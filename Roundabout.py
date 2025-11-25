from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Deque, Dict, Optional, Tuple
from collections import deque
import math, random, argparse, statistics as stats

# ------------------------------ helpers ------------------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def ahead_distance(a: float, b: float, C: float) -> float:
    """Signed-arc distance from position a to b along the ring (>=0, wraps at C)."""
    d = b - a
    if d < 0:
        d += C
    return d

def mmss(t: float) -> str:
    m = int(t // 60)
    s = int(t - m*60)
    return f"{m:02d}:{s:02d}"

# ------------------------------ config -------------------------------------

@dataclass
class DriverParams:
    a_max: float = 1.5
    b_comf: float = 2.0
    delta: int = 4
    s0: float = 2.0
    T: float = 1.2
    tau: float = 0.8
    v0_ring: float = 12.0  # m/s (≈43 km/h)
    t_cross_lane: float = 0.8  # sec to traverse ONE circulating lane during entry

@dataclass
class GapParams:
    crit_gap_mean: float = 3.0
    crit_gap_sd: float = 0.6
    followup_mean: float = 2.0
    followup_sd: float = 0.3

@dataclass
class Geometry:
    diameter: float = 45.0
    lanes: int = 3  # 1, 2, or 3
    g: float = 9.81
    a_lat_max: float = 2.2  # m/s^2 lateral comfort limit

    def circumference(self) -> float:
        return math.pi * self.diameter

    def vmax_by_curvature(self) -> float:
        # v = sqrt(a_lat_max * R), R = diameter/2
        R = max(5.0, 0.5 * self.diameter)
        return math.sqrt(self.a_lat_max * R)

@dataclass
class Demand:
    arrival: Tuple[float, float, float, float] = (0.18, 0.12, 0.20, 0.15)  # veh/s per arm
    turning: Tuple[float, float, float] = (0.25, 0.55, 0.20)  # (L, T, R), sums to 1.0

@dataclass
class SimConfig:
    seed: int = 42
    horizon: float = 3600.0
    report_every: float = 300.0
    dt: float = 0.2
    # NEW: seed a small circulating flow so lane effects are visible without cranking demand
    initial_ring_density: float = 0.02  # vehicles per meter per lane (~20 veh/km)
    driver: DriverParams = field(default_factory=DriverParams)
    gaps: GapParams = field(default_factory=GapParams)
    geo: Geometry = field(default_factory=Geometry)
    dem: Demand = field(default_factory=Demand)

# ------------------------------ vehicle ------------------------------------

@dataclass
class Vehicle:
    id: int
    arm: int               # origin arm (0..3)
    turn: str              # 'L','T','R' (dummy for seeded ring cars)
    lane_choice: int       # target circulating lane index at merge (0 outer ... L-1 inner-most)
    steps_to_exit: int     # 1=R, 2=T, 3=L (seeded ring cars use a large sentinel)
    in_ring: bool = False
    pos: float = 0.0
    speed: float = 0.0
    t_queue_start: float = 0.0
    t_enter_ring: Optional[float] = None
    crit_gap: float = 0.0
    followup: float = 0.0

# ------------------------------ simulator ----------------------------------

class RoundaboutSim:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        random.seed(cfg.seed)

        # geometry and kinematics
        self.C = cfg.geo.circumference()
        self.N = 4
        self.entry_pos = [i * self.C / self.N for i in range(self.N)]
        self.exit_pos = self.entry_pos[:]
        self.vmax_ring = min(cfg.driver.v0_ring, cfg.geo.vmax_by_curvature())

        # state: circulating lanes as lists of Vehicle
        L = max(1, min(3, cfg.geo.lanes))
        self.L = L
        self.lanes: List[List[Vehicle]] = [[] for _ in range(L)]

        # per-arm queues
        self.queues: List[Deque[Vehicle]] = [deque() for _ in range(self.N)]
        self.next_needed_headway: List[float] = [0.0] * self.N  # 0 => use crit_gap of next vehicle

        # history for DDE-like leader snapshot
        self.history: Deque[Dict[int, Tuple[float, float]]] = deque()
        self.max_hist_len = max(1, int(round(self.cfg.driver.tau / self.cfg.dt)))

        # ids and time
        self._veh_seq = 0
        self.t = 0.0

        # metrics
        self.total_arrivals = 0
        self.total_exits = 0
        self.delays_all: List[float] = []
        self.max_queue_len_global = [0] * self.N

        # window (per report interval)
        self.win_arrivals = 0
        self.win_exits = 0
        self.win_delays: List[float] = []
        self.win_queue_max = [0] * self.N

        # diagnostics
        self.merge_denied = 0
        self.merge_attempts = 0
        self.entered_per_lane: List[int] = [0 for _ in range(self.L)]

        # warm-start circulating traffic to make lane effects visible
        self._seed_ring(cfg.initial_ring_density)

    # ------------------------------ policies --------------------------------

    def _lane_allowed(self, turn: str, lane: int) -> bool:
        """Whether a (turn) may use (lane) under the user's rules."""
        if self.L == 1:
            return lane == 0
        if self.L == 2:
            # lane 0: R,T | lane 1: T,L
            return (lane == 0 and turn in ('R', 'T')) or (lane == 1 and turn in ('T', 'L'))
        # L == 3
        # lane 0: R,T | lane 1: T | lane 2: T,L
        if lane == 0:
            return turn in ('R', 'T')
        if lane == 1:
            return turn == 'T'
        if lane == 2:
            return turn in ('T', 'L')
        return False

    def _allowed_lanes(self, turn: str) -> List[int]:
        return [ln for ln in range(self.L) if self._lane_allowed(turn, ln)]

    def _steps_for_turn(self, turn: str) -> int:
        return {'R': 1, 'T': 2, 'L': 3}[turn]

    # ------------------------------ warm start -------------------------------

    def _seed_ring(self, density_per_m: float) -> None:
        """Place a small number of 'background' circulating vehicles in each lane to
        create realistic merge pressure even at moderate external demand.
        These background vehicles do NOT exit (they have negative ids)."""
        if density_per_m <= 0.0:
            return
        for ln in range(self.L):
            n = max(1, int(self.C * density_per_m))
            # random phase so entries see different spacings per lane
            phase = random.random() * self.C
            for i in range(n):
                pos = (phase + i * (self.C / n)) % self.C
                v = Vehicle(
                    id=-(1000000 + ln * 10000 + i),  # negative = permanent circulating
                    arm=0, turn='T', lane_choice=ln, steps_to_exit=10_000_000,
                    in_ring=True, pos=pos, speed=0.85 * self.vmax_ring,
                    t_queue_start=0.0, t_enter_ring=0.0, crit_gap=0.0, followup=0.0
                )
                self.lanes[ln].append(v)

    # ------------------------------ arrivals --------------------------------

    def _draw_turn(self) -> str:
        Lp, Tp, Rp = self.cfg.dem.turning
        u = random.random()
        if u < Lp: return 'L'
        if u < Lp + Tp: return 'T'
        return 'R'

    def _draw_crit(self) -> float:
        m, s = self.cfg.gaps.crit_gap_mean, self.cfg.gaps.crit_gap_sd
        x = random.gauss(m, s) if s > 0 else m
        return max(0.5, x)

    def _draw_follow(self) -> float:
        m, s = self.cfg.gaps.followup_mean, self.cfg.gaps.followup_sd
        x = random.gauss(m, s) if s > 0 else m
        return max(0.2, x)

    def _spawn_arrivals(self) -> None:
        """Bernoulli thinning approximation to Poisson(λ dt) per arm per step."""
        lam = self.cfg.dem.arrival
        dt = self.cfg.dt
        for arm in range(self.N):
            if random.random() < lam[arm] * dt:
                turn = self._draw_turn()
                # temporary lane; final decision happens right before entry
                lane = 0
                if self.L == 3 and turn == 'T':
                    lane = 1
                if not self._lane_allowed(turn, lane):
                    for ln in self._allowed_lanes(turn):
                        lane = ln; break
                v = Vehicle(
                    id=self._veh_seq, arm=arm, turn=turn, lane_choice=lane,
                    steps_to_exit=self._steps_for_turn(turn),
                    t_queue_start=self.t, crit_gap=self._draw_crit(), followup=self._draw_follow()
                )
                self._veh_seq += 1
                self.queues[arm].append(v)
                self.total_arrivals += 1
                self.win_arrivals += 1

    # ------------------------------ history ---------------------------------

    def _push_history(self) -> None:
        snap: Dict[int, Tuple[float, float]] = {}
        for lane in self.lanes:
            for v in lane:
                snap[v.id] = (v.pos, v.speed)
        self.history.append(snap)
        if len(self.history) > self.max_hist_len:
            self.history.popleft()

    def _snapshot(self, steps_back: int) -> Dict[int, Tuple[float, float]]:
        if steps_back <= 0 or steps_back > len(self.history):
            return {}
        return list(self.history)[-steps_back]

    # ---------------------------- merge checks -------------------------------

    def _nearest_time_in_lane(self, entry_pos: float, lane_idx: int) -> float:
        """Time until the nearest vehicle in lane_idx *reaches the entry* (forward time)."""
        if lane_idx < 0 or lane_idx >= self.L:
            return float('inf')
        vmin = 1e-3
        best_t = float('inf')
        C = self.C
        for v in self.lanes[lane_idx]:
            d_fwd = ahead_distance(v.pos, entry_pos, C)
            t_fwd = d_fwd / max(vmin, v.speed)
            best_t = min(best_t, t_fwd)
        return best_t

    def _min_ahead_in_lane(self, entry_pos: float, lane_idx: int) -> float:
        """Shortest arc from entry to any vehicle ahead in lane_idx (space check)."""
        if lane_idx < 0 or lane_idx >= self.L:
            return float('inf')
        C = self.C
        best = float('inf')
        for v in self.lanes[lane_idx]:
            d = ahead_distance(entry_pos, v.pos, C)
            best = min(best, d)
        return best

    def _must_cross_lanes(self, target_lane: int) -> List[int]:
        """Lanes crossed when entering target_lane from the outside (outer=0)."""
        return list(range(0, max(0, target_lane)))

    def _merge_feasible(self, arm: int, v: Vehicle, needed_headway: float, lane_override: Optional[int]=None) -> bool:
        """
        Gap acceptance with lane benefit & realistic crossing:
        - Time headway enforced in the TARGET lane only (>= needed_headway).
        - For EACH crossed lane, require forward time headway >= t_cross_lane.
        - Space-ahead buffers at the entry for all involved lanes.
        """
        entry = self.entry_pos[arm]
        drv = self.cfg.driver
        target = v.lane_choice if lane_override is None else lane_override

        # Space buffers
        s_target = drv.s0 + 2.0
        s_cross  = drv.s0 + 0.5

        # 1) Target-lane time & space
        t_near = self._nearest_time_in_lane(entry, target)
        if t_near < needed_headway:
            return False
        if self._min_ahead_in_lane(entry, target) < s_target:
            return False

        # 2) Crossing-lane time & space for each lane you traverse to reach target
        for ln in self._must_cross_lanes(target):
            if self._nearest_time_in_lane(entry, ln) < drv.t_cross_lane:
                return False
            if self._min_ahead_in_lane(entry, ln) < s_cross:
                return False

        return True

    # ------------------------------ dynamic lane choice ----------------------

    def _best_lane_now(self, arm: int, turn: str, needed_headway: float) -> int:
        """Pick among allowed lanes using a greedy score based on near-term mergeability."""
        entry = self.entry_pos[arm]
        allowed = self._allowed_lanes(turn)
        if len(allowed) == 1:
            return allowed[0]

        best_lane = allowed[0]
        best_score = float('inf')
        for ln in allowed:
            feasible = self._merge_feasible(arm, Vehicle(-1, arm, turn, ln, self._steps_for_turn(turn)),
                                            needed_headway, lane_override=ln)
            if feasible:
                score = 0.0 + 0.2 * len(self._must_cross_lanes(ln))
            else:
                t = self._nearest_time_in_lane(entry, ln)
                score = (needed_headway - t if t < needed_headway else 0.1 * t) + 0.3 * len(self._must_cross_lanes(ln))
            if score < best_score:
                best_score = score
                best_lane = ln
        return best_lane

    # ------------------------------ entries ----------------------------------

    def _attempt_entries(self) -> None:
        for i in range(self.N):
            q = self.queues[i]
            if not q:
                self.next_needed_headway[i] = 0.0
                continue
            v = q[0]
            needed = self.next_needed_headway[i] or v.crit_gap

            # Dynamically (re)choose the lane just before attempting to enter
            v.lane_choice = self._best_lane_now(i, v.turn, needed)

            self.merge_attempts += 1
            if self._merge_feasible(i, v, needed):
                q.popleft()
                v.in_ring = True
                v.pos = (self.entry_pos[i] + 1.0) % self.C
                v.speed = min(4.0, self.vmax_ring)
                v.t_enter_ring = self.t
                delay = self.t - v.t_queue_start
                self.delays_all.append(delay)
                self.win_delays.append(delay)
                self.lanes[v.lane_choice].append(v)
                self.entered_per_lane[v.lane_choice] += 1
                self.next_needed_headway[i] = v.followup
            else:
                self.merge_denied += 1
                if self.next_needed_headway[i] != 0.0:
                    self.next_needed_headway[i] = min(8.0, self.next_needed_headway[i] * 1.02)

            # Track per-window and global max queues
            qlen = len(q)
            self.win_queue_max[i] = max(self.win_queue_max[i], qlen)
            self.max_queue_len_global[i] = max(self.max_queue_len_global[i], qlen)

    # ------------------------------ ring step --------------------------------

    def _leader_at_delayed_time(self, lane: List[Vehicle],
                                pos_d: float,
                                snap: Dict[int, Tuple[float, float]]) -> Tuple[Optional[Vehicle], float, float]:
        """Return (leader, delayed_gap, leader_speed) w.r.t. a delayed ego position pos_d in the given lane."""
        C = self.C
        if not lane:
            return None, float('inf'), 0.0
        best_gap = float('inf')
        leader_speed = 0.0
        leader = None
        for u in lane:
            u_pos, u_spd = snap.get(u.id, (u.pos, u.speed))
            gap = ahead_distance(pos_d, u_pos, C)
            if 1e-6 < gap < best_gap:
                best_gap = gap
                leader_speed = u_spd
                leader = u
        if math.isinf(best_gap):
            return None, float('inf'), 0.0
        return leader, best_gap, leader_speed

    def _advance_ring(self) -> None:
        dt = self.cfg.dt
        steps_back = int(round(self.cfg.driver.tau / dt))
        snap = self._snapshot(steps_back) if steps_back > 0 else {}

        for lane in self.lanes:
            if not lane:
                continue
            lane.sort(key=lambda u: u.pos)
            next_state: List[Tuple[Vehicle, float, float]] = []
            for v in lane:
                pos_d, v_d = snap.get(v.id, (v.pos, v.speed))
                _, gap_d, vL_d = self._leader_at_delayed_time(lane, pos_d, snap)

                drv = self.cfg.driver
                v0 = min(self.vmax_ring, drv.v0_ring)
                s0, T, a_max, b, delta = drv.s0, drv.T, drv.a_max, drv.b_comf, drv.delta

                if math.isinf(gap_d):
                    gap_d, vL_d = 1e9, v_d
                dv = v_d - vL_d
                s_star = s0 + v_d * T + (v_d * dv) / max(1e-6, 2.0 * math.sqrt(a_max * b))
                s_star = max(s0, s_star)
                acc = a_max * (1.0 - (v_d / max(0.1, v0)) ** delta - (s_star / max(1e-3, gap_d)) ** 2)
                v_new = clamp(v.speed + acc * dt, 0.0, v0)
                x_new = (v.pos + v_new * dt) % self.C
                next_state.append((v, x_new, v_new))

            for v, x_new, v_new in next_state:
                v.pos = x_new
                v.speed = v_new

        # exits (allow exits from any lane for simplicity)
        survivors_by_lane: List[List[Vehicle]] = [[] for _ in range(self.L)]
        for lane_idx, lane in enumerate(self.lanes):
            for v in lane:
                # permanent circulating background cars never exit
                if v.id < 0:
                    survivors_by_lane[lane_idx].append(v)
                    continue
                target_angle = (self.entry_pos[v.arm] + v.steps_to_exit * (self.C / self.N)) % self.C
                if ahead_distance(v.pos, target_angle, self.C) < 1.0 and v.t_enter_ring is not None and (self.t - v.t_enter_ring) > 0.5:
                    self.total_exits += 1
                    self.win_exits += 1
                    continue
                survivors_by_lane[lane_idx].append(v)
        self.lanes = survivors_by_lane

    # ------------------------------ reporting --------------------------------

    def _report_window(self, end_time: float) -> None:
        if self.win_delays:
            avg_delay = sum(self.win_delays) / len(self.win_delays)
            p95 = stats.quantiles(self.win_delays, n=20)[18] if len(self.win_delays) >= 20 else avg_delay
        else:
            avg_delay = 0.0
            p95 = 0.0
        thr_win = (self.win_exits * 3600.0) / max(1e-9, self.cfg.report_every)
        max_q = f"[{' ,'.join(str(x) for x in self.win_queue_max)}]"
        print(f"[{mmss(end_time)}] lanes={self.L}  arrivals={self.win_arrivals}  exits={self.win_exits}  "
              f"throughput={thr_win:.0f} veh/hr  avg_delay={avg_delay:.1f}s  p95={p95:.1f}s  "
              f"max_q={max_q}")
        self.win_arrivals = 0
        self.win_exits = 0
        self.win_delays = []
        self.win_queue_max = [0] * self.N

    # ------------------------------ main loop --------------------------------

    def run(self) -> None:
        dt = self.cfg.dt
        next_report = self.cfg.report_every
        while self.t < self.cfg.horizon + 1e-9:
            self._spawn_arrivals()
            self._attempt_entries()
            self._push_history()
            self._advance_ring()
            self.t += dt
            if self.t + 1e-9 >= next_report:
                self._report_window(next_report)
                next_report += self.cfg.report_every

        # final summary
        hr_throughput = (self.total_exits * 3600.0) / max(1e-9, self.cfg.horizon)
        avg_dly = (sum(self.delays_all) / len(self.delays_all)) if self.delays_all else 0.0
        p95 = stats.quantiles(self.delays_all, n=20)[18] if len(self.delays_all) >= 20 else avg_dly
        deny_rate = (self.merge_denied / max(1, self.merge_attempts)) * 100.0
        print("\n=== Hourly Summary ===")
        print(f"lanes={self.L}  arrivals={self.total_arrivals}  exits={self.total_exits}  "
              f"throughput={hr_throughput:.0f} veh/hr  avg_delay={avg_dly:.1f}s  p95={p95:.1f}s")
        print(f"max_queue_per_arm={self.max_queue_len_global}")
        print(f"merge_attempts={self.merge_attempts}  denied={self.merge_denied} ({deny_rate:.1f}% denied)")
        print(f"entered_per_lane={self.entered_per_lane}")

# ------------------------------ CLI ----------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multilane Roundabout Microsim (1–3 lanes)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--horizon", type=float, default=3600.0)
    p.add_argument("--report-every", type=float, default=300.0)
    p.add_argument("--dt", type=float, default=0.2)
    p.add_argument("--diameter", type=float, default=45.0)
    p.add_argument("--lanes", type=int, default=2, choices=[1,2,3])

    p.add_argument("--arrival", type=float, nargs=4, default=[0.18,0.12,0.20,0.15],
                   help="veh/s per arm (4 numbers)")
    p.add_argument("--turning", type=float, nargs=3, default=[0.25,0.55,0.20],
                   help="L T R (sums to 1)")
    p.add_argument("--crit-gap-mean", type=float, default=3.0)
    p.add_argument("--crit-gap-sd", type=float, default=0.6)
    p.add_argument("--followup-mean", type=float, default=2.0)
    p.add_argument("--followup-sd", type=float, default=0.3)
    p.add_argument("--initial-ring-density", type=float, default=0.02,
                   help="background circulating vehicles per meter per lane (0 to disable)")
    return p

def from_args(args: argparse.Namespace) -> SimConfig:
    geo = Geometry(diameter=args.diameter, lanes=args.lanes)
    dem = Demand(arrival=tuple(args.arrival), turning=tuple(args.turning))
    gaps = GapParams(crit_gap_mean=args.crit_gap_mean, crit_gap_sd=args.crit_gap_sd,
                     followup_mean=args.followup_mean, followup_sd=args.followup_sd)
    cfg = SimConfig(seed=args.seed, horizon=args.horizon, report_every=args.report_every, dt=args.dt,
                    geo=geo, dem=dem, gaps=gaps, initial_ring_density=args.initial_ring_density)
    return cfg

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    cfg = from_args(args)
    sim = RoundaboutSim(cfg)
    sim.run()