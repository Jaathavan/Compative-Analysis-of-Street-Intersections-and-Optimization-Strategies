# Comparative Analysis of Street Intersections and Optimization Strategies

By: Jaathavan Ranjanathan, Massi Afzal, Bach Vu

A comprehensive study comparing the performance of roundabouts and signalized intersections under various traffic conditions using both our own text-based simulator and a microscopic traffic simulation (SUMO).

## Project Overview

This project implements and compares two intersection control strategies (note signalized was omitted in final paper due to limited time):

1. **Roundabout**: Unsignalized circular intersections with priority-based gap acceptance
2. **Signalized Intersection**: Fixed-time traffic signal control with dedicated phases

The analysis uses:
- **Text-based Simulations** (implemented using DDE's (delay differentiable equations), IDM (intelligent driver module), poisson process, ... )
- **Microscopic simulation** (SUMO - Simulation of Urban MObility)
- **Performance metrics**: throughput, delay, queue length

## Final Report

The complete analysis, methodology, and findings are documented in the [Final Report (PDF)](./Final_Report.pdf), which includes:

- Detailed comparison of roundabout intersection performance
- Mathematical models and simulation methodologies
- Comprehensive results analysis and visualization
- Conclusions and recommendations for intersection optimization strategies

## Results Summary (Roundabout Study)

This section summarizes the final report results for both simulators:

- Text-based simulator demand set: $\lambda \in \{0.05, 0.07, 0.10\}$ veh/s/arm
- SUMO demand set: $\lambda \in \{0.05, 0.10, 0.15, 0.20, 0.25\}$ veh/s/arm
- KPIs: throughput (veh/hr), average delay (s), 95th-percentile delay (s), and maximum queue length (veh)

### Report Tables (Key Metrics)

#### Base-demand comparison at $\lambda = 0.10$ (from report Tables 2, 4, 6, 3, 5, 7)

| Simulator | Lanes | Throughput (veh/hr) | Avg Delay (s) | P95 Delay (s) | Max Queue |
|---|---:|---:|---:|---:|---:|
| Text-based | 1 | ~764 | ~718.0 | ~1506.0 | ~189 |
| Text-based | 2 | ~1414 | ~26.0 | ~131.2 | ~27 |
| Text-based | 3 | ~1417 | ~9.8 | ~67.1 | ~20 |
| SUMO | 1 | ~1140 | ~8.8 | ~54.6 | 38 |
| SUMO | 2 | ~1260 | ~2.2 | ~9.5 | 5 |
| SUMO | 3 | ~1248 | ~1.2 | ~6.1 | 4 |

#### High-demand comparison at $\lambda = 0.25$ (from report Tables 3, 5, 7)

| Simulator | Lanes | Throughput (veh/hr) | Avg Delay (s) | P95 Delay (s) | Max Queue |
|---|---:|---:|---:|---:|---:|
| SUMO | 1 | ~1320 | ~71.9 | ~166.9 | 169 |
| SUMO | 2 | ~2244 | ~30.5 | ~99.3 | 98 |
| SUMO | 3 | ~2772 | ~8.2 | ~24.4 | 91 |

#### Breakdown thresholds (from report Table 8 and Section 6.5.2)

| Source | Configuration | Observed threshold behavior |
|---|---|---|
| Text-based | 2-lane | Practical breakdown begins around $\lambda \approx 0.12$; throughput saturates near ~1500 veh/hr while delay/queue spike |
| Text-based | 3-lane | Stable at $\lambda = 0.12$ and breaks down between $0.12$ and $0.15$; practical capacity ~1700-1750 veh/hr |
| SUMO | 2-lane (30m) | Throughput peaks around ~2172 veh/hr at $\lambda = 0.20$ and plateaus near ~2244 at $\lambda = 0.25$ |
| SUMO | 3-lane (30m) | No breakdown seen through $\lambda = 0.25$; throughput reaches ~2772 veh/hr with low delay |

#### Diameter sensitivity at fixed $\lambda = 0.10$ (from report Tables 9 and 10)

| Source | Configuration | Best reported diameter zone | Worst reported diameter zone |
|---|---|---|---|
| Text-based | 2-lane | 20-30m: throughput ~1421-1424, avg delay ~2-3s | 90m: throughput ~277, avg delay ~142.5s, very large queues |
| Text-based | 3-lane | 20-30m: throughput ~1420, avg delay ~1-2s | 90m: throughput ~915, avg delay ~590s, severe queues |
| SUMO | 1/2/3-lane | 30m consistently best throughput and delay | 50m reduces throughput and increases delay across lane counts |

### Figures: Text-Based Simulator

#### Text-based arrival-rate metrics (report Figure 1 style)

![Text simulation: average delay vs arrival rate](results/text_simulation_results/Figure_1.png)

![Text simulation: throughput vs arrival rate](results/text_simulation_results/Figure_2.png)

![Text simulation: 95th percentile delay vs arrival rate](results/text_simulation_results/Figure_3.png)

![Text simulation: max queue vs arrival rate](results/text_simulation_results/Figure_4.png)

#### Text-based 2-lane vs 3-lane breakpoint comparisons (report Figure 3 style)

![Text simulation: delay breakpoint comparison](results/text_simulation_results/Figure_11.png)

![Text simulation: p95 breakpoint comparison](results/text_simulation_results/Figure_22.png)

![Text simulation: throughput breakpoint comparison](results/text_simulation_results/Figure_33.png)

![Text simulation: queue breakpoint comparison](results/text_simulation_results/Figure_44.png)

#### Text-based diameter sensitivity at $\lambda = 0.10$ (report Figure 5 style)

![Text simulation: average delay vs diameter](results/text_simulation_results/F1.png)

![Text simulation: throughput vs diameter](results/text_simulation_results/F2.png)

![Text simulation: max queue vs diameter](results/text_simulation_results/F3.png)

![Text simulation: p95 delay vs diameter](results/text_simulation_results/F4.png)

### Figures: SUMO

#### SUMO arrival-rate metrics (report Figure 2 style)

![SUMO: throughput vs arrival rate by number of lanes](results/roundabout_comparisons/2_throughput_vs_arrival.png)

![SUMO: average delay vs arrival rate](results/roundabout_comparisons/1_delay_vs_arrival.png)

![SUMO: 95th percentile delay vs arrival rate](results/roundabout_comparisons/3_p95_delay_vs_arrival.png)

![SUMO: maximum queue length vs arrival rate](results/roundabout_comparisons/4_max_queue_vs_arrival.png)

#### SUMO 2-lane vs 3-lane direct comparison with breakpoints (report Figure 4 style)

![SUMO: throughput with capacity breakpoints](results/roundabout_comparisons/9_throughput_vs_arrival_2lane_3lane.png)

![SUMO: average delay with performance breakpoints](results/roundabout_comparisons/10_avg_delay_vs_arrival_2lane_3lane.png)

![SUMO: p95 delay with reliability breakpoints](results/roundabout_comparisons/11_p95_delay_vs_arrival_2lane_3lane.png)

![SUMO: maximum queue with storage breakpoints](results/roundabout_comparisons/12_max_queue_vs_arrival_2lane_3lane.png)

#### SUMO diameter sensitivity at $\lambda = 0.10$ (report Figure 6 style)

![SUMO: throughput vs diameter](results/roundabout_comparisons/6_throughput_vs_diameter.png)

![SUMO: average delay vs diameter](results/roundabout_comparisons/5_delay_vs_diameter.png)

![SUMO: p95 delay vs diameter](results/roundabout_comparisons/7_p95_delay_vs_diameter.png)

![SUMO: maximum queue vs diameter](results/roundabout_comparisons/8_max_queue_vs_diameter.png)

### Final Summary Statement (From Report Conclusion)

Across both the Python text-based simulator and SUMO, the report concludes that lane count and diameter jointly determine roundabout performance: 1-lane designs reach practical saturation earliest, adding a second lane delivers the largest capacity gain, and a third lane mainly improves resilience and delay at higher demand. The report also finds that compact multi-lane designs (about 20-30m, with 30m consistently strongest in SUMO) outperform larger diameters, while oversized designs can significantly worsen queues and delay. Within the tested range, a 3-lane roundabout around 30m diameter is the most robust-performing configuration.

## Project Structure

```
.
├── README.md                           # Project documentation (this file)
├── requirements.txt                    # Python dependencies
│
├── Roundabout.py                       # Text-based roundabout simulation (queueing model)
├── Signalized.py                       # Text-based signalized intersection simulation
│
├── results/                            # Consolidated results and comparisons
│   ├── roundabout_text_results.csv     # Analytical model outputs
│   ├── signalized_text_results.csv     # Analytical model outputs
│   ├── roundabout_comparisons/         # SUMO simulation results
│   │   ├── simulation_data.csv         # Performance metrics (lanes, diameter, arrival rate)
│   │   └── *.png                       # Comparison graphs
│   ├── text_simulation_results         # Text simulation results
│   │   └── *.png                       # Comparison graphs
│   └── sumo_demo/                      # Demo simulation outputs
│
├── roundabout/                         # SUMO-based roundabout simulation framework
│   ├── quickstart.py                   # Quick-start demo script
│   ├── config/
│   │   └── config.yaml                 # Network and demand configuration
│   ├── src/
│   │   ├── generate_network.py         # Roundabout network generator
│   │   ├── generate_routes.py          # Traffic demand generator
│   │   ├── run_simulation.py           # SUMO simulation runner
│   │   └── analyze_results.py          # Performance analysis
│   ├── quickstart_output/              # Demo outputs
│   └── results/                        # Batch simulation results
│
├── signalized/                         # SUMO-based signalized intersection framework
│   ├── quickstart.py                   # Quick-start demo script
│   ├── config/
│   │   └── config.yaml                 # Network and signal timing configuration
│   ├── src/
│   │   ├── generate_network.py         # Intersection network generator
│   │   ├── generate_routes.py          # Traffic demand generator
│   │   └── run_simulation.py           # SUMO simulation runner
│   ├── quickstart_output/              # Demo outputs
│   └── results/                        # Batch simulation results
│
├── dashboard_visualizations/           # Interactive visualization dashboard
│   ├── streamlit_app.py                # Web-based visualization interface
│   ├── Roundabout.py                   # Roundabout model (for dashboard)
│   ├── traffic_signal.py               # Signalized intersection model
│   ├── signal_dataset.csv              # Sample data for visualization
│   └── docs/                           # Dashboard documentation
│
├── test_sumo/                          # SUMO installation verification
│   └── test.net.xml                    # Minimal test network
│
└── Final_Report.pdf                    # Final Overall Report
```
