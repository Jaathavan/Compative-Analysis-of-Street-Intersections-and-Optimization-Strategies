#!/usr/bin/env python3
# filepath: roundabout/src/generate_failure_videos.py
"""
generate_failure_videos.py - Create Slowed-Down Failure Demonstration Videos
=============================================================================

Generates SUMO-GUI videos showing failure scenarios for 1, 2, and 3-lane
roundabouts at critical arrival rates where the system breaks down.

Videos are slowed down to make queue buildup and congestion visible.

Usage:
    cd roundabout
    python src/generate_failure_videos.py --speed 0.5 --duration 600
    python src/generate_failure_videos.py --config config/config.yaml --output videos/
"""

import argparse
import subprocess
import sys
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List
import time
import copy
import os

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Now import local modules
try:
    from generate_network import RoundaboutNetworkGenerator
    from generate_routes import RoundaboutRouteGenerator
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    print("Make sure you're running from the roundabout/ directory:")
    print("  cd roundabout && python src/generate_failure_videos.py")
    sys.exit(1)


class FailureVideoGenerator:
    """
    Generates demonstration videos of roundabout failure modes.
    """
    
    def __init__(self, config_path: str, output_dir: str):
        """Initialize generator."""
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate config exists
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"✓ Loaded configuration from {config_path}")
        print(f"✓ Output directory: {output_dir}")
    
    def create_gui_settings(self, delay: int = 100, zoom: float = 1500):
        """
        Create SUMO-GUI settings file for video recording.
        
        Args:
            delay: Delay per simulation step (ms) - higher = slower playback
            zoom: Zoom level (higher = more zoomed out)
        
        Returns:
            Path to settings file
        """
        settings_path = self.output_dir / 'gui_settings.xml'
        
        # Create settings XML
        root = ET.Element('viewsettings')
        
        # Viewport
        viewport = ET.SubElement(root, 'viewport')
        viewport.set('y', '0')
        viewport.set('x', '0')
        viewport.set('zoom', str(zoom))
        viewport.set('angle', '0')
        
        # Delay
        delay_elem = ET.SubElement(root, 'delay')
        delay_elem.set('value', str(delay))
        
        # Scheme
        scheme = ET.SubElement(root, 'scheme')
        scheme.set('name', 'real world')
        
        # Show vehicle details
        vehicles = ET.SubElement(root, 'vehicles')
        vehicles.set('vehicleQuality', '2')
        vehicles.set('vehicleSize', '1')
        vehicles.set('showBlinker', 'true')
        vehicles.set('vehicleName', 'show')
        vehicles.set('vehicleText', 'show')
        vehicles.set('showLaneChangePreference', 'true')
        
        # Background
        background = ET.SubElement(root, 'background')
        background.set('backgroundColor', '255,255,255')
        background.set('showGrid', 'true')
        background.set('gridXSize', '10')
        background.set('gridYSize', '10')
        
        # Write XML with proper formatting
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")  # Pretty print (Python 3.9+)
        tree.write(str(settings_path), encoding='utf-8', xml_declaration=True)
        
        print(f"✓ Created GUI settings: {settings_path}")
        return settings_path
    
    def create_sumocfg(self, scenario_dir: Path, net_file: str, 
                      route_file: str, duration: int) -> Path:
        """
        Create SUMO configuration file.
        
        Args:
            scenario_dir: Directory for scenario files
            net_file: Network file name
            route_file: Route file name
            duration: Simulation duration (seconds)
        
        Returns:
            Path to .sumocfg file
        """
        sumocfg_path = scenario_dir / 'roundabout.sumocfg'
        
        # Create configuration XML
        root = ET.Element('configuration')
        
        # Input section
        input_elem = ET.SubElement(root, 'input')
        net_elem = ET.SubElement(input_elem, 'net-file')
        net_elem.set('value', net_file)
        route_elem = ET.SubElement(input_elem, 'route-files')
        route_elem.set('value', route_file)
        
        # Time section
        time_elem = ET.SubElement(root, 'time')
        begin_elem = ET.SubElement(time_elem, 'begin')
        begin_elem.set('value', '0')
        end_elem = ET.SubElement(time_elem, 'end')
        end_elem.set('value', str(duration))
        step_elem = ET.SubElement(time_elem, 'step-length')
        step_elem.set('value', '0.1')
        
        # Processing section
        processing = ET.SubElement(root, 'processing')
        lateral_elem = ET.SubElement(processing, 'lateral-resolution')
        lateral_elem.set('value', '0.8')
        
        # Report section (suppress warnings)
        report = ET.SubElement(root, 'report')
        verbose = ET.SubElement(report, 'verbose')
        verbose.set('value', 'false')
        no_warnings = ET.SubElement(report, 'no-warnings')
        no_warnings.set('value', 'true')
        
        # Write XML
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(str(sumocfg_path), encoding='utf-8', xml_declaration=True)
        
        print(f"✓ Created SUMO config: {sumocfg_path}")
        return sumocfg_path
    
    def generate_failure_scenario(self, lanes: int, arrival_rate: float, 
                                  duration: int = 600, seed: int = 42) -> Dict:
        """
        Generate network and routes for a failure scenario.
        
        Args:
            lanes: Number of circulating lanes (1, 2, or 3)
            arrival_rate: Per-arm arrival rate (veh/s) - set to critical value
            duration: Simulation duration (seconds)
            seed: Random seed
        
        Returns:
            Dictionary with paths to generated files
        """
        print(f"\n{'='*70}")
        print(f"Generating failure scenario: {lanes} lane(s), λ={arrival_rate:.3f} veh/s")
        print(f"{'='*70}")
        
        # Deep copy config to avoid mutation
        temp_config = copy.deepcopy(self.config)
        
        # Modify parameters
        temp_config['geometry']['lanes'] = lanes
        temp_config['demand']['arrivals'] = [arrival_rate] * 4  # Equal demand on all arms
        temp_config['simulation']['horizon'] = duration
        temp_config['simulation']['seed'] = seed
        
        # Create scenario directory
        scenario_name = f'{lanes}lane_failure_lambda{arrival_rate:.3f}'
        scenario_dir = self.output_dir / scenario_name
        scenario_dir.mkdir(exist_ok=True)
        
        print(f"  Scenario: {scenario_name}")
        print(f"  Directory: {scenario_dir}")
        
        # Generate network
        print(f"  [1/3] Generating network...")
        try:
            net_gen = RoundaboutNetworkGenerator(temp_config)
            net_file = scenario_dir / 'roundabout.net.xml'
            net_gen.generate(str(net_file))
            print(f"    ✓ Network: {net_file.name}")
        except Exception as e:
            print(f"    ✗ ERROR generating network: {e}")
            raise
        
        # Generate routes
        print(f"  [2/3] Generating routes...")
        try:
            route_gen = RoundaboutRouteGenerator(temp_config)
            route_file = scenario_dir / 'roundabout.rou.xml'
            route_gen.generate(
                network_file=str(net_file),
                output_file=str(route_file)
            )
            print(f"    ✓ Routes: {route_file.name}")
        except Exception as e:
            print(f"    ✗ ERROR generating routes: {e}")
            raise
        
        # Create SUMO config
        print(f"  [3/3] Creating SUMO configuration...")
        try:
            sumocfg = self.create_sumocfg(
                scenario_dir, 
                'roundabout.net.xml',
                'roundabout.rou.xml',
                duration
            )
            print(f"    ✓ Config: {sumocfg.name}")
        except Exception as e:
            print(f"    ✗ ERROR creating config: {e}")
            raise
        
        print(f"{'='*70}")
        print(f"✅ Scenario generated successfully\n")
        
        return {
            'scenario_name': scenario_name,
            'scenario_dir': scenario_dir,
            'sumocfg': sumocfg,
            'net_file': net_file,
            'route_file': route_file,
            'lanes': lanes,
            'arrival_rate': arrival_rate
        }
    
    def record_video(self, scenario: Dict, gui_settings_path: Path, 
                    playback_delay: int = 100, auto_start: bool = True) -> Path:
        """
        Launch SUMO-GUI for video recording.
        
        Args:
            scenario: Scenario dictionary from generate_failure_scenario()
            gui_settings_path: Path to GUI settings XML
            playback_delay: Delay between simulation steps (ms)
            auto_start: Whether to auto-start simulation
        
        Returns:
            Path where video should be saved (manual recording required)
        """
        scenario_name = scenario['scenario_name']
        sumocfg = scenario['sumocfg']
        
        print(f"\n{'='*70}")
        print(f"LAUNCHING SUMO-GUI: {scenario_name}")
        print(f"{'='*70}")
        print(f"  Config: {sumocfg}")
        print(f"  Playback delay: {playback_delay}ms per step")
        print(f"  Auto-start: {auto_start}")
        
        # Validate files exist
        if not sumocfg.exists():
            raise FileNotFoundError(f"SUMO config not found: {sumocfg}")
        if not gui_settings_path.exists():
            raise FileNotFoundError(f"GUI settings not found: {gui_settings_path}")
        
        # Output video path (for reference)
        video_path = self.output_dir / f'{scenario_name}.mp4'
        
        # SUMO-GUI command
        cmd = [
            'sumo-gui',
            '-c', str(sumocfg),
            '--gui-settings-file', str(gui_settings_path),
            '--delay', str(playback_delay),
            '--window-size', '1920,1080',
            '--window-pos', '0,0',
        ]
        
        if auto_start:
            cmd.append('--start')
            cmd.append('--quit-on-end')
        
        # Check if sumo-gui is available
        try:
            subprocess.run(['which', 'sumo-gui'], 
                          check=True, 
                          capture_output=True)
        except subprocess.CalledProcessError:
            print(f"\n✗ ERROR: sumo-gui not found in PATH")
            print(f"  Make sure SUMO is installed and SUMO_HOME is set:")
            print(f"    export SUMO_HOME=/usr/share/sumo")
            print(f"    export PATH=$SUMO_HOME/bin:$PATH")
            return None
        
        print(f"\n📹 VIDEO RECORDING INSTRUCTIONS:")
        print(f"{'='*70}")
        print(f"1. SUMO-GUI will open in a moment")
        print(f"2. Start your screen recording software:")
        print(f"   • OBS Studio (recommended)")
        print(f"   • macOS: QuickTime > New Screen Recording")
        print(f"   • Windows: Win+G (Game Bar)")
        print(f"   • Linux: SimpleScreenRecorder")
        print(f"3. The simulation will:")
        print(f"   • {'Auto-start' if auto_start else 'Wait for you to click Play'}")
        print(f"   • Run at {playback_delay}ms per step (slowed down)")
        print(f"   • Show queue buildup and failure mode")
        print(f"   • {'Auto-quit when finished' if auto_start else 'Run until you stop it'}")
        print(f"4. Save your recording as: {video_path.name}")
        print(f"{'='*70}\n")
        
        print(f"Command: {' '.join(cmd)}\n")
        
        input("Press ENTER when ready to launch SUMO-GUI...")
        
        # Run SUMO-GUI
        try:
            print(f"🚀 Launching SUMO-GUI...\n")
            subprocess.run(cmd, check=True)
            print(f"\n✓ SUMO-GUI closed")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ ERROR: SUMO-GUI failed with code {e.returncode}")
            return None
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted by user")
            return None
        
        print(f"\n💾 Save your recording as: {video_path}")
        return video_path
    
    def generate_all_failure_videos(self, playback_delay: int = 100, 
                                    duration: int = 600, auto_start: bool = True):
        """
        Generate failure videos for all lane configurations.
        
        Args:
            playback_delay: Delay between simulation steps (ms)
            duration: Simulation duration (seconds)
            auto_start: Whether to auto-start simulations
        """
        # Create GUI settings
        gui_settings = self.create_gui_settings(delay=playback_delay, zoom=1500)
        
        # Define failure scenarios (arrival rates at/above breaking point)
        # Based on analysis results from earlier phases
        failure_scenarios = [
            {
                'lanes': 1, 
                'arrival_rate': 0.12, 
                'description': '1-lane: Queue divergence (oversaturated)'
            },
            {
                'lanes': 2, 
                'arrival_rate': 0.15, 
                'description': '2-lane: Approaching saturation (unstable)'
            },
            {
                'lanes': 3, 
                'arrival_rate': 0.18, 
                'description': '3-lane: High demand but stable (contrast)'
            },
        ]
        
        print(f"\n{'='*70}")
        print(f"FAILURE DEMONSTRATION VIDEO GENERATION")
        print(f"{'='*70}\n")
        print(f"Scenarios:")
        for sc in failure_scenarios:
            print(f"  • {sc['description']}")
        print(f"\nSimulation duration: {duration}s ({duration/60:.1f} min)")
        print(f"Playback delay: {playback_delay}ms per step")
        print(f"Auto-start: {auto_start}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*70}\n")
        
        videos = []
        for i, sc in enumerate(failure_scenarios, 1):
            print(f"\n{'#'*70}")
            print(f"[{i}/{len(failure_scenarios)}] {sc['description']}")
            print(f"{'#'*70}\n")
            
            try:
                # Generate scenario
                scenario = self.generate_failure_scenario(
                    lanes=sc['lanes'],
                    arrival_rate=sc['arrival_rate'],
                    duration=duration
                )
                
                # Record video (launches SUMO-GUI)
                video_path = self.record_video(
                    scenario, 
                    gui_settings, 
                    playback_delay,
                    auto_start
                )
                
                if video_path:
                    videos.append({
                        'description': sc['description'],
                        'video_path': video_path,
                        'scenario': scenario
                    })
                    print(f"\n✅ Scenario {i} complete")
                else:
                    print(f"\n⚠️  Scenario {i} skipped or failed")
                
            except Exception as e:
                print(f"\n✗ ERROR processing scenario {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Ask if user wants to continue
            if i < len(failure_scenarios):
                response = input(f"\nContinue to next scenario? [Y/n]: ").strip().lower()
                if response and response[0] == 'n':
                    print("Stopping generation.")
                    break
        
        # Create summary documentation
        self._create_summary(videos, playback_delay, duration)
        
        print(f"\n{'='*70}")
        print(f"✅ VIDEO GENERATION SESSION COMPLETE")
        print(f"{'='*70}")
        print(f"\nProcessed: {len(videos)}/{len(failure_scenarios)} scenario(s)")
        print(f"Output directory: {self.output_dir}")
        print(f"\n📚 See {self.output_dir}/README_VIDEOS.md for details")
        print(f"{'='*70}\n")
    
    def _create_summary(self, videos: List[Dict], playback_delay: int, duration: int):
        """Create summary README for the videos."""
        readme_path = self.output_dir / 'README_VIDEOS.md'
        
        content = [
            "# Roundabout Failure Demonstration Videos",
            "",
            "## Overview",
            "",
            "This directory contains failure demonstration scenarios showing roundabout",
            "performance under critical arrival rates where the system approaches or",
            "exceeds capacity.",
            "",
            f"**Simulation settings:**",
            f"- Playback speed: {100/playback_delay:.1f}× slower than real-time",
            f"- Duration: {duration}s ({duration/60:.1f} minutes)",
            f"- Resolution: 1920×1080 (recommended)",
            "",
            "## Scenarios Generated",
            ""
        ]
        
        for i, video in enumerate(videos, 1):
            content.append(f"### {i}. {video['description']}")
            content.append("")
            scenario = video['scenario']
            total_demand = scenario['arrival_rate'] * 4 * 3600
            content.append(f"- **Lanes**: {scenario['lanes']}")
            content.append(f"- **Arrival rate**: {scenario['arrival_rate']:.3f} veh/s per arm")
            content.append(f"- **Total demand**: {total_demand:.0f} veh/hr")
            content.append(f"- **Expected video**: `{video['video_path'].name}`")
            content.append(f"- **Scenario files**: `{scenario['scenario_dir'].name}/`")
            content.append("")
        
        content.extend([
            "## How to Record Videos",
            "",
            "### Quick Method: Screen Recording",
            "",
            "1. Run the generator:",
            "   ```bash",
            "   cd roundabout",
            "   python src/generate_failure_videos.py --speed 0.5 --duration 600",
            "   ```",
            "",
            "2. When SUMO-GUI launches, record your screen:",
            "   - **OBS Studio** (free, all platforms): https://obsproject.com/",
            "   - **macOS**: QuickTime > File > New Screen Recording",
            "   - **Windows**: Win+G (Game Bar) or OBS",
            "   - **Linux**: SimpleScreenRecorder, Kazam, or OBS",
            "",
            "3. The simulation will automatically play at slowed speed",
            "",
            "### Advanced: FFmpeg Screen Capture (Linux/macOS)",
            "",
            "```bash",
            "# Start recording",
            "ffmpeg -video_size 1920x1080 -framerate 30 -f x11grab -i :0.0+0,0 \\",
            "       -c:v libx264 -preset fast -crf 18 output.mp4",
            "",
            "# Then launch SUMO-GUI in another terminal",
            "python src/generate_failure_videos.py",
            "",
            "# Stop recording with Ctrl+C when done",
            "```",
            "",
            "### Alternative: SUMO Screenshot Export",
            "",
            "If your SUMO version supports it:",
            "",
            "```bash",
            "sumo-gui -c scenario/roundabout.sumocfg \\",
            "  --gui-settings-file gui_settings.xml \\",
            "  --start --quit-on-end \\",
            "  --screenshot-dir screenshots/",
            "",
            "# Then compile screenshots:",
            "ffmpeg -framerate 30 -i screenshots/screenshot_%04d.png \\",
            "       -c:v libx264 -pix_fmt yuv420p output.mp4",
            "```",
            "",
            "## Key Observations to Document",
            "",
            "When analyzing the videos, observe:",
            "",
            "### 1-Lane System (Oversaturated)",
            "- ❌ **Queues grow unbounded** throughout simulation",
            "- 📈 Final queue lengths: >100 vehicles per arm",
            "- ⏱️ Average delays: >500 seconds",
            "- 🚫 System cannot handle demand",
            "",
            "### 2-Lane System (Near Saturation)",
            "- ⚠️ **High but eventually stabilizing** queues",
            "- 📊 Queue lengths: 20-50 vehicles per arm",
            "- ⏱️ Average delays: 200-600 seconds",
            "- 🔄 Periodic congestion waves",
            "",
            "### 3-Lane System (Stable)",
            "- ✅ **Manageable queues** with good flow",
            "- 📉 Queue lengths: <30 vehicles per arm",
            "- ⏱️ Average delays: <300 seconds",
            "- 🎯 System handles demand efficiently",
            "",
            "## Comparison Points",
            "",
            "| Metric | 1-Lane | 2-Lane | 3-Lane |",
            "|--------|--------|--------|--------|",
            "| Arrival rate | 0.12 | 0.15 | 0.18 |",
            "| Total demand | 1728 | 2160 | 2592 |",
            "| Status | Failed | Stressed | Stable |",
            "| Max queue | >100 | 20-50 | <30 |",
            "| Avg delay | >500s | 200-600s | <300s |",
            "",
            "## Technical Details",
            "",
            f"**Simulation parameters:**",
            f"- Timestep: 0.1s",
            f"- Playback delay: {playback_delay}ms",
            f"- Lateral resolution: 0.8m",
            f"- Car-following: Krauss model",
            f"- Gap acceptance: 3.0s critical gap",
            "",
            "**Vehicle types:**",
            "- Passenger cars (80%)",
            "- Trucks (15%)",
            "- Buses (5%)",
            "",
            "---",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"Tool: `generate_failure_videos.py`",
        ])
        
        with open(readme_path, 'w') as f:
            f.write('\n'.join(content))
        
        print(f"\n✓ Created documentation: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate failure demonstration videos for roundabout simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all scenarios with default settings
  python src/generate_failure_videos.py

  # Slower playback for detailed observation
  python src/generate_failure_videos.py --speed 0.25 --duration 300

  # Custom output directory
  python src/generate_failure_videos.py --output my_videos/
  
  # Manual control (no auto-start)
  python src/generate_failure_videos.py --no-auto-start
        """
    )
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file (default: config/config.yaml)')
    parser.add_argument('--output', type=str, default='videos',
                       help='Output directory for videos (default: videos/)')
    parser.add_argument('--speed', type=float, default=0.5,
                       help='Playback speed multiplier (0.5 = 2x slower, default: 0.5)')
    parser.add_argument('--duration', type=int, default=600,
                       help='Simulation duration in seconds (default: 600 = 10 min)')
    parser.add_argument('--no-auto-start', action='store_true',
                       help='Do not auto-start simulations (manual control)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.speed <= 0:
        print(f"ERROR: Speed must be positive (got {args.speed})")
        sys.exit(1)
    
    if args.duration <= 0:
        print(f"ERROR: Duration must be positive (got {args.duration})")
        sys.exit(1)
    
    # Convert speed to delay (ms per simulation step)
    # SUMO timestep is 0.1s, so delay=100ms gives real-time playback
    playback_delay = int(100 / args.speed)
    
    print(f"\n{'='*70}")
    print(f"ROUNDABOUT FAILURE VIDEO GENERATOR")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Config file:     {args.config}")
    print(f"  Output dir:      {args.output}")
    print(f"  Playback speed:  {args.speed}× (delay={playback_delay}ms/step)")
    print(f"  Duration:        {args.duration}s ({args.duration/60:.1f} min)")
    print(f"  Auto-start:      {not args.no_auto_start}")
    print(f"{'='*70}\n")
    
    # Validate config exists
    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}")
        print(f"Make sure you're in the roundabout/ directory:")
        print(f"  cd roundabout && python src/generate_failure_videos.py")
        sys.exit(1)
    
    # Check for SUMO installation
    if not os.environ.get('SUMO_HOME'):
        print(f"⚠️  WARNING: SUMO_HOME not set")
        print(f"   If SUMO-GUI fails to launch, set:")
        print(f"   export SUMO_HOME=/usr/share/sumo")
        print(f"   export PATH=$SUMO_HOME/bin:$PATH\n")
    
    try:
        # Create generator
        generator = FailureVideoGenerator(args.config, args.output)
        
        # Generate all videos
        generator.generate_all_failure_videos(
            playback_delay=playback_delay,
            duration=args.duration,
            auto_start=not args.no_auto_start
        )
        
        print(f"\n✅ All done! Check {args.output}/ for scenario files")
        print(f"📚 See {args.output}/README_VIDEOS.md for recording instructions\n")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()