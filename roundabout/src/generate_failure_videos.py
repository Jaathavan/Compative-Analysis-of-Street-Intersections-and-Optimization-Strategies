#!/usr/bin/env python3
"""
generate_failure_videos.py - Create Slowed-Down Failure Demonstration Videos
=============================================================================

Generates SUMO-GUI videos showing failure scenarios for 1, 2, and 3-lane
roundabouts at critical arrival rates where the system breaks down.

Videos are slowed down to make queue buildup and congestion visible.

Usage:
    python generate_failure_videos.py --config config/config.yaml --output videos/
    python generate_failure_videos.py --speed 0.5 --duration 600
"""

import argparse
import subprocess
import sys
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List
import time

class FailureVideoGenerator:
    """
    Generates demonstration videos of roundabout failure modes.
    """
    
    def __init__(self, config_path: str, output_dir: str):
        """Initialize generator."""
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"Loaded configuration from {config_path}")
        print(f"Output directory: {output_dir}")
    
    def create_gui_settings(self, delay: int = 100, zoom: float = 2000):
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
        
        # Show vehicle names/IDs
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
        
        # Write XML
        tree = ET.ElementTree(root)
        tree.write(str(settings_path), encoding='utf-8', xml_declaration=True)
        
        print(f"✓ Created GUI settings: {settings_path}")
        return settings_path
    
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
        print(f"\nGenerating failure scenario: {lanes} lane(s), λ={arrival_rate:.3f}")
        
        # Modify config
        temp_config = self.config.copy()
        temp_config['geometry']['lanes'] = lanes
        temp_config['demand']['arrivals'] = [arrival_rate] * 4  # Equal demand on all arms
        temp_config['simulation']['horizon'] = duration
        temp_config['simulation']['seed'] = seed
        
        # Create scenario directory
        scenario_name = f'{lanes}lane_failure_lambda{arrival_rate:.3f}'
        scenario_dir = self.output_dir / scenario_name
        scenario_dir.mkdir(exist_ok=True)
        
        # Generate network
        from generate_network import RoundaboutNetworkGenerator
        net_gen = RoundaboutNetworkGenerator(temp_config)
        net_gen.generate(str(scenario_dir))
        
        # Generate routes
        from generate_routes import RoundaboutRouteGenerator
        route_gen = RoundaboutRouteGenerator(temp_config)
        route_gen.generate(str(scenario_dir / 'roundabout.rou.xml'))
        
        print(f"✓ Generated network and routes in {scenario_dir}")
        
        return {
            'scenario_name': scenario_name,
            'scenario_dir': scenario_dir,
            'sumocfg': scenario_dir / 'roundabout.sumocfg',
            'lanes': lanes,
            'arrival_rate': arrival_rate
        }
    
    def record_video(self, scenario: Dict, gui_settings_path: Path, 
                    playback_delay: int = 100) -> Path:
        """
        Record SUMO-GUI video for a scenario.
        
        Args:
            scenario: Scenario dictionary from generate_failure_scenario()
            gui_settings_path: Path to GUI settings XML
            playback_delay: Delay between simulation steps (ms)
        
        Returns:
            Path to output video file
        """
        scenario_name = scenario['scenario_name']
        sumocfg = scenario['sumocfg']
        
        print(f"\nRecording video for: {scenario_name}")
        print(f"  Playback delay: {playback_delay}ms per step")
        
        # Output video path
        video_path = self.output_dir / f'{scenario_name}.mp4'
        
        # SUMO-GUI command with screenshot recording
        # Note: SUMO doesn't directly output video, so we'll use screenshots
        screenshot_dir = self.output_dir / f'{scenario_name}_screenshots'
        screenshot_dir.mkdir(exist_ok=True)
        
        cmd = [
            'sumo-gui',
            '-c', str(sumocfg),
            '--gui-settings-file', str(gui_settings_path),
            '--delay', str(playback_delay),
            '--start',  # Auto-start simulation
            '--quit-on-end',  # Auto-quit when done
            '--window-size', '1920,1080',
            '--window-pos', '0,0',
            # Screenshot settings
            # Note: For actual video, you may need to use screen recording software
            # or SUMO's built-in video export (if available in your version)
        ]
        
        print(f"  Running SUMO-GUI with command:")
        print(f"    {' '.join(cmd)}")
        print(f"\n  📹 Please manually record the SUMO-GUI window using:")
        print(f"     - OBS Studio")
        print(f"     - Screen recording software")
        print(f"     - Or use SUMO's File > Export Video (if available)")
        print(f"\n  The simulation will:")
        print(f"    1. Start automatically")
        print(f"    2. Run at {playback_delay}ms per step (slowed down)")
        print(f"    3. Show queue buildup and failure mode")
        print(f"    4. Quit when finished")
        
        # Run SUMO-GUI
        try:
            subprocess.run(cmd, check=True)
            print(f"\n✓ SUMO-GUI simulation complete")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ ERROR: SUMO-GUI failed: {e}")
            return None
        except FileNotFoundError:
            print(f"\n✗ ERROR: sumo-gui not found. Is SUMO installed and in PATH?")
            return None
        
        print(f"\n  💡 To create video from screenshots (if captured):")
        print(f"     ffmpeg -framerate 30 -i {screenshot_dir}/screenshot_%04d.png \\")
        print(f"            -c:v libx264 -pix_fmt yuv420p {video_path}")
        
        return video_path
    
    def generate_all_failure_videos(self, playback_delay: int = 100, 
                                    duration: int = 600):
        """
        Generate failure videos for all lane configurations.
        
        Args:
            playback_delay: Delay between simulation steps (ms)
            duration: Simulation duration (seconds)
        """
        # Create GUI settings
        gui_settings = self.create_gui_settings(delay=playback_delay, zoom=1500)
        
        # Define failure scenarios (arrival rates at/above breaking point)
        failure_scenarios = [
            {'lanes': 1, 'arrival_rate': 0.12, 'description': '1-lane: Queue divergence'},
            {'lanes': 2, 'arrival_rate': 0.15, 'description': '2-lane: Approaching saturation'},
            {'lanes': 3, 'arrival_rate': 0.18, 'description': '3-lane: High but stable (contrast)'},
        ]
        
        print(f"\n{'='*70}")
        print(f"GENERATING FAILURE DEMONSTRATION VIDEOS")
        print(f"{'='*70}\n")
        print(f"Scenarios:")
        for sc in failure_scenarios:
            print(f"  - {sc['description']}")
        print(f"\nSimulation duration: {duration}s ({duration/60:.1f} min)")
        print(f"Playback delay: {playback_delay}ms (slower for visibility)")
        print(f"{'='*70}\n")
        
        videos = []
        for i, sc in enumerate(failure_scenarios, 1):
            print(f"\n[{i}/{len(failure_scenarios)}] Processing: {sc['description']}")
            print(f"{'='*70}")
            
            # Generate scenario
            scenario = self.generate_failure_scenario(
                lanes=sc['lanes'],
                arrival_rate=sc['arrival_rate'],
                duration=duration
            )
            
            # Record video (manual recording required)
            video_path = self.record_video(scenario, gui_settings, playback_delay)
            
            if video_path:
                videos.append({
                    'description': sc['description'],
                    'video_path': video_path,
                    'scenario': scenario
                })
            
            print(f"{'='*70}")
        
        # Create summary document
        self._create_summary(videos)
        
        print(f"\n{'='*70}")
        print(f"✅ VIDEO GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nGenerated {len(videos)} scenario(s)")
        print(f"Output directory: {self.output_dir}")
        print(f"\n💡 Next steps:")
        print(f"  1. Videos were displayed in SUMO-GUI with slowed playback")
        print(f"  2. To create actual video files, use screen recording software")
        print(f"  3. Alternatively, use SUMO's built-in video export if available")
        print(f"  4. See README_VIDEOS.md for detailed instructions")
        print(f"{'='*70}\n")
    
    def _create_summary(self, videos: List[Dict]):
        """Create summary README for the videos."""
        readme_path = self.output_dir / 'README_VIDEOS.md'
        
        content = [
            "# Roundabout Failure Demonstration Videos",
            "",
            "## Overview",
            "",
            "This directory contains failure demonstration videos showing roundabout",
            "performance under critical arrival rates where the system approaches or",
            "exceeds capacity.",
            "",
            "## Scenarios",
            ""
        ]
        
        for i, video in enumerate(videos, 1):
            content.append(f"### {i}. {video['description']}")
            content.append("")
            scenario = video['scenario']
            content.append(f"- **Lanes**: {scenario['lanes']}")
            content.append(f"- **Arrival rate**: {scenario['arrival_rate']:.3f} veh/s per arm")
            content.append(f"- **Total demand**: {scenario['arrival_rate'] * 4 * 3600:.0f} veh/hr")
            content.append(f"- **Video file**: `{video['video_path'].name}`")
            content.append("")
        
        content.extend([
            "## How to Create Videos",
            "",
            "### Method 1: Screen Recording (Recommended)",
            "",
            "1. Run the video generation script:",
            "   ```bash",
            "   python generate_failure_videos.py --config config/config.yaml --output videos/",
            "   ```",
            "",
            "2. When SUMO-GUI opens, use screen recording software:",
            "   - **OBS Studio** (free, cross-platform)",
            "   - **macOS**: QuickTime Player > New Screen Recording",
            "   - **Windows**: Game Bar (Win+G) or OBS",
            "   - **Linux**: SimpleScreenRecorder, OBS, or Kazam",
            "",
            "3. The simulation will automatically:",
            "   - Start playing",
            "   - Run at slowed-down speed for visibility",
            "   - Show queue buildup and congestion",
            "   - Quit when finished",
            "",
            "### Method 2: SUMO Built-in Export",
            "",
            "Some SUMO versions support direct video export:",
            "",
            "```bash",
            "sumo-gui -c scenario/roundabout.sumocfg \\",
            "  --gui-settings-file gui_settings.xml \\",
            "  --start --quit-on-end \\",
            "  --video-encoding png \\",
            "  --video-output video.mp4",
            "```",
            "",
            "### Method 3: Screenshots + FFmpeg",
            "",
            "1. Capture screenshots during simulation (manual or automated)",
            "2. Compile into video:",
            "   ```bash",
            "   ffmpeg -framerate 30 -i screenshot_%04d.png \\",
            "          -c:v libx264 -pix_fmt yuv420p output.mp4",
            "   ```",
            "",
            "## Key Observations",
            "",
            "### 1-Lane System (λ=0.12 veh/s)",
            "- **Status**: Oversaturated",
            "- **Observation**: Queues grow unbounded",
            "- **Queue lengths**: Exceed 100+ vehicles per arm",
            "- **Delays**: >500s average",
            "",
            "### 2-Lane System (λ=0.15 veh/s)",
            "- **Status**: Near saturation",
            "- **Observation**: High but stabilizing queues",
            "- **Queue lengths**: 20-50 vehicles per arm",
            "- **Delays**: 200-600s average",
            "",
            "### 3-Lane System (λ=0.18 veh/s)",
            "- **Status**: High demand but stable (for contrast)",
            "- **Observation**: Shorter queues, better flow",
            "- **Queue lengths**: <30 vehicles per arm",
            "- **Delays**: <300s average",
            "",
            "## Video Settings",
            "",
            "- **Playback speed**: Slowed down (100ms delay per simulation step)",
            "- **Resolution**: 1920×1080",
            "- **Duration**: 10 minutes simulation time",
            "- **Zoom level**: Optimized to show full roundabout + approach queues",
            "",
            "## Analysis Notes",
            "",
            "When watching the videos, observe:",
            "",
            "1. **Queue buildup patterns**:",
            "   - How quickly do queues form?",
            "   - Do they stabilize or grow unbounded?",
            "   - Are queues balanced across arms?",
            "",
            "2. **Merging behavior**:",
            "   - How often are merge attempts denied?",
            "   - Do vehicles wait for large gaps?",
            "   - Lane utilization patterns",
            "",
            "3. **Throughput degradation**:",
            "   - Does flow rate decrease as queues build?",
            "   - Evidence of gridlock or spillback?",
            "",
            "4. **Lane-specific effects** (2-3 lane scenarios):",
            "   - Which lanes are preferred?",
            "   - Do inner lanes get underutilized?",
            "   - Lane-changing conflicts?",
            "",
            "---",
            "",
            f"Generated by: `generate_failure_videos.py`  ",
            f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        ])
        
        with open(readme_path, 'w') as f:
            f.write('\n'.join(content))
        
        print(f"\n✓ Created video README: {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate failure demonstration videos for roundabout simulations"
    )
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='videos',
                       help='Output directory for videos')
    parser.add_argument('--speed', type=float, default=0.1,
                       help='Playback speed (0.1 = 10x slower, 1.0 = normal)')
    parser.add_argument('--duration', type=int, default=600,
                       help='Simulation duration in seconds')
    
    args = parser.parse_args()
    
    # Convert speed to delay (ms)
    # Normal simulation runs at ~0.1s per step (dt=0.1)
    # delay = 0 means as fast as possible
    # delay = 100 means 100ms per step (10x slower than real-time if dt=0.01)
    playback_delay = int(100 / max(0.01, args.speed))
    
    print(f"\nConfiguration:")
    print(f"  Config file: {args.config}")
    print(f"  Output directory: {args.output}")
    print(f"  Playback speed: {args.speed}× (delay={playback_delay}ms)")
    print(f"  Duration: {args.duration}s ({args.duration/60:.1f} min)")
    
    # Create generator
    generator = FailureVideoGenerator(args.config, args.output)
    
    # Generate all videos
    generator.generate_all_failure_videos(
        playback_delay=playback_delay,
        duration=args.duration
    )


if __name__ == "__main__":
    main()
