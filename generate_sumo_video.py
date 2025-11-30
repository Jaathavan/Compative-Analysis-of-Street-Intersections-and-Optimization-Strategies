#!/usr/bin/env python3
"""
generate_sumo_video.        net_cmd = [
            'python3', 'roundabout/src/generate_network.py',
            '--config', 'roundabout/config/config.yaml',
            '--diameter', str(self.diameter),
            '--lanes', str(self.lanes),
            '--output', str(self.config_dir)
        ]UMO Simulation Video Generator
==========================================================

Generates video recordings of SUMO roundabout simulations using sumo-gui.

Usage:
    python generate_sumo_video.py --lanes 2 --diameter 40 --arrival 0.15 --duration 300
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time


class SUMOVideoGenerator:
    """Generates SUMO simulation videos."""
    
    def __init__(self, lanes: int, diameter: int, arrival_rate: float, 
                 duration: int, output_dir: str):
        """Initialize video generator."""
        self.lanes = lanes
        self.diameter = diameter
        self.arrival_rate = arrival_rate
        self.duration = duration
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_name = f"video_{lanes}lane_d{diameter}_arr{arrival_rate:.2f}"
        self.config_dir = self.output_dir / self.config_name
        
    def generate_network(self):
        """Generate SUMO network."""
        print(f"\n{'='*70}")
        print(f"Generating Network")
        print(f"{'='*70}")
        print(f"  Lanes: {self.lanes}")
        print(f"  Diameter: {self.diameter}m")
        print(f"  Arrival Rate: {self.arrival_rate} veh/s/arm")
        
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        net_output = self.config_dir / 'roundabout.net.xml'
        
        cmd = [
            'python3', 'roundabout/src/generate_network.py',
            '--diameter', str(self.diameter),
            '--lanes', str(self.lanes),
            '--output', str(net_output)
        ]
        
        try:
            subprocess.run(cmd, check=True, timeout=30)
            print(f"  ✓ Network generated: {net_output}")
            return True
        except Exception as e:
            print(f"  ✗ Error generating network: {e}")
            return False
    
    def generate_routes(self):
        """Generate SUMO routes."""
        print(f"\n{'='*70}")
        print(f"Generating Routes")
        print(f"{'='*70}")
        
        # Convert veh/s to veh/hr
        demand_vehhr = self.arrival_rate * 3600
        
        route_output = self.config_dir / 'routes.rou.xml'
        
        cmd = [
            'python3', 'roundabout/src/generate_routes.py',
            '--demand', str(demand_vehhr),
            '--duration', str(self.duration),
            '--output', str(route_output)
        ]
        
        try:
            subprocess.run(cmd, check=True, timeout=30)
            print(f"  ✓ Routes generated: {route_output}")
            return True
        except Exception as e:
            print(f"  ✗ Error generating routes: {e}")
            return False
    
    def create_config_file(self):
        """Create SUMO configuration file."""
        print(f"\n{'='*70}")
        print(f"Creating Configuration File")
        print(f"{'='*70}")
        
        sumocfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="roundabout.net.xml"/>
        <route-files value="routes.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{self.duration}"/>
        <step-length value="0.1"/>
    </time>
    <output>
        <summary-output value="summary.xml"/>
        <tripinfo-output value="tripinfo.xml"/>
    </output>
    <gui_only>
        <start value="true"/>
        <quit-on-end value="true"/>
        <delay value="50"/>
    </gui_only>
</configuration>"""
        
        config_path = self.config_dir / 'roundabout.sumocfg'
        with open(config_path, 'w') as f:
            f.write(sumocfg_content)
        
        print(f"  ✓ Config file created: {config_path}")
        return config_path
    
    def create_view_settings(self):
        """Create view settings for better visualization."""
        view_settings = """<?xml version="1.0" encoding="UTF-8"?>
<viewsettings>
    <scheme name="real world"/>
    <delay value="50"/>
    <viewport y="0" x="0" zoom="100"/>
    <snapshot file="screenshot.png"/>
</viewsettings>"""
        
        view_path = self.config_dir / 'view.xml'
        with open(view_path, 'w') as f:
            f.write(view_settings)
        
        return view_path
    
    def run_sumo_gui(self):
        """Run SUMO with GUI for visualization/recording."""
        print(f"\n{'='*70}")
        print(f"Launching SUMO-GUI")
        print(f"{'='*70}")
        print(f"\nInstructions:")
        print(f"  1. SUMO-GUI will open automatically")
        print(f"  2. To record video:")
        print(f"     - Click 'Edit' → 'Edit Visualization'")
        print(f"     - Go to 'OpenGL' tab")
        print(f"     - Enable 'Screenshot' and set filename")
        print(f"     - Click 'Start Simulation' (Play button)")
        print(f"  3. Alternatively, use screen recording software")
        print(f"  4. The simulation will run for {self.duration} seconds")
        print(f"\nStarting in 3 seconds...")
        time.sleep(3)
        
        config_path = self.config_dir / 'roundabout.sumocfg'
        view_path = self.create_view_settings()
        
        cmd = [
            'sumo-gui',
            '-c', str(config_path),
            '--gui-settings-file', str(view_path),
            '--start', 'true',
            '--delay', '50',
            '--quit-on-end', 'false'
        ]
        
        print(f"\nRunning: {' '.join(cmd)}\n")
        
        try:
            subprocess.run(cmd)
            print(f"\n✓ Simulation complete")
            return True
        except KeyboardInterrupt:
            print(f"\n✗ Simulation interrupted by user")
            return False
        except Exception as e:
            print(f"\n✗ Error running SUMO-GUI: {e}")
            return False
    
    def generate_screenshots(self):
        """Generate screenshots at key time points."""
        print(f"\n{'='*70}")
        print(f"Generating Screenshots")
        print(f"{'='*70}")
        
        config_path = self.config_dir / 'roundabout.sumocfg'
        screenshots_dir = self.config_dir / 'screenshots'
        screenshots_dir.mkdir(exist_ok=True)
        
        # Time points to capture (in seconds)
        time_points = [0, 60, 120, 180, 240, 300]
        
        for t in time_points[:min(len(time_points), self.duration//60 + 1)]:
            print(f"  Capturing at t={t}s...")
            
            screenshot_path = screenshots_dir / f'screenshot_t{t:04d}.png'
            
            cmd = [
                'sumo-gui',
                '-c', str(config_path),
                '--start', 'true',
                '--quit-on-end', 'true',
                '--no-warnings', 'true',
                '--step-length', '1.0',
                '--begin', str(t),
                '--end', str(t + 1),
                '--window-size', '1920,1080',
                '--screenshot', str(screenshot_path)
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, timeout=30)
                if screenshot_path.exists():
                    print(f"    ✓ {screenshot_path}")
                else:
                    print(f"    ✗ Failed to create screenshot")
            except Exception as e:
                print(f"    ✗ Error: {e}")
        
        print(f"\n✓ Screenshots saved to: {screenshots_dir}")
    
    def generate_video_with_ffmpeg(self):
        """Generate video from screenshots using ffmpeg."""
        print(f"\n{'='*70}")
        print(f"Creating Video with ffmpeg")
        print(f"{'='*70}")
        
        screenshots_dir = self.config_dir / 'screenshots'
        if not screenshots_dir.exists():
            print("  ✗ No screenshots directory found. Run with --screenshots first.")
            return False
        
        video_output = self.output_dir / f'{self.config_name}.mp4'
        
        cmd = [
            'ffmpeg',
            '-framerate', '30',
            '-pattern_type', 'glob',
            '-i', str(screenshots_dir / '*.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-y',
            str(video_output)
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"  ✓ Video created: {video_output}")
            return True
        except FileNotFoundError:
            print("  ✗ ffmpeg not found. Install with: sudo apt install ffmpeg")
            return False
        except Exception as e:
            print(f"  ✗ Error creating video: {e}")
            return False


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Generate SUMO simulation video"
    )
    parser.add_argument('--lanes', type=int, default=2, choices=[1, 2, 3],
                       help='Number of lanes per approach')
    parser.add_argument('--diameter', type=int, default=40,
                       help='Roundabout diameter (m)')
    parser.add_argument('--arrival', type=float, default=0.15,
                       help='Arrival rate (veh/s per arm)')
    parser.add_argument('--duration', type=int, default=300,
                       help='Simulation duration (s)')
    parser.add_argument('--output-dir', type=str, default='results/sumo_videos',
                       help='Output directory')
    parser.add_argument('--mode', type=str, default='gui',
                       choices=['gui', 'screenshots', 'video'],
                       help='Mode: gui (interactive), screenshots, or video')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SUMO VIDEO GENERATOR")
    print("="*70)
    
    generator = SUMOVideoGenerator(
        lanes=args.lanes,
        diameter=args.diameter,
        arrival_rate=args.arrival,
        duration=args.duration,
        output_dir=args.output_dir
    )
    
    # Generate network and routes
    if not generator.generate_network():
        sys.exit(1)
    
    if not generator.generate_routes():
        sys.exit(1)
    
    generator.create_config_file()
    
    # Run based on mode
    if args.mode == 'gui':
        generator.run_sumo_gui()
    elif args.mode == 'screenshots':
        generator.generate_screenshots()
    elif args.mode == 'video':
        generator.generate_screenshots()
        generator.generate_video_with_ffmpeg()
    
    print("\n" + "="*70)
    print("VIDEO GENERATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
