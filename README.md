# Comparative Analysis of Street Intersections and Optimization Strategies

A comprehensive study comparing the performance of roundabouts and signalized intersections under various traffic conditions using both our own text-based simulator and a microscopic traffic simulation (SUMO).

## Project Overview

This project implements and compares two intersection control strategies (note signalized was omitted in final paper due to limited time):

1. **Roundabout**: Unsignalized circular intersections with priority-based gap acceptance
2. **Signalized Intersection**: Fixed-time traffic signal control with dedicated phases

The analysis uses:
- **Text-based Simulations** (implemented using DDE's (delay differentiable equations), IDM (intelligent driver module), poisson process, ... )
- **Microscopic simulation** (SUMO - Simulation of Urban MObility)
- **Performance metrics**: throughput, delay, queue length

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

## Final Report

The complete analysis, methodology, and findings are documented in the [Final Report (PDF)](./Final_Report.pdf), which includes:

- Detailed comparison of roundabout intersection performance
- Mathematical models and simulation methodologies
- Comprehensive results analysis and visualization
- Conclusions and recommendations for intersection optimization strategies