# God's Dice - A Quantitative Stock Market Model Based on Behavioral Finance

![GitHub](https://img.shields.io/github/license/your-username/gods-dice)

> "God does not play dice." - Albert Einstein

This is a quantitative simulation system for studying randomness in financial markets. By simulating different types of investor behaviors and simplified market mechanisms, it researches: 1. Randomness and butterfly effect in financial markets; 2. Impact of different investor behaviors on financial markets; 3. Performance of different investor types in financial markets.

## Project Features

- **Multiple Investor Types**:
  - Value Investor
  - Chase Investor
  - Trend Investor
  - Random Investor
  - Insider Investor
  - Message Investor
  - Never Stop Loss Investor
  - Bottom Fishing Investor

- **Realistic Market Mechanisms**:
  - Call auction pricing mechanism
  - Trading volume and liquidity impact
  - Transaction fee system
  - Price discovery process
  - Capital injection and withdrawal mechanisms

- **Comprehensive Analysis Tools**:
  - Monte Carlo simulation
  - Butterfly effect analysis
  - Market impact measurement
  - Price deviation statistics
  - Detailed trading logs

## Core Modules

### 1. Market Simulation
- `main_ca_3.2.1.1.py`: Call auction trading mechanism (legacy version)
- `main_OHLC_2.0.5.1.py`: Core market simulation engine

### 2. Butterfly Effect Studies
- `butterfly_effect_simulation_1.*.py`: Butterfly effect simulation and related analysis

### 3. Monte Carlo Simulation
- `MC_simulation_1.0-1.4.py`: Monte Carlo method implementations
- `MC_simulation_asset_return_analysis_1.*.py`: Asset return analysis for different investor types
- `MC_simulation_bias_std.py`: Market impact study of value investors
- `MC_simulation_chase_period.py`: Market impact study of chase investors
- `MC_simulation_trend_period.py`: Market impact study of trend investors

### 4. Brownian Motion Research
- `brownian_motion_test_1.0.py`: Brownian motion verification tool

## Requirements
- Python 3.7+
- NumPy
- Matplotlib
- Pandas (optional, for data analysis)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments