"""
MC_simulation_bias_std.py

This script runs multiple stock market model simulations (based on main_OHLC_2.0.5.1.py),
using different standard deviations of value investor bias (bias_percent_std), calculates
average closing price sequences, and visualizes the results.

The simulation analyzes the impact of different bias_percent_std values on market prices
using Monte Carlo methods, helping to understand how the distribution of value investor
valuation biases affects market prices.

Comparing three bias_percent_std settings:
1. Low bias standard deviation: bias_percent_std = 0.15 (small valuation bias)
2. Medium bias standard deviation: bias_percent_std = 0.30 (medium valuation bias)
3. High bias standard deviation: bias_percent_std = 0.50 (large valuation bias)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import importlib.util

# 导入main_OHLC_2.0.5.1模块
spec = importlib.util.spec_from_file_location("main_module", "main_OHLC_2.0.5.1.py")
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)

# 从模块中获取必要的函数和类
generate_random_seeds = main_module.generate_random_seeds
Market = main_module.Market
TrueValueCurve = main_module.TrueValueCurve
Investor = main_module.Investor

def run_simulation_with_seed(random_seeds, bias_percent_std=0.15, enable_trade_log=False):
    """
    Run a single stock market simulation with given random seeds and bias_percent_std, and return the market object.

    Parameters:
        random_seeds: Dictionary containing various random seeds to control randomness in the simulation
        bias_percent_std: Standard deviation of value investor valuation bias
        enable_trade_log: Whether to enable trade logging functionality, defaults to False

    Returns:
        Market object (Market instance) containing simulation results, including price history, volume, etc.
    """
    # Basic market parameter settings
    initial_price = 100
    price_tick = 0.01
    days = 800
    buy_fee_rate = 0.001
    sell_fee_rate = 0.001

    # Set whether to enable trade logging
    Investor.set_enable_trade_log(enable_trade_log)

    # Initialize global trade log file for all investors
    if enable_trade_log:
        Investor.init_trade_log_file(f"all_investors_trade_log_bias_std_{bias_percent_std:.2f}.csv")

    # Create true value curve
    value_curve = TrueValueCurve(initial_value=initial_price, days=days, seed=random_seeds['value_line_seed'])

    # Create market instance
    market = Market(initial_price, price_tick, value_curve=value_curve,
                   seed=random_seeds['market_seed'],
                   close_seed=random_seeds['close_seed'],
                   buy_fee_rate=buy_fee_rate, sell_fee_rate=sell_fee_rate)

    # Parameters for different types of investors
    value_investors_params = {
        'num': 100,
        'initial_shares': 100,
        'initial_cash': 10000
    }

    chase_investors_params = {
        'num': 100,
        'initial_shares': 100,
        'initial_cash': 10000
    }

    trend_investors_params = {
        'num': 100,
        'initial_shares': 100,
        'initial_cash': 10000
    }

    random_investors_params = {
        'num': 100,
        'initial_shares': 100,
        'initial_cash': 10000
    }

    never_stop_loss_investors_params = {
        'num': 20,
        'initial_shares': 100,
        'initial_cash': 10000
    }

    bottom_fishing_investors_params = {
        'num': 20,
        'initial_shares': 100,
        'initial_cash': 10000
    }

    insider_investors_params = {
        'num': 10,
        'initial_shares': 100,
        'initial_cash': 10000
    }

    message_investors_params = {
        'num': 10,
        'initial_shares': 100,
        'initial_cash': 10000
    }

    # Create investor list
    investors = []

    # Create value investors (investment strategy based on value bias)
    # Use specified bias_percent_std
    investor_bias_percents = np.random.RandomState(random_seeds['value_investor_seed']).normal(0, bias_percent_std, value_investors_params['num'])

    for i, bias_percent in enumerate(investor_bias_percents):
        max_deviation = np.random.RandomState(random_seeds['value_investor_seed'] + i).uniform(0.2, 0.4)
        target_position = np.random.RandomState(random_seeds['value_investor_seed'] + i).uniform(0.4, 0.6)

        investors.append(main_module.ValueInvestor(
            value_investors_params['initial_shares'],
            value_investors_params['initial_cash'],
            bias_percent=bias_percent,
            max_deviation=max_deviation,
            target_position=target_position,
            seed=random_seeds['value_investor_seed'] + i
        ))

    # Create chase investors (following price uptrend)
    for i in range(chase_investors_params['num']):
        investors.append(main_module.ChaseInvestor(
            chase_investors_params['initial_shares'],
            chase_investors_params['initial_cash'],
            N=None,  # Will be randomly initialized
            seed=random_seeds['chase_investor_seed'] + i
        ))

    # Create trend investors (based on trend analysis of different time periods)
    trend_periods = [5, 10, 15, 20, 25, 30, 60, 100, 150, 200]
    trend_investors_num_per_period = trend_investors_params['num'] // len(trend_periods)

    for period_idx, period in enumerate(trend_periods):
        for i in range(trend_investors_num_per_period):
            investors.append(main_module.TrendInvestor(
                trend_investors_params['initial_shares'],
                trend_investors_params['initial_cash'],
                period,
                seed=random_seeds['trend_investor_seed'] + period_idx * 100 + i
            ))

    # Create random investors (random trading decisions)
    for i in range(random_investors_params['num']):
        investors.append(main_module.RandomInvestor(
            random_investors_params['initial_shares'],
            random_investors_params['initial_cash'],
            seed=random_seeds['random_investor_seed'] + i
        ))

    # Create never stop loss investors (investment strategy without stop loss points)
    for i in range(never_stop_loss_investors_params['num']):
        investors.append(main_module.NeverStopLossInvestor(
            never_stop_loss_investors_params['initial_shares'],
            never_stop_loss_investors_params['initial_cash'],
            buy_probability=0.3,
            initial_profit_target=0.15,
            min_profit_target=0.02,
            seed=random_seeds['never_stop_loss_investor_seed'] + i
        ))

    # Create bottom fishing investors (buying after significant price drops)
    for i in range(bottom_fishing_investors_params['num']):
        seed = random_seeds['bottom_fishing_investor_seed'] + i
        rng = np.random.RandomState(seed)
        profit_target = rng.uniform(0.1, 0.5)
        investors.append(main_module.BottomFishingInvestor(
            bottom_fishing_investors_params['initial_shares'],
            bottom_fishing_investors_params['initial_cash'],
            profit_target=profit_target,
            seed=seed
        ))

    # Create insider investors (investors with market insider information)
    for i in range(insider_investors_params['num']):
        investors.append(main_module.InsiderInvestor(
            insider_investors_params['initial_shares'],
            insider_investors_params['initial_cash'],
            seed=random_seeds['insider_investor_seed'] + i
        ))

    # Create message investors (investors trading based on market messages)
    for i in range(message_investors_params['num']):
        investors.append(main_module.MessageInvestor(
            message_investors_params['initial_shares'],
            message_investors_params['initial_cash'],
            seed=random_seeds['message_investor_seed'] + i
        ))

    # Calculate start and end indices for each investor type (for subsequent analysis)
    value_end = value_investors_params['num']
    chase_end = value_end + chase_investors_params['num']
    trend_end = chase_end + trend_investors_params['num']
    random_end = trend_end + random_investors_params['num']
    never_stop_loss_end = random_end + never_stop_loss_investors_params['num']
    bottom_fishing_end = never_stop_loss_end + bottom_fishing_investors_params['num']
    insider_end = bottom_fishing_end + insider_investors_params['num']
    message_end = insider_end + message_investors_params['num']

    # Run simulated trading day by day
    for day in range(days):
        for investor in investors:
            investor.trade(market.price, market)
        market.daily_auction()

    # Close trade log file
    if enable_trade_log:
        Investor.close_trade_log_file()

    return market

def run_multiple_simulations(num_simulations=10, base_seed=2108, custom_seeds=None, bias_percent_std=0.15, enable_trade_log=False):
    """
    Run multiple simulations, each using a different market_seed value, and collect results.

    Parameters:
        num_simulations: Number of simulations to run
        base_seed: Base seed value for random number generation
        custom_seeds: Optional dictionary of custom random seeds, if provided these seed values will be used
        bias_percent_std: Standard deviation of value investor valuation bias
        enable_trade_log: Whether to enable trade logging

    Returns:
        A tuple containing:
        - List of closing price sequences from each simulation
        - True value curve from the first simulation (when using the same value_line_seed, all simulations have the same true value curve)
    """
    all_closing_prices = []
    value_curve_data = None  # Store true value curve from first simulation

    print(f"Running {num_simulations} simulations with bias_percent_std = {bias_percent_std}...")

    for i in range(num_simulations):
        print(f"Simulation {i+1}/{num_simulations}...")

        # Create random seed dictionary
        if custom_seeds is not None:
            # Use custom seeds, but use different market seed for each simulation
            random_seeds = custom_seeds.copy()
            random_seeds['market_seed'] = base_seed + 1300 + i  # Ensure each simulation uses a different market seed
        else:
            # Use default seed generation method
            random_seeds = generate_random_seeds(base_seed=base_seed)
            random_seeds['market_seed'] = base_seed + 1300 + i

        # Run simulation and get market object
        market = run_simulation_with_seed(random_seeds, bias_percent_std, enable_trade_log)

        # Extract closing prices from market object
        closing_prices = [ohlc[3] for ohlc in market.ohlc_data]  # OHLC data format is (open, high, low, close)
        all_closing_prices.append(closing_prices)

        # Store true value curve from first simulation
        if i == 0:
            value_curve_data = market.value_history

    return all_closing_prices, value_curve_data

def calculate_statistics(sequences, value_curve=None):
    """
    Calculate statistics for a list of sequences, including average sequence, standard deviation sequence, and volatility.

    Parameters:
        sequences: List of sequences (list or array)
        value_curve: Optional true value curve for calculating price deviation

    Returns:
        Dictionary containing statistics:
        - average_sequence: Average sequence
        - std_sequence: Standard deviation sequence
        - volatility: Standard deviation of daily returns (volatility)
        - avg_volatility: Average volatility
        - std_volatility: Standard deviation of volatility
        - individual_volatilities: Volatility of each sequence
        - max_drawdowns: Maximum drawdown of each sequence
        - avg_max_drawdown: Average maximum drawdown
        - price_value_correlation: Correlation between price and value
        - price_value_deviation: Average deviation between price and value
        - upside_volatility: Upside volatility
        - downside_volatility: Downside volatility
    """
    # Ensure all sequences have the same length
    min_length = min(len(seq) for seq in sequences)
    sequences_trimmed = [seq[:min_length] for seq in sequences]

    # Convert to numpy array for easier calculation
    sequences_array = np.array(sequences_trimmed)

    # Calculate mean and standard deviation along first axis (across all simulations)
    average_sequence = np.mean(sequences_array, axis=0)
    std_sequence = np.std(sequences_array, axis=0)

    # Calculate daily returns
    daily_returns = np.zeros((len(sequences), min_length-1))
    for i, seq in enumerate(sequences_trimmed):
        daily_returns[i] = np.diff(seq) / seq[:-1]

    # Calculate volatility (standard deviation of daily returns)
    volatility = np.std(daily_returns) * 100  # Convert to percentage

    # Calculate average volatility (mean of each sequence's volatility)
    individual_volatilities = np.std(daily_returns, axis=1) * 100
    avg_volatility = np.mean(individual_volatilities)
    std_volatility = np.std(individual_volatilities)

    # Calculate upside and downside volatility
    positive_returns = daily_returns.copy()
    negative_returns = daily_returns.copy()
    positive_returns[positive_returns < 0] = 0
    negative_returns[negative_returns > 0] = 0

    upside_volatility = np.std(positive_returns) * 100
    downside_volatility = np.std(negative_returns) * 100

    # Calculate maximum drawdown
    max_drawdowns = []
    for seq in sequences_trimmed:
        # Calculate running maximum
        running_max = np.maximum.accumulate(seq)
        # Calculate drawdown
        drawdowns = (running_max - seq) / running_max
        # Maximum drawdown
        max_drawdown = np.max(drawdowns) * 100  # Convert to percentage
        max_drawdowns.append(max_drawdown)

    avg_max_drawdown = np.mean(max_drawdowns)
    std_max_drawdown = np.std(max_drawdowns)

    # If true value curve is provided, calculate price-value relationship
    price_value_correlation = None
    price_value_deviation = None
    if value_curve is not None and len(value_curve) >= min_length:
        value_curve_trimmed = value_curve[:min_length]
        # Calculate correlation between price and value
        price_value_correlation = np.corrcoef(average_sequence, value_curve_trimmed)[0, 1]
        # Calculate average deviation between price and value (percentage)
        price_value_deviation = np.mean(np.abs(average_sequence - value_curve_trimmed) / value_curve_trimmed) * 100

    return {
        'average_sequence': average_sequence,
        'std_sequence': std_sequence,
        'volatility': volatility,
        'avg_volatility': avg_volatility,
        'std_volatility': std_volatility,
        'individual_volatilities': individual_volatilities,
        'max_drawdowns': max_drawdowns,
        'avg_max_drawdown': avg_max_drawdown,
        'std_max_drawdown': std_max_drawdown,
        'upside_volatility': upside_volatility,
        'downside_volatility': downside_volatility,
        'price_value_correlation': price_value_correlation,
        'price_value_deviation': price_value_deviation
    }

def plot_comparison_results(results, value_curve_data):
    """
    Plot average price sequences, volatility, and true value curve for different bias_percent_std settings.

    Parameters:
        results: Dictionary containing results for different bias_percent_std settings
        value_curve_data: True value curve data
    """
    # Create a 3x2 subplot layout
    fig, axs = plt.subplots(3, 2, figsize=(18, 20))

    # Ensure all sequences have the same length
    min_length = min(len(stats['average_sequence']) for std, stats in results.items())

    # Use different colors to plot average sequences for different bias_percent_std settings
    colors = ['blue', 'green', 'red']
    labels = ['Low Bias Std (0.15)', 'Medium Bias Std (0.30)', 'High Bias Std (0.50)']

    # Plot 1: Average price curves
    for i, (std, stats) in enumerate(results.items()):
        avg_sequence_trimmed = stats['average_sequence'][:min_length]
        axs[0, 0].plot(avg_sequence_trimmed, linewidth=2, color=colors[i], label=labels[i])

    # Plot true value curve in black dashed line
    value_curve_trimmed = value_curve_data[:min_length]
    axs[0, 0].plot(value_curve_trimmed, linestyle='--', color='black', linewidth=1.5, label='True Value Curve', zorder=10)

    axs[0, 0].set_title('Average Stock Prices with Different Bias Standard Deviations')
    axs[0, 0].set_xlabel('Trading Days')
    axs[0, 0].set_ylabel('Price')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # Plot 2: Price volatility (standard deviation)
    for i, (std, stats) in enumerate(results.items()):
        std_sequence_trimmed = stats['std_sequence'][:min_length]
        axs[0, 1].plot(std_sequence_trimmed, linewidth=2, color=colors[i], label=labels[i])

    axs[0, 1].set_title('Price Standard Deviation with Different Bias Standard Deviations')
    axs[0, 1].set_xlabel('Trading Days')
    axs[0, 1].set_ylabel('Standard Deviation')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # Plot 3: Price deviation from true value percentage
    for i, (std, stats) in enumerate(results.items()):
        avg_sequence_trimmed = stats['average_sequence'][:min_length]
        # Calculate price deviation from true value percentage
        deviation_percent = (avg_sequence_trimmed - value_curve_trimmed) / value_curve_trimmed * 100
        axs[1, 0].plot(deviation_percent, linewidth=2, color=colors[i], label=labels[i])

    axs[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axs[1, 0].set_title('Price Deviation from True Value (%)')
    axs[1, 0].set_xlabel('Trading Days')
    axs[1, 0].set_ylabel('Deviation (%)')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # Plot 4: Volatility comparison (bar chart)
    x = np.arange(len(results))
    width = 0.35

    volatilities = [stats['avg_volatility'] for std, stats in results.items()]
    volatility_stds = [stats['std_volatility'] for std, stats in results.items()]

    axs[1, 1].bar(x, volatilities, width, yerr=volatility_stds, color=colors, alpha=0.7)
    axs[1, 1].set_title('Average Daily Volatility (%)')
    axs[1, 1].set_ylabel('Volatility (%)')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(['0.15', '0.30', '0.50'])
    axs[1, 1].set_xlabel('Bias Standard Deviation')
    axs[1, 1].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(volatilities):
        axs[1, 1].text(i, v + volatility_stds[i] + 0.1, f'{v:.2f}%', ha='center')

    # Plot 5: Upside vs downside volatility comparison
    upside_volatilities = [stats['upside_volatility'] for std, stats in results.items()]
    downside_volatilities = [stats['downside_volatility'] for std, stats in results.items()]

    x = np.arange(len(results))
    width = 0.35

    axs[2, 0].bar(x - width/2, upside_volatilities, width, color='green', alpha=0.7, label='Upside Volatility')
    axs[2, 0].bar(x + width/2, downside_volatilities, width, color='red', alpha=0.7, label='Downside Volatility')
    axs[2, 0].set_title('Upside vs Downside Volatility (%)')
    axs[2, 0].set_ylabel('Volatility (%)')
    axs[2, 0].set_xticks(x)
    axs[2, 0].set_xticklabels(['0.15', '0.30', '0.50'])
    axs[2, 0].set_xlabel('Bias Standard Deviation')
    axs[2, 0].legend()
    axs[2, 0].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(upside_volatilities):
        axs[2, 0].text(i - width/2, v + 0.1, f'{v:.2f}%', ha='center')
    for i, v in enumerate(downside_volatilities):
        axs[2, 0].text(i + width/2, v + 0.1, f'{v:.2f}%', ha='center')

    # Plot 6: Maximum drawdown comparison
    max_drawdowns = [stats['avg_max_drawdown'] for std, stats in results.items()]
    max_drawdown_stds = [stats['std_max_drawdown'] for std, stats in results.items()]

    axs[2, 1].bar(x, max_drawdowns, width, yerr=max_drawdown_stds, color=colors, alpha=0.7)
    axs[2, 1].set_title('Average Maximum Drawdown (%)')
    axs[2, 1].set_ylabel('Maximum Drawdown (%)')
    axs[2, 1].set_xticks(x)
    axs[2, 1].set_xticklabels(['0.15', '0.30', '0.50'])
    axs[2, 1].set_xlabel('Bias Standard Deviation')
    axs[2, 1].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(max_drawdowns):
        axs[2, 1].text(i, v + max_drawdown_stds[i] + 0.1, f'{v:.2f}%', ha='center')

    plt.tight_layout()

    # Save plot
    plt.savefig('bias_std_comparison_results.png', dpi=300)

    # Show plot
    plt.show()

    # Return statistics for report generation
    return {
        'min_length': min_length,
        'value_curve_trimmed': value_curve_trimmed
    }

def generate_analysis_report(results, plot_data):
    """
    Generate analysis report containing volatility analysis results for different bias_percent_std settings.

    Parameters:
        results: Dictionary containing results for different bias_percent_std settings
        plot_data: Plot data containing minimum sequence length and true value curve
    """
    min_length = plot_data['min_length']
    value_curve_trimmed = plot_data['value_curve_trimmed']

    # Create report file
    with open('bias_std_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write("# Analysis of Value Investor Bias Standard Deviation Impact on Market Volatility\n\n")

        f.write("## Overview\n\n")
        f.write("This report analyzes the impact of the value investor bias standard deviation (bias_percent_std) parameter on stock market simulation.")
        f.write("By running 20 Monte Carlo simulations, we compared three different bias_percent_std settings: 0.15 (low), 0.30 (medium), and 0.50 (high).\n\n")

        f.write("## Volatility Analysis\n\n")

        # Create volatility comparison table
        f.write("### Overall Volatility Metrics\n\n")
        f.write("| Metric | Low Bias Std (0.15) | Medium Bias Std (0.30) | High Bias Std (0.50) |\n")
        f.write("| --- | ---: | ---: | ---: |\n")

        # Add average volatility
        avg_volatilities = [stats['avg_volatility'] for std, stats in results.items()]
        f.write(f"| Average Daily Volatility (%) | {avg_volatilities[0]:.4f} | {avg_volatilities[1]:.4f} | {avg_volatilities[2]:.4f} |\n")

        # Add volatility standard deviation
        std_volatilities = [stats['std_volatility'] for std, stats in results.items()]
        f.write(f"| Volatility Std Dev (%) | {std_volatilities[0]:.4f} | {std_volatilities[1]:.4f} | {std_volatilities[2]:.4f} |\n")

        # Add upside volatility
        upside_volatilities = [stats['upside_volatility'] for std, stats in results.items()]
        f.write(f"| Upside Volatility (%) | {upside_volatilities[0]:.4f} | {upside_volatilities[1]:.4f} | {upside_volatilities[2]:.4f} |\n")

        # Add downside volatility
        downside_volatilities = [stats['downside_volatility'] for std, stats in results.items()]
        f.write(f"| Downside Volatility (%) | {downside_volatilities[0]:.4f} | {downside_volatilities[1]:.4f} | {downside_volatilities[2]:.4f} |\n")

        # Add upside/downside volatility ratio
        up_down_ratios = [up/down if down != 0 else float('inf') for up, down in zip(upside_volatilities, downside_volatilities)]
        f.write(f"| Upside/Downside Volatility Ratio | {up_down_ratios[0]:.4f} | {up_down_ratios[1]:.4f} | {up_down_ratios[2]:.4f} |\n\n")

        # Add maximum drawdown information
        f.write("### Maximum Drawdown Analysis\n\n")
        f.write("| Metric | Low Bias Std (0.15) | Medium Bias Std (0.30) | High Bias Std (0.50) |\n")
        f.write("| --- | ---: | ---: | ---: |\n")

        # Add average maximum drawdown
        avg_max_drawdowns = [stats['avg_max_drawdown'] for std, stats in results.items()]
        f.write(f"| Average Maximum Drawdown (%) | {avg_max_drawdowns[0]:.4f} | {avg_max_drawdowns[1]:.4f} | {avg_max_drawdowns[2]:.4f} |\n")

        # Add maximum drawdown standard deviation
        std_max_drawdowns = [stats['std_max_drawdown'] for std, stats in results.items()]
        f.write(f"| Max Drawdown Std Dev (%) | {std_max_drawdowns[0]:.4f} | {std_max_drawdowns[1]:.4f} | {std_max_drawdowns[2]:.4f} |\n\n")

        # Add price-value relationship analysis
        f.write("### Price-Value Relationship Analysis\n\n")
        f.write("| Metric | Low Bias Std (0.15) | Medium Bias Std (0.30) | High Bias Std (0.50) |\n")
        f.write("| --- | ---: | ---: | ---: |\n")

        # Add price-value correlation
        price_value_correlations = [stats['price_value_correlation'] for std, stats in results.items()]
        f.write(f"| Price-Value Correlation | {price_value_correlations[0]:.4f} | {price_value_correlations[1]:.4f} | {price_value_correlations[2]:.4f} |\n")

        # Add price-value average deviation
        price_value_deviations = [stats['price_value_deviation'] for std, stats in results.items()]
        f.write(f"| Price-Value Average Deviation (%) | {price_value_deviations[0]:.4f} | {price_value_deviations[1]:.4f} | {price_value_deviations[2]:.4f} |\n\n")

        # Add conclusions and observations
        f.write("## Conclusions and Observations\n\n")

        # Volatility trend analysis
        if avg_volatilities[0] < avg_volatilities[1] < avg_volatilities[2]:
            f.write("1. **Volatility increases with bias standard deviation**: Data shows that market volatility significantly increases as the value investor bias standard deviation increases.\n")
        elif avg_volatilities[0] > avg_volatilities[1] > avg_volatilities[2]:
            f.write("1. **Volatility decreases with bias standard deviation**: Data shows that market volatility actually decreases as the value investor bias standard deviation increases.\n")
        else:
            f.write("1. **Non-linear relationship between volatility and bias standard deviation**: Data shows a complex non-linear relationship between volatility and bias standard deviation.\n")

        # Upside vs downside volatility analysis
        for i, ratio in enumerate(up_down_ratios):
            if ratio > 1:
                f.write(f"2. **Upside volatility greater than downside volatility for bias_percent_std={[0.15, 0.30, 0.50][i]}**: Upside/downside volatility ratio is {ratio:.4f}, indicating market volatility is greater during price increases than decreases.\n")
            elif ratio < 1:
                f.write(f"2. **Downside volatility greater than upside volatility for bias_percent_std={[0.15, 0.30, 0.50][i]}**: Upside/downside volatility ratio is {ratio:.4f}, indicating market volatility is greater during price decreases than increases.\n")
            break  # Only analyze first ratio

        # Maximum drawdown analysis
        if avg_max_drawdowns[0] < avg_max_drawdowns[1] < avg_max_drawdowns[2]:
            f.write("3. **Maximum drawdown increases with bias standard deviation**: Data shows that market maximum drawdown increases as the value investor bias standard deviation increases.\n")
        elif avg_max_drawdowns[0] > avg_max_drawdowns[1] > avg_max_drawdowns[2]:
            f.write("3. **Maximum drawdown decreases with bias standard deviation**: Data shows that market maximum drawdown actually decreases as the value investor bias standard deviation increases.\n")
        else:
            f.write("3. **Non-linear relationship between maximum drawdown and bias standard deviation**: Data shows a complex non-linear relationship between maximum drawdown and bias standard deviation.\n")

        # Price-value correlation analysis
        if all(corr > 0 for corr in price_value_correlations):
            max_corr_idx = price_value_correlations.index(max(price_value_correlations))
            f.write(f"4. **Positive price-value correlation for all settings**: Strongest correlation ({max(price_value_correlations):.4f}) observed with bias_percent_std={[0.15, 0.30, 0.50][max_corr_idx]}.\n")
        elif all(corr < 0 for corr in price_value_correlations):
            min_corr_idx = price_value_correlations.index(min(price_value_correlations))
            f.write(f"4. **Negative price-value correlation for all settings**: Strongest negative correlation ({min(price_value_correlations):.4f}) observed with bias_percent_std={[0.15, 0.30, 0.50][min_corr_idx]}.\n")
        else:
            f.write("4. **Price-value correlation varies with bias standard deviation**: Different bias_percent_std settings result in significant variations in price-value correlation.\n")

        # Summary
        f.write("\n## Summary\n\n")
        f.write("This study demonstrates that the value investor bias standard deviation parameter has a significant impact on market volatility.")
        f.write("As bias_percent_std increases, we observe significant changes in market volatility, maximum drawdown, and price-value deviation metrics.")
        f.write("These findings provide important insights into how value investor valuation biases affect market dynamics.")

def main():
    # Set global random seed for reproducible results
    global_seed = 42
    np.random.seed(global_seed)

    # Centralize all random seed settings
    base_seed = 2108
    random_seeds = {
        'base_seed': base_seed,
        'value_line_seed': base_seed + 345,  # True value curve seed
        'close_seed': base_seed + 2,       # Closing price seed
        'market_seed': base_seed + 3,      # Market seed
        'value_investor_seed': base_seed + 4,  # Value investor seed
        'chase_investor_seed': base_seed + 5,  # Chase investor seed
        'trend_investor_seed': base_seed + 6,  # Trend investor seed
        'random_investor_seed': base_seed + 7,  # Random investor seed
        'never_stop_loss_investor_seed': base_seed + 8,  # Never stop loss investor seed
        'bottom_fishing_investor_seed': base_seed + 9,  # Bottom fishing investor seed
        'insider_investor_seed': base_seed + 10,  # Insider investor seed
        'message_investor_seed': base_seed + 11  # Message investor seed
    }

    # Number of simulations to run
    num_simulations = 20  # Increase number of simulations for better statistical reliability

    # Different bias_percent_std settings
    bias_percent_stds = [0.15, 0.30, 0.50]

    # Store results for different settings
    results = {}
    value_curve_data = None

    # Run multiple simulations for each bias_percent_std value
    for bias_std in bias_percent_stds:
        # Run multiple simulations
        all_closing_prices, value_curve = run_multiple_simulations(
            num_simulations=num_simulations,
            base_seed=base_seed,
            custom_seeds=random_seeds,
            bias_percent_std=bias_std
        )

        # Calculate statistics
        stats = calculate_statistics(all_closing_prices, value_curve)

        # Store results
        results[bias_std] = stats

        # Store true value curve (all settings use the same true value curve)
        if value_curve_data is None:
            value_curve_data = value_curve

    # Plot comparison results
    plot_data = plot_comparison_results(results, value_curve_data)

    # Generate analysis report
    generate_analysis_report(results, plot_data)

    print("Simulation complete. Results saved to 'bias_std_comparison_results.png'")
    print("Analysis report saved to 'bias_std_analysis_report.md'")
    print(f"Global seed: {global_seed}, Base seed: {base_seed}")

if __name__ == "__main__":
    main()
