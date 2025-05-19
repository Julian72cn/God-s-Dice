"""
MC_simulation_1.4.py

This script runs multiple stock market model simulations (based on main_OHLC_2.0.5.1.py),
focusing on simulating the long-term impact of different transaction costs on the market,
and analyzing the effect of transaction costs on stock price averages using Monte Carlo methods.

Simulating four transaction cost scenarios:
1. No transaction cost: buy_fee_rate = 0, sell_fee_rate = 0
2. Low transaction cost: buy_fee_rate = 0.0003, sell_fee_rate = 0.0003 (0.03%)
3. Medium transaction cost: buy_fee_rate = 0.001, sell_fee_rate = 0.001 (0.1%)
4. High transaction cost: buy_fee_rate = 0.01, sell_fee_rate = 0.01 (1%)

Main features:
1. Simulating market performance under different transaction costs
2. Comparing the effects of transaction costs on market liquidity, price volatility, and trading volume
3. Analyzing the impact of transaction costs on different types of investors
4. Reducing the influence of random factors through multiple simulations to obtain more reliable conclusions

This simulation helps understand the impact of transaction costs on market operational efficiency,
providing reference for market design and regulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import importlib.util

# Import main_OHLC_2.0.5.1 module
spec = importlib.util.spec_from_file_location("main_module", "main_OHLC_2.0.5.1.py")
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)

# Get necessary functions and classes from the module
generate_random_seeds = main_module.generate_random_seeds
Market = main_module.Market
TrueValueCurve = main_module.TrueValueCurve
Investor = main_module.Investor

def run_simulation_with_seed(random_seeds, enable_trade_log=False, capital_events=None, buy_fee_rate=0.001, sell_fee_rate=0.001):
    """
    Run a single stock market simulation with the given random seeds and return the market object.
    This is a modified version of the simulate_stock_market function, specifically designed to return the market object for further analysis.

    Parameters:
        random_seeds: Dictionary containing various random seeds to control randomness in the simulation
        enable_trade_log: Whether to enable trade log recording, default is False
        capital_events: List of capital injection or withdrawal events, default is None
        buy_fee_rate: Buy transaction fee rate, default is 0.001 (0.1%)
        sell_fee_rate: Sell transaction fee rate, default is 0.001 (0.1%)

    Returns:
        Market object (Market instance) containing simulation results, including price history, trading volume, etc.
    """
    # Basic market parameter settings
    initial_price = 100
    price_tick = 0.01
    days = 1200
    # buy_fee_rate and sell_fee_rate are now passed as function parameters

    # Set whether to enable trade log recording
    Investor.set_enable_trade_log(enable_trade_log)

    # Initialize global trade log file for all investors
    Investor.init_trade_log_file("all_investors_trade_log.csv")

    # Create true value curve
    value_curve = TrueValueCurve(initial_value=initial_price, days=days, seed=random_seeds['value_line_seed'])

    # Create market instance
    market = Market(initial_price, price_tick, value_curve=value_curve,
                   seed=random_seeds['market_seed'],
                   close_seed=random_seeds['close_seed'],
                   buy_fee_rate=buy_fee_rate, sell_fee_rate=sell_fee_rate)

    # Parameter settings for different types of investors
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

    # Create value investors (investment strategy based on value deviation)
    bias_percent_std = 0.30 # Originally 0.15
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

    # Create momentum investors (chasing price uptrends)
    for i in range(chase_investors_params['num']):
        investors.append(main_module.ChaseInvestor(
            chase_investors_params['initial_shares'],
            chase_investors_params['initial_cash'],
            N=None,  # Will be randomly initialized
            seed=random_seeds['chase_investor_seed'] + i
        ))

    # Create trend investors (based on trend analysis over different time periods)
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

    # Create never-stop-loss investors (investment strategy without stop-loss points)
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

    # Create message investors (trading based on market messages)
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

    # Run simulation trading by day
    for day in range(days):
        # Execute investor trading
        for investor in investors:
            investor.trade(market.price, market)

        # Execute market auction
        market.daily_auction()

    # Close trade log file
    Investor.close_trade_log_file()

    return market

def run_multiple_simulations(num_simulations=10, base_seed=2108, custom_seeds=None, enable_trade_log=False, capital_events=None, buy_fee_rate=0.001, sell_fee_rate=0.001):
    """
    Run multiple simulations, each using different random seed values (except value_line_seed remains constant), and collect results.

    Parameters:
        num_simulations: Number of simulations to run
        base_seed: Base seed value for random number generation
        custom_seeds: Optional custom random seed dictionary, if provided these seed values will be used
        enable_trade_log: Whether to enable trade log recording
        capital_events: List of capital injection or withdrawal events, default is None
        buy_fee_rate: Buy transaction fee rate, default is 0.001 (0.1%)
        sell_fee_rate: Sell transaction fee rate, default is 0.001 (0.1%)

    Returns:
        A tuple containing:
        - List of closing price sequences for each simulation
        - True value curve from the first simulation (when using the same value_line_seed, all simulations have the same true value curve)
        - List of trading volume sequences for all simulations
    """
    all_closing_prices = []
    value_curve_data = None  # Store the true value curve from the first simulation

    print(f"Running {num_simulations} simulations...")

    # Save the value_line_seed value to ensure all simulations use the same true value curve
    value_line_seed = custom_seeds['value_line_seed'] if custom_seeds is not None and 'value_line_seed' in custom_seeds else (base_seed + 345)

    for i in range(num_simulations):
        print(f"Simulation {i+1}/{num_simulations}...")

        # Generate new random seeds for each simulation
        # Use current time and simulation index to generate random seeds, ensuring each simulation has different randomness
        current_seed = np.random.randint(0, 100000)

        # Create random seed dictionary
        if custom_seeds is not None:
            # Only keep value_line_seed, regenerate other seeds
            random_seeds = {
                'base_seed': current_seed,
                'value_line_seed': value_line_seed,  # Keep value_line_seed constant
                'close_seed': np.random.randint(0, 100000),
                'market_seed': np.random.randint(0, 100000),
                'value_investor_seed': np.random.randint(0, 100000),
                'chase_investor_seed': np.random.randint(0, 100000),
                'trend_investor_seed': np.random.randint(0, 100000),
                'random_investor_seed': np.random.randint(0, 100000),
                'never_stop_loss_investor_seed': np.random.randint(0, 100000),
                'bottom_fishing_investor_seed': np.random.randint(0, 100000),
                'insider_investor_seed': np.random.randint(0, 100000),
                'message_investor_seed': np.random.randint(0, 100000)
            }
        else:
            # Generate completely new random seed dictionary, but keep value_line_seed constant
            random_seeds = generate_random_seeds(base_seed=current_seed)
            random_seeds['value_line_seed'] = value_line_seed  # Keep value_line_seed constant

        # Run simulation and get market object
        market = run_simulation_with_seed(random_seeds, enable_trade_log, capital_events, buy_fee_rate, sell_fee_rate)

        # Extract closing prices from market object
        closing_prices = [ohlc[3] for ohlc in market.ohlc_data]  # OHLC data format is (open, high, low, close)
        all_closing_prices.append(closing_prices)

        # Store the true value curve from the first simulation
        if i == 0:
            value_curve_data = market.value_history

    return all_closing_prices, value_curve_data

def calculate_average_sequence(sequences):
    """
    Calculate the average sequence from a list of sequences.

    Parameters:
        sequences: List of sequences (list or array)

    Returns:
        Average sequence as a numpy array
    """
    # Ensure all sequences have the same length
    min_length = min(len(seq) for seq in sequences)
    sequences_trimmed = [seq[:min_length] for seq in sequences]

    # Convert to numpy array for calculation
    sequences_array = np.array(sequences_trimmed)

    # Calculate mean along the first axis (across all simulations)
    average_sequence = np.mean(sequences_array, axis=0)

    return average_sequence

def run_multiple_simulations_with_fee_rates(num_simulations=10, base_seed=2108, fee_rates=None, enable_trade_log=False):
    """
    Run multiple simulations to compare the effects of different transaction costs.

    Parameters:
        num_simulations: Number of simulations to run for each transaction cost
        base_seed: Base seed value for random number generation
        fee_rates: Dictionary of different transaction costs, format {'cost_name': (buy_fee_rate, sell_fee_rate)}
        enable_trade_log: Whether to enable trade log recording

    Returns:
        A dictionary containing average closing price sequences and true value curve for each transaction cost
    """
    # Initialize results dictionary
    results = {}
    value_curve_data = None

    # Ensure all simulations use the same true value curve
    value_line_seed = base_seed + 345
    custom_seeds = {'value_line_seed': value_line_seed}

    # Set default transaction cost rates
    if fee_rates is None:
        fee_rates = {
            'no_fee': (0.0, 0.0),                # No transaction cost
            'low_fee': (0.0003, 0.0003),         # Low transaction cost (0.03%)
            'medium_fee': (0.001, 0.001),         # Medium transaction cost (0.1%)
            'high_fee': (0.01, 0.01)              # High transaction cost (1%)
        }

    # Run multiple simulations for each transaction cost
    for scenario_name, (buy_fee, sell_fee) in fee_rates.items():
        print(f"\nRunning transaction cost scenario: {scenario_name} (buy_fee_rate={buy_fee}, sell_fee_rate={sell_fee})")

        # Run multiple simulations
        all_closing_prices, scenario_value_curve = run_multiple_simulations(
            num_simulations=num_simulations,
            base_seed=base_seed,
            custom_seeds=custom_seeds,
            enable_trade_log=enable_trade_log,
            capital_events=None,  # Don't use capital injection/withdrawal
            buy_fee_rate=buy_fee,
            sell_fee_rate=sell_fee
        )

        # Calculate average sequence
        average_sequence = calculate_average_sequence(all_closing_prices)

        # Store results
        results[scenario_name] = {
            'average_sequence': average_sequence,
            'all_sequences': all_closing_prices,
            'buy_fee_rate': buy_fee,
            'sell_fee_rate': sell_fee
        }

        # Save true value curve (all scenarios use the same true value curve)
        if value_curve_data is None:
            value_curve_data = scenario_value_curve

    # Add true value curve to results
    results['value_curve'] = value_curve_data

    return results

def plot_results(all_sequences, average_sequence, value_curve_data):
    """
    Plot all sequences, average sequence, and true value curve.

    Parameters:
        all_sequences: List of all individual simulation sequences
        average_sequence: Average sequence
        value_curve_data: True value curve data
    """
    plt.figure(figsize=(12, 8))

    # Ensure all sequences have the same length as the average sequence
    min_length = len(average_sequence)
    all_sequences_trimmed = [seq[:min_length] for seq in all_sequences]

    # Plot individual simulation sequences with low opacity
    for i, sequence in enumerate(all_sequences_trimmed):
        plt.plot(sequence, alpha=0.3, color='gray', label='Individual Simulation' if i == 0 else "")

    # Plot average sequence with high opacity and thicker line
    plt.plot(average_sequence, linewidth=2, color='blue', label='Average Sequence')

    # Plot true value curve with red dashed line (displayed above other lines)
    value_curve_trimmed = value_curve_data[:min_length]
    plt.plot(value_curve_trimmed, linestyle='--', color='red', linewidth=1.5, label='True Value Curve', zorder=10)

    plt.title('Stock Market Simulations with Different Market Seeds')
    plt.xlabel('Trading Days')
    plt.ylabel('Price / Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save chart
    plt.savefig('average_market_simulation_results.png', dpi=300)

    # Display chart
    plt.show()

def plot_fee_rate_comparison(results, global_seed=42, base_seed=2108):
    """
    Plot comparison charts for different transaction costs, using subplots.

    Parameters:
        results: Results dictionary returned by run_multiple_simulations_with_fee_rates function
        global_seed: Global random seed
        base_seed: Base random seed
    """
    # Get true value curve
    value_curve_data = results['value_curve']

    # Determine minimum length of all sequences
    min_length = min(len(results[scenario]['average_sequence']) for scenario in results if scenario != 'value_curve')
    min_length = min(min_length, len(value_curve_data))
    value_curve_trimmed = value_curve_data[:min_length]

    # Calculate number of subplots and layout
    num_scenarios = len([s for s in results if s != 'value_curve'])
    rows = num_scenarios  # One row per transaction cost
    cols = 1  # Only one column

    # Create chart and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(18, 4 * rows), sharex=True, sharey=True)
    if rows == 1 and cols == 1:
        axes = np.array([axes])  # Ensure axes is iterable when there's only one subplot
    axes = axes.flatten()  # Flatten multi-dimensional array to one dimension

    # Choose different colors for different transaction costs
    colors = {
        'no_fee': 'blue',
        'low_fee': 'green',
        'medium_fee': 'purple',
        'high_fee': 'red'
    }

    # Plot subplot for each transaction cost
    scenario_idx = 0
    for scenario in results:
        if scenario == 'value_curve':
            continue

        # Get current subplot
        ax = axes[scenario_idx]
        scenario_idx += 1

        # Get average sequence and all individual simulation sequences for current transaction cost
        average_sequence = results[scenario]['average_sequence'][:min_length]
        all_sequences = results[scenario]['all_sequences']
        buy_fee_rate = results[scenario]['buy_fee_rate']
        sell_fee_rate = results[scenario]['sell_fee_rate']

        # Choose color
        color = colors.get(scenario, 'blue')

        # Plot individual simulation sequences (dark gray)
        for sequence in all_sequences:
            ax.plot(sequence[:min_length], alpha=0.35, color='darkgray', linewidth=0.8, zorder=2)

        # Plot true value curve
        ax.plot(value_curve_trimmed, linestyle='--', color='red', linewidth=1.5, label='True Value', zorder=10)

        # Plot average sequence
        ax.plot(average_sequence, linewidth=2.5, color=color, label='Average Price', zorder=5)

        # Calculate price-value difference
        price_value_diff = np.mean(np.abs(average_sequence - value_curve_trimmed[:len(average_sequence)]))
        price_value_ratio = np.mean(average_sequence / value_curve_trimmed[:len(average_sequence)])

        # Calculate price volatility
        price_volatility = np.std(np.diff(average_sequence) / average_sequence[:-1])

        # Set subplot title and grid
        title = f'Fee Rate: {scenario} (Buy: {buy_fee_rate*100:.2f}%, Sell: {sell_fee_rate*100:.2f}%)'
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=12)

        # Add statistical information
        stats_text = f"Avg Price/Value Diff: {price_value_diff:.2f}\n" \
                    f"Avg Price/Value Ratio: {price_value_ratio:.4f}\n" \
                    f"Price Volatility: {price_volatility*100:.4f}%"
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))

        # Set axis labels
        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Price / Value', fontsize=12)

        # Adjust axis tick font size
        ax.tick_params(axis='both', which='major', labelsize=10)

    # If number of subplots is less than number of rows and columns, hide extra subplots
    for i in range(scenario_idx, len(axes)):
        fig.delaxes(axes[i])

    # Adjust subplot spacing
    plt.tight_layout()

    # Add overall title
    fig.suptitle('Comparison of Different Transaction Fee Rates', fontsize=18, y=1.01)

    # Save chart, filename includes random seed values
    filename = f'transaction_fee_comparison_g{global_seed}_b{base_seed}.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight')

    # Display chart
    plt.show()

    return filename

def main():
    # Set global random seed for reproducible results
    global_seed = 42
    np.random.seed(global_seed)

    # Set base seed
    base_seed = 2108

    # Number of simulations to run
    num_simulations = 50  # Number of simulations for each transaction cost

    # Define different transaction cost rates
    fee_rates = {
        'no_fee': (0.0, 0.0),                # No transaction cost
        'low_fee': (0.0003, 0.0003),         # Low transaction cost (0.03%)
        'medium_fee': (0.001, 0.001),         # Medium transaction cost (0.1%)
        'high_fee': (0.01, 0.01)              # High transaction cost (1%)
    }

    # Run multiple simulations, comparing different transaction costs
    results = run_multiple_simulations_with_fee_rates(
        num_simulations=num_simulations,
        base_seed=base_seed,
        fee_rates=fee_rates,
        enable_trade_log=False
    )

    # Plot comparison chart
    filename = plot_fee_rate_comparison(results, global_seed, base_seed)

    print(f"Simulation complete. Results saved to '{filename}'")
    print(f"Global seed: {global_seed}, Base seed: {base_seed}")
    print(f"Ran {num_simulations} simulations for each transaction cost")
    print("Transaction cost rates:")
    for name, (buy_fee, sell_fee) in fee_rates.items():
        print(f"  {name}: buy_fee_rate={buy_fee*100:.2f}%, sell_fee_rate={sell_fee*100:.2f}%")

if __name__ == "__main__":
    main()
