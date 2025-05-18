"""
butterfly_effect_simulation.py

This script simulates the butterfly effect in stock markets by modifying closing prices on specific days 
(with positive changes of 1%, 2%, 3%, 5% ratios) and observing the impact of these small changes 
on subsequent market behavior.

All random seeds are deterministic to ensure reproducibility and comparability of simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import importlib.util
import copy
from matplotlib.ticker import PercentFormatter

# Import main_OHLC_2.0.5.1 module
spec = importlib.util.spec_from_file_location("main_module", "main_OHLC_2.0.5.1.py")
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)

# Get necessary functions and classes from the module
generate_random_seeds = main_module.generate_random_seeds
Market = main_module.Market
TrueValueCurve = main_module.TrueValueCurve
Investor = main_module.Investor

def run_baseline_simulation(random_seeds, days=2000, enable_trade_log=False):
    """
    Run baseline simulation (without modifying closing prices).

    Parameters:
        random_seeds: Dictionary containing various random seeds to control simulation randomness
        days: Number of trading days to simulate
        enable_trade_log: Whether to enable trade logging, defaults to False

    Returns:
        Market object containing simulation results (Market instance)
    """
    # Basic market parameters setup
    initial_price = 100
    price_tick = 0.01
    buy_fee_rate = 0.001
    sell_fee_rate = 0.001

    # Set whether to enable trade logging
    Investor.set_enable_trade_log(enable_trade_log)

    # Initialize global trade log file for all investors
    if enable_trade_log:
        Investor.init_trade_log_file("baseline_trade_log.csv")

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
        'num': 50,
        'initial_shares': 100,
        'initial_cash': 10000
    }

    trend_investors_params = {
        'num': 50,
        'initial_shares': 100,
        'initial_cash': 10000
    }

    random_investors_params = {
        'num': 30,
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
        'num': 5,
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
    bias_percent_std = 0.30  # previously 0.15
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

    # Create chase investors (following upward price trends)
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

    # Create bottom-fishing investors
    for i in range(bottom_fishing_investors_params['num']):
        # Create profit target using random number generator
        seed = random_seeds['bottom_fishing_investor_seed'] + i
        rng = np.random.RandomState(seed)
        profit_target = rng.uniform(0.1, 0.5)
        investors.append(main_module.BottomFishingInvestor(
            bottom_fishing_investors_params['initial_shares'],
            bottom_fishing_investors_params['initial_cash'],
            profit_target=profit_target,
            seed=seed
        ))

    # Create insider traders
    for i in range(insider_investors_params['num']):
        investors.append(main_module.InsiderInvestor(
            insider_investors_params['initial_shares'],
            insider_investors_params['initial_cash'],
            seed=random_seeds['insider_investor_seed'] + i
        ))

    # Create message investors
    for i in range(message_investors_params['num']):
        investors.append(main_module.MessageInvestor(
            message_investors_params['initial_shares'],
            message_investors_params['initial_cash'],
            seed=random_seeds['message_investor_seed'] + i
        ))

    # Calculate the start and end indices for each type of investor (for later analysis)
    value_end = value_investors_params['num']
    chase_end = value_end + chase_investors_params['num']
    trend_end = chase_end + trend_investors_params['num']
    random_end = trend_end + random_investors_params['num']
    never_stop_loss_end = random_end + never_stop_loss_investors_params['num']
    bottom_fishing_end = never_stop_loss_end + bottom_fishing_investors_params['num']
    insider_end = bottom_fishing_end + insider_investors_params['num']
    message_end = insider_end + message_investors_params['num']

    # Simulate trading day by day
    for day in range(days):
        for investor in investors:
            investor.trade(market.price, market)
        market.daily_auction()

    # Close trade log file
    if enable_trade_log:
        Investor.close_trade_log_file()

    return market

def run_modified_simulation(random_seeds, modification_day, price_change_ratio, days=1000, enable_trade_log=False):
    """
    Run modified simulation, modifying the closing price on a specified day.

    Parameters:
        random_seeds: Dictionary containing various random seeds to control simulation randomness
        modification_day: Trading day to modify the closing price (0-indexed)
        price_change_ratio: Price change ratio (e.g., 0.01 for a 1% increase)
        days: Number of trading days to simulate
        enable_trade_log: Whether to enable trade logging, defaults to False

    Returns:
        Market object containing simulation results (Market instance)
    """
    # Basic market parameters setup
    initial_price = 100
    price_tick = 0.01
    buy_fee_rate = 0.001
    sell_fee_rate = 0.001

    # Set whether to enable trade logging
    Investor.set_enable_trade_log(enable_trade_log)

    # Initialize global trade log file for all investors
    if enable_trade_log:
        log_file_name = f"modified_trade_log_{int(price_change_ratio*100)}percent_day{modification_day}.csv"
        Investor.init_trade_log_file(log_file_name)

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
        'num': 50,
        'initial_shares': 100,
        'initial_cash': 10000
    }

    trend_investors_params = {
        'num': 50,
        'initial_shares': 100,
        'initial_cash': 10000
    }

    random_investors_params = {
        'num': 30,
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
        'num': 5,
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
    bias_percent_std = 0.30  # previously 0.15
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

    # Create chase investors (following upward price trends)
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

    # Create bottom-fishing investors
    for i in range(bottom_fishing_investors_params['num']):
        # Create profit target using random number generator
        seed = random_seeds['bottom_fishing_investor_seed'] + i
        rng = np.random.RandomState(seed)
        profit_target = rng.uniform(0.1, 0.5)
        investors.append(main_module.BottomFishingInvestor(
            bottom_fishing_investors_params['initial_shares'],
            bottom_fishing_investors_params['initial_cash'],
            profit_target=profit_target,
            seed=seed
        ))

    # Create insider traders
    for i in range(insider_investors_params['num']):
        investors.append(main_module.InsiderInvestor(
            insider_investors_params['initial_shares'],
            insider_investors_params['initial_cash'],
            seed=random_seeds['insider_investor_seed'] + i
        ))

    # Create message investors
    for i in range(message_investors_params['num']):
        investors.append(main_module.MessageInvestor(
            message_investors_params['initial_shares'],
            message_investors_params['initial_cash'],
            seed=random_seeds['message_investor_seed'] + i
        ))

    # Calculate the start and end indices for each type of investor (for later analysis)
    value_end = value_investors_params['num']
    chase_end = value_end + chase_investors_params['num']
    trend_end = chase_end + trend_investors_params['num']
    random_end = trend_end + random_investors_params['num']
    never_stop_loss_end = random_end + never_stop_loss_investors_params['num']
    bottom_fishing_end = never_stop_loss_end + bottom_fishing_investors_params['num']
    insider_end = bottom_fishing_end + insider_investors_params['num']
    message_end = insider_end + message_investors_params['num']

    # Simulate trading day by day
    for day in range(days):
        for investor in investors:
            investor.trade(market.price, market)
        market.daily_auction()

        # Modify closing price on the specified day
        if day == modification_day:
            # Get current closing price
            current_close = market.price_history[-1]
            # Calculate new closing price
            new_close = current_close * (1 + price_change_ratio)
            # Modify closing price
            market.price_history[-1] = new_close
            # Update market current price
            market.price = new_close
            # Update OHLC data
            open_price, high_price, low_price, _ = market.ohlc_data[-1]
            # Ensure the highest price is not lower than the new closing price
            high_price = max(high_price, new_close)
            market.ohlc_data[-1] = (open_price, high_price, low_price, new_close)

            print(f"Modified closing price on day {day}: {current_close:.2f} -> {new_close:.2f} (change rate: {price_change_ratio*100:.1f}%)")

    # Close trade log file
    if enable_trade_log:
        Investor.close_trade_log_file()

    return market

def calculate_price_differences(baseline_prices, modified_prices):
    """
    Calculate the differences between baseline prices and modified prices.

    Parameters:
        baseline_prices: Baseline simulation price series
        modified_prices: Modified simulation price series

    Returns:
        Percentage series of price differences
    """
    # Ensure both sequences have the same length
    min_length = min(len(baseline_prices), len(modified_prices))
    baseline_prices = baseline_prices[:min_length]
    modified_prices = modified_prices[:min_length]

    # Calculate percentage of price differences
    price_diff_percent = [(modified - baseline) / baseline * 100 for baseline, modified in zip(baseline_prices, modified_prices)]

    return price_diff_percent

def plot_butterfly_effect(baseline_market, modified_markets, modification_day, price_change_ratios):
    """
    Plot visual charts of the butterfly effect.

    Parameters:
        baseline_market: Baseline simulation market object
        modified_markets: List of modified simulation market objects
        modification_day: Trading day of the modified closing price
        price_change_ratios: List of price change ratios
    """
    # Extract closing prices and value curve
    baseline_prices = [ohlc[3] for ohlc in baseline_market.ohlc_data]
    value_curve = baseline_market.value_history

    # Create charts, increasing height to accommodate more legends
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})

    # Plot value curve (at the bottom layer)
    ax1.plot(value_curve, label='True Value', color='dimgray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

    # Unified color scheme, regardless of positive or negative changes
    all_colors = ['royalblue', 'forestgreen', 'crimson', 'darkorchid', 'darkorange', 'deeppink', 'darkturquoise', 'gold', 'limegreen', 'slateblue']

    for i, (market, ratio) in enumerate(zip(modified_markets, price_change_ratios)):
        modified_prices = [ohlc[3] for ohlc in market.ohlc_data]

        # Select color and label based on the magnitude of change
        color = all_colors[i % len(all_colors)]  # Cycle through colors
        # Retain positive and negative signs to distinguish the direction of change
        if ratio > 0:
            label = f'Modified +{ratio*100:.1f}%'
        else:
            label = f'Modified {ratio*100:.1f}%'  # Negative numbers carry their negative sign
        alpha = 0.8

        # All modified lines use a uniform line style and width, but thinner than baseline
        ax1.plot(modified_prices, label=label, color=color, alpha=alpha, linestyle='-',
                 linewidth=0.8, zorder=2+i)

    # Add vertical line to mark the modification date
    ax1.axvline(x=modification_day, color='gray', linestyle='--', alpha=0.7)
    ax1.text(modification_day+5, min(baseline_prices), f'Modification Day: {modification_day}',
             verticalalignment='bottom', alpha=0.7)

    # Plot baseline line (at the top layer)
    ax1.plot(baseline_prices, label='Baseline', color='black', linewidth=1.2, zorder=20)

    ax1.set_title('Butterfly Effect Simulation in Stock Prices')
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Price')
    # Use a more compact legend layout, displayed in two columns
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize='small')
    ax1.grid(True, alpha=0.3)

    # Plot price difference percentages
    for i, (market, ratio) in enumerate(zip(modified_markets, price_change_ratios)):
        modified_prices = [ohlc[3] for ohlc in market.ohlc_data]
        price_diff_percent = calculate_price_differences(baseline_prices, modified_prices)

        # Select color and label based on the magnitude of change
        color = all_colors[i % len(all_colors)]  # Cycle through colors
        # Retain positive and negative signs to distinguish the direction of change
        if ratio > 0:
            label = f'Diff +{ratio*100:.1f}%'
        else:
            label = f'Diff {ratio*100:.1f}%'  # Negative numbers carry their negative sign
        alpha = 0.8

        # All lines use a uniform line style and width
        ax2.plot(price_diff_percent, label=label, color=color, alpha=alpha, linestyle='-', linewidth=1.0)

    # Add vertical line to mark the modification date
    ax2.axvline(x=modification_day, color='gray', linestyle='--', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    ax2.set_title('Price Difference Percentage from Baseline')
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Price Difference (%)')
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    # Use a more compact legend layout, displayed in two columns
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small')
    ax2.grid(True, alpha=0.3)

    # Adjust layout to leave more space for legends
    plt.tight_layout(pad=3.0, h_pad=3.0)
    plt.subplots_adjust(bottom=0.15)  # Increase bottom space to accommodate legends
    plt.savefig('butterfly_effect_simulation.png', dpi=600)
    plt.show()

def analyze_butterfly_effect(baseline_market, modified_markets, price_change_ratios):
    """
    Analyze the impact of the butterfly effect.

    Parameters:
        baseline_market: Baseline simulation market object
        modified_markets: List of modified simulation market objects
        price_change_ratios: List of price change ratios
    """
    baseline_prices = [ohlc[3] for ohlc in baseline_market.ohlc_data]

    print("\nButterfly Effect Analysis Results:")
    print("-" * 80)
    print(f"{'Change %':^10}{'Max Diff':^15}{'Final Diff':^15}{'Avg Diff':^15}{'Amplification':^15}{'Direction':^10}")
    print("-" * 80)

    for market, ratio in zip(modified_markets, price_change_ratios):
        modified_prices = [ohlc[3] for ohlc in market.ohlc_data]
        price_diff_percent = calculate_price_differences(baseline_prices, modified_prices)

        # Calculate statistics
        max_diff = max(abs(diff) for diff in price_diff_percent)
        final_diff = price_diff_percent[-1]
        avg_diff = sum(abs(diff) for diff in price_diff_percent) / len(price_diff_percent)

        # Calculate amplification factor (ratio of max difference to initial change)
        amplification = max_diff / abs(ratio * 100)

        # Determine if the direction of the final difference is consistent with the initial change
        if (final_diff > 0 and ratio > 0) or (final_diff < 0 and ratio < 0):
            direction = "Same"
        else:
            direction = "Opposite"

        print(f"{ratio*100:^10.1f}%{max_diff:^15.2f}%{final_diff:^15.2f}%{avg_diff:^15.2f}%{amplification:^15.2f}x{direction:^10}")

    print("-" * 80)

def main():
    # Set global random seed for reproducible results
    global_seed = 44
    np.random.seed(global_seed)

    # Centrally set all random seeds
    base_seed = 2298
    random_seeds = {
        'base_seed': base_seed,
        'value_line_seed': base_seed + 345,  # True value curve seed
        'close_seed': base_seed + 2,       # Closing price seed
        'market_seed': base_seed + 3,      # Market seed
        'value_investor_seed': base_seed + 4,  # Value investor seed
        'chase_investor_seed': base_seed + 5,  # Chase investor seed
        'trend_investor_seed': base_seed + 6,  # Trend investor seed
        'random_investor_seed': base_seed + 7,  # Random investor seed
        'never_stop_loss_investor_seed': base_seed + 8,  # Never stop-loss investor seed
        'bottom_fishing_investor_seed': base_seed + 9,  # Bottom fishing investor seed
        'insider_investor_seed': base_seed + 10,  # Insider investor seed
        'message_investor_seed': base_seed + 11  # Message investor seed
    }

    # Simulation parameters
    days = 1000  # Number of trading days to simulate
    modification_day = 300  # Trading day to modify closing price
    price_change_ratios = [0.005, 0.015, 0.04, -0.005, -0.015, -0.04]  # Price change ratios (±1%, ±2%, ±3%, ±5%, ±8%)

    print("Starting butterfly effect simulation...")

    # Run baseline simulation
    print("Running baseline simulation...")
    baseline_market = run_baseline_simulation(random_seeds, days=days)

    # Run modified simulations
    modified_markets = []
    for ratio in price_change_ratios:
        print(f"Running modified simulation (change rate: +{ratio*100:.1f}%)...")
        modified_market = run_modified_simulation(random_seeds, modification_day, ratio, days=days)
        modified_markets.append(modified_market)

    # Plot results
    plot_butterfly_effect(baseline_market, modified_markets, modification_day, price_change_ratios)

    # Analyze butterfly effect
    analyze_butterfly_effect(baseline_market, modified_markets, price_change_ratios)

    print("\nSimulation completed. Results saved to 'butterfly_effect_simulation.png'")
    print(f"Global seed: {global_seed}, Base seed: {base_seed}")

if __name__ == "__main__":
    main()
