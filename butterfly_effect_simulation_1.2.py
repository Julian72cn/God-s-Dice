"""
butterfly_effect_simulation_1.2.py

This script simulates the butterfly effect in the stock market by introducing a sudden investor 
who starts trading on a specific date, then observes how this small change affects the subsequent 
market behavior.

Changes:
1. Removed the original SingleDayInvestor class
2. Added SuddenInvestor class, who suddenly enters the market on a specified date and buys stocks at a premium
3. Compare market behavior with and without this investor's participation
4. Monitor whether the investor's trades are successfully executed

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

class SuddenInvestor(Investor):
    """
    Sudden Investor - Enters the market suddenly on a specified date and buys stocks at a premium price

    Attributes:
        action_day: Trading day when the buy operation is executed
        initial_cash: Initial cash amount
        has_acted: Whether the buy operation has been executed
        price_premium: Price premium percentage, determines the buy price relative to current price
        trade_executed: Whether the trade has been executed
    """
    def __init__(self, shares, cash, action_day, price_premium=0.05, seed=None):
        # Ensure initial state has only cash, no shares
        super().__init__(0, cash)  # Set initial shares to 0
        self.action_day = action_day
        self.initial_cash = cash
        self.has_acted = False
        self.price_premium = price_premium  # Price premium, default 5%
        self.trade_executed = False  # Track if trade has been executed
        self._rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))

    def decide_price(self, current_price, market):
        """Decide trading price and quantity"""
        current_day = len(market.price_history) - 1

        # Only execute buy operation on specified day and only once
        if current_day == self.action_day and not self.has_acted and self.cash > 0:
            # Calculate maximum shares that can be bought (considering fees)
            max_shares = int(self.cash / (current_price * (1 + market.buy_fee_rate)))
            if max_shares > 0:
                self.has_acted = True
                # Set buy price slightly higher than current price
                buy_price = current_price * (1 + self.price_premium)
                return ('buy', buy_price, max_shares)

        return ('hold', 0, 0)

    def trade(self, price, market):
        """Override trade method to ensure trades are recorded and monitored"""
        current_day = len(market.price_history) - 1

        # Record pre-trade state
        old_shares = self.shares
        old_cash = self.cash

        action, order_price, shares = self.decide_price(price, market)

        if action == 'buy':
            # Record entry price
            self.entry_price = price
            # Record pre-order cash and shares
            pre_order_cash = self.cash
            pre_order_shares = self.shares
            # Place order directly
            market.place_order('buy', order_price, shares, self)
            # Calculate actual execution results
            actual_bought_shares = self.shares - pre_order_shares
            actual_spent_cash = pre_order_cash - self.cash
            execution_rate = actual_bought_shares / shares if shares > 0 else 0

            # Update trade execution status
            self.trade_executed = actual_bought_shares > 0

            # Print trade information
            print(f"SuddenInvestor placed order: Buy {shares} shares at {order_price:.2f} (premium: {self.price_premium:.4f})")
            print(f"Execution result: Bought {actual_bought_shares} shares ({execution_rate*100:.2f}% executed), spent {actual_spent_cash:.2f}")
            print(f"Trade executed: {self.trade_executed}")

        # Record post-trade state changes
        if old_shares != self.shares or old_cash != self.cash:
            profit_ratio = 0
            if self.shares > 0 and self.entry_price is not None:
                profit_ratio = (price - self.entry_price) / self.entry_price
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

def run_baseline_simulation(random_seeds, days=2000, enable_trade_log=False):
    """
    Run baseline simulation (without adding sudden investor).

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

    # Calculate start and end indices for each investor type (for subsequent analysis)
    value_end = value_investors_params['num']
    chase_end = value_end + chase_investors_params['num']
    trend_end = chase_end + trend_investors_params['num']
    random_end = trend_end + random_investors_params['num']
    never_stop_loss_end = random_end + never_stop_loss_investors_params['num']
    bottom_fishing_end = never_stop_loss_end + bottom_fishing_investors_params['num']
    insider_end = bottom_fishing_end + insider_investors_params['num']
    message_end = insider_end + message_investors_params['num']

    # Run daily simulation
    for day in range(days):
        for investor in investors:
            investor.trade(market.price, market)
        market.daily_auction()

    # Close trade log file
    if enable_trade_log:
        Investor.close_trade_log_file()

    return market

def run_modified_simulation(random_seeds, action_day, investor_cash, days=1000, enable_trade_log=False):
    """
    Run modified simulation by adding a sudden investor on a specific day.

    Parameters:
        random_seeds: Dictionary containing various random seeds to control simulation randomness
        action_day: Trading day when the sudden investor executes buy operation (counting from 0)
        investor_cash: Initial capital of the sudden investor
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
        log_file_name = f"modified_trade_log_cash{investor_cash}_day{action_day}.csv"
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

    # Calculate start and end indices for each investor type (for subsequent analysis)
    value_end = value_investors_params['num']
    chase_end = value_end + chase_investors_params['num']
    trend_end = chase_end + trend_investors_params['num']
    random_end = trend_end + random_investors_params['num']
    never_stop_loss_end = random_end + never_stop_loss_investors_params['num']
    bottom_fishing_end = never_stop_loss_end + bottom_fishing_investors_params['num']
    insider_end = bottom_fishing_end + insider_investors_params['num']
    message_end = insider_end + message_investors_params['num']

    # Add sudden investor
    # Set price premium
    price_premium = 0.05  # Price premium of 5%, increased to make orders more likely to execute
    sudden_investor = SuddenInvestor(
        shares=0,  # No initial shares
        cash=investor_cash,  # Set initial capital
        action_day=action_day,  # Set action day
        price_premium=price_premium,  # Set price premium
        seed=random_seeds['base_seed'] + 999  # Use independent random seed
    )
    # Add this investor separately, not in regular investor list, to ensure it only trades on specified day
    sudden_investor_list = [sudden_investor]
    sudden_investor_end = message_end + 1

    # Run daily simulation
    for day in range(days):
        # Regular investors trading
        for investor in investors:
            investor.trade(market.price, market)

        # Let sudden investor participate on specified day
        if day == action_day:
            print(f"Day {day}: SuddenInvestor with {investor_cash:.2f} cash entered the market")
            print(f"Market price before SuddenInvestor: {market.price:.2f}")
            # Print sudden investor's initial state
            print(f"Initial state - Shares: {sudden_investor.shares}, Cash: {sudden_investor.cash:.2f}")
            # Record pre-trade state
            pre_trade_shares = sudden_investor.shares
            pre_trade_cash = sudden_investor.cash
            # Let sudden investor trade
            sudden_investor.trade(market.price, market)

        # Execute daily auction
        market.daily_auction()

        # Print market conditions and check trade execution on specified day
        if day == action_day:
            # Print day's market price and volume
            print(f"Market price after auction: {market.price:.2f}, Daily volume: {market.executed_volume}")
            # Check post-trade state changes
            post_trade_shares = sudden_investor.shares
            post_trade_cash = sudden_investor.cash
            # Update trade execution status
            if post_trade_shares > pre_trade_shares or post_trade_cash < pre_trade_cash:
                sudden_investor.trade_executed = True
            print(f"Post-auction state - Shares: {sudden_investor.shares}, Cash: {sudden_investor.cash:.2f}")
            print(f"Trade executed: {sudden_investor.trade_executed}")

        # Check trade status again on the day after action
        if day == action_day + 1:
            print(f"Next day - Trade execution status: {sudden_investor.trade_executed}")
            print(f"Investor final shares: {sudden_investor.shares}, cash: {sudden_investor.cash:.2f}")

    # Close trade log file
    if enable_trade_log:
        Investor.close_trade_log_file()

    return market

def calculate_price_differences(baseline_prices, modified_prices):
    """
    Calculate the differences between baseline prices and modified prices.

    Parameters:
        baseline_prices: Price sequence from baseline simulation
        modified_prices: Price sequence from modified simulation

    Returns:
        List of percentage differences in prices
    """
    # Ensure both sequences have the same length
    min_length = min(len(baseline_prices), len(modified_prices))
    baseline_prices = baseline_prices[:min_length]
    modified_prices = modified_prices[:min_length]

    # Calculate percentage differences in prices
    price_diff_percent = [(modified - baseline) / baseline * 100 for baseline, modified in zip(baseline_prices, modified_prices)]

    return price_diff_percent

def plot_butterfly_effect(baseline_market, modified_markets, action_day, investor_cash_amounts):
    """
    Plot visualization of the butterfly effect.

    Parameters:
        baseline_market: Market object from baseline simulation
        modified_markets: List of market objects from modified simulations
        action_day: Trading day when sudden investor executes buy operation
        investor_cash_amounts: List of initial capital amounts for sudden investor
    """
    # Extract closing prices and value curve
    baseline_prices = [ohlc[3] for ohlc in baseline_market.ohlc_data]
    value_curve = baseline_market.value_history

    # Create chart, increase height to accommodate more legends
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})

    # Plot value curve (in background)
    ax1.plot(value_curve, label='True Value', color='dimgray', linestyle='--', linewidth=1.5, alpha=0.7, zorder=1)

    # Use unified color scheme
    all_colors = ['royalblue', 'forestgreen', 'crimson', 'darkorchid', 'darkorange', 'deeppink', 'darkturquoise', 'gold', 'limegreen', 'slateblue']

    for i, (market, cash) in enumerate(zip(modified_markets, investor_cash_amounts)):
        modified_prices = [ohlc[3] for ohlc in market.ohlc_data]

        # Choose color and label based on amount
        color = all_colors[i % len(all_colors)]  # Use cycling colors
        # Use capital amount as label
        label = f'Cash {cash:.0f}'
        alpha = 0.8

        # All modified lines use same style but thinner than baseline
        ax1.plot(modified_prices, label=label, color=color, alpha=alpha, linestyle='-',
                 linewidth=0.8, zorder=2+i)

    # Add vertical line marking investor action day
    ax1.axvline(x=action_day, color='gray', linestyle='--', alpha=0.7)
    ax1.text(action_day+5, min(baseline_prices), f'Investor Action Day: {action_day}',
             verticalalignment='bottom', alpha=0.7)

    # Plot baseline line (on top)
    ax1.plot(baseline_prices, label='Baseline', color='black', linewidth=1.2, zorder=20)

    ax1.set_title('Butterfly Effect of Sudden Investor in Stock Prices')
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Price')
    # Use compact legend layout in two columns
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6, fontsize='small')
    ax1.grid(True, alpha=0.3)

    # Plot price difference percentages
    for i, (market, cash) in enumerate(zip(modified_markets, investor_cash_amounts)):
        modified_prices = [ohlc[3] for ohlc in market.ohlc_data]
        price_diff_percent = calculate_price_differences(baseline_prices, modified_prices)

        # Choose color and label based on amount
        color = all_colors[i % len(all_colors)]  # Use cycling colors
        # Use capital amount as label
        label = f'Diff {cash:.0f}'
        alpha = 0.8

        # All lines use same style and width
        ax2.plot(price_diff_percent, label=label, color=color, alpha=alpha, linestyle='-', linewidth=1.0)

    # Add vertical line marking investor action day
    ax2.axvline(x=action_day, color='gray', linestyle='--', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    ax2.set_title('Price Difference Percentage from Baseline')
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Price Difference (%)')
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    # Use compact legend layout in two columns
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize='small')
    ax2.grid(True, alpha=0.3)

    # Adjust layout to leave space for legends
    plt.tight_layout(pad=3.0, h_pad=3.0)
    plt.subplots_adjust(bottom=0.15)  # Increase bottom margin for legends
    plt.savefig('butterfly_effect_sudden_investor.png', dpi=600)
    plt.show()

def analyze_butterfly_effect(baseline_market, modified_markets, investor_cash_amounts):
    """
    Analyze the impact level of butterfly effect.

    Parameters:
        baseline_market: Market object from baseline simulation
        modified_markets: List of market objects from modified simulations
        investor_cash_amounts: List of initial capital amounts for sudden investor
    """
    baseline_prices = [ohlc[3] for ohlc in baseline_market.ohlc_data]

    print("\nButterfly Effect Analysis Results:")
    print("-" * 80)
    print(f"{'Cash':^10}{'Max Diff':^15}{'Final Diff':^15}{'Avg Diff':^15}{'Impact Ratio':^15}{'Direction':^10}")
    print("-" * 80)

    for market, cash in zip(modified_markets, investor_cash_amounts):
        modified_prices = [ohlc[3] for ohlc in market.ohlc_data]
        price_diff_percent = calculate_price_differences(baseline_prices, modified_prices)

        # Calculate statistics
        max_diff = max(abs(diff) for diff in price_diff_percent)
        final_diff = price_diff_percent[-1]
        avg_diff = sum(abs(diff) for diff in price_diff_percent) / len(price_diff_percent)

        # Calculate impact ratio (max difference / investor capital, unit: percent/10k)
        impact_ratio = max_diff / (cash / 10000) if cash > 0 else 0

        # Determine final difference direction
        if final_diff > 0:
            direction = "Positive"
        elif final_diff < 0:
            direction = "Negative"
        else:
            direction = "Neutral"

        print(f"{cash:^10.0f}{max_diff:^15.2f}%{final_diff:^15.2f}%{avg_diff:^15.2f}%{impact_ratio:^15.2f}{direction:^10}")

    print("-" * 80)

def main():
    # Set global random seed for reproducible results
    global_seed = 44
    np.random.seed(global_seed)

    # Set all random seeds centrally
    base_seed = 2298
    random_seeds = {
        'base_seed': base_seed,
        'value_line_seed': base_seed + 345,  # True value curve seed
        'close_seed': base_seed + 2,       # Closing price seed
        'market_seed': base_seed + 3,      # Market seed
        'value_investor_seed': base_seed + 4,  # Value investor seed
        'chase_investor_seed': base_seed + 5,  # Momentum investor seed
        'trend_investor_seed': base_seed + 6,  # Trend investor seed
        'random_investor_seed': base_seed + 7,  # Random investor seed
        'never_stop_loss_investor_seed': base_seed + 8,  # Never-stop-loss investor seed
        'bottom_fishing_investor_seed': base_seed + 9,  # Bottom-fishing investor seed
        'insider_investor_seed': base_seed + 10,  # Insider trader seed
        'message_investor_seed': base_seed + 11  # Message-driven investor seed
    }

    # Simulation parameters
    days = 1000  # Number of trading days to simulate
    action_day = 300  # Day when sudden investor executes buy operation
    investor_cash_amounts = [300, 1000, 5000, 10000, 20000, 50000, 100000]  # List of initial capital amounts for sudden investor

    print("Starting butterfly effect simulation with sudden investor...")

    # Run baseline simulation
    print("Running baseline simulation...")
    baseline_market = run_baseline_simulation(random_seeds, days=days)

    # Run modified simulations
    modified_markets = []
    for cash in investor_cash_amounts:
        print(f"Running modified simulation (investor cash: {cash:.0f})...")
        modified_market = run_modified_simulation(random_seeds, action_day, cash, days=days)
        modified_markets.append(modified_market)

    # Plot results
    plot_butterfly_effect(baseline_market, modified_markets, action_day, investor_cash_amounts)

    # Analyze butterfly effect
    analyze_butterfly_effect(baseline_market, modified_markets, investor_cash_amounts)

    print("\nSimulation completed. Results saved to 'butterfly_effect_sudden_investor.png'")
    print(f"Global seed: {global_seed}, Base seed: {base_seed}")

if __name__ == "__main__":
    main()
