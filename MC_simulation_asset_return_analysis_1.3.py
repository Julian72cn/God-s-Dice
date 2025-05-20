"""
MC_simulation_asset_return_analysis.py

This script uses the Monte Carlo method to run multiple stock market simulations (based on main_OHLC_2.0.5.1.py),
aiming to analyze the average asset changes and returns of different types of investors under fixed true value curves
and varying random market behaviors.

Core functionality:
1. Import necessary modules and classes from main_OHLC_2.0.5.1.py.
2. Define a main simulation function that accepts a fixed random seed for generating the stock's true value curve,
   and accepts variable random seeds for market behavior, investor decisions, and other random factors.
3. Run multiple simulations (Monte Carlo iterations):
    a. In each iteration, the true value curve uses the same seed for consistency.
    b. Other random factors (such as market order matching, random initialization of investor decision parameters, etc.)
       use different seeds.
4. Collect data: Record the daily total assets of various types of investors (such as value investors, chase investors,
   trend investors, etc.) in each simulation.
5. Analyze results:
    a. Calculate the average daily total assets of various types of investors across all simulations to obtain average
       asset change curves.
    b. Calculate the average final returns of various types of investors.
    c. Visualize and display average asset curves and related statistical indicators.
"""

import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import os
from collections import defaultdict
import time

# Import main_OHLC_2.0.5.1 module
main_module_path = os.path.join(os.path.dirname(__file__), "main_OHLC_2.0.5.1.py")
spec = importlib.util.spec_from_file_location("main_module", main_module_path)
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)

# Get necessary functions and classes from the module
generate_random_seeds = main_module.generate_random_seeds
Market = main_module.Market
TrueValueCurve = main_module.TrueValueCurve
Investor = main_module.Investor
ValueInvestor = main_module.ValueInvestor
ChaseInvestor = main_module.ChaseInvestor
TrendInvestor = main_module.TrendInvestor
RandomInvestor = main_module.RandomInvestor
NeverStopLossInvestor = main_module.NeverStopLossInvestor
BottomFishingInvestor = main_module.BottomFishingInvestor
InsiderInvestor = main_module.InsiderInvestor
MessageInvestor = main_module.MessageInvestor

def run_single_simulation(days=800, enable_trade_log=False, value_line_seed=None, simulation_seed=None):
    """
    Run a single stock market simulation.

    Parameters:
        days (int): Total number of simulation days.
        enable_trade_log (bool): Whether to enable trade log recording.
        value_line_seed (int, optional): Random seed for generating the true value curve. If None, randomly generated.
        simulation_seed (int, optional): Random seed for other random factors in the simulation (market, investors, etc.).
                                       If None, randomly generated.

    Returns:
        tuple: (market, investors_daily_assets, trend_daily_assets_by_period, seeds)
               market: Market object after simulation ends.
               investors_daily_assets: A dictionary where keys are investor type strings and values are lists of daily
                                     total assets for all investors of that type during the simulation.
               trend_daily_assets_by_period: A dictionary where keys are trend periods and values are lists of daily
                                           total assets for trend investors of that period.
               seeds: Dictionary containing all random seeds used in this simulation.
    """
    # If seeds are not provided, generate them randomly
    if value_line_seed is None:
        value_line_seed = np.random.randint(1, 100000)
    if simulation_seed is None:
        simulation_seed = np.random.randint(1, 100000)

    # Record seeds used in this simulation for subsequent analysis
    used_seeds = {'value_line_seed': value_line_seed, 'simulation_seed': simulation_seed}

    # Generate seeds for market and investors
    random_seeds = generate_random_seeds(simulation_seed)
    random_seeds['value_line_seed'] = value_line_seed  # Set value curve seed

    # Basic market parameter settings
    initial_price = 100
    price_tick = 0.01
    buy_fee_rate = 0.001
    sell_fee_rate = 0.001

    Investor.set_enable_trade_log(enable_trade_log)
    if enable_trade_log:
        log_dir = "trade_logs_monte_carlo"
        os.makedirs(log_dir, exist_ok=True)
        Investor.init_trade_log_file(os.path.join(log_dir, f"mc_trade_log_sim_{simulation_seed}.csv"))

    # Create true value curve
    value_curve = TrueValueCurve(initial_value=initial_price, days=days, seed=random_seeds['value_line_seed'])

    # Create market instance
    market = Market(initial_price, price_tick, value_curve=value_curve,
                   seed=random_seeds['market_seed'],
                   close_seed=random_seeds['close_seed'],
                   buy_fee_rate=buy_fee_rate, sell_fee_rate=sell_fee_rate)

    # Parameter settings for different types of investors
    investor_params_template = {'initial_shares': 100, 'initial_cash': 10000}
    investor_configs = {
        "ValueInvestor": {'num': 50, **investor_params_template},
        "ChaseInvestor": {'num': 50, **investor_params_template},
        "TrendInvestor": {'num': 50, **investor_params_template},
        "RandomInvestor": {'num': 50, **investor_params_template},
        "NeverStopLossInvestor": {'num': 10, **investor_params_template},
        "BottomFishingInvestor": {'num': 10, **investor_params_template},
        "InsiderInvestor": {'num': 5, **investor_params_template},
        "MessageInvestor": {'num': 5, **investor_params_template}
    }

    investors = []
    investor_groups = defaultdict(list)

    # Create value investors
    params = investor_configs["ValueInvestor"]
    bias_percent_std = 0.30
    investor_bias_percents = np.random.RandomState(random_seeds['value_investor_seed']).normal(0, bias_percent_std, params['num'])
    for i, bias_percent in enumerate(investor_bias_percents):
        max_deviation = np.random.RandomState(random_seeds['value_investor_seed'] + i).uniform(0.2, 0.4)
        target_position = np.random.RandomState(random_seeds['value_investor_seed'] + i).uniform(0.4, 0.6)
        inv = ValueInvestor(params['initial_shares'], params['initial_cash'], bias_percent=bias_percent, max_deviation=max_deviation, target_position=target_position, seed=random_seeds['value_investor_seed'] + i)
        investors.append(inv)
        investor_groups["ValueInvestor"].append(inv)

    # Create chase investors
    params = investor_configs["ChaseInvestor"]
    for i in range(params['num']):
        inv = ChaseInvestor(params['initial_shares'], params['initial_cash'], N=np.random.RandomState(random_seeds['chase_investor_seed'] + i).randint(5, 21), seed=random_seeds['chase_investor_seed'] + i)
        investors.append(inv)
        investor_groups["ChaseInvestor"].append(inv)

    # Create trend investors
    params = investor_configs["TrendInvestor"]
    trend_periods = [5, 10, 15, 20, 25, 30, 60, 100, 150, 200]
    num_per_period = params['num'] // len(trend_periods)
    extra_investors = params['num'] % len(trend_periods)
    current_investor_idx = 0
    for period_idx, period in enumerate(trend_periods):
        num_to_create = num_per_period + (1 if period_idx < extra_investors else 0)
        for i in range(num_to_create):
            inv = TrendInvestor(params['initial_shares'], params['initial_cash'], period, seed=random_seeds['trend_investor_seed'] + current_investor_idx)
            investors.append(inv)
            investor_groups["TrendInvestor"].append(inv)
            current_investor_idx += 1

    # Create random investors
    params = investor_configs["RandomInvestor"]
    for i in range(params['num']):
        inv = RandomInvestor(params['initial_shares'], params['initial_cash'], seed=random_seeds['random_investor_seed'] + i)
        investors.append(inv)
        investor_groups["RandomInvestor"].append(inv)

    # Create never stop loss investors
    params = investor_configs["NeverStopLossInvestor"]
    for i in range(params['num']):
        inv = NeverStopLossInvestor(params['initial_shares'], params['initial_cash'], buy_probability=0.3, initial_profit_target=0.15, min_profit_target=0.02, seed=random_seeds['never_stop_loss_investor_seed'] + i)
        investors.append(inv)
        investor_groups["NeverStopLossInvestor"].append(inv)

    # Create bottom fishing investors
    params = investor_configs["BottomFishingInvestor"]
    for i in range(params['num']):
        seed = random_seeds['bottom_fishing_investor_seed'] + i
        rng_bf = np.random.RandomState(seed)
        profit_target = rng_bf.uniform(0.1, 0.5)
        inv = BottomFishingInvestor(params['initial_shares'], params['initial_cash'], profit_target=profit_target, seed=seed)
        investors.append(inv)
        investor_groups["BottomFishingInvestor"].append(inv)

    # Create insider investors
    params = investor_configs["InsiderInvestor"]
    for i in range(params['num']):
        inv = InsiderInvestor(params['initial_shares'], params['initial_cash'], seed=random_seeds['insider_investor_seed'] + i)
        investors.append(inv)
        investor_groups["InsiderInvestor"].append(inv)

    # Create message investors
    params = investor_configs["MessageInvestor"]
    for i in range(params['num']):
        inv = MessageInvestor(params['initial_shares'], params['initial_cash'], seed=random_seeds['message_investor_seed'] + i)
        investors.append(inv)
        investor_groups["MessageInvestor"].append(inv)

    # Initialize daily asset recorder
    investors_daily_assets = {investor_type: [0.0 for _ in range(days)] for investor_type in investor_groups.keys()}

    # Create separate categories for trend investors by period
    trend_investors_by_period = defaultdict(list)
    for inv in investor_groups["TrendInvestor"]:
        trend_investors_by_period[inv.M].append(inv)

    # Initialize daily asset records for trend investors by period
    trend_daily_assets_by_period = {period: [0.0 for _ in range(days)] for period in trend_investors_by_period.keys()}

    # Run simulation trading by day
    for day in range(days):
        # Collect trading orders from investors
        for investor in investors:
            investor.trade(market.price, market)
        # Market daily auction (including opening, continuous trading, and closing)
        market.daily_auction()

        # Record total assets of each type of investor for the day
        for investor_type, group in investor_groups.items():
            # Calculate total assets = cash + number of shares * current price
            current_day_total_assets = sum(inv.cash + inv.shares * market.price_history[-1] for inv in group)
            investors_daily_assets[investor_type][day] = current_day_total_assets

        # Record daily assets for trend investors by period
        for period, group in trend_investors_by_period.items():
            period_total_assets = sum(inv.cash + inv.shares * market.price_history[-1] for inv in group)
            trend_daily_assets_by_period[period][day] = period_total_assets

        if (day + 1) % 100 == 0:
            print(f"  Sim {simulation_seed}, Day {day + 1}/{days} completed.")

    return market, investors_daily_assets, trend_daily_assets_by_period, used_seeds

def calculate_trend_performance(trend_daily_assets_by_period, days):
    """
    Calculate performance metrics for trend investors of different periods.

    Parameters:
        trend_daily_assets_by_period: Daily asset data for trend investors categorized by period
        days: Total number of simulation days

    Returns:
        Dictionary containing performance metrics for trend investors of each period
    """
    performance = {}

    for period, assets in trend_daily_assets_by_period.items():
        initial_assets = assets[0]
        final_assets = assets[-1]

        # Calculate total return
        total_return = (final_assets - initial_assets) / initial_assets

        # Calculate annualized return
        years = days / 365
        annualized_return = (1 + total_return) ** (1 / years) - 1

        performance[period] = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'average_assets': np.mean(assets)
        }

    return performance


def save_trend_performance_results(trend_performance, output_dir="trend_performance_results"):
    """
    Save trend investor performance results to CSV file.

    Parameters:
        trend_performance: Trend investor performance data
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save total return and annualized return
    with open(os.path.join(output_dir, "trend_performance.csv"), "w") as f:
        f.write("Period,Total Return,Annualized Return,Average Assets\n")
        for period, perf in trend_performance.items():
            f.write(f"{period},{perf['total_return']},{perf['annualized_return']},{perf['average_assets']}\n")

    print(f"Trend investor performance results saved to {output_dir} directory")


def plot_trend_performance(trend_performance, output_dir="trend_performance_results"):
    """
    Plot trend investor performance charts.

    Parameters:
        trend_performance: Trend investor performance data
        output_dir: Output directory
    """
    periods = sorted(trend_performance.keys())
    total_returns = [trend_performance[p]['total_return'] for p in periods]
    annualized_returns = [trend_performance[p]['annualized_return'] for p in periods]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(periods, total_returns)
    plt.title("Total Returns for Trend Investors by Period")
    plt.xlabel("Period (Days)")
    plt.ylabel("Total Return")

    plt.subplot(1, 2, 2)
    plt.bar(periods, annualized_returns)
    plt.title("Annualized Returns for Trend Investors by Period")
    plt.xlabel("Period (Days)")
    plt.ylabel("Annualized Return")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trend_performance.png"))
    plt.close()

    print(f"Trend investor performance charts saved to {output_dir} directory")


def run_monte_carlo_simulation(num_simulations, days=800, enable_trade_log=False, use_fixed_value_line=False):
    """
    Run Monte Carlo simulation.

    Parameters:
        num_simulations (int): Total number of simulations.
        days (int): Number of days for each simulation.
        enable_trade_log (bool): Whether to enable trade log.
        use_fixed_value_line (bool): Whether to use the same value curve in all simulations. If True, the value curve
                                    seed from the first simulation will be used for all subsequent simulations.

    Returns:
        dict: Dictionary containing average daily assets, average final returns, trend investors' average daily assets
              by period, and used seeds.
              {'avg_daily_assets': {investor_type: [avg_asset_day1, ..., avg_asset_dayN]},
               'avg_final_returns': {investor_type: avg_return},
               'avg_trend_daily_assets': {period: [avg_asset_day1, ..., avg_asset_dayN]},
               'trend_performance': {period: {'total_return': value, 'annualized_return': value}},
               'seeds': [List of seeds used for each simulation]}
    """
    all_sim_investor_assets = defaultdict(lambda: [0.0] * days)
    all_sim_investor_final_assets = defaultdict(list)
    all_sim_trend_assets_by_period = defaultdict(lambda: [0.0] * days)
    all_sim_trend_final_assets_by_period = defaultdict(list)
    initial_total_assets_per_type = defaultdict(float)
    initial_trend_assets_by_period = defaultdict(float)
    all_seeds = []
    fixed_value_seed = None

    print(f"Starting Monte Carlo simulation with {num_simulations} runs...")
    total_start_time = time.time()

    for i in range(num_simulations):
        sim_start_time = time.time()
        print(f"Running simulation {i + 1}/{num_simulations}...")

        # If using fixed value curve and not the first simulation, use the value curve seed from the first simulation
        value_seed = fixed_value_seed if (use_fixed_value_line and i > 0) else None

        market, daily_assets, trend_daily_assets_by_period, seeds = run_single_simulation(
            days=days,
            enable_trade_log=enable_trade_log,
            value_line_seed=value_seed,
            simulation_seed=None  # Use randomly generated simulation seed each time
        )

        # If it's the first simulation and fixed value curve is needed, save the value curve seed
        if i == 0 and use_fixed_value_line:
            fixed_value_seed = seeds['value_line_seed']

        # Record seeds used in this simulation
        all_seeds.append(seeds)
        sim_end_time = time.time()
        print(f"Simulation {i + 1} finished in {sim_end_time - sim_start_time:.2f} seconds.")

        first_run = (i == 0)
        # Process data for all investor types
        for investor_type, assets_over_time in daily_assets.items():
            if first_run:
                # Get initial total assets for this type of investor (based on day 0 data from first simulation)
                # Assume that initial configuration (shares and cash) is the same for all investors of the same type
                # Therefore, it's reasonable to get initial total assets from day 0 of the first simulation
                # Need to ensure run_single_simulation correctly records assets at the end of day 0
                # Or calculate initial total assets immediately after creating investors
                # For simplicity, let's assume initial assets are based on day 0 of the first sim
                # A more robust way would be to calculate it from initial_shares * initial_price + initial_cash for each investor type
                # Let's refine this: calculate initial total assets before simulation starts for each type
                if investor_type not in initial_total_assets_per_type:
                    # Calculate initial total assets based on investor configuration in first simulation
                    # Use market and daily_assets from first simulation to get initial assets
                    # Here we directly use the sum of assets on day 0 as initial assets
                    initial_total_assets_per_type[investor_type] = assets_over_time[0]

            for day_idx, day_asset in enumerate(assets_over_time):
                all_sim_investor_assets[investor_type][day_idx] += day_asset
            all_sim_investor_final_assets[investor_type].append(assets_over_time[-1])

        # Process data for trend investors by period
        for period, assets_over_time in trend_daily_assets_by_period.items():
            if first_run:
                if period not in initial_trend_assets_by_period:
                    initial_trend_assets_by_period[period] = assets_over_time[0]

            for day_idx, day_asset in enumerate(assets_over_time):
                all_sim_trend_assets_by_period[period][day_idx] += day_asset
            all_sim_trend_final_assets_by_period[period].append(assets_over_time[-1])

    avg_daily_assets = {
        inv_type: [total_asset / num_simulations for total_asset in daily_totals]
        for inv_type, daily_totals in all_sim_investor_assets.items()
    }

    avg_trend_daily_assets = {
        period: [total_asset / num_simulations for total_asset in daily_totals]
        for period, daily_totals in all_sim_trend_assets_by_period.items()
    }

    avg_final_returns = {}
    for inv_type, final_assets_list in all_sim_investor_final_assets.items():
        if initial_total_assets_per_type[inv_type] > 0:
            avg_final_asset = np.mean(final_assets_list)
            avg_final_returns[inv_type] = (avg_final_asset - initial_total_assets_per_type[inv_type]) / initial_total_assets_per_type[inv_type]
        else:
            avg_final_returns[inv_type] = 0 # Avoid division by zero if no initial assets

    # Calculate performance for trend investors by period
    trend_performance = {}
    for period, final_assets_list in all_sim_trend_final_assets_by_period.items():
        if initial_trend_assets_by_period[period] > 0:
            avg_final_asset = np.mean(final_assets_list)
            total_return = (avg_final_asset - initial_trend_assets_by_period[period]) / initial_trend_assets_by_period[period]
            # Calculate annualized return (assuming 250 trading days per year)
            annualized_return = ((1 + total_return) ** (250 / days) - 1) if total_return > -1 else -1
            trend_performance[period] = {
                'total_return': total_return,
                'annualized_return': annualized_return
            }
        else:
            trend_performance[period] = {
                'total_return': 0,
                'annualized_return': 0
            }

    total_end_time = time.time()
    print(f"Monte Carlo simulation completed in {total_end_time - total_start_time:.2f} seconds.")

    return {
        'avg_daily_assets': avg_daily_assets,
        'avg_final_returns': avg_final_returns,
        'avg_trend_daily_assets': avg_trend_daily_assets,
        'trend_performance': trend_performance,
        'seeds': all_seeds
    }

def plot_average_asset_curves(avg_daily_assets, days, seeds, num_simulations, use_fixed_value_line=False):
    """
    绘制各类投资者平均每日资产增值百分比曲线。
    """
    plt.figure(figsize=(14, 8))

    # 计算每个投资者类型的资产增值百分比
    for investor_type, assets_curve in avg_daily_assets.items():
        if len(assets_curve) == days:
            # 计算初始资产值
            initial_asset = assets_curve[0]
            # 计算每日资产增值百分比
            asset_percent_change = [(asset / initial_asset - 1) * 100 for asset in assets_curve]
            plt.plot(range(days), asset_percent_change, label=f'{investor_type} (Avg)')
        else:
            print(f"Warning: Asset curve length mismatch for {investor_type}. Expected {days}, got {len(assets_curve)}.")

    # 获取标题信息
    title = f'Average Daily Asset Percentage Change (Monte Carlo: {num_simulations} runs)'
    if use_fixed_value_line and seeds and len(seeds) > 0:
        value_seed = seeds[0]['value_line_seed']
        title += f'\nFixed Value Line Seed: {value_seed}'

    plt.title(title)
    plt.xlabel('Day')
    plt.ylabel('Asset Change (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    results_dir = "monte_carlo_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = os.path.join(results_dir, f"avg_asset_curves_t{timestamp}_n{num_simulations}.png")
    plt.savefig(filename)
    print(f"Average asset curves plot saved to {filename}")
    # plt.show() # Comment out if running in a non-interactive environment or saving many plots
    plt.close()


def plot_trend_asset_curves(avg_trend_daily_assets, days, seeds, num_simulations, use_fixed_value_line=False):
    """
    绘制不同周期趋势投资者的平均每日资产增值百分比曲线。
    """
    plt.figure(figsize=(14, 8))

    # 计算每个周期趋势投资者的资产增值百分比
    for period, assets_curve in sorted(avg_trend_daily_assets.items()):
        if len(assets_curve) == days:
            # 计算初始资产值
            initial_asset = assets_curve[0]
            # 计算每日资产增值百分比
            asset_percent_change = [(asset / initial_asset - 1) * 100 for asset in assets_curve]
            plt.plot(range(days), asset_percent_change, label=f'Period {period} (Avg)')
        else:
            print(f"Warning: Asset curve length mismatch for period {period}. Expected {days}, got {len(assets_curve)}.")

    # 获取标题信息
    title = f'Average Daily Asset Percentage Change by Trend Period (Monte Carlo: {num_simulations} runs)'
    if use_fixed_value_line and seeds and len(seeds) > 0:
        value_seed = seeds[0]['value_line_seed']
        title += f'\nFixed Value Line Seed: {value_seed}'

    plt.title(title)
    plt.xlabel('Day')
    plt.ylabel('Asset Change (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    results_dir = "monte_carlo_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = os.path.join(results_dir, f"trend_asset_curves_t{timestamp}_n{num_simulations}.png")
    plt.savefig(filename)
    print(f"Trend investor asset curves plot saved to {filename}")
    # plt.show() # Comment out if running in a non-interactive environment or saving many plots
    plt.close()

def plot_average_final_returns(avg_final_returns, seeds, num_simulations, use_fixed_value_line=False, days=250):
    """
    绘制各类投资者平均最终收益率和年化收益率的条形图。

    参数:
        avg_final_returns: 各类投资者的平均最终收益率
        seeds: 使用的随机种子列表
        num_simulations: 模拟次数
        use_fixed_value_line: 是否使用固定价值曲线
        days: 模拟天数，用于计算年化收益率
    """
    investor_types = list(avg_final_returns.keys())
    returns = [avg_final_returns[it] * 100 for it in investor_types] # 转换为百分比

    # 计算年化收益率 (假设一年有250个交易日)
    annual_returns = [((1 + avg_final_returns[it]) ** (250 / days) - 1) * 100 for it in investor_types]

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))

    # 设置柱状图的位置
    x = np.arange(len(investor_types))
    width = 0.35

    # 绘制总收益率柱状图
    bars1 = ax.bar(x - width/2, returns, width, label='Total Return', color='steelblue')

    # 绘制年化收益率柱状图
    bars2 = ax.bar(x + width/2, annual_returns, width, label='Annualized Return', color='lightcoral')

    # 获取标题信息
    title = f'Average Returns per Investor Type (Monte Carlo: {num_simulations} runs)'
    if use_fixed_value_line and seeds and len(seeds) > 0:
        value_seed = seeds[0]['value_line_seed']
        title += f'\nFixed Value Line Seed: {value_seed}'

    ax.set_title(title)
    ax.set_xlabel('Investor Type')
    ax.set_ylabel('Return (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(investor_types, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--')

    # 在柱状图上添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top')

    add_labels(bars1)
    add_labels(bars2)

    plt.tight_layout()

    # 保存图像
    results_dir = "monte_carlo_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = os.path.join(results_dir, f"avg_final_returns_t{timestamp}_n{num_simulations}.png")
    plt.savefig(filename)
    print(f"Average final returns plot saved to {filename}")
    # plt.show()
    plt.close()


def plot_trend_returns(trend_performance, seeds, num_simulations, use_fixed_value_line=False):
    """
    绘制不同周期趋势投资者的平均最终收益率和年化收益率的条形图。

    参数:
        trend_performance: 不同周期趋势投资者的绩效数据
        seeds: 使用的随机种子列表
        num_simulations: 模拟次数
        use_fixed_value_line: 是否使用固定价值曲线
    """
    periods = sorted(trend_performance.keys())
    total_returns = [trend_performance[p]['total_return'] * 100 for p in periods]  # 转换为百分比
    annual_returns = [trend_performance[p]['annualized_return'] * 100 for p in periods]  # 转换为百分比

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))

    # 设置柱状图的位置
    x = np.arange(len(periods))
    width = 0.35

    # 绘制总收益率柱状图
    bars1 = ax.bar(x - width/2, total_returns, width, label='Total Return', color='steelblue')

    # 绘制年化收益率柱状图
    bars2 = ax.bar(x + width/2, annual_returns, width, label='Annualized Return', color='lightcoral')

    # 获取标题信息
    title = f'Average Returns by Trend Period (Monte Carlo: {num_simulations} runs)'
    if use_fixed_value_line and seeds and len(seeds) > 0:
        value_seed = seeds[0]['value_line_seed']
        title += f'\nFixed Value Line Seed: {value_seed}'

    ax.set_title(title)
    ax.set_xlabel('Trend Period (Days)')
    ax.set_ylabel('Return (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(periods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--')

    # 在柱状图上添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top')

    add_labels(bars1)
    add_labels(bars2)

    plt.tight_layout()

    # 保存图像
    results_dir = "monte_carlo_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = os.path.join(results_dir, f"trend_returns_t{timestamp}_n{num_simulations}.png")
    plt.savefig(filename)
    print(f"Trend investor returns plot saved to {filename}")
    # plt.show()
    plt.close()

if __name__ == "__main__":
    # Simulation parameters
    num_simulations = 100  # Total number of simulations
    days = 800  # Number of days for each simulation
    enable_trade_log = False  # Whether to enable trade log
    use_fixed_value_line = True  # Whether to use the same value curve in all simulations

    # Run Monte Carlo simulation
    results = run_monte_carlo_simulation(
        num_simulations=num_simulations,
        days=days,
        enable_trade_log=enable_trade_log,
        use_fixed_value_line=use_fixed_value_line
    )

    # Create output directory
    output_dir = "monte_carlo_results"
    os.makedirs(output_dir, exist_ok=True)

    # Plot average asset curves
    plot_average_asset_curves(
        results['avg_daily_assets'],
        results['seeds'],
        num_simulations,
        output_dir
    )

    # Plot average final returns
    plot_average_final_returns(
        results['avg_final_returns'],
        results['seeds'],
        num_simulations,
        days,
        output_dir
    )

    # Calculate and save trend performance results
    trend_performance = calculate_trend_performance(
        results['avg_trend_daily_assets'],
        days
    )
    save_trend_performance_results(trend_performance, output_dir)

    # Plot trend performance
    plot_trend_performance(trend_performance, output_dir)

    # Output random seeds used in simulations
    print("\nRandom seeds used in simulations:")
    for i, seeds in enumerate(results['seeds']):
        print(f"Simulation {i + 1}:")
        print(f"  Value line seed: {seeds['value_line_seed']}")
        print(f"  Simulation seed: {seeds['simulation_seed']}")
