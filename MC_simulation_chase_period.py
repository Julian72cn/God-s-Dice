"""
MC_simulation_chase_period.py

This script runs multiple stock market model simulations (based on main_OHLC_2.0.5.1.py),
analyzing the impact of ChaseInvestor under different observation periods (N) on the market,
and uses Monte Carlo method for simulation.

The simulation analyzes the impact of different observation period N values on market prices
using Monte Carlo method, helping to understand how ChaseInvestor's behavior affects market prices.

Comparing three observation period settings:
1. Short-term observation period: N = 3
2. Medium-term observation period: N = 10
3. Long-term observation period: N = 20
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
ChaseInvestor = main_module.ChaseInvestor # 显式导入 ChaseInvestor

def run_simulation_with_seed(random_seeds, chase_period=10, enable_trade_log=False):
    """
    Run a single stock market simulation with given random seeds and chase_period, and return the market object.

    Parameters:
        random_seeds: Dictionary containing various random seeds to control randomness in simulation
        chase_period: Observation period N for ChaseInvestor
        enable_trade_log: Whether to enable trade log recording, defaults to False

    Returns:
        Market object containing simulation results, including price history, volume, and other data
    """
    # Basic market parameter settings
    initial_price = 100
    price_tick = 0.01
    days = 800
    buy_fee_rate = 0.001
    sell_fee_rate = 0.001

    # Set whether to enable trade log recording
    Investor.set_enable_trade_log(enable_trade_log)

    # Initialize global trade log file for all investors
    if enable_trade_log:
        Investor.init_trade_log_file(f"all_investors_trade_log_chase_period_{chase_period}.csv")

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
    # Fixed bias_percent_std as this script focuses on chase_period
    fixed_bias_percent_std = 0.30  # Can be adjusted as needed
    investor_bias_percents = np.random.RandomState(random_seeds['value_investor_seed']).normal(0, fixed_bias_percent_std, value_investors_params['num'])

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
        investors.append(ChaseInvestor(  # Use imported ChaseInvestor
            chase_investors_params['initial_shares'],
            chase_investors_params['initial_cash'],
            N=chase_period,  # Use fixed observation period
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

    # Run simulation trading by day
    for day in range(days):
        for investor in investors:
            investor.trade(market.price, market)
        market.daily_auction()

    # Close trade log file
    if enable_trade_log:
        Investor.close_trade_log_file()

    return market

def run_multiple_simulations(num_simulations=10, base_seed=2108, custom_seeds=None, chase_period=10, enable_trade_log=False):
    """
    运行多次模拟，每次使用不同的market_seed值，并收集结果。

    参数:
        num_simulations: 要运行的模拟次数
        base_seed: 随机数生成的基础种子值
        custom_seeds: 可选的自定义随机种子字典，如果提供则使用这些种子值
        chase_period: ChaseInvestor的观察周期N
        enable_trade_log: 是否启用交易日志记录

    返回:
        一个元组，包含：
        - 每次模拟的收盘价格序列列表
        - 第一次模拟的真实价值曲线（当使用相同的value_line_seed时，所有模拟的真实价值曲线相同）
    """
    all_closing_prices = []
    value_curve_data = None  # 存储第一次模拟的真实价值曲线

    print(f"正在运行 {num_simulations} 次模拟，chase_period = {chase_period}...")

    for i in range(num_simulations):
        print(f"模拟 {i+1}/{num_simulations}...")

        # 创建随机种子字典
        if custom_seeds is not None:
            # 使用自定义种子，但为每次模拟使用不同的市场种子
            random_seeds = custom_seeds.copy()
            random_seeds['market_seed'] = base_seed + 1300 + i  # 确保每次模拟使用不同的市场种子
        else:
            # 使用默认方式生成种子
            random_seeds = generate_random_seeds(base_seed=base_seed)
            random_seeds['market_seed'] = base_seed + 1300 + i

        # 运行模拟并获取市场对象
        market = run_simulation_with_seed(random_seeds, chase_period, enable_trade_log)

        # 从市场对象中提取收盘价格
        closing_prices = [ohlc[3] for ohlc in market.ohlc_data]  # OHLC数据格式为(开盘价，最高价，最低价，收盘价)
        all_closing_prices.append(closing_prices)

        # 存储第一次模拟的真实价值曲线
        if i == 0:
            value_curve_data = market.value_history

    return all_closing_prices, value_curve_data

def calculate_statistics(sequences, value_curve=None):
    """
    计算序列列表的统计数据，包括平均序列、标准差序列和波动率。

    参数:
        sequences: 序列列表（列表或数组）
        value_curve: 可选的真实价值曲线，用于计算价格偏离度

    返回:
        包含统计数据的字典：
        - average_sequence: 平均序列
        - std_sequence: 标准差序列
        - volatility: 日收益率的标准差（波动率）
        - avg_volatility: 平均波动率
        - std_volatility: 波动率的标准差
        - individual_volatilities: 每个序列的波动率
        - max_drawdowns: 每个序列的最大回撤
        - avg_max_drawdown: 平均最大回撤
        - price_value_correlation: 价格与价值的相关性
        - price_value_deviation: 价格与价值的平均偏离度
        - upside_volatility: 上行波动率
        - downside_volatility: 下行波动率
    """
    # 确保所有序列具有相同的长度
    min_length = min(len(seq) for seq in sequences)
    sequences_trimmed = [seq[:min_length] for seq in sequences]

    # 转换为numpy数组以便于计算
    sequences_array = np.array(sequences_trimmed)

    # 沿第一个轴计算平均值和标准差（跨所有模拟）
    average_sequence = np.mean(sequences_array, axis=0)
    std_sequence = np.std(sequences_array, axis=0)

    # 计算日收益率
    daily_returns = np.zeros((len(sequences), min_length-1))
    for i, seq in enumerate(sequences_trimmed):
        daily_returns[i] = np.diff(seq) / seq[:-1]

    # 计算波动率（日收益率的标准差）
    volatility = np.std(daily_returns) * 100  # 转换为百分比

    # 计算平均波动率（每个序列的波动率的平均值）
    individual_volatilities = np.std(daily_returns, axis=1) * 100
    avg_volatility = np.mean(individual_volatilities)
    std_volatility = np.std(individual_volatilities)

    # 计算上行和下行波动率
    positive_returns = daily_returns.copy()
    negative_returns = daily_returns.copy()
    positive_returns[positive_returns < 0] = 0
    negative_returns[negative_returns > 0] = 0

    upside_volatility = np.std(positive_returns) * 100
    downside_volatility = np.std(negative_returns) * 100

    # 计算最大回撤
    max_drawdowns = []
    for seq in sequences_trimmed:
        # 计算累计最大值
        running_max = np.maximum.accumulate(seq)
        # 计算回撤
        drawdowns = (running_max - seq) / running_max
        # 最大回撤
        max_drawdown = np.max(drawdowns) * 100  # 转换为百分比
        max_drawdowns.append(max_drawdown)

    avg_max_drawdown = np.mean(max_drawdowns)
    std_max_drawdown = np.std(max_drawdowns)

    # 如果提供了真实价值曲线，计算价格与价值的关系
    price_value_correlation = None
    price_value_deviation = None
    if value_curve is not None and len(value_curve) >= min_length:
        value_curve_trimmed = value_curve[:min_length]
        # 计算价格与价值的相关性
        price_value_correlation = np.corrcoef(average_sequence, value_curve_trimmed)[0, 1]
        # 计算价格与价值的平均偏离度（百分比）
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

def plot_comparison_results(results, value_curve_data, chase_periods_to_plot):
    """
    绘制不同chase_period设置下的平均价格序列、波动性和真实价值曲线。

    参数:
        results: 包含不同chase_period设置下的结果字典
        value_curve_data: 真实价值曲线数据
        chase_periods_to_plot: 用于图例标签的chase_period值列表
    """
    # 创建一个3x2的子图布局
    fig, axs = plt.subplots(3, 2, figsize=(18, 20))

    # 确保所有序列具有相同的长度
    min_length = min(len(stats['average_sequence']) for std, stats in results.items())

    # 使用不同颜色绘制不同chase_period设置下的平均序列
    colors = ['blue', 'green', 'red']
    # labels = [f'Chase Period N={N}' for N in results.keys()] # results.keys() 顺序可能不一致
    labels = [f'Chase Period N={N}' for N in chase_periods_to_plot]

    # 图1: 平均价格曲线
    for i, (std, stats) in enumerate(results.items()):
        avg_sequence_trimmed = stats['average_sequence'][:min_length]
        axs[0, 0].plot(avg_sequence_trimmed, linewidth=2, color=colors[i], label=labels[i])

    # 使用黑色虚线绘制真实价值曲线
    value_curve_trimmed = value_curve_data[:min_length]
    axs[0, 0].plot(value_curve_trimmed, linestyle='--', color='black', linewidth=1.5, label='True Value Curve', zorder=10)

    axs[0, 0].set_title('Average Stock Prices with Different ChaseInvestor Observation Periods (N)')
    axs[0, 0].set_xlabel('Trading Days')
    axs[0, 0].set_ylabel('Price')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # 图2: 价格波动性（标准差）
    for i, (std, stats) in enumerate(results.items()):
        std_sequence_trimmed = stats['std_sequence'][:min_length]
        axs[0, 1].plot(std_sequence_trimmed, linewidth=2, color=colors[i], label=labels[i])

    axs[0, 1].set_title('Price Standard Deviation with Different ChaseInvestor Observation Periods (N)')
    axs[0, 1].set_xlabel('Trading Days')
    axs[0, 1].set_ylabel('Standard Deviation')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # 图3: 价格与真实价值的偏离百分比
    for i, (std, stats) in enumerate(results.items()):
        avg_sequence_trimmed = stats['average_sequence'][:min_length]
        # 计算价格与真实价值的偏离百分比
        deviation_percent = (avg_sequence_trimmed - value_curve_trimmed) / value_curve_trimmed * 100
        axs[1, 0].plot(deviation_percent, linewidth=2, color=colors[i], label=labels[i])

    axs[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axs[1, 0].set_title('Price Deviation from True Value (%)')
    axs[1, 0].set_xlabel('Trading Days')
    axs[1, 0].set_ylabel('Deviation (%)')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # 图4: 波动率比较（条形图）
    x = np.arange(len(results))
    width = 0.35

    volatilities = [stats['avg_volatility'] for std, stats in results.items()]
    volatility_stds = [stats['std_volatility'] for std, stats in results.items()]

    axs[1, 1].bar(x, volatilities, width, yerr=volatility_stds, color=colors, alpha=0.7)
    axs[1, 1].set_title('Average Daily Volatility (%)')
    axs[1, 1].set_ylabel('Volatility (%)')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels([str(N) for N in chase_periods_to_plot])
    axs[1, 1].set_xlabel('ChaseInvestor Observation Period (N)')
    axs[1, 1].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, v in enumerate(volatilities):
        axs[1, 1].text(i, v + volatility_stds[i] + 0.1, f'{v:.2f}%', ha='center')

    # 图5: 上行和下行波动率比较
    upside_volatilities = [stats['upside_volatility'] for std, stats in results.items()]
    downside_volatilities = [stats['downside_volatility'] for std, stats in results.items()]

    x = np.arange(len(results))
    width = 0.35

    axs[2, 0].bar(x - width/2, upside_volatilities, width, color='green', alpha=0.7, label='Upside Volatility')
    axs[2, 0].bar(x + width/2, downside_volatilities, width, color='red', alpha=0.7, label='Downside Volatility')
    axs[2, 0].set_title('Upside vs Downside Volatility (%)')
    axs[2, 0].set_ylabel('Volatility (%)')
    axs[2, 0].set_xticks(x)
    axs[2, 0].set_xticklabels([str(N) for N in chase_periods_to_plot])
    axs[2, 0].set_xlabel('ChaseInvestor Observation Period (N)')
    axs[2, 0].legend()
    axs[2, 0].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, v in enumerate(upside_volatilities):
        axs[2, 0].text(i - width/2, v + 0.1, f'{v:.2f}%', ha='center')
    for i, v in enumerate(downside_volatilities):
        axs[2, 0].text(i + width/2, v + 0.1, f'{v:.2f}%', ha='center')

    # 图6: 最大回撤比较
    max_drawdowns = [stats['avg_max_drawdown'] for std, stats in results.items()]
    max_drawdown_stds = [stats['std_max_drawdown'] for std, stats in results.items()]

    axs[2, 1].bar(x, max_drawdowns, width, yerr=max_drawdown_stds, color=colors, alpha=0.7)
    axs[2, 1].set_title('Average Maximum Drawdown (%)')
    axs[2, 1].set_ylabel('Maximum Drawdown (%)')
    axs[2, 1].set_xticks(x)
    axs[2, 1].set_xticklabels([str(N) for N in chase_periods_to_plot])
    axs[2, 1].set_xlabel('ChaseInvestor Observation Period (N)') # 添加X轴标签
    axs[2, 1].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, v in enumerate(max_drawdowns):
        axs[2, 1].text(i, v + max_drawdown_stds[i] + 0.1, f'{v:.2f}%', ha='center')

    # 保存图表前调整布局
    plt.tight_layout()

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.35)

    # 保存图表
    plt.savefig('chase_period_comparison_results.png', dpi=300, bbox_inches='tight')

    # 显示图表
    plt.show()

    # 返回统计数据以便于生成报告
    return {
        'min_length': min_length,
        'value_curve_trimmed': value_curve_trimmed
    }

def generate_analysis_report(results, plot_data, chase_periods_for_report):
    """
    Generate analysis report containing volatility analysis results under different chase_period settings.

    Parameters:
        results: Dictionary containing results under different chase_period settings
        plot_data: Plotting data, containing minimum sequence length and true value curve
        chase_periods_for_report: List of chase_period values for report tables
    """
    min_length = plot_data['min_length']
    value_curve_trimmed = plot_data['value_curve_trimmed']

    # Create report file
    with open('chase_period_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write("# Analysis of ChaseInvestor Observation Period Impact on Market Volatility\n\n")

        f.write("## Overview\n\n")
        f.write("This report analyzes the impact of ChaseInvestor's observation period (N) parameter on stock market simulation.")
        f.write(f"By running 20 Monte Carlo simulations, we compared three different observation period settings: {chase_periods_for_report[0]} (short), {chase_periods_for_report[1]} (medium), and {chase_periods_for_report[2]} (long).\n\n")

        f.write("## Volatility Analysis\n\n")

        # Create volatility comparison table
        f.write("### Overall Volatility Metrics\n\n")
        f.write(f"| Metric | Observation Period N={chase_periods_for_report[0]} | Observation Period N={chase_periods_for_report[1]} | Observation Period N={chase_periods_for_report[2]} |\n")
        f.write("| --- | ---: | ---: | ---: |\n")

        # Add average volatility
        avg_volatilities = [stats['avg_volatility'] for std, stats in results.items()]
        f.write(f"| Average Daily Volatility (%) | {avg_volatilities[0]:.4f} | {avg_volatilities[1]:.4f} | {avg_volatilities[2]:.4f} |\n")

        # Add volatility standard deviation
        std_volatilities = [stats['std_volatility'] for std, stats in results.items()]
        f.write(f"| Volatility Standard Deviation (%) | {std_volatilities[0]:.4f} | {std_volatilities[1]:.4f} | {std_volatilities[2]:.4f} |\n")

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
        f.write(f"| Metric | Observation Period N={chase_periods_for_report[0]} | Observation Period N={chase_periods_for_report[1]} | Observation Period N={chase_periods_for_report[2]} |\n")
        f.write("| --- | ---: | ---: | ---: |\n")

        # Add average maximum drawdown
        avg_max_drawdowns = [stats['avg_max_drawdown'] for std, stats in results.items()]
        f.write(f"| Average Maximum Drawdown (%) | {avg_max_drawdowns[0]:.4f} | {avg_max_drawdowns[1]:.4f} | {avg_max_drawdowns[2]:.4f} |\n")

        # Add maximum drawdown standard deviation
        std_max_drawdowns = [stats['std_max_drawdown'] for std, stats in results.items()]
        f.write(f"| Maximum Drawdown Standard Deviation (%) | {std_max_drawdowns[0]:.4f} | {std_max_drawdowns[1]:.4f} | {std_max_drawdowns[2]:.4f} |\n\n")

        # Add price-value relationship analysis
        f.write("### Price-Value Relationship Analysis\n\n")
        f.write(f"| Metric | Observation Period N={chase_periods_for_report[0]} | Observation Period N={chase_periods_for_report[1]} | Observation Period N={chase_periods_for_report[2]} |\n")
        f.write("| --- | ---: | ---: | ---: |\n")

        # Add price-value correlation
        price_value_correlations = [stats['price_value_correlation'] for std, stats in results.items()]
        f.write(f"| Price-Value Correlation | {price_value_correlations[0]:.4f} | {price_value_correlations[1]:.4f} | {price_value_correlations[2]:.4f} |\n")

        # Add average price-value deviation
        price_value_deviations = [stats['price_value_deviation'] for std, stats in results.items()]
        f.write(f"| Average Price-Value Deviation (%) | {price_value_deviations[0]:.4f} | {price_value_deviations[1]:.4f} | {price_value_deviations[2]:.4f} |\n\n")

        # Add conclusions and observations
        f.write("## Conclusions and Observations\n\n")

        # Volatility trend analysis
        if avg_volatilities[0] < avg_volatilities[1] < avg_volatilities[2]:
            f.write("1. **Volatility increases with observation period**: Data shows that market volatility significantly increases as ChaseInvestor's observation period increases.\n")
        elif avg_volatilities[0] > avg_volatilities[1] > avg_volatilities[2]:
            f.write("1. **Volatility decreases with observation period**: Data shows that market volatility actually decreases as ChaseInvestor's observation period increases.\n")
        else:
            f.write("1. **Non-linear relationship between volatility and observation period**: Data shows a complex non-linear relationship between volatility and observation period.\n")

        # Upside vs downside volatility analysis
        for i, ratio in enumerate(up_down_ratios):
            if ratio > 1:
                f.write(f"2. **Upside volatility greater than downside volatility for N={chase_periods_for_report[i]}**: Upside/downside volatility ratio is {ratio:.4f}, indicating greater market volatility during price increases than decreases.\n")
            elif ratio < 1:
                f.write(f"2. **Downside volatility greater than upside volatility for N={chase_periods_for_report[i]}**: Upside/downside volatility ratio is {ratio:.4f}, indicating greater market volatility during price decreases than increases.\n")

        # Maximum drawdown analysis
        if avg_max_drawdowns[0] < avg_max_drawdowns[1] < avg_max_drawdowns[2]:
            f.write("3. **Maximum drawdown increases with observation period**: Data shows that market's maximum drawdown increases as ChaseInvestor's observation period increases.\n")
        elif avg_max_drawdowns[0] > avg_max_drawdowns[1] > avg_max_drawdowns[2]:
            f.write("3. **Maximum drawdown decreases with observation period**: Data shows that market's maximum drawdown actually decreases as ChaseInvestor's observation period increases.\n")
        else:
            f.write("3. **Non-linear relationship between maximum drawdown and observation period**: Data shows a complex non-linear relationship between maximum drawdown and observation period.\n")

        # Price-value correlation analysis
        if all(corr > 0 for corr in price_value_correlations):
            max_corr_idx = price_value_correlations.index(max(price_value_correlations))
            f.write(f"4. **Positive price-value correlation for all settings**: Strongest correlation ({max(price_value_correlations):.4f}) observed for N={chase_periods_for_report[max_corr_idx]}.\n")
        elif all(corr < 0 for corr in price_value_correlations):
            min_corr_idx = price_value_correlations.index(min(price_value_correlations))
            f.write(f"4. **Negative price-value correlation for all settings**: Strongest negative correlation ({min(price_value_correlations):.4f}) observed for N={chase_periods_for_report[min_corr_idx]}.\n")
        else:
            f.write("4. **Varying price-value correlation across observation periods**: Different N settings result in significantly different price-value correlations.\n")

        # Summary
        f.write("\n## Summary\n\n")
        f.write("This study demonstrates that ChaseInvestor's observation period parameter has a significant impact on market volatility.")
        f.write("As the observation period N changes, we observe significant changes in market volatility, maximum drawdown, and price-value deviation metrics.")
        f.write("These findings provide important insights into understanding how ChaseInvestor's behavior affects market dynamics.\n")

def main():
    # 设置全局随机数种子，使结果可复现
    global_seed = 42
    np.random.seed(global_seed)

    # 集中设置所有随机数种子
    base_seed = 2108
    random_seeds = {
        'base_seed': base_seed,
        'value_line_seed': base_seed + 345,  # 真实价值曲线种子
        'close_seed': base_seed + 2,       # 收盘价种子
        'market_seed': base_seed + 3,      # 市场种子
        'value_investor_seed': base_seed + 4,  # 价值投资者种子
        'chase_investor_seed': base_seed + 5,  # 追涨投资者种子
        'trend_investor_seed': base_seed + 6,  # 趋势投资者种子
        'random_investor_seed': base_seed + 7,  # 随机投资者种子
        'never_stop_loss_investor_seed': base_seed + 8,  # 永不止损投资者种子
        'bottom_fishing_investor_seed': base_seed + 9,  # 抄底投资者种子
        'insider_investor_seed': base_seed + 10,  # 内幕投资者种子
        'message_investor_seed': base_seed + 11  # 消息投资者种子
    }

    # 要运行的模拟次数
    num_simulations = 20  # 增加模拟次数以提高统计可靠性

    # 不同的chase_period设置
    chase_periods = [3, 10, 20]

    # 存储不同设置的结果
    results = {}
    value_curve_data = None

    # 对每个chase_period值运行多次模拟
    for chase_N in chase_periods:
        # 运行多次模拟
        all_closing_prices, value_curve = run_multiple_simulations(
            num_simulations=num_simulations,
            base_seed=base_seed,
            custom_seeds=random_seeds,
            chase_period=chase_N
        )

        # 计算统计数据
        stats = calculate_statistics(all_closing_prices, value_curve)

        # 存储结果
        results[chase_N] = stats

        # 存储真实价值曲线（所有设置都使用相同的真实价值曲线）
        if value_curve_data is None:
            value_curve_data = value_curve

    # 绘制比较结果
    plot_data = plot_comparison_results(results, value_curve_data, chase_periods)

    # 生成分析报告
    generate_analysis_report(results, plot_data, chase_periods)

    print("模拟完成。结果已保存到 'chase_period_comparison_results.png'")
    print("分析报告已保存到 'chase_period_analysis_report.md'")
    print(f"全局种子: {global_seed}, 基础种子: {base_seed}")

if __name__ == "__main__":
    main()
