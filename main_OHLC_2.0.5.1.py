"""
Updates to main_OHLC_2.0.5.1.py:

- Added independent random seeds for all random number generation
- Added the generate_random_seeds function to generate a dictionary of random seeds
- Modified the simulate_stock_market function to accept a random_seeds dictionary parameter
- Updated the ChaseInvestor class by adding a seed parameter and initializing the random number generator
- Updated the TrendInvestor class by adding a seed parameter and initializing the random number generator
- Modified the code for creating various investor types to use seeds from the random_seeds dictionary
- Updated the main function to provide multiple ways of setting random seeds
These changes allow all random seeds to be centrally configured, making it easier to adjust and reproduce simulation results

"""

import numpy as np
import matplotlib.pyplot as plt

class TrueValueCurve:
    """
    股票真实价值曲线生成器
    - 根据随机种子生成特定的价值曲线
    - 支持通过索引访问价值数据
    - 在指定日期范围内生成随机方波变化的价值序列

    属性:
        initial_value: 初始价值
        days: 生成的天数
        seed: 随机数种子
        values: 生成的价值序列
    """
    def __init__(self, initial_value=100, days=2000, seed=422):
        self.initial_value = initial_value
        self.days = days
        self.seed = seed
        self.values = self._generate_curve()

    def _generate_curve(self):
        """生成价值曲线,使用随机方波变化"""
        rng = np.random.RandomState(self.seed)
        values = [self.initial_value]
        true_value = self.initial_value
        last_jump_day = 0
        next_jump_interval = rng.randint(30, 50)

        for day in range(1, self.days+1):
            if day - last_jump_day >= next_jump_interval:
                if rng.rand() < 0.33:  # 33%的概率发生价值跳变
                    change = rng.uniform(10, 30) * (1 if rng.rand() < 0.5 else -1)
                    true_value += change
                    true_value = max(0, true_value)  # 确保价值不为负
                last_jump_day = day
                next_jump_interval = rng.randint(30, 50)
            values.append(true_value)

        return values

    def __getitem__(self, idx):
        """支持用索引访问价值"""
        return self.values[idx]

    def __len__(self):
        """支持len()函数"""
        return len(self.values)

class Investor:
    """
    投资者基类 - 定义了所有投资者的基本属性和方法
    Base Investor class - Defines basic properties and methods for all investors

    属性 (Properties):
        shares: 持有的股票数量 (Number of shares held)
        cash: 持有的现金 (Cash held)
    """
    # 类变量，用于存储所有投资者的交易日志文件
    trade_log_file = None
    investor_count = 0
    # 日志记录开关，默认开启
    enable_trade_log = True

    @classmethod
    def init_trade_log_file(cls, filename="all_investors_trade_log.csv"):
        """初始化交易日志文件，写入表头"""
        if cls.enable_trade_log:
            cls.trade_log_file = open(filename, "w")
            cls.trade_log_file.write("Day,Investor_Type,Investor_ID,Action,Price,Shares,Current_Shares,Cash,Entry_Price,Profit_Ratio,Additional_Info\n")
            cls.trade_log_file.flush()

    @classmethod
    def close_trade_log_file(cls):
        """关闭交易日志文件"""
        if cls.trade_log_file:
            cls.trade_log_file.close()
            cls.trade_log_file = None

    @classmethod
    def set_enable_trade_log(cls, enable=True):
        """设置是否启用交易日志记录"""
        cls.enable_trade_log = enable

    def __init__(self, shares, cash):
        self.shares = shares  # 持有股票数量 (shares held)
        self.cash = cash      # 持有现金 (cash held)
        # 为每个投资者分配一个唯一ID
        self.investor_id = Investor.investor_count
        Investor.investor_count += 1
        # 记录买入价格/入场价格
        self.entry_price = None

    def get_investor_type(self):
        """获取投资者类型名称"""
        return self.__class__.__name__

    def log_trade(self, day, action, price, shares, profit_ratio=0.0, additional_info=None):
        """记录交易到日志文件"""
        if Investor.enable_trade_log and Investor.trade_log_file:
            entry_price_str = str(self.entry_price) if hasattr(self, 'entry_price') and self.entry_price is not None else 'None'
            Investor.trade_log_file.write(f"{day},{self.get_investor_type()},{self.investor_id},{action},{price:.2f},{shares},{self.shares},{self.cash:.2f},{entry_price_str},{profit_ratio:.4f},{additional_info if additional_info is not None else ''}\n")
            Investor.trade_log_file.flush()

    def trade(self, price, market):
        """执行交易的方法，由子类实现 (Trading method to be implemented by subclasses)"""
        pass

    def decide_price(self, current_price, market):
        """决定交易价格和数量的方法，由子类实现 (Method to decide trading price and quantity, implemented by subclasses)
        返回值 (Returns): (action, price, shares)
            action: 'buy', 'sell' 或 'hold'
            price: 交易价格 (trading price)
            shares: 交易数量 (number of shares)
        """
        return ('hold', 0, 0)

class ValueInvestor(Investor):
    """
    价值投资者 - 基于对股票内在价值的估计进行交易
    Value Investor - Trades based on estimated intrinsic value of the stock

    策略 (Strategy):
    - 每个投资者有固定的估值偏差百分比(bias_percent)
    - 所有投资者的偏差百分比整体服从正态分布
    - 估值 = 真实价值 * (1 + 偏差百分比)
    - 投资者不会立即对价值变化做出反应，而是有1-20天的滞后期
    - 通过持仓比例控制买卖行为：
      - 当价格等于估值时，保持目标持仓比例（默认50%）
      - 当价格高于估值时，减少持仓比例，偏离达到最大值时空仓
      - 当价格低于估值时，增加持仓比例，偏离达到最大值时满仓

    属性 (Properties):
        bias_percent: 固定的估值偏差百分比 (Fixed value estimation bias percentage)
        revaluation_delay: 重新估值的延迟天数 (Days of delay before revaluation)
        last_valuation_day: 上次估值的日期 (Day of last valuation)
        value_estimate: 当前估值 (Current value estimation)
        max_deviation: 最大偏离比例，达到此比例时满仓或空仓 (Maximum deviation ratio)
        target_position: 目标持仓比例，价格等于估值时的持仓比例 (Target position ratio)
    """
    def __init__(self, shares, cash, bias_percent, max_deviation=0.3, target_position=0.5, seed=None):
        super().__init__(shares, cash)
        self.bias_percent = bias_percent  # 固定的估值偏差百分比
        self.max_deviation = max_deviation  # 最大偏离比例，达到此比例时满仓或空仓
        self.target_position = target_position  # 目标持仓比例，价格等于估值时的持仓比例
        self._rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))
        self.revaluation_delay = self._rng.randint(1, 21)  # 随机生成1-20天的重新估值延迟
        self.last_valuation_day = 0  # 初始化为第0天
        self.value_estimate = None  # 初始估值为None

    def decide_price(self, current_price, market):
        """根据上一交易日收盘价与估值的差异决定交易行为"""
        current_day = len(market.price_history) - 1

        # 使用上一交易日的收盘价而不是当前价格
        # 如果是第一天交易（没有上一日收盘价），则使用当前价格
        if current_day > 0:
            decision_price = market.price_history[current_day - 1]  # 上一交易日收盘价
        else:
            decision_price = current_price  # 第一天使用当前价格

        # 初始化估值或达到重新估值时间时更新估值
        if self.value_estimate is None or (current_day - self.last_valuation_day) >= self.revaluation_delay:
            true_value = market.value_curve[current_day]
            self.value_estimate = true_value * (1 + self.bias_percent)
            self.last_valuation_day = current_day

        # 如果还没有估值，不进行交易
        if self.value_estimate is None:
            return ('hold', 0, 0)

        # 计算价格偏离度 - 使用上一交易日收盘价
        # 防止除以零错误
        if self.value_estimate == 0:
            # 如果估值为零，则将差异设为一个大值，表示价格远高于估值
            diff = 1.0 if decision_price > 0 else -1.0
        else:
            diff = (decision_price - self.value_estimate) / self.value_estimate

        # 计算目标持仓比例 - 根据价格偏离度线性调整
        # 当diff=0时，持仓比例为target_position
        # 当diff=max_deviation时，持仓比例为0（空仓）
        # 当diff=-max_deviation时，持仓比例为1（满仓）
        target_ratio = max(0, min(1, self.target_position - (diff / self.max_deviation) * self.target_position))

        # 计算当前总资产价值 - 仍使用当前价格计算总资产
        total_assets = self.cash + self.shares * current_price

        # 计算目标持股数量 - 使用当前价格计算目标持股数量
        target_shares = int((total_assets * target_ratio) / current_price)

        # 计算需要买入或卖出的股票数量
        shares_diff = target_shares - self.shares

        if shares_diff > 0:  # 需要买入
            # 检查现金是否足够
            max_affordable = int(self.cash / current_price)
            buy_shares = min(shares_diff, max_affordable)
            if buy_shares > 0:
                buy_price = current_price * 1.01  # 稍高于市价的买入价格
                return ('buy', buy_price, buy_shares)
        elif shares_diff < 0:  # 需要卖出
            sell_shares = -shares_diff  # 转为正数
            if sell_shares > 0:
                sell_price = current_price * 0.99  # 稍低于市价的卖出价格
                return ('sell', sell_price, sell_shares)

        return ('hold', 0, 0)  # 无交易信号

    def trade(self, price, market):
        """执行具体的交易操作"""
        current_day = len(market.price_history) - 1

        # 记录交易前状态
        old_shares = self.shares
        old_cash = self.cash

        # 直接执行交易决策，不需要更新估值
        action, order_price, shares = self.decide_price(price, market)

        # 计算当前盈亏比例
        profit_ratio = 0
        if self.shares > 0 and self.entry_price is not None:
            profit_ratio = (price - self.entry_price) / self.entry_price

        # 记录交易决策
        value_estimate_str = f"{self.value_estimate:.2f}" if self.value_estimate is not None else "0"
        self.log_trade(current_day, f"decide_{action}", price, shares, profit_ratio,
                      f"order_price={order_price:.2f}, value_estimate={value_estimate_str}")

        if action == 'buy':
            # 买入时记录入场价格
            if self.shares == 0 or self.entry_price is None:  # 新建仓位或入场价格为None
                self.entry_price = price
            else:  # 增加仓位，计算加权平均成本
                self.entry_price = (self.entry_price * self.shares + price * shares) / (self.shares + shares)
            market.place_order(action, order_price, shares, self)
        elif action == 'sell':
            # 卖出全部持仓时重置入场价格
            if shares >= self.shares:
                self.entry_price = None
            market.place_order(action, order_price, shares, self)

        # 记录交易后的状态变化
        if old_shares != self.shares or old_cash != self.cash:
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

class ChaseInvestor(Investor):
    """
    追涨杀跌投资者 - 仅基于价格变化速度进行交易
    Chase Investor - Trades based solely on price change velocity

    策略 (Strategy):
    - 当股价上涨速度越快，买入越多（按剩余现金比例） (Buy more when price rises faster, proportional to remaining cash)
    - 当股价下跌速度越快，卖出越多（按剩余股票比例） (Sell more when price falls faster, proportional to remaining shares)
    - 交易量与价格变化速度成正比 (Trading volume proportional to price change velocity)
    - 不受市场情绪影响 (Not affected by market sentiment)

    属性 (Properties):
        N: 观察周期，用于计算价格变化速度 (Observation period for calculating price change velocity)
    """
    def __init__(self, shares, cash, N=None, seed=None):
        super().__init__(shares, cash)
        self.N = N  # 观察周期 (observation period)
        self._n_initialized = False  # 标记是否已初始化N
        self._rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))

    def calculate_velocity(self, prices):
        """计算价格变化速度
        Calculate price change velocity"""
        if len(prices) < 2:
            return 0.0
        # 计算价格变化率 (Calculate price change rate)
        price_changes = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:  # 确保除数大于0，避免除以零或负数
                change = (prices[i] - prices[i-1]) / prices[i-1]
                price_changes.append(change)
        # 检查是否有有效的价格变化数据
        if not price_changes:
            return 0.0
        # 返回平均变化率 (Return average change rate)
        return float(sum(price_changes)) / len(price_changes)  # 确保返回浮点数，并再次检查列表非空

    def decide_price(self, current_price, market):
        """根据上一交易日收盘价的变化速度决定交易行为
        Decide trading action based on previous day's closing price velocity"""
        current_day = len(market.price_history) - 1

        # 使用上一交易日的收盘价而不是当前价格
        # 如果是第一天交易（没有上一日收盘价），则使用当前价格
        if current_day > 0:
            decision_price = market.price_history[current_day - 1]  # 上一交易日收盘价
        else:
            decision_price = current_price  # 第一天使用当前价格

        # 确保 N 已经被初始化
        if self.N is None and not self._n_initialized:
            self.N = market._rng.choice([3, 5, 10, 15, 20])
            self._n_initialized = True

        if len(market.price_history) >= self.N:
            recent_prices = market.price_history[-self.N:]  # 获取近期价格历史 (get recent price history)
            velocity = self.calculate_velocity(recent_prices)  # 计算价格变化速度 (calculate price change velocity)

            if velocity > 0:  # 价格上涨趋势 (Price rising trend)
                # 买入比例仅与速度成正比
                buy_ratio = min(abs(velocity) * 5, 1)  # 速度影响增大
                # 确保现金充足
                if self.cash > 0:
                    buy_shares = int((self.cash * buy_ratio) / current_price)
                    if buy_shares > 0:
                        # 固定的买入价格溢价
                        buy_price = current_price * 1.02
                        return ('buy', buy_price, buy_shares)
            elif velocity < 0:  # 价格下跌趋势 (Price falling trend)
                # 卖出比例仅与速度成正比
                sell_ratio = min(abs(velocity) * 5, 1)  # 速度影响增大
                # 确保持股充足
                if self.shares > 0:
                    sell_shares = int(self.shares * sell_ratio)
                    if sell_shares > 0:
                        # 固定的卖出价格折扣
                        sell_price = current_price * 0.98
                        return ('sell', sell_price, sell_shares)

            # 考虑交易量反馈
            if len(market.executed_volume_history) > 5:
                volume_change = market.executed_volume_history[-1] / max(1, np.mean(market.executed_volume_history[-6:-1]))
                if volume_change > 1.2 and velocity > 0:  # 交易量增加且价格上涨
                    # 量价齐升，增强追涨力度，但仅基于速度和交易量
                    buy_ratio = min(1.0, abs(velocity) * 5 * min(2.0, volume_change))
                    # 确保现金充足
                    if self.cash > 0:
                        buy_shares = int((self.cash * buy_ratio) / current_price)
                        if buy_shares > 0:
                            buy_price = current_price * 1.02  # 固定的买入价格溢价
                            return ('buy', buy_price, buy_shares)

        return ('hold', 0, 0)  # 无交易信号 (no trading signal)

    def trade(self, price, market):
        """执行具体的交易操作
        Execute specific trading operation"""
        current_day = len(market.price_history) - 1

        # 记录交易前状态
        old_shares = self.shares
        old_cash = self.cash

        action, order_price, shares = self.decide_price(price, market)

        # 计算当前盈亏比例
        profit_ratio = 0
        if self.shares > 0 and self.entry_price is not None:
            profit_ratio = (price - self.entry_price) / self.entry_price

        # 记录交易决策
        velocity = 0
        if len(market.price_history) >= self.N:
            recent_prices = market.price_history[-self.N:]
            velocity = self.calculate_velocity(recent_prices)
        self.log_trade(current_day, f"decide_{action}", price, shares, profit_ratio,
                      f"order_price={order_price:.2f}, velocity={velocity:.4f}, N={self.N}")

        if action == 'buy':
            # 买入时记录入场价格
            if self.shares == 0 or self.entry_price is None:  # 新建仓位或入场价格为None
                self.entry_price = price
            else:  # 增加仓位，计算加权平均成本
                self.entry_price = (self.entry_price * self.shares + price * shares) / (self.shares + shares)
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            # 卖出全部持仓时重置入场价格
            if shares >= self.shares:
                self.entry_price = None
            market.place_order('sell', order_price, shares, self)

        # 记录交易后的状态变化
        if old_shares != self.shares or old_cash != self.cash:
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

class TrendInvestor(Investor):
    """
    趋势投资者 - 基于移动平均线进行交易
    Trend Investor - Trades based on moving average

    策略 (Strategy):
    - 当价格首次上穿移动平均线时全仓买入 (Buy all when price first crosses above moving average)
    - 当价格首次下穿移动平均线时全仓卖出 (Sell all when price first crosses below moving average)

    属性 (Properties):
        M: 移动平均线周期 (Moving average period)
        above_ma: 记录价格是否在均线上方 (Record if price is above MA)
    """
    def __init__(self, shares, cash, M, seed=None):
        super().__init__(shares, cash)
        self.M = M  # 移动平均线周期 (moving average period)
        self.above_ma = None  # 记录价格是否在均线上方 (Record if price is above MA)
        self._rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))

    def decide_price(self, current_price, market):
        """根据上一交易日收盘价与移动平均线的关系决定交易行为
        Decide trading action based on previous day's closing price's relationship with moving average"""
        current_day = len(market.price_history) - 1

        # 使用上一交易日的收盘价而不是当前价格
        # 如果是第一天交易（没有上一日收盘价），则使用当前价格
        if current_day > 0:
            decision_price = market.price_history[current_day - 1]  # 上一交易日收盘价
        else:
            decision_price = current_price  # 第一天使用当前价格

        if len(market.price_history) >= self.M:
            # 计算简单移动平均线 (calculate simple moving average)
            recent_prices = market.price_history[-self.M:]
            if not recent_prices or len(recent_prices) < self.M:  # 检查是否有足够的价格数据
                return ('hold', 0, 0)
            sma = float(sum(recent_prices)) / len(recent_prices)  # 已经检查过列表非空，直接使用长度作为除数

            # 使用上一交易日收盘价与移动平均线比较
            current_above_ma = decision_price > sma

            # 检测价格穿越均线 (detect price crossing MA)
            if self.above_ma is None:
                self.above_ma = current_above_ma
            elif current_above_ma != self.above_ma:  # 价格穿越均线 (price crosses MA)
                self.above_ma = current_above_ma
                if current_above_ma:  # 上穿均线，全仓买入 (crosses above MA, buy all)
                    buy_shares = self.cash // current_price  # 计算可买入的最大股数 (calculate maximum shares to buy)
                    if buy_shares > 0:
                        buy_price = current_price * 1.01
                        return ('buy', buy_price, buy_shares)
                else:  # 下穿均线，全仓卖出 (crosses below均线，sell all)
                    if self.shares > 0:
                        sell_price = current_price * 0.99
                        return ('sell', sell_price, self.shares)
        return ('hold', 0, 0)  # 无交易信号 (no trading signal)

    def trade(self, price, market):
        current_day = len(market.price_history) - 1

        # 记录交易前状态
        old_shares = self.shares
        old_cash = self.cash

        action, order_price, shares = self.decide_price(price, market)

        # 计算当前盈亏比例
        profit_ratio = 0
        if self.shares > 0 and self.entry_price is not None:
            profit_ratio = (price - self.entry_price) / self.entry_price

        # 记录交易决策
        ma = 0
        if len(market.price_history) >= self.M:
            ma = sum(market.price_history[-self.M:]) / self.M
        self.log_trade(current_day, f"decide_{action}", price, shares, profit_ratio,
                      f"order_price={order_price:.2f}, ma={ma:.2f}, M={self.M}")

        if action == 'buy':
            # 买入时记录入场价格
            if self.shares == 0 or self.entry_price is None:  # 新建仓位或入场价格为None
                self.entry_price = price
            else:  # 增加仓位，计算加权平均成本
                self.entry_price = (self.entry_price * self.shares + price * shares) / (self.shares + shares)
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            # 卖出全部持仓时重置入场价格
            if shares >= self.shares:
                self.entry_price = None
            market.place_order('sell', order_price, shares, self)

        # 记录交易后的状态变化
        if old_shares != self.shares or old_cash != self.cash:
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

class RandomInvestor(Investor):
    """
    随机投资者 - 模拟受市场情绪影响的非理性交易行为
    Random Investor - Simulates irrational trading behavior influenced by market sentiment

    策略 (Strategy):
    - 随机决定买入或卖出，但受市场情绪影响 (Randomly decides to buy or sell, influenced by market sentiment)
    - 市场情绪高涨时买入概率增加，卖出概率减小 (Higher buy probability and lower sell probability when sentiment is high)
    - 市场情绪低迎时买入概率减小，卖出概率增加 (Lower buy probability and higher sell probability when sentiment is low)
    - 交易比例受市场情绪影响 (Trading ratio influenced by market sentiment)
    - 价格偏离范围固定，不受市场情绪影响 (Fixed price deviation range, not influenced by market sentiment)

    属性 (Properties):
        p: 基础交易概率，实际概率受市场情绪影响 (Base probability of trading, actual probability influenced by market sentiment)
    """
    def __init__(self, shares, cash, p=0.2, seed=None):
        super().__init__(shares, cash)
        self.p = p  # 交易概率 (trading probability)
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))

    def decide_price(self, current_price, market):
        """基于市场情绪和随机因素决定交易行为、价格和交易比例
        Decide trading action, price and trading ratio based on market sentiment and random factors"""
        # 根据市场情绪调整买入和卖出概率
        # 市场情绪高涨时，增加买入概率，减小卖出概率
        # 市场情绪低迎时，减小买入概率，增加卖出概率
        sentiment = market.market_sentiment  # 市场情绪值（0-1之间）

        # 调整买入和卖出概率
        buy_prob = self.p * (0.5 + sentiment)  # 市场情绪高时买入概率增加
        sell_prob = self.p * (1.5 - sentiment)  # 市场情绪高时卖出概率减小

        # 确保概率总和不超过1
        hold_prob = max(0, 1 - (buy_prob + sell_prob))

        # 随机选择交易行为：买入、卖出或持有
        action = self._rng_investor.choice(['buy', 'sell', 'hold'], p=[buy_prob, sell_prob, hold_prob])

        if action == 'buy':
            # 市场情绪高时交易比例范围增大
            min_ratio = 0.1 + 0.2 * sentiment  # 范围从0.1到0.3
            max_ratio = 0.6 + 0.4 * sentiment  # 范围从0.6到1.0
            random_ratio = self._rng_investor.uniform(min_ratio, max_ratio)

            buy_shares = int(self.cash * random_ratio / current_price)
            if buy_shares > 0 and self.cash >= buy_shares * current_price:
                # 使用固定的价格偏离范围，不受市场情绪影响
                min_price_factor = 0.95  # 固定的最小价格因子
                max_price_factor = 1.05  # 固定的最大价格因子
                price_factor = market._rng.uniform(min_price_factor, max_price_factor)
                buy_price = current_price * price_factor
                return ('buy', buy_price, buy_shares)
        elif action == 'sell':
            # 市场情绪低时交易比例范围增大
            min_ratio = 0.1 + 0.2 * (1 - sentiment)  # 范围从0.1到0.3
            max_ratio = 0.6 + 0.4 * (1 - sentiment)  # 范围从0.6到1.0
            random_ratio = self._rng_investor.uniform(min_ratio, max_ratio)

            sell_shares = int(self.shares * random_ratio)
            if sell_shares > 0:
                # 使用固定的价格偏离范围，不受市场情绪影响
                min_price_factor = 0.95  # 固定的最小价格因子
                max_price_factor = 1.05  # 固定的最大价格因子
                price_factor = market._rng.uniform(min_price_factor, max_price_factor)
                sell_price = current_price * price_factor
                return ('sell', sell_price, sell_shares)
        return ('hold', 0, 0)  # 无交易信号 (No trading signal)

    def trade(self, price, market):
        """执行具体的交易操作
        Execute specific trading operation"""
        current_day = len(market.price_history) - 1

        # 记录交易前状态
        old_shares = self.shares
        old_cash = self.cash

        action, order_price, shares = self.decide_price(price, market)

        # 计算当前盈亏比例
        profit_ratio = 0
        if self.shares > 0 and self.entry_price is not None:
            profit_ratio = (price - self.entry_price) / self.entry_price

        # 记录交易决策
        sentiment_str = f"{market.market_sentiment:.2f}" if hasattr(market, 'market_sentiment') else "N/A"
        self.log_trade(current_day, f"decide_{action}", price, shares, profit_ratio,
                      f"order_price={order_price:.2f}, p={self.p:.2f}, sentiment={sentiment_str}")

        if action == 'buy':
            # 买入时记录入场价格
            if self.shares == 0 or self.entry_price is None:  # 新建仓位或入场价格为None
                self.entry_price = price
            else:  # 增加仓位，计算加权平均成本
                self.entry_price = (self.entry_price * self.shares + price * shares) / (self.shares + shares)
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            # 卖出全部持仓时重置入场价格
            if shares >= self.shares:
                self.entry_price = None
            market.place_order('sell', order_price, shares, self)

        # 记录交易后的状态变化
        if old_shares != self.shares or old_cash != self.cash:
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

class NeverStopLossInvestor(Investor):
    """
    永不止损投资者 - 买入后如果浮亏则持续持有直到回本
    Never Stop Loss Investor - Holds positions until price recovers to entry point

    策略 (Strategy):
    - 随机时刻可能全仓买入 (May buy all-in at random moments)
    - 买入后如果浮亏，持续持有 (If in loss after buying, continues to hold)
    - 只有当股价回升到买入价格以上时才考虑卖出 (Only considers selling when price rises above entry point)
    - 盈利目标动态调整：没有经历大幅浮亏时目标较高，大幅亏损持有时间越长目标越低 (Profit target adjusts dynamically)

    属性 (Properties):
        buy_price: 买入价格，如果未持仓则为None (Entry price, None if not holding)
        buy_probability: 买入概率 (Probability of buying)
        initial_profit_target: 初始目标盈利比例 (Initial target profit percentage)
        min_profit_target: 最小目标盈利比例 (Minimum target profit percentage)
        max_loss_ratio: 记录持有期间的最大亏损比例 (Maximum loss ratio during holding period)
        loss_days: 大幅亏损状态的持有天数 (Days held during significant loss)
    """
    # 类变量
    investor_count = 0

    def __init__(self, shares, cash, buy_probability=0.001, initial_profit_target=0.15, min_profit_target=0.02, seed=None):
        super().__init__(shares, cash)
        self.buy_price = None  # 买入价格，初始为None (Entry price, initially None)
        self.buy_probability = buy_probability  # 买入概率 (Probability of buying)
        self.initial_profit_target = initial_profit_target  # 初始目标盈利比例 (Initial target profit percentage)
        self.min_profit_target = min_profit_target  # 最小目标盈利比例 (Minimum target profit percentage)
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))

        # 新增属性用于跟踪浮亏状态
        self.max_loss_ratio = 0.0  # 记录持有期间的最大亏损比例
        self.loss_days = 0  # 大幅亏损状态的持有天数
        self.significant_loss_threshold = -0.1  # 大幅亏损的阈值，默认-10%

        # 为每个投资者分配一个唯一ID
        self.investor_id = NeverStopLossInvestor.investor_count
        NeverStopLossInvestor.investor_count += 1

    # 移除特定日志文件记录方法

    def calculate_dynamic_profit_target(self):
        """根据浮亏状态计算动态盈利目标"""
        # 如果没有经历大幅浮亏，使用初始目标
        if self.max_loss_ratio > self.significant_loss_threshold or self.loss_days == 0:
            return self.initial_profit_target

        # 根据浮亏幅度和持有天数计算动态盈利目标
        # 浮亏越大，持有天数越长，盈利目标越低
        loss_factor = abs(self.max_loss_ratio / self.significant_loss_threshold)  # 浮亏因子，越大越接近或超过1
        time_factor = min(1.0, self.loss_days / 15.0)  # 时间因子，最多为1（当持有15天或以上）

        # 综合因子，范围0-1，越大表示盈利目标越低
        adjustment_factor = loss_factor * time_factor

        # 计算动态盈利目标，从初始目标线性降低到最小目标
        dynamic_target = self.initial_profit_target - (self.initial_profit_target - self.min_profit_target) * adjustment_factor

        # 确保不低于最小目标
        return max(dynamic_target, self.min_profit_target)

    def decide_price(self, current_price, market):
        """根据当前持仓状态和价格决定交易行为
        Decide trading action based on current position status and price"""
        # 强制打印当前状态
        current_day = len(market.price_history) - 1
        random_value = self._rng_investor.random()

        # 不再记录状态到特定日志文件

        # 如果没有持股，应该重置buy_price为None
        if self.shares == 0 and self.buy_price is not None:
            old_buy_price = self.buy_price
            self.buy_price = None
            self.max_loss_ratio = 0.0
            self.loss_days = 0
            # 不再记录重置操作到特定日志文件

        # 如果没有持仓，考虑买入
        if self.shares == 0 and self.buy_price is None:
            # 随机决定是否买入
            if random_value < self.buy_probability:
                # 确保现金充足
                if self.cash > 0:
                    # 全仓买入
                    buy_shares = int(self.cash / current_price)
                    if buy_shares > 0:
                        self.buy_price = current_price  # 记录买入价格
                        buy_price = current_price * 1.01  # 接受稍高的买入价格
                        # 重置浮亏跟踪参数
                        self.max_loss_ratio = 0.0
                        self.loss_days = 0
                        # 不再记录买入操作到特定日志文件
                        return ('buy', buy_price, buy_shares)

        # 如果有持仓，更新浮亏状态和考虑卖出
        elif self.shares > 0 and self.buy_price is not None:
            # 计算当前盈亏比例
            profit_ratio = (current_price - self.buy_price) / self.buy_price

            # 更新最大亏损比例
            if profit_ratio < 0 and profit_ratio < self.max_loss_ratio:
                self.max_loss_ratio = profit_ratio

            # 更新大幅亏损状态的持有天数
            if profit_ratio < self.significant_loss_threshold:
                self.loss_days += 1

            # 计算动态盈利目标
            current_profit_target = self.calculate_dynamic_profit_target()

            # 不再记录持有状态到特定日志文件

            # 如果盈利达到动态目标，卖出
            if profit_ratio >= current_profit_target:
                # 确保持股充足
                if self.shares > 0:
                    sell_price = current_price * 0.99  # 接受稍低的卖出价格
                    sell_shares = self.shares  # 全部卖出
                    self.buy_price = None  # 重置买入价格
                    # 重置浮亏跟踪参数
                    self.max_loss_ratio = 0.0
                    self.loss_days = 0
                    # 不再记录卖出操作到特定日志文件
                    return ('sell', sell_price, sell_shares)

        return ('hold', 0, 0)  # 无交易信号或持有等待回本

    def trade(self, price, market):
        """执行具体的交易操作
        Execute specific trading operation"""
        current_day = len(market.price_history) - 1

        # 记录交易前状态
        old_shares = self.shares
        old_cash = self.cash

        # 检查是否持有股票但buy_price为None的异常情况
        if self.shares > 0 and self.buy_price is None:
            # 如果持有股票但没有买入价格记录，使用当前价格作为买入价格
            self.buy_price = price
            # 重置浮亏跟踪参数
            self.max_loss_ratio = 0.0
            self.loss_days = 0
            # 记录到全局日志
            self.log_trade(current_day, "fix_buy_price", price, self.shares, 0.0,
                          f"max_loss_ratio={self.max_loss_ratio:.4f}, loss_days={self.loss_days}")

        action, order_price, shares = self.decide_price(price, market)

        # 计算当前盈亏比例
        profit_ratio = 0
        if self.shares > 0 and self.buy_price is not None:
            profit_ratio = (price - self.buy_price) / self.buy_price

        # 记录交易决策到全局日志
        dynamic_target = self.calculate_dynamic_profit_target() if self.shares > 0 else None
        buy_price_str = f"{self.buy_price}" if self.buy_price is not None else "None"
        dynamic_target_str = f"{dynamic_target:.4f}" if dynamic_target is not None else "0"
        self.log_trade(current_day, f"decide_{action}", price, shares, profit_ratio,
                      f"order_price={order_price:.2f}, buy_price={buy_price_str}, "
                      f"max_loss_ratio={self.max_loss_ratio:.4f}, loss_days={self.loss_days}, "
                      f"dynamic_target={dynamic_target_str}")

        # 执行交易
        if action == 'buy':
            # 记录入场价格
            self.entry_price = price
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            # 卖出全部持仓时重置入场价格
            if shares >= self.shares:
                self.entry_price = None
            market.place_order('sell', order_price, shares, self)

        # 记录交易后的状态变化到全局日志
        if old_shares != self.shares or old_cash != self.cash:
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

        # 交易后再次检查是否持有股票但buy_price为None的异常情况
        if self.shares > 0 and self.buy_price is None:
            self.buy_price = price
            self.max_loss_ratio = 0.0
            self.loss_days = 0
            # 记录修复操作到全局日志
            self.log_trade(current_day, "fix_buy_price_after_trade", price, self.shares, 0.0,
                          f"max_loss_ratio={self.max_loss_ratio:.4f}, loss_days={self.loss_days}")

class BottomFishingInvestor(Investor):
    """
    抄底投资者 - 从100天内最高价下跌x%开始分批买入
    Bottom Fishing Investor - Buys in batches after price drops x% from 100-day high

    属性 (Properties):
        profit_target: 盈利目标比例（10%-50%）
        avg_cost: 持仓加权平均成本
        trigger_drop: 触发买入的下跌比例(5%-15%)
        step_drop: 每次下跌买入的比例(5%-15%)
    """
    def __init__(self, shares, cash, profit_target=None, seed=None):
        super().__init__(shares, cash)
        self.avg_cost = None  # 持仓加权平均成本
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))
        self.profit_target = profit_target if profit_target is not None else self._rng_investor.uniform(0.1, 0.5)
        self.trigger_drop = self._rng_investor.uniform(0.10, 0.20)  # 触发买入的下跌比例
        self.step_drop = self._rng_investor.uniform(0.10, 0.30)    # 每次下跌买入的比例
        self.last_buy_price = None  # 记录上次买入价格

    def decide_price(self, current_price, market):
        # 卖出逻辑：达到盈利目标就卖出全部
        if self.shares > 0 and self.avg_cost is not None:
            profit_ratio = (current_price - self.avg_cost) / self.avg_cost
            if profit_ratio >= self.profit_target:
                # 确保持股充足
                if self.shares > 0:
                    sell_price = current_price * 0.99
                    return ('sell', sell_price, self.shares)

        # 买入逻辑：从100天内最高价下跌trigger_drop%开始，每下跌step_drop%买入一次
        if len(market.price_history) >= 50:
            peak_price = max(market.price_history[-100:])
            drop_from_peak = (peak_price - current_price) / peak_price

            # 检查是否达到触发买入的下跌比例
            if drop_from_peak >= self.trigger_drop:
                # 计算下跌了多少个step
                drop_steps = int((drop_from_peak - self.trigger_drop) / self.step_drop)
                if drop_steps > 0:
                    # 下跌越多，买入比例越高（最大使用80%现金）
                    buy_ratio = min(0.8, 0.1 + drop_steps * 0.1)
                    # 确保现金充足
                    if self.cash > 0:
                        buy_shares = int((self.cash * buy_ratio) / current_price)
                        if buy_shares > 0:
                            buy_price = current_price * 1.01
                            return ('buy', buy_price, buy_shares)

        return ('hold', 0, 0)

    def trade(self, price, market):
        current_day = len(market.price_history) - 1

        # 记录交易前状态
        old_shares = self.shares
        old_cash = self.cash

        action, order_price, shares = self.decide_price(price, market)

        # 计算当前盈亏比例
        profit_ratio = 0
        if self.shares > 0 and self.avg_cost is not None:
            profit_ratio = (price - self.avg_cost) / self.avg_cost

        # 记录交易决策
        # 计算从最高点下跌的百分比
        drop_from_peak = 0
        if len(market.price_history) >= 50:
            peak_price = max(market.price_history[-100:])
            drop_from_peak = (peak_price - price) / peak_price

        avg_cost_str = f"{self.avg_cost:.2f}" if self.avg_cost is not None else "None"
        self.log_trade(current_day, f"decide_{action}", price, shares, profit_ratio,
                      f"order_price={order_price:.2f}, avg_cost={avg_cost_str}, "
                      f"drop_from_peak={drop_from_peak:.4f}, trigger_drop={self.trigger_drop:.4f}, step_drop={self.step_drop:.4f}")

        if action == 'buy':
            # 更新加权平均成本
            total_cost = (self.avg_cost * self.shares if self.avg_cost is not None else 0) + order_price * shares
            total_shares = self.shares + shares
            self.avg_cost = total_cost / total_shares if total_shares > 0 else None
            # 记录入场价格
            self.entry_price = self.avg_cost
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)
            self.avg_cost = None  # 清空持仓后成本重置
            self.entry_price = None  # 清空持仓后入场价格重置

        # 记录交易后的状态变化
        if old_shares != self.shares or old_cash != self.cash:
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

class InsiderInvestor(Investor):
    """
    内幕交易者 - 能提前知道股票价值变化的投资者
    属性:
        prediction_days: 提前知道价值变化的天数(1-5天)
        profit_target: 目标盈利比例(默认15%)
        stop_loss: 止损比例(默认20%)
        max_hold_days: 最大持有天数(默认30天)
        holding_days: 当前持仓天数
        entry_price: 买入价格
    """
    def __init__(self, shares, cash, prediction_days=None, profit_target=0.15, stop_loss=0.20, max_hold_days=30, seed=None):
        super().__init__(shares, cash)
        self._rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))
        self.prediction_days = prediction_days if prediction_days is not None else self._rng.randint(1, 6)
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_hold_days = max_hold_days
        self.holding_days = 0
        self.entry_price = None

    def decide_price(self, current_price, market):
        # 更新持仓天数
        if self.shares > 0 and self.entry_price is not None:
            self.holding_days += 1
            profit_ratio = (current_price - self.entry_price) / self.entry_price
            if (profit_ratio >= self.profit_target or
                profit_ratio <= -self.stop_loss or
                self.holding_days >= self.max_hold_days):
                # 确保持股充足
                if self.shares > 0:
                    sell_price = current_price * 0.99
                    sell_shares = self.shares
                    self.entry_price = None
                    self.holding_days = 0
                    return ('sell', sell_price, sell_shares)

        # 检查未来价值变化
        future_value_change = market.get_future_value_change(self.prediction_days)
        if future_value_change is not None:
            if future_value_change > 0 and self.shares == 0:
                # 确保现金充足
                if self.cash > 0:
                    buy_shares = int(self.cash * 0.8 / current_price)
                    if buy_shares > 0:
                        buy_price = current_price * 1.01
                        self.entry_price = current_price
                        self.holding_days = 0
                        return ('buy', buy_price, buy_shares)
            elif future_value_change < 0 and self.shares > 0:
                # 确保持股充足
                if self.shares > 0:
                    sell_price = current_price * 0.99
                    sell_shares = self.shares
                    self.entry_price = None
                    self.holding_days = 0
                    return ('sell', sell_price, sell_shares)
        return ('hold', 0, 0)

    def trade(self, price, market):
        current_day = len(market.price_history) - 1

        # 记录交易前状态
        old_shares = self.shares
        old_cash = self.cash

        action, order_price, shares = self.decide_price(price, market)

        # 计算当前盈亏比例
        profit_ratio = 0
        if self.shares > 0 and self.entry_price is not None:
            profit_ratio = (price - self.entry_price) / self.entry_price

        # 记录交易决策
        future_value_change = market.get_future_value_change(self.prediction_days)
        future_value_change_str = f"{future_value_change:.4f}" if future_value_change is not None else "None"
        self.log_trade(current_day, f"decide_{action}", price, shares, profit_ratio,
                      f"order_price={order_price:.2f}, prediction_days={self.prediction_days}, "
                      f"future_value_change={future_value_change_str}, "
                      f"holding_days={self.holding_days}")

        if action in ['buy', 'sell']:
            market.place_order(action, order_price, shares, self)

        # 记录交易后的状态变化
        if old_shares != self.shares or old_cash != self.cash:
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

class MessageInvestor(Investor):
    """
    消息投资者 - 在价值变化后1-5天才获知股票价值变化的投资者
    属性:
        delay_days: 获知价值变化的延迟天数(1-5天)
        profit_target: 目标盈利比例(默认15%)
        stop_loss: 止损比例(默认20%)
        max_hold_days: 最大持有天数(默认30天)
        holding_days: 当前持仓天数
        entry_price: 买入价格
    """
    def __init__(self, shares, cash, delay_days=None, profit_target=0.15, stop_loss=0.20, max_hold_days=30, seed=None):
        super().__init__(shares, cash)
        self._rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))
        self.delay_days = delay_days if delay_days is not None else self._rng.randint(1, 6)
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_hold_days = max_hold_days
        self.holding_days = 0
        self.entry_price = None

    def decide_price(self, current_price, market):
        # 更新持仓天数
        if self.shares > 0 and self.entry_price is not None:
            self.holding_days += 1
            profit_ratio = (current_price - self.entry_price) / self.entry_price
            if (profit_ratio >= self.profit_target or
                profit_ratio <= -self.stop_loss or
                self.holding_days >= self.max_hold_days):
                # 确保持股充足
                if self.shares > 0:
                    sell_price = current_price * 0.99
                    sell_shares = self.shares
                    self.entry_price = None
                    self.holding_days = 0
                    return ('sell', sell_price, sell_shares)

        # 获取延迟的价值变化信息
        current_day = len(market.price_history) - 1
        if market.value_curve is not None and current_day >= self.delay_days:
            past_value = market.value_curve[current_day - self.delay_days]
            current_value = market.value_curve[current_day]
            value_change = current_value - past_value

            if abs(value_change) > 0:
                if value_change > 0 and self.shares == 0:
                    # 确保现金充足
                    if self.cash > 0:
                        buy_shares = int(self.cash * 0.8 / current_price)
                        if buy_shares > 0:
                            buy_price = current_price * 1.01
                            self.entry_price = current_price
                            self.holding_days = 0
                            return ('buy', buy_price, buy_shares)
                elif value_change < 0 and self.shares > 0:
                    # 确保持股充足
                    if self.shares > 0:
                        sell_price = current_price * 0.99
                        sell_shares = self.shares
                        self.entry_price = None
                        self.holding_days = 0
                        return ('sell', sell_price, sell_shares)
        return ('hold', 0, 0)

    def trade(self, price, market):
        current_day = len(market.price_history) - 1

        # 记录交易前状态
        old_shares = self.shares
        old_cash = self.cash

        action, order_price, shares = self.decide_price(price, market)

        # 计算当前盈亏比例
        profit_ratio = 0
        if self.shares > 0 and self.entry_price is not None:
            profit_ratio = (price - self.entry_price) / self.entry_price

        # 记录交易决策
        past_value_change = None
        if len(market.value_history) > self.delay_days:
            past_value_change = market.value_history[-1] - market.value_history[-1-self.delay_days]

        past_value_change_str = f"{past_value_change:.4f}" if past_value_change is not None else "None"
        self.log_trade(current_day, f"decide_{action}", price, shares, profit_ratio,
                      f"order_price={order_price:.2f}, delay_days={self.delay_days}, "
                      f"past_value_change={past_value_change_str}, "
                      f"holding_days={self.holding_days}")

        if action in ['buy', 'sell']:
            market.place_order(action, order_price, shares, self)

        # 记录交易后的状态变化
        if old_shares != self.shares or old_cash != self.cash:
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

class Market:
    """
    股票市场 - 实现集合竞价机制的市场模拟器
    Stock Market - Market simulator implementing call auction mechanism

    策略 (Strategy):
    - 收集买卖订单 (Collect buy and sell orders)
    - 通过集合竞价确定成交价格 (Determine clearing price through call auction)
    - 按价格优先原则执行交易 (Execute trades based on price priority)
    - 股票价值为随机方波变化 (Stock value changes as a random square wave)
    - 支持在指定日期注入或抽离资金 (Support capital injection and withdrawal on specified dates)
    - 实现交易手续费机制 (Implement transaction fee mechanism)

    属性 (Properties):
        price: 当前市场价格 (Current market price)
        price_tick: 最小价格变动单位 (Minimum price movement unit)
        price_history: 价格历史记录 (Price history)
        buy_orders: 买单队列 (Buy order queue)
        sell_orders: 卖单队列 (Sell order queue)
        executed_volume: 成交量 (Executed trading volume)
        true_value: 股票真实价值 (True stock value)
        value_history: 价值历史 (Value history)
        seed: 随机数种子 (Random seed)
        capital_change_history: 资金变动历史记录 (Capital change history)
        buy_fee_rate: 买入交易手续费率 (Buy transaction fee rate)
        sell_fee_rate: 卖出交易手续费率 (Sell transaction fee rate)
        fee_income: 累计手续费收入 (Accumulated fee income)
    """
    def __init__(self, initial_price, price_tick=0.01, value_curve=None, seed=None, close_seed=None, buy_fee_rate=0.001, sell_fee_rate=0.002):
        self.price = initial_price
        self.price_tick = price_tick
        self.price_history = [initial_price]
        self.buy_orders = []
        self.sell_orders = []
        self.executed_volume = 0
        self.value_curve = value_curve  # 引用外部传入的价值曲线
        self.true_value = value_curve[0] if value_curve is not None else initial_price
        self.value_history = [self.true_value]
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)
        self._rng = np.random.RandomState(self.seed)

        # 为收盘价设置独立的随机数种子
        self.close_seed = close_seed if close_seed is not None else (seed + 20240501 if seed is not None else np.random.randint(0, 1000000))
        self._rng_close = np.random.RandomState(self.close_seed)

        self.executed_volume_history = []
        self.capital_change_history = []
        self.buy_fee_rate = buy_fee_rate
        self.sell_fee_rate = sell_fee_rate
        self.fee_income = 0

        # 市场情绪相关属性
        self.market_sentiment = 0.5  # 初始化为中性情绪（0-1之间，0.5为中性）
        self.sentiment_history = [0.5]  # 市场情绪历史记录
        self.volume_ma = None  # 交易量移动平均线
        self.volume_ma_period = 10  # 交易量移动平均线周期

    def update_market_sentiment(self):
        """更新市场情绪指标
        基于价格变化、交易量变化和当前情绪状态计算新的市场情绪指标
        """
        # 计算价格变化
        if len(self.price_history) > 1:
            price_change = (self.price_history[-1] / self.price_history[-2] - 1)
        else:
            price_change = 0

        # 计算交易量变化
        if len(self.executed_volume_history) > 1:
            # 更新交易量移动平均线
            if len(self.executed_volume_history) >= self.volume_ma_period:
                self.volume_ma = np.mean(self.executed_volume_history[-self.volume_ma_period:])
            else:
                self.volume_ma = np.mean(self.executed_volume_history)

            # 计算交易量相对于移动平均线的变化
            if self.volume_ma > 0:
                volume_factor = self.executed_volume_history[-1] / self.volume_ma
            else:
                volume_factor = 1.0
        else:
            volume_factor = 1.0

        # 更新市场情绪
        # 价格上涨且交易量增加，情绪变得更积极
        sentiment_change = 0

        # 价格因素
        if price_change > 0:
            sentiment_change += 0.05 * price_change  # 价格上涨提升情绪
        else:
            sentiment_change += 0.03 * price_change  # 价格下跌降低情绪，但影响较小

        # 交易量因素
        if volume_factor > 1.0:
            # 交易量增加时，影响取决于价格变化方向
            if price_change > 0:
                # 量价齐升，大幅提升情绪
                sentiment_change += 0.05 * (volume_factor - 1) * (1 + price_change * 10)
            else:
                # 量增价跌，轻微降低情绪
                sentiment_change -= 0.02 * (volume_factor - 1)
        else:
            # 交易量减少时，轻微降低情绪
            sentiment_change -= 0.01 * (1 - volume_factor)

        # 情绪惯性因素，使情绪缓慢回归到中性状态
        sentiment_change -= 0.01 * (self.market_sentiment - 0.5)

        # 更新市场情绪，确保在0-1范围内
        self.market_sentiment = max(0.0, min(1.0, self.market_sentiment + sentiment_change))
        self.sentiment_history.append(self.market_sentiment)

    def place_order(self, order_type, price, shares, investor):
        if order_type == 'buy':
            self.buy_orders.append((price, shares, investor))
        elif order_type == 'sell':
            self.sell_orders.append((price, shares, investor))

    def call_auction(self, buy_orders, sell_orders, last_price):
        # 检查订单列表是否为空
        if not buy_orders or not sell_orders:
            return last_price, 0, set(), set()
        buy_orders_sorted = sorted(enumerate(buy_orders), key=lambda x: x[1][0], reverse=True)
        sell_orders_sorted = sorted(enumerate(sell_orders), key=lambda x: x[1][0])
        possible_prices = sorted(set([order[0] for order in buy_orders + sell_orders]))
        if not possible_prices:
            return last_price, 0, set(), set()
        max_volume = 0
        clearing_price = last_price
        for test_price in possible_prices:
            buy_volume = sum(shares for price, shares, _ in buy_orders if price >= test_price)
            sell_volume = sum(shares for price, shares, _ in sell_orders if price <= test_price)
            executed = min(buy_volume, sell_volume)
            if executed > max_volume:
                max_volume = executed
                clearing_price = test_price
            elif executed == max_volume and abs(test_price - last_price) < abs(clearing_price - last_price):
                clearing_price = test_price
        executed_buy_idx = set()
        executed_sell_idx = set()
        remain_buy = max_volume
        for idx, (price, shares, investor) in buy_orders_sorted:
            if price >= clearing_price and remain_buy > 0:
                exec_shares = min(shares, remain_buy)
                remain_buy -= exec_shares
                executed_buy_idx.add(idx)
        remain_sell = max_volume
        for idx, (price, shares, investor) in sell_orders_sorted:
            if price <= clearing_price and remain_sell > 0:
                exec_shares = min(shares, remain_sell)
                remain_sell -= exec_shares
                executed_sell_idx.add(idx)
        return clearing_price, max_volume, executed_buy_idx, executed_sell_idx

    def execute_trades(self, clearing_price, max_volume, buy_orders, sell_orders, executed_buy_idx, executed_sell_idx):
        # 确保所有参数有效
        if max_volume <= 0 or not buy_orders or not sell_orders or not executed_buy_idx or not executed_sell_idx:
            return

        # 执行买单
        remain = max_volume
        buy_orders_sorted = sorted(enumerate(buy_orders), key=lambda x: x[1][0], reverse=True)
        if buy_orders_sorted:  # 确保有买单
            for idx, (price, shares, investor) in buy_orders_sorted:
                if idx in executed_buy_idx and remain > 0:
                    exec_shares = min(shares, remain)
                    remain -= exec_shares

                    # 计算交易金额和买入手续费
                    trade_amount = exec_shares * clearing_price
                    buy_fee = trade_amount * self.buy_fee_rate
                    total_cost = trade_amount + buy_fee

                    # 检查投资者是否有足够的现金
                    if investor.cash >= total_cost:
                        # 更新投资者持仓和现金（扣除手续费）
                        investor.shares += exec_shares
                        investor.cash -= total_cost

                        # 累计手续费收入
                        self.fee_income += buy_fee
                    else:
                        # 如果现金不足，调整购买数量
                        affordable_shares = int((investor.cash / (clearing_price * (1 + self.buy_fee_rate))))
                        if affordable_shares > 0:
                            # 重新计算交易金额和手续费
                            trade_amount = affordable_shares * clearing_price
                            buy_fee = trade_amount * self.buy_fee_rate
                            total_cost = trade_amount + buy_fee

                            # 更新投资者持仓和现金
                            investor.shares += affordable_shares
                            investor.cash -= total_cost

                            # 累计手续费收入
                            self.fee_income += buy_fee

                    # 手续费收入已在上面累计

                    # 不再在这里更新价值投资者的估值，而是让他们自己在decide_price中根据延迟时间更新

        # 执行卖单
        remain = max_volume
        sell_orders_sorted = sorted(enumerate(sell_orders), key=lambda x: x[1][0])
        if sell_orders_sorted:  # 确保有卖单
            for idx, (price, shares, investor) in sell_orders_sorted:
                if idx in executed_sell_idx and remain > 0:
                    exec_shares = min(shares, remain)
                    remain -= exec_shares

                    # 计算交易金额和卖出手续费
                    trade_amount = exec_shares * clearing_price
                    sell_fee = trade_amount * self.sell_fee_rate

                    # 检查投资者是否有足够的股票
                    if investor.shares >= exec_shares:
                        # 更新投资者持仓和现金（扣除手续费）
                        investor.shares -= exec_shares
                        investor.cash += (trade_amount - sell_fee)

                        # 累计手续费收入
                        self.fee_income += sell_fee
                    else:
                        # 如果股票不足，只卖出实际持有的股票
                        available_shares = investor.shares
                        if available_shares > 0:
                            # 重新计算交易金额和手续费
                            trade_amount = available_shares * clearing_price
                            sell_fee = trade_amount * self.sell_fee_rate

                            # 更新投资者持仓和现金
                            investor.shares = 0
                            investor.cash += (trade_amount - sell_fee)

                            # 累计手续费收入
                            self.fee_income += sell_fee

                    # 不再在这里更新价值投资者的估值，而是让他们自己在decide_price中根据延迟时间更新

    def inject_capital(self, investors, amount=None, percentage=None, day=None):
        """
        为投资者注入资金
        Inject capital to investors

        Args:
            investors (list): 投资者列表 (list of investors)
            amount (float, optional): 注入的固定金额 (fixed amount to inject)
            percentage (float, optional): 注入的资金比例 (percentage of current cash to inject)
            day (int, optional): 注入资金的日期，默认为当前日期 (day of injection, default is current day)

        注意：必须提供amount或percentage其中之一，但不能同时提供两者
        Note: Either amount or percentage must be provided, but not both
        """
        if amount is None and percentage is None:
            raise ValueError("必须提供固定金额(amount)或比例(percentage)参数之一")
        if amount is not None and percentage is not None:
            raise ValueError("不能同时提供固定金额(amount)和比例(percentage)参数")

        # 确定当前日期
        current_day = day if day is not None else len(self.price_history) - 1

        # 为每个投资者注入资金
        for investor in investors:
            if amount is not None:
                # 注入固定金额
                investor.cash += amount
                injection_amount = amount
            else:
                # 按比例注入资金
                injection_amount = investor.cash * percentage
                investor.cash += injection_amount

        # 记录资金注入历史
        if amount is not None:
            self.capital_change_history.append((current_day, "inject", amount, None))
        else:
            self.capital_change_history.append((current_day, "inject", None, percentage))

    def withdraw_capital(self, investors, amount=None, percentage=None, day=None):
        """
        从投资者抽离资金
        Withdraw capital from investors

        Args:
            investors (list): 投资者列表 (list of investors)
            amount (float, optional): 抽离的固定金额 (fixed amount to withdraw)
            percentage (float, optional): 抽离的资金比例 (percentage of current cash to withdraw)
            day (int, optional): 抽离资金的日期，默认为当前日期 (day of withdrawal, default is current day)

        注意：必须提供amount或percentage其中之一，但不能同时提供两者
        Note: Either amount or percentage must be provided, but not both
        """
        if amount is None and percentage is None:
            raise ValueError("必须提供固定金额(amount)或比例(percentage)参数之一")
        if amount is not None and percentage is not None:
            raise ValueError("不能同时提供固定金额(amount)和比例(percentage)参数")

        # 确定当前日期
        current_day = day if day is not None else len(self.price_history) - 1

        # 从每个投资者抽离资金
        for investor in investors:
            if amount is not None:
                # 抽离固定金额，但不能超过投资者持有的现金
                withdrawal_amount = min(amount, investor.cash)
                investor.cash -= withdrawal_amount
            else:
                # 按比例抽离资金
                withdrawal_amount = investor.cash * percentage
                investor.cash -= withdrawal_amount

        # 记录资金抽离历史
        if amount is not None:
            self.capital_change_history.append((current_day, "withdraw", amount, None))
        else:
            self.capital_change_history.append((current_day, "withdraw", None, percentage))

    def daily_auction(self):
        # 更新市场情绪
        self.update_market_sentiment()

        # 初始化当日价格数据
        current_day = len(self.price_history) - 1
        last_price = self.price_history[-1]
        open_price = last_price  # 默认开盘价为前一天收盘价
        high_price = last_price  # 初始化最高价
        low_price = last_price   # 初始化最低价
        close_price = last_price # 初始化收盘价
        daily_volume = 0         # 初始化当日成交量

        # 临时存储开盘前的买卖订单
        opening_buy_orders = []
        opening_sell_orders = []

        # 临时存储收盘前的买卖订单
        closing_buy_orders = []
        closing_sell_orders = []

        # 1. 开盘前随机分配一部分订单用于开盘集合竞价
        # 随机选择30%-50%的订单参与开盘集合竞价
        opening_ratio = self._rng.uniform(0.3, 0.5)

        # 随机选择买单参与开盘集合竞价
        if self.buy_orders:
            num_opening_buy = max(1, int(len(self.buy_orders) * opening_ratio))
            opening_indices = self._rng.choice(len(self.buy_orders), num_opening_buy, replace=False)
            for i in opening_indices:
                opening_buy_orders.append(self.buy_orders[i])
            # 从原始订单列表中移除这些订单
            self.buy_orders = [order for i, order in enumerate(self.buy_orders) if i not in opening_indices]

        # 随机选择卖单参与开盘集合竞价
        if self.sell_orders:
            num_opening_sell = max(1, int(len(self.sell_orders) * opening_ratio))
            opening_indices = self._rng.choice(len(self.sell_orders), num_opening_sell, replace=False)
            for i in opening_indices:
                opening_sell_orders.append(self.sell_orders[i])
            # 从原始订单列表中移除这些订单
            self.sell_orders = [order for i, order in enumerate(self.sell_orders) if i not in opening_indices]

        # 2. 执行开盘集合竞价
        if opening_buy_orders and opening_sell_orders:
            open_price, open_volume, executed_buy_idx, executed_sell_idx = self.call_auction(
                opening_buy_orders, opening_sell_orders, last_price)

            # 执行开盘集合竞价交易
            self.execute_trades(open_price, open_volume, opening_buy_orders, opening_sell_orders,
                               executed_buy_idx, executed_sell_idx)
            daily_volume += open_volume

            # 更新最高最低价
            high_price = max(high_price, open_price)
            low_price = min(low_price, open_price)

            # 清空已执行的订单
            opening_buy_orders = [order for i, order in enumerate(opening_buy_orders) if i not in executed_buy_idx]
            opening_sell_orders = [order for i, order in enumerate(opening_sell_orders) if i not in executed_sell_idx]

            # 将未成交的开盘订单添加回主订单列表
            self.buy_orders.extend(opening_buy_orders)
            self.sell_orders.extend(opening_sell_orders)

        # 3. 盘中交易 - 模拟订单陆续到达并撮合
        max_iterations = 100  # 设置最大迭代次数，防止无限循环
        iteration = 0

        while (self.buy_orders or self.sell_orders) and iteration < max_iterations:
            # 随机决定是否进行本轮撮合
            if self._rng.random() < 0.8:  # 80%的概率进行撮合
                # 临时存储本轮参与撮合的订单
                current_buy_orders = []
                current_sell_orders = []

                # 随机选择一部分买单参与本轮撮合
                if self.buy_orders:
                    num_current_buy = max(1, int(len(self.buy_orders) * self._rng.uniform(0.1, 0.3)))
                    current_indices = self._rng.choice(len(self.buy_orders), min(num_current_buy, len(self.buy_orders)), replace=False)
                    for i in current_indices:
                        current_buy_orders.append(self.buy_orders[i])
                    # 从原始订单列表中移除这些订单
                    self.buy_orders = [order for i, order in enumerate(self.buy_orders) if i not in current_indices]

                # 随机选择一部分卖单参与本轮撮合
                if self.sell_orders:
                    num_current_sell = max(1, int(len(self.sell_orders) * self._rng.uniform(0.1, 0.3)))
                    current_indices = self._rng.choice(len(self.sell_orders), min(num_current_sell, len(self.sell_orders)), replace=False)
                    for i in current_indices:
                        current_sell_orders.append(self.sell_orders[i])
                    # 从原始订单列表中移除这些订单
                    self.sell_orders = [order for i, order in enumerate(self.sell_orders) if i not in current_indices]

                # 如果本轮有买单和卖单，尝试撮合
                if current_buy_orders and current_sell_orders:
                    match_price, match_volume, executed_buy_idx, executed_sell_idx = self.call_auction(
                        current_buy_orders, current_sell_orders, open_price if iteration == 0 else self.price)

                    # 如果有成交
                    if match_volume > 0:
                        # 执行交易
                        self.execute_trades(match_price, match_volume, current_buy_orders, current_sell_orders,
                                          executed_buy_idx, executed_sell_idx)
                        daily_volume += match_volume

                        # 更新最高最低价
                        high_price = max(high_price, match_price)
                        low_price = min(low_price, match_price)

                        # 更新当前价格
                        self.price = match_price

                    # 将未成交的订单添加回主订单列表
                    current_buy_orders = [order for i, order in enumerate(current_buy_orders) if i not in executed_buy_idx]
                    current_sell_orders = [order for i, order in enumerate(current_sell_orders) if i not in executed_sell_idx]

                    # 随机决定未成交订单的去向
                    # 一部分返回主订单列表，一部分进入收盘集合竞价，一部分取消
                    for order in current_buy_orders:
                        r = self._rng.random()
                        if r < 0.4:  # 40%概率返回主订单列表
                            self.buy_orders.append(order)
                        elif r < 0.7:  # 30%概率进入收盘集合竞价
                            closing_buy_orders.append(order)
                        # 剩余30%概率取消订单

                    for order in current_sell_orders:
                        r = self._rng.random()
                        if r < 0.4:  # 40%概率返回主订单列表
                            self.sell_orders.append(order)
                        elif r < 0.7:  # 30%概率进入收盘集合竞价
                            closing_sell_orders.append(order)
                        # 剩余30%概率取消订单

            # 随机决定是否将主订单列表中的一部分订单移至收盘集合竞价
            if iteration > max_iterations * 0.7 and (self.buy_orders or self.sell_orders):  # 在迭代后期
                # 随机选择一部分买单移至收盘集合竞价
                if self.buy_orders:
                    num_closing_buy = max(1, int(len(self.buy_orders) * self._rng.uniform(0.1, 0.2)))
                    closing_indices = self._rng.choice(len(self.buy_orders), min(num_closing_buy, len(self.buy_orders)), replace=False)
                    for i in closing_indices:
                        closing_buy_orders.append(self.buy_orders[i])
                    # 从原始订单列表中移除这些订单
                    self.buy_orders = [order for i, order in enumerate(self.buy_orders) if i not in closing_indices]

                # 随机选择一部分卖单移至收盘集合竞价
                if self.sell_orders:
                    num_closing_sell = max(1, int(len(self.sell_orders) * self._rng.uniform(0.1, 0.2)))
                    closing_indices = self._rng.choice(len(self.sell_orders), min(num_closing_sell, len(self.sell_orders)), replace=False)
                    for i in closing_indices:
                        closing_sell_orders.append(self.sell_orders[i])
                    # 从原始订单列表中移除这些订单
                    self.sell_orders = [order for i, order in enumerate(self.sell_orders) if i not in closing_indices]

            iteration += 1

        # 4. 将剩余的主订单列表中的订单添加到收盘集合竞价
        closing_buy_orders.extend(self.buy_orders)
        closing_sell_orders.extend(self.sell_orders)
        self.buy_orders = []
        self.sell_orders = []

        # 5. 执行收盘集合竞价
        # if closing_buy_orders and closing_sell_orders:
        close_price, close_volume, executed_buy_idx, executed_sell_idx = self.call_auction(
            closing_buy_orders, closing_sell_orders, self.price)

        # 执行收盘集合竞价交易
        self.execute_trades(close_price, close_volume, closing_buy_orders, closing_sell_orders,
                           executed_buy_idx, executed_sell_idx)
        daily_volume += close_volume

        # 更新最高最低价
        high_price = max(high_price, close_price)
        low_price = min(low_price, close_price)
        # else:
        #     # 如果没有收盘集合竞价，使用随机生成的收盘价
        #     if high_price > low_price:
        #         # 收盘价在最高价和最低价之间随机生成，但偏向于开盘价和最后一次交易价格的加权平均
        #         weight = self._rng_close.uniform(0.3, 0.7)  # 使用收盘价专用随机数生成器生成随机权重
        #         base_price = weight * open_price + (1 - weight) * self.price
        #         # 在基准价格附近随机波动，但确保在最高价和最低价之间
        #         close_price = max(low_price, min(high_price,
        #                                        base_price + self._rng_close.uniform(-0.5, 0.5) * (high_price - low_price)))
        #     else:
        #         close_price = high_price  # 如果最高价等于最低价，则收盘价等于最高价

        # 更新市场状态
        self.price = close_price
        self.price_history.append(close_price)
        self.executed_volume = daily_volume
        self.executed_volume_history.append(daily_volume)

        # 记录当日K线数据（开盘价、最高价、最低价、收盘价）
        if not hasattr(self, 'ohlc_data'):
            self.ohlc_data = []
        self.ohlc_data.append((open_price, high_price, low_price, close_price))

        # 清空剩余订单
        self.buy_orders = []
        self.sell_orders = []

        # 更新真实价值
        if self.value_curve is not None and current_day < len(self.value_curve):
            self.true_value = self.value_curve[current_day]
        self.value_history.append(self.true_value)

    def get_future_value_change(self, prediction_days):
        """获取未来价值变化"""
        current_day = len(self.price_history) - 1
        if self.value_curve is not None:
            future_idx = current_day + prediction_days
            if future_idx < len(self.value_curve):
                change = self.value_curve[future_idx] - self.value_curve[current_day]
                return change
        return None

def generate_random_seeds(base_seed=None):
    """生成随机数种子字典

    参数:
        base_seed: 基础种子，如果为None，则生成一个随机的基础种子

    返回:
        包含所有随机数种子的字典
    """
    if base_seed is None:
        base_seed = np.random.randint(0, 1000000)

    random_seeds = {
        'base_seed': base_seed,
        'value_line_seed': base_seed,
        'close_seed': base_seed + 1,
        'market_seed': base_seed + 2,
        'value_investor_seed': base_seed + 3,
        'chase_investor_seed': base_seed + 4,
        'trend_investor_seed': base_seed + 5,
        'random_investor_seed': base_seed + 6,
        'never_stop_loss_investor_seed': base_seed + 7,
        'bottom_fishing_investor_seed': base_seed + 8,
        'insider_investor_seed': base_seed + 9,
        'message_investor_seed': base_seed + 10
    }

    return random_seeds

def simulate_stock_market(random_seeds=None, enable_trade_log=True):
    # 如果没有提供随机数种子字典，创建一个新的
    if random_seeds is None:
        random_seeds = generate_random_seeds()

    # 基础市场参数设置
    initial_price = 100
    price_tick = 0.01
    days = 1000
    buy_fee_rate = 0.001
    sell_fee_rate = 0.001

    # 设置是否启用交易日志记录
    Investor.set_enable_trade_log(enable_trade_log)

    # 初始化所有投资者的全局日志文件
    Investor.init_trade_log_file("all_investors_trade_log.csv")

    # 不再初始化NeverStopLossInvestor的特定日志文件

    # 创建价值曲线
    value_curve = TrueValueCurve(initial_value=initial_price, days=days, seed=random_seeds['value_line_seed'])

    # 创建市场实例，传入价值曲线
    market = Market(initial_price, price_tick, value_curve=value_curve,
                   seed=random_seeds['market_seed'],
                   close_seed=random_seeds['close_seed'],
                   buy_fee_rate=buy_fee_rate, sell_fee_rate=sell_fee_rate)

    # 不同类型投资者的参数设置
    value_investors_params = {
        'num': 100,  # 价值投资者人数较多
        'initial_shares': 100,
        'initial_cash': 10000
    }

    chase_investors_params = {
        'num': 100,  # 追涨杀跌投资者数量适中
        'initial_shares': 100,
        'initial_cash': 10000
    }

    trend_investors_params = {
        'num': 100,  # 趋势投资者相对较少
        'initial_shares': 100,
        'initial_cash': 10000
    }

    random_investors_params = {
        'num': 100,  # 随机投资者数量较多
        'initial_shares': 100,
        'initial_cash': 10000
    }

    never_stop_loss_investors_params = {
        'num': 100,  # 永不止损投资者数量适中
        'initial_shares': 0,  # 初始无持股，从0开始
        'initial_cash': 20000  # 增加初始现金
    }

    bottom_fishing_investors_params = {
        'num': 100,  # 抄底投资者数量较少
        'initial_shares': 100,
        'initial_cash': 10000
    }

    insider_investors_params = {
        'num': 10,  # 内幕交易者数量很少
        'initial_shares': 100,
        'initial_cash': 10000
    }

    message_investors_params = {
        'num': 10,  # 消息投资者数量较少但比内幕交易者多
        'initial_shares': 100,
        'initial_cash': 10000
    }

    investors = []
    trend_periods = [5, 10, 15, 20, 25, 30, 60, 100, 150, 200]
    trend_investors_by_period = {}

    # 创建价值投资者，预先生成所有投资者的估值偏差百分比
    # 使用正态分布生成固定的估值偏差百分比，标准差为15%
    bias_percent_std = 0.15  # 偏差百分比的标准差为15%
    investor_bias_percents = market._rng.normal(0, bias_percent_std, value_investors_params['num'])

    # 创建价值投资者，每个投资者使用预生成的固定偏差百分比和随机的重新估值延迟
    for i, bias_percent in enumerate(investor_bias_percents):
        # 随机生成最大偏离比例(0.2-0.4之间)
        max_deviation = market._rng.uniform(0.2, 0.4)
        # 随机生成目标持仓比例(0.4-0.6之间)
        target_position = market._rng.uniform(0.4, 0.6)

        investors.append(ValueInvestor(
            value_investors_params['initial_shares'],
            value_investors_params['initial_cash'],
            bias_percent=bias_percent,  # 每个投资者有固定的估值偏差百分比
            max_deviation=max_deviation,  # 最大偏离比例
            target_position=target_position,  # 目标持仓比例
            seed=random_seeds['value_investor_seed'] + i  # 使用不同的种子确保每个投资者有不同的重新估值延迟
        ))

    # 创建追涨杀跌投资者
    for i in range(chase_investors_params['num']):
        investors.append(ChaseInvestor(
            chase_investors_params['initial_shares'],
            chase_investors_params['initial_cash'],
            seed=random_seeds['chase_investor_seed'] + i  # 使用不同的种子确保每个投资者有不同的随机行为
        ))

    # 创建趋势投资者
    trend_investors_num_per_period = trend_investors_params['num'] // len(trend_periods)
    for period_idx, period in enumerate(trend_periods):
        trend_investors_by_period[period] = []
        for i in range(trend_investors_num_per_period):
            investor = TrendInvestor(
                trend_investors_params['initial_shares'],
                trend_investors_params['initial_cash'],
                period,
                seed=random_seeds['trend_investor_seed'] + period_idx * 100 + i  # 使用不同的种子确保每个投资者有不同的随机行为
            )
            investors.append(investor)
            trend_investors_by_period[period].append(investor)

    # 创建随机投资者
    for i in range(random_investors_params['num']):
        investors.append(RandomInvestor(
            random_investors_params['initial_shares'],
            random_investors_params['initial_cash'],
            seed=random_seeds['random_investor_seed'] + i  # 使用不同的种子确保每个投资者有不同的随机行为
        ))

    # 创建永不止损投资者
    for i in range(never_stop_loss_investors_params['num']):
        investor = NeverStopLossInvestor(
            never_stop_loss_investors_params['initial_shares'],
            never_stop_loss_investors_params['initial_cash'],
            buy_probability=0.3,  # 30%的概率全仓买入，增加买入概率
            initial_profit_target=0.15,  # 15%的初始目标盈利比例
            min_profit_target=0.02,  # 2%的最小目标盈利比例
            seed=random_seeds['never_stop_loss_investor_seed'] + i  # 使用不同的种子确保每个投资者有不同的随机行为
        )
        investors.append(investor)

    # 创建抄底投资者
    for i in range(bottom_fishing_investors_params['num']):
        # 使用随机数生成器创建盈利目标
        seed = random_seeds['bottom_fishing_investor_seed'] + i
        rng = np.random.RandomState(seed)
        profit_target = rng.uniform(0.1, 0.5)
        investors.append(BottomFishingInvestor(
            bottom_fishing_investors_params['initial_shares'],
            bottom_fishing_investors_params['initial_cash'],
            profit_target=profit_target,
            seed=seed  # 使用不同的种子确保每个投资者有不同的随机行为
        ))

    # 创建内幕交易者
    for i in range(insider_investors_params['num']):
        investors.append(InsiderInvestor(
            insider_investors_params['initial_shares'],
            insider_investors_params['initial_cash'],
            seed=random_seeds['insider_investor_seed'] + i  # 使用不同的种子确保每个投资者有不同的随机行为
        ))

    # 创建消息投资者
    for i in range(message_investors_params['num']):
        investors.append(MessageInvestor(
            message_investors_params['initial_shares'],
            message_investors_params['initial_cash'],
            seed=random_seeds['message_investor_seed'] + i  # 使用不同的种子确保每个投资者有不同的随机行为
        ))

    # 计算每种类型投资者的起始和结束索引
    value_end = value_investors_params['num']
    chase_end = value_end + chase_investors_params['num']
    trend_end = chase_end + trend_investors_params['num']
    random_end = trend_end + random_investors_params['num']
    never_stop_loss_end = random_end + never_stop_loss_investors_params['num']
    bottom_fishing_end = never_stop_loss_end + bottom_fishing_investors_params['num']
    insider_end = bottom_fishing_end + insider_investors_params['num']
    message_end = insider_end + message_investors_params['num']

    shares_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': [], 'NeverStopLoss': [], 'BottomFishing': [], 'Insider': [], 'Message': []}
    cash_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': [], 'NeverStopLoss': [], 'BottomFishing': [], 'Insider': [], 'Message': []}
    wealth_by_type = {'Value': [], 'Chase': [], 'Trend': [], 'Random': [], 'NeverStopLoss': [], 'BottomFishing': [], 'Insider': [], 'Message': []}
    trend_assets_by_period = {period: [] for period in trend_periods}

    # 定义资金注入和抽离的日期和参数
    # Define dates and parameters for capital injection and withdrawal
    # capital_events = [
    #     {'day': 500, 'type': 'inject', 'amount': 10000, 'investors_range': (0, len(investors))},  # 第1000天为所有投资者注入5000资金
    #     {'day': 800, 'type': 'inject', 'percentage': 0.8, 'investors_range': (0, value_end)},  # 第2000天为价值投资者注入20%资金
    #     {'day': 1000, 'type': 'withdraw', 'percentage': 0.5, 'investors_range': (0, len(investors))},  # 第3000天从所有投资者抽离10%资金
    #     {'day': 1500, 'type': 'withdraw', 'amount': 2000, 'investors_range': (random_end, never_stop_loss_end)}  # 第4000天从永不止损投资者抽离2000资金
    # ]

    # capital_events = [
    #     {'day': 400, 'type': 'inject', 'percentage': 2, 'investors_range': (0, len(investors))},  # 第2000天为价值投资者注入20%资金
    # ]

    capital_events = []

    for day in range(days):
        # 检查是否有资金注入或抽离事件
        for event in capital_events:
            if day == event['day']:
                start, end = event['investors_range']
                affected_investors = investors[start:end]
                if event['type'] == 'inject':
                    if 'amount' in event:
                        market.inject_capital(affected_investors, amount=event['amount'], day=day)
                    else:
                        market.inject_capital(affected_investors, percentage=event['percentage'], day=day)
                else:  # withdraw
                    if 'amount' in event:
                        market.withdraw_capital(affected_investors, amount=event['amount'], day=day)
                    else:
                        market.withdraw_capital(affected_investors, percentage=event['percentage'], day=day)

        for investor in investors:
            investor.trade(market.price, market)
        # 执行每日集合竞价 (Execute daily call auction)
        market.daily_auction()
        # 不需要再次添加价格，因为daily_auction已经添加了
        # prices.append(market.price)

        # 记录每种类型投资者的平均持仓、现金和总财富
        # Record average holdings, cash and total wealth for each type of investor
        type_ranges = [
            ('Value', 0, value_end),
            ('Chase', value_end, chase_end),
            ('Trend', chase_end, trend_end),
            ('Random', trend_end, random_end),
            ('NeverStopLoss', random_end, never_stop_loss_end),
            ('BottomFishing', never_stop_loss_end, bottom_fishing_end),
            ('Insider', bottom_fishing_end, insider_end),
            ('Message', insider_end, message_end)
        ]

        for type_name, start, end in type_ranges:
            # 确保有投资者数据才计算平均值
            if start < end:
                shares_list = [inv.shares for inv in investors[start:end]]
                cash_list = [inv.cash for inv in investors[start:end]]
                if shares_list and cash_list:  # 确保列表非空
                    avg_shares = np.mean(shares_list)  # 计算平均持股数 / Calculate average shares held
                    avg_cash = np.mean(cash_list)      # 计算平均现金 / Calculate average cash
                    avg_wealth = avg_cash + avg_shares * market.price  # 计算平均总财富 / Calculate average total wealth
                else:
                    avg_shares = avg_cash = avg_wealth = 0.0
            else:
                avg_shares = avg_cash = avg_wealth = 0.0
            shares_by_type[type_name].append(avg_shares)
            cash_by_type[type_name].append(avg_cash)
            wealth_by_type[type_name].append(avg_wealth)

        # 记录不同周期趋势投资者的总资产
        # Record total assets for trend investors with different periods
        for period in trend_periods:
            investors_list = trend_investors_by_period[period]
            if investors_list:  # 确保有投资者数据
                assets_list = [inv.cash + inv.shares * market.price for inv in investors_list]
                if assets_list:  # 确保列表非空
                    avg_assets = np.mean(assets_list)
                else:
                    avg_assets = 0.0
            else:
                avg_assets = 0.0
            trend_assets_by_period[period].append(avg_assets)

    # 创建图表展示模拟结果
    # Create plots to display simulation results
    # 创建可视化图表 - 包含5个子图 / Create visualization - 5 subplots
    fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)  # 共享x轴 / Share x-axis

    # 绘制K线图和真实价值(第一个子图) / Plot candlestick chart and true value (first subplot)
    # 准备K线图数据
    if hasattr(market, 'ohlc_data') and market.ohlc_data:
        # 提取OHLC数据
        opens = [data[0] for data in market.ohlc_data]
        highs = [data[1] for data in market.ohlc_data]
        lows = [data[2] for data in market.ohlc_data]
        closes = [data[3] for data in market.ohlc_data]

        # 绘制K线图
        # 计算K线宽度
        width = 0.6
        # 创建x轴位置
        x = np.arange(len(opens))

        # 绘制影线(最高价到最低价的线)
        for i in range(len(opens)):
            # 根据开盘价和收盘价的关系确定K线颜色
            if closes[i] >= opens[i]:  # 收盘价高于开盘价，为阳线(红色)
                color = 'red'
                # 绘制实体(开盘价到收盘价的矩形)
                axs[0].add_patch(plt.Rectangle((i-width/2, opens[i]), width, closes[i]-opens[i],
                                              fill=True, color=color, alpha=0.6))
            else:  # 收盘价低于开盘价，为阴线(绿色)
                color = 'green'
                # 绘制实体(开盘价到收盘价的矩形)
                axs[0].add_patch(plt.Rectangle((i-width/2, closes[i]), width, opens[i]-closes[i],
                                              fill=True, color=color, alpha=0.6))

            # 绘制最高价到最低价的线，使用与K线实体相同的颜色
            axs[0].plot([i, i], [lows[i], highs[i]], color=color, linewidth=1)

        # 绘制真实价值曲线
        axs[0].plot(market.value_history, label='True Value', linestyle='--', color='blue', alpha=0.7)

        axs[0].set_ylabel('Price')  # 设置y轴标签 / Set y-axis label
        axs[0].legend(loc='upper left')  # 添加图例 / Add legend
        axs[0].set_title('OHLC Candlestick Chart and True Value')  # 设置标题 / Set title

        # 在图例中添加K线说明
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
        legend_elements = [
            Line2D([0], [0], color='blue', linestyle='--', label='True Value'),
            Rectangle((0, 0), 1, 1, color='red', alpha=0.6, label='Bullish Candle'),
            Rectangle((0, 0), 1, 1, color='green', alpha=0.6, label='Bearish Candle')
        ]
        axs[0].legend(handles=legend_elements, loc='upper left')
    else:
        # 如果没有OHLC数据，则绘制普通价格曲线
        axs[0].plot(market.price_history, label='Stock Price', color='blue')  # 股票价格 / Stock price
        axs[0].plot(market.value_history, label='True Value', linestyle='--', color='green')  # 真实价值 / True value
        axs[0].set_ylabel('Price', color='blue')  # 设置y轴标签 / Set y-axis label
        axs[0].tick_params(axis='y', labelcolor='blue')  # 设置刻度颜色 / Set tick color
        axs[0].legend(loc='upper left')  # 添加图例 / Add legend

    # 在第一个子图上添加成交量(第二个y轴) / Add trading volume (second y-axis) to first subplot
    ax2 = axs[0].twinx()  # 创建共享x轴的双y轴 / Create twin y-axis sharing x-axis
    # 确保成交量历史与价格历史长度一致
    # Ensure volume history length matches price history length
    daily_volumes = market.executed_volume_history
    # 确保绘制的成交量数据长度与days一致
    ax2.bar(range(len(daily_volumes)), daily_volumes, alpha=0.3, color='gray', label='Trading Volume')  # 成交量柱状图 / Volume bar chart
    ax2.set_ylabel('Volume', color='gray')  # 设置y轴标签 / Set y-axis label
    ax2.tick_params(axis='y', labelcolor='gray')  # 设置刻度颜色 / Set tick color
    ax2.legend(loc='upper right')  # 添加图例 / Add legend

    # 在第一个子图上添加资金注入和抽离事件标记 / Add capital injection and withdrawal event markers to first subplot
    if market.capital_change_history:
        for day, event_type, amount, percentage in market.capital_change_history:
            if event_type == 'inject':
                color = 'green'  # 注入资金用绿色表示 / Green for capital injection
                marker = '^'     # 上三角形表示注入 / Up triangle for injection
                label = f'Inject: {amount if amount else percentage*100:.1f}%'
            else:  # withdraw
                color = 'red'    # 抽离资金用红色表示 / Red for capital withdrawal
                marker = 'v'     # 下三角形表示抽离 / Down triangle for withdrawal
                label = f'Withdraw: {amount if amount else percentage*100:.1f}%'

            # 在图上标记事件 / Mark the event on the plot
            axs[0].axvline(x=day, color=color, alpha=0.3, linestyle='--')
            axs[0].plot(day, market.price_history[day], marker=marker, markersize=10, color=color, label=label)

    # 绘制各类投资者的平均持股(第二个子图) / Plot average shares by investor type (second subplot)
    for type_name in shares_by_type:
        axs[1].plot(shares_by_type[type_name], label=f'{type_name} Investors')  # 绘制持股曲线 / Plot shares curve
    axs[1].set_ylabel('Shares')  # 设置y轴标签 / Set y-axis label
    axs[1].set_title('Average Shares Held by Investor Type')  # 设置标题 / Set title
    axs[1].legend()  # 添加图例 / Add legend

    # 绘制各类投资者的平均现金(第三个子图) / Plot average cash by investor type (third subplot)
    for type_name in cash_by_type:
        axs[2].plot(cash_by_type[type_name], label=f'{type_name} Investors')  # 绘制现金曲线 / Plot cash curve
    axs[2].set_ylabel('Cash')  # 设置y轴标签 / Set y-axis label
    axs[2].set_title('Average Cash Held by Investor Type')  # 设置标题 / Set title
    axs[2].legend()  # 添加图例 / Add legend

    # 绘制各类投资者的平均总财富(第四个子图) / Plot average total wealth by investor type (fourth subplot)
    for type_name in wealth_by_type:
        axs[3].plot(wealth_by_type[type_name], label=f'{type_name} Investors')  # 绘制财富曲线 / Plot wealth curve
    axs[3].set_ylabel('Total Wealth')  # 设置y轴标签 / Set y-axis label
    axs[3].set_title('Average Total Wealth by Investor Type')  # 设置标题 / Set title
    axs[3].legend()  # 添加图例 / Add legend

    # 绘制不同周期趋势投资者的表现(第五个子图) / Plot trend investors' performance by period (fifth subplot)
    for period in trend_periods:
        axs[4].plot(trend_assets_by_period[period], label=f'MA{period}')  # 绘制资产曲线 / Plot assets curve
    axs[4].set_ylabel('Total Assets')  # 设置y轴标签 / Set y-axis label
    axs[4].set_title('Trend Investors Performance by MA Period')  # 设置标题 / Set title
    axs[4].legend()  # 添加图例 / Add legend

    # Set common x-label for the fifth subplot
    axs[4].set_xlabel('Day')

    # 为第一个子图添加资金变动事件的图例 / Add legend for capital change events in the first subplot
    if market.capital_change_history:
        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[0].legend(by_label.values(), by_label.keys(), loc='upper left')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # 关闭所有投资者的全局日志文件
    Investor.close_trade_log_file()

# 运行模拟
# Run simulation
if __name__ == "__main__":
    # 可以通过传入不同的随机数种子字典来生成不同的模拟结果
    # 例如，可以使用固定的基础种子生成随机数种子字典
    #random_seeds = generate_random_seeds(base_seed=2108)
    # 或者使用完全随机的种子
    # random_seeds = generate_random_seeds()
    # 或者手动指定每个种子
    random_seeds = {
        'base_seed': 2108,
        'value_line_seed': 2102,
        'close_seed': 2109,
        'market_seed': 2111,
        'value_investor_seed': 2121,
        'chase_investor_seed': 2111,
        'trend_investor_seed': 2113,
        'random_investor_seed': 2114,
        'never_stop_loss_investor_seed': 2115,
        'bottom_fishing_investor_seed': 2116,
        'insider_investor_seed': 2117,
        'message_investor_seed': 2118
    }
    # 设置enable_trade_log=True开启日志记录，False关闭日志记录
    simulate_stock_market(random_seeds=random_seeds, enable_trade_log=False)