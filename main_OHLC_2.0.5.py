
"""Stock Market Simulation System (God's Dice)

This program simulates a stock market environment with different types of investors and trading strategies.

Core Components:
1. TrueValueCurve
   - Generates underlying true value of stocks using random square wave patterns
   - Simulates fundamental value changes in the market

2. Investor Classes
   - Base Investor: Abstract base class with common trading functionalities
   - ValueInvestor: Trades based on estimated intrinsic value with individual bias
   - ChaseInvestor: Momentum-based trading using price velocity
   - TrendInvestor: Moving average based trading strategy

Key Features:
- Realistic price formation through investor interactions
- Multiple trading strategies implementation
- Detailed trade logging and analysis
- Configurable market parameters and investor behaviors

The system aims to study how different trading strategies interact and affect market dynamics,
providing insights into price formation and market behavior under various conditions.
"""

import numpy as np
import matplotlib.pyplot as plt

class TrueValueCurve:
    """
    Stock True Value Curve Generator
    - Generates specific value curves based on random seeds
    - Supports accessing value data through indexing
    - Generates value sequences with random square wave changes within specified date ranges

    Properties:
        initial_value: Initial value
        days: Number of days to generate
        seed: Random number seed
        values: Generated value sequence
    """
    def __init__(self, initial_value=100, days=2000, seed=422):
        self.initial_value = initial_value
        self.days = days
        self.seed = seed
        self.values = self._generate_curve()

    def _generate_curve(self):
        """Generate value curve using random square wave changes"""
        rng = np.random.RandomState(self.seed)
        values = [self.initial_value]
        true_value = self.initial_value
        last_jump_day = 0
        next_jump_interval = rng.randint(30, 50)

        for day in range(1, self.days+1):
            if day - last_jump_day >= next_jump_interval:
                if rng.rand() < 0.33:  # 33% probability of value jump
                    change = rng.uniform(10, 30) * (1 if rng.rand() < 0.5 else -1)
                    true_value += change
                    true_value = max(0, true_value)  # Ensure value is not negative
                last_jump_day = day
                next_jump_interval = rng.randint(30, 50)
            values.append(true_value)

        return values

    def __getitem__(self, idx):
        """Support accessing value by index"""
        return self.values[idx]

    def __len__(self):
        """Support len() function"""
        return len(self.values)

class Investor:
    """
    Base Investor class - Defines basic properties and methods for all investors

    Properties:
        shares: Number of shares held
        cash: Cash held
    """
    # Class variable for storing trade log files of all investors
    trade_log_file = None
    investor_count = 0
    # Log recording switch, enabled by default
    enable_trade_log = True

    @classmethod
    def init_trade_log_file(cls, filename="all_investors_trade_log.csv"):
        """Initialize trade log file, write header"""
        if cls.enable_trade_log:
            cls.trade_log_file = open(filename, "w")
            cls.trade_log_file.write("Day,Investor_Type,Investor_ID,Action,Price,Shares,Current_Shares,Cash,Entry_Price,Profit_Ratio,Additional_Info\n")
            cls.trade_log_file.flush()

    @classmethod
    def close_trade_log_file(cls):
        """Close trade log file"""
        if cls.trade_log_file:
            cls.trade_log_file.close()
            cls.trade_log_file = None

    @classmethod
    def set_enable_trade_log(cls, enable=True):
        """Set whether to enable trade log recording"""
        cls.enable_trade_log = enable

    def __init__(self, shares, cash):
        self.shares = shares  # shares held
        self.cash = cash      # cash held
        # Assign a unique ID to each investor
        self.investor_id = Investor.investor_count
        Investor.investor_count += 1
        # Record purchase price/entry price
        self.entry_price = None

    def get_investor_type(self):
        """Get investor type name"""
        return self.__class__.__name__

    def log_trade(self, day, action, price, shares, profit_ratio=0.0, additional_info=None):
        """Record trade to log file"""
        if Investor.enable_trade_log and Investor.trade_log_file:
            entry_price_str = str(self.entry_price) if hasattr(self, 'entry_price') and self.entry_price is not None else 'None'
            Investor.trade_log_file.write(f"{day},{self.get_investor_type()},{self.investor_id},{action},{price:.2f},{shares},{self.shares},{self.cash:.2f},{entry_price_str},{profit_ratio:.4f},{additional_info if additional_info is not None else ''}\n")
            Investor.trade_log_file.flush()

    def trade(self, price, market):
        """Trading method to be implemented by subclasses"""
        pass

    def decide_price(self, current_price, market):
        """Method to decide trading price and quantity, implemented by subclasses
        Returns: (action, price, shares)
            action: 'buy', 'sell' or 'hold'
            price: trading price
            shares: number of shares
        """
        return ('hold', 0, 0)

class ValueInvestor(Investor):
    """
    Value Investor - Trades based on estimated intrinsic value of the stock

    Strategy:
    - Each investor has a fixed valuation bias percentage (bias_percent)
    - The bias percentages of all investors follow a normal distribution
    - Valuation = True Value * (1 + bias_percent)
    - Investors do not react immediately to value changes, but have a 1-20 day lag period
    - Control buying and selling behavior through position ratio:
      - When price equals valuation, maintain target position ratio (default 50%)
      - When price is higher than valuation, reduce position ratio, empty position when deviation reaches maximum
      - When price is lower than valuation, increase position ratio, full position when deviation reaches maximum

    Properties:
        bias_percent: Fixed value estimation bias percentage
        revaluation_delay: Days of delay before revaluation
        last_valuation_day: Day of last valuation
        value_estimate: Current value estimation
        max_deviation: Maximum deviation ratio, full position or empty position at this ratio
        target_position: Target position ratio, position ratio when price equals valuation
    """
    def __init__(self, shares, cash, bias_percent, max_deviation=0.3, target_position=0.5, seed=None):
        super().__init__(shares, cash)
        self.bias_percent = bias_percent  # Fixed valuation bias percentage
        self.max_deviation = max_deviation  # Maximum deviation ratio, full position or empty position at this ratio
        self.target_position = target_position  # Target position ratio, position ratio when price equals valuation
        self._rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))
        self.revaluation_delay = self._rng.randint(1, 21)  # Randomly generate 1-20 days revaluation delay
        self.last_valuation_day = 0  # Initialize to day 0
        self.value_estimate = None  # Initial valuation is None

    def decide_price(self, current_price, market):
        """Decide trading behavior based on the difference between previous trading day's closing price and valuation"""
        current_day = len(market.price_history) - 1

        # Use previous trading day's closing price instead of current price
        # If it's the first day of trading (no previous day's closing price), use current price
        if current_day > 0:
            decision_price = market.price_history[current_day - 1]  # Previous trading day's closing price
        else:
            decision_price = current_price  # Use current price on first day

        # Initialize valuation or update valuation when revaluation time is reached
        if self.value_estimate is None or (current_day - self.last_valuation_day) >= self.revaluation_delay:
            true_value = market.value_curve[current_day]
            self.value_estimate = true_value * (1 + self.bias_percent)
            self.last_valuation_day = current_day

        # If there is no valuation yet, do not trade
        if self.value_estimate is None:
            return ('hold', 0, 0)

        # Calculate price deviation - using previous trading day's closing price
        # Prevent division by zero error
        if self.value_estimate == 0:
            # If valuation is zero, set difference to a large value, indicating price is far above valuation
            diff = 1.0 if decision_price > 0 else -1.0
        else:
            diff = (decision_price - self.value_estimate) / self.value_estimate

        # Calculate target position ratio - linear adjustment based on price deviation
        # When diff=0, position ratio is target_position
        # When diff=max_deviation, position ratio is 0 (empty position)
        # When diff=-max_deviation, position ratio is 1 (full position)
        target_ratio = max(0, min(1, self.target_position - (diff / self.max_deviation) * self.target_position))

        # Calculate current total asset value - still use current price to calculate total assets
        total_assets = self.cash + self.shares * current_price

        # Calculate target number of shares - use current price to calculate target number of shares
        target_shares = int((total_assets * target_ratio) / current_price)

        # Calculate number of shares to buy or sell
        shares_diff = target_shares - self.shares

        if shares_diff > 0:  # Need to buy
            # Check if cash is sufficient
            max_affordable = int(self.cash / current_price)
            buy_shares = min(shares_diff, max_affordable)
            if buy_shares > 0:
                buy_price = current_price * 1.01  # Buy price slightly higher than market price
                return ('buy', buy_price, buy_shares)
        elif shares_diff < 0:  # Need to sell
            sell_shares = -shares_diff  # Convert to positive number
            if sell_shares > 0:
                sell_price = current_price * 0.99  # Sell price slightly lower than market price
                return ('sell', sell_price, sell_shares)

        return ('hold', 0, 0)  # No trading signal

    def trade(self, price, market):
        """Execute specific trading operations"""
        current_day = len(market.price_history) - 1

        # Record state before trading
        old_shares = self.shares
        old_cash = self.cash

        # Execute trading decision directly, no need to update valuation
        action, order_price, shares = self.decide_price(price, market)

        # Calculate current profit/loss ratio
        profit_ratio = 0
        if self.shares > 0 and self.entry_price is not None:
            profit_ratio = (price - self.entry_price) / self.entry_price

        # Record trading decision
        value_estimate_str = f"{self.value_estimate:.2f}" if self.value_estimate is not None else "0"
        self.log_trade(current_day, f"decide_{action}", price, shares, profit_ratio,
                      f"order_price={order_price:.2f}, value_estimate={value_estimate_str}")

        if action == 'buy':
            # Record entry price when buying
            if self.shares == 0 or self.entry_price is None:  # New position or entry price is None
                self.entry_price = price
            else:  # Increase position, calculate weighted average cost
                self.entry_price = (self.entry_price * self.shares + price * shares) / (self.shares + shares)
            market.place_order(action, order_price, shares, self)
        elif action == 'sell':
            # Reset entry price when selling all positions
            if shares >= self.shares:
                self.entry_price = None
            market.place_order(action, order_price, shares, self)

        # Record state changes after trading
        if old_shares != self.shares or old_cash != self.cash:
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

class ChaseInvestor(Investor):
    """
    Chase Investor - Trades based solely on price change velocity

    Strategy:
    - Buy more when price rises faster, proportional to remaining cash
    - Sell more when price falls faster, proportional to remaining shares
    - Trading volume proportional to price change velocity
    - Not affected by market sentiment

    Properties:
        N: Observation period for calculating price change velocity
    """
    def __init__(self, shares, cash, N=None):
        super().__init__(shares, cash)
        self.N = N  # observation period
        self._n_initialized = False  # flag whether N has been initialized

    def calculate_velocity(self, prices):
        """Calculate price change velocity"""
        if len(prices) < 2:
            return 0.0
        # Calculate price change rate
        price_changes = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:  # Ensure denominator is greater than 0, avoid division by zero or negative number
                change = (prices[i] - prices[i-1]) / prices[i-1]
                price_changes.append(change)
        # Check if there is valid price change data
        if not price_changes:
            return 0.0
        # Return average change rate
        return float(sum(price_changes)) / len(price_changes)  # Ensure return value is float, and check again that the list is not empty

    def decide_price(self, current_price, market):
        """Decide trading action based on previous day's closing price velocity"""
        current_day = len(market.price_history) - 1

        # Use previous trading day's closing price instead of current price
        # If it's the first day of trading (no previous day's closing price), use current price
        if current_day > 0:
            decision_price = market.price_history[current_day - 1]  # Previous trading day's closing price
        else:
            decision_price = current_price  # Use current price on first day

        # Ensure N has been initialized
        if self.N is None and not self._n_initialized:
            self.N = market._rng.choice([3, 5, 10, 15, 20])
            self._n_initialized = True

        if len(market.price_history) >= self.N:
            recent_prices = market.price_history[-self.N:]  # Get recent price history
            velocity = self.calculate_velocity(recent_prices)  # Calculate price change velocity

            if velocity > 0:  # Price rising trend
                # Buy ratio is proportional to velocity only
                buy_ratio = min(abs(velocity) * 5, 1)  # Velocity impact increased
                # Ensure cash is sufficient
                if self.cash > 0:
                    buy_shares = int((self.cash * buy_ratio) / current_price)
                    if buy_shares > 0:
                        # Fixed buy price premium
                        buy_price = current_price * 1.02
                        return ('buy', buy_price, buy_shares)
            elif velocity < 0:  # Price falling trend
                # Sell ratio is proportional to velocity only
                sell_ratio = min(abs(velocity) * 5, 1)  # Velocity impact increased
                # Ensure shares are sufficient
                if self.shares > 0:
                    sell_shares = int(self.shares * sell_ratio)
                    if sell_shares > 0:
                        # Fixed sell price discount
                        sell_price = current_price * 0.98
                        return ('sell', sell_price, sell_shares)

            # Consider trading volume feedback
            if len(market.executed_volume_history) > 5:
                volume_change = market.executed_volume_history[-1] / max(1, np.mean(market.executed_volume_history[-6:-1]))
                if volume_change > 1.2 and velocity > 0:  # Trading volume increases and price rises
                    # Volume and price rise together, enhance chasing strength, but only based on velocity and volume
                    buy_ratio = min(1.0, abs(velocity) * 5 * min(2.0, volume_change))
                    # Ensure cash is sufficient
                    if self.cash > 0:
                        buy_shares = int((self.cash * buy_ratio) / current_price)
                        if buy_shares > 0:
                            buy_price = current_price * 1.02  # Fixed buy price premium
                            return ('buy', buy_price, buy_shares)

        return ('hold', 0, 0)  # No trading signal

    def trade(self, price, market):
        """Execute specific trading operation"""
        current_day = len(market.price_history) - 1

        # Record state before trading
        old_shares = self.shares
        old_cash = self.cash

        action, order_price, shares = self.decide_price(price, market)

        # Calculate current profit/loss ratio
        profit_ratio = 0
        if self.shares > 0 and self.entry_price is not None:
            profit_ratio = (price - self.entry_price) / self.entry_price

        # Record trading decision
        velocity = 0
        if len(market.price_history) >= self.N:
            recent_prices = market.price_history[-self.N:]
            velocity = self.calculate_velocity(recent_prices)
        self.log_trade(current_day, f"decide_{action}", price, shares, profit_ratio,
                      f"order_price={order_price:.2f}, velocity={velocity:.4f}, N={self.N}")

        if action == 'buy':
            # Record entry price when buying
            if self.shares == 0 or self.entry_price is None:  # New position or entry price is None
                self.entry_price = price
            else:  # Increase position, calculate weighted average cost
                self.entry_price = (self.entry_price * self.shares + price * shares) / (self.shares + shares)
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            # Reset entry price when selling all positions
            if shares >= self.shares:
                self.entry_price = None
            market.place_order('sell', order_price, shares, self)

        # Record state changes after trading
        if old_shares != self.shares or old_cash != self.cash:
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

class TrendInvestor(Investor):
    """
    Trend Investor - Trades based on moving average

    Strategy:
    - Buy all when price first crosses above moving average
    - Sell all when price first crosses below moving average

    Properties:
        M: Moving average period
        above_ma: Record if price is above MA
    """
    def __init__(self, shares, cash, M):
        super().__init__(shares, cash)
        self.M = M  # Moving average period
        self.above_ma = None  # Record if price is above MA

    def decide_price(self, current_price, market):
        """Decide trading action based on previous day's closing price's relationship with moving average"""
        current_day = len(market.price_history) - 1

        # Use previous trading day's closing price instead of current price
        # If it's the first day of trading (no previous day's closing price), use current price
        if current_day > 0:
            decision_price = market.price_history[current_day - 1]  # Previous trading day's closing price
        else:
            decision_price = current_price  # Use current price on first day

        if len(market.price_history) >= self.M:
            # Calculate simple moving average
            recent_prices = market.price_history[-self.M:]
            if not recent_prices or len(recent_prices) < self.M:  # Check if there is enough price data
                return ('hold', 0, 0)
            sma = float(sum(recent_prices)) / len(recent_prices)  # Already checked list is not empty, use length directly as divisor

            # Compare previous trading day's closing price with moving average
            current_above_ma = decision_price > sma

            # Detect price crossing MA
            if self.above_ma is None:
                self.above_ma = current_above_ma
            elif current_above_ma != self.above_ma:  # Price crosses MA
                self.above_ma = current_above_ma
                if current_above_ma:  # Crosses above MA, buy all
                    buy_shares = self.cash // current_price  # Calculate maximum shares to buy
                    if buy_shares > 0:
                        buy_price = current_price * 1.01
                        return ('buy', buy_price, buy_shares)
                else:  # Crosses below MA, sell all
                    if self.shares > 0:
                        sell_price = current_price * 0.99
                        return ('sell', sell_price, self.shares)
        return ('hold', 0, 0)  # No trading signal

    def trade(self, price, market):
        current_day = len(market.price_history) - 1

        # Record state before trading
        old_shares = self.shares
        old_cash = self.cash

        action, order_price, shares = self.decide_price(price, market)

        # Calculate current profit/loss ratio
        profit_ratio = 0
        if self.shares > 0 and self.entry_price is not None:
            profit_ratio = (price - self.entry_price) / self.entry_price

        # Record trading decision
        ma = 0
        if len(market.price_history) >= self.M:
            ma = sum(market.price_history[-self.M:]) / self.M
        self.log_trade(current_day, f"decide_{action}", price, shares, profit_ratio,
                      f"order_price={order_price:.2f}, ma={ma:.2f}, M={self.M}")

        if action == 'buy':
            # Record entry price when buying
            if self.shares == 0 or self.entry_price is None:  # New position or entry price is None
                self.entry_price = price
            else:  # Increase position, calculate weighted average cost
                self.entry_price = (self.entry_price * self.shares + price * shares) / (self.shares + shares)
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            # Reset entry price when selling all positions
            if shares >= self.shares:
                self.entry_price = None
            market.place_order('sell', order_price, shares, self)

        # Record state changes after trading
        if old_shares != self.shares or old_cash != self.cash:
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

class RandomInvestor(Investor):
    """
    Random Investor - Simulates irrational trading behavior influenced by market sentiment

    Strategy:
    - Randomly decides to buy or sell, influenced by market sentiment
    - Higher buy probability and lower sell probability when sentiment is high
    - Lower buy probability and higher sell probability when sentiment is low
    - Trading ratio influenced by market sentiment
    - Fixed price deviation range, not influenced by market sentiment

    Properties:
        p: Base probability of trading, actual probability influenced by market sentiment
    """
    def __init__(self, shares, cash, p=0.2, seed=None):
        super().__init__(shares, cash)
        self.p = p  # Trading probability
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))

    def decide_price(self, current_price, market):
        """Decide trading action, price and trading ratio based on market sentiment and random factors"""
        # Adjust buy and sell probabilities based on market sentiment
        # When market sentiment is high, increase buy probability, decrease sell probability
        # When market sentiment is low, decrease buy probability, increase sell probability
        sentiment = market.market_sentiment  # Market sentiment value (between 0-1)

        # Adjust buy and sell probabilities
        buy_prob = self.p * (0.5 + sentiment)  # Buy probability increases when market sentiment is high
        sell_prob = self.p * (1.5 - sentiment)  # Sell probability decreases when market sentiment is high

        # Ensure the sum of probabilities does not exceed 1
        hold_prob = max(0, 1 - (buy_prob + sell_prob))

        # Randomly select trading action: buy, sell or hold
        action = self._rng_investor.choice(['buy', 'sell', 'hold'], p=[buy_prob, sell_prob, hold_prob])

        if action == 'buy':
            # Trading ratio range increases when market sentiment is high
            min_ratio = 0.1 + 0.2 * sentiment  # Range from 0.1 to 0.3
            max_ratio = 0.6 + 0.4 * sentiment  # Range from 0.6 to 1.0
            random_ratio = self._rng_investor.uniform(min_ratio, max_ratio)

            buy_shares = int(self.cash * random_ratio / current_price)
            if buy_shares > 0 and self.cash >= buy_shares * current_price:
                # Use fixed price deviation range, not influenced by market sentiment
                min_price_factor = 0.95  # Fixed minimum price factor
                max_price_factor = 1.05  # Fixed maximum price factor
                price_factor = market._rng.uniform(min_price_factor, max_price_factor)
                buy_price = current_price * price_factor
                return ('buy', buy_price, buy_shares)
        elif action == 'sell':
            # Trading ratio range increases when market sentiment is low
            min_ratio = 0.1 + 0.2 * (1 - sentiment)  # Range from 0.1 to 0.3
            max_ratio = 0.6 + 0.4 * (1 - sentiment)  # Range from 0.6 to 1.0
            random_ratio = self._rng_investor.uniform(min_ratio, max_ratio)

            sell_shares = int(self.shares * random_ratio)
            if sell_shares > 0:
                # Use fixed price deviation range, not influenced by market sentiment
                min_price_factor = 0.95  # Fixed minimum price factor
                max_price_factor = 1.05  # Fixed maximum price factor
                price_factor = market._rng.uniform(min_price_factor, max_price_factor)
                sell_price = current_price * price_factor
                return ('sell', sell_price, sell_shares)
        return ('hold', 0, 0)  # No trading signal

    def trade(self, price, market):
        """Execute specific trading operation"""
        current_day = len(market.price_history) - 1

        # Record state before trading
        old_shares = self.shares
        old_cash = self.cash

        action, order_price, shares = self.decide_price(price, market)

        # Calculate current profit/loss ratio
        profit_ratio = 0
        if self.shares > 0 and self.entry_price is not None:
            profit_ratio = (price - self.entry_price) / self.entry_price

        # Record trading decision
        sentiment_str = f"{market.market_sentiment:.2f}" if hasattr(market, 'market_sentiment') else "N/A"
        self.log_trade(current_day, f"decide_{action}", price, shares, profit_ratio,
                      f"order_price={order_price:.2f}, p={self.p:.2f}, sentiment={sentiment_str}")

        if action == 'buy':
            # Record entry price when buying
            if self.shares == 0 or self.entry_price is None:  # New position or entry price is None
                self.entry_price = price
            else:  # Increase position, calculate weighted average cost
                self.entry_price = (self.entry_price * self.shares + price * shares) / (self.shares + shares)
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            # Reset entry price when selling all positions
            if shares >= self.shares:
                self.entry_price = None
            market.place_order('sell', order_price, shares, self)

        # Record state changes after trading
        if old_shares != self.shares or old_cash != self.cash:
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

class NeverStopLossInvestor(Investor):
    """
    Never Stop Loss Investor - Holds positions until price recovers to entry point

    Strategy:
    - May buy all-in at random moments
    - If in loss after buying, continues to hold
    - Only considers selling when price rises above entry point
    - Profit target adjusts dynamically: higher target when no significant loss experienced,
      lower target as duration of significant loss increases

    Properties:
        buy_price: Entry price, None if not holding
        buy_probability: Probability of buying
        initial_profit_target: Initial target profit percentage
        min_profit_target: Minimum target profit percentage
        max_loss_ratio: Maximum loss ratio during holding period
        loss_days: Days held during significant loss
    """
    # 类变量
    investor_count = 0

    def __init__(self, shares, cash, buy_probability=0.001, initial_profit_target=0.15, min_profit_target=0.02, seed=None):
        super().__init__(shares, cash)
        self.buy_price = None  # Entry price, initially None
        self.buy_probability = buy_probability  # Probability of buying
        self.initial_profit_target = initial_profit_target  # Initial target profit percentage
        self.min_profit_target = min_profit_target  # Minimum target profit percentage
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))

        # Additional properties for tracking floating loss status
        self.max_loss_ratio = 0.0  # Maximum loss ratio during holding period
        self.loss_days = 0  # Days held during significant loss
        self.significant_loss_threshold = -0.1  # Significant loss threshold, default -10%

        # Assign a unique ID to each investor
        self.investor_id = NeverStopLossInvestor.investor_count
        NeverStopLossInvestor.investor_count += 1

    # 移除特定日志文件记录方法

    def calculate_dynamic_profit_target(self):
        """Calculate dynamic profit target based on floating loss status"""
        # Use initial target if no significant loss experienced
        if self.max_loss_ratio > self.significant_loss_threshold or self.loss_days == 0:
            return self.initial_profit_target

        # Calculate dynamic profit target based on loss magnitude and holding duration
        # Larger loss and longer holding period lead to lower profit target
        loss_factor = abs(self.max_loss_ratio / self.significant_loss_threshold)  # Loss factor, closer to or exceeds 1 as loss increases
        time_factor = min(1.0, self.loss_days / 15.0)  # Time factor, max 1 (when held for 15 days or more)

        # Combined factor, range 0-1, larger value indicates lower profit target
        adjustment_factor = loss_factor * time_factor

        # Calculate dynamic profit target, linearly decreasing from initial to minimum target
        dynamic_target = self.initial_profit_target - (self.initial_profit_target - self.min_profit_target) * adjustment_factor

        # Ensure not below minimum target
        return max(dynamic_target, self.min_profit_target)

    def decide_price(self, current_price, market):
        """Decide trading action based on current position status and price"""
        # Get current state
        current_day = len(market.price_history) - 1
        random_value = self._rng_investor.random()

        # Reset buy_price to None if no shares held
        if self.shares == 0 and self.buy_price is not None:
            old_buy_price = self.buy_price
            self.buy_price = None
            self.max_loss_ratio = 0.0
            self.loss_days = 0

        # Consider buying if no position
        if self.shares == 0 and self.buy_price is None:
            # Random decision to buy
            if random_value < self.buy_probability:
                # Ensure sufficient cash
                if self.cash > 0:
                    # Buy all-in
                    buy_shares = int(self.cash / current_price)
                    if buy_shares > 0:
                        self.buy_price = current_price  # Record entry price
                        buy_price = current_price * 1.01  # Accept slightly higher buy price
                        # Reset floating loss tracking parameters
                        self.max_loss_ratio = 0.0
                        self.loss_days = 0
                        return ('buy', buy_price, buy_shares)

        # Update floating loss status and consider selling if holding position
        elif self.shares > 0 and self.buy_price is not None:
            # Calculate current profit/loss ratio
            profit_ratio = (current_price - self.buy_price) / self.buy_price

            # Update maximum loss ratio
            if profit_ratio < 0 and profit_ratio < self.max_loss_ratio:
                self.max_loss_ratio = profit_ratio

            # Update days held during significant loss
            if profit_ratio < self.significant_loss_threshold:
                self.loss_days += 1

            # Calculate dynamic profit target
            current_profit_target = self.calculate_dynamic_profit_target()

            # Sell if profit reaches dynamic target
            if profit_ratio >= current_profit_target:
                # Ensure sufficient shares
                if self.shares > 0:
                    sell_price = current_price * 0.99  # Accept slightly lower sell price
                    sell_shares = self.shares  # Sell all
                    self.buy_price = None  # Reset entry price
                    # Reset floating loss tracking parameters
                    self.max_loss_ratio = 0.0
                    self.loss_days = 0
                    return ('sell', sell_price, sell_shares)

        return ('hold', 0, 0)  # No trading signal或持有等待回本

    def trade(self, price, market):
        """Execute specific trading operation"""
        current_day = len(market.price_history) - 1

        # Record state before trading
        old_shares = self.shares
        old_cash = self.cash

        # Check for anomaly where shares are held but buy_price is None
        if self.shares > 0 and self.buy_price is None:
            # If holding shares but no purchase price record, use current price as purchase price
            self.buy_price = price
            # Reset floating loss tracking parameters
            self.max_loss_ratio = 0.0
            self.loss_days = 0
            # Record to global log
            self.log_trade(current_day, "fix_buy_price", price, self.shares, 0.0,
                          f"max_loss_ratio={self.max_loss_ratio:.4f}, loss_days={self.loss_days}")

        action, order_price, shares = self.decide_price(price, market)

        # Calculate current profit/loss ratio
        profit_ratio = 0
        if self.shares > 0 and self.buy_price is not None:
            profit_ratio = (price - self.buy_price) / self.buy_price

        # Record trading decision to global log
        dynamic_target = self.calculate_dynamic_profit_target() if self.shares > 0 else None
        buy_price_str = f"{self.buy_price}" if self.buy_price is not None else "None"
        dynamic_target_str = f"{dynamic_target:.4f}" if dynamic_target is not None else "0"
        self.log_trade(current_day, f"decide_{action}", price, shares, profit_ratio,
                      f"order_price={order_price:.2f}, buy_price={buy_price_str}, "
                      f"max_loss_ratio={self.max_loss_ratio:.4f}, loss_days={self.loss_days}, "
                      f"dynamic_target={dynamic_target_str}")

        # Execute trade
        if action == 'buy':
            # Record entry price
            self.entry_price = price
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            # Reset entry price when selling all positions
            if shares >= self.shares:
                self.entry_price = None
            market.place_order('sell', order_price, shares, self)

        # Record state changes after trading to global log
        if old_shares != self.shares or old_cash != self.cash:
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

        # Check again after trading for anomaly where shares are held but buy_price is None
        if self.shares > 0 and self.buy_price is None:
            self.buy_price = price
            self.max_loss_ratio = 0.0
            self.loss_days = 0
            # Record fix operation to global log
            self.log_trade(current_day, "fix_buy_price_after_trade", price, self.shares, 0.0,
                          f"max_loss_ratio={self.max_loss_ratio:.4f}, loss_days={self.loss_days}")

class BottomFishingInvestor(Investor):
    """
    Bottom Fishing Investor - Buys in batches after price drops x% from 100-day high

    Properties:
        profit_target: Target profit ratio (10%-50%)
        avg_cost: Weighted average cost of holdings
        trigger_drop: Price drop ratio that triggers buying (5%-15%)
        step_drop: Additional drop ratio for each batch purchase (5%-15%)
    """
    def __init__(self, shares, cash, profit_target=None, seed=None):
        super().__init__(shares, cash)
        self.avg_cost = None  # Weighted average cost of holdings
        self.profit_target = profit_target if profit_target is not None else np.random.uniform(0.1, 0.5)
        self._rng_investor = np.random.RandomState(seed if seed is not None else np.random.randint(0, 1000000))
        self.trigger_drop = self._rng_investor.uniform(0.10, 0.20)  # Price drop ratio that triggers buying
        self.step_drop = self._rng_investor.uniform(0.10, 0.30)    # Additional drop ratio for each batch purchase
        self.last_buy_price = None  # Record last purchase price

    def decide_price(self, current_price, market):
        # Selling logic: Sell all when profit target is reached
        if self.shares > 0 and self.avg_cost is not None:
            profit_ratio = (current_price - self.avg_cost) / self.avg_cost
            if profit_ratio >= self.profit_target:
                # Ensure sufficient shares
                if self.shares > 0:
                    sell_price = current_price * 0.99
                    return ('sell', sell_price, self.shares)

        # Buying logic: Start buying when price drops trigger_drop% from 100-day high, buy more for each step_drop%
        if len(market.price_history) >= 50:
            peak_price = max(market.price_history[-100:])
            drop_from_peak = (peak_price - current_price) / peak_price

            # Check if price drop reaches trigger ratio
            if drop_from_peak >= self.trigger_drop:
                # Calculate number of drop steps
                drop_steps = int((drop_from_peak - self.trigger_drop) / self.step_drop)
                if drop_steps > 0:
                    # Higher drop leads to higher buy ratio (max 80% cash)
                    buy_ratio = min(0.8, 0.1 + drop_steps * 0.1)
                    # Ensure sufficient cash
                    if self.cash > 0:
                        buy_shares = int((self.cash * buy_ratio) / current_price)
                        if buy_shares > 0:
                            buy_price = current_price * 1.01
                            return ('buy', buy_price, buy_shares)

        return ('hold', 0, 0)

    def trade(self, price, market):
        current_day = len(market.price_history) - 1

        # Record state before trading
        old_shares = self.shares
        old_cash = self.cash

        action, order_price, shares = self.decide_price(price, market)

        # Calculate current profit/loss ratio
        profit_ratio = 0
        if self.shares > 0 and self.avg_cost is not None:
            profit_ratio = (price - self.avg_cost) / self.avg_cost

        # Record trading decision
        # Calculate percentage drop from peak
        drop_from_peak = 0
        if len(market.price_history) >= 50:
            peak_price = max(market.price_history[-100:])
            drop_from_peak = (peak_price - price) / peak_price

        avg_cost_str = f"{self.avg_cost:.2f}" if self.avg_cost is not None else "None"
        self.log_trade(current_day, f"decide_{action}", price, shares, profit_ratio,
                      f"order_price={order_price:.2f}, avg_cost={avg_cost_str}, "
                      f"drop_from_peak={drop_from_peak:.4f}, trigger_drop={self.trigger_drop:.4f}, step_drop={self.step_drop:.4f}")

        if action == 'buy':
            # Update weighted average cost
            total_cost = (self.avg_cost * self.shares if self.avg_cost is not None else 0) + order_price * shares
            total_shares = self.shares + shares
            self.avg_cost = total_cost / total_shares if total_shares > 0 else None
            # Record entry price
            self.entry_price = self.avg_cost
            market.place_order('buy', order_price, shares, self)
        elif action == 'sell':
            market.place_order('sell', order_price, shares, self)
            self.avg_cost = None  # Reset cost after emptying position
            self.entry_price = None  # Reset entry price after emptying position

        # Record state changes after trading
        if old_shares != self.shares or old_cash != self.cash:
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

class InsiderInvestor(Investor):
    """
    Insider Trader - Investor who can predict stock value changes in advance
    Properties:
        prediction_days: Number of days to predict value changes (1-5 days)
        profit_target: Target profit ratio (default 15%)
        stop_loss: Stop loss ratio (default 20%)
        max_hold_days: Maximum holding period (default 30 days)
        holding_days: Current holding days
        entry_price: Purchase price
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
        # Update holding period
        if self.shares > 0 and self.entry_price is not None:
            self.holding_days += 1
            profit_ratio = (current_price - self.entry_price) / self.entry_price
            if (profit_ratio >= self.profit_target or
                profit_ratio <= -self.stop_loss or
                self.holding_days >= self.max_hold_days):
                # Ensure sufficient shares
                if self.shares > 0:
                    sell_price = current_price * 0.99
                    sell_shares = self.shares
                    self.entry_price = None
                    self.holding_days = 0
                    return ('sell', sell_price, sell_shares)

        # Check future value changes
        future_value_change = market.get_future_value_change(self.prediction_days)
        if future_value_change is not None:
            if future_value_change > 0 and self.shares == 0:
                # Ensure sufficient cash
                if self.cash > 0:
                    buy_shares = int(self.cash * 0.8 / current_price)
                    if buy_shares > 0:
                        buy_price = current_price * 1.01
                        self.entry_price = current_price
                        self.holding_days = 0
                        return ('buy', buy_price, buy_shares)
            elif future_value_change < 0 and self.shares > 0:
                # Ensure sufficient shares
                if self.shares > 0:
                    sell_price = current_price * 0.99
                    sell_shares = self.shares
                    self.entry_price = None
                    self.holding_days = 0
                    return ('sell', sell_price, sell_shares)
        return ('hold', 0, 0)

    def trade(self, price, market):
        current_day = len(market.price_history) - 1

        # Record state before trading
        old_shares = self.shares
        old_cash = self.cash

        action, order_price, shares = self.decide_price(price, market)

        # Calculate current profit/loss ratio
        profit_ratio = 0
        if self.shares > 0 and self.entry_price is not None:
            profit_ratio = (price - self.entry_price) / self.entry_price

        # Record trading decision
        future_value_change = market.get_future_value_change(self.prediction_days)
        future_value_change_str = f"{future_value_change:.4f}" if future_value_change is not None else "None"
        self.log_trade(current_day, f"decide_{action}", price, shares, profit_ratio,
                      f"order_price={order_price:.2f}, prediction_days={self.prediction_days}, "
                      f"future_value_change={future_value_change_str}, "
                      f"holding_days={self.holding_days}")

        if action in ['buy', 'sell']:
            market.place_order(action, order_price, shares, self)

        # Record state changes after trading
        if old_shares != self.shares or old_cash != self.cash:
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

class MessageInvestor(Investor):
    """Message Investor - Investors who learn about stock value changes 1-5 days after the change occurs
    Properties:
        delay_days: Delay days before learning value changes (1-5 days)
        profit_target: Target profit ratio (default 15%)
        stop_loss: Stop loss ratio (default 20%)
        max_hold_days: Maximum holding period (default 30 days)
        holding_days: Current holding period
        entry_price: Purchase price
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
        # Update holding period
        if self.shares > 0 and self.entry_price is not None:
            self.holding_days += 1
            profit_ratio = (current_price - self.entry_price) / self.entry_price
            if (profit_ratio >= self.profit_target or
                profit_ratio <= -self.stop_loss or
                self.holding_days >= self.max_hold_days):
                # Ensure sufficient shares
                if self.shares > 0:
                    sell_price = current_price * 0.99
                    sell_shares = self.shares
                    self.entry_price = None
                    self.holding_days = 0
                    return ('sell', sell_price, sell_shares)

        # Get delayed value change information
        current_day = len(market.price_history) - 1
        if market.value_curve is not None and current_day >= self.delay_days:
            past_value = market.value_curve[current_day - self.delay_days]
            current_value = market.value_curve[current_day]
            value_change = current_value - past_value

            if abs(value_change) > 0:
                if value_change > 0 and self.shares == 0:
                    # Ensure sufficient cash
                    if self.cash > 0:
                        buy_shares = int(self.cash * 0.8 / current_price)
                        if buy_shares > 0:
                            buy_price = current_price * 1.01
                            self.entry_price = current_price
                            self.holding_days = 0
                            return ('buy', buy_price, buy_shares)
                elif value_change < 0 and self.shares > 0:
                    # Ensure sufficient shares
                    if self.shares > 0:
                        sell_price = current_price * 0.99
                        sell_shares = self.shares
                        self.entry_price = None
                        self.holding_days = 0
                        return ('sell', sell_price, sell_shares)
        return ('hold', 0, 0)

    def trade(self, price, market):
        current_day = len(market.price_history) - 1

        # Record state before trading
        old_shares = self.shares
        old_cash = self.cash

        action, order_price, shares = self.decide_price(price, market)

        # Calculate current profit/loss ratio
        profit_ratio = 0
        if self.shares > 0 and self.entry_price is not None:
            profit_ratio = (price - self.entry_price) / self.entry_price

        # Record trading decision
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

        # Record state changes after trading
        if old_shares != self.shares or old_cash != self.cash:
            self.log_trade(current_day, "trade_result", price, self.shares, profit_ratio,
                          f"{old_shares}->{self.shares}, {old_cash:.2f}->{self.cash:.2f}")

class Market:
    """Stock Market - Market simulator implementing call auction mechanism

    Strategy:
    - Collect buy and sell orders
    - Determine clearing price through call auction
    - Execute trades based on price priority
    - Stock value changes as a random square wave
    - Support capital injection and withdrawal on specified dates
    - Implement transaction fee mechanism

    Properties:
        price: Current market price
        price_tick: Minimum price movement unit
        price_history: Price history
        buy_orders: Buy order queue
        sell_orders: Sell order queue
        executed_volume: Executed trading volume
        true_value: True stock value
        value_history: Value history
        seed: Random seed
        capital_change_history: Capital change history
        buy_fee_rate: Buy transaction fee rate
        sell_fee_rate: Sell transaction fee rate
        fee_income: Accumulated fee income
    """
    def __init__(self, initial_price, price_tick=0.01, value_curve=None, seed=None, close_seed=None, buy_fee_rate=0.001, sell_fee_rate=0.002):
        self.price = initial_price
        self.price_tick = price_tick
        self.price_history = [initial_price]
        self.buy_orders = []
        self.sell_orders = []
        self.executed_volume = 0
        self.value_curve = value_curve  # Reference to externally provided value curve
        self.true_value = value_curve[0] if value_curve is not None else initial_price
        self.value_history = [self.true_value]
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)
        self._rng = np.random.RandomState(self.seed)

        # Set independent random seed for closing price
        self.close_seed = close_seed if close_seed is not None else (seed + 20240501 if seed is not None else np.random.randint(0, 1000000))
        self._rng_close = np.random.RandomState(self.close_seed)

        self.executed_volume_history = []
        self.capital_change_history = []
        self.buy_fee_rate = buy_fee_rate
        self.sell_fee_rate = sell_fee_rate
        self.fee_income = 0

        # Market sentiment related properties
        self.market_sentiment = 0.5  # Initialize as neutral sentiment (between 0-1, 0.5 is neutral)
        self.sentiment_history = [0.5]  # Market sentiment history
        self.volume_ma = None  # Volume moving average
        self.volume_ma_period = 10  # Volume moving average period

    def update_market_sentiment(self):
        """Update market sentiment indicator
        Calculate new market sentiment indicator based on price changes,
        volume changes and current sentiment state
        """
        # Calculate price change
        if len(self.price_history) > 1:
            price_change = (self.price_history[-1] / self.price_history[-2] - 1)
        else:
            price_change = 0

        # Calculate trading volume change
        if len(self.executed_volume_history) > 1:
            # Update volume moving average
            if len(self.executed_volume_history) >= self.volume_ma_period:
                self.volume_ma = np.mean(self.executed_volume_history[-self.volume_ma_period:])
            else:
                self.volume_ma = np.mean(self.executed_volume_history)

            # Calculate volume change relative to moving average
            if self.volume_ma > 0:
                volume_factor = self.executed_volume_history[-1] / self.volume_ma
            else:
                volume_factor = 1.0
        else:
            volume_factor = 1.0

        # Update market sentiment
        # Price rise with volume increase leads to more positive sentiment
        sentiment_change = 0

        # Price factor
        if price_change > 0:
            sentiment_change += 0.05 * price_change  # Price rise improves sentiment
        else:
            sentiment_change += 0.03 * price_change  # Price fall reduces sentiment, but with less impact

        # Volume factor
        if volume_factor > 1.0:
            # Volume increase impact depends on price change direction
            if price_change > 0:
                # Volume and price rise together, significantly improves sentiment
                sentiment_change += 0.05 * (volume_factor - 1) * (1 + price_change * 10)
            else:
                # Volume up but price down, slightly reduces sentiment
                sentiment_change -= 0.02 * (volume_factor - 1)
        else:
            # Volume decrease slightly reduces sentiment
            sentiment_change -= 0.01 * (1 - volume_factor)

        # Sentiment inertia factor, makes sentiment slowly return to neutral state
        sentiment_change -= 0.01 * (self.market_sentiment - 0.5)

        # Update market sentiment, ensure within 0-1 range
        self.market_sentiment = max(0.0, min(1.0, self.market_sentiment + sentiment_change))
        self.sentiment_history.append(self.market_sentiment)

    def place_order(self, order_type, price, shares, investor):
        if order_type == 'buy':
            self.buy_orders.append((price, shares, investor))
        elif order_type == 'sell':
            self.sell_orders.append((price, shares, investor))

    def call_auction(self, buy_orders, sell_orders, last_price):
        # Check if order lists are empty
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
        # Ensure all parameters are valid
        if max_volume <= 0 or not buy_orders or not sell_orders or not executed_buy_idx or not executed_sell_idx:
            return

        # Execute buy orders
        remain = max_volume
        buy_orders_sorted = sorted(enumerate(buy_orders), key=lambda x: x[1][0], reverse=True)
        if buy_orders_sorted:  # Ensure there are buy orders
            for idx, (price, shares, investor) in buy_orders_sorted:
                if idx in executed_buy_idx and remain > 0:
                    exec_shares = min(shares, remain)
                    remain -= exec_shares

                    # Calculate trade amount and buy fee
                    trade_amount = exec_shares * clearing_price
                    buy_fee = trade_amount * self.buy_fee_rate
                    total_cost = trade_amount + buy_fee

                    # Check if investor has enough cash
                    if investor.cash >= total_cost:
                        # Update investor's position and cash (deduct fees)
                        investor.shares += exec_shares
                        investor.cash -= total_cost

                        # Accumulate fee income
                        self.fee_income += buy_fee
                    else:
                        # If cash insufficient, adjust purchase quantity
                        affordable_shares = int((investor.cash / (clearing_price * (1 + self.buy_fee_rate))))
                        if affordable_shares > 0:
                            # Recalculate trade amount and fees
                            trade_amount = affordable_shares * clearing_price
                            buy_fee = trade_amount * self.buy_fee_rate
                            total_cost = trade_amount + buy_fee

                            # Update investor's position and cash
                            investor.shares += affordable_shares
                            investor.cash -= total_cost

                            # Accumulate fee income
                            self.fee_income += buy_fee

                    # Fee income already accumulated above

                    # No longer update value investor's valuation here, let them update in decide_price based on delay time

        # Execute sell orders
        remain = max_volume
        sell_orders_sorted = sorted(enumerate(sell_orders), key=lambda x: x[1][0])
        if sell_orders_sorted:  # Ensure there are sell orders
            for idx, (price, shares, investor) in sell_orders_sorted:
                if idx in executed_sell_idx and remain > 0:
                    exec_shares = min(shares, remain)
                    remain -= exec_shares

                    # Calculate trade amount and sell fee
                    trade_amount = exec_shares * clearing_price
                    sell_fee = trade_amount * self.sell_fee_rate

                    # Check if investor has enough shares
                    if investor.shares >= exec_shares:
                        # Update investor's position and cash (deduct fees)
                        investor.shares -= exec_shares
                        investor.cash += (trade_amount - sell_fee)

                        # Accumulate fee income
                        self.fee_income += sell_fee
                    else:
                        # If shares insufficient, only sell available shares
                        available_shares = investor.shares
                        if available_shares > 0:
                            # Recalculate trade amount and fees
                            trade_amount = available_shares * clearing_price
                            sell_fee = trade_amount * self.sell_fee_rate

                            # Update investor's position and cash
                            investor.shares = 0
                            investor.cash += (trade_amount - sell_fee)

                            # Accumulate fee income
                            self.fee_income += sell_fee

                    # No longer update value investor's valuation here, let them update in decide_price based on delay time

    def inject_capital(self, investors, amount=None, percentage=None, day=None):
        """
        Inject capital to investors

        Args:
            investors (list): List of investors
            amount (float, optional): Fixed amount to inject
            percentage (float, optional): Percentage of current cash to inject
            day (int, optional): Day of injection, default is current day

        Note: Either amount or percentage must be provided, but not both
        """
        if amount is None and percentage is None:
            raise ValueError("必须提供固定金额(amount)或比例(percentage)参数之一")
        if amount is not None and percentage is not None:
            raise ValueError("不能同时提供固定金额(amount)和比例(percentage)参数")

        # Determine current day
        current_day = day if day is not None else len(self.price_history) - 1

        # Inject capital for each investor
        for investor in investors:
            if amount is not None:
                # Inject fixed amount
                investor.cash += amount
                injection_amount = amount
            else:
                # Inject proportional amount
                injection_amount = investor.cash * percentage
                investor.cash += injection_amount

        # Record capital injection history
        if amount is not None:
            self.capital_change_history.append((current_day, "inject", amount, None))
        else:
            self.capital_change_history.append((current_day, "inject", None, percentage))

    def withdraw_capital(self, investors, amount=None, percentage=None, day=None):
        """
        Withdraw capital from investors

        Args:
            investors (list): List of investors
            amount (float, optional): Fixed amount to withdraw
            percentage (float, optional): Percentage of current cash to withdraw
            day (int, optional): Day of withdrawal, default is current day

        Note: Either amount or percentage must be provided, but not both
        """
        if amount is None and percentage is None:
            raise ValueError("必须提供固定金额(amount)或比例(percentage)参数之一")
        if amount is not None and percentage is not None:
            raise ValueError("不能同时提供固定金额(amount)和比例(percentage)参数")

        # Determine current day
        current_day = day if day is not None else len(self.price_history) - 1

        # Withdraw capital from each investor
        for investor in investors:
            if amount is not None:
                # Withdraw fixed amount, but not exceeding investor's cash
                withdrawal_amount = min(amount, investor.cash)
                investor.cash -= withdrawal_amount
            else:
                # Withdraw proportional amount
                withdrawal_amount = investor.cash * percentage
                investor.cash -= withdrawal_amount

        # Record capital withdrawal history
        if amount is not None:
            self.capital_change_history.append((current_day, "withdraw", amount, None))
        else:
            self.capital_change_history.append((current_day, "withdraw", None, percentage))

    def daily_auction(self):
        # Update market sentiment
        self.update_market_sentiment()

        # Initialize daily price data
        current_day = len(self.price_history) - 1
        last_price = self.price_history[-1]
        open_price = last_price  # Default open price is previous day's close
        high_price = last_price  # Initialize high price
        low_price = last_price   # Initialize low price
        close_price = last_price # Initialize close price
        daily_volume = 0         # Initialize daily volume

        # Temporarily store pre-opening buy/sell orders
        opening_buy_orders = []
        opening_sell_orders = []

        # Temporarily store pre-closing buy/sell orders
        closing_buy_orders = []
        closing_sell_orders = []

        # 1. Randomly allocate some orders for opening call auction
        # Randomly select 30%-50% of orders to participate in opening call auction
        opening_ratio = self._rng.uniform(0.3, 0.5)

        # Randomly select buy orders for opening call auction
        if self.buy_orders:
            num_opening_buy = max(1, int(len(self.buy_orders) * opening_ratio))
            opening_indices = self._rng.choice(len(self.buy_orders), num_opening_buy, replace=False)
            for i in opening_indices:
                opening_buy_orders.append(self.buy_orders[i])
            # Remove these orders from original order list
            self.buy_orders = [order for i, order in enumerate(self.buy_orders) if i not in opening_indices]

        # Randomly select sell orders for opening call auction
        if self.sell_orders:
            num_opening_sell = max(1, int(len(self.sell_orders) * opening_ratio))
            opening_indices = self._rng.choice(len(self.sell_orders), num_opening_sell, replace=False)
            for i in opening_indices:
                opening_sell_orders.append(self.sell_orders[i])
            # Remove these orders from original order list
            self.sell_orders = [order for i, order in enumerate(self.sell_orders) if i not in opening_indices]

        # 2. Execute opening call auction
        if opening_buy_orders and opening_sell_orders:
            open_price, open_volume, executed_buy_idx, executed_sell_idx = self.call_auction(
                opening_buy_orders, opening_sell_orders, last_price)

            # Execute opening call auction trades
            self.execute_trades(open_price, open_volume, opening_buy_orders, opening_sell_orders,
                               executed_buy_idx, executed_sell_idx)
            daily_volume += open_volume

            # Update high/low prices
            high_price = max(high_price, open_price)
            low_price = min(low_price, open_price)

            # Clear executed orders
            opening_buy_orders = [order for i, order in enumerate(opening_buy_orders) if i not in executed_buy_idx]
            opening_sell_orders = [order for i, order in enumerate(opening_sell_orders) if i not in executed_sell_idx]

            # Add unfilled opening orders back to main order list
            self.buy_orders.extend(opening_buy_orders)
            self.sell_orders.extend(opening_sell_orders)

        # 3. Intraday trading - simulate orders arriving and matching
        max_iterations = 100  # Set max iterations to prevent infinite loop
        iteration = 0

        while (self.buy_orders or self.sell_orders) and iteration < max_iterations:
            # Randomly decide whether to conduct this round of matching
            if self._rng.random() < 0.8:  # 80% probability of matching
                # Temporarily store orders participating in this round
                current_buy_orders = []
                current_sell_orders = []

                # Randomly select a portion of buy orders for this round
                if self.buy_orders:
                    num_current_buy = max(1, int(len(self.buy_orders) * self._rng.uniform(0.1, 0.3)))
                    current_indices = self._rng.choice(len(self.buy_orders), min(num_current_buy, len(self.buy_orders)), replace=False)
                    for i in current_indices:
                        current_buy_orders.append(self.buy_orders[i])
                    # Remove these orders from the original order list
                    self.buy_orders = [order for i, order in enumerate(self.buy_orders) if i not in current_indices]

                # Randomly select a portion of sell orders for this round
                if self.sell_orders:
                    num_current_sell = max(1, int(len(self.sell_orders) * self._rng.uniform(0.1, 0.3)))
                    current_indices = self._rng.choice(len(self.sell_orders), min(num_current_sell, len(self.sell_orders)), replace=False)
                    for i in current_indices:
                        current_sell_orders.append(self.sell_orders[i])
                    # Remove these orders from the original order list
                    self.sell_orders = [order for i, order in enumerate(self.sell_orders) if i not in current_indices]

                # If there are both buy and sell orders in this round, attempt matching
                if current_buy_orders and current_sell_orders:
                    match_price, match_volume, executed_buy_idx, executed_sell_idx = self.call_auction(
                        current_buy_orders, current_sell_orders, open_price if iteration == 0 else self.price)

                    # If there are trades executed
                    if match_volume > 0:
                        # Execute trades
                        self.execute_trades(match_price, match_volume, current_buy_orders, current_sell_orders,
                                          executed_buy_idx, executed_sell_idx)
                        daily_volume += match_volume

                        # Update high/low prices
                        high_price = max(high_price, match_price)
                        low_price = min(low_price, match_price)

                        # Update current price
                        self.price = match_price

                    # Add unexecuted orders back to main order list
                    current_buy_orders = [order for i, order in enumerate(current_buy_orders) if i not in executed_buy_idx]
                    current_sell_orders = [order for i, order in enumerate(current_sell_orders) if i not in executed_sell_idx]

                    # Randomly determine the fate of unexecuted orders
                    # Some return to main order list, some enter closing auction, some get cancelled
                    for order in current_buy_orders:
                        r = self._rng.random()
                        if r < 0.4:  # 40% probability of returning to main order list
                            self.buy_orders.append(order)
                        elif r < 0.7:  # 30% probability of entering closing auction
                            closing_buy_orders.append(order)
                        # Remaining 30% probability of order cancellation

                    for order in current_sell_orders:
                        r = self._rng.random()
                        if r < 0.4:  # 40% probability of returning to main order list
                            self.sell_orders.append(order)
                        elif r < 0.7:  # 30% probability of entering closing auction
                            closing_sell_orders.append(order)
                        # Remaining 30% probability of order cancellation

            # Randomly decide whether to move some orders from main list to closing auction
            if iteration > max_iterations * 0.7 and (self.buy_orders or self.sell_orders):  # In later iterations
                # Randomly select some buy orders to move to closing auction
                if self.buy_orders:
                    num_closing_buy = max(1, int(len(self.buy_orders) * self._rng.uniform(0.1, 0.2)))
                    closing_indices = self._rng.choice(len(self.buy_orders), min(num_closing_buy, len(self.buy_orders)), replace=False)
                    for i in closing_indices:
                        closing_buy_orders.append(self.buy_orders[i])
                    # Remove these orders from the original order list
                    self.buy_orders = [order for i, order in enumerate(self.buy_orders) if i not in closing_indices]

                # Randomly select some sell orders to move to closing auction
                if self.sell_orders:
                    num_closing_sell = max(1, int(len(self.sell_orders) * self._rng.uniform(0.1, 0.2)))
                    closing_indices = self._rng.choice(len(self.sell_orders), min(num_closing_sell, len(self.sell_orders)), replace=False)
                    for i in closing_indices:
                        closing_sell_orders.append(self.sell_orders[i])
                    # Remove these orders from the original order list
                    self.sell_orders = [order for i, order in enumerate(self.sell_orders) if i not in closing_indices]

            iteration += 1

        # 4. Add remaining orders from the main order list to the closing call auction
        closing_buy_orders.extend(self.buy_orders)
        closing_sell_orders.extend(self.sell_orders)
        self.buy_orders = []
        self.sell_orders = []

        # 5. Execute closing call auction
        # if closing_buy_orders and closing_sell_orders:
        close_price, close_volume, executed_buy_idx, executed_sell_idx = self.call_auction(
            closing_buy_orders, closing_sell_orders, self.price)

        # Execute trades from closing call auction
        self.execute_trades(close_price, close_volume, closing_buy_orders, closing_sell_orders,
                           executed_buy_idx, executed_sell_idx)
        daily_volume += close_volume

        # Update high and low prices
        high_price = max(high_price, close_price)
        low_price = min(low_price, close_price)
        # else:
        #     # If no closing call auction, use randomly generated closing price
        #     if high_price > low_price:
        #         # Generate closing price randomly between high and low, but biased towards weighted average of open price and last trade price
        #         weight = self._rng_close.uniform(0.3, 0.7)  # Use dedicated random number generator for closing price to generate random weight
        #         base_price = weight * open_price + (1 - weight) * self.price
        #         # Random fluctuation around base price, ensuring it stays between high and low
        #         close_price = max(low_price, min(high_price,
        #                                        base_price + self._rng_close.uniform(-0.5, 0.5) * (high_price - low_price)))
        #     else:
        #         close_price = high_price  # If high price equals low price, closing price equals high price

        # Update market state
        self.price = close_price
        self.price_history.append(close_price)
        self.executed_volume = daily_volume
        self.executed_volume_history.append(daily_volume)

        # Record daily OHLC data (Open, High, Low, Close)
        if not hasattr(self, 'ohlc_data'):
            self.ohlc_data = []
        self.ohlc_data.append((open_price, high_price, low_price, close_price))

        # Clear remaining orders
        self.buy_orders = []
        self.sell_orders = []

        # Update true value
        if self.value_curve is not None and current_day < len(self.value_curve):
            self.true_value = self.value_curve[current_day]
        self.value_history.append(self.true_value)

    def get_future_value_change(self, prediction_days):
        """Get future value change"""
        current_day = len(self.price_history) - 1
        if self.value_curve is not None:
            future_idx = current_day + prediction_days
            if future_idx < len(self.value_curve):
                change = self.value_curve[future_idx] - self.value_curve[current_day]
                return change
        return None

def simulate_stock_market(value_line_seed=2133, close_seed=None, enable_trade_log=True):
    # Basic market parameter settings
    initial_price = 100
    price_tick = 0.01
    days = 1000
    buy_fee_rate = 0.001
    sell_fee_rate = 0.001

    # Set whether to enable trade log recording
    Investor.set_enable_trade_log(enable_trade_log)

    # Initialize global log file for all investors
    Investor.init_trade_log_file("all_investors_trade_log.csv")

    # No longer initialize specific log file for NeverStopLossInvestor

    # Create value curve
    value_curve = TrueValueCurve(initial_value=initial_price, days=days, seed=value_line_seed)

    # Create market instance with value curve
    market = Market(initial_price, price_tick, value_curve=value_curve, seed=value_line_seed, close_seed=close_seed,
                   buy_fee_rate=buy_fee_rate, sell_fee_rate=sell_fee_rate)

    # Parameter settings for different types of investors
    value_investors_params = {
        'num': 100,  # Larger number of value investors
        'initial_shares': 100,
        'initial_cash': 10000
    }

    chase_investors_params = {
        'num': 100,  # Moderate number of momentum investors
        'initial_shares': 100,
        'initial_cash': 10000
    }

    trend_investors_params = {
        'num': 100,  # Relatively fewer trend investors
        'initial_shares': 100,
        'initial_cash': 10000
    }

    random_investors_params = {
        'num': 100,  # More random investors
        'initial_shares': 100,
        'initial_cash': 10000
    }

    never_stop_loss_investors_params = {
        'num': 100,  # Moderate number of never-stop-loss investors
        'initial_shares': 0,  # No initial shares, start from 0
        'initial_cash': 20000  # Increased initial cash
    }

    bottom_fishing_investors_params = {
        'num': 100,  # Fewer bottom-fishing investors
        'initial_shares': 100,
        'initial_cash': 10000
    }

    insider_investors_params = {
        'num': 10,  # Very few insider traders
        'initial_shares': 100,
        'initial_cash': 10000
    }

    message_investors_params = {
        'num': 10,  # Fewer message investors but more than insider traders
        'initial_shares': 100,
        'initial_cash': 10000
    }

    investors = []
    trend_periods = [5, 10, 15, 20, 25, 30, 60, 100, 150, 200]
    trend_investors_by_period = {}

    # Create value investors, pre-generate valuation bias percentages for all investors
    # Use normal distribution to generate fixed valuation bias percentages, standard deviation is 15%
    bias_percent_std = 0.15  # Standard deviation of bias percentage is 15%
    investor_bias_percents = market._rng.normal(0, bias_percent_std, value_investors_params['num'])

    # Create value investors, each investor uses pre-generated fixed bias percentage and random revaluation delay
    for i, bias_percent in enumerate(investor_bias_percents):
        # Randomly generate maximum deviation ratio (between 0.2-0.4)
        max_deviation = market._rng.uniform(0.2, 0.4)
        # Randomly generate target position ratio (between 0.4-0.6)
        target_position = market._rng.uniform(0.4, 0.6)

        investors.append(ValueInvestor(
            value_investors_params['initial_shares'],
            value_investors_params['initial_cash'],
            bias_percent=bias_percent,  # Each investor has a fixed valuation bias percentage
            max_deviation=max_deviation,  # Maximum deviation ratio
            target_position=target_position,  # Target position ratio
            seed=value_line_seed + i  # Use different seeds to ensure each investor has different revaluation delay
        ))

    # Create momentum investors (chase up and down)
    for _ in range(chase_investors_params['num']):
        investors.append(ChaseInvestor(
            chase_investors_params['initial_shares'],
            chase_investors_params['initial_cash']
        ))

    # Create trend investors
    trend_investors_num_per_period = trend_investors_params['num'] // len(trend_periods)
    for period in trend_periods:
        trend_investors_by_period[period] = []
        for _ in range(trend_investors_num_per_period):
            investor = TrendInvestor(
                trend_investors_params['initial_shares'],
                trend_investors_params['initial_cash'],
                period
            )
            investors.append(investor)
            trend_investors_by_period[period].append(investor)

    # Create random investors
    for _ in range(random_investors_params['num']):
        investors.append(RandomInvestor(
            random_investors_params['initial_shares'],
            random_investors_params['initial_cash']
        ))

    # Create never-stop-loss investors
    for _ in range(never_stop_loss_investors_params['num']):
        investor = NeverStopLossInvestor(
            never_stop_loss_investors_params['initial_shares'],
            never_stop_loss_investors_params['initial_cash'],
            buy_probability=0.3,  # 30% probability of full position buying, increased buying probability
            initial_profit_target=0.15,  # 15% initial target profit ratio
            min_profit_target=0.02  # 2% minimum target profit ratio
        )
        investors.append(investor)

    # Create bottom fishing investors
    for _ in range(bottom_fishing_investors_params['num']):
        profit_target = np.random.uniform(0.1, 0.5)
        investors.append(BottomFishingInvestor(
            bottom_fishing_investors_params['initial_shares'],
            bottom_fishing_investors_params['initial_cash'],
            profit_target=profit_target
        ))

    # Create insider traders
    for _ in range(insider_investors_params['num']):
        investors.append(InsiderInvestor(
            insider_investors_params['initial_shares'],
            insider_investors_params['initial_cash']
        ))

    # Create message-driven investors
    for _ in range(message_investors_params['num']):
        investors.append(MessageInvestor(
            message_investors_params['initial_shares'],
            message_investors_params['initial_cash']
        ))

    # Calculate start and end indices for each type of investor
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

    # Define dates and parameters for capital injection and withdrawal
    # capital_events = [
    #     {'day': 500, 'type': 'inject', 'amount': 10000, 'investors_range': (0, len(investors))},  # Day 1000: Inject 5000 capital to all investors
    #     {'day': 800, 'type': 'inject', 'percentage': 0.8, 'investors_range': (0, value_end)},  # Day 2000: Inject 20% capital to value investors
    #     {'day': 1000, 'type': 'withdraw', 'percentage': 0.5, 'investors_range': (0, len(investors))},  # Day 3000: Withdraw 10% capital from all investors
    #     {'day': 1500, 'type': 'withdraw', 'amount': 2000, 'investors_range': (random_end, never_stop_loss_end)}  # Day 4000: Withdraw 2000 capital from never-stop-loss investors
    # ]

    # capital_events = [
    #     {'day': 400, 'type': 'inject', 'percentage': 2, 'investors_range': (0, len(investors))},  # 第2000天为价值投资者注入20%资金
    # ]

    capital_events = []

    for day in range(days):
        # Check for capital injection or withdrawal events
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
        # Execute daily call auction
        market.daily_auction()
        # No need to append price again as daily_auction has already done it
        # prices.append(market.price)

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
            # Only calculate averages if there are investors
            if start < end:
                shares_list = [inv.shares for inv in investors[start:end]]
                cash_list = [inv.cash for inv in investors[start:end]]
                if shares_list and cash_list:  # Ensure lists are not empty
                    avg_shares = np.mean(shares_list)  # Calculate average shares held
                    avg_cash = np.mean(cash_list)      # Calculate average cash
                    avg_wealth = avg_cash + avg_shares * market.price  # Calculate average total wealth
                else:
                    avg_shares = avg_cash = avg_wealth = 0.0
            else:
                avg_shares = avg_cash = avg_wealth = 0.0
            shares_by_type[type_name].append(avg_shares)
            cash_by_type[type_name].append(avg_cash)
            wealth_by_type[type_name].append(avg_wealth)

        # Record total assets for trend investors with different periods
        for period in trend_periods:
            investors_list = trend_investors_by_period[period]
            if investors_list:  # Ensure there are investors
                assets_list = [inv.cash + inv.shares * market.price for inv in investors_list]
                if assets_list:  # Ensure list is not empty
                    avg_assets = np.mean(assets_list)
                else:
                    avg_assets = 0.0
            else:
                avg_assets = 0.0
            trend_assets_by_period[period].append(avg_assets)

    # Create plots to display simulation results
    # Create visualization - 5 subplots
    fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)  # Share x-axis

    # Plot candlestick chart and true value (first subplot)
    # Prepare candlestick data
    if hasattr(market, 'ohlc_data') and market.ohlc_data:
        # Extract OHLC data
        opens = [data[0] for data in market.ohlc_data]
        highs = [data[1] for data in market.ohlc_data]
        lows = [data[2] for data in market.ohlc_data]
        closes = [data[3] for data in market.ohlc_data]

        # Draw candlestick chart
        # Calculate candlestick width
        width = 0.6
        # Create x-axis positions
        x = np.arange(len(opens))

        # Draw shadows (lines from high to low)
        for i in range(len(opens)):
            # Determine candlestick color based on open and close prices
            if closes[i] >= opens[i]:  # Close higher than open, bullish (red)
                color = 'red'
                # Draw body (rectangle from open to close)
                axs[0].add_patch(plt.Rectangle((i-width/2, opens[i]), width, closes[i]-opens[i],
                                              fill=True, color=color, alpha=0.6))
            else:  # Close lower than open, bearish candle (green)
                color = 'green'
                # Draw body (rectangle from close to open)
                axs[0].add_patch(plt.Rectangle((i-width/2, closes[i]), width, opens[i]-closes[i],
                                              fill=True, color=color, alpha=0.6))

            # Draw line from high to low using the same color as the candlestick body
            axs[0].plot([i, i], [lows[i], highs[i]], color=color, linewidth=1)

        # Draw true value curve
        axs[0].plot(market.value_history, label='True Value', linestyle='--', color='blue', alpha=0.7)

        axs[0].set_ylabel('Price')
        axs[0].legend(loc='upper left')
        axs[0].set_title('OHLC Candlestick Chart and True Value')

        # Add candlestick explanation to legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
        legend_elements = [
            Line2D([0], [0], color='blue', linestyle='--', label='True Value'),
            Rectangle((0, 0), 1, 1, color='red', alpha=0.6, label='Bullish Candle'),
            Rectangle((0, 0), 1, 1, color='green', alpha=0.6, label='Bearish Candle')
        ]
        axs[0].legend(handles=legend_elements, loc='upper left')
    else:
        # If no OHLC data, draw regular price curve
        axs[0].plot(market.price_history, label='Stock Price', color='blue')
        axs[0].plot(market.value_history, label='True Value', linestyle='--', color='green')
        axs[0].set_ylabel('Price', color='blue')
        axs[0].tick_params(axis='y', labelcolor='blue')
        axs[0].legend(loc='upper left')

    # Add trading volume to first subplot (second y-axis)
    ax2 = axs[0].twinx()  # Create twin y-axis sharing x-axis
    # Ensure volume history length matches price history length
    daily_volumes = market.executed_volume_history
    # Ensure volume data length matches days
    ax2.bar(range(len(daily_volumes)), daily_volumes, alpha=0.3, color='gray', label='Trading Volume')
    ax2.set_ylabel('Volume', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.legend(loc='upper right')

    # Add capital injection and withdrawal event markers to first subplot
    if market.capital_change_history:
        for day, event_type, amount, percentage in market.capital_change_history:
            if event_type == 'inject':
                color = 'green'  # Green for capital injection
                marker = '^'     # Up triangle for injection
                label = f'Inject: {amount if amount else percentage*100:.1f}%'
            else:  # withdraw
                color = 'red'    # Red for capital withdrawal
                marker = 'v'     # Down triangle for withdrawal
                label = f'Withdraw: {amount if amount else percentage*100:.1f}%'

            # Mark the event on the plot
            axs[0].axvline(x=day, color=color, alpha=0.3, linestyle='--')
            axs[0].plot(day, market.price_history[day], marker=marker, markersize=10, color=color, label=label)

    # Plot average shares by investor type (second subplot)
    for type_name in shares_by_type:
        axs[1].plot(shares_by_type[type_name], label=f'{type_name} Investors')
    axs[1].set_ylabel('Shares')
    axs[1].set_title('Average Shares Held by Investor Type')
    axs[1].legend()

    # Plot average cash by investor type (third subplot)
    for type_name in cash_by_type:
        axs[2].plot(cash_by_type[type_name], label=f'{type_name} Investors')
    axs[2].set_ylabel('Cash')
    axs[2].set_title('Average Cash Held by Investor Type')
    axs[2].legend()

    # Plot average total wealth by investor type (fourth subplot)
    for type_name in wealth_by_type:
        axs[3].plot(wealth_by_type[type_name], label=f'{type_name} Investors')
    axs[3].set_ylabel('Total Wealth')
    axs[3].set_title('Average Total Wealth by Investor Type')
    axs[3].legend()

    # Plot trend investors' performance by period (fifth subplot)
    for period in trend_periods:
        axs[4].plot(trend_assets_by_period[period], label=f'MA{period}')
    axs[4].set_ylabel('Total Assets')
    axs[4].set_title('Trend Investors Performance by MA Period')
    axs[4].legend()

    # Set common x-label for the fifth subplot
    axs[4].set_xlabel('Day')

    # Add legend for capital change events in the first subplot
    if market.capital_change_history:
        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[0].legend(by_label.values(), by_label.keys(), loc='upper left')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Close the global log file for all investors
    Investor.close_trade_log_file()

# Run simulation
if __name__ == "__main__":
    # You can generate different value curves by passing different value_line_seed
    # You can generate different closing price sequences by passing different close_seed
    # For example, try different value_line_seed values: 2108, 2133, 2200, etc.
    # Set enable_trade_log=True to enable logging, False to disable logging
    simulate_stock_market(value_line_seed=2108, close_seed=None, enable_trade_log=False)