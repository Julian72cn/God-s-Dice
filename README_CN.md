# 上帝的骰子 - 基于行为金融学的股票市场量化模型

![GitHub](https://img.shields.io/github/license/your-username/gods-dice)

> "上帝不会掷骰子。" - 爱因斯坦

这是一个用于研究金融市场中随机性的量化模拟系统。通过模拟不同类型的投资者行为和设定简化的市场机制，研究：1.金融市场的随机性和蝴蝶效应的相关问题；2.不同类型的投资者的行为对金融市场的影响；3.不同类型的投资者在金融市场的表现。

## 项目特点

- **多种投资者类型**：
  - 价值投资者 (Value Investor)
  - 追涨杀跌投资者 (Chase Investor)
  - 趋势投资者 (Trend Investor)
  - 随机投资者 (Random Investor)
  - 内幕交易者 (Insider Investor)
  - 基于消息的投资者 (Message Investor)
  - 永不止损投资者 (Never Stop Loss Investor)
  - 抄底投资者 (Bottom Fishing Investor)

- **接近真实市场交易机制**：
  - 集合竞价定价机制
  - 交易量和流动性影响
  - 交易费用系统
  - 价格发现过程
  - 资金注入和撤出机制

- **全面的分析工具**：
  - 蒙特卡罗模拟
  - 蝴蝶效应分析
  - 市场影响度量
  - 价格偏差统计
  - 详细的交易日志

## 核心模块

### 1. 市场模拟 (Market Simulation)
- `main_ca_3.2.1.1.py`: 集合竞价的交易机制实现（旧版）
- `main_OHLC_2.0.5.1.py`: 核心市场模拟引擎

### 2. 蝴蝶效应研究 (Butterfly Effect Studies)
- `butterfly_effect_simulation_1.*.py`: 蝴蝶效应模拟和相关分析研究


### 3. 蒙特卡罗模拟 (Monte Carlo Simulation)
- `MC_simulation_1.0-1.4.py`: 蒙特卡罗方法系列实现
- `MC_simulation_asset_return_analysis_1.*.py`: 不同类型投资者资产收益分析
- `MC_simulation_bias_std.py`: 价值投资者的市场影响研究
- `MC_simulation_chase_period.py`: 追涨杀跌这的市场影响研究
- `MC_simulation_trend_period.py`: 趋势投资者的市场影响研究

### 4. 布朗运动研究 (Brownian Motion)
- `brownian_motion_test_1.0.py`: 布朗运动验证工具


## 环境要求
- Python 3.7+
- NumPy
- Matplotlib
- Pandas (可选，用于数据分析)


## 许可证

本项目采用 **[Apache License 2.0](LICENSE)** 开源协议。  
完整协议内容请见项目根目录下的 `LICENSE` 文件。

## 致谢



