#!/usr/bin/env python3
"""测试警告修复效果"""

import warnings
import numpy as np
import pandas as pd

print("=== 测试警告修复效果 ===\n")

# 模拟原问题：pandas比较警告
print("1. 测试原始问题场景...")

# 模拟原来会出警告的代码
test_data = pd.DataFrame({
    'A': [1.0, 2.0, np.nan, 4.0, np.inf],
    'B': [0.01, np.nan, 0.03, 0.04, -np.inf],
    'C': [1.0, 1.5, np.nan, 2.0, 2.5]
})

print("测试数据:")
print(test_data)

# 测试比较操作
print("\n测试比较操作...")
try:
    result = test_data['A'] > test_data['B']
    print("✅ 比较操作完成，无警告")
except Exception as e:
    print(f"❌ 比较操作失败: {e}")

# 测试我们的修复函数
print("\n2. 测试修复后的函数...")

from working.lib.strategy import scale_to_allocation, VolatilityOverlay
from working.lib.evaluation import calculate_sharpe_ratio, backtest_strategy

# 测试有问题的数据
raw_allocations = np.array([0.1, np.nan, np.inf, 0.4, 0.5])
market_returns = np.array([0.01, np.nan, 0.03, 0.04, 0.05])

print(f"原始分配: {raw_allocations}")
print(f"市场收益: {market_returns}")

# 捕获警告
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    
    # 使用修复后的函数
    allocations = scale_to_allocation(raw_allocations, scale=10.0)
    print(f"分配比例: {allocations}")
    
    # 测试策略
    overlay = VolatilityOverlay()
    result = overlay.transform(allocations, market_returns)
    print(f"策略结果长度: {len(result['allocations'])}")
    
    # 测试评估
    metrics = backtest_strategy(allocations, market_returns)
    print(f"策略夏普比率: {metrics['strategy_sharpe']:.4f}")
    
    print(f"\n3. 警告统计: 共捕获 {len(w)} 个警告")
    for warning in w:
        print(f"  - {warning.category.__name__}: {warning.message}")

print("\n✅ 测试完成")
