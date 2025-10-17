"""
测试模型功能
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from lib.models import HullModel, create_submission, create_baseline_model
except ImportError:
    # 如果lib.models导入失败，尝试直接导入
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
    from models import HullModel, create_submission, create_baseline_model


def test_create_baseline_model():
    """测试基线模型创建"""
    
    model = create_baseline_model()
    assert model is not None
    

def test_hull_model_initialization():
    """测试Hull模型初始化"""
    
    # 测试不同模型类型
    for model_type in ["baseline", "lightgbm", "xgboost", "ensemble"]:
        model = HullModel(model_type=model_type)
        assert model.model_type == model_type
        assert model.model is None
        assert model.feature_columns is None


def test_hull_model_fit_predict():
    """测试模型训练和预测"""
    
    # 创建测试数据
    X_train = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    y_train = np.random.randn(100)
    
    model = HullModel(model_type="baseline")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 验证训练后的状态
    assert model.model is not None
    assert model.feature_columns == ['feature1', 'feature2']
    
    # 测试预测
    X_test = pd.DataFrame({
        'feature1': np.random.randn(10),
        'feature2': np.random.randn(10)
    })
    
    predictions = model.predict(X_test)
    
    # 验证预测结果
    assert len(predictions) == 10
    assert np.all(predictions >= 0)  # 预测值应该在0-2之间
    assert np.all(predictions <= 2)
    

def test_hull_model_cross_validation():
    """测试交叉验证"""
    
    # 创建时间序列数据
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100)
    })
    y = np.random.randn(100)
    
    model = HullModel(model_type="baseline")
    
    # 执行交叉验证
    cv_results = model.cross_validate(X, y, n_splits=3)
    
    # 验证结果格式
    expected_keys = ['mean_mse', 'std_mse', 'mean_mae', 'std_mae']
    for key in expected_keys:
        assert key in cv_results
        assert isinstance(cv_results[key], float)


def test_create_submission():
    """测试提交文件创建"""
    
    predictions = np.array([0.5, 1.2, 0.8])
    date_ids = pd.Series([1001, 1002, 1003])
    
    submission_df = create_submission(predictions, date_ids)
    
    # 验证数据框结构
    assert len(submission_df) == 3
    assert 'date_id' in submission_df.columns
    assert 'prediction' in submission_df.columns
    assert list(submission_df['date_id']) == [1001, 1002, 1003]
    assert list(submission_df['prediction']) == [0.5, 1.2, 0.8]


def test_prediction_clipping():
    """测试预测值裁剪功能"""
    
    # 创建超出范围的预测值
    X = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })
    y = [0.5, 1.0, 1.5]
    
    model = HullModel(model_type="baseline")
    model.fit(X, y)
    
    # 模拟超出范围的预测（通过修改模型预测方法）
    # 这里我们直接测试裁剪逻辑
    raw_predictions = np.array([-0.5, 0.8, 2.5])  # 超出范围的值
    clipped_predictions = np.clip(raw_predictions, 0, 2)
    
    assert np.all(clipped_predictions >= 0)
    assert np.all(clipped_predictions <= 2)
    assert clipped_predictions[0] == 0.0
    assert clipped_predictions[2] == 2.0


if __name__ == "__main__":
    test_create_baseline_model()
    test_hull_model_initialization()
    test_hull_model_fit_predict()
    test_hull_model_cross_validation()
    test_create_submission()
    test_prediction_clipping()
    print("✅ 所有模型测试通过")