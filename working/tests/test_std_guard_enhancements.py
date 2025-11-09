"""
std_guard增强功能测试
测试自适应阈值、多模型回退和噪声注入功能
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch
import sys
import os

# 添加working目录到路径
working_dir = os.path.dirname(os.path.dirname(__file__))
if working_dir not in sys.path:
    sys.path.insert(0, working_dir)

try:
    from lib.models import HullModel
    from lib.data import load_train_data, load_test_data
    from lib.features import FeaturePipeline
    from lib.env import detect_run_environment, get_data_paths
except ImportError as e:
    print(f"导入错误: {e}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path}")
    raise


class TestHullModelEnhancements:
    """测试HullModel的增强功能"""

    def setup_method(self):
        """设置测试数据"""
        np.random.seed(42)
        self.train_data = pd.DataFrame({
            'date_id': range(200),
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200),
            'feature3': np.random.randn(200),
            'forward_returns': np.random.randn(200) * 0.01,
        })
        
        self.test_data = self.train_data.head(10).copy()
        self.target = self.train_data['forward_returns'].fillna(0)

    def test_noise_injection(self):
        """测试噪声注入功能"""
        # 设置固定随机种子以确保测试稳定性
        np.random.seed(123)
        
        model = HullModel('baseline', {})
        model.fit(self.train_data, self.target)
        
        # 无噪声预测
        preds_no_noise = model.predict(self.test_data, noise_std=0.0)
        
        # 有噪声预测
        preds_with_noise = model.predict(self.test_data, noise_std=0.01)
        
        # 验证噪声注入了
        assert not np.array_equal(preds_no_noise, preds_with_noise), \
            "噪声注入未生效"
        
        # 验证形状一致
        assert preds_with_noise.shape == preds_no_noise.shape, \
            "噪声注入改变了预测形状"
        
        # 验证噪声确实存在
        noise_diff = np.abs(preds_with_noise - preds_no_noise)
        assert noise_diff.max() > 0, "噪声注入后应该与原预测有差异"

    def test_clip_functionality(self):
        """测试裁剪功能"""
        # 分离特征和目标
        feature_cols = ['date_id', 'feature1', 'feature2', 'feature3']
        X_train = self.train_data[feature_cols]
        y_train = self.target
        
        # 小的测试数据，使用相同的特征列
        X_test = X_train.head(3)
        
        model = HullModel('baseline', {})
        model.fit(X_train, y_train)
        
        # 无裁剪预测
        preds_no_clip = model.predict(X_test, clip=False)
        
        # 有裁剪预测
        preds_with_clip = model.predict(X_test, clip=True)
        
        # 验证裁剪功能
        assert np.all(preds_with_clip >= 0), "裁剪后预测值不应小于0"
        assert np.all(preds_with_clip <= 2), "裁剪后预测值不应大于2"
        assert preds_with_clip.shape == preds_no_clip.shape, "裁剪不应改变形状"


class TestStdGuardAdaptation:
    """测试std_guard自适应功能"""

    def test_prediction_variability_detection(self):
        """测试预测变异性检测"""
        # 创建低变异性预测
        low_variability_preds = np.array([0.001] * 10)
        
        # 创建高变异性预测
        high_variability_preds = np.array([0.001, 0.002, 0.003, 0.001, 0.002, 0.003, 0.001, 0.002, 0.003, 0.001])
        
        low_std = float(np.std(low_variability_preds))
        high_std = float(np.std(high_variability_preds))
        
        assert low_std < high_std, f"低变异性std应小于高变异性: {low_std} vs {high_std}"
        assert high_std > 0.0005, f"高变异性std应大于阈值: {high_std}"

    def test_adaptive_threshold_calculation(self):
        """测试自适应阈值计算逻辑"""
        # 模拟不同的训练std情况
        train_stds = [0.00001, 0.0001, 0.001, 0.01]
        default_threshold = 0.15
        
        for train_std in train_stds:
            if train_std > 0:
                target_std_ratio = max(0.1, min(1.0, train_std / 0.01))
                adaptive_threshold = max(0.001, min(default_threshold, train_std * target_std_ratio))
            else:
                adaptive_threshold = max(0.001, default_threshold * 0.1)
            
            # 验证自适应阈值的合理性
            assert adaptive_threshold > 0, f"自适应阈值应为正数: {adaptive_threshold}"
            assert adaptive_threshold <= default_threshold, \
                f"自适应阈值不应超过默认阈值: {adaptive_threshold} vs {default_threshold}"

    def test_multi_model_enhancement(self):
        """测试多模型增强逻辑"""
        # 模拟std_guard触发场景
        low_variability_preds = np.array([0.001] * 10)
        high_variability_preds = np.array([0.002, 0.003, 0.001, 0.002, 0.003] * 2)
        
        # 验证噪声注入可以增加变异性
        noise_scale = 0.001
        noise = np.random.normal(0, noise_scale, size=low_variability_preds.shape)
        enhanced_preds = low_variability_preds + noise
        enhanced_std = float(np.std(enhanced_preds))
        original_std = float(np.std(low_variability_preds))
        
        assert enhanced_std > original_std, \
            f"噪声注入应增加变异性: {enhanced_std} vs {original_std}"


class TestModelIntegration:
    """测试模型集成功能"""

    def test_real_data_prediction_variability(self):
        """测试真实数据的预测变异性"""
        env = detect_run_environment()
        data_paths = get_data_paths(env)
        
        train_data = load_train_data(data_paths)
        test_data = load_test_data(data_paths)
        
        # 使用增强的FeaturePipeline
        pipeline = FeaturePipeline(extra_group_stats=True)
        features = pipeline.fit_transform(train_data)
        test_features = pipeline.transform(test_data)
        
        target = train_data['forward_returns'].fillna(train_data['forward_returns'].median())
        
        # 测试baseline模型
        baseline_model = HullModel('baseline', {})
        baseline_model.fit(features, target)
        baseline_preds = baseline_model.predict(test_features, clip=False)
        
        # 验证baseline模型有合理的预测变异性
        baseline_std = float(np.std(baseline_preds))
        assert baseline_std > 0, f"baseline预测应有一定变异性: {baseline_std}"
        
        # 验证预测值在合理范围内
        assert np.isfinite(baseline_preds).all(), "所有预测值应为有限数"
        assert np.isfinite(baseline_std), "std计算应得到有限结果"

    def test_ensemble_model_functionality(self):
        """测试集成模型功能"""
        env = detect_run_environment()
        data_paths = get_data_paths(env)
        
        train_data = load_train_data(data_paths)
        test_data = load_test_data(data_paths)
        
        pipeline = FeaturePipeline(extra_group_stats=True)
        features = pipeline.fit_transform(train_data)
        test_features = pipeline.transform(test_data)
        
        target = train_data['forward_returns'].fillna(train_data['forward_returns'].median())
        
        try:
            # 尝试使用集成模型
            ensemble_model = HullModel('ensemble', {
                'weights': {'lightgbm': 1.0, 'xgboost': 1.0, 'catboost': 1.0}
            })
            ensemble_model.fit(features, target)
            ensemble_preds = ensemble_model.predict(test_features, clip=False)
            
            # 验证集成模型能产生预测
            assert ensemble_preds.shape[0] == test_features.shape[0], \
                f"集成模型预测数量应匹配测试数据: {ensemble_preds.shape[0]} vs {test_features.shape[0]}"
            
        except ImportError:
            # 如果模型库未安装，跳过测试
            pytest.skip("LightGBM/XGBoost/CatBoost未安装")


if __name__ == "__main__":
    # 运行基本测试
    model_test = TestHullModelEnhancements()
    model_test.setup_method()
    model_test.test_noise_injection()
    model_test.test_clip_functionality()
    
    std_test = TestStdGuardAdaptation()
    std_test.test_prediction_variability_detection()
    std_test.test_adaptive_threshold_calculation()
    std_test.test_multi_model_enhancement()
    
    print("✅ std_guard增强功能测试通过")
