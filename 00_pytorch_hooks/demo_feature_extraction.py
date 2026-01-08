"""
中间层特征提取实战

演示使用 Forward Hook 从神经网络中提取中间层特征：
1. 从自定义 CNN 提取各层特征图
2. 特征图的形状和统计信息
3. 使用闭包工厂批量注册 hook

这是 Hook 最常见的实际应用之一，广泛用于：
- 特征可视化
- 迁移学习中的特征复用
- 模型可解释性分析

参考: https://pytorch.org/docs/stable/notes/modules.html#module-hooks
"""

import torch
import torch.nn as nn


# ============================================================
# 定义一个简单的 CNN
# ============================================================
class SimpleCNN(nn.Module):
    """简单的 CNN，用于演示特征提取"""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # conv1
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 16x16

            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # conv2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 8x8

            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # conv3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),                  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# 特征提取器类
# ============================================================
class FeatureExtractor:
    """使用 Hook 从模型中提取指定层的特征"""

    def __init__(self, model, layer_names):
        """
        Args:
            model: PyTorch 模型
            layer_names: 要提取特征的层名列表
        """
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        """为指定的层注册 forward hook"""
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                # 使用闭包捕获层名
                handle = module.register_forward_hook(self._make_hook(name))
                self._handles.append(handle)

    def _make_hook(self, name):
        """创建 hook 函数的工厂方法"""
        def hook(module, args, output):
            # 使用 detach() 避免内存泄漏
            self.features[name] = output.detach()
        return hook

    def __call__(self, x):
        """前向传播并返回特征的副本"""
        self.features.clear()
        output = self.model(x)
        # 返回特征的副本，避免后续调用覆盖
        return output, {k: v.clone() for k, v in self.features.items()}

    def remove_hooks(self):
        """移除所有 hook"""
        for h in self._handles:
            h.remove()
        self._handles.clear()


def demo_feature_extraction():
    """演示从 CNN 提取中间层特征"""
    print("=" * 60)
    print("Demo 1: 从 CNN 提取中间层特征")
    print("=" * 60)

    model = SimpleCNN(num_classes=10)
    model.eval()

    # 查看模型结构
    print("\n模型结构 (named_modules):")
    for name, module in model.named_modules():
        if name and '.' not in name:
            print(f"  {name}: {module.__class__.__name__}")
        elif name.count('.') == 1:
            print(f"    {name}: {module.__class__.__name__}")

    # 指定要提取特征的层
    target_layers = [
        'features.0',   # 第一个 Conv2d
        'features.4',   # 第二个 Conv2d
        'features.8',   # 第三个 Conv2d
        'classifier.1', # 全连接层
    ]

    extractor = FeatureExtractor(model, target_layers)

    # 创建输入 (batch_size=4, channels=3, height=32, width=32)
    x = torch.randn(4, 3, 32, 32)

    output, features = extractor(x)

    print(f"\n输入 shape: {x.shape}")
    print(f"输出 shape: {output.shape}")
    print("\n提取的中间层特征:")
    print(f"{'层名':>20s} | {'Shape':>25s} | {'均值':>10s} | {'标准差':>10s}")
    print("-" * 75)
    for name in target_layers:
        feat = features[name]
        print(f"{name:>20s} | {str(feat.shape):>25s} | "
              f"{feat.mean():>10.4f} | {feat.std():>10.4f}")

    extractor.remove_hooks()
    print()


def demo_activation_statistics():
    """演示收集所有层的激活值统计信息"""
    print("=" * 60)
    print("Demo 2: 收集所有层激活值统计信息")
    print("=" * 60)

    model = SimpleCNN(num_classes=10)
    model.eval()

    activation_stats = {}

    def stats_hook(name):
        def hook(module, args, output):
            if isinstance(output, torch.Tensor):
                activation_stats[name] = {
                    'shape': tuple(output.shape),
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'zero_pct': (output == 0).float().mean().item() * 100,
                }
        return hook

    # 为所有有实际计算的层注册 hook
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d)):
            h = module.register_forward_hook(stats_hook(name))
            handles.append(h)

    # 前向传播
    x = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        _ = model(x)

    # 打印统计信息
    print(f"\n{'层名':>20s} | {'Shape':>20s} | {'均值':>8s} | {'标准差':>8s} | {'零值%':>6s}")
    print("-" * 75)
    for name, stats in activation_stats.items():
        print(f"{name:>20s} | {str(stats['shape']):>20s} | "
              f"{stats['mean']:>8.4f} | {stats['std']:>8.4f} | "
              f"{stats['zero_pct']:>5.1f}%")

    for h in handles:
        h.remove()
    print()


def demo_feature_comparison():
    """演示比较不同输入的特征差异"""
    print("=" * 60)
    print("Demo 3: 比较不同输入的特征相似度")
    print("=" * 60)

    model = SimpleCNN(num_classes=10)
    model.eval()

    # 提取最后一个卷积层的特征
    features_store = {}

    def capture_hook(module, args, output):
        features_store['feat'] = output.detach()

    handle = model.features[-2].register_forward_hook(capture_hook)  # ReLU before pool

    # 输入 1: 随机图像
    torch.manual_seed(0)
    img1 = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        model(img1)
    feat1 = features_store['feat'].flatten()

    # 输入 2: 相似图像 (加少量噪声)
    img2 = img1 + torch.randn_like(img1) * 0.1
    with torch.no_grad():
        model(img2)
    feat2 = features_store['feat'].flatten()

    # 输入 3: 完全不同的图像
    torch.manual_seed(999)
    img3 = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        model(img3)
    feat3 = features_store['feat'].flatten()

    # 计算余弦相似度
    cos_sim = nn.CosineSimilarity(dim=0)
    sim_1_2 = cos_sim(feat1, feat2).item()
    sim_1_3 = cos_sim(feat1, feat3).item()
    sim_2_3 = cos_sim(feat2, feat3).item()

    print(f"\n  图像 1 vs 图像 2 (加少量噪声): 余弦相似度 = {sim_1_2:.4f}")
    print(f"  图像 1 vs 图像 3 (完全不同):   余弦相似度 = {sim_1_3:.4f}")
    print(f"  图像 2 vs 图像 3 (完全不同):   余弦相似度 = {sim_2_3:.4f}")
    print("\n  结论: 相似图像的特征更接近 (相似度更高)")

    handle.remove()
    print()


def demo_multi_layer_features_for_loss():
    """演示提取多层特征用于 perceptual loss"""
    print("=" * 60)
    print("Demo 4: 多层特征提取 (感知损失场景)")
    print("=" * 60)

    model = SimpleCNN(num_classes=10)
    model.eval()

    # 提取多个层的特征
    conv_layers = ['features.0', 'features.4', 'features.8']
    extractor = FeatureExtractor(model, conv_layers)

    # 两张图片
    torch.manual_seed(42)
    img_a = torch.randn(1, 3, 32, 32)
    img_b = img_a + torch.randn_like(img_a) * 0.5

    with torch.no_grad():
        _, features_a = extractor(img_a)
        _, features_b = extractor(img_b)

    # 计算多层特征的 L2 距离 (类似 perceptual loss)
    print("\n  逐层特征距离 (L2):")
    total_loss = 0
    for name in conv_layers:
        feat_a = features_a[name]
        feat_b = features_b[name]
        layer_loss = nn.MSELoss()(feat_a, feat_b).item()
        total_loss += layer_loss
        print(f"    {name:>12s}: {layer_loss:.6f}")

    print(f"    {'总感知损失':>12s}: {total_loss:.6f}")

    extractor.remove_hooks()
    print()


if __name__ == '__main__':
    torch.manual_seed(42)
    demo_feature_extraction()
    demo_activation_statistics()
    demo_feature_comparison()
    demo_multi_layer_features_for_loss()
