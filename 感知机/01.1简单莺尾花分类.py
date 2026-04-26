import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# ============================
# 1. 环境与数据准备
# ============================

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 解决中文显示（Mac用'Arial Unicode MS'）
plt.rcParams['axes.unicode_minus'] = False

print("=" * 40)
print("感知机实战 - 鸢尾花二分类")
print("任务：用花萼特征区分山鸢尾(0)和变色鸢尾(1)")
print("=" * 40)

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 只取前两类（线性可分）
idx = np.where((y == 0) | (y == 1))
X = X[idx]
y = y[idx]

# 只取前两个特征（方便可视化）
X = X[:, :2]

print(f"\n数据形状: {X.shape}")
print(f"类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"训练集: {X_train.shape[0]} 样本 | 测试集: {X_test.shape[0]} 样本")

# ============================
# 2. 感知机模型（原始形式）
# ============================
class Perceptron:
    """原始形式感知机，用于二分类（标签需为-1和+1）"""
    
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.errors = []  # 每轮的误分类数

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 初始化权重和偏置
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 标签转换：0→-1，1→+1
        y_conv = np.where(y == 1, 1, -1)

        for epoch in range(self.n_iters):
            n_errors = 0
            for idx, x_i in enumerate(X):
                # 误分类条件：y_i * (w·x_i + b) ≤ 0
                if y_conv[idx] * (np.dot(x_i, self.weights) + self.bias) <= 0:
                    # 核心更新公式：w := w + η * y_i * x_i
                    self.weights += self.lr * y_conv[idx] * x_i
                    self.bias    += self.lr * y_conv[idx]
                    n_errors += 1
            
            self.errors.append(n_errors)
            if n_errors == 0:
                print(f"✅ 第 {epoch+1} 轮迭代，误分类点数降为 0，模型收敛！")
                break
        
        if self.errors[-1] > 0:
            print(f"⚠️ 达到最大迭代次数 {self.n_iters}，最终误分类点数：{self.errors[-1]}")
        
        return self

    def predict(self, X):
        # 符号函数：>0 → +1，≤0 → -1
        linear = np.dot(X, self.weights) + self.bias
        return np.where(linear > 0, 1, -1)

# ============================
# 3. 训练与评估
# ============================
print("\n开始训练感知机模型...")
model = Perceptron(learning_rate=0.1, n_iters=1000)
model.fit(X_train, y_train)

# 预测并转换回原始标签
y_pred_conv = model.predict(X_test)
y_pred = np.where(y_pred_conv == 1, 1, 0)  # +1→1，-1→0

accuracy = np.mean(y_pred == y_test) * 100
print(f"\n测试集准确率: {accuracy:.2f}%")
print(f"真实标签: {y_test.tolist()}")
print(f"预测标签: {y_pred.tolist()}")
print(f"\n最终权重: {model.weights}")
print(f"最终偏置: {model.bias:.4f}")

# ============================
# 4. 可视化
# ============================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ---- 图1：误分类下降曲线 ----
ax = axes[0]
ax.plot(range(1, len(model.errors)+1), model.errors, marker='o', color='green')
ax.set_xlabel('迭代次数（Epoch）')
ax.set_ylabel('误分类点数')
ax.set_title('感知机训练收敛曲线')
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=-0.5)

# ---- 图2：决策边界 ----
ax = axes[1]
# 画训练集
ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 
           c='blue', marker='o', edgecolors='k', s=60, label='训练集-山鸢尾')
ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
           c='red', marker='o', edgecolors='k', s=60, label='训练集-变色鸢尾')
# 画测试集
ax.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], 
           c='blue', marker='x', s=100, linewidths=2, label='测试集-山鸢尾')
ax.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], 
           c='red', marker='x', s=100, linewidths=2, label='测试集-变色鸢尾')

# 绘制分类面
x_min, x_max = X[:, 0].min()-0.3, X[:, 0].max()+0.3
y_min, y_max = X[:, 1].min()-0.3, X[:, 1].max()+0.3
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
ax.contour(xx, yy, Z, colors='k', linewidths=1.5)
ax.set_xlabel('花萼长度 (cm)')
ax.set_ylabel('花萼宽度 (cm)')
ax.set_title(f'感知机决策边界（准确率: {accuracy:.1f}%）')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ---- 额外的关键发现 ----
print("\n" + "=" * 40)
print("实验结论")
print("=" * 40)
print(f"1. 感知机在 {len(model.errors)} 次迭代内收敛。")
print(f"2. 从误分类曲线可以看出，模型以‘错误驱动’的方式快速学习。")
print(f"3. 决策边界是一条清晰的直线，完美分割两类，验证了线性可分性。")