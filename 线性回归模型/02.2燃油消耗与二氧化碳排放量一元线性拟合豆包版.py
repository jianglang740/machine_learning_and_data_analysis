#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:39:32 2026

@author: clinking
"""

# /usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Mon Mar 23 19:02:29 2026
# @author: clinking

# 库导入（新增评估工具）
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 解决中文显示（Mac用'Arial Unicode MS'）
plt.rcParams['axes.unicode_minus'] = False

# 数据提取
df = pd.read_csv("/Users/clinking/开发/线性回归与分类数据集/co2_mini.csv")

# 数据预处理
print("数据概览:")
print(df.head())
print(df.describe())
print(df.info())

# -------------------------- 1. 改进：标准划分训练集/测试集 --------------------------
# 提取特征X和标签y
X = df["consumption"].values.reshape(-1, 1)
y = df["co2"].values

# 8:2拆分，固定随机种子
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"\n训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")

# -------------------------- 2. 模型训练（仅用训练集） --------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# 提取模型参数
alpha = model.intercept_
beta = model.coef_[0]
print(f"\n模型参数:")
print(f"截距(alpha): {alpha:.4f}, 斜率(beta): {beta:.4f}")

# -------------------------- 3. 模型评估（核心检测环节） --------------------------
# 预测
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 计算指标
print("\n" + "="*50)
print("【训练集性能】")
print(f"R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")

print("\n【测试集性能】")
print(f"R²: {r2_score(y_test, y_pred_test):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
print("="*50)

# 5折交叉验证（进阶评估）
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"\n5折交叉验证R²: {cv_scores.round(4)}")
print(f"平均R²: {cv_scores.mean():.4f}, 标准差: {cv_scores.std():.4f}")

# -------------------------- 4. 可视化 --------------------------
# 图1：训练集+测试集拟合对比
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color="red", alpha=0.5, label="训练集数据")
plt.scatter(X_test, y_test, color="blue", alpha=0.5, label="测试集数据")
X_all = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_all_pred = model.predict(X_all)
plt.plot(X_all, y_all_pred, color="green", alpha=0.8, linewidth=2, 
         label=f"拟合线: y = {alpha:.3f} + {beta:.3f}x, R²(测试)={r2_score(y_test, y_pred_test):.3f}")

plt.xlabel("燃油消耗量")
plt.ylabel("二氧化碳排放量")
plt.title("燃油消耗与CO₂排放：训练集+测试集线性拟合")
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 图2：测试集残差图
residuals = y_test - y_pred_test
plt.figure(figsize=(10, 4))
plt.scatter(y_pred_test, residuals, color="purple", alpha=0.6)
plt.axhline(y=0, color="black", linestyle="--")
plt.xlabel("预测值")
plt.ylabel("残差（真实值-预测值）")
plt.title("测试集残差图（验证模型假设）")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()