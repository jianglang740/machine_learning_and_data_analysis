# 导入数据分析库
import pandas as pd
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt
# 导入线性回归模型
from sklearn.linear_model import LinearRegression
# 导入数据集划分工具（你已掌握）
from sklearn.model_selection import train_test_split
# 设置中文字体（避免绘图中文乱码）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取CSV数据（路径已匹配你上传的文件）
df = pd.read_csv('/Users/clinking/开发/线性回归与分类数据集/house_mini.csv')

# 2. 查看数据前5行（了解数据长什么样）
print("数据前5行：")
print(df.head())

# 3. 查看数据基本信息（列名、数据类型、是否有缺失值）
print("\n数据基本信息：")
print(df.info())

# 4. 查看数据统计描述（均值、标准差、最值等，帮你快速了解数据范围）
print("\n数据统计描述：")
print(df.describe())

# 1. 选择特征变量X（多个特征，体现“多元”回归）
# 这里选3个最直观的房价影响因素，你也可以根据理解调整
X = df[['sqft_living', 'bedrooms', 'bathrooms']]

# 2. 选择目标变量y（我们要预测的房价）
y = df['price']

# 查看选择后的X和y形状（确保数据行数一致）
print(f"\n特征变量X的形状：{X.shape}（行数：{X.shape[0]}，列数：{X.shape[1]}）")
print(f"目标变量y的形状：{y.shape}（行数：{y.shape[0]}）")

# 1. 检查每列的缺失值数量
print("\n各列缺失值数量：")
print(X.isnull().sum())  # 检查特征变量X的缺失值
print(f"目标变量y缺失值数量：{y.isnull().sum()}")

# 2. 删除特征变量X中的缺失值行（如果有的话）
X_clean = X.dropna()  # dropna()：删除包含空值的行

# 3. 确保目标变量y和X_clean行数一致（避免数据不匹配）
# 用X_clean的索引筛选y，保证一一对应
y_clean = y[X_clean.index]

# 验证处理后的数据形状
print(f"\n处理后X的形状：{X_clean.shape}")
print(f"处理后y的形状：{y_clean.shape}")

# 划分训练集（80%）和测试集（20%）
# random_state=42：固定随机种子，确保每次运行结果一致
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=42
)

# 查看划分后的数据量
print(f"\n训练集：X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
print(f"测试集：X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")

# 1. 创建多元线性回归模型实例
model = LinearRegression()

# 2. 用训练集训练模型（核心步骤：让模型学习特征和房价的关系）
model.fit(X_train, y_train)

# 3. 查看模型参数（理解模型的“预测公式”）
# 多元线性回归公式：price = β0 + β1*sqft_living + β2*bedrooms + β3*bathrooms
print("\n多元线性回归模型参数：")
print(f"截距 β0：{model.intercept_:.2f}（当所有特征为0时，房价的基准值）")
# 遍历特征和对应的系数，解释每个特征对房价的影响
for feature, coef in zip(X.columns, model.coef_):
    print(f"特征 {feature} 的系数 β：{coef:.2f}（该特征每增加1单位，房价平均变化{coef:.2f}美元）")

# 1. 用测试集做预测（让模型对“没见过”的数据做判断）
y_pred = model.predict(X_test)

# 2. 查看前10个预测值和真实值（直观对比）
print("\n测试集前10个预测值与真实值对比：")
comparison = pd.DataFrame({
    '真实房价': y_test.values[:10],
    '预测房价': y_pred[:10],
    '误差': y_pred[:10] - y_test.values[:10]
})
print(comparison.round(2))

# 计算R²分数
from sklearn.metrics import r2_score, mean_squared_error  # 导入评估指标

r2 = r2_score(y_test, y_pred)
# 计算MSE（均方误差）
mse = mean_squared_error(y_test, y_pred)

print(f"\n模型评估结果：")
print(f"R² 分数：{r2:.4f}（越接近1越好，说明模型能解释{r2*100:.2f}%的房价变化）")
print(f"MSE（均方误差）：{mse:.2f}（数值越小，预测误差越小）")

# 绘制“真实值vs预测值”散点图
plt.figure(figsize=(10, 6))  # 设置图的大小
# 散点图：x轴是真实值，y轴是预测值
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='真实值vs预测值')
# 理想预测线：y=x（如果预测完全准确，所有点都会落在这条线上）
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='理想预测线（y=x）')
# 添加标签和标题
plt.xlabel('真实房价（美元）', fontsize=12)
plt.ylabel('预测房价（美元）', fontsize=12)
plt.title('多元线性回归：房价真实值vs预测值', fontsize=14)
plt.legend()  # 显示图例
plt.grid(True, alpha=0.3)  # 添加网格线，方便查看
plt.show()

# 绘制“误差分布”直方图（看误差是否接近正态分布，判断模型是否稳定）
plt.figure(figsize=(10, 6))
error = y_pred - y_test  # 计算每个预测的误差
plt.hist(error, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('预测误差（美元）', fontsize=12)
plt.ylabel('频次', fontsize=12)
plt.title('房价预测误差分布', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()