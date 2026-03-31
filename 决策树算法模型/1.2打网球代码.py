import pandas as pd
from sklearn.tree import DecisionTreeClassifier #决策树模型
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #评估指标
from sklearn.preprocessing import OrdinalEncoder #预处理、变换 —— 类别特征编码
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree #决策树可视化


# 数据（直接粘贴上面的 CSV 内容）
data = """
Outlook,Temperature,Humidity,Windy,Play
Sunny,Hot,High,False,No
Sunny,Hot,High,True,No
Overcast,Hot,High,False,Yes
Rainy,Mild,High,False,Yes
Rainy,Cool,Normal,False,Yes
Rainy,Cool,Normal,True,No
Overcast,Cool,Normal,True,Yes
Sunny,Mild,High,False,No
Sunny,Cool,Normal,False,Yes
Rainy,Mild,Normal,False,Yes
Sunny,Mild,Normal,True,Yes
Overcast,Mild,High,True,Yes
Overcast,Hot,Normal,False,Yes
Rainy,Mild,High,True,No
Sunny,Hot,Normal,False,Yes
Sunny,Hot,Normal,True,Yes
Overcast,Cool,High,False,Yes
Rainy,Cool,High,True,No
Rainy,Mild,Normal,True,Yes
Sunny,Cool,High,False,No
Overcast,Mild,Normal,False,Yes
Sunny,Mild,High,True,No
Rainy,Hot,Normal,False,Yes
Rainy,Hot,High,False,No
Overcast,Hot,High,True,Yes
Sunny,Cool,High,True,No
Overcast,Cool,Normal,False,Yes
Rainy,Mild,High,True,No
Sunny,Hot,High,False,No
Overcast,Hot,Normal,True,Yes
"""
'''
这行代码 from io import StringIO 引入的是 Python 标准库 io 模块中的 StringIO 类，
核心作用是在内存中模拟一个 “虚拟文件”，让字符串可以像文件一样被读写。
'''
from io import StringIO
df = pd.read_csv(StringIO(data.strip()))

print(df.head())


# 特征和标签
X = df.drop('Play', axis=1) #axis = 1 表示沿列操作，axis = 0 表示沿行操作
y = df['Play'] #基于目标计算基尼

# 类别特征编码（决策树可以处理数字编码），将特征映射成唯一的编码（数字、字符等）
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# 划分训练集和测试集（70% 训练，30% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)

# 训练决策树（基于基尼系数进行分裂，最大深度3）
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 预测与评估
y_pred = clf.predict(X_test)
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.2f}")

# 可选：查看特征重要性
print("\n特征重要性:")
for name, imp in zip(X.columns, clf.feature_importances_):
    print(f"{name}: {imp:.3f}")

plt.figure(figsize=(12, 8))
# 如果分类标签是中文或字符串，可以传入 class_names
plot_tree(clf, 
          feature_names = X.columns, 
          class_names=['No', 'Yes'],  # 注意顺序：按 clf.classes_ 排序，通常是 ['No','Yes']
          filled=True,                # 填充颜色
          rounded=True,               # 圆角框
          fontsize=10)
plt.title("Decision Tree for Play Tennis")
plt.show()
'''
在 sklearn 的决策树中（无论是分类树还是回归树）,feature_importances_ 的计算逻辑是：
每个特征的重要性 = 该特征在所有节点上带来的不纯度减少量的加权和
权重 = 该节点的样本数 / 总样本数
对于分类树（基尼系数或信息熵），不纯度减少量 = 父节点不纯度 - 子节点加权平均不纯度。
对于回归树(MSE),不纯度减少量 = MSE 减少量。
然后将所有特征的重要性归一化到 0~1 之间，总和为 1。
'''