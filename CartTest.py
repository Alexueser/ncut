import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import graphviz
import pydotplus

# 生成随机数据
X, y = make_classification(n_samples=100, n_features=4, n_informative=3,
                           n_redundant=1, n_classes=2, random_state=42)

# 将数据转换为DataFrame格式
df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])
df['Target'] = y

# 将数据集划分为训练集和测试集
train, test = train_test_split(df, test_size=0.3, random_state=42)

# 选择特征和目标列
features = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
target = 'Target'

# 创建决策树分类器对象
dt = DecisionTreeClassifier(criterion='entropy')

# 使用训练数据训练决策树分类器
dt.fit(train[features], train[target])

# 可视化决策树
dot_data = export_graphviz(dt, out_file=None,
                           feature_names=features,
                           class_names=['0', '1'],
                           filled=True, rounded=True,
                           special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("dtr_white_background.pdf")

