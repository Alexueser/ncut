import numpy as np
import math

# 定义一个节点类
class Node:
    def __init__(self, is_leaf=False, label=None, feature=None):
        self.is_leaf = is_leaf  # 是否是叶子节点
        self.label = label  # 叶子节点的类别标签
        self.feature = feature  # 分裂特征的索引
        self.children = {}  # 子节点字典，格式为{特征取值：子节点}

# 定义一个ID3算法类
class ID3:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth  # 最大深度
        self.min_samples_split = min_samples_split  # 最小分裂样本数

    # 计算数据集的信息熵
    def entropy(self, y):
        unique_labels = np.unique(y)  # 获取标签类别
        entropy = 0
        for label in unique_labels:
            prob = len(y[y == label]) / len(y)  # 计算每个类别的概率
            entropy -= prob * math.log2(prob)  # 累加信息熵
        return entropy

    # 计算数据集的信息增益
    def information_gain(self, X, y, feature):
        entropy_parent = self.entropy(y)  # 计算父节点的信息熵
        unique_values = np.unique(X[:, feature])  # 获取特征的取值
        entropy_children = 0
        for value in unique_values:
            child_indices = np.where(X[:, feature] == value)[0]  # 获取特征取值为value的样本索引
            child_entropy = self.entropy(y[child_indices])  # 计算子节点的信息熵
            entropy_children += len(child_indices) / len(y) * child_entropy  # 加权累加子节点的信息熵
        information_gain = entropy_parent - entropy_children  # 计算信息增益
        return information_gain

    # 选择最佳特征
    def select_best_feature(self, X, y):
        best_feature = None
        best_information_gain = -1
        for feature in range(X.shape[1]):
            information_gain = self.information_gain(X, y, feature)  # 计算特征的信息增益
            if information_gain > best_information_gain:
                best_feature = feature  # 更新最佳特征
                best_information_gain = information_gain  # 更新最大信息增益
        return best_feature

    # 构建决策树
    def build_tree(self, X, y, depth=0):
        if depth == self.max_depth:  # 判断是否达到最大深度
            return Node(is_leaf=True, label=np.bincount(y).argmax())  # 返回叶子节点
        if len(X) < self.min_samples_split:  # 判断是否达到最小分裂样本数
            return Node(is_leaf=True, label=np.bincount(y).argmax())  # 返回叶子节点
        if np.unique(y).shape[0] == 1:  # 判断是否所有样本都属于同一类别
            return Node(is_leaf=True, label=y[0])  # 返回叶子节点
        best_feature = self.select_best_feature(X, y)  # 选择最佳特征
        if best_feature is None:  # 如果无法选择最佳特征，则返回叶子节点
            return Node(is_leaf=True, label=np.bincount(y).argmax())

        node = Node(feature=best_feature)  # 创建节点
        unique_values = np.unique(X[:, best_feature])  # 获取最佳特征的取值
        for value in unique_values:
            child_indices = np.where(X[:, best_feature] == value)[0]  # 获取特征取值为value的样本索引
            child_X = X[child_indices]  # 获取子节点的特征矩阵
            child_y = y[child_indices]  # 获取子节点的标签向量
            child_node = self.build_tree(child_X, child_y, depth=depth + 1)  # 递归构建子节点
            node.children[value] = child_node  # 添加子节点
        return node

    # 预测单个样本的类别
    def predict_one(self, x, node):
        if node.is_leaf:  # 判断是否为叶子节点
            return node.label  # 返回类别标签
        value = x[node.feature]  # 获取样本在节点特征上的取值
        if value not in node.children:  # 如果取值未出现过，返回父节点的类别标签
            return node.label
        child_node = node.children[value]  # 获取对应子节点
        return self.predict_one(x, child_node)  # 递归预测子节点

    # 预测多个样本的类别
    def predict(self, X, tree):
        return [self.predict_one(x, tree) for x in X]

    # 计算预测准确率
    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    # 可视化决策树
    def visualize(self, node, depth=0):
        if node.is_leaf:  # 如果是叶子节点，直接输出类别标签
            print(" " * depth, "类别标签:", node.label)
            return
        print(" " * depth, "分裂特征:", node.feature)
        for value, child_node in node.children.items():
            print(" " * depth, "特征取值:", value)
            self.visualize(child_node, depth=depth+1)

# 生成随机数据
np.random.seed(0)
X = np.random.randint(2, size=(100, 5))
y = np.random.randint(2, size=(100,))

# 构建决策树
id3 = ID3(max_depth=3)
tree = id3.build_tree(X, y)

# 可视化决策树
id3.visualize(tree)

# 预测类别
y_pred = id3.predict(X, tree)

# 计算准确率
accuracy = id3.accuracy(y, y_pred)
print("预测准确率:", accuracy)



