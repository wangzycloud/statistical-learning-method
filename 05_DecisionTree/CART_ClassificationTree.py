"""
CART分类树，是一颗二叉树，以某个特征以及该特征对应的一个值为结点，故相对ID3算法，最大的不同就是特征可以使用多次
https://blog.csdn.net/slx_share/article/details/79992846?depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-5&utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-5
"""
import time
import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

def randomDATA(x_domain, y_domain, rate=0.3):
    '''
    划分训练集、测试集
    '''
    x_train, x_test, y_train, y_test = train_test_split(x_domain, y_domain, test_size=rate)
    return x_train, x_test, y_train, y_test

class Node:
    # 定义二叉树结点，结点内容包括：特征、特征对应的值、类别标记lab、左右子树指针
    def __init__(self, fea=-1, val=None, lab=None, right=None, left=None):
        self.fea = fea  # 特征
        self.val = val  # 特征对应的值
        self.lab = lab  # 叶结点标记
        self.right = right
        self.left = left

class CART_CLF:
    # 定义CART分类树
    def __init__(self, epsilon=1e-3, min_sample=1):
        self.epsilon = epsilon
        # 叶结点含有的最少样本数
        self.min_sample = min_sample  
        self.tree = None

    def getGini(self, y_data):
        # 计算基尼指数 Counter方法以字典形式，返回数据集y_data的统计数据，如{'1':30,'2':20}，表示数据1有30个，数据2有20个
        c = Counter(y_data)
        # 公式(5.24)
        GD = 1 - sum([(val / y_data.shape[0]) ** 2 for val in c.values()])
        return GD

    def getFeaGini(self, set1, set2):
        # 计算某个特征及相应特征值组成的切分结点的基尼指数
        # 注意，按照CART算法的思想，根据样本点对特征A=ai的测试为"是"、"否"，将D切分为D1=set1,D2=set2
        # 分别计算D1、D2的基尼指数，乘上相应权重(该特征是、否)，最后加和
        num = set1.shape[0] + set2.shape[0]
        # 公式(5.25)
        GDA = set1.shape[0] / num * self.getGini(set1) + set2.shape[0] / num * self.getGini(set2)
        return GDA

    def bestSplit(self, splits_set, x_data, y_data):
        '''
        该函数得到所有切分点的基尼指数，以字典形式存储。
        :param splits_set: 切分点数据集
        :param x_data: 训练数据
        :param y_data: 训练数据标签
        :return: 所有切分点的基尼指数，以字典形式存储。key为切分点(split)
                 split是一个元组，第一个元素为最优切分特征，第二个为该特征对应的最优切分值
        '''
        # 数据标签集的基尼指数
        pre_gini = self.getGini(y_data)
        # 切分点以及相应的样本点的索引。该算法采用样本点的索引来进行操作！
        # {key：(特征,特征值),value：样本点索引}
        subdata_inds = defaultdict(list)
        # 对于切分点集合中的每一个切分点，寻找属于该切分点的所有数据，记录其索引值
        for split in splits_set:
            for ind, sample in enumerate(x_data):
                # 特征n，取到切分点元组的值
                if sample[split[0]] == split[1]:
                    subdata_inds[split].append(ind)
        # 寻找最优切分点
        min_gini = 1
        best_split = None
        best_set = None
        # 对于每一个切分点，计算该特征条件下的基尼指数
        for split, set1_inds in subdata_inds.items():
            # 满足切分点的条件('是')，则为左子树
            set1 = y_data[set1_inds]
            # 不满足切分点的条件('否')，则为右子树。左子树索引的补集
            set2_inds = list(set(range(y_data.shape[0])) - set(set1_inds))
            set2 = y_data[set2_inds]
            # 判断是否满足叶结点最少样本数的条件
            if set1.shape[0] < 1 or set2.shape[0] < 1:
                continue
            # 得到该切分点下的基尼指数
            now_gini = self.getFeaGini(set1, set2)
            # 遍历过程只需要保存最小基尼指数对应的切分点
            if now_gini < min_gini:
                min_gini = now_gini
                best_split = split
                best_set = (set1_inds, set2_inds)
        # 若切分后基尼指数下降未超过阈值则停止切分
        if abs(pre_gini - min_gini) < self.epsilon:
            best_split = None
        return best_split, best_set, min_gini

    def buildTree(self, splits_set, x_data, y_data):
        # 数据集小于阈值直接设为叶结点
        if y_data.shape[0] < self.min_sample:
            # 当前数据中实例数最大的类别作为该结点的类别标记
            return Node(lab=Counter(y_data).most_common(1)[0][0])
        best_split, best_set, min_gini = self.bestSplit(splits_set, x_data, y_data)
        # 基尼指数小于阈值，则终止切分，设为叶结点
        if best_split is None:
            return Node(lab=Counter(y_data).most_common(1)[0][0])
        else:
            # 移除最优特征后，递归创建左右子树
            splits_set.remove(best_split)
            left = self.buildTree(splits_set, x_data[best_set[0]], y_data[best_set[0]])
            right = self.buildTree(splits_set, x_data[best_set[1]], y_data[best_set[1]])
            return Node(fea=best_split[0], val=best_split[1], right=right, left=left)

    def fit(self, x_data, y_data):
        # 训练模型，CART分类树与ID3最大的不同是，CART建立的是二叉树，每个结点是特征及其对应的某个值组成的元组
        # 特征可以多次使用
        splits_set = []
        # 根据特征每个可能的取值，建立切分点列表
        for fea in range(x_data.shape[1]):
            # 得到特征每个可能的取值
            unique_vals = np.unique(x_data[:, fea])
            if unique_vals.shape[0] < 2:
                continue
            # 若特征取值只有2个，则只有一个切分点，非此即彼
            elif unique_vals.shape[0] == 2:
                splits_set.append((fea, unique_vals[0]))
            # 若特征取值大于2个，则针对每个取值，定为切分点
            else:
                for val in unique_vals:
                    splits_set.append((fea, val))
        self.tree = self.buildTree(splits_set, x_data, y_data)
        return

    def predict(self, x):
        def helper(x, tree):
            if tree.lab is not None:  # 表明到达叶结点
                return tree.lab
            else:
                if x[tree.fea] == tree.val:  # "是" 返回左子树
                    branch = tree.left
                else:
                    branch = tree.right
                return helper(x, branch)

        return helper(x, self.tree)

    def disp_tree(self):
        # 打印树
        self.disp_helper(self.tree)
        return

    def disp_helper(self, current_Node):
        # 前序遍历
        print(current_Node.fea, current_Node.val, current_Node.lab)
        if current_Node.lab is not None:
            return
        self.disp_helper(current_Node.left)
        self.disp_helper(current_Node.right)
        return

if __name__ == '__main__':
    from sklearn.datasets import load_iris

    x_data = load_iris().data
    y_data = load_iris().target
    x_train, x_test, y_train, y_test = randomDATA(x_data, y_data)

    t1 = time.time()
    clf = CART_CLF()
    clf.fit(x_train, y_train)

    t2 = time.time()
    correct_num = 0
    for X, y in zip(x_test, y_test):
        if clf.predict(X) == y:
            correct_num += 1

    t3 = time.time()
    print('训练耗时：', t2 - t1)
    print('预测耗时：', t3 - t2)
    print('正确率为：{:.2f}'.format(correct_num / len(x_test)))