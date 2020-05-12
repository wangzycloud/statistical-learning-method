"""
CART+最小二乘法构建CART回归树
https://blog.csdn.net/slx_share/article/details/79992846?depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-5&utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-5
"""
import numpy as np

class node:
    # 定义二叉树结点，结点内容包括：特征、特征对应的值、类别标记lab、左右子树指针
    def __init__(self, fea=-1, val=None, lab=None, right=None, left=None):
        self.fea = fea  # 特征
        self.val = val  # 特征对应的值
        self.lab = lab  # 叶结点标记
        self.right = right
        self.left = left

class CART_REG:
    # 定义CART回归树
    def __init__(self, epsilon=0.1, min_sample=10):
        self.epsilon = epsilon
        # 叶结点含有的最少样本数
        self.min_sample = min_sample
        self.tree = None

    def err(self, y_data):
        # 用平方误差最小化的准则求解每个单元上的最优输出值
        # 子数据集的输出变量y与均值的差的平方和
        return y_data.var() * y_data.shape[0]

    def leaf(self, y_data):
        # 叶节点取值，为子数据集输出y的均值
        return y_data.mean()

    def split(self, fea, val, x_data):
        # 公式(5.18) 根据某个特征，以及特征下的某个取值，将数据集进行切分
        set1_inds = np.where(x_data[:, fea] <= val)[0]
        set2_inds = list(set(range(x_data.shape[0]))-set(set1_inds))
        return set1_inds, set2_inds

    def getBestSplit(self, x_data, y_data):
        # 求最优切分点
        best_err = self.err(y_data)
        best_split = None
        subsets_inds = None
        for fea in range(x_data.shape[1]):
            for val in x_data[:, fea]:
                set1_inds, set2_inds = self.split(fea, val, x_data)
                # 若切分后某个子集大小不足2，则不切分
                if len(set1_inds) < 2 or len(set2_inds) < 2:
                    continue
                now_err = self.err(y_data[set1_inds]) + self.err(y_data[set2_inds])
                if now_err < best_err:
                    best_err = now_err
                    best_split = (fea, val)
                    subsets_inds = (set1_inds, set2_inds)
        return best_err, best_split, subsets_inds

    def buildTree(self, x_data, y_data):
        # 递归构建二叉树
        if y_data.shape[0] < self.min_sample:
            return node(lab=self.leaf(y_data))
        best_err, best_split, subsets_inds = self.getBestSplit(x_data, y_data)
        if subsets_inds is None:
            return node(lab=self.leaf(y_data))
        if best_err < self.epsilon:
            return node(lab=self.leaf(y_data))
        else:
            left = self.buildTree(x_data[subsets_inds[0]], y_data[subsets_inds[0]])
            right = self.buildTree(x_data[subsets_inds[1]], y_data[subsets_inds[1]])
            return node(fea=best_split[0], val=best_split[1], right=right, left=left)

    def fit(self, x_data, y_data):
        self.tree = self.buildTree(x_data, y_data)
        return

    def predict(self, x):
        # 对输入变量进行预测
        def helper(x, tree):
            if tree.lab is not None:
                return tree.lab
            else:
                # 递归向下一层进行遍历。由根结点的特征情况(此分支结点选择了哪个特征进行切分)
                # 判断预测数据x的该特征取值，根据取值情况进行不同的分支，进行下沉
                # 下一个要进行切分的根结点，表示为branch
                if x[tree.fea] <= tree.val:
                    branch = tree.left
                else:
                    branch = tree.right
                return helper(x, branch)

        return helper(x, self.tree)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x_data_raw = np.linspace(-3, 3, 50)
    np.random.shuffle(x_data_raw)
    y_data = np.sin(x_data_raw)
    x_data = np.transpose([x_data_raw])
    y_data = y_data + 0.1 * np.random.randn(y_data.shape[0])

    clf = CART_REG(epsilon=1e-4, min_sample=1)
    clf.fit(x_data, y_data)

    lab = []
    for i in range(x_data.shape[0]):
        lab.append(clf.predict(x_data[i]))
    p1 = plt.scatter(x_data_raw, y_data)
    p2 = plt.scatter(x_data_raw, lab, marker='*')
    plt.legend([p1,p2],['real','pred'],loc='upper left')
    plt.show()