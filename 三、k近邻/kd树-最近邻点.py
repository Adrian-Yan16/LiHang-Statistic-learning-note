import time
from random import random
# kd 树搜索
from collections import namedtuple
import math


class KNode:
    def __init__(self,dom_elt,split,left,right):
        self.dom_elt = dom_elt  # k 维向量节点（k 维空间中的样本点——结点值）
        self.split = split  # 进行分割维度的序号
        self.left = left
        self.right = right

class KdTree:
    def __init__(self,data):
        self.k = len(data[0])
        self.root = self.fit(0,data)
        # 定义一个namedtuple，用来存储最近坐标点，最近距离和已访问的节点数
        self.result = namedtuple('result_tuple','nearest_point neareast_distance')

    # 在第split维划分数据集创建 kdnode
    def fit(self,split,data_set):
        if not data_set:
            return None
        # key 的参数是一个函数，此函数只有一个参数且返回一个值进行比较
        # operator模块提供的 itemgetter 函数用于获取对象的哪些维的数据，参数为需要获取的数据在对象中的序号
        # data_set.sort(key=itemgetter(split))  按要进行分割的那一维数据排序
        data_set.sort(key=lambda x:x[split])
        split_pos = len(data_set) // 2
        median = data_set[split_pos]
        split_next = (split + 1) % self.k

        return KNode(median,split,
                     self.fit(split_next,data_set[:split_pos]),# 左子树
                     self.fit(split_next,data_set[split_pos + 1:]))# 右子树

    # kd 树先序遍历
    def preorder(self,root):
        if not root:
            return
        temp = root
        temp_list = []
        while temp or temp_list:
            while temp:
                print(temp.dom_elt)
                temp_list.append(temp)
                temp = temp.left
            temp = temp_list.pop()
            temp = temp.right

    def find_nearest(self,point):
        # 数据维度
        k = len(point)
        def search(kd_node,target):
            if not kd_node:
                return self.result([0] * k,float('inf'))

            # 获取在当前的分割轴上的点和分割的维度
            split = kd_node.split
            parent_node = kd_node.dom_elt

            # 如果目标值当前维度的值小于当前根节点的值，则目标离左子树更近，反之与右子树更近
            if target[split] < parent_node[split]:
                nearer = kd_node.left
                further = kd_node.right
            else:
                nearer = kd_node.right
                further = kd_node.left

            # 从根结点开始向下递归查找包含目标点的叶结点
            temp1 = search(nearer,target)

            # 以当前叶结点作为 “当前最近结点”
            nearest = temp1.nearest_point
            distance = temp1.neareast_distance

            hyperplane_distance = abs(target[split] - parent_node[split])
            if hyperplane_distance > distance:
                return self.result(nearest,distance)

            # 计算目标点与分割点的欧式距离，如果小于max_distance则更新
            hyperplane_distance = math.sqrt(sum([(p1 - p2) ** 2 for p1,p2 in zip(parent_node,target)]))
            if hyperplane_distance < distance:
                nearest = parent_node
                distance = hyperplane_distance
                

            # 在另一子结点所在区域查找是否有更近的点，如果有则更新
            temp2 = search(further,target)
            if temp2.neareast_distance < distance:
                nearest = temp2.nearest_point
                distance = temp2.neareast_distance
            return self.result(nearest,distance)

        return search(self.root,point)


# 产生一个 k 为随机向量，每维的分量都在0~1之间
def random_point(k):
    return [random() for _ in range(k)]


def random_points(k,n):
    return [random_point(k) for _ in range(n)]


if __name__ == '__main__':
    data = [[2, 3],[3,4], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    kd = KdTree(data)
    # preorder(kd.root)

    ret = kd.find_nearest([3,4.5])
    print(ret)

    # N = 400000
    # t0 = time.time()
    # kd2 = KdTree(random_points(3, N))  # 构建包含四十万个3维空间样本点的kd树
    # ret2 = find_nearest(kd2, [0.1, 0.5, 0.8])  # 四十万个样本点中寻找离目标最近的点
    # t1 = time.time()
    # print("time: ", t1 - t0, "s")
    # print(ret2)







































