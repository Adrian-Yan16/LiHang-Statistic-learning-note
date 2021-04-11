import numpy as np
from sklearn.neighbors import KDTree

train_data = np.array([(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)])
tree = KDTree(train_data,leaf_size=2)
dist,ind = tree.query(np.array([(3,4.5)]),k=1)
res = train_data[ind[0]][0]


print('(3,4.5)最近点为',res)