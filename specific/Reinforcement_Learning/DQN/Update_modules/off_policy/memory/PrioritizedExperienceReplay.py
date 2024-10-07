import numpy as np
from collections import namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# from operator import itemgetter 
# b = [1,3,5]
# print(itemgetter(*b)(self.memory_pool.memory))
# exit()

class SumTree:
    def __init__(self, capacity):
        # 存储叶子节点优先级的树结构，大小为 2 * capacity - 1
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        # 存储实际经验
        self.data = np.zeros(capacity, dtype=object)
        # 当前存储的经验数
        self.data_pointer = 0

    def push(self, priority, data): # add
        """添加新经验，并更新优先级"""
        # 获取叶子节点的位置
        tree_index = self.data_pointer + self.capacity - 1
        # 存储数据
        self.data[self.data_pointer] = data
        # 更新叶子节点优先级
        self.update(tree_index, priority)
        # 更新数据指针
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0  # 循环覆盖

    def update(self, tree_index, priority):
        """更新叶子节点的优先级，并更新父节点的和"""
        # 计算优先级变化
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        # 递归更新父节点
        self._propagate(tree_index, change)

    def _propagate(self, tree_index, change):
        """向上递归更新父节点"""
        parent = (tree_index - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def get_leaf(self, value):
        """按优先级比例采样，获取相应的叶子节点"""
        parent = 0
        while True:
            left_child = 2 * parent + 1
            right_child = left_child + 1
            # 检查是否是叶子节点
            if left_child >= len(self.tree):
                leaf_index = parent
                break
            else:
                if value <= self.tree[left_child]:
                    parent = left_child
                else:
                    value -= self.tree[left_child]
                    parent = right_child
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def total_priority(self):
        """返回总的优先级和"""
        return self.tree[0]
    
    def __len__(self):
        return self.data_pointer