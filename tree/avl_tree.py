import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import sys
import array
import mmap
import os


import struct

class MemoryPool:
    def __init__(self, size=1000):
        self.size = size
        self.node_size = struct.calcsize('iiih')  # Correctly calculate node_size
        self.memory = mmap.mmap(-1, size * self.node_size)
        self.available = array.array('I', range(size))
        self.lock = threading.Lock()

    def allocate(self):
        with self.lock:
            if not self.available:
                return None
            index = self.available.pop()
            return index

    def deallocate(self, index):
        with self.lock:
            self.available.append(index)

    def get(self, index):
        if index is None:
            return None
        offset = index * self.node_size
        val, left, right, height = struct.unpack('iiih', self.memory[offset:offset + self.node_size])
        left = None if left == -1 else left
        right = None if right == -1 else right
        return (val, left, right, height)

    def set(self, index, val, left, right, height):
        offset = index * self.node_size
        left = -1 if left is None else left
        right = -1 if right is None else right
        self.memory[offset:offset + self.node_size] = struct.pack('iiih', val, left, right, height)


class AVLTree:
    def __init__(self, pool_size=1000):
        self.root = None
        self.memory_pool = MemoryPool(pool_size)
        self.lock = threading.Lock()

    def _height(self, node):
        return node[3] if node else 0

    def _balance(self, node):
        return self._height(self.memory_pool.get(node[1])) - self._height(self.memory_pool.get(node[2]))

    def _update_height(self, index):
        node = self.memory_pool.get(index)
        height = 1 + max(self._height(self.memory_pool.get(node[1])), self._height(self.memory_pool.get(node[2])))
        self.memory_pool.set(index, node[0], node[1], node[2], height)

    def _rotate_right(self, y_index):
        y = self.memory_pool.get(y_index)
        x_index = y[1]
        x = self.memory_pool.get(x_index)
        T2 = x[2]

        self.memory_pool.set(x_index, x[0], x[1], y_index, 0)
        self.memory_pool.set(y_index, y[0], T2, y[2], 0)

        self._update_height(y_index)
        self._update_height(x_index)

        return x_index

    def _rotate_left(self, x_index):
        x = self.memory_pool.get(x_index)
        y_index = x[2]
        y = self.memory_pool.get(y_index)
        T2 = y[1]

        self.memory_pool.set(y_index, y[0], x_index, y[2], 0)
        self.memory_pool.set(x_index, x[0], x[1], T2, 0)

        self._update_height(x_index)
        self._update_height(y_index)

        return y_index

    def insert(self, val):
        def _insert(node_index):
            if node_index is None:
                new_index = self.memory_pool.allocate()
                if new_index is None:
                    raise MemoryError("Memory pool exhausted")
                self.memory_pool.set(new_index, val, None, None, 1)
                return new_index

            node = self.memory_pool.get(node_index)
            if val < node[0]:
                left = _insert(node[1])
                self.memory_pool.set(node_index, node[0], left, node[2], node[3])
            elif val > node[0]:
                right = _insert(node[2])
                self.memory_pool.set(node_index, node[0], node[1], right, node[3])
            else:
                return node_index

            self._update_height(node_index)
            balance = self._balance(self.memory_pool.get(node_index))

            if balance > 1:
                if val < self.memory_pool.get(self.memory_pool.get(node_index)[1])[0]:
                    return self._rotate_right(node_index)
                self.memory_pool.set(node_index, node[0], self._rotate_left(node[1]), node[2], node[3])
                return self._rotate_right(node_index)
            if balance < -1:
                if val > self.memory_pool.get(self.memory_pool.get(node_index)[2])[0]:
                    return self._rotate_left(node_index)
                self.memory_pool.set(node_index, node[0], node[1], self._rotate_right(node[2]), node[3])
                return self._rotate_left(node_index)

            return node_index

        with self.lock:
            self.root = _insert(self.root)

    def delete(self, val):
        def _delete(node_index):
            if node_index is None:
                return None

            node = self.memory_pool.get(node_index)
            if val < node[0]:
                left = _delete(node[1])
                self.memory_pool.set(node_index, node[0], left, node[2], node[3])
            elif val > node[0]:
                right = _delete(node[2])
                self.memory_pool.set(node_index, node[0], node[1], right, node[3])
            else:
                if node[1] is None:
                    temp = node[2]
                    self.memory_pool.deallocate(node_index)
                    return temp
                elif node[2] is None:
                    temp = node[1]
                    self.memory_pool.deallocate(node_index)
                    return temp
                temp_index = self._find_min(node[2])
                temp = self.memory_pool.get(temp_index)
                self.memory_pool.set(node_index, temp[0], node[1], node[2], node[3])
                right = _delete(node[2])
                self.memory_pool.set(node_index, temp[0], node[1], right, node[3])

            if node_index is None:
                return None

            self._update_height(node_index)
            balance = self._balance(self.memory_pool.get(node_index))

            if balance > 1:
                if self._balance(self.memory_pool.get(self.memory_pool.get(node_index)[1])) >= 0:
                    return self._rotate_right(node_index)
                self.memory_pool.set(node_index, node[0], self._rotate_left(node[1]), node[2], node[3])
                return self._rotate_right(node_index)
            if balance < -1:
                if self._balance(self.memory_pool.get(self.memory_pool.get(node_index)[2])) <= 0:
                    return self._rotate_left(node_index)
                self.memory_pool.set(node_index, node[0], node[1], self._rotate_right(node[2]), node[3])
                return self._rotate_left(node_index)

            return node_index

        with self.lock:
            self.root = _delete(self.root)

    def _find_min(self, node_index):
        current = self.memory_pool.get(node_index)
        while current[1] is not None:
            current = self.memory_pool.get(current[1])
        return current

    def search(self, val):
        current_index = self.root
        while current_index is not None:
            current = self.memory_pool.get(current_index)
            if val == current[0]:
                return current
            current_index = current[1] if val < current[0] else current[2]
        return None

    def inorder(self):
        result = []

        def _inorder(node_index):
            if node_index is None:
                return
            node = self.memory_pool.get(node_index)
            _inorder(node[1])
            result.append(node[0])
            _inorder(node[2])

        _inorder(self.root)
        return result

    def level_order(self):
        if self.root is None:
            return []
        result = []
        queue = deque([(self.root, 0)])
        while queue:
            node_index, level = queue.popleft()
            node = self.memory_pool.get(node_index)
            if len(result) == level:
                result.append([])
            result[level].append(node[0])
            if node[1] is not None:
                queue.append((node[1], level + 1))
            if node[2] is not None:
                queue.append((node[2], level + 1))
        return result

    def parallel_operation(self, operation, data):
        chunk_size = max(1, len(data) // (os.cpu_count() or 1))
        with ThreadPoolExecutor() as executor:
            list(executor.map(operation, data, chunksize=chunk_size))

    def parallel_insert(self, values):
        self.parallel_operation(self.insert, values)

    def parallel_delete(self, values):
        self.parallel_operation(self.delete, values)

    def height(self):
        return self._height(self.memory_pool.get(self.root))

    def is_balanced(self):
        def _is_balanced(node_index):
            if node_index is None:
                return True, 0
            node = self.memory_pool.get(node_index)
            left_balanced, left_height = _is_balanced(node[1])
            if not left_balanced:
                return False, 0
            right_balanced, right_height = _is_balanced(node[2])
            if not right_balanced:
                return False, 0
            return (abs(left_height - right_height) <= 1,
                    1 + max(left_height, right_height))

        return _is_balanced(self.root)[0]

    def serialize(self):
        if self.root is None:
            return '[]'
        result = []
        queue = deque([self.root])
        while queue:
            node_index = queue.popleft()
            if node_index is not None:
                node = self.memory_pool.get(node_index)
                result.append(str(node[0]))
                queue.append(node[1])
                queue.append(node[2])
            else:
                result.append('null')
        while result[-1] == 'null':
            result.pop()
        return '[' + ','.join(result) + ']'

    @classmethod
    def deserialize(cls, data, pool_size=1000):
        if data == '[]':
            return cls(pool_size)
        values = data[1:-1].split(',')
        tree = cls(pool_size)
        root_index = tree.memory_pool.allocate()
        tree.memory_pool.set(root_index, int(values[0]), None, None, 1)
        queue = deque([root_index])
        i = 1
        while queue and i < len(values):
            parent_index = queue.popleft()
            parent = tree.memory_pool.get(parent_index)
            if values[i] != 'null':
                left_index = tree.memory_pool.allocate()
                tree.memory_pool.set(left_index, int(values[i]), None, None, 1)
                tree.memory_pool.set(parent_index, parent[0], left_index, parent[2], parent[3])
                queue.append(left_index)
            i += 1
            if i < len(values) and values[i] != 'null':
                right_index = tree.memory_pool.allocate()
                tree.memory_pool.set(right_index, int(values[i]), None, None, 1)
                tree.memory_pool.set(parent_index, parent[0], parent[1], right_index, parent[3])
                queue.append(right_index)
            i += 1
        tree.root = root_index
        return tree


# Usage example
