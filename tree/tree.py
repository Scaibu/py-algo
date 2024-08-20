class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class BinaryTree:
    def __init__(self):
        self.root = None

    # 1. Insert
    def insert(self, value):
        if not self.root:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._insert_recursive(node.right, value)

    # 2. Search
    def search(self, value):
        return self._search_recursive(self.root, value)

    def _search_recursive(self, node, value):
        if node is None or node.value == value:
            return node
        if value < node.value:
            return self._search_recursive(node.left, value)
        return self._search_recursive(node.right, value)

    # 3. Delete
    def delete(self, value):
        self.root = self._delete_recursive(self.root, value)

    def _delete_recursive(self, node, value):
        if node is None:
            return node

        if value < node.value:
            node.left = self._delete_recursive(node.left, value)
        elif value > node.value:
            node.right = self._delete_recursive(node.right, value)
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left

            min_node = self._find_min(node.right)
            node.value = min_node.value
            node.right = self._delete_recursive(node.right, min_node.value)

        return node

    # 4. Find minimum
    def find_min(self):
        if self.root is None:
            return None
        return self._find_min(self.root).value

    def _find_min(self, node):
        current = node
        while current.left:
            current = current.left
        return current

    # 5. Find maximum
    def find_max(self):
        if self.root is None:
            return None
        return self._find_max(self.root).value

    def _find_max(self, node):
        current = node
        while current.right:
            current = current.right
        return current

    # 6. Inorder traversal
    def inorder_traversal(self):
        result = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

    # 7. Preorder traversal
    def preorder_traversal(self):
        result = []
        self._preorder_recursive(self.root, result)
        return result

    def _preorder_recursive(self, node, result):
        if node:
            result.append(node.value)
            self._preorder_recursive(node.left, result)
            self._preorder_recursive(node.right, result)

    # 8. Postorder traversal
    def postorder_traversal(self):
        result = []
        self._postorder_recursive(self.root, result)
        return result

    def _postorder_recursive(self, node, result):
        if node:
            self._postorder_recursive(node.left, result)
            self._postorder_recursive(node.right, result)
            result.append(node.value)

    # 9. Level order traversal
    def level_order_traversal(self):
        if not self.root:
            return []

        result = []
        queue = [self.root]

        while queue:
            level_size = len(queue)
            level = []

            for _ in range(level_size):
                node = queue.pop(0)
                level.append(node.value)

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(level)

        return result

    # 10. Height of the tree
    def height(self):
        return self._height_recursive(self.root)

    def _height_recursive(self, node):
        if node is None:
            return -1
        left_height = self._height_recursive(node.left)
        right_height = self._height_recursive(node.right)
        return max(left_height, right_height) + 1

    # 11. Size of the tree
    def size(self):
        return self._size_recursive(self.root)

    def _size_recursive(self, node):
        if node is None:
            return 0
        return 1 + self._size_recursive(node.left) + self._size_recursive(node.right)

    # 12. Check if tree is empty
    def is_empty(self):
        return self.root is None

    # 13. Clear the tree
    def clear(self):
        self.root = None

    # 14. Check if tree is balanced
    def is_balanced(self):
        return self._is_balanced_recursive(self.root) != -1

    def _is_balanced_recursive(self, node):
        if node is None:
            return 0

        left_height = self._is_balanced_recursive(node.left)
        if left_height == -1:
            return -1

        right_height = self._is_balanced_recursive(node.right)
        if right_height == -1:
            return -1

        if abs(left_height - right_height) > 1:
            return -1

        return max(left_height, right_height) + 1

    # 15. Get leaf nodes
    def get_leaf_nodes(self):
        leaves = []
        self._get_leaf_nodes_recursive(self.root, leaves)
        return leaves

    def _get_leaf_nodes_recursive(self, node, leaves):
        if node:
            if not node.left and not node.right:
                leaves.append(node.value)
            self._get_leaf_nodes_recursive(node.left, leaves)
            self._get_leaf_nodes_recursive(node.right, leaves)

    # 16. Count leaf nodes
    def count_leaf_nodes(self):
        return len(self.get_leaf_nodes())

    # 17. Get internal nodes
    def get_internal_nodes(self):
        internal = []
        self._get_internal_nodes_recursive(self.root, internal)
        return internal

    def _get_internal_nodes_recursive(self, node, internal):
        if node:
            if node.left or node.right:
                internal.append(node.value)
            self._get_internal_nodes_recursive(node.left, internal)
            self._get_internal_nodes_recursive(node.right, internal)

    # 18. Count internal nodes
    def count_internal_nodes(self):
        return len(self.get_internal_nodes())

    # 19. Get nodes at a specific level
    def get_nodes_at_level(self, level):
        result = []
        self._get_nodes_at_level_recursive(self.root, level, 0, result)
        return result

    def _get_nodes_at_level_recursive(self, node, target_level, current_level, result):
        if node is None:
            return

        if current_level == target_level:
            result.append(node.value)

        self._get_nodes_at_level_recursive(node.left, target_level, current_level + 1, result)
        self._get_nodes_at_level_recursive(node.right, target_level, current_level + 1, result)

    # 20. Get width of the tree
    def width(self):
        if not self.root:
            return 0

        max_width = 0
        level = 0

        while True:
            width = len(self.get_nodes_at_level(level))
            if width == 0:
                break
            max_width = max(max_width, width)
            level += 1

        return max_width

    # 21. Check if two trees are identical
    def is_identical(self, other_tree):
        return self._is_identical_recursive(self.root, other_tree.root)

    def _is_identical_recursive(self, node1, node2):
        if node1 is None and node2 is None:
            return True
        if node1 is None or node2 is None:
            return False
        return (node1.value == node2.value and
                self._is_identical_recursive(node1.left, node2.left) and
                self._is_identical_recursive(node1.right, node2.right))

    # 22. Get mirror image of the tree
    def mirror(self):
        self._mirror_recursive(self.root)

    def _mirror_recursive(self, node):
        if node:
            node.left, node.right = node.right, node.left
            self._mirror_recursive(node.left)
            self._mirror_recursive(node.right)

    # 23. Check if tree is a mirror of another tree
    def is_mirror(self, other_tree):
        return self._is_mirror_recursive(self.root, other_tree.root)

    def _is_mirror_recursive(self, node1, node2):
        if node1 is None and node2 is None:
            return True
        if node1 is None or node2 is None:
            return False
        return (node1.value == node2.value and
                self._is_mirror_recursive(node1.left, node2.right) and
                self._is_mirror_recursive(node1.right, node2.left))

    # 24. Get lowest common ancestor
    def lowest_common_ancestor(self, value1, value2):
        return self._lca_recursive(self.root, value1, value2)

    def _lca_recursive(self, node, value1, value2):
        if node is None:
            return None

        if node.value > value1 and node.value > value2:
            return self._lca_recursive(node.left, value1, value2)

        if node.value < value1 and node.value < value2:
            return self._lca_recursive(node.right, value1, value2)

        return node.value

    # 25. Check if a value exists in the tree
    def contains(self, value):
        return self.search(value) is not None

    # 26. Get the depth of a node
    def depth(self, value):
        return self._depth_recursive(self.root, value, 0)

    def _depth_recursive(self, node, value, current_depth):
        if node is None:
            return -1
        if node.value == value:
            return current_depth
        left_depth = self._depth_recursive(node.left, value, current_depth + 1)
        if left_depth != -1:
            return left_depth
        return self._depth_recursive(node.right, value, current_depth + 1)

    # 27. Get the diameter of the tree
    def diameter(self):
        self.max_diameter = 0
        self._diameter_recursive(self.root)
        return self.max_diameter

    def _diameter_recursive(self, node):
        if node is None:
            return 0

        left_height = self._diameter_recursive(node.left)
        right_height = self._diameter_recursive(node.right)

        self.max_diameter = max(self.max_diameter, left_height + right_height)

        return max(left_height, right_height) + 1

    # 28. Check if the tree is a binary search tree
    def is_bst(self):
        return self._is_bst_recursive(self.root, float('-inf'), float('inf'))

    def _is_bst_recursive(self, node, min_value, max_value):
        if node is None:
            return True

        if node.value <= min_value or node.value >= max_value:
            return False

        return (self._is_bst_recursive(node.left, min_value, node.value) and
                self._is_bst_recursive(node.right, node.value, max_value))

    # 29. Get the sum of all node values
    def sum_of_nodes(self):
        return self._sum_of_nodes_recursive(self.root)

    def _sum_of_nodes_recursive(self, node):
        if node is None:
            return 0
        return node.value + self._sum_of_nodes_recursive(node.left) + self._sum_of_nodes_recursive(node.right)

    # 30. Get the maximum path sum
    def max_path_sum(self):
        self.max_sum = float('-inf')
        self._max_path_sum_recursive(self.root)
        return self.max_sum

    def _max_path_sum_recursive(self, node):
        if node is None:
            return 0

        left_sum = max(0, self._max_path_sum_recursive(node.left))
        right_sum = max(0, self._max_path_sum_recursive(node.right))

        self.max_sum = max(self.max_sum, left_sum + right_sum + node.value)

        return max(left_sum, right_sum) + node.value

    def to_list(self):
        return self.inorder_traversal()

    # 32. Create tree from sorted list
    def from_sorted_list(self, sorted_list):
        self.root = self._from_sorted_list_recursive(sorted_list, 0, len(sorted_list) - 1)

    def _from_sorted_list_recursive(self, sorted_list, start, end):
        if start > end:
            return None
        mid = (start + end) // 2
        node = TreeNode(sorted_list[mid])
        node.left = self._from_sorted_list_recursive(sorted_list, start, mid - 1)
        node.right = self._from_sorted_list_recursive(sorted_list, mid + 1, end)
        return node

    # 33. Check if tree is complete
    def is_complete(self):
        if not self.root:
            return True
        queue = [self.root]
        flag = False
        while queue:
            node = queue.pop(0)
            if node.left:
                if flag:
                    return False
                queue.append(node.left)
            else:
                flag = True
            if node.right:
                if flag:
                    return False
                queue.append(node.right)
            else:
                flag = True
        return True

    # 34. Check if tree is perfect
    def is_perfect(self):
        height = self.height()
        size = self.size()
        return size == (2 ** (height + 1) - 1)

    # 35. Check if tree is full
    def is_full(self):
        return self._is_full_recursive(self.root)

    def _is_full_recursive(self, node):
        if node is None:
            return True
        if node.left is None and node.right is None:
            return True
        if node.left and node.right:
            return self._is_full_recursive(node.left) and self._is_full_recursive(node.right)
        return False

    # 36. Get all paths from root to leaves
    def all_paths(self):
        paths = []
        self._all_paths_recursive(self.root, [], paths)
        return paths

    def _all_paths_recursive(self, node, path, paths):
        if node is None:
            return
        path.append(node.value)
        if node.left is None and node.right is None:
            paths.append(list(path))
        else:
            self._all_paths_recursive(node.left, path, paths)
            self._all_paths_recursive(node.right, path, paths)
        path.pop()

    # 37. Get path to a specific node
    def path_to_node(self, value):
        path = []
        self._path_to_node_recursive(self.root, value, path)
        return path[::-1] if path else []

    def _path_to_node_recursive(self, node, value, path):
        if node is None:
            return False
        if node.value == value:
            path.append(node.value)
            return True
        if self._path_to_node_recursive(node.left, value, path) or \
                self._path_to_node_recursive(node.right, value, path):
            path.append(node.value)
            return True
        return False

    # 38. Get the level of a node
    def level_of_node(self, value):
        return self._level_of_node_recursive(self.root, value, 0)

    def _level_of_node_recursive(self, node, value, level):
        if node is None:
            return -1
        if node.value == value:
            return level
        left_level = self._level_of_node_recursive(node.left, value, level + 1)
        if left_level != -1:
            return left_level
        return self._level_of_node_recursive(node.right, value, level + 1)

    # 39. Get the sibling of a node
    def get_sibling(self, value):
        parent = self._find_parent(self.root, value)
        if parent is None:
            return None
        if parent.left and parent.left.value == value:
            return parent.right.value if parent.right else None
        return parent.left.value if parent.left else None

    def _find_parent(self, node, value):
        if node is None or (node.left and node.left.value == value) or (node.right and node.right.value == value):
            return node
        left_parent = self._find_parent(node.left, value)
        if left_parent:
            return left_parent
        return self._find_parent(node.right, value)

    # 40. Get the uncle of a node
    def get_uncle(self, value):
        parent = self._find_parent(self.root, value)
        if parent is None:
            return None
        grandparent = self._find_parent(self.root, parent.value)
        if grandparent is None:
            return None
        if grandparent.left == parent:
            return grandparent.right.value if grandparent.right else None
        return grandparent.left.value if grandparent.left else None

    # 41. Check if a node is a leaf
    def is_leaf(self, value):
        node = self.search(value)
        return node is not None and node.left is None and node.right is None

    # 42. Get the maximum width of the tree
    def max_width(self):
        if not self.root:
            return 0
        queue = [(self.root, 0)]
        max_width = 0
        while queue:
            level_size = len(queue)
            level_start = queue[0][1]
            level_end = queue[-1][1]
            max_width = max(max_width, level_end - level_start + 1)
            for _ in range(level_size):
                node, index = queue.pop(0)
                if node.left:
                    queue.append((node.left, 2 * index))
                if node.right:
                    queue.append((node.right, 2 * index + 1))
        return max_width

    # 43. Get the vertical sum of the tree
    def vertical_sum(self):
        sums = {}
        self._vertical_sum_recursive(self.root, 0, sums)
        return [sums[key] for key in sorted(sums.keys())]

    def _vertical_sum_recursive(self, node, hd, sums):
        if node is None:
            return
        sums[hd] = sums.get(hd, 0) + node.value
        self._vertical_sum_recursive(node.left, hd - 1, sums)
        self._vertical_sum_recursive(node.right, hd + 1, sums)

    # 44. Check if tree is symmetric
    def is_symmetric(self):
        if not self.root:
            return True
        return self._is_symmetric_recursive(self.root.left, self.root.right)

    def _is_symmetric_recursive(self, left, right):
        if left is None and right is None:
            return True
        if left is None or right is None:
            return False
        return (left.value == right.value and
                self._is_symmetric_recursive(left.left, right.right) and
                self._is_symmetric_recursive(left.right, right.left))

    # 45. Get the boundary nodes of the tree
    def boundary_traversal(self):
        if not self.root:
            return []
        result = [self.root.value]
        self._left_boundary(self.root.left, result)
        self._leaves(self.root.left, result)
        self._leaves(self.root.right, result)
        self._right_boundary(self.root.right, result)
        return result

    def _left_boundary(self, node, result):
        if node is None or (node.left is None and node.right is None):
            return
        result.append(node.value)
        if node.left:
            self._left_boundary(node.left, result)
        else:
            self._left_boundary(node.right, result)

    def _right_boundary(self, node, result):
        if node is None or (node.left is None and node.right is None):
            return
        if node.right:
            self._right_boundary(node.right, result)
        else:
            self._right_boundary(node.left, result)
        result.append(node.value)

    def _leaves(self, node, result):
        if node is None:
            return
        if node.left is None and node.right is None:
            result.append(node.value)
            return
        self._leaves(node.left, result)
        self._leaves(node.right, result)

    # 46. Get the diagonal traversal of the tree
    def diagonal_traversal(self):
        result = []
        node = self.root
        queue = []
        while node:
            result.append(node.value)
            if node.left:
                queue.append(node.left)
            if node.right:
                node = node.right
            else:
                if queue:
                    node = queue.pop(0)
                else:
                    node = None
        return result

    # 47. Check if a binary tree is subtree of another binary tree
    def is_subtree(self, subtree):
        if subtree.root is None:
            return True
        return self._is_subtree_recursive(self.root, subtree.root)

    def _is_subtree_recursive(self, t, s):
        if t is None:
            return False
        if self._is_identical_recursive(t, s):
            return True
        return self._is_subtree_recursive(t.left, s) or self._is_subtree_recursive(t.right, s)

    # 48. Get the vertical order traversal of the tree
    def vertical_order_traversal(self):
        if not self.root:
            return []
        column_table = {}
        queue = [(self.root, 0)]
        min_column = max_column = 0
        while queue:
            node, column = queue.pop(0)
            if column not in column_table:
                column_table[column] = []
            column_table[column].append(node.value)
            min_column = min(min_column, column)
            max_column = max(max_column, column)
            if node.left:
                queue.append((node.left, column - 1))
            if node.right:
                queue.append((node.right, column + 1))
        return [column_table[i] for i in range(min_column, max_column + 1)]

    # 49. Get the top view of the tree
    def top_view(self):
        if not self.root:
            return []
        column_table = {}
        queue = [(self.root, 0)]
        min_column = max_column = 0
        while queue:
            node, column = queue.pop(0)
            if column not in column_table:
                column_table[column] = node.value
            min_column = min(min_column, column)
            max_column = max(max_column, column)
            if node.left:
                queue.append((node.left, column - 1))
            if node.right:
                queue.append((node.right, column + 1))
        return [column_table[i] for i in range(min_column, max_column + 1)]

    # 50. Get the bottom view of the tree
    def bottom_view(self):
        if not self.root:
            return []
        column_table = {}
        queue = [(self.root, 0)]
        min_column = max_column = 0
        while queue:
            node, column = queue.pop(0)
            column_table[column] = node.value
            min_column = min(min_column, column)
            max_column = max(max_column, column)
            if node.left:
                queue.append((node.left, column - 1))
            if node.right:
                queue.append((node.right, column + 1))
        return [column_table[i] for i in range(min_column, max_column + 1)]

    # 51. Convert tree to its mirror image
    def convert_to_mirror(self):
        self._convert_to_mirror_recursive(self.root)

    def _convert_to_mirror_recursive(self, node):
        if node is None:
            return
        self._convert_to_mirror_recursive(node.left)
        self._convert_to_mirror_recursive(node.right)
        node.left, node.right = node.right, node.left

    # 52. Check if two nodes are cousins
    def are_cousins(self, value1, value2):
        level1 = self.level_of_node(value1)
        level2 = self.level_of_node(value2)
        if level1 != level2:
            return False
        parent1 = self._find_parent(self.root, value1)
        parent2 = self._find_parent(self.root, value2)
        return parent1 != parent2

    # 53. Get the deepest node in the tree
    def deepest_node(self):
        if not self.root:
            return None
        queue = [self.root]
        deepest = None
        while queue:
            size = len(queue)
            for _ in range(size):
                node = queue.pop(0)
                deepest = node
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return deepest.value

    # 54. Check if all leaves are at the same level
    def are_leaves_at_same_level(self):
        if not self.root:
            return True
        return self._are_leaves_at_same_level_recursive(self.root, 0, [None])

    def _are_leaves_at_same_level_recursive(self, node, level, leaf_level):
        if node is None:
            return True
        if node.left is None and node.right is None:
            if leaf_level[0] is None:
                leaf_level[0] = level
            return level == leaf_level[0]
        return (self._are_leaves_at_same_level_recursive(node.left, level + 1, leaf_level) and
                self._are_leaves_at_same_level_recursive(node.right, level + 1, leaf_level))
    def density(self):
        size = self.size()
        height = self.height()
        return size / (2 ** (height + 1) - 1) if height >= 0 else 0

    # 62. Check if the tree is height-balanced
    def is_height_balanced(self):
        return self._is_height_balanced_recursive(self.root)[0]

    def _is_height_balanced_recursive(self, node):
        if not node:
            return True, -1
        left_balanced, left_height = self._is_height_balanced_recursive(node.left)
        right_balanced, right_height = self._is_height_balanced_recursive(node.right)
        balanced = left_balanced and right_balanced and abs(left_height - right_height) <= 1
        height = max(left_height, right_height) + 1
        return balanced, height

    # 63. Get the maximum level sum
    def max_level_sum(self):
        if not self.root:
            return 0
        max_sum = float('-inf')
        queue = [self.root]
        while queue:
            level_sum = 0
            level_size = len(queue)
            for _ in range(level_size):
                node = queue.pop(0)
                level_sum += node.value
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            max_sum = max(max_sum, level_sum)
        return max_sum

    # 64. Print all nodes at distance K from root
    def nodes_at_distance_k(self, k):
        return self.get_nodes_at_level(k)

    # 65. Check if a binary tree is a sum tree
    def is_sum_tree(self):
        return self._is_sum_tree_recursive(self.root)[0]

    def _is_sum_tree_recursive(self, node):
        if not node or (not node.left and not node.right):
            return True, node.value if node else 0
        left_is_sum, left_sum = self._is_sum_tree_recursive(node.left)
        right_is_sum, right_sum = self._is_sum_tree_recursive(node.right)
        is_sum_tree = left_is_sum and right_is_sum and node.value == left_sum + right_sum
        return is_sum_tree, node.value + left_sum + right_sum

    # 66. Convert a tree to its sum tree
    def to_sum_tree(self):
        self._to_sum_tree_recursive(self.root)

    def _to_sum_tree_recursive(self, node):
        if not node:
            return 0
        old_value = node.value
        node.value = self._to_sum_tree_recursive(node.left) + self._to_sum_tree_recursive(node.right)
        return node.value + old_value

    # 67. Check if a binary tree contains all possible heights
    def has_all_heights(self):
        max_height = self.height()
        heights = set(range(max_height + 1))
        self._check_heights_recursive(self.root, 0, heights)
        return len(heights) == 0

    def _check_heights_recursive(self, node, current_height, heights):
        if not node:
            return
        if current_height in heights:
            heights.remove(current_height)
        self._check_heights_recursive(node.left, current_height + 1, heights)
        self._check_heights_recursive(node.right, current_height + 1, heights)

    # 68. Get the maximum width of the tree using level order traversal
    def max_width_level_order(self):
        if not self.root:
            return 0
        max_width = 0
        queue = [self.root]
        while queue:
            level_size = len(queue)
            max_width = max(max_width, level_size)
            for _ in range(level_size):
                node = queue.pop(0)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return max_width

    # 69. Print nodes in vertical order
    def print_vertical_order(self):
        if not self.root:
            return
        column_table = {}
        self._vertical_order_recursive(self.root, 0, column_table)
        for column in sorted(column_table.keys()):
            print(f"Column {column}: {column_table[column]}")

    def _vertical_order_recursive(self, node, column, column_table):
        if not node:
            return
        if column not in column_table:
            column_table[column] = []
        column_table[column].append(node.value)
        self._vertical_order_recursive(node.left, column - 1, column_table)
        self._vertical_order_recursive(node.right, column + 1, column_table)

    # 70. Check if a binary tree is a complete binary tree
    def is_complete_binary_tree(self):
        if not self.root:
            return True
        queue = [self.root]
        flag = False
        while queue:
            node = queue.pop(0)
            if node.left:
                if flag:
                    return False
                queue.append(node.left)
            else:
                flag = True
            if node.right:
                if flag:
                    return False
                queue.append(node.right)
            else:
                flag = True
        return True

    # 71. Get the maximum consecutive increasing path length in the tree
    def max_consecutive_path_length(self):
        return self._max_consecutive_path_length_recursive(self.root, None, 0)[1]

    def _max_consecutive_path_length_recursive(self, node, parent, length):
        if not node:
            return parent, length
        current_length = length + 1 if parent and node.value == parent.value + 1 else 1
        left_parent, left_length = self._max_consecutive_path_length_recursive(node.left, node, current_length)
        right_parent, right_length = self._max_consecutive_path_length_recursive(node.right, node, current_length)
        max_length = max(current_length, left_length, right_length)
        return node, max_length

    # 72. Print all paths with a specified sum
    def print_paths_with_sum(self, target_sum):
        self._print_paths_with_sum_recursive(self.root, target_sum, [])

    def _print_paths_with_sum_recursive(self, node, target_sum, path):
        if not node:
            return
        path.append(node.value)
        if sum(path) == target_sum:
            print(path)
        self._print_paths_with_sum_recursive(node.left, target_sum, path)
        self._print_paths_with_sum_recursive(node.right, target_sum, path)
        path.pop()

    # 73. Get the maximum difference between node and its ancestor
    def max_diff_node_ancestor(self):
        return self._max_diff_node_ancestor_recursive(self.root)[1]

    def _max_diff_node_ancestor_recursive(self, node):
        if not node:
            return float('inf'), float('-inf')
        if not node.left and not node.right:
            return node.value, float('-inf')
        left_min, left_diff = self._max_diff_node_ancestor_recursive(node.left)
        right_min, right_diff = self._max_diff_node_ancestor_recursive(node.right)
        curr_min = min(node.value, left_min, right_min)
        curr_diff = max(node.value - left_min, node.value - right_min, left_diff, right_diff)
        return curr_min, curr_diff

    # 74. Check if the binary tree is a min-heap
    def is_min_heap(self):
        return self._is_min_heap_recursive(self.root)

    def _is_min_heap_recursive(self, node):
        if not node:
            return True
        if node.left and node.left.value < node.value:
            return False
        if node.right and node.right.value < node.value:
            return False
        return self._is_min_heap_recursive(node.left) and self._is_min_heap_recursive(node.right)

    # 75. Print all nodes that don't have a sibling
    def print_nodes_without_sibling(self):
        self._print_nodes_without_sibling_recursive(self.root)

    def _print_nodes_without_sibling_recursive(self, node):
        if not node:
            return
        if node.left and not node.right:
            print(node.left.value)
        elif node.right and not node.left:
            print(node.right.value)
        self._print_nodes_without_sibling_recursive(node.left)
        self._print_nodes_without_sibling_recursive(node.right)

    # 76. Check if the binary tree is a foldable tree
    def is_foldable(self):
        if not self.root:
            return True
        return self._is_foldable_recursive(self.root.left, self.root.right)

    def _is_foldable_recursive(self, left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        return (self._is_foldable_recursive(left.left, right.right) and
                self._is_foldable_recursive(left.right, right.left))

    # 77. Get the maximum sum of a path between two leaves
    def max_sum_leaf_to_leaf(self):
        return self._max_sum_leaf_to_leaf_recursive(self.root)[1]

    def _max_sum_leaf_to_leaf_recursive(self, node):
        if not node:
            return 0, float('-inf')
        if not node.left and not node.right:
            return node.value, float('-inf')
        left_sum, left_max = self._max_sum_leaf_to_leaf_recursive(node.left)
        right_sum, right_max = self._max_sum_leaf_to_leaf_recursive(node.right)
        curr_sum = max(left_sum, right_sum) + node.value
        curr_max = max(left_max, right_max, left_sum + right_sum + node.value)
        return curr_sum, curr_max

    # 78. Check if the binary tree is isomorphic to another binary tree
    def is_isomorphic(self, other_tree):
        return self._is_isomorphic_recursive(self.root, other_tree.root)

    def _is_isomorphic_recursive(self, node1, node2):
        if not node1 and not node2:
            return True
        if not node1 or not node2:
            return False
        if node1.value != node2.value:
            return False
        return ((self._is_isomorphic_recursive(node1.left, node2.left) and
                 self._is_isomorphic_recursive(node1.right, node2.right)) or
                (self._is_isomorphic_recursive(node1.left, node2.right) and
                 self._is_isomorphic_recursive(node1.right, node2.left)))

    # 79. Print all ancestors of a given node
    def print_ancestors(self, value):
        self._print_ancestors_recursive(self.root, value)

    def _print_ancestors_recursive(self, node, value):
        if not node:
            return False
        if node.value == value:
            return True
        if (self._print_ancestors_recursive(node.left, value) or
            self._print_ancestors_recursive(node.right, value)):
            print(node.value)
            return True
        return False

    # 80. Find the largest BST subtree in a binary tree
    def largest_bst_subtree(self):
        return self._largest_bst_subtree_recursive(self.root)[2]

    def _largest_bst_subtree_recursive(self, node):
        if not node:
            return float('inf'), float('-inf'), 0, True
        left_min, left_max, left_size, left_is_bst = self._largest_bst_subtree_recursive(node.left)
        right_min, right_max, right_size, right_is_bst = self._largest_bst_subtree_recursive(node.right)
        if left_is_bst and right_is_bst and left_max < node.value < right_min:
            return min(left_min, node.value), max(right_max, node.value), left_size + right_size + 1, True
        return float('-inf'), float('inf'), max(left_size, right_size), False

    # 81. Check if the binary tree is a sum tree
    def is_sum_tree(self):
        return self._is_sum_tree_recursive(self.root)[0]

    def _is_sum_tree_recursive(self, node):
        if not node or (not node.left and not node.right):
            return True, node.value if node else 0
        left_is_sum, left_sum = self._is_sum_tree_recursive(node.left)
        right_is_sum, right_sum = self._is_sum_tree_recursive(node.right)
        is_sum_tree = left_is_sum and right_is_sum and node.value == left_sum + right_sum
        return is_sum_tree, node.value + left_sum + right_sum

    # 82. Get the maximum sum of a path from root to leaf
    def max_sum_root_to_leaf(self):
        return self._max_sum_root_to_leaf_recursive(self.root)

    def _max_sum_root_to_leaf_recursive(self, node):
        if not node:
            return 0
        left_sum = self._max_sum_root_to_leaf_recursive(node.left)
        right_sum = self._max_sum_root_to_leaf_recursive(node.right)
        return node.value + max(left_sum, right_sum)

    # 83. Check if the binary tree is a perfect binary tree
    def is_perfect_binary_tree(self):
        depth = self._leftmost_depth(self.root)
        return self._is_perfect_binary_tree_recursive(self.root, depth, 0)

    def _leftmost_depth(self, node):
        depth = 0
        while node:
            depth += 1
            node = node.left
        return depth

    def _is_perfect_binary_tree_recursive(self, node, depth, level):
        if not node:
            return True
        if not node.left and not node.right:
            return depth == level + 1
        if not node.left or not node.right:
            return False
        return (self._is_perfect_binary_tree_recursive(node.left, depth, level + 1) and
                self._is_perfect_binary_tree_recursive(node.right, depth, level + 1))
    def _print_nodes_down(self, node, k):
        if not node or k < 0:
            return
        if k == 0:
            print(node.value)
            return
        self._print_nodes_down(node.left, k - 1)
        self._print_nodes_down(node.right, k - 1)

    # 85. Check if the binary tree is a continuous tree
    def is_continuous_tree(self):
        return self._is_continuous_tree_recursive(self.root)

    def _is_continuous_tree_recursive(self, node):
        if not node:
            return True
        if not node.left and not node.right:
            return True
        if node.left and not node.right:
            return abs(node.value - node.left.value) == 1 and self._is_continuous_tree_recursive(node.left)
        if node.right and not node.left:
            return abs(node.value - node.right.value) == 1 and self._is_continuous_tree_recursive(node.right)
        return (abs(node.value - node.left.value) == 1 and
                abs(node.value - node.right.value) == 1 and
                self._is_continuous_tree_recursive(node.left) and
                self._is_continuous_tree_recursive(node.right))

    # 86. Get the maximum width of the binary tree
    def max_width(self):
        if not self.root:
            return 0
        queue = [(self.root, 0)]
        max_width = 0
        while queue:
            level_size = len(queue)
            level_start = queue[0][1]
            level_end = queue[-1][1]
            max_width = max(max_width, level_end - level_start + 1)
            for _ in range(level_size):
                node, index = queue.pop(0)
                if node.left:
                    queue.append((node.left, 2 * index))
                if node.right:
                    queue.append((node.right, 2 * index + 1))
        return max_width

    # 87. Check if the binary tree is a complete binary tree
    def is_complete_binary_tree(self):
        if not self.root:
            return True
        queue = [self.root]
        flag = False
        while queue:
            node = queue.pop(0)
            if node.left:
                if flag:
                    return False
                queue.append(node.left)
            else:
                flag = True
            if node.right:
                if flag:
                    return False
                queue.append(node.right)
            else:
                flag = True
        return True

    # 88. Get the deepest left leaf node
    def deepest_left_leaf(self):
        return self._deepest_left_leaf_recursive(self.root, 0, False)[0]

    def _deepest_left_leaf_recursive(self, node, level, is_left):
        if not node:
            return None, 0
        if not node.left and not node.right and is_left:
            return node, level
        left_node, left_level = self._deepest_left_leaf_recursive(node.left, level + 1, True)
        right_node, right_level = self._deepest_left_leaf_recursive(node.right, level + 1, False)
        if left_level > right_level:
            return left_node, left_level
        elif right_level > left_level:
            return right_node, right_level
        else:
            return left_node, left_level

    # 89. Check if the binary tree is a sum tree
    def is_sum_tree(self):
        return self._is_sum_tree_recursive(self.root)[0]

    def _is_sum_tree_recursive(self, node):
        if not node or (not node.left and not node.right):
            return True, node.value if node else 0
        left_is_sum, left_sum = self._is_sum_tree_recursive(node.left)
        right_is_sum, right_sum = self._is_sum_tree_recursive(node.right)
        is_sum_tree = left_is_sum and right_is_sum and node.value == left_sum + right_sum
        return is_sum_tree, node.value + left_sum + right_sum

    # 90. Convert a binary tree to a doubly linked list
    def to_doubly_linked_list(self):
        self.head = self.prev = None
        self._to_doubly_linked_list_recursive(self.root)
        return self.head

    def _to_doubly_linked_list_recursive(self, node):
        if not node:
            return
        self._to_doubly_linked_list_recursive(node.left)
        if not self.prev:
            self.head = node
        else:
            node.left = self.prev
            self.prev.right = node
        self.prev = node
        self._to_doubly_linked_list_recursive(node.right)

 # 91. Find the closest leaf node for a given value
    def closest_leaf(self, value):
        target_node = self.search(value)
        if not target_node:
            return None
        return self._closest_leaf_recursive(self.root, target_node)

    def _closest_leaf_recursive(self, node, target):
        if not node:
            return float('inf'), None
        if node == target:
            if not node.left and not node.right:
                return 0, node
            left_dist, left_leaf = self._closest_leaf_recursive(node.left, target)
            right_dist, right_leaf = self._closest_leaf_recursive(node.right, target)
            if left_dist < right_dist:
                return left_dist + 1, left_leaf
            else:
                return right_dist + 1, right_leaf
        if not node.left and not node.right:
            return 0, node
        left_dist, left_leaf = self._closest_leaf_recursive(node.left, target)
        right_dist, right_leaf = self._closest_leaf_recursive(node.right, target)
        if left_dist < right_dist:
            return left_dist + 1, left_leaf
        else:
            return right_dist + 1, right_leaf

    # 92. Check if the binary tree is a min-heap
    def is_min_heap(self):
        return self._is_min_heap_recursive(self.root)

    def _is_min_heap_recursive(self, node):
        if not node:
            return True
        if node.left and node.left.value < node.value:
            return False
        if node.right and node.right.value < node.value:
            return False
        return self._is_min_heap_recursive(node.left) and self._is_min_heap_recursive(node.right)

    # 93. Convert binary tree to sum tree
    def to_sum_tree(self):
        self._to_sum_tree_recursive(self.root)

    def _to_sum_tree_recursive(self, node):
        if not node:
            return 0
        old_value = node.value
        node.value = (self._to_sum_tree_recursive(node.left) +
                      self._to_sum_tree_recursive(node.right))
        return node.value + old_value

    # 94. Check if two nodes are cousins
    def are_cousins(self, value1, value2):
        level1, parent1 = self._find_level_and_parent(self.root, value1, 0, None)
        level2, parent2 = self._find_level_and_parent(self.root, value2, 0, None)
        return level1 == level2 and parent1 != parent2

    def _find_level_and_parent(self, node, value, level, parent):
        if not node:
            return -1, None
        if node.value == value:
            return level, parent
        left_level, left_parent = self._find_level_and_parent(node.left, value, level + 1, node)
        if left_level != -1:
            return left_level, left_parent
        return self._find_level_and_parent(node.right, value, level + 1, node)

    # 95. Print all paths from root to leaf
    def print_all_paths(self):
        self._print_all_paths_recursive(self.root, [])

    def _print_all_paths_recursive(self, node, path):
        if not node:
            return
        path.append(node.value)
        if not node.left and not node.right:
            print(path)
        self._print_all_paths_recursive(node.left, path)
        self._print_all_paths_recursive(node.right, path)
        path.pop()

    # 96. Check if a binary tree is subtree of another binary tree
    def is_subtree(self, subtree):
        return self._is_subtree_recursive(self.root, subtree.root)

    def _is_subtree_recursive(self, t, s):
        if not s:
            return True
        if not t:
            return False
        if self._are_identical(t, s):
            return True
        return self._is_subtree_recursive(t.left, s) or self._is_subtree_recursive(t.right, s)

    def _are_identical(self, root1, root2):
        if not root1 and not root2:
            return True
        if not root1 or not root2:
            return False
        return (root1.value == root2.value and
                self._are_identical(root1.left, root2.left) and
                self._are_identical(root1.right, root2.right))

    # 97. Find the maximum path sum between two leaves
    def max_path_sum_leaves(self):
        self.max_sum = float('-inf')
        self._max_path_sum_leaves_recursive(self.root)
        return self.max_sum

    def _max_path_sum_leaves_recursive(self, node):
        if not node:
            return 0
        if not node.left and not node.right:
            return node.value
        left_sum = self._max_path_sum_leaves_recursive(node.left)
        right_sum = self._max_path_sum_leaves_recursive(node.right)
        if node.left and node.right:
            self.max_sum = max(self.max_sum, left_sum + right_sum + node.value)
            return max(left_sum, right_sum) + node.value
        return (left_sum + node.value) if node.left else (right_sum + node.value)

    # 98. Check if a binary tree is height balanced
    def is_height_balanced(self):
        return self._is_height_balanced_recursive(self.root)[0]

    def _is_height_balanced_recursive(self, node):
        if not node:
            return True, 0
        left_balanced, left_height = self._is_height_balanced_recursive(node.left)
        right_balanced, right_height = self._is_height_balanced_recursive(node.right)
        balanced = left_balanced and right_balanced and abs(left_height - right_height) <= 1
        height = max(left_height, right_height) + 1
        return balanced, height

    # 99. Find the diameter of the binary tree
    def diameter(self):
        return self._diameter_recursive(self.root)[1]

    def _diameter_recursive(self, node):
        if not node:
            return 0, 0
        left_height, left_diameter = self._diameter_recursive(node.left)
        right_height, right_diameter = self._diameter_recursive(node.right)
        height = max(left_height, right_height) + 1
        diameter = max(left_height + right_height, left_diameter, right_diameter)
        return height, diameter

    # 100. Check if a binary tree is symmetric
    def is_symmetric(self):
        if not self.root:
            return True
        return self._is_symmetric_recursive(self.root.left, self.root.right)

    def _is_symmetric_recursive(self, left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        return (left.value == right.value and
                self._is_symmetric_recursive(left.left, right.right) and
                self._is_symmetric_recursive(left.right, right.left))

    # 101. Find the lowest common ancestor of two nodes
    def lowest_common_ancestor(self, value1, value2):
        return self._lowest_common_ancestor_recursive(self.root, value1, value2)

    def _lowest_common_ancestor_recursive(self, node, value1, value2):
        if not node:
            return None
        if node.value == value1 or node.value == value2:
            return node
        left_lca = self._lowest_common_ancestor_recursive(node.left, value1, value2)
        right_lca = self._lowest_common_ancestor_recursive(node.right, value1, value2)
        if left_lca and right_lca:
            return node
        return left_lca if left_lca else right_lca

    # 102. Serialize and deserialize binary tree
    def serialize(self):
        return self._serialize_recursive(self.root)

    def _serialize_recursive(self, node):
        if not node:
            return "null"
        return f"{node.value},{self._serialize_recursive(node.left)},{self._serialize_recursive(node.right)}"

    def deserialize(self, data):
        def helper():
            val = next(values)
            if val == "null":
                return None
            node = TreeNode(int(val))
            node.left = helper()
            node.right = helper()
            return node

        values = iter(data.split(","))
        return helper()

    # 103. Find the maximum width of the binary tree
    def max_width(self):
        if not self.root:
            return 0
        queue = [(self.root, 0, 0)]
        cur_depth = left = ans = 0
        for node, depth, pos in queue:
            if node:
                queue.append((node.left, depth+1, pos*2))
                queue.append((node.right, depth+1, pos*2 + 1))
                if cur_depth != depth:
                    cur_depth = depth
                    left = pos
                ans = max(pos - left + 1, ans)
        return ans

    # 104. Check if a binary tree is complete
    def is_complete(self):
        if not self.root:
            return True
        queue = [self.root]
        flag = False
        while queue:
            node = queue.pop(0)
            if node.left:
                if flag:
                    return False
                queue.append(node.left)
            else:
                flag = True
            if node.right:
                if flag:
                    return False
                queue.append(node.right)
            else:
                flag = True
        return True

    # 105. Find the vertical sum of the binary tree
    def vertical_sum(self):
        sums = {}
        self._vertical_sum_recursive(self.root, 0, sums)
        return [sums[x] for x in sorted(sums)]

    def _vertical_sum_recursive(self, node, hd, sums):
        if not node:
            return
        sums[hd] = sums.get(hd, 0) + node.value
        self._vertical_sum_recursive(node.left, hd - 1, sums)
        self._vertical_sum_recursive(node.right, hd + 1, sums)

    # 106. Check if a binary tree is a BST
    def is_bst(self):
        return self._is_bst_recursive(self.root, float('-inf'), float('inf'))

    def _is_bst_recursive(self, node, min_val, max_val):
        if not node:
            return True
        if node.value <= min_val or node.value >= max_val:
            return False
        return (self._is_bst_recursive(node.left, min_val, node.value) and
                self._is_bst_recursive(node.right, node.value, max_val))

    # 107. Convert binary tree to its mirror
    def convert_to_mirror(self):
        self._convert_to_mirror_recursive(self.root)

    def _convert_to_mirror_recursive(self, node):
        if not node:
            return
        node.left, node.right = node.right, node.left
        self._convert_to_mirror_recursive(node.left)
        self._convert_to_mirror_recursive(node.right)

    # 108. Find the maximum element in the binary tree
    def find_max(self):
        return self._find_max_recursive(self.root)

    def _find_max_recursive(self, node):
        if not node:
            return float('-inf')
        return max(node.value, self._find_max_recursive(node.left), self._find_max_recursive(node.right))

    # 109. Print nodes at k distance from root
    def print_k_distance_nodes(self, k):
        self._print_k_distance_nodes_recursive(self.root, k)

    def _print_k_distance_nodes_recursive(self, node, k):
        if not node:
            return
        if k == 0:
            print(node.value, end=" ")
        else:
            self._print_k_distance_nodes_recursive(node.left, k-1)
            self._print_k_distance_nodes_recursive(node.right, k-1)

    # 110. Check if all leaf nodes are at same level
    def are_leaves_at_same_level(self):
        if not self.root:
            return True
        queue = [(self.root, 0)]
        leaf_level = None
        while queue:
            node, level = queue.pop(0)
            if not node.left and not node.right:
                if leaf_level is None:
                    leaf_level = level
                elif level != leaf_level:
                    return False
            if node.left:
                queue.append((node.left, level + 1))
            if node.right:
                queue.append((node.right, level + 1))
        return True