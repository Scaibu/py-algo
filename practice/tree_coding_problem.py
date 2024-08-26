from typing import Optional

class TreeNode:
    def __init__(self, value: int):
        self.value: int = value
        self.left: Optional[TreeNode] = None
        self.right: Optional[TreeNode] = None

class BinaryTree:
    def __init__(self):
        self.root: Optional[TreeNode] = None

    def insert(self, value: int) -> None:
        if self.root is None:
            self.root = TreeNode(value)
        else:
            self._insert_recursively(self.root, value)

    def _insert_recursively(self, node: TreeNode, value: int) -> None:
        if node.value > value:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._insert_recursively(node.left, value)
        else:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._insert_recursively(node.right, value)

    def inorder(self, node: Optional[TreeNode]) -> None:
        if node is None:
            return
        self.inorder(node.left)
        print(node.value)
        self.inorder(node.right)

    def search(self, value: int) -> Optional[TreeNode]:
        return self._search_recursively(self.root, value)

    def _search_recursively(self, node: Optional[TreeNode], value: int) -> Optional[TreeNode]:
        if node is None or node.value == value:
            return node
        elif value < node.value:
            return self._search_recursively(node.left, value)
        else:
            return self._search_recursively(node.right, value)

    def print_tree(self):
        self._print_tree(self.root, "", True)

    def _print_tree(self, node: TreeNode, prefix: str, is_left: bool):
        if node is not None:
            if node.right is not None:
                self._print_tree(node.right, prefix + ("│   " if is_left else "    "), False)
            print(prefix + ("└── " if is_left else "┌── ") + str(node.value))
            if node.left is not None:
                self._print_tree(node.left, prefix + ("    " if is_left else "│   "), True)

if __name__ == '__main__':
    bt = BinaryTree()
    bt.insert(5)
    bt.insert(10)
    bt.insert(9)
    bt.insert(20)
    bt.insert(24)
    bt.insert(28)
    bt.insert(15)
    bt.insert(25)
    bt.print_tree()

