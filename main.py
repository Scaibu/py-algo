# This is a sample Python script.
from heap.max_heap import MaxHeap
from heap.min_heap import MinHeap
from tree.tree import BinaryTree


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def quick_sort_dp(arr, memo=None):
    if memo is None:
        memo = {}

    # Convert the array to a tuple (immutable) to use as a key in the memo dictionary
    arr_tuple = tuple(arr)

    if arr_tuple in memo:
        return memo[arr_tuple]

    if len(arr) <= 1:
        memo[arr_tuple] = arr
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    # Sort left and right partitions, using memoization
    sorted_left = quick_sort_dp(left, memo)
    sorted_right = quick_sort_dp(right, memo)

    # Combine sorted subarrays and memoize the result
    sorted_arr = sorted_left + middle + sorted_right
    memo[arr_tuple] = sorted_arr

    return sorted_arr

def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)


# Example usage:
def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    max_heap = MaxHeap()
    max_heap.insert(10)
    max_heap.insert(20)
    max_heap.insert(5)
    max_heap.insert(30)

    print("Max value:", max_heap.get_max())  # Output: 30
    print("Extracted max:", max_heap.extract_max())  # Output: 30
    print("Max value after extraction:", max_heap.get_max())  # Output: 20

    min_heap = MinHeap()
    min_heap.insert(10)
    min_heap.insert(20)
    min_heap.insert(5)
    min_heap.insert(30)

    print("Min value:", min_heap.get_min())  # Output: 5
    print("Extracted min:", min_heap.extract_min())  # Output: 5
    print("Min value after extraction:", min_heap.get_min())  # Output: 10



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
