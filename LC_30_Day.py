import collections

"""
# Day 1 : Single Number

import collections

def singleNumber(nums):
    num_frequency = collections.Counter(nums)
    for k, v in num_frequency.items():
        if v == 1:
            return k


# Day 2 :Happy Number

def isHappy(n):
    seen = set()
    result = 0
    while result != 1:
        result = 0
        while n > 0:
            result += (n % 10) * (n % 10)
            n = n // 10
        if result == 1:
            return True
        elif result in seen:
            return False
        else:
            n = result
            seen.add(n)

#  Day 3

# def max_subarray_algo(nums):
#     for i in range(1, len(nums)):
#         if nums[i-1] > 0:
#             nums[i] += nums[i-1]
#     return max(nums)


# def max_subarray(nums):
#     curSum = maxSum = nums[0]
#     for num in nums[1:]:
#         curSum = max(num, curSum + num)
#         maxSum = max(curSum, maxSum)
#     return maxSum


# def max_subarray_fast(nums):
#     length = len(nums)
#     if length == 1:
#         return nums[0]
#     globalmax = localmax = nums[0]
#     for i in range(1, length):
#         if localmax < 0:
#             localmax = nums[i]
#         else:
#             localmax = nums[i]+localmax
#         if localmax > globalmax:
#             globalmax = localmax
#     return globalmax

# Divide and Conquer
# public:
# int maxSubArray(int A[], int n) {
#     // IMPORTANT: Please reset any member data you declared, as
#     // the same Solution instance will be reused for each test case.
#     if(n==0) return 0;
#     return maxSubArrayHelperFunction(A,0,n-1);
# }

# int maxSubArrayHelperFunction(int A[], int left, int right) {
#     if(right == left) return A[left];
#     int middle = (left+right)/2;
#     int leftans = maxSubArrayHelperFunction(A, left, middle);
#     int rightans = maxSubArrayHelperFunction(A, middle+1, right);
#     int leftmax = A[middle];
#     int rightmax = A[middle+1];
#     int temp = 0;
#     for(int i=middle;i>=left;i--) {
#         temp += A[i];
#         if(temp > leftmax) leftmax = temp;
#     }
#     temp = 0;
#     for(int i=middle+1;i<=right;i++) {
#         temp += A[i];
#         if(temp > rightmax) rightmax = temp;
#     }
#     return max(max(leftans, rightans),leftmax+rightmax);
# }


Day 4

def move_zeroes(nums):
    for i in range(len(nums))[::-1]:
        print(i)
        if nums[i] == 0:
            nums.pop(i)
            nums.append(0)
    return


test_input = [0, 1, 0, 3, 12, 0, 0, 13]
move_zeroes(test_input)
print(test_input)


def move_zeroes_1(nums):
    left = right = 0
    while right < len(nums):
        if nums[right] != 0:
            nums[left] = nums[right]
            left += 1
        right += 1

    while left < len(nums):
        nums[left] = 0
        left += 1
    return

# Day 5 : Best time to sell stock 2

def max_profit_rough(prices):
    stock_on_hand = False
    buy_value = sell_value = profit = 0
    for i in range(len(prices)):
        if not stock_on_hand and i != len(prices)-1:
            if prices[i] == prices[i+1]:
                pass
            elif i == 0:
                if prices[i] < prices[i+1]:
                    buy_value = prices[i]
                    stock_on_hand = True
            elif prices[i] <= prices[i-1] and prices[i] < prices[i+1]:
                buy_value = prices[i]
                stock_on_hand = True
        elif stock_on_hand:
            sell_value = prices[i]
            if buy_value < sell_value:
                # Confirm selling at better price
                if i < len(prices)-1 and prices[i] > prices[i+1]:
                    profit += sell_value - buy_value
                    buy_value = 0
                    stock_on_hand = False
                # Sell if sell value larger and is last day of market
                elif i == len(prices)-1:
                    profit += sell_value - buy_value
                    buy_value = 0
                    stock_on_hand = False
    return profit


def max_profit_fastest(prices):
    # return = if only one price
    if len(prices) < 2:
        return 0

    diff_list = []

    for i in range(1, len(prices)):
        # if today price > ytd price
        if prices[i] > prices[i-1]:
            #
            diff_list.append(prices[i] - prices[i-1])
    return sum(diff_list)


def max_profit_memory(prices):
    if len(prices) < 2:
        return 0

    total_profit = 0
    current_price = float("inf")

    for price in prices:
        profit = price - current_price
        if profit > 0:
            total_profit += profit
        current_price = price

    return total_profit

Day 6 : Group Anagrams


def group_anagrams(strs):
    ans = collections.defaultdict(list)
    for s in strs:
        ans[tuple(sorted(s))].append(s)
    return list(ans.values())

def group_anagrams_2(strs):
    ans = collections.defaultdict(list)
    for s in strs:
        count = [0] * 26
        for ch in s:
            count[ord(ch)- ord('a')] += 1
        ans[tuple(count)].append(s)
    return ans.values()

# Day 7 : Count Elements

def count_elements(arr):
    num_freq = collections.Counter(arr)
    return sum(num_freq[x] for x in num_freq if x+1 in num_freq)

# Day 8 : Finding Middle Node of linked list

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


def middle_node(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    return slow

# Day 9: Backspace String Compare

def backspace_compare(S, T):
    if not S or not T or len(S) > 200 or len(T) > 200:
        return None

    s_list = []
    t_list = []
    for s in S:
        if s == '#':
            if not s_list:
                pass
            else:
                s_list.pop()
        else:
            s_list.append(s)
    for t in T:
        if t == '#':
            if not t_list:
                pass
            else:
                t_list.pop()
        else:
            t_list.append(t)
    return "".join(s_list) == "".join(t_list)


def backspace_compare_memory(S, T):
    import itertools

    def F(S):
        skip = 0
        for x in reversed(S):
            if x == '#':
                skip += 1
            elif skip:
                skip -= 1
            else:
                yield x
    return all(x == y for x, y in itertools.zip_longest(F(S), F(T)))

# Day 10: MinStack with push,pop,getMin,top

    # Class to make a Node
class Node:
    # Constructor which assign argument to nade's value
    def __init__(self, value):
        self.value = value
        self.next = None

    # This method returns the string representation of the object.
    def __str__(self):
        return "Node({})".format(self.value)

    # __repr__ is same as __str__
    __repr__ = __str__


class Stack:
    # Stack Constructor initialise top of stack and counter.
    def __init__(self):
        self.top = None
        self.count = 0
        self.minimum = None

    # This method returns the string representation of the object (stack).
    def __str__(self):
        temp = self.top
        out = []
        while temp:
            out.append(str(temp.value))
            temp = temp.next
        out = '\n'.join(out)
        return ('Top {} \n\nStack :\n{}'.format(self.top, out))

    # __repr__ is same as __str__
    __repr__ = __str__

    # This method is used to get minimum element of stack
    def getMin(self):
        if self.top is None:
            return "Stack is empty"
        else:
            print("Minimum Element in the stack is: {}" .format(self.minimum))

    # Method to check if Stack is Empty or not

    def isEmpty(self):
        # If top equals to None then stack is empty
        if self.top == None:
            return True
        else:
            # If top not equal to None then stack is empty
            return False

    # This method returns length of stack
    def __len__(self):
        self.count = 0
        tempNode = self.top
        while tempNode:
            tempNode = tempNode.next
            self.count += 1
        return self.count

    # This method returns top of stack
    def peek(self):
        if self.top is None:
            print("Stack is empty")
        else:
            if self.top.value < self.minimum:
                print("Top Most Element is: {}" .format(self.minimum))
            else:
                print("Top Most Element is: {}" .format(self.top.value))

    # This method is used to add node to stack
    def push(self, value):
        if self.top is None:
            self.top = Node(value)
            self.minimum = value

        elif value < self.minimum:
            temp = (2 * value) - self.minimum
            new_node = Node(temp)
            new_node.next = self.top
            self.top = new_node
            self.minimum = value
        else:
            new_node = Node(value)
            new_node.next = self.top
            self.top = new_node
        print("Number Inserted: {}" .format(value))

    # This method is used to pop top of stack
    def pop(self):
        if self.top is None:
            print("Stack is empty")
        else:
            removedNode = self.top.value
            self.top = self.top.next
            if removedNode < self.minimum:
                print("Top Most Element Removed :{} " .format(self.minimum))
                self.minimum = ((2 * self.minimum) - removedNode)
            else:
                print("Top Most Element Removed : {}" .format(removedNode))


# Driver program to test above class
stack = Stack()

stack.push(3)
stack.push(5)
stack.getMin()
stack.push(2)
stack.push(1)
stack.getMin()
stack.pop()
stack.getMin()
stack.pop()
stack.peek()

# This code is contributed by Blinkii

Day 11 : Diameter of Binary Tree

def diameter_of_binary_tree(self, root):
    self.ans = 1

    def depth(node):
        # base condition
        if node == None:
            return 0

        L = depth(node.left)
        R = depth(node.right)
        self.ans = max(self.ans, L+R+1)
        return max(L, R) + 1
    depth(root)
    # returns edges instead of the no. of nodes
    return self.ans - 1


# Day 13: Longest Contiguous Array

def find_max_length(nums):
    #  add entry for initial count = 0 and index -1
    table = {0: -1}
    #  initialize max_length and count variable
    maxlen = count = 0

    for index, num in enumerate(nums):
        count += num or -1
        if count in table:
            maxlen = max(maxlen, index-table.get(count))
        else:
            table[count] = index
    return maxlen
"""

"""
Day 14: Perform String Shifts


def string_shift(s, shift):
    # left shift +ve, right shift -ves
    count = 0
    for direction, amount in shift:
        if direction:
            count -= amount
        else:
            count += amount
    # if count > length of string, cycled, modulus
    # can take care of that
    count %= len(s)
    return s[count:] + s[:count]


Day 15:


def productExceptSelf_time_exceeded(nums):
    output = []
    i = 0
    while (len(nums) > i):
        result = 1
        for j in range(1, len(nums)):
            result *= nums[j]
        output.append(result)
        nums.append(nums.pop(0))
        i += 1
    return output


def productExceptSelf(nums):
    output = []
    num_freq = collections.Counter(nums)
    for num in nums:
        result = 1
        num_freq[num] -= 1
        for k, v in num_freq.items():
            result *= (k**v)
        output.append(result)
        num_freq[num] += 1
    return output


def productExceptSelf_algo1(nums):
    # The length of the input array
    length = len(nums)

    # The left and right arrays as described in the algorithm
    L, R, answer = [0]*length, [0]*length, [0]*length

    # L[i] contains the product of all the elements to the left
    # Note: for the element at index '0', there are no elements to the left,
    # so the L[0] would be 1
    L[0] = 1
    for i in range(1, length):

        # L[i - 1] already contains the product of elements to the left of 'i - 1'
        # Simply multiplying it with nums[i - 1] would give the product of all
        # elements to the left of index 'i'
        L[i] = nums[i - 1] * L[i - 1]

    # R[i] contains the product of all the elements to the right
    # Note: for the element at index 'length - 1', there are no elements to the right,
    # so the R[length - 1] would be 1
    R[length - 1] = 1
    for i in reversed(range(length - 1)):

        # R[i + 1] already contains the product of elements to the right of 'i + 1'
        # Simply multiplying it with nums[i + 1] would give the product of all
        # elements to the right of index 'i'
        R[i] = nums[i + 1] * R[i + 1]

    # Constructing the answer array
    for i in range(length):
        # For the first element, R[i] would be product except self
        # For the last element of the array, product except self would be L[i]
        # Else, multiple product of all elements to the left and to the right
        answer[i] = L[i] * R[i]

    return answer


def productExceptSelf_algo2(nums):
    # The length of the input array
    length = len(nums)

    # The answer array to be returned
    answer = [0]*length

    # answer[i] contains the product of all the elements to the left
    # Note: for the element at index '0', there are no elements to the left,
    # so the answer[0] would be 1
    answer[0] = 1
    for i in range(1, length):

        # answer[i - 1] already contains the product of elements to the left of 'i - 1'
        # Simply multiplying it with nums[i - 1] would give the product of all
        # elements to the left of index 'i'
        answer[i] = nums[i - 1] * answer[i - 1]

    # R contains the product of all the elements to the right
    # Note: for the element at index 'length - 1', there are no elements to the right,
    # so the R would be 1
    R = 1
    for i in reversed(range(length)):

        # For the index 'i', R would contain the
        # product of all elements to the right. We update R accordingly
        answer[i] = answer[i] * R
        R *= nums[i]

    return answer

Dat 16 : Valid Parentheses String

def check_valid_string(s):
    lo = hi = 0
    for c in s:
        lo += 1 if c == '(' else -1
        hi += 1 if c != ')' else -1
        if (hi < 0):
            break
        lo = max(lo, 0)
    return lo == 0


test_input = ')('
print(check_valid_string(test_input))

Day 17: Is island

def num_of_islands(grid):
    def depth_first_search(grid, i, j):
        # base condition, if out of border(i=j = 0 or larger than len(i) or len(j)) OR the cell is not '1'
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != '1':
            return
        # changes the 1 to #
        grid[i][j] = '#'
        # run through the grid for neighbouring 1
        depth_first_search(grid, i+1, j)
        depth_first_search(grid, i-1, j)
        depth_first_search(grid, i, j+1)
        depth_first_search(grid, i, j-1)
    if not grid:
        return 0

    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                # if found '1', then pass the grid and the current postion to the function
                depth_first_search(grid, i, j)
                count += 1
    return count

Day 18: Min Path Sum

def min_path_sum(grid):
    # finds the min sum to reach each cell other than the first cell
    # slowly add up and finally returning the value of the destination cell
    m, n = len(grid), len(grid[0])
    for i in range(1, n):
        grid[0][i] += grid[0][i-1]
    for i in range(1, m):
        grid[i][0] += grid[i-1][0]
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i-1][j], grid[i][j-1])
    return grid[-1][-1]


test_cases = [[[1, 3, 1],
               [1, 5, 1],
               [4, 2, 1]]]


Day 19 : Binary search in rotated sorted list
def search(nums, target):
    if not nums:
        return -1

    low, high = 0, len(nums) - 1

    while low <= high:
        mid = (low + high) // 2

        if target == nums[mid]:
            # if target is the middle of our search, return
            return mid

        if nums[low] <= nums[mid]:
            # if nums between num[low] and mid is sorted
            if nums[low] <= target <= nums[mid]:
                # check if target between low and mid
                high = mid - 1
            else:
                # target is between mid and high
                low = mid + 1

        else:
            # If num between low and mid not sorted
            # then num between mid and high must be sorted
            if nums[mid] <= target <= nums[high]:
                # check if target between mid and high
                low = mid+1
            else:
                # target is in low and mid
                high = mid - 1

    return -1


test_cases = [[6, 7, 0, 1, 2, 4, 5], 0]

Day 20: Reconstruct a bst from a preorder traversal list

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def bstFromPreorder(preorder):
    import bisect

    def helper(i, j):
        if i == j:
            return None
        root = TreeNode(preorder[i])
        mid = bisect.bisect(preorder, preorder[i], i+1, j)
        root.left = helper(i+1, mid)
        root.right = helper(mid, j)
        return root
    return helper(0, len(preorder))

test_case = [8,5,1,7,10,12]

Day 21 :  Leftmost column with one


def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
    row, column = binaryMatrix.dimensions()
    i = 0
    j = column - 1
    column_index = -1 
        
    while (i < row and j >= 0):
        if binaryMatrix.get(i,j) == 0:
            i += 1
        else: 
            column_index = j
            j -= 1
        
    return column_index

Day 22: sub array sum equals k

def subarraySum(nums, k):
    d = {0: 1}
    total = count = 0
    for num in nums:
        total += num
        if total - k in d:
            count += d[total-k]

        d[total] = d.get(total, 0) + 1
    return count


test_case = [2, 2, 3, 1]
target = 4

Day 23: Bitwise AND of numbers range

def rangeBitwiseAnd(self, m: int, n: int) -> int:
        i = 0
        while m != n:
            m >>= 1
            n >>= 1
            i += 1
        return n << i

"""
