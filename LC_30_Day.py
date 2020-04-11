import collections

"""
# Day 1 : Single Number

import collections

def singleNumber(nums):
    num_frequency = collections.Counter(nums)
    for k, v in num_frequency.items():
        if v == 1:
            return k

"""

# Day 2 :Happy Number

"""
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
"""

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

"""
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

"""

"""
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

"""
"""
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


"""

"""
Day 7 : Count Elements

def count_elements(arr):
    num_freq = collections.Counter(arr)
    return sum(num_freq[x] for x in num_freq if x+1 in num_freq)

"""

"""
Day 8 : Finding Middle Node of linked list

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

"""
"""

Day 9: Backspace String Compare

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

    """

"""
Day 10: MinStack with push,pop,getMin,top

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

    """
