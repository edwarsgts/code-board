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
