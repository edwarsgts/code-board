from timeit import timeit


def findLucky(arr):
    """
    Given an array of integers arr, a lucky integer is an integer which has a frequency in the array equal to its value.
    Return a lucky integer in the array. If there are multiple lucky integers return the largest of them. If there is no lucky integer return -1.
    Example 1:
    Input: arr = [2,2,3,4]
    Output: 2
    Explanation: The only lucky number in the array is 2 because frequency[2] == 2.
    Example 2:
    Input: arr = [1,2,2,3,3,3]
    Output: 3
    Explanation: 1, 2 and 3 are all lucky numbers, return the largest of them.
    """
    cnt = [0] * 501
    for a in arr:
        cnt[a] += 1
    for i in range(500, 0, -1):
        if cnt[i] == i:
            return i
    return -1


def createTargetArray(nums, index):
    # 1389. Create Target Array in the Given Order
    target = []
    for item in zip(index, nums):
        target.insert(item[0], item[1])
    return target


def findTheDifference(s, t):
    # 389. Find the Difference
    for i in t:
        if s.count(i) != t.count(i):
            return i


def singleNumber(nums):
    import collections
    # 136. Single Number
    #
    # for i in set(nums):
    #     if nums.count(i) == 1:
    #         return i
    numsCount = collections.Counter(nums)
    for k, v in numsCount.items():
        if v == 1:
            return k


def missingNumber(self, nums):
    # 268. Missing Number
    expected_sum = len(nums)*(len(nums)+1)//2
    actual_sum = sum(nums)
    return expected_sum - actual_sum


def luckyNumbers(matrix):
    # 1380. Lucky Numbers in a Matrix
    # Hint 1 : find out and save the min of each row
    #  and max of each column in two lists
    min_row = {min(row) for row in matrix}
    max_column = {max(column) for column in zip(*matrix)}
    return list(min_row & max_column)


def lastStoneWeight(stones):
    # 1046. Last Stone weight
    while len(stones) > 1:
        stones = sorted(stones)
        y1 = stones.pop()
        y2 = stones.pop()
        if y1-y2 > 0:
            stones.append(y1-y2)
    if stones != []:
        return stones[0]
    else:
        return 0


def defangIPaddr(address):
    return address.replace(".", "[.]")


def smallerNumbersThanCurrent(nums):
    #  1365. How Many Numbers Are Smaller Than the Current Number
    #  self-written, slower
    result = []
    for i in nums:
        counter = 0
        clone = nums[:]
        clone.remove(i)
        for j in clone:
            if i > j:
                counter += 1
        result.append(counter)
    return result


def smallerNumbersThanCurrent_fast(nums):
    n = len(nums)
    if n == 0:
        return []
    count = {}
    for num in nums:
        count[num] = count.get(num, 0) + 1
    else:
        # take keys(input list) and sort
        keys = sorted(count.keys())
        # create new dict containing
        smaller = {keys[0]: 0}
        # because sorted so start from 1
        for i in range(1, len(keys)):
            smaller[keys[i]] = smaller[keys[i-1]] + count[keys[i-1]]
        else:
            for k, v in enumerate(nums):
                nums[k] = smaller[v]
            else:
                return nums


def findMostRepeartedChar(string):
    from pprint import pprint
    # calculate and put into dict
    charCount = {x: string.count(x) for x in set(string)}

    # check dictionary if needed
    # pprint(charCount, width=1)

    # sort dictionary by passing lamdbda that returns value in kv
    charCountSorted = sorted(
        charCount.items(),
        key=lambda kv: kv[1],
        reverse=True)

    # return first item in sorted dictionary
    return charCountSorted[0]


def num_jewels_in_stones(J, S):
    # def oneline_njis(J, S):
    # return sum(map(S.count, J))
    result = 0
    for jewel in J:
        result += S.count(jewel)
    return result

