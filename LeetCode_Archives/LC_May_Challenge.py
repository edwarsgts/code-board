#  May LeetCode Challenge Archives

# Day 1 : First Bad Version


def first_bad_version_0(n):
    # Using Binary Search
    left = 1
    right = n
    while left < right:
        mid = (left + right) // 2
        if isBadVersion(mid):
            right = mid
        else:
            left = mid + 1
    return left


# def first_bad_version_1(n):
#     import bisect
#     # using Bisect
#     self.__getitem__ = isBadVersion
#     return bisect.bisect(self, True, 1, n)


def isBadVersion():
    # function provided by question
    return

# Day 2 : Jewels in Stones


def num_jewels_in_stones(J, S):
    result = 0
    for jewel in J:
        result += S.count(jewel)
    return result

# Day 3: Ransom Note


def can_construct(ransomNote, magazine):
    for ch in ransomNote:
        if ch not in magazine:
            return False
        magazine = magazine.replace(ch, "", 1)
    return True

#  Day 4: Number Complement


def find_complement(num):
    b_num = bin(num)
    res = ''.join(['0' if r == '1' else '1' for r in b_num[2:]])
    return int(res, 2)

# Day 5 First Unique Character in a String


def first_uniq_char(s):
    import collections
    freq = collections.Counter(s)
    for k, v in freq.items():
        if v == 1:
            return s.index(k)
    return -1

# Day 6 : Majority Element


def majority_element(nums):
    # if count of item is > len (num)/2,
    # it is consiered a majority element
    nums_set = set(nums)
    for i in nums_set:
        if nums.count(i) > len(nums)/2:
            return i

# Day 7 : Cousins of a Binary Tree


def is_cousin(root, x, y):
    def depth(root, x, par, level):
        if not root:
            return

        if root.val == x:
            return [level, par]

        return depth(root.left, x, root, level+1) or depth(root.right, x, root, level+1)

        a = depth(root, x, None, 1)
        b = depth(root, y, None, 1)
        if a[0] == b[0] and a[1] != b[1]:
            return True
        return False

# Day 8: Check if it is a straight line
def check_straight_line(coordinates):
    #  To check if it is a straight line, using
    # gradient to check, the expression is:
    # (y - y1) / (x - x1) = (y1 - y0) / (x1 - x0)
    # But to avoid zero division error, use multiplication form:
    # dx * (y - y1) = dy * (x - x1), where dx = x1 - x0 and dy = y1 - y0
    (x0, y0), (x1, y1) = coordinates[: 2]
    for x, y in coordinates:
        if (x1 - x0) * (y - y1) != (x - x1) * (y1 - y0):
            return False
    return True

def check_straight_line_2(coordinates):
    (x0, y0), (x1,y1) = coordinates[:2]
    return all((x1-x0) * (y - y1) == (x-x1) * (y1 - y0) for x,y in coordinates)