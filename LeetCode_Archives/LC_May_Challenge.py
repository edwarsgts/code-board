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

# Day 3 Ransom Note


def can_construct(ransomNote, magazine):
    for ch in ransomNote:
        if ch not in magazine:
            return False
        magazine = magazine.replace(ch, "", 1)
    return True
