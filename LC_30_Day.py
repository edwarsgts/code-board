"""
# LC day one : Single Number

import collections

def singleNumber(nums):
    num_frequency = collections.Counter(nums)
    for k, v in num_frequency.items():
        if v == 1:
            return k

"""

# LC day two :Happy Number

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
