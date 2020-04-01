"""
# LC day one : Single Number

import collections

def singleNumber(nums):
    num_frequency = collections.Counter(nums)
    for k, v in num_frequency.items():
        if v == 1:
            return k


test_input = [2, 2, 1]
print(singleNumber(test_input))
"""
