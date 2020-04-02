
"""
# Exercise
# find most repeated character

from pprint import pprint
sentence = "This is a common interview question"

#  calculate and put into dict
charCount = {x: sentence.count(x) for x in set(sentence)}

# pprint(charCount, width=1)
# print(max(charCount, key=lambda parameter_list: expression))

# sort the dictionary, pass lambda that returns value
charCountSorted = sorted(
    charCount.items(),
    key=lambda kv: kv[1],
    reverse=True)

# return first item in descending dictionary
print(charCountSorted[0])


# Section: Exceptions

try:
    age = int(input("Age :"))
except ValueError as ex:
    print("You didn't enter a valid age")
    print(ex)
    print(type(ex))
else:
    # if no exception, else block code will be executed
    print("No exceptions were thrown")
    # similar to for else loops

# Handling different exceptions
try:
    age = int(input("Age :"))
    xfactor = 10/age
except (ValueError, ZeroDivisionError):
    print("You didn't enter a valid age")
else:
    print("No exceptions were thrown")

# python only xecutes the first except clause

#  Cleaning up
try:
    # the with will close the file after execution"
    with open("app.py") as file:
        print("File opened.")
    age = int(input("Age :"))
    xfactor = 10/age
except (ValueError, ZeroDivisionError):
    print("You didn't enter a valid age")
else:
    print("No exceptions were thrown")


# Raising exceptions

def calculate_xfactor(age):
    if age <= 0:
        raise ValueError("Age cannot be 0 or less.")
        #  similar to throw in java
        #  raising exceptions is costly
        #  can refer to documentation of python's exception
    return 10/age

try:
    calculate_xfactor(-1)
except ValueError as error:
    print(error)

"""

#  Cost of Raising exceptions

# from timeit import timeit

# code1 = """
# def calculate_xfactor(age):
#     if age <= 0:
#         raise ValueError("Age cannot be 0 or less.")
#         #  similar to throw in java
#         #  raising exceptions is costly
#         #  can refer to documentation of python's exception
#     return 10/age

# try:
#     calculate_xfactor(-1)
# except ValueError as error:
#     pass
# """

# code2 = """
# def calculate_xfactor(age):
#     if age <= 0:
#         return None
#         #  similar to throw in java
#         #  raising exceptions is costly
#         #  can refer to documentation of python's exception
#     return 10/age

# xfactor = calculate_xfactor(-1)
# if xfactor == None:
#     pass
# """

# print("first code", timeit(code1, number=10000))
# print("second code", timeit(code2, number=10000))
# #  returns the performance of running the code 10000 times
# #  simple functions will not be affected but
# #  still it is better to not raise exceptions
# #  unless necessary
