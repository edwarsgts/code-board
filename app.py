
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
"""

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
    with open("app.py") as file:
        print("File opened.")
    age = int(input("Age :"))
    xfactor = 10/age
except (ValueError, ZeroDivisionError):
    print("You didn't enter a valid age")
else:
    print("No exceptions were thrown")
