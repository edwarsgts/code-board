import random
import string

print(random.random())
print(random.choices([1, 2, 31, 2, 31, 32, 123, 1, 23, 13, 1], k=4))
print("".join(random.choices(string.ascii_letters+string.digits, k=4)))

# random.shuffle(list)
