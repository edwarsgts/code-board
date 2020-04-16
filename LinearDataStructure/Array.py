class Array():

    def __init__(self, length=0):
        self.length = length
        self.items = []
        self.count = 0

    def print(self):
        for i in range(self.count):
            print(self.items[i])

    def insert(self, value):
        self.items.append(value)
        self.count += 1
        if self.count > self.length:
            self.length *= 2

    def removeAt(self, index):
        if index >= self.count or index < 0:
            raise Exception(
                f"index not valid. Index needs to be larger than 0 and lower than {self.count}")
        else:
            removed = self.items.pop(index)
            self.count -= 1
            return removed

    def indexOf(self, value):
        if value in self.items:
            return self.items.index(value)
        else:
            return -1

    def max_val(self):
        import math
        max_val = -math.inf
        for item in self.items:
            if item > max_val:
                max_val = item
        return max_val

    def intersect(self, other):
        intersection = []
        for item in self.items:
            if item in other.items:
                intersection.append(item)
        return intersection

    def reverse(self):
        start, end = 0, self.count - 1
        while start > end:
            self.items[start], self.items[end] = self.items[end], self.items[start]
            start += 1
            end -= 1

    def insertAt(self, value, index):
        if index < self.count:
            self.items.append(None)
            for i in reversed(range(index, self.count)):
                self.items[i+1] = self.items[i]
            self.items[index] = value
            self.count += 1
        else:
            while index >= self.count:
                self.items.append(0)
                self.count += 1
            else:
                self.items[index] = value
