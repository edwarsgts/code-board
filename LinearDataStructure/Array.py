class Array():
    items = []
    count = 0

    def __init__(self, length):
        self.length = length

    def print(self):
        for i in self.items:
            print(i)

    def insert(self, value):
        self.items.append(value)
        self.count += 1
        if self.count > self.length:
            self.length *= 2

    def removeAt(self, index):
        if index > self.length:
            print("index not available")
        else:
            removed = self.items.pop(index)
            self.count -= 1
            return removed

    def indexOf(self, value):
        if value in self.items:
            return self.items.index(value)
        else:
            return -1
