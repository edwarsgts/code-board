class Array():

    def __init__(self, __length=1):
        self.__length = __length
        self.items = []
        self.count = 0

    def get_length(self):
        return self.__length

    def print(self):
        for i in range(self.count):
            print(self.items[i])

    def insert(self, value):
        self.items.append(value)
        self.count += 1
        if self.count > self.__length:
            self.__length *= 2

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


class LinkedList():
    # Lookup is O(n)
    # Insert is O(1) at the beginning/end
    # Inserting in the middle O(n)
    # Deleting in the beginning O(1)
    # Deleting last item is o(n)
    # Deleting middle item is o(n)
    # Methods
    # add first, add last,
    # deleteLast, deleteFirst, remove(index)
    # contains(value), indexOf(value)
    # size() number of items in the list
    # toArray() convert the linkedlist into an array

    class __Node():
        def __init__(self, value):
            self.value = value
            self.next = None

        def __contains__(self, value):
            return value

        def __eq__(self, other):
            return self.value == other.value

        def __nonzero__(self):
            return self.value

    def __init__(self):
        self.first = None
        self.last = None
        self.count = 0

    def addLast(self, value):
        new_node = self.__Node(value)
        try:
            if not self.first:
                self.first = self.last = new_node
            else:
                self.last.next = new_node
                self.last = new_node
        finally:
            self.count += 1

    def addFirst(self, value):
        new_node = self.__Node(value)
        try:
            if not self.first:
                self.first = self.last = new_node

            new_node.next = self.first
            self.first = new_node
        finally:
            self.count += 1

    def indexOf(self, value):
        # return first occurence
        index = 0
        if not self.first:
            return -1
        pointer = self.first
        while pointer:
            if pointer.value == value:
                return index
            pointer = pointer.next
            index += 1
        else:
            return -1

    def contains(self, value):
        return self.indexOf(value) != -1

    def removeFirst(self):
        if not self.first:
            return "list is empty"

        try:

            if self.first == self.last:
                self.first = self.last = None
                return

            self.first = self.first.next
        finally:
            self.count -= 1

    def removeLast(self):
        if not self.first:
            return "list is empty"
        try:
            if self.first == self.last:
                self.first = self.last = None
                return

            pointer = self.getPreviousNode(self.last)
            self.last = pointer
            self.last.next = None
        finally:
            self.count -= 1

    def getPreviousNode(self, node):
        if node == self.first:
            return False

        pointer = self.first
        while pointer.next != node:
            pointer = pointer.next
        return pointer

    def toArray(self):
        result = []
        pointer = self.first
        while pointer:
            result.append(pointer.value)
            pointer = pointer.next
        return result

    def print(self):
        result = self.toArray()
        print(result)

    def size(self):
        return self.count

    def reversed(self):
        # I wrote this and this is slower
        pointer = self.last
        while self.getPreviousNode(pointer):
            node = self.getPreviousNode(pointer)
            node.next, pointer.next = None, node
            pointer = pointer.next
        self.first, self.last = self.last, pointer

    def reverse(self):
        # This is from the tutorial and is faster
        if not self.first:
            return

        prev = self.first
        current = self.first.next

        while current:
            nex = current.next
            current.next = prev
            prev = current
            current = nex

        self.last = self.first
        self.last.next = None
        self.first = prev

    def kth_node_from_the_end(self, k):
        if not self.first:
            return "List is empty"

        # if k > self.count:
        #     return "list smaller than target"

        distance = k - 1
        p1 = p2 = self.first

        while distance:
            p2 = p2.next
            if not p2:
                return "list smaller than target"
            distance -= 1
        while p2 != self.last:
            p1 = p1.next
            p2 = p2.next
        return p1.value
