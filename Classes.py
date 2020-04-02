#  Class: blueprint for creating new objects
#  Object: instance of a class
#  think java
#  Pascal naming
#  variable and functions use underscore and lowercase


class Point:
    # Class level attributes are shared and changed together
    default_color = "red"

    # Constructor
    # Self is a reference to the point object
    def __init__(self, x, y):
        self.x = x
        self.y = y

    #  magic methods
    def __str__(self):
        return f"({self.x},{self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    # defining greater than, python will automatically
    # intepret less than
    def __gt__(self, other):
        return self.x > other.x and self.y > other.y

    # arithmetic operations
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    # Factory method , method that creates an object
    @classmethod  # decorator
    def zero(cls):
        return cls(0, 0)

    #  Methods , in python it should at
    # least have the self parameter
    def draw(self):
        print(f"Point ({self.x}, {self.y})")


# Point.default_color = "Yellow"

# point = Point(1, 2)
# print(type(point))
# print(isinstance(point, Point))
# point.draw()
# point.z = 10  # additional parameters

another = Point(3, 4)
another.draw()

o = Point.zero()
o.y = 3
y = Point.zero()

print(another + o)
o.draw()
print(str(o))


class TagCloud:
    # private members but can still be accessed using different names
    def __init__(self):
        self.__tags = {}

    def add(self, tag):
        self.__tags[tag.lower()] = self.__tags.get(tag.lower(), 0) + 1

    def __getitem__(self, tag):
        return self.__tags.get(tag.lower(), 0)

    def __setitem__(self, tag, count):
        self.__tags[tag.lower()] = count

    def __len__(self):
        return len(self.__tags)

    def __iter__(self):
        return iter(self.__tags)


class Product:
    def __init__(self, price):
        self.price = price

    @property
    def price(self):
        return self.__price

    @price.setter
    def price(self, value):
        if value < 0:
            raise ValueError("Price cannot be negative.")
        self.__price = value
