#  Class: blueprint for creating new objects
#  Object: instance of a class
#  think java
#  Pascal naming
#  variable and functions use underscore and lowercase
from collections import namedtuple
from abc import ABC, abstractmethod


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

# another = Point(3, 4)
# another.draw()

# o = Point.zero()
# o.y = 3
# y = Point.zero()

# print(another + o)
# o.draw()
# print(str(o))


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

#  Inheritance
#  DRY

# Inheritance can inherit methods and attributes


class Animal:
    def __init__(self):
        self.age = 1

    def eat(self):
        print("eat")


class Mammal(Animal):
    def walk(self):
        print("walk")


class Fish(Animal):
    def swim(self):
        print("swim")


# m = Mammal()

# m.eat()
# print(m.age)

#  Object Class

#  Mammal inherits from Animal class
#  An instance of the mammal class is also an instance of the animal class
#  Object class is the base class for all classes in python

#  Method Overriding
#  The constructor in subclasses overrides the base class
#  Use super keyword to access base class

class Mammal2(Animal):
    def __init__(self):
        self.weight = 2
        super().__init__()

    def walk(self):
        print("walk")


# m2 = Mammal2()
# print(m2.age)
# print(m2.weight)


# Multi-level inheritance
# too many level will make your program complicated
# focus on your problem
# Limit inheritance to 1-2 levels

class Bird(Animal):
    def fly(self):
        print("fly")


class Chicken(Bird):
    pass

# Multiple Inheritance


class Employee:
    def greet(self):
        print("Employee greet")


class Person:
    def greet(self):
        print("Person greet")


class Manager(Employee, Person):
    pass


# manager = Manager()
# manager.greet()

# python takes the method according to the order of the inheritance
# when you have thing in common, will be confusing

# This is a good example of multiple inheritance
# The flyer class and swimmer class do not have similar methods


class Flyer:
    def fly(self):
        pass


class Swimmer:
    def swim(self):
        pass


class FlyingFish(Flyer, Swimmer):
    pass

# A Good Example of Inheritance
#  Abstract base class, reinforce the methods needed to be overrided
#  Again, think java
#  Need to force a common interface/contract
#  Modelling from data


class InvalidOperationError(Exception):
    pass


class Stream(ABC):
    def __init__(self):
        self.opened = False

    def open(self):
        if self.opened:
            raise InvalidOperationError("Stream is already open")
        self.opened = True

    def close(self):
        if not self.opened:
            raise InvalidOperationError("Stream is already closed")
        self.opened = False

    @abstractmethod
    def read(self):
        pass


class FileStream(Stream):
    def read(self):
        print("Reading data from a file")


class NetworkStream(Stream):
    def read(self):
        print("Reading data from a network")


class MemoryStream(Stream):
    def read(self):
        print("Reading data from a memory stream")


# ms = MemoryStream()
# ms.open()
# print(ms.opened)


# Polymorphism
class UIControl(ABC):
    @abstractmethod
    def draw(self):
        pass


class TextBox(UIControl):
    def draw(self):
        print("Textbox")


class DropDownList(UIControl):
    def draw(self):
        print("DropDownList")

# the draw method has many forms depending on
# the type/class of the argument
# duck typing, python doesn't check for the type
# only checks if the item has the called function


def draw(controls):
    for control in controls:
        control.draw()


# ddl = DropDownList()
# tb = TextBox()
# draw([ddl, tb])

# print(isinstance(ddl, UIControl))

# Extending Built-in Types
class Text(str):
    def duplicate(self):
        return self + self


class TrackableList(list):
    def append(self, object):
        print("Append called")
        super().append(object)


# Data Classes
# However, namedtuples are immutable

Point1 = namedtuple("Point", ["x", "y"])
p1 = Point1(x=1, y=2)
# to change the value(tuples are immutable), create a new object
p1 = Point1(x=10, y=2)
