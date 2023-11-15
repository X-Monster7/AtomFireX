"""


================
@Author: zhicun Zeng / Alan
@Date: 2023/11/14 17:27
================
"""


class Decorator1:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print("Decorator 1")
        return self.func(*args, **kwargs)


def decorator2(func):
    def wrapper(*args, **kwargs):
        print("Decorator 2")
        return func(*args, **kwargs)

    return wrapper


@Decorator1
@decorator2
def my_function():
    print("Hello, world!")


my_function()
