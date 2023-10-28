"""类装饰器为什么需要使用__get__方法？
参考资料: https://www.zhihu.com/question/35957619/answer/2717043867 且听风吟

==================================
Author: Alan / Zeng Zhicun
Institution: CSU, China, changsha
Date: 2023/10/28
==================================
"""

import types
from functools import wraps
from pprint import pprint


class Profiled:
    def __init__(self, func):
        # print(func)
        # <function Spam.bar at 0x00000198FDBDA280>
        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        # 不使用描述符方法时：
        # <__main__.Profiled object at 0x0000015C40343CD0>
        # (3,)
        # {}
        # print(self)
        # print(args)
        # print(kwargs)
        """=================="""
        # 使用描述符方法时：
        # 使用types.MethodType(self, instance)
        # 生成一个types.MethodType对象，
        # 它把Spam对象绑定到Profiled对象的__call__函数上，之后调用就可以传入self了
        # 那么args中就有了self，如下
        # (<__main__.Spam object at 0x00000226ABC43E20>, 3)
        # print(args)
        return self.__wrapped__(*args, **kwargs)

    def __get__(self, instance, cls):
        print(f"__get__中的self {self}、instance {instance}和cls {cls}")
        if instance is None:
            return self
        else:
            return types.MethodType(self, instance)


class Spam:
    @Profiled
    def bar(self, x):
        print(f"bar 中的self {self}和x {x}")
        return x


if __name__ == '__main__':
    """
    经过类装饰器装饰之后，
    @Profiled
    def bar(self, x):
        print(self)
        print(self, x)
    相当于Spam 中多了一个类变量 : _ = Profiled(bar), 这里的bar是<function >
    """
    # <__main__.Profiled object at 0x000002E9BB4C3CD0>
    # 不会触发__get__
    pprint(Spam.__dict__['bar'])
    # or
    # 触发__get__, 通过类名可以访问到类变量 _ = Profiled(bar)
    pprint("通过类名访问：")
    pprint(Spam.bar)
    pprint("======")
    pprint("通过类实例访问：")
    s = Spam()
    """
    在Profiled初始化时, 类变量_没有记录Spam self的信息, __wrapped函数__函数也没有记录哪一个Spam的self
    调用顺序：s.bar -> Profiled(bar) -> Profiled.__call__ -> self.__wrapped__(即bar)
     -> 发现少了一个self，因为Profiled初始化的时候，压根没记录Spam的self的信息。
     因此，传到bar里面的其实就是一个x，没有self，所以会报错：缺少x参数
    """
    pprint(s.bar(3))
    pprint("======")
