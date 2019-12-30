#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/6/20 16:35
# @Author  : Guicheng.Zhou
# @File    : ExceptionClass.py
# @Software: IntelliJ IDEA
# @Describe: 异常类

# 定义两个类处理意外
class TooManyFaces(Exception):
    def __init__(self):
        Exception.__init__(self, "Too Many Faces!")


class NoFaces(Exception):
    def __init__(self):
        Exception.__init__(self, "No Face!")
