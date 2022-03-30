#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @author ZSQ
# @date 2022/3/30
# @file test.py
def getkeys(d,value):
    return [k for k,v in d.items() if v==value]
temp = {'a':'001','b':'002'}
print(getkeys(temp,'001'))