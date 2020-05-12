#!/usr/bin/python          
# -*- coding: UTF-8 -*-

# file_name: 说明文档
# author: catherine.qin
# create: 2020/5/12 5:49 下午


'''

Request对象的重要属性如下所列：


method - 当前请求方法。Form - 它是一个字典对象，包含表单参数及其值的键和值对。
args - 解析查询字符串的内容，它是问号（？）之后的URL的一部分。
Cookies  - 保存Cookie名称和值的字典对象。
files - 与上传文件有关的数据。


可以利用Flask所基于的Jinja2模板引擎的地方。而不是从函数返回硬编码HTML，可以通过render_template()函数呈现HTML文件

Flask类有一个redirect()函数。调用时，它返回一个响应对象，并将用户重定向到具有指定状态代码的另一个目标位置。
redirect()函数的原型如下：
    Flask.redirect(location, statuscode, response)
在上述函数中：
    location参数是应该重定向响应的URL。
    statuscode发送到浏览器标头，默认为302。
    response参数用于实例化响应。






'''