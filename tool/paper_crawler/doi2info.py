# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/9/19 10:21
@version: 1.0.0
"""

import requests
import re


def doi2info(doi: str):
    # crossref通过doi查询论文信息的URL
    url = 'https://api.crossref.org/works/{}'
    # 发送查询请求
    response = requests.get(url.format(doi), timeout=5000)

    # 检查响应是否成功
    if response.status_code == 200:
        # 成功获取响应
        data = response.json()  # 假设响应是JSON格式的数据
        # \/ 用来匹配斜杠 / 字符，而 ? 表示匹配前面的元素零次或一次。
        # 因此，\/? 匹配零个或一个斜杠字符。
        abstract = re.sub(r'<\/?jats:p>', '',
                          data['message']['abstract'])
        print(abstract)
    else:
        # 响应失败
        print(f"doi2info请求失败，状态码：{response.status_code}")
