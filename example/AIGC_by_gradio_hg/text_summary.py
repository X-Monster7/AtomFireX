#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/10/12 12:52
@version: 1.0.0
"""
from transformers import pipeline
import os

os.environ["http_proxy"] = "http://127.0.0.1:1080"
os.environ["https_proxy"] = "http://127.0.0.1:1080"


def summarize(input_text):
    def get_completion():
        return pipeline("summarization")

    output = get_completion()
    return output[0]['summary_text']


if __name__ == "__main__":
    text = ('''The tower is 324 metres (1,063 ft) tall, about the same height
            as an 81-storey building, and the tallest structure in Paris. 
            Its base is square, measuring 125 metres (410 ft) on each side. 
            During its construction, the Eiffel Tower surpassed the Washington 
            Monument to become the tallest man-made structure in the world,
            a title it held for 41 years until the Chrysler Building
            in New York City was finished in 1930. It was the first structure 
            to reach a height of 300 metres. Due to the addition of a broadcasting 
            aerial at the top of the tower in 1957, it is now taller than the 
            Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the 
            Eiffel Tower is the second tallest free-standing structure in France 
            after the Millau Viaduct.''')

    print(summarize(text))
