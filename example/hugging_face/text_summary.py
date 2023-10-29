#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/10/12 12:52
@version: 1.0.0
"""
import os
import gradio as gr
from transformers import pipeline
from dotenv import load_dotenv, find_dotenv


def summarize(input_text):
    return model(input_text)[0]['summary_text']


if __name__ == "__main__":
    # 使用clash代理软件和代码配置系统代理环境，解决网络问题
    _ = load_dotenv(find_dotenv("./env/summary.env"))

    model = pipeline("summarization")
    gr.close_all()
    demo = gr.Interface(
        fn = summarize,
        inputs = "text",
        outputs = "text",
    )
    demo.launch(share = True)
