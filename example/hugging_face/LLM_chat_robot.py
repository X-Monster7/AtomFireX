#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Alan Tsang / Zeng Zhicun
@institution: CSU, China, changsha
@date: 2023/10/13 12:25
@version: 1.0.0
"""
import os
import io
import IPython.display
from PIL import Image
import base64
import requests

requests.adapters.DEFAULT_TIMEOUT = 60

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
hf_api_key = os.environ['HF_API_KEY']

# Helper function
import requests, json
from text_generation import Client

# FalcomLM-instruct endpoint on the text_generation library
client = Client(os.environ['HF_API_FALCOM_BASE'], headers = {"Authorization": f"Basic {hf_api_key}"}, timeout = 120)

prompt = "Has math been invented or discovered?"
client.generate(prompt, max_new_tokens = 256).generated_text

# Back to Lesson 2, time flies!
import gradio as gr


def generate(input, slider):
    output = client.generate(input, max_new_tokens = slider).generated_text
    return output


demo = gr.Interface(
    fn = generate, inputs = [gr.Textbox(label = "Prompt"), gr.Slider(
        label = "Max new tokens", value = 20, maximum = 1024, minimum = 1
    )], outputs = [gr.Textbox(label = "Completion")]
)
gr.close_all()
demo.launch(share = True, server_port = int(os.environ['PORT1']))

import random


def respond(message, chat_history):
    # No LLM here, just respond with a random pre-made message
    bot_message = random.choice(
        ["Tell me more about it",
         "Cool, but I'm not interested",
         "Hmmmm, ok then"]
    )
    chat_history.append((message, bot_message))
    return "", chat_history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height = 240)  # just to fit the notebook
    msg = gr.Textbox(label = "Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components = [msg, chatbot], value = "Clear console")

    btn.click(respond, inputs = [msg, chatbot], outputs = [msg, chatbot])
    msg.submit(respond, inputs = [msg, chatbot], outputs = [msg, chatbot])  # Press enter to submit
gr.close_all()
demo.launch(share = True, server_port = int(os.environ['PORT2']))


def format_chat_prompt(message, chat_history, instruction):
    prompt = f"System:{instruction}"
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt


def respond(message, chat_history, instruction, temperature = 0.7):
    prompt = format_chat_prompt(message, chat_history, instruction)
    chat_history = chat_history + [[message, ""]]
    stream = client.generate_stream(
        prompt,
        max_new_tokens = 1024,
        stop_sequences = ["\nUser:", "<|endoftext|>"],
        temperature = temperature
    )
    # stop_sequences to not generate the user answer
    acc_text = ""
    # Streaming the tokens
    for idx, response in enumerate(stream):
        text_token = response.token.text

        if response.details:
            return

        if idx == 0 and text_token.startswith(" "):
            text_token = text_token[1:]

        acc_text += text_token
        last_turn = list(chat_history.pop(-1))
        last_turn[-1] += acc_text
        chat_history = chat_history + [last_turn]
        yield "", chat_history
        acc_text = ""


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height = 240)  # just to fit the notebook
    msg = gr.Textbox(label = "Prompt")
    with gr.Accordion(label = "Advanced options", open = False):
        system = gr.Textbox(
            label = "System message", lines = 2,
            value = "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."
        )
        temperature = gr.Slider(label = "temperature", minimum = 0.1, maximum = 1, value = 0.7, step = 0.1)
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components = [msg, chatbot], value = "Clear console")

    btn.click(respond, inputs = [msg, chatbot, system], outputs = [msg, chatbot])
    msg.submit(respond, inputs = [msg, chatbot, system], outputs = [msg, chatbot])  # Press enter to submit
gr.close_all()
demo.queue().launch(share = True, server_port = int(os.environ['PORT4']))


def format_chat_prompt(message, chat_history, instruction):
    prompt = f"System:{instruction}"
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt


def respond(message, chat_history, instruction, temperature = 0.7):
    prompt = format_chat_prompt(message, chat_history, instruction)
    chat_history = chat_history + [[message, ""]]
    stream = client.generate_stream(
        prompt,
        max_new_tokens = 1024,
        stop_sequences = ["\nUser:", "<|endoftext|>"],
        temperature = temperature
    )
    # stop_sequences to not generate the user answer
    acc_text = ""
    # Streaming the tokens
    for idx, response in enumerate(stream):
        text_token = response.token.text

        if response.details:
            return

        if idx == 0 and text_token.startswith(" "):
            text_token = text_token[1:]

        acc_text += text_token
        last_turn = list(chat_history.pop(-1))
        last_turn[-1] += acc_text
        chat_history = chat_history + [last_turn]
        yield "", chat_history
        acc_text = ""


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height = 240)  # just to fit the notebook
    msg = gr.Textbox(label = "Prompt")
    with gr.Accordion(label = "Advanced options", open = False):
        system = gr.Textbox(
            label = "System message", lines = 2,
            value = "A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers."
        )
        temperature = gr.Slider(label = "temperature", minimum = 0.1, maximum = 1, value = 0.7, step = 0.1)
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components = [msg, chatbot], value = "Clear console")

    btn.click(respond, inputs = [msg, chatbot, system], outputs = [msg, chatbot])
    msg.submit(respond, inputs = [msg, chatbot, system], outputs = [msg, chatbot])  # Press enter to submit
gr.close_all()
demo.queue().launch(share = True, server_port = int(os.environ['PORT4']))
