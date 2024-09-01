import os

HF_CACHE_DIR = "/home/students/kolber/seminars/kolber/.cache/"
os.environ["HF_HOME"] = HF_CACHE_DIR

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
)

import os
import random


def create_token_list(fp):
    token_list = []
    with open("/home/students/kolber/seminars/kolber/token_list.txt", "r") as file:
        for line in file:
            token_list.append(line.strip())
    return token_list


HF_CACHE_DIR = "/home/students/kolber/seminars/kolber/.cache/"
N = 10000
ARROW = "=>"

model = T5ForConditionalGeneration.from_pretrained("t5-large", cache_dir=HF_CACHE_DIR)
tokenizer = AutoTokenizer.from_pretrained("t5-large", cache_dir=HF_CACHE_DIR)

token_list = create_token_list("/home/students/kolber/seminars/kolber/token_list.txt")

for i in range(N):
    token1 = random.choice(token_list)
    token2 = random.choice(token_list)

    accuracy = 0

    for j in range(10):
        token3 = random.choice(token_list)
        inputs = tokenizer(
            f"""summarize: {token1}{ARROW}{token1};{token2}{ARROW}{token2};{token3}{ARROW}
            """,
            return_tensors="pt",
        )
        outputs = model.generate(**inputs)
        generated_seq = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        repeated_token = generated_seq.split("=>")[-1]
        if repeated_token == token3:
            accuracy += 1
    accuracy /= 10
    if accuracy == 1:
        print(f"Token1: {token1}, Token2: {token2}")
