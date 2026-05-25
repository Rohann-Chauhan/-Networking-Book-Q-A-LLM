from tkinter import Tk
from tkinter.filedialog import askopenfilename

import os
import re
import fitz

from train import train_model
from generate import generate_text

# FILE PICKER

Tk().withdraw()

file_path = askopenfilename()

if not os.path.exists(file_path):

    print("❌ File Not Found")
    exit()

# PDF EXTRACT

doc = fitz.open(file_path)

all_text = ""

for page_num in range(len(doc)):

    page = doc[page_num]

    all_text += page.get_text() + "\n"

# CLEANING

all_text = all_text.replace("￾", " ")

all_text = re.sub(r"\s+", " ", all_text)

all_text = re.sub(r"\b\d+\b", "", all_text)

all_text = all_text.strip()

# SAVE CLEAN TEXT

with open(
    "clean_book.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(all_text)

# VOCAB

preprocessed = re.split(
    r'([,.:;?_!"()\']|--|\s)',
    all_text
)

tokens = []

for item in preprocessed:

    cleaned_item = item.strip()

    if cleaned_item != "":

        tokens.append(cleaned_item)

all_tokens = sorted(list(set(tokens)))

all_tokens.extend([
    "<unk>",
    "<pad>",
    "<start>",
    "<end>"
])

vocab = {

    token: integer

    for integer, token in enumerate(all_tokens)

}

print("📖 Vocabulary:", len(vocab))

# TRAIN

model = train_model(
    all_text,
    vocab
)

# CHAT

print("\n🚀 GPT READY")

while True:

    text = input("\nYou: ")

    if text.lower() == "exit":

        break

    output = generate_text(
        model,
        vocab,
        text
    )

    print("\nAI:", output)