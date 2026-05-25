import torch
import torch.nn.functional as F

from tokenizer import Simpletokenizer

from utils import causal_mask

from config import *

def generate_text(

    model,
    vocab,
    prompt,
    max_new_tokens=80,
    temperature=0.8,
    top_k=40,
    repetition_penalty=1.2

):

    model.eval()

    tokenizer = Simpletokenizer(vocab)

    idx = tokenizer.encode(prompt)

    idx = torch.tensor(
        [idx],
        dtype=torch.long
    ).to(device)

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -max_seq_len:]

        mask = causal_mask(
            idx_cond.size(1)
        ).to(device)

        with torch.no_grad():

            logits = model(
                idx_cond,
                mask
            )

        logits = logits[:, -1, :]

        for token_id in set(idx[0].tolist()):

            logits[:, token_id] /= repetition_penalty

        logits = logits / temperature

        top_k_values, top_k_indices = torch.topk(
            logits,
            top_k
        )

        probs = F.softmax(
            top_k_values,
            dim=-1
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        next_token = top_k_indices.gather(
            -1,
            next_token
        )

        idx = torch.cat(
            [idx, next_token],
            dim=1
        )

    return tokenizer.decode(
        idx[0].tolist()
    )