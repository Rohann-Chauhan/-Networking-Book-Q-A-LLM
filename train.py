import os
import torch
import torch.nn as nn

from tqdm import tqdm

from config import *
from model import GPTModel
from dataset import create_dataloader_v1
from utils import causal_mask

def train_model(
    all_text,
    vocab
):

    train_loader, val_loader = create_dataloader_v1(
        all_text,
        vocab
    )

    model = GPTModel(
        len(vocab)
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    ckpt_path = "checkpoint.pt"

    start_epoch = 0

    if os.path.exists(ckpt_path):

        checkpoint = torch.load(
            ckpt_path,
            map_location=device
        )

        model.load_state_dict(
            checkpoint["model_state"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state"]
        )

        start_epoch = checkpoint["epoch"] + 1

        print(
            f"🔁 Resuming Epoch {start_epoch}"
        )

    for epoch in range(
        start_epoch,
        epochs
    ):

        print(
            f"\n🚀 Epoch {epoch+1}/{epochs}"
        )

        model.train()

        total_train_loss = 0

        train_bar = tqdm(train_loader)

        for x, y in train_bar:

            x = x.to(device)

            y = y.to(device)

            mask = causal_mask(
                x.size(1)
            ).to(device)

            output = model(
                x,
                mask
            )

            loss = loss_fn(

                output.view(
                    -1,
                    len(vocab)
                ),

                y.reshape(-1)

            )

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_train_loss += loss.item()

            train_bar.set_postfix(
                loss=loss.item()
            )

        avg_train_loss = (
            total_train_loss
            / len(train_loader)
        )

        # VALIDATION

        model.eval()

        total_val_loss = 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)

                y = y.to(device)

                mask = causal_mask(
                    x.size(1)
                ).to(device)

                output = model(
                    x,
                    mask
                )

                loss = loss_fn(

                    output.view(
                        -1,
                        len(vocab)
                    ),

                    y.reshape(-1)

                )

                total_val_loss += loss.item()

        avg_val_loss = (
            total_val_loss
            / len(val_loader)
        )

        scheduler.step()

        print(
            f"🔥 Train Loss: {avg_train_loss:.4f}"
        )

        print(
            f"🧾 Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:

            best_val_loss = avg_val_loss

            torch.save(
                model.state_dict(),
                "best_model.pt"
            )

            print("✅ Best Model Saved")

        torch.save({

            "epoch": epoch,

            "model_state": model.state_dict(),

            "optimizer_state": optimizer.state_dict()

        }, ckpt_path)

        print("💾 Checkpoint Saved")

    return model