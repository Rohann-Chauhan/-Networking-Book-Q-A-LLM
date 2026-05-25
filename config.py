import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# MODEL CONFIG

d_model = 384
num_heads = 6
num_layers = 6
d_ff = 1536

dropout = 0.1

max_seq_len = 256

batch_size = 8

epochs = 24

learning_rate = 3e-4

stride = 128