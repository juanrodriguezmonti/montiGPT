"""Monti GPT. A small GPT trained on Python functions (CodeSearchNet + synthetic).

Created by : Juan Rodríguez Monti.
rmontijuan@gmail.com
This is released under the GPL v2.

Entry points: ``--train``, ``--test``, and ``--both`` for training,
testing, or both.
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

# CONFIG

MODEL_PATH = "mini_gpt_code.pt"
TOKENS_CACHE = "tokens_cache.pt"

BLOCK_SIZE = 128
EMBED_SIZE = 384
HEADS = 6
LAYERS = 6

BATCH_SIZE = 32
STEPS = 8000
MAX_SECONDS = 3600
CHECKPOINT_EVERY = 500

WARMUP_STEPS = 200
LR = 5e-4
MIN_LR = 5e-5
GRAD_CLIP = 1.0

DATASET_FRACTION = "train[:10%]"
VAL_FRACTION = 0.05

TOP_K = 20
TEMPERATURE = 0.4
MAX_NEW_TOKENS = 120

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

torch.manual_seed(42)

# TOKENIZER

tokenizer = AutoTokenizer.from_pretrained("gpt2")


def encode(x):
    """Return GPT-2 token ids for *x* (``str`` or list of str)."""
    return tokenizer.encode(x)


def decode(x):
    """Decode token ids to text; *x* is a list of ids or a single id."""
    return tokenizer.decode(x)


VOCAB_SIZE = tokenizer.vocab_size

# DATASET


def load_data():
    """Load or build a 1D token tensor; chunk by ``BLOCK_SIZE`` and EOS.

    Reuses on-disk ``TOKENS_CACHE`` when present. Filters raw snippets
    (length, imports, classes, etc.) and appends synthetic examples.
    """
    if os.path.exists(TOKENS_CACHE):
        return torch.load(TOKENS_CACHE)

    dataset = load_dataset(
        "code_search_net", "python", split=DATASET_FRACTION
    )

    texts = []

    for item in dataset:
        code = item["whole_func_string"]

        if not code:
            continue
        if len(code) > 300:
            continue
        if "import" in code:
            continue
        if "class " in code:
            continue
        if "self" in code:
            continue
        if '"""' in code:
            continue
        if "return" not in code:
            continue

        texts.append(code)

    # 🔥 sintético balanceado
    synthetic = [
        """def suma(a, b):
    return a + b
""",
        """def resta(a, b):
    return a - b
""",
        """def es_par(n):
    return n % 2 == 0
""",
        """def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
""",
    ]

    texts.extend(synthetic * 100)

    print(f"[INFO] total funciones: {len(texts)}")

    tokens = []
    sep = tokenizer.eos_token_id

    for t in texts:
        enc = encode(t)

        for i in range(0, len(enc), BLOCK_SIZE):
            tokens.extend(enc[i : i + BLOCK_SIZE])
            tokens.append(sep)

    data = torch.tensor(tokens)
    torch.save(data, TOKENS_CACHE)

    print(f"[INFO] tokens: {len(data)}")
    return data


def split_data(data):
    """Split *data* into (train, validation) using ``VAL_FRACTION``."""
    n = int(len(data) * (1 - VAL_FRACTION))
    return data[:n], data[n:]


def get_batch(data):
    """A random batch: *x* and *y* shifted by one token."""
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack(
        [data[i + 1 : i + BLOCK_SIZE + 1] for i in ix]
    )
    return x.to(DEVICE), y.to(DEVICE)

# MODEL


class Block(nn.Module):
    """Transformer block: causal self-attention, residual paths, and GELU FFN."""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(EMBED_SIZE)
        self.ln2 = nn.LayerNorm(EMBED_SIZE)

        self.qkv = nn.Linear(EMBED_SIZE, EMBED_SIZE * 3)
        self.proj = nn.Linear(EMBED_SIZE, EMBED_SIZE)

        self.ff = nn.Sequential(
            nn.Linear(EMBED_SIZE, EMBED_SIZE * 4),
            nn.GELU(),
            nn.Linear(EMBED_SIZE * 4, EMBED_SIZE),
        )

    def forward(self, x):
        """
        *x* has shape (B, T, C): multi-head causal attention, residuals, FFN.
        Returns a tensor of the same shape.
        """
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)

        att = F.scaled_dot_product_attention(
            q.view(B, T, HEADS, -1).transpose(1, 2),
            k.view(B, T, HEADS, -1).transpose(1, 2),
            v.view(B, T, HEADS, -1).transpose(1, 2),
            is_causal=True,
        )

        att = att.transpose(1, 2).contiguous().view(B, T, C)

        x = x + self.proj(att)
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """Causal LM: token + position embeddings, ``Block`` stack, weight-tied head."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)
        self.pos = nn.Embedding(BLOCK_SIZE, EMBED_SIZE)

        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.ln = nn.LayerNorm(EMBED_SIZE)

        self.head = nn.Linear(EMBED_SIZE, VOCAB_SIZE, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x, targets=None):
        """
        *x* (B, T) token indices; *targets* same shape for the loss.

        Returns ``(logits, loss)``; *loss* is None when *targets* is None.
        """
        B, T = x.shape

        tok = self.embed(x)
        pos = self.pos(torch.arange(T, device=x.device))
        x = tok + pos

        for b in self.blocks:
            x = b(x)

        x = self.ln(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        return logits, loss

# LR


def lr_at(step):
    """Learning rate: linear warmup, then cosine decay down to ``MIN_LR``."""
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / (STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))

# SAVE / LOAD


def save_model(model, step):
    """Save model state and training step to ``MODEL_PATH``."""
    torch.save(
        {"model": model.state_dict(), "step": step},
        MODEL_PATH,
    )
    print(f"[INFO] saved step {step}")


def load_model():
    """Load weights and step from disk, or a fresh model at step 0."""
    model = MiniGPT().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state["model"])
        return model, state["step"]
    return model, 0

# TRAIN


def train(model, train_data, start_step):
    """Train until ``STEPS`` or ``MAX_SECONDS``; write checkpoints and final state."""
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    start = time.time()

    for step in range(start_step, STEPS):

        if time.time() - start > MAX_SECONDS:
            print("⏹️ corte por tiempo")
            save_model(model, step)
            return

        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr

        xb, yb = get_batch(train_data)
        _, loss = model(xb, yb)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        if step % 100 == 0:
            print(f"[TRAIN] {step} loss {loss.item():.4f}")

        if step % CHECKPOINT_EVERY == 0:
            save_model(model, step)

# GENERATION


def sample(logits):
    """Sample one id from top-*k* logits (with temperature)."""
    logits = logits / TEMPERATURE
    values, _ = torch.topk(logits, TOP_K)
    min_topk = values[:, -1].unsqueeze(-1)
    logits[logits < min_topk] = -float("inf")
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)


@torch.no_grad()
def generate(model, prompt):
    """Autoregressive text generation; stops on a double newline in a piece."""
    model.eval()
    idx = torch.tensor(encode(prompt)).unsqueeze(0).to(DEVICE)

    out = prompt

    for _ in range(MAX_NEW_TOKENS):
        logits, _ = model(idx[:, -BLOCK_SIZE:])
        nxt = sample(logits[:, -1, :])
        token = decode([nxt.item()])
        out += token
        idx = torch.cat([idx, nxt], dim=1)
        if "\n\n" in token:
            break

    return out

# EVAL


def evaluate_code(code):
    """(True, None) if *code* runs without error; else (False, error message)."""
    try:
        exec(code, {})
        return True, None
    except Exception as e:
        return False, str(e)


def test_generation(model):
    """Run fixed *prompts* through generation and print whether ``exec`` succeeds."""
    prompts = [
        "def suma(a, b):",
        "def es_par(n):",
        "def factorial(n):",
        "def maximo(a, b):",
        "def promedio(lista):",
    ]

    for p in prompts:
        print("\n--- TEST ---")
        code = generate(model, p)
        print(code)

        ok, err = evaluate_code(code)

        if ok:
            print("OK")
        else:
            print("Failed", err)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--both", action="store_true")
    args = parser.parse_args()

    model, step = load_model()

    if args.train or args.both:
        data = load_data()
        train_data, _ = split_data(data)
        train_data = train_data.to(DEVICE)
        train(model, train_data, step)

    if args.test or args.both:
        test_generation(model)
