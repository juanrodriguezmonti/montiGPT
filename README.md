# MontiGPT — A real and small GPT you can actually understand

Inspired by the “build it from scratch” philosophy of Andrej Karpathy, this project is my take on a **fully working, minimal-but-real GPT model** — designed not just to run, but to *teach*.

Not a toy. Not a wrapper.  
A real Transformer you can read end-to-end.

And yes — it actually generates code.

---

## Why this exists

Modern models like GPT-4 feel like black boxes.

This project is the opposite.

The goal is simple:

> Build a GPT-style model that is small enough to understand, but powerful enough to be interesting.

---

## What it can do

- Generate **coherent Python functions**
- Learn real syntax and patterns (not just character noise)
- Complete code from partial prompts
- Produce surprisingly clean outputs for its size

### Example output

    def factorial(n):
        if n == 0:
            return 1
        return n * factorial(n - 1)

---

## ⚙️ Features

### Real Transformer architecture
- Multi-head self-attention  
- Residual connections  
- MLP blocks  
- Layer normalization  

### Proper tokenization
- Uses GPT-2 tokenizer (BPE)  
- No character-level shortcuts  
- Handles real code structure  

### Smart dataset design
- Based on CodeSearchNet  
- Filtered for clarity and signal  
- Augmented with synthetic examples to reinforce patterns  

### Modern training setup
- Learning rate warmup  
- Cosine decay schedule  
- Gradient clipping  

### Controlled generation
- Temperature sampling  
- Top-k filtering  

---

## Design philosophy

This is not about scale.

This is about **clarity**.

Every part of the model is:
- Readable  
- Hackable  
- Modifiable  

You can trace:
- how tokens become embeddings  
- how attention moves information  
- how gradients update weights  

No abstractions hiding the important parts.

---

## What this is (and isn’t)

| ✅ This is | ❌ This is not |
|----------|--------------|
| A real GPT-style model | A production LLM |
| A learning tool | A Copilot replacement |
| Small but meaningful | Massive or state-of-the-art |
| Transparent | Black box magic |

---

##  What you can learn from this

- How embeddings actually work  
- How attention is computed step by step  
- How training loops shape behavior  
- How small models can still generalize  

If you've ever thought:

> “I know what a Transformer is… but I don’t *really* get it”

This project is for you.

---

## Why I built this

Because understanding beats memorizing.

Because scaling laws are cool, but intuition is better.

And because at some point, I wanted you to stop *using* models —  
and start actually *understanding* them.

---

## Contributing / Feedback

If you:
- want to extend it  
- break it  
- optimize it  
- or just understand it better  

feel free to open an issue or reach out.

---

## Final note

This won’t replace GPT-4.

But it *will* help you understand why models like that even work.

And once you see that clearly…  
you’re playing a completely different game.
