# 🤖 Simple Demo - Chat with a Local AI

The friendliest way to run an AI model on your own computer. **No API key, no cloud, no cost.**

## Quick Start

**If you have a .gguf model:**
```bash
go run ./cmd/simpledemo
```

The demo auto-finds models in these locations:
- `C:\Users\You\models\*.gguf` (Windows)
- `~/.cache/fak-models/gguf/*.gguf`
- `~/Downloads/*.gguf`

**First time? No model?**

The demo will show you exactly how to get one. Just run:
```bash
go run ./cmd/simpledemo
```

---

## What You'll See

```
🤖 Found model: Qwen2.5-0.5B-Instruct-Q8_0.gguf

📦 Loading model...
✅ Loaded Qwen2.5-0.5B in 0.8s
🎯 Temperature: 0.5 | Max tokens: 128

💬 Chat with your AI! Type a message and press Enter.
   Commands: /clear = new chat, /exit = quit

You: What is the capital of France?
AI: The capital of France is Paris.

📊 15 tok in, 8 tok out (12.5 tok/s) | 1.3s total
```

---

## Commands

| Command | What It Does |
|---------|--------------|
| `/exit` or `/quit` | Quit the demo |
| `/clear` | Start a fresh conversation |

---

## Model Recommendations

| Model | Size | Quality | Speed | RAM |
|-------|------|---------|-------|-----|
| **0.5B Q8** | 500MB | Good | ⚡⚡⚡ | 2GB |
| **1.5B Q8** | 1.6GB | Better | ⚡⚡ | 3GB |
| **3B Q4_K_M** | 2GB | Best | ⚡ | 5GB |
| **27B Q4_K_M** | 16GB | Excellent | Needs GPU | 16GB+ |

**Download from:** [HuggingFace](https://huggingface.co/models?search=gguf qwen2.5 instruct)

Save to `C:\Users\You\models\` (Windows) or `~/models/` (Linux/Mac).

---

## Advanced Usage

```powershell
# Use a specific model
.\simpledemo.exe -gguf C:\path\to\model.gguf

# Adjust response length
.\simpledemo.exe -n 256

# Change creativity (temperature)
.\simpledemo.exe -temp 0.3  # Focused
.\simpledemo.exe -temp 0.9  # Creative

# Custom system prompt
.\simpledemo.exe -sys "You are a coding expert. Be concise."
```

---

## Tips for Small Models

1. **Keep prompts short** - One question at a time
2. **Be specific** - "Write a function to sort a list" beats "Help me code"
3. **Use `/clear`** - Start fresh if the model gets confused
4. **Lower temperature** - Use `-temp 0.3` for factual answers

---

## Troubleshooting

### "No model found"

Place a `.gguf` file in one of these locations:
- Windows: `C:\Users\You\models\`
- Linux/Mac: `~/models/`
- Or anywhere, then use: `-gguf /path/to/model.gguf`

### "Tokenizer not found"

Rare — the demo uses the tokenizer **embedded in the `.gguf`** by default, so most models
need no separate file. You only hit this if your GGUF embeds no usable tokenizer; then
download `tokenizer.json` from the same HuggingFace page as your model and save it next to
the `.gguf` file (or pass `-tok /path/to/dir`).

### Slow responses

- Try a smaller model (0.5B is fastest)
- Close other programs to free RAM
- First response is always slower (model "reads" your prompt)

### Garbled or repetitive output

Make sure you're on a build that includes the NEOX-rope GGUF fix: Qwen/Gemma/Phi GGUFs
used to decode as repetitive token-salad ("mand mand…") before it, and no temperature
setting fixes that. The same model/build mismatch makes the reply collapse into a loop
(`2 2 2 …` when sampling, `.assistant.assistant…` under greedy `-temp 0`) — issue #91.

The demo now **detects** that case and prints a `⚠️ That reply looks degenerate` warning
so a first run never silently hands you gibberish. On a fixed build the reply is coherent;
if output still looks off, try a lower temperature: `-temp 0.3`.

The greedy non-degeneracy guard is regression-tested. With a model on disk:

```bash
FAK_SIMPLEDEMO_GGUF="$HOME/.cache/fak-models/gguf/Qwen2.5-1.5B-Instruct.Q8_0.gguf" \
    go -C fak test ./cmd/simpledemo/ -run TestGreedyNonDegenerate -v
```

(The detector's own unit tests run with no model: `go -C fak test ./cmd/simpledemo/`.)

---

## Behind the Scenes

This demo uses **fak's in-kernel model engine**:

- Model runs inside the same process
- No external server needed
- Full ChatML prompt formatting
- Quantized (Q8) for speed while keeping quality

---

## What's Next?

- Explore the full fak system: `fak/GETTING-STARTED.md`
- Learn about tool-calling agents: `docs/cli-reference.md`
- Try different models from [HuggingFace](https://huggingface.co/models?search=gguf qwen2.5 instruct)
