# Using with Claude Code (or Claude Desktop)

Run a local AI model as a backend for Claude Code or Claude Desktop. **No API key required.**

## Quick Start

### Step 1: Start the fak server

```powershell
.\fak.exe serve --addr 127.0.0.1:8137 `
  --gguf ~/.cache/fak-models/gguf/Qwen2.5-1.5B-Instruct-Q8_0.gguf `
  --model qwen2.5-1.5b
```

Linux/macOS:
```bash
./fak serve --addr 127.0.0.1:8137 \
  --gguf ~/.cache/fak-models/gguf/Qwen2.5-1.5B-Instruct-Q8_0.gguf \
  --model qwen2.5-1.5b
```

### Step 2: Configure Claude Code

Edit your Claude Code config (`~/.claude/settings.json`):

```json
{
  "providers": [
    {
      "name": "local-fak",
      "baseApiUrl": "http://127.0.0.1:8137",
      "apiKey": "dummy-key",
      "models": [
        {
          "id": "qwen2.5-1.5b",
          "displayName": "Local Qwen 1.5B"
        }
      ]
    }
  ]
}
```

### Step 3: Use it!

In Claude Code:
```
/provider local-fak
/model qwen2.5-1.5b
```

Now all your conversations run through your local model.

---

## Verify It's Working

Test the server:
```powershell
curl http://127.0.0.1:8137/healthz
```

Should return:
```json
{"engine":"inkernel","model":"qwen2.5-1.5b","ok":true}
```

---

## Model Recommendations

| For... | Use this model | Size | RAM |
|--------|----------------|------|-----|
| **Testing** | Qwen2.5-0.5B-Q8 | 500MB | 2GB |
| **Daily use** | Qwen2.5-1.5B-Q8 | 1.6GB | 3GB |
| **Better quality** | Qwen2.5-3B-Q8 | 3.2GB | 5GB |

Download from [HuggingFace](https://huggingface.co/models?search=gguf qwen2.5 instruct).

---

## Tips for Best Results with Small Models

### 1. Keep System Messages Short

❌ Too long (confuses small models):
```
You are a highly sophisticated AI assistant designed to help users with
a wide variety of tasks including coding, writing, analysis, and more...
```

✅ Better:
```
You are a helpful assistant. Give clear, concise answers.
```

### 2. One Thing at a Time

❌ Too complex:
```
Analyze this code, fix all bugs, add tests, write documentation, and
create a deployment pipeline.
```

✅ Better:
```
Review this function for bugs: [paste code]
```

### 3. Use Lower Temperature

For factual/technical work, use lower temperature:
```json
{
  "temperature": 0.3
}
```

### 4. Clear Context Periodically

Small models have smaller context windows. Start fresh conversations occasionally.

---

## What Works Well

| Task | How to Ask |
|------|------------|
| **Code help** | "Write a Python function to reverse a list" |
| **Debugging** | "Find the bug in this code: [paste]" |
| **Explanations** | "Explain REST APIs like I'm 12" |
| **Drafting** | "Draft an email requesting a meeting" |
| **Summaries** | "Summarize this text: [paste]" |

## What Doesn't Work Well (Yet)

Small models struggle with:
- Complex multi-step reasoning
- Very long codebases
- Nuanced creative writing
- Factual knowledge beyond training data

For these, you'd want a larger model (7B+) or cloud API.

---

## Troubleshooting

### "Could not connect to server"

1. Make sure `fak serve` is running
2. Check the port: `curl http://127.0.0.1:8137/healthz`
3. Verify the URL in your config matches

### "Model not found"

1. Download a model from HuggingFace
2. Save to `~/.cache/fak-models/gguf/`
3. Restart `fak serve`

### Responses seem off

0. Make sure you're on a build with the NEOX-rope GGUF fix and the embedded-tokenizer
   serve fallback — without them, Qwen/Gemma replies were gibberish or canned regardless
   of the knobs below.
1. Try a smaller system prompt
2. Use temperature 0.3-0.5
3. Start a fresh conversation
4. Try a larger model if you have RAM

---

## Performance Expectations

On a typical laptop (8-16GB RAM):

| Model | First Response | Subsequent | Quality |
|-------|----------------|------------|---------|
| 0.5B | ~3 seconds | ~20-50 tok/s | Good |
| 1.5B | ~5 seconds | ~15-30 tok/s | Better |
| 3B | ~10 seconds | ~8-15 tok/s | Best |

First response is slower because the model needs to "read" your prompt (prefill).

---

## Why Use a Local Model?

| Pro | Con |
|-----|-----|
| ✅ Privacy (data never leaves) | ❌ Slower than cloud |
| ✅ No API costs | ❌ Smaller models |
| ✅ Works offline | ❌ Less capable |
| ✅ No rate limits | ❌ Uses your RAM |

Perfect for: private code, sensitive documents, offline work, experimentation.
