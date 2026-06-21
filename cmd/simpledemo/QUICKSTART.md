# 🚀 Quick Start - Chat with a Local AI in 30 Seconds

**One command. That's it.**

**From repo root:**
```bash
go -C fak run ./cmd/simpledemo -download
```

**Or from the fak directory:**
```bash
go run ./cmd/simpledemo -download
```

First run downloads a ~500MB model. Then just chat!

---

## What You'll See

```
╔════════════════════════════════════════════════════════════════════════╗
║     🤖 Simple Demo - Chat with a Local AI (No API Key Required!)      ║
╚════════════════════════════════════════════════════════════════════════╝

📥 Downloading model (first time only)...
   Model: Qwen2.5-0.5B-Instruct-Q8_0.gguf
   Downloading: 500.0 MB...
   ✅ 500.0 MB in 45.2s (11.1 MB/s)

📦 Loading model...
✅ Loaded qwen2.5 in 2.3s

💬 Chat with your AI! Type a message and press Enter.
   Commands: /clear = new chat, /exit = quit

You: Hello!
AI: Hello! How can I help you today?

You: What is 2+2?
AI: 2+2 equals 4.
```

---

## Commands

| Type | What Happens |
|------|--------------|
| Your message | AI responds |
| `/exit` | Quit |
| `/clear` | New conversation |

---

## Already Have a Model?

```bash
go run ./cmd/simpledemo
```

It will auto-detect models in:
- `~/.cache/fak-models/gguf/`
- `~/Downloads/`

---

## Having Trouble?

1. **Need a better model?** See [README.md](README.md)
2. **Using with Claude Code?** See [CLAUDE.md](CLAUDE.md)
3. **Something broken?** Check [README.md](README.md#troubleshooting)

---

**That's everything. Go chat! 🎉**
