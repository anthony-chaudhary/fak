---
title: "fak install paths: which one command do I run?"
description: "A decision table that routes each reader to their one correct fak install command: prebuilt binary, go install, from source, or MCP client. One verified command per row, with the honest caveat for each."
slug: install-paths
keywords:
  - install fak
  - go install fak
  - fak prebuilt binary
  - fak from source
  - fak mcp server
  - one static go binary
  - agent kernel install
date: 2026-07-03
---

# fak install paths: which one command do I run?

fak is one static Go binary that pulls in two `golang.org/x` modules and nothing
else — no Python, no CUDA toolchain. There is no wrong way
to get it, but there is a fastest way for each situation. Find your row, run the
one command, done. Every command here is the same one the
[getting-started guide](../../GETTING-STARTED.md) documents; this page just
routes you to yours.

| I want to... | Pick | The one command | Caveat |
|---|---|---|---|
| Just try it, no clone, no Go | Prebuilt binary | `curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh \| sh` | Linux/macOS only. Reads the script before piping to `sh` if you prefer; it is checksum-verified and honors `FAK_VERSION` / `FAK_INSTALL_DIR`. Windows: grab the `.zip` from the [latest release](https://github.com/anthony-chaudhary/fak/releases/latest) instead. |
| I already have Go | `go install` | `go install github.com/anthony-chaudhary/fak/cmd/fak@latest` | Needs Go 1.26+. Lands in `$(go env GOBIN)` or `$GOPATH/bin`, so that has to be on your `PATH`. `@latest` pins to the newest tag, not to trunk. |
| I want to hack on it | From source | `git clone https://github.com/anthony-chaudhary/fak.git && cd fak && go build -o fak ./cmd/fak` | Needs Go 1.26+ and the clone. On Windows build with `-o fak.exe` (the extensionless `-o fak` leaves a file cmd.exe / PowerShell cannot launch by name). This is the only path that lets you change the code. |
| I drive an MCP client (Claude Code, etc.) | MCP server | Paste [`examples/mcp/.mcp.json`](../../examples/mcp/.mcp.json) into your project root as `.mcp.json` | Needs the `fak` binary already installed (use one of the rows above first). The stdio transport needs no listener, no port, and no API key. Adjust the `--policy` path to a floor you trust. |

## If you are not sure

- Reach for the **prebuilt binary** if you want to run fak, not read its code.
  It is the shortest path from zero to a working `fak` command.
- Reach for **`go install`** if you have a Go toolchain already; it skips the
  clone and drops the binary straight onto your Go bin path.
- Reach for **from source** only when you intend to build, test, or change fak.
  It is the heaviest of the four and the only one that gives you an editable tree.
- Reach for the **MCP server** row after you have the binary, when the goal is to
  let an agent route its proposed tool calls through fak rather than to run fak
  yourself.

## Verify it worked

Whichever path you took, the same check confirms the binary runs:

```bash
fak help          # or ./fak help  /  .\fak.exe help  if it is not on your PATH
```

Then follow the tiers in the [getting-started guide](../../GETTING-STARTED.md):
Tier 0 is offline and needs nothing further; Tier 1 puts fak in front of a model
server you already run.

## What each path does not do

None of these install a model, a GPU driver, or a Python environment. fak is the
kernel; the model is separate and only some tiers need one (the getting-started
guide marks which). The MCP row does not start a network listener. The `go
install` and prebuilt-binary rows do not give you the source tree, so you cannot
edit fak from them. Pick the from-source row if that is what you are after.
