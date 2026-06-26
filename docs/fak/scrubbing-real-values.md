---
title: "Keeping real hosts and credentials out of the public tree"
description: "The convention for public-safe placeholder defaults plus machine-local overrides, and the scrub/leak-scan tooling that enforces it."
---

# Scrubbing real values

`fak` is a **public** GitHub repo. Real tailnet IPs, SSH usernames, internal
hostnames, model node names, and credentials must never land in tracked source —
not even in a default, a doc example, or a test fixture. This page is the
convention every contributor (human or agent) follows so those values stay out
of git while the tools still work on a real machine.

## The pattern: public-safe default + machine-local override

Tracked source ships a **placeholder default** that is obviously not real and
does not resolve. The real value lives in a **gitignored local file** the user
supplies. The placeholder is the public-safe value *by policy* — restoring a
real value into the tracked default is a regression, not a fix.

`fak claude-mac-fak` is the worked example:

| Layer | Value | Where |
| --- | --- | --- |
| Tracked default | `node-macos-a.local`, `user@node-macos-a.local` | `cmd/fak/claude_mac_fak.go` |
| Override (env) | `FAK_MAC_GATEWAY`, `FAK_MAC_SSH_HOST`, `FAK_MAC_MODEL`, `FAK_GATEWAY_KEY` | your shell |
| Machine-local file | real tailnet host + user | `fak-mac.local.ps1` (gitignored) |
| Public template | placeholders only | `fak-mac.local.ps1.example` (tracked) |

So "restore the default" means **restore your working default via the override**
(dot-source `fak-mac.local.ps1`), not edit the placeholder in tracked source.

```powershell
cp fak-mac.local.ps1.example fak-mac.local.ps1   # edit with real values
. .\fak-mac.local.ps1                             # sets FAK_MAC_* for this shell
go run ./cmd/fak claude-mac-fak
```

The cleanest skip is `FAK_GATEWAY_KEY` set directly — then the in-binary ssh
key-fetch (and its unresolved-placeholder failure) never runs at all.

## How values are kept out of git

`.gitignore` already excludes the local-override families; this is where a new
secret-bearing local file goes:

```
.env  .env.*           # !.env.example stays tracked
*.local.json  *.local.yaml  *.local.yml
*.local.ps1  *.local.sh      # !*.local.ps1.example / !*.local.sh.example tracked
*.pem  *.key  *.p12
```

Rule of thumb: **a real value goes in a `*.local.*` (or `.env`) file; a tracked
`*.example` carries placeholders only.** Name the local file so one of the
patterns above already ignores it — do not invent a new path and hope.

## The enforcing tooling

These exist; run the relevant one before committing anything that *could* carry a
real value (a new default, a runbook, a fixture):

- `tools/leak_scan.py` — scans the working tree for leaked hosts/credentials.
- `tools/history_leak_audit.py` — scans git **history** (a scrub is only real if
  the value was never committed; if it was, history must be rewritten).
- `tools/scrub_public_copy.py` — produces a public-safe copy with reals removed.
- `tools/scrub_hardware_names.py` — replaces internal hardware/node names.

A scrub that leaves the value in history is not a scrub. For
`claude-mac-fak` the real IP was **never committed** — verify the same for any
value you replace (`git log -S "<value>"` returns nothing) before trusting the
placeholder.

## Checklist for adding a new real-value surface

1. Ship a placeholder default in tracked source (must not resolve / obviously fake).
2. Read the override from an env var (`FAK_*`) **and** the placeholder, env wins.
3. Put real values in a `*.local.*` / `.env` file; add a tracked `*.example` twin.
4. `git log -S "<real value>"` is empty; `tools/leak_scan.py` is clean.
5. Never restore a real value into the tracked default to "fix" a resolve error —
   point the user at the override instead.
