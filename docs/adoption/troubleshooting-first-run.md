---
title: "It didn't work: the five most likely fak first-run failures"
description: "The first five things that go wrong the first time you run fak — binary not on PATH, base URL pointed at the wrong place (or empty, so you hit the offline mock), the upstream key env not set for a keyed provider, the default-deny policy refusing a tool call you expected, and the serve port already in use — each with the exact symptom, the cause, and the one-line fix. fak is one static Go binary; most first-run failures are wiring, not the binary."
slug: troubleshooting-first-run
keywords:
  - fak troubleshooting
  - first run
  - fak serve
  - base url
  - api key env
  - policy denied
  - address already in use
  - static go binary
date: 2026-07-17
---

# It didn't work

fak is **one static Go binary** — no runtime, no Python, no CUDA. So the first-run
failures are almost never the binary; they are wiring: where the binary is, where it
points, what key it forwards, what the policy allows, and which port is free. Here are
the five most likely, each with the symptom you see, the cause, and the fix.

If your symptom is a build/toolchain one instead (`go.mod requires go >= 1.26`,
`fetch-model.sh: need python3`), see the build table in
[`GETTING-STARTED.md`](../../GETTING-STARTED.md#troubleshooting) — this page is about the
first *run*.

## 1. `fak: command not found` — the binary isn't on PATH

**Symptom.** `fak: command not found` (macOS/Linux) or `'fak' is not recognized as an
internal or external command` (Windows).

**Cause.** You built `fak` in the repo but it isn't on your `PATH`. The build drops the
binary in the working directory, not a system bin dir.

**Fix.** Run it by path from the repo — `./fak serve …` — or put it on PATH. On Windows
the dogfood installer does the lock-safe copy + shim for you:
`scripts/dogfood-claude.ps1 --install` (writes `fak.exe` to `<home>\bin`). Then a fresh
shell finds `fak` anywhere.

## 2. Connection refused, 404s, or canned replies — the base URL is wrong (or empty)

**Symptom.** `fak serve` starts, but requests fail with connection-refused / 404, or the
replies look scripted and identical no matter what you ask.

**Cause.** `--base-url` must point at an **OpenAI-compatible `/v1` endpoint** of your
upstream (e.g. Ollama at `http://localhost:11434/v1`, or a hosted provider). Two traps:
pointing it at the host root instead of the `/v1` path, or leaving `--base-url` **empty**
— with no base URL the gateway runs **offline against a scripted mock**, which is why the
replies look canned.

**Fix.** Pass the full `/v1` URL and confirm the upstream is actually up first:

```
curl -sf http://localhost:11434/api/tags >/dev/null   # is the upstream bound?
./fak serve --addr 127.0.0.1:8080 --base-url http://localhost:11434/v1 --model <name>
curl -s http://127.0.0.1:8080/healthz                 # is fak up?
```

Then point your OpenAI client at `http://127.0.0.1:8080/v1`.

## 3. Upstream `401 Unauthorized` — the key env isn't set

**Symptom.** Requests reach a hosted provider but come back `401`/`403` from upstream.

**Cause.** For a keyed provider, fak forwards a bearer token read **from an environment
variable you name with `--api-key-env VAR`** — the flag names the env var, never the
literal key. If `VAR` is unset (or you never passed the flag), nothing is forwarded and
the upstream rejects the call.

**Fix.** Export the variable and name it:

```
export MYPROVIDER_KEY=sk-...
./fak serve --provider openai --api-key-env MYPROVIDER_KEY --base-url https://api.…/v1 --model <name>
```

(The request's model name passes through to the upstream verbatim, so your existing
prompts and tool definitions stay unchanged.) Note this is the *upstream* key;
`--require-key-env` is the separate knob that makes clients authenticate to *fak's* own
gateway.

## 4. A tool call you expected got refused — the default-deny policy blocked it

**Symptom.** Your agent runs, but a tool call it should have been able to make is
blocked, with a structured refusal naming a reason.

**Cause.** This is working as designed: fak ships an **embedded default-deny capability
floor**. The kernel adjudicates every tool call the model proposes and refuses anything
the policy does not explicitly allow — with a reason from a closed vocabulary, not a
free-text error. A first run often trips this because the floor is deliberately tight.

**Fix.** Inspect the floor you are running under, then widen it deliberately for the
capability you actually want:

```
fak manage --dump-policy     # print the embedded floor and every rule
```

Read the refusal's reason, find the matching rule, and allow that capability explicitly
— rather than disabling the gate. The default-deny is the point; the fix is to grant,
not to turn it off.

## 5. `address already in use` — the serve port is taken

**Symptom.** `fak serve` exits immediately with `address already in use` (or
`bind: address already in use`).

**Cause.** Another process — often a previous `fak serve` you didn't stop — already holds
the `--addr` port.

**Fix.** Pick another port, or free the one you want:

```
./fak serve --addr 127.0.0.1:8137 …          # just use a different port
# or find and stop the holder:
#   lsof -i :8080        (macOS/Linux)
#   Get-NetTCPConnection -LocalPort 8080   (Windows)
```

---

**Related:** [`GETTING-STARTED.md`](../../GETTING-STARTED.md) (the full first-session
walkthrough and the build-time troubleshooting table) ·
[`docs/integrations/claude.md`](../integrations/claude.md) (the one-command
`fak guard -- claude` front door) ·
[`docs/fak/tutorial.md`](../fak/tutorial.md) (the guided first session with captured
output).

_Dimension F (Developer experience & onramp) of the
[concept-popularization epic](../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md)._
