---
title: "Linting Agent Code at the Kernel"
description: "When an agent writes code, who checks it before it lands? fak already adjudicates every tool call, so a write that produces broken code is checkable at the same boundary. codelint adds language-server packs to the kernel: route each file to the pack that owns its extension, report the parse/compile errors, and feed them back to the model so it self-corrects."
slug: code-linting-at-the-kernel
keywords:
  - language server
  - LSP
  - linting
  - agent code
  - tool call adjudication
  - tensor code
  - kernel boundary
  - SWE-bench
date: 2026-06-23
---

# Linting agent code at the kernel

A coding agent's main move is to write a file. It reads some source, decides on an
edit, and calls `write_file`. Nothing in a normal loop checks that the bytes it just
wrote even parse. The agent finds out a few turns later, when a build fails or a test
errors, and spends a turn recovering. Multiply that across a fleet and the wasted
turns add up.

fak already sits on that boundary. Every tool call an agent makes crosses one
in-process kernel that decides whether the call may run. A `write_file` is just a
tool call carrying a path and some content. So the same place that asks "is this
agent allowed to write here?" is the natural place to also ask "is what it's writing
valid code?" That second question is what `codelint` answers.

## A pack is one language's diagnostics

The unit is a `Pack`: one language, the file extensions it owns, and a `Check` that
turns a file into findings.

```go
type Pack struct {
	Lang  string
	Exts  []string
	Check func(ctx context.Context, path string) ([]Finding, error)
}
```

`DefaultRegistry` ships four. Go and JSON check in-process, because the Go standard
library already parses both — no external tool, always an opinion. Python and CUDA
shell out to their own toolchains (the stdlib Python compiler, `nvcc`), because that
is where those languages' parsers live. Python is the canonical tensor language
(PyTorch, JAX, NumPy), and CUDA is the tensor code that runs on the GPU, so "lint
code like tensors" is just the Python and CUDA packs doing their job.

Adding a language is one entry. A pack that shells out is a few lines: the binaries
to try, the argv to build, and the closed code its errors carry. A full long-lived
language server — gopls, pyright, rust-analyzer over JSON-RPC — is a future `Pack`
with the same `Check` shape; the simplest realization just runs the language's
one-shot checker, and that is what ships today.

## The constraint that shapes the design

The kernel's decide path is not allowed to start a subprocess. That rule has its own
gate, `architest`'s `TestHotPathHasNoExec`, and a reason: the whole point of fak is
that a tool-call decision costs microseconds in one address space, not the
hundreds of milliseconds a per-call `fork`/`exec` would cost. A language server is a
subprocess. So `codelint` cannot live on the decide path.

It doesn't. `codelint` is a foundation leaf that runs off the hot path, the same
place the SWE-bench fleet already runs `git` and `bash`. Each external pack is
bounded by a timeout and degrades to "no opinion" when its checker is absent — most
hosts have no `nvcc`, and that is fine, the CUDA pack simply says nothing there and
lints for real on a GPU box. Linting is a quality signal, never a security gate, so
it fails open: an absent or wedged checker never blocks a write.

## Two decisions that follow from "the input is untrusted"

boundarylint, the kernel's static linter for its own Go source, honors a
`//boundarylint:ignore` comment. A human owns that source, so letting them mark an
exception is reasonable. `codelint` honors no such comment. The code it reads was
written by the model, and the model must not be able to switch the gate off by
emitting a magic comment — the same rule that keeps the model from talking past the
adjudicator. The lint is a kernel judgment, not a model-suppressible one.

`codelint` also reports only hard errors: code that does not parse or compile.
Semantic checks — "undefined name", "wrong argument type" — need the whole package's
context, which a single agent-written file in isolation does not have, and would fire
false positives on perfectly good code. The parse/compile tier is the one with no
false positives, which is exactly the tier you want gating an automated loop.

## A worked example: the write that fixes itself

Say a coding agent is editing a Go file and fumbles a method signature. It
calls `write_file` with content that stops mid-declaration:

```go
package store

func (s *Store) Put(
```

In a normal loop that file just sits there. The agent moves on, and three turns
later a `go build` fails. Now it has to stop, read a compiler error, trace its
way back to this file, and spend a turn recovering — having long since paged the
mistake out of its working context.

With `--lint-writes` on, the SWE-bench fleet runs the Go pack the moment the
write lands and staples the result onto the tool output the agent already gets
back:

```
codelint: the file you just wrote has errors — fix them before continuing:
store.go:3:8: error: expected ')', found 'EOF' (go/GO_PARSE)
store.go:3:8: error: expected 'IDENT', found 'EOF' (go/GO_PARSE)
```

The agent reads the parse error on the same turn it made it, while the file is
still in front of it, and re-issues a correct write. The breakage never reaches
the build. And the write still landed — the diagnostic is advice clipped to the
result, not a veto — so a checker that is ever wrong cannot wedge the loop.

You can watch the pack do exactly this with no model in the loop:

```bash
printf 'package store\n\nfunc (s *Store) Put(\n' > /tmp/broken.go
go run ./cmd/fak codelint /tmp/broken.go
# codelint: 1 file(s) checked, 5 finding(s) (5 error, 0 warning)
# /tmp/broken.go:3:8: error: expected ')', found 'EOF' (go/GO_PARSE)
# ...
echo "exit $?"   # -> 1
```

A clean file is silent and exits 0; a file in a language no pack owns is left
unlinted, never an error. That is the whole contract.

## Where it's wired

Two surfaces, today:

- `fak codelint PATH...` runs the packs over files or whole directories and exits
  non-zero on a hard error. It is the code-content dual of `fak lint`, which checks
  the tool registry. Point it at a repo in CI, or pipe one file through it.
- The SWE-bench fleet runs the same packs (`Registry.LintFile`) over every agent
  file write under `--lint-writes`. When the agent writes broken code, the parse
  error is appended to
  the tool result it sees, so it fixes the file on the next turn instead of
  discovering the breakage downstream. It is off by default, so a benchmark run's
  numbers don't move unless you ask for it, and the write always lands — the
  diagnostic is advice, not a veto.

The deeper integration — a write-scoped verdict in the adjudicator itself, so a
broken write is refused with a `MALFORMED` reason the model can act on — is a natural
next rung. It would route the lint through the kernel's existing out-of-process
verification seam rather than forcing exec onto the decide path, keeping the rule
above intact. The pieces are in place; the leaf is the part that was missing.

## Try every pack

```bash
go build ./cmd/fak
./fak codelint --list
# codelint can lint: cuda, go, json, python
./fak codelint --json ./path/to/changes   # machine-readable findings, e.g. for a CI gate
```

Go and JSON always have an opinion, because the Go standard library parses both
with no external tool. Python and CUDA report for real wherever `python3` or
`nvcc` is on PATH and stay quiet where they are not, so the one command is safe
to drop into any host or CI image.

To see where this sits in the kernel's design — why feeding errors back at the
write boundary is the concrete payoff of the write gate — work through FAK 318 in
the [learning path](https://github.com/anthony-chaudhary/fak/blob/main/LEARNING-PATH.md).
