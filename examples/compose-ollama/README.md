# fak in front of Ollama (docker compose)

A copy-paste stack that brings up a **governed local model** in one command: [Ollama](https://ollama.com)
generates, and `fak serve` sits in front of it as an [OpenAI-compatible](https://platform.openai.com/docs/api-reference)
gateway that **adjudicates every tool call** the model proposes against a default-deny
capability floor and writes an audit log. Ollama's own port is never published — the
only way in is the governed front door on `:8080`.

## Run it

You'll need Docker (with `docker compose`), `curl`, and `openssl` — no Go, no Python,
no model on the host. On Windows, run it from Git Bash or WSL.

```bash
bash run.sh        # or ./run.sh once it is executable
```

That's it: it mints an ephemeral `FAK_GATEWAY_KEY`, brings the stack up, waits (with a
bounded retry, never forever) for the front door, and prints the checks below. With the
images and the model already cached it completes in well under a minute; the **first**
run takes several minutes while Docker pulls ~2 GB.

Or drive it by hand:

```bash
export FAK_GATEWAY_KEY="$(openssl rand -hex 32)"   # clients must present this as a Bearer token
docker compose up                                  # first run pulls qwen2.5:1.5b (~1 GB) into a named volume
```

`compose up` starts Ollama, pulls the model once (a one-shot `ollama-pull`
service that fak waits on), then starts `fak serve` fronting it. When it's up:

```bash
curl -s http://localhost:8080/healthz             # {"ok":true,...}   (always unauthenticated)

curl -s http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $FAK_GATEWAY_KEY" \
  -d '{"model":"qwen2.5:1.5b","messages":[{"role":"user","content":"Say OK."}]}'
```

Point any OpenAI client at `http://localhost:8080/v1` with that Bearer key and you have
a governed local model — no code change to the client.

## What you'll see

`run.sh` prints three checks in order: `/healthz` answering `{"ok":true,…}`
unauthenticated, an unauthenticated `/v1/models` call coming back refused with its
status code, and an authenticated chat call returning an OpenAI-shaped completion. It
exits `0` when the run completes.

The captured transcript — plus the two offline checks you can reproduce in seconds
with no Docker and no model — is in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## What each piece does

| Service | Role |
|---|---|
| `ollama` | Runs the model. Its `:11434` is **internal to the compose network only** — not published to the host, so nothing can bypass the gate. |
| `ollama-pull` | A one-shot that pulls `qwen2.5:1.5b` into the shared `ollama-models` volume, then exits. `fak` `depends_on` its successful completion, so the first request never races the download. |
| `fak` | `fak serve` with `--base-url=http://ollama:11434/v1`. The **only published port** (`:8080`). Enforces `--policy=/etc/fak/policy.json`, requires `Authorization: Bearer $FAK_GATEWAY_KEY` on every route except `/healthz`, and writes an audit journal to the `fak-audit` volume. |

## The policy

[`policy.json`](policy.json) is the deployable capability floor — a `fak-policy/v1`
manifest, validated offline with `fak policy --check policy.json`. It ships **fail
closed**: only tools matching an allow rule are permitted, everything else is refused
with a closed-vocabulary reason. The starter floor here allows read-only-shaped tools
(`read_`, `get_`, `search_`, `list_`, `lookup_`, `find_` prefixes, plus `calculate`)
and explicitly denies `exfiltrate` (`SECRET_EXFIL`) and `shell_rm_rf` (`POLICY_BLOCK`).

Edit it for your tools, then re-validate before restarting:

```bash
fak policy --dump  > policy.json     # start from the built-in default floor and edit, OR
fak policy --check policy.json       # validate your edited manifest (prints the floor it admits)
docker compose up -d --force-recreate fak
```

## Notes and honest fences

- **Model choice.** `qwen2.5:1.5b` is a small, fast default so the recipe runs on a
  laptop. Swap the tag in both `compose.yaml` (`ollama-pull` entrypoint and the fak
  `--model`) for any Ollama model; larger models need more RAM and a higher
  `FAK_HTTP_WRITE_TIMEOUT_S`.
- **Image tag.** The `fak` service uses `ghcr.io/anthony-chaudhary/fak:latest`. Pin a
  release tag for production, or build from source (`build: https://github.com/anthony-chaudhary/fak.git`).
- **Small-model agentic quality is a ramp.** A 1.5B model is for wiring and local
  experimentation; for frontier-quality coding, front a hosted provider instead
  (`--provider anthropic --base-url … --api-key-env …`). The governance is identical
  either way — that's the point.
- **This governs the tool-call boundary, not the model's weights.** fak decides which
  tool calls are allowed and quarantines tool results; it does not change what the model
  generates.
- **Determinism, split honestly.** The **gate** is deterministic: a verdict is a pure
  function of `(policy.json, the proposed call)`, so the same proposed call yields the
  same reason and disposition on every run and every machine — that is why the verdicts
  in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md) are byte-stable and safe to re-run. The
  **model** is not: `qwen2.5:1.5b` samples, so its text varies run to run.
- **What this does not claim.** It does not demonstrate that a denied tool call is
  prevented from *executing* inside your agent — fak returns the verdict; the runtime
  that honours it is `fak manage` / the managed runtime. It does not prove the model is
  safe, it does not benchmark answer quality, and because the images are pinned at
  `:latest` it is not a reproducible-by-digest deployment.

See [`docs/fak/deployment-guide.md`](../../docs/fak/deployment-guide.md) for the
production compose/Kubernetes patterns this recipe is distilled from, and
[`docs/integrations/claude.md`](../../docs/integrations/claude.md) for the one-command
`fak manage -- claude` front door.
