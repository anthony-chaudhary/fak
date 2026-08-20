<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak — configure your agents for the task at hand

AI agents can do more work than one fixed pile of prompts, tools, permissions, and model
settings can handle well. fak lets you run your agents with a small configuration chosen for
this task, while one boundary manages their context, models, tools, and record of what happened.

## Run your agents

Install fak, then keep using the Codex login and models you already have:

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/anthony-chaudhary/fak/main/install.sh | sh

# Any host with Go 1.26+
go install github.com/anthony-chaudhary/fak/cmd/fak@latest

# See the shipped configuration levels (no key or model call required)
fak agent profiles
```

For a small coding change, ask Ponytail to push toward the smallest correct implementation and
Caveman to keep the answer compact:

```bash
fak manage --output-profile caveman:medium --work-profile ponytail:high -- codex \
  "Remove the duplicate cache without adding a dependency."
```

For a risky investigation, use the same Codex harness with lighter simplicity pressure and room
for a fuller explanation:

```bash
fak manage --output-profile caveman:low --work-profile ponytail:low -- codex \
  "Trace the intermittent checkout failure and explain the evidence."
```

That is the idea: **your agent, configured for this task**. Ponytail controls how strongly the
agent resists avoidable code and machinery. Caveman controls how compactly it reports back. The
profiles never remove explicit requirements, safety checks, tests, diagnostics, or evidence.

`fak manage -- codex` uses the balanced `ponytail:medium` + `caveman:medium` defaults. Change
either axis per run, or use `standard` / `full` to turn that axis off. `fak guard` remains the
compatible name for `fak manage`.

### Next, make the harness yours

When a per-run configuration is not enough, generate a small harness whose identity,
instructions, and tools you own:

```bash
fak harness init --dir ./my-agent --module example.com/my-agent
cd ./my-agent
go run ./cmd/product --selfcheck
go run ./cmd/product
```

The generated product uses fak's managed boundary without making you assemble an agent loop from
scratch. Start with the [harness guide](docs/harness-init.md) when you are ready to customize it.

## More details

| If you want to… | Start here |
|---|---|
| Choose Ponytail and Caveman levels | [Work profiles](docs/work-profiles.md) · [response profiles](docs/response-profiles.md) |
| Connect another agent or model | [Codex](docs/integrations/openai-codex.md) · [Claude Code](docs/integrations/claude.md) · [all integrations](docs/integrations/) |
| Understand what fak manages | [Architecture](docs/architecture.md) · [capability map](docs/CAPABILITIES.md) |
| Check the proof and limits | [Claims](CLAIMS.md) · [benchmark authority](BENCHMARK-AUTHORITY.md) · [security](SECURITY.md) |
| Build on fak | [Go API](pkg/) · [harness contract](docs/harness-kit-contract.md) · [contributing](CONTRIBUTING.md) |
| Explore everything else | [Documentation index](docs/index.md) · [interactive showcase](docs/showcase.html) · [front-page archive](docs/README-legacy.md) |

Apache-2.0 licensed.

<!-- readme-verified: 2026-08-20 vs VERSION 0.44.0 + BENCHMARK-AUTHORITY · process: tools/readme_freshness_audit.py + /refresh-readme -->
