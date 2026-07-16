# Start here

This page is the human route map for **fak**. Choose the job you have now; each route points to its current authority and one next action. For the product overview and first commands, use the [README](README.md).

**Default:** if you are evaluating fak for the first time, run the [one-minute offline proof](docs/repro-packet.md). It needs no key, model download, or GPU.

## Choose your route

| You want to… | Current route | Next action |
|---|---|---|
| Understand what fak manages | [README](README.md) | Choose `fak guard`, `fak serve`, or the offline proof from the first-screen table. |
| Evaluate the kernel locally | [Reproducibility packet](docs/repro-packet.md) | Run its copy-paste proof and compare the three expected results. |
| Add fak beside one agent | [`fak guard` quickstart](README.md#manage-one-local-agent-fak-guard) | Launch the agent you already use through `fak guard`. |
| Run a shared or durable endpoint | [Server quickstart](docs/fak/server-quickstart.md) | Start `fak serve`, then call its health and model endpoints. |
| Integrate a client or agent | [Integration guides](docs/integrations/) | Select the guide for your client and follow its smallest working path. |
| Operate or deploy fak | [Deployment chooser](docs/deployment.md) | Choose local, fleet, cloud, or air-gapped operation by requirements, then follow its health check. |
| Contribute code or docs | [Contributing guide](CONTRIBUTING.md) | Read the repository workflow, then choose a scoped issue. |
| Find documentation by job or lifecycle | [Documentation index](INDEX.md) | Choose an evaluator, builder, operator, contributor, or research route before entering the detailed map. |

## Current product paths

The routes above describe the current generation of fak unless a page marks itself historical, experimental, simulated, or superseded.

- **`fak guard`** manages one existing local agent and is the default integration path.
- **`fak serve`** runs the same kernel as a shared or durable OpenAI, Anthropic, or MCP endpoint.
- **Offline proof** demonstrates adjudication and agent behavior without external compute.

Deployment support depends on the selected backend and environment. Each quickstart names its own prerequisites and supported boundary; the [README](README.md#one-managed-agent-two-ways-to-run-the-kernel) is authoritative for choosing a mode.

## Deeper maps

- **Machines and coding agents:** [`llms.txt`](llms.txt) maps tasks to authoritative files.
- **All documentation:** [`INDEX.md`](INDEX.md) is the curated repository-wide index.
- **Repository agents:** [`AGENTS.md`](AGENTS.md) contains build, proof, commit, and shared-tree rules.
- **Claims:** [`CLAIMS.md`](CLAIMS.md) records shipped, simulated, and stub status.
- **Project evolution:** [`docs/notes/`](docs/notes/) preserves dated research, decisions, and historical context.

If you are still deciding, use the default: [run the one-minute offline proof](docs/repro-packet.md).
