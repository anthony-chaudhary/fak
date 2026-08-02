# Start here

This page is the human route map for **fak**. Choose the job you have now; each route points to its current authority and one next action. For the product overview and first commands, use the [README](README.md).

## See one verdict in one command — no key, model, or GPU

If you are evaluating fak for the first time, the shortest proof that it manages a
tool-using agent is a single copy-pasteable command:

```bash
fak agent --offline
```

**Expected output** (abridged — the check is deterministic, so you should see these
same verdict lines; the [tutorial](docs/fak/tutorial.md) shows the full capture):

```
injection in context                YES           no
destructive op executed             YES           no
task completed (booked)             YES          YES

HEADLINE
  poisoned result blocked   : YES
  destructive op prevented  : YES
```

It runs a deterministic end-to-end check against a mock planner and prints three
adjudicated results — `task completed (booked) YES / YES`, `poisoned result blocked YES`,
and `destructive op prevented YES` — so you see the managed-agent path and the policy
boundary decide, with no key, model download, or GPU. For the expanded policy, routing,
and benchmark sequence, use the [one-minute offline proof](docs/repro-packet.md).

Everything below is the route map — choose the job you have now.

## Choose your route

| You want to… | Current route | Next action |
|---|---|---|
| Understand what fak manages | [README](README.md) | Choose `fak guard`, `fak serve`, or the offline proof from the first-screen table. |
| Stand up a governed agent, offline, in ~10 min | [Governed-agent quickstart](docs/fak/governed-agent-quickstart.md) | Run the one-command governed session, then the server-side floor + audit + DENY path. |
| Evaluate the kernel locally | [Reproducibility packet](docs/repro-packet.md) | Run its copy-paste proof and compare the three expected results. |
| Add fak beside one agent | [`fak guard` quickstart](README.md#manage-one-local-agent-fak-guard) | Launch the agent you already use through `fak guard`. |
| Run a shared or durable endpoint | [Server quickstart](docs/fak/server-quickstart.md) | Start `fak serve`, then call its health and model endpoints. |
| Integrate a client or agent | [Integration guides](docs/integrations/) | Select the guide for your client and follow its smallest working path. |
| Fix a first command that did not work | [First-run troubleshooting](docs/adoption/troubleshooting-first-run.md) | Match your symptom — binary not found, canned replies, an upstream `401`, a refused tool call, or a taken port — to its cause and one-line fix. |
| Operate or deploy fak | [Deployment chooser](docs/deployment.md) | Choose local, fleet, cloud, or air-gapped operation by requirements, then follow its health check. |
| Contribute code or docs | [Contributing guide](CONTRIBUTING.md) | No write access? Follow [fork and open a pull request](CONTRIBUTING.md#fork-and-open-a-pull-request). Maintainers: read the shared-checkout workflow, then choose a scoped issue. |
| Find documentation by job or lifecycle | [Documentation index](INDEX.md) | Choose an evaluator, builder, operator, contributor, or research route before entering the detailed map. |

## Builder routes

Choose the route that matches the next builder decision; each destination owns its prerequisites and support boundary.

| You want to… | Current route | Next action |
|---|---|---|
| Connect an agent, client, or protocol | [Integration path chooser](docs/integrations/README.md) | Select the row for the agent you run and follow its smallest supported path. |
| Choose a cloud or deployment target | [Supported clouds](docs/supported/clouds.md) | Match the target to its stated support and evidence before deploying. |
| Exercise a runnable outcome | [Demo chooser](docs/run-the-demos.md) | Pick a demo by outcome, time, hardware, and proof, then run its self-check. |
| Resolve a setup or usage question | [FAQ](docs/FAQ.md) | Choose the task question and follow its linked authority. |

## Operator routes

Use these routes after choosing to operate or deploy fak; they expose current boundaries, recovery, hardware evidence, and one production architecture.

| You want to… | Current route | Next action |
|---|---|---|
| Check current maturity and boundaries | [Product status](docs/PRODUCT-STATUS.md) | Verify the capability state and evidence before relying on it. |
| Recover from a bad upgrade or release | [Rollback procedure](docs/ROLLBACK.md) | Select the affected deployment and follow the bounded recovery sequence. |
| Match a workload to supported hardware | [Hardware matrix](docs/HARDWARE-MATRIX.md) | Choose the backend whose evidence satisfies the workload requirement. |
| Review a production vendor-cloud shape | [Neo Cloud reference architecture](docs/vendor/neo-cloud-reference-architecture.md) | Compare its assumptions with the target environment before adopting it. |

## Current product paths

The routes above describe the current generation of fak unless a page marks itself historical, experimental, simulated, or superseded.

- **`fak guard`** manages one existing local agent and is the default integration path.
- **`fak serve`** runs the same kernel as a shared or durable OpenAI, Anthropic, or MCP endpoint.
- **Offline proof** demonstrates adjudication and agent behavior without external compute.

Deployment support depends on the selected backend and environment. Each quickstart names its own prerequisites and supported boundary; the [README](README.md#one-managed-agent-two-ways-to-run-the-kernel) is authoritative for choosing a mode.

## Deeper maps

- **Words this project coined or overloaded:** [`docs/glossary.md`](docs/glossary.md) pins down kernel, context, adjudicator, vDSO, context-MMU, and the capability floor / result admission pair.
- **Machines and coding agents:** [`llms.txt`](llms.txt) maps tasks to authoritative files.
- **All documentation:** [`INDEX.md`](INDEX.md) is the curated repository-wide index.
- **Repository agents:** [`AGENTS.md`](AGENTS.md) contains build, proof, commit, and shared-tree rules.
- **Claims:** [`CLAIMS.md`](CLAIMS.md) records shipped, simulated, and stub status.
- **Project evolution:** [`docs/notes/`](docs/notes/) preserves dated research, decisions, and historical context.
- **Participation:** the [Code of Conduct](.github/CODE_OF_CONDUCT.md) governs issues, pull requests, and reviews, and names the route for reporting a problem.

If you are still deciding, use the default: [run the one-minute offline proof](docs/repro-packet.md).
