# Start here

*This page owns one job: taking a task you already have — operate, integrate, deploy, contribute — and naming the single current authority for it plus one next action. It is for a reader who has already decided to use fak and needs a destination, not an explanation.* If you have not decided yet, the [README](README.md) is the front door; to install and run, use [Getting started](GETTING-STARTED.md); to look a document up by name, use [INDEX.md](INDEX.md); to learn the concepts in order, use [LEARNING-PATH.md](LEARNING-PATH.md).

This page routes; it does not re-explain. The repository's configured homepage is the published copy of [`docs/showcase.html`](docs/showcase.html), which the README audit keeps in step with the README.

For the current product focus—fewer tokens, fewer turns, better model routing, and controllable long sessions—start with [Spend fewer tokens and turns](docs/CAPABILITIES.md).

For the durable product thesis—“don’t make me think,” without surrendering control—read
[The problems fak exists to solve](docs/problems-we-solve.md). It defines four checks applied
to every change—managed context, net-true efficiency, bounded adaptation, and integrated
operations—and a separate centrality signal for deciding which work is closest to those
connected user problems.

For the current investment boundary and capability classification, read [Project orientation: the agent-kernel center](docs/project-orientation.md).

## See one verdict in one command — no key, model, or GPU

If you are evaluating fak for the first time, the shortest proof that it manages a
tool-using agent is a single copy-pasteable command:

```bash
fak agent --offline
```

It runs a deterministic end-to-end check against a mock planner and prints three
adjudicated results — `task completed (booked) YES / YES`, `poisoned result blocked YES`,
and `destructive op prevented YES` — so you see the managed-agent path and the policy
boundary decide, with no key, model download, or GPU.

**Expected output:** this page deliberately does not reprint it. The verbatim block lives
in one place, [Getting started § Tier 0](GETTING-STARTED.md#2-tier-0--try-the-kernel-zero-downloads-2-min),
and the complete unabridged capture is in the [tutorial](docs/fak/tutorial.md). For the
expanded policy, routing, and benchmark sequence, use the
[one-minute offline proof](docs/repro-packet.md).

Everything below is the route map — choose the job you have now.

## Choose your route

| If you want to… | Go here |
|---|---|
| Discover token, turn, cache, routing, and session-control capabilities | [Performance-first capability map](docs/CAPABILITIES.md) |


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
| Look a document up by name, component, or artifact | [Documentation index](INDEX.md) | Search the exhaustive map for the page you already know the name of, or its versioned/historical route. |

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

## FAK or DOS?

Use **FAK** for the agent execution path (tool calls, context, tokens, models, cache, policy, and serving). Use **DOS** for the work decision/proof path (admission, leases, verification, witnesses, liveness, and operator decisions). FAK's repository workflows compose DOS; they do not make the two kernels interchangeable. See [FAK and DOS: which layer owns what](docs/fak-vs-dos.md).
## Deeper maps

- **Words this project coined or overloaded:** [`docs/glossary.md`](docs/glossary.md) pins down kernel, context, adjudicator, vDSO, context-MMU, and the capability floor / result admission pair.
- **Machines and coding agents:** [`llms.txt`](llms.txt) maps tasks to authoritative files.
- **All documentation:** [`INDEX.md`](INDEX.md) is the exhaustive repository-wide index — use it when you already know a document, component, or artifact name.
- **Repository agents:** [`AGENTS.md`](AGENTS.md) contains build, proof, commit, and shared-tree rules. It is written for automated contributors working inside the maintainers' shared checkout; human contributors want [`CONTRIBUTING.md`](CONTRIBUTING.md) instead.
- **Claims:** [`CLAIMS.md`](CLAIMS.md) records shipped, simulated, and stub status.
- **Project evolution:** [`docs/notes/`](docs/notes/) preserves dated research, decisions, and historical context.
- **Participation:** the [Code of Conduct](.github/CODE_OF_CONDUCT.md) governs issues, pull requests, and reviews, and names the route for reporting a problem.

If you are still deciding, use the default: [run the one-minute offline proof](docs/repro-packet.md).
