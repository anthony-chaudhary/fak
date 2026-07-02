---
title: "fak adoption visuals: how to think about using it"
description: "Five figures for adopting fak: where the binary sits, the rung-by-rung on-ramp, the honest split between fak-authored and provider-authored value, which integration shape fits what you already run, and when the performance win is real."
---

# Adoption Visuals — How to Think About Using fak

These five figures are the mental model for adopting `fak`. They are for someone
deciding whether to put it in front of an agent they already run. Figure 68 shows
where the binary sits and why nothing else changes. Figure 69 shows the adoption
ladder, one rung at a time. Figure 70 splits the value fak authors from the value it
relays. Figure 71 picks an integration shape from what you already run. Figure 72
gives the two conditions that gate the performance claim. Numbers trace to
[`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) and the
result docs it links. The figures repeat the honest fences instead of smoothing them
over.

Sources:

- [`docs/integrations/adopter-playbook.md`](integrations/adopter-playbook.md): the
  external-adopter playbook, with shapes A/B/C and the production checklist.
- [`docs/integrations/CLAUDE.md`](integrations/CLAUDE.md): the `fak guard` front door,
  the per-turn debug line, and the gateway-transit proof.
- [`docs/concepts-and-story.md`](concepts-and-story.md): the two-gate trust model and
  the "when does the win kick in" tables.
- [`fak/GETTING-STARTED.md`](https://github.com/anthony-chaudhary/fak/blob/main/GETTING-STARTED.md):
  install and first run.

| # | Figure | Point |
|---|---|---|
| 68 | Adoption seat map | Adopting fak is re-pointing a base URL: the agent and the model stay what they were; the seam between them becomes reviewable policy. |
| 69 | The on-ramp ladder | Each rung adds a witness instead of a rewrite. You can stop at any rung. |
| 70 | The honest value split | fak authors the floor, the quarantine, and the audit trail on any seat; prompt-cache speed on a proxy seat is the provider's win, relayed as OBSERVED. |
| 71 | Which shape fits | One binary, four integration shapes: pick by what you already run. |
| 72 | When the perf win is real | Two gates (multiple sharers, a real shared prefix), then ~60× only versus a naive loop and ~1.5–4× versus a tuned stack. |

---

## 68 - Adoption seat map

Killer line: adopting fak means re-pointing one base URL. The agent and the model
stay what they were. The seam between them becomes policy you can review, version,
and audit.

[SVG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/68-adoption-seat-map.svg) - [PNG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/68-adoption-seat-map.png) - [source](https://github.com/anthony-chaudhary/fak/blob/main/visuals/68-adoption-seat-map.mmd)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'Segoe UI, Helvetica, Arial, sans-serif','fontSize':'14px','lineColor':'#5B6B7B','primaryColor':'#DCE9FB','primaryTextColor':'#1B3A66','primaryBorderColor':'#3B6FB5','clusterBkg':'#F7F9FB','clusterBorder':'#AFC0CE'}}}%%
flowchart LR
  subgraph AGENTS["Your agent - unchanged"]
    direction TB
    claude["Claude Code<br/>subscription OAuth by default"]
    codex["Codex / OpenCode<br/>OpenAI-compatible wire"]
    sdk["Your own app<br/>Anthropic / OpenAI SDK or MCP client"]
  end

  subgraph FAK["fak - one static Go binary, the seam you own"]
    boundary["Tool calls adjudicated: allow / deny / repair / quarantine<br/>Tool results admitted before the model reads them<br/>Every decision counted; hash-chained journal opt-in<br/>Provider 429s and retries absorbed"]
  end

  subgraph UPSTREAM["The model - also unchanged"]
    direction TB
    anthropic["Anthropic API<br/>your subscription or API key"]
    openaiC["Any OpenAI-compatible server<br/>vLLM, SGLang, Ollama, llama-server, cloud"]
    gguf["Local GGUF in-kernel<br/>--gguf: no key, no network"]
  end

  claude -->|"ANTHROPIC_BASE_URL<br/>injected into the child only"| boundary
  codex -->|"OPENAI_BASE_URL"| boundary
  sdk -->|"base_url = the gateway"| boundary
  boundary --> anthropic
  boundary --> openaiC
  boundary --> gguf

  note["Adoption is re-pointing a base URL.<br/>The agent and the model stay what they were;<br/>the seam between them becomes reviewable, executable policy."]
  FAK -.-> note

  classDef untrusted fill:#FFE9D6,stroke:#E8833A,stroke-width:2px,color:#7A3E12;
  classDef kernel fill:#DCE9FB,stroke:#3B6FB5,stroke-width:2px,color:#1B3A66;
  classDef note fill:#F7F9FB,stroke:#AFC0CE,stroke-width:1px,color:#46586A;

  class claude,codex,sdk,anthropic,openaiC,gguf untrusted;
  class boundary kernel;
  class note note;
```

**Terms used:**

- "injected into the child only": `fak guard` sets the base URL in the wrapped
  process's environment. Your shell, your `settings.json`, and any other agent in
  another terminal are untouched.
- "hash-chained journal opt-in": the durable audit trail exists when
  `FAK_AUDIT_JOURNAL` is set. The in-memory exit summary is always on.

---

## 69 - The on-ramp ladder

Killer line: each rung adds a witness rather than a rewrite. Stop at any rung and
keep everything the previous rung proved.

[SVG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/69-adoption-onramp.svg) - [PNG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/69-adoption-onramp.png) - [source](https://github.com/anthony-chaudhary/fak/blob/main/visuals/69-adoption-onramp.mmd)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'Segoe UI, Helvetica, Arial, sans-serif','fontSize':'14px','lineColor':'#5B6B7B','primaryColor':'#DCE9FB','primaryTextColor':'#1B3A66','primaryBorderColor':'#3B6FB5','clusterBkg':'#F7F9FB','clusterBorder':'#AFC0CE'}}}%%
flowchart TB
  r0["Rung 0 - wrap what you already run<br/>fak guard -- claude<br/>gain: embedded default-deny floor, exit summary of every verdict,<br/>429 absorption - on your existing subscription"]
  r1["Rung 1 - own the floor<br/>fak policy --dump then edit, fak policy --check<br/>gain: the allow-list is a reviewed JSON file in YOUR git,<br/>checkable offline with no model and no key"]
  r2["Rung 2 - turn on the record<br/>FAK_AUDIT_JOURNAL=audit.jsonl, --log FILE, GET /metrics<br/>gain: a hash-chained decision trail an auditor re-verifies,<br/>plus Prometheus counters per verdict"]
  r3["Rung 3 - front your own stack<br/>fak serve --base-url (production checklist) or serve --stdio (MCP)<br/>or --gguf (local model in-kernel)<br/>gain: the same floor for every client of your endpoint"]
  r4["Rung 4 - fleet and referee<br/>many sessions over one boundary<br/>gain: per-trace metrics across agents, and claim verification<br/>from artifacts instead of worker self-reports"]

  r0 -->|"cost to climb: edit one JSON file"| r1
  r1 -->|"cost: set one env var"| r2
  r2 -->|"cost: run the same binary as a server"| r3
  r3 -->|"cost: point more sessions at it"| r4

  note["Each rung adds a witness, not a rewrite.<br/>Stop at any rung: the binary and the policy<br/>keep the same shape underneath you."]
  r4 -.-> note

  classDef pass fill:#DBF3E1,stroke:#3FA45F,stroke-width:2px,color:#1C5230;
  classDef kernel fill:#DCE9FB,stroke:#3B6FB5,stroke-width:2px,color:#1B3A66;
  classDef note fill:#F7F9FB,stroke:#AFC0CE,stroke-width:1px,color:#46586A;

  class r0 pass;
  class r1,r2,r3,r4 kernel;
  class note note;
```

**Terms used:**

- "witness": an artifact a third party can re-check, as opposed to a self-report.
  Examples: a policy file in git, a hash-chained journal row, a metrics counter.
- "claim verification": at rung 4, commit and task claims are checked against git
  evidence rather than taken from the worker's own summary.

---

## 70 - The honest value split

Killer line: adopt fak for the left column, which holds on any seat. On a proxy
seat, "faster" belongs to the provider column. fak relays that slice marked OBSERVED
instead of claiming it.

[SVG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/70-adoption-value-split.svg) - [PNG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/70-adoption-value-split.png) - [source](https://github.com/anthony-chaudhary/fak/blob/main/visuals/70-adoption-value-split.mmd)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'Segoe UI, Helvetica, Arial, sans-serif','fontSize':'14px','lineColor':'#5B6B7B','primaryColor':'#DCE9FB','primaryTextColor':'#1B3A66','primaryBorderColor':'#3B6FB5','clusterBkg':'#F7F9FB','clusterBorder':'#AFC0CE'}}}%%
flowchart TB
  turn["One served turn crosses the boundary.<br/>Who authored each slice of its value?"]

  subgraph FAKOWN["fak-authored - WITNESSED on any seat, including proxy"]
    direction TB
    floor["Capability-floor refusals<br/>rm -rf denied, git push gated, self-modify blocked"]
    quar["Result quarantine<br/>hostile tool output held out of context"]
    audit["Legibility<br/>per-turn verdict line, /metrics,<br/>hash-chained audit journal"]
    resil["Operational absorption<br/>provider 429s / retries eaten;<br/>deny-all false stops auto-resumed"]
    shed["Compaction shed<br/>the fak= token slice of the debug line"]
  end

  subgraph PROV["provider-authored - OBSERVED, honestly relayed"]
    direction TB
    pcache["Prompt-cache read rebates<br/>the prov= token slice of the debug line"]
    speed["Serving speed and model quality"]
  end

  subgraph LOCAL["fak-authored only in-kernel (--gguf / local model)"]
    direction TB
    kv["KV prefix as a kernel object<br/>cross-agent prefix reuse"]
    evict["Poison eviction<br/>quarantined bytes never enter the KV cache"]
  end

  turn --> FAKOWN
  turn --> PROV
  turn --> LOCAL

  note["On a proxy or subscription seat, faster belongs to the provider column -<br/>fak relays it, marked OBSERVED, and does not claim it.<br/>Adopt for the left column; the right column arrives only when<br/>the model lives inside the kernel."]
  PROV -.-> note

  classDef pass fill:#DBF3E1,stroke:#3FA45F,stroke-width:2px,color:#1C5230;
  classDef gate fill:#FFF3CC,stroke:#C9A227,stroke-width:2px,color:#6B5410;
  classDef kernel fill:#DCE9FB,stroke:#3B6FB5,stroke-width:2px,color:#1B3A66;
  classDef note fill:#F7F9FB,stroke:#AFC0CE,stroke-width:1px,color:#46586A;

  class turn note;
  class floor,quar,audit,resil,shed pass;
  class pcache,speed gate;
  class kv,evict kernel;
  class note note;
```

**Terms used:**

- "prov= / fak=": the two token-saving slices on the per-turn debug line `fak guard`
  prints. `prov=` is the provider prompt-cache net saving, read rebate minus write
  premium, relayed from provider counters. `fak=` is the slice fak itself authored:
  compaction shed, plus in-kernel KV-prefix reuse on a local model.
- "OBSERVED": a value fak reports but did not produce. Provider counters pass
  through with their origin labeled and are never folded into fak's own claim.
- "deny-all false stops auto-resumed": when the floor refuses every tool call in a
  turn, the harness would otherwise stop early. Guard blocks that spurious stop and
  re-prompts the agent to pick an allowed alternative. The retry is bounded and on
  by default.

---

## 71 - Which shape fits

Killer line: one binary, four shapes. You add flags rather than components. Pick by
what you already run.

[SVG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/71-adoption-shape-picker.svg) - [PNG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/71-adoption-shape-picker.png) - [source](https://github.com/anthony-chaudhary/fak/blob/main/visuals/71-adoption-shape-picker.mmd)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'Segoe UI, Helvetica, Arial, sans-serif','fontSize':'14px','lineColor':'#5B6B7B','primaryColor':'#DCE9FB','primaryTextColor':'#1B3A66','primaryBorderColor':'#3B6FB5','clusterBkg':'#F7F9FB','clusterBorder':'#AFC0CE'}}}%%
flowchart TB
  start["What do you already run?"]

  q1{"An agent CLI you launch yourself?<br/>claude / codex / opencode"}
  q2{"A model endpoint that many<br/>clients already hit?"}
  q3{"An agent that executes its own tools<br/>and should ask for a verdict first?"}
  q4{"No model in the loop -<br/>you just want the floor checked?"}

  guard["fak guard -- your-agent<br/>in-process gateway on a private loopback port;<br/>base URL injected into the child only; zero config"]
  serve["fak serve --base-url ...<br/>the bare-serve production path:<br/>policy file + auth-key env + /healthz + /metrics"]
  mcp["fak serve --stdio<br/>manual MCP server: the agent asks the kernel<br/>about a call before running it itself"]
  ci["fak policy --check / fak preflight<br/>no model, no key, no listener -<br/>author and gate the floor in CI"]
  ggufN["Air-gapped or privacy-bound?<br/>add --gguf: a local model in-kernel,<br/>no key, no network after the first pull"]

  start --> q1
  q1 -->|"yes"| guard
  q1 -->|"no"| q2
  q2 -->|"yes"| serve
  q2 -->|"no"| q3
  q3 -->|"yes"| mcp
  q3 -->|"no"| q4
  q4 -->|"yes"| ci
  guard -.-> ggufN
  serve -.-> ggufN

  note["One binary, four shapes: you add flags, not components.<br/>Pick by what you already run,<br/>not by what you are willing to rewrite."]
  ci -.-> note

  classDef gate fill:#FFF3CC,stroke:#C9A227,stroke-width:2px,color:#6B5410;
  classDef pass fill:#DBF3E1,stroke:#3FA45F,stroke-width:2px,color:#1C5230;
  classDef kernel fill:#DCE9FB,stroke:#3B6FB5,stroke-width:2px,color:#1B3A66;
  classDef note fill:#F7F9FB,stroke:#AFC0CE,stroke-width:1px,color:#46586A;

  class start note;
  class q1,q2,q3,q4 gate;
  class guard pass;
  class serve,mcp,ci,ggufN kernel;
  class note note;
```

**Terms used:**

- "shape": one of the four integration forms from the
  [adopter playbook](integrations/adopter-playbook.md). Wrap an agent CLI with
  `fak guard`. Front a model endpoint with `serve --base-url`. Advise a
  self-executing agent over `serve --stdio` MCP. Or check the policy floor in CI
  with no model at all.

---

## 72 - When the perf win is real

Killer line: the performance claim is gated, never ambient. It needs two sharers and
a real shared prefix, or the honest answer is "roughly zero". The trust value is
what holds at one agent, one turn.

[SVG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/72-adoption-perf-gate.svg) - [PNG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/72-adoption-perf-gate.png) - [source](https://github.com/anthony-chaudhary/fak/blob/main/visuals/72-adoption-perf-gate.mmd)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'Segoe UI, Helvetica, Arial, sans-serif','fontSize':'14px','lineColor':'#5B6B7B','primaryColor':'#DCE9FB','primaryTextColor':'#1B3A66','primaryBorderColor':'#3B6FB5','clusterBkg':'#F7F9FB','clusterBorder':'#AFC0CE'}}}%%
flowchart TB
  claim["When is the performance win real?"]

  g1{"Two or more things share one prompt?<br/>many turns in a row, or agents side by side"}
  g2{"A shared prefix worth reusing?<br/>a few hundred words or more"}
  base{"What are you replacing?"}

  zero["Roughly zero benefit.<br/>A single agent on a single short turn<br/>is a slight LOSS - adopt for trust, not speed"]
  naive["A naive re-send-everything loop:<br/>up to ~60x, growing with turns and agents"]
  tuned["A tuned warm-cache / prefix-sharing stack:<br/>~1.5-4x, growing with prompt size and agent count"]
  writes["Write-heavy fleet caution: around a 1% shared-state<br/>write rate, cross-agent sharing can flip NEGATIVE<br/>on defaults - read-heavy fleets only"]

  claim --> g1
  g1 -->|"no"| zero
  g1 -->|"yes"| g2
  g2 -->|"no"| zero
  g2 -->|"yes"| base
  base -->|"naive baseline"| naive
  base -->|"tuned baseline"| tuned
  naive -.-> writes
  tuned -.-> writes

  note["The trust value - floor, quarantine, audit - is not gated<br/>by any of this. It holds at one agent, one turn."]
  zero -.-> note

  classDef gate fill:#FFF3CC,stroke:#C9A227,stroke-width:2px,color:#6B5410;
  classDef pass fill:#DBF3E1,stroke:#3FA45F,stroke-width:2px,color:#1C5230;
  classDef deny fill:#FBDDDD,stroke:#C9453F,stroke-width:2px,color:#6E1F1B;
  classDef note fill:#F7F9FB,stroke:#AFC0CE,stroke-width:1px,color:#46586A;

  class claim note;
  class g1,g2,base gate;
  class naive,tuned pass;
  class zero,writes deny;
  class note note;
```

**Terms used:**

- "~60× / ~1.5–4×": the two honest baselines from
  [concepts-and-story](concepts-and-story.md). The large figure is only versus a
  naive loop that re-sends the whole growing conversation every turn. Versus a tuned
  warm-cache stack the gain is a few-fold.
- "read-heavy fleets only": measured on the fleet sweep. Global invalidation turns
  the cross-agent benefit negative at roughly a 1% shared-state write rate on
  default settings. Scoped invalidation recovers most of it, but the caution stands.

---

## The honest scope, in one place

- The figures describe mechanisms that exist today: `fak guard`, `fak serve`,
  `serve --stdio`, `policy --check`, `--gguf`, the audit journal, and `/metrics`.
  None of this claims market adoption. The on-ramp is what an adopter *would* climb,
  and never a report that anyone has.
- On a proxy or subscription seat there is no local KV cache, so KV poison-eviction
  is a structural no-op there. That row of value is real only on the in-kernel
  (`--gguf` / local model) path.
- The perf figures are geometry- and sweep-derived ratios, and no substitute for an
  independent benchmark of your workload. Read figure 72's gates before quoting any
  of them.
