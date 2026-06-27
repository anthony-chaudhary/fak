---
title: "fak permission systems visual companion (2026-06-18)"
description: "Figures contrasting prompt fatigue, broad grants, classifier risk, and dangerous bypass against FAK/DOS, the checked-in executable kernel policy for tool calls."
---

# Permission Systems Visual Companion

> Visual companion for the permission-system benchmark and API-host bridge proof.
> These figures draw the current claim: users should not have to choose between
> repeated prompts, broad class grants, auto-classified permissions, or dangerous
> bypass. FAK/DOS makes the tool boundary a checked-in, executable kernel policy.

Sources:

- [`docs/permission-system-benchmark-methodology.md`](../permission-system-benchmark-methodology.md)
- [`fak/experiments/permission-systems/permission-system-benchmark.md`](https://github.com/anthony-chaudhary/fak/blob/main/experiments/permission-systems/permission-system-benchmark.md)
- [`fak/experiments/permission-systems/permission-source-audit.md`](https://github.com/anthony-chaudhary/fak/blob/main/experiments/permission-systems/permission-source-audit.md)
- [`fak/experiments/api-host-bridge/api-host-bridge-gate.md`](https://github.com/anthony-chaudhary/fak/blob/main/experiments/api-host-bridge/api-host-bridge-gate.md)
- [`fak/experiments/api-host-bridge/api-host-bridge-proof.md`](https://github.com/anthony-chaudhary/fak/blob/main/experiments/api-host-bridge/api-host-bridge-proof.md)
- [`fak/experiments/api-host-bridge/api-host-bridge-verify-all.md`](https://github.com/anthony-chaudhary/fak/blob/main/experiments/api-host-bridge/api-host-bridge-verify-all.md)

| # | Figure | Point |
|---|---|---|
| 35 | Permission options map | The bad alternatives are fatigue, broad authority, classifier miss risk, or no boundary; FAK/DOS is the executable floor. |
| 36 | Deterministic risk coverage | FAK/DOS is 6/6 hard coverage; the next highest rows are Codex sandboxing at 3/6 (50.0%) and GitHub Copilot cloud agent at 2/6 (33.3%). |
| 37 | Two-gate permission path | Tool calls and tool results are both admitted at the gateway boundary before the model host sees them. |
| 38 | API-host bridge proof stack | The bridge is scope-bounded, executable proof: 7/7 source audit, 9/9 witness commands, 16/16 proof rollup, 35/35 verify-all. |

---

## 35 - Permission options map

Killer line: the incumbent choices trade between fatigue, over-broad authority,
probabilistic recall, after-the-fact review, or no boundary.

[SVG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/35-permission-options-map.svg) - [PNG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/35-permission-options-map.png) - [source](https://github.com/anthony-chaudhary/fak/blob/main/visuals/35-permission-options-map.mmd)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'Segoe UI, Helvetica, Arial, sans-serif','fontSize':'14px','lineColor':'#5B6B7B','primaryColor':'#DCE9FB','primaryTextColor':'#1B3A66','primaryBorderColor':'#3B6FB5','clusterBkg':'#F7F9FB','clusterBorder':'#AFC0CE'}}}%%
flowchart LR
  need["User wants useful autonomy<br/>without handing over the keys"]

  prompts["Manual prompts<br/>control: PROMPT<br/>hard risk coverage: 0/6<br/>failure: click-through tax"]
  classes["Broad sandbox / class grants<br/>control: boundary + review<br/>hard risk coverage: partial<br/>failure: allowed class is too wide"]
  auto["Auto-classified permissioning<br/>control: learned classifier<br/>hard risk coverage: 0/6<br/>known max FNR: 17%"]
  skip["Dangerous skip<br/>control: none<br/>hard risk coverage: 0/6<br/>unguarded risk allows: 6"]

  fak["FAK/DOS gateway<br/>control: manifest + arg rules + IFC + witness<br/>hard risk coverage: 6/6<br/>result admission: QUARANTINE<br/>bridge controls: 6/6"]

  proof["Executable proof stack<br/>source audit: 7/7<br/>witness gate: 9/9<br/>proof rollup: 16/16<br/>verify-all: 35/35"]

  need --> prompts
  need --> classes
  need --> auto
  need --> skip
  need --> fak
  fak --> proof

  bad["Incumbent choices trade off fatigue,<br/>over-broad authority, probabilistic recall,<br/>after-the-fact review, or no boundary."]
  prompts -.-> bad
  classes -.-> bad
  auto -.-> bad
  skip -.-> bad

  classDef untrusted fill:#FFE9D6,stroke:#E8833A,stroke-width:2px,color:#7A3E12;
  classDef kernel fill:#DCE9FB,stroke:#3B6FB5,stroke-width:2px,color:#1B3A66;
  classDef gate fill:#FFF3CC,stroke:#C9A227,stroke-width:2px,color:#6B5410;
  classDef pass fill:#DBF3E1,stroke:#3FA45F,stroke-width:2px,color:#1C5230;
  classDef deny fill:#FBDDDD,stroke:#C9453F,stroke-width:2px,color:#6E1F1B;
  classDef note fill:#F7F9FB,stroke:#AFC0CE,stroke-width:1px,color:#46586A;

  class need note;
  class prompts,auto,skip deny;
  class classes gate;
  class fak pass;
  class proof kernel;
  class bad note;
```

**Terms used:**
- "unguarded risk allows": The count of risky tool-call scenarios (from the six benchmark rows) that are permitted without any guard, review, or control mechanism.
- "known max FNR": The worst-case False Negative Rate of the auto-classifier—the probability it incorrectly permits a dangerous tool call—measured against the benchmark dataset.

---

## 36 - Deterministic risk coverage

Killer line: FAK/DOS is the only row with hard control over all six risky
scenarios in the benchmark.

[SVG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/36-permission-risk-coverage.svg) - [PNG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/36-permission-risk-coverage.png) - [source](https://github.com/anthony-chaudhary/fak/blob/main/visuals/36-permission-risk-coverage.mmd)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'Segoe UI, Helvetica, Arial, sans-serif','xyChart':{'backgroundColor':'#F7F9FB','titleColor':'#1B3A66','xAxisLabelColor':'#1B3A66','yAxisLabelColor':'#1B3A66','xAxisTitleColor':'#1B3A66','yAxisTitleColor':'#1B3A66','plotColorPalette':'#3B6FB5'}}}}%%
xychart-beta
  title "Fig 36 - Deterministic risk coverage over six risky rows"
  x-axis ["FAK/DOS", "Claude auto", "Codex", "Copilot", "Prompts", "Skip"]
  y-axis "hard control (%)" 0 --> 100
  bar [100, 0, 50, 33.3, 0, 0]
```

---

## 37 - Two-gate permission path

Killer line: the host stays swappable, but the tool boundary becomes executable
policy owned by the gateway.

[SVG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/37-permission-two-gate-path.svg) - [PNG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/37-permission-two-gate-path.png) - [source](https://github.com/anthony-chaudhary/fak/blob/main/visuals/37-permission-two-gate-path.mmd)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'Segoe UI, Helvetica, Arial, sans-serif','fontSize':'14px','lineColor':'#5B6B7B','primaryColor':'#DCE9FB','primaryTextColor':'#1B3A66','primaryBorderColor':'#3B6FB5','clusterBkg':'#F7F9FB','clusterBorder':'#AFC0CE'}}}%%
flowchart TB
  host["Compatible API host<br/>(model stays behind it)"]

  subgraph GW["FAK/DOS gateway owns the tool boundary"]
    direction TB
    proposed["Upstream proposes tool call"]
    subgraph CALL["Gate 1 - pre-execution tool-call admission"]
      direction TB
      nameGate{"tool name allowed<br/>by manifest?"}
      argGate{"scalar arg rules pass?<br/>glob / regex / byte cap"}
      flowGate{"IFC / preflight / witness<br/>verdict passes?"}
      admitted["Admitted tool call<br/>returned to client"]
      clientExec["Client executes admitted tool<br/>(any language)"]
      deny1["DENY<br/>DEFAULT_DENY or POLICY_BLOCK"]
      deny2["DENY<br/>bounded claim, no arg leak"]
      hold["HOLD<br/>needs witness or repair"]
    end
    subgraph RESULT["Gate 2 - pre-send result admission"]
      direction TB
      resultGate{"pre-send result admission<br/>before upstream sees bytes"}
      clean["clean result<br/>sent upstream"]
      hostClean["Host receives<br/>sanitized result"]
      quarantine["QUARANTINE<br/>hostile result held out"]
    end
  end

  host --> proposed
  proposed --> nameGate
  nameGate -->|"no"| deny1
  nameGate -->|"yes"| argGate
  argGate -->|"no"| deny2
  argGate -->|"yes"| flowGate
  flowGate -->|"hold"| hold
  flowGate -->|"pass"| admitted
  admitted --> clientExec --> resultGate
  resultGate -->|"clean"| clean --> hostClean
  resultGate -->|"hostile"| quarantine

  note["The product move: keep the model host swappable,<br/>but make permission a checked-in kernel policy<br/>instead of a prompt, class guess, classifier, or skip flag."]
  GW -.-> note

  classDef untrusted fill:#FFE9D6,stroke:#E8833A,stroke-width:2px,color:#7A3E12;
  classDef kernel fill:#DCE9FB,stroke:#3B6FB5,stroke-width:2px,color:#1B3A66;
  classDef gate fill:#FFF3CC,stroke:#C9A227,stroke-width:2px,color:#6B5410;
  classDef pass fill:#DBF3E1,stroke:#3FA45F,stroke-width:2px,color:#1C5230;
  classDef deny fill:#FBDDDD,stroke:#C9453F,stroke-width:2px,color:#6E1F1B;
  classDef note fill:#F7F9FB,stroke:#AFC0CE,stroke-width:1px,color:#46586A;

  class host,proposed,clientExec untrusted;
  class nameGate,argGate,flowGate,resultGate gate;
  class admitted,clean,hostClean pass;
  class deny1,deny2,hold,quarantine deny;
  class note note;
```

---

## 38 - API-host bridge proof stack

Killer line: the bridge proof is green and scope-bounded; it does not claim every
API host on the internet or external paid/keyed state.

[SVG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/38-api-host-bridge-proof-stack.svg) - [PNG](https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/38-api-host-bridge-proof-stack.png) - [source](https://github.com/anthony-chaudhary/fak/blob/main/visuals/38-api-host-bridge-proof-stack.mmd)

```mermaid
%%{init: {'theme':'base','themeVariables':{'fontFamily':'Segoe UI, Helvetica, Arial, sans-serif','fontSize':'14px','lineColor':'#5B6B7B','primaryColor':'#DCE9FB','primaryTextColor':'#1B3A66','primaryBorderColor':'#3B6FB5','clusterBkg':'#F7F9FB','clusterBorder':'#AFC0CE'}}}%%
flowchart TB
  claim["Bridge claim<br/>compatible API host stays behind FAK<br/>while FAK owns the tool boundary"]

  source["Source audit<br/>external permission claims verified: 7/7"]
  bench["Permission benchmark<br/>FAK/DOS hard risk coverage: 6/6<br/>bridge controls: 6/6"]
  gate["Executed witness gate<br/>required commands passed: 9/9"]
  proof["Proof rollup<br/>requirements proven: 16/16<br/>scope: BRIDGE_PROVEN<br/>SCOPE_BOUNDED"]
  verify["Verify-all<br/>executed steps passed: 35/35<br/>scope-bounded gate: yes"]

  residual["Residual scope<br/>not proven: every API host on the internet<br/>external state: paid/keyed live hosts"]

  claim --> source --> bench --> gate --> proof --> verify
  proof -.-> residual
  verify -.-> residual

  classDef kernel fill:#DCE9FB,stroke:#3B6FB5,stroke-width:2px,color:#1B3A66;
  classDef gate fill:#FFF3CC,stroke:#C9A227,stroke-width:2px,color:#6B5410;
  classDef pass fill:#DBF3E1,stroke:#3FA45F,stroke-width:2px,color:#1C5230;
  classDef deny fill:#FBDDDD,stroke:#C9453F,stroke-width:2px,color:#6E1F1B;
  classDef note fill:#F7F9FB,stroke:#AFC0CE,stroke-width:1px,color:#46586A;

  class claim kernel;
  class source,bench,gate,proof,verify pass;
  class residual deny;
```
