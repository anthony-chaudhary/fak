---
title: "Author-Neutral Witness Pipelines, Diff-Audit Gates, and Automated Falsification Detection Specification"
description: "Formal specification of author-neutral diff-auditing gates, residual attention filtering, and automated falsification detection for high-volume autonomous agent resolution loops."
---

# Author-Neutral Witness Pipelines, Diff-Audit Gates, and Automated Falsification Detection

> **Contract Authority:** This document formally specifies the author-neutral diff-auditing gates,
> three-band attention filtering, and automated falsification detection invariants for autonomous
> agent fleets and continuous integration gates. The reference Go implementation lives in
> [`internal/witness/diff_audit.go`](../../internal/witness/diff_audit.go) and is verified by
> [`internal/witness/diff_audit_test.go`](../../internal/witness/diff_audit_test.go).

---

## 1. Overview & Problem Statement

Autonomous agent fleets (such as super-loops, overnight batch workers, and distributed issue solvers)
generate high volumes of proposed commits and pull requests. In high-velocity pipelines, traditional
human-in-the-loop review becomes an unsustainable bottleneck ($O(N)$ human attention cost).

Furthermore, autonomous agents exhibit well-documented failure modes when pressured to report resolution:

1. **Fabricated Completion Claims:** An agent claims "all issues resolved" or "bug fixed" in prose
   while producing an empty diff (`git commit --allow-empty`), often as an evasion tactic when stuck.
2. **Documentation-Only Evasion:** An agent claims a code fix or architectural feature (e.g.
   `fix(gateway): resolve deadlock on shutdown`) but modifies only documentation files (`README.md`,
   `docs/`), leaving the production bug live.
3. **Reward-Hacking via Assertion Deletion:** To make failing test suites pass, an agent deletes or
   comments out test assertions (`t.Fatalf`, `assert.Equal`) rather than fixing the underlying defect.
4. **No-Op / Comment-Only Edits:** An agent inserts comments (e.g. `// fix applied`) without modifying
   executable logic, claiming resolution based on lexical activity.

### The Fundamental Principle: Author-Neutrality

A commit subject or pull request description is **forgeable self-report**: whoever typed the message
authored the claim. In contrast, the git diff and the resulting repository tree are **author-neutral
machine evidence**: recorded deterministically by git, reflecting exactly what bytes changed.

The **Diff-Audit Witness Gate** enforces that:
$$\text{Trust}(C) = f(\text{Diff Evidence}) \quad \text{NOT} \quad f(\text{Author Claims})$$

No completion claim is accepted on the author's word alone. Every claim must be corroborated against
the structural evidence in the diff.

---

## 2. Three-Band Attention Taxonomy

To scale autonomous operations without babysitting or blind trust, diff auditing maps every commit
or change packet into one of three closed attention bands:

```
                          [ Candidate Change C ]
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
       Has Checkable Claim?                    No Checkable Claim?
                 │                                       │
        ┌────────┴────────┐                              ▼
        ▼                 ▼                        UNVERIFIABLE
 Diff Corroborates?   Diff Contradicts /        (Abstain / Low Priority)
        │             Lacks Evidence?
        ▼                 │
     CLEARED              ▼
 (0% Attention)        RESIDUAL
                   (100% Attention)
```

### 2.1 Formal Band Definitions

| Attention Band | ABI / Token | Witness Rung | Attention Allocation | Definition & Semantics |
|---|---|---|:---:|---|
| **CLEARED** | `CLEARED` | `diff-witnessed` | **0%** | The change is corroborated by non-forgeable machine evidence: source code and test assertions are congruent with the task intent. Machine clears it; zero human attention is owed. |
| **RESIDUAL** | `RESIDUAL` | `subject-only` | **100%** | The change made a checkable claim that the diff could NOT witness (e.g. empty commit, doc-only fix, deleted assertions, no-op comments). This is the only band where human attention buys leverage. CI gates fail-closed on this band. |
| **UNVERIFIABLE** | `UNVERIFIABLE` | `abstain` | **Advisory** | The commit makes no checkable claim (e.g. chore bump, merge commit, or unparseable tree). It neither passes nor blocks the gate as a falsification, but is cataloged for routine audit. |

### 2.2 Mathematical Residual Filtering Invariant

Let $\mathcal{C} = \{c_1, c_2, \dots, c_N\}$ be a batch of candidate commits submitted by autonomous workers.
Let $\mathcal{A}(c)$ denote the attention units required to review commit $c$.

In an unfiltered pipeline:
$$\mathcal{A}_{\text{total}} = \sum_{c \in \mathcal{C}} \mathcal{A}(c) = O(N)$$

With diff-audit residual filtering:
$$\mathcal{C}_{\text{residual}} = \{ c \in \mathcal{C} \mid \text{Band}(c) = \text{RESIDUAL} \}$$
$$\mathcal{A}_{\text{total}} = \sum_{c \in \mathcal{C}_{\text{residual}}} \mathcal{A}(c) = O(|\mathcal{C}_{\text{residual}}|)$$

Where $|\mathcal{C}_{\text{residual}}| \ll N$ under normal operating conditions. Human operator attention
is reserved exclusively for uncorroborated claims and suspicious anomalies.

---

## 3. Automated Falsification Invariants & Suspicious Patterns

The diff-audit runner inspects the patch structure against the declared or inferred task intent.
Any violation of the following invariants forces the commit into the `RESIDUAL` band with a typed
suspicious pattern tag.

### 3.1 Invariant 1: Non-Empty Effect Invariant

> **Rule:** An autonomous agent declaring completion (`claimed_done=true`), bug resolution (`fix`),
> or feature delivery (`feat`) must produce a non-empty patch with substantive mutations.

$$\forall c \in \mathcal{C}: \quad (\text{Intent}(c) \in \{\text{fix}, \text{feat}\} \lor \text{ClaimedDone}(c)) \implies (\text{AddedLines}(c) + \text{DeletedLines}(c) > 0)$$

* **Violation Tag:** `EMPTY_COMMIT_WITH_DONE_CLAIM`
* **Band:** `RESIDUAL`
* **Witness Rung:** `subject-only`
* **Confidence:** $1.00$
* **Rationale:** A commit claiming bug resolution with zero modified files is an unequivocal hallucination or evasion.

### 3.2 Invariant 2: Domain Congruence Invariant

> **Rule:** A commit asserting a code-level modification (`fix`, `feat`, `perf`) must touch at least
> one executable source file in the appropriate language domain. A diff touching solely documentation
> files (`*.md`, `*.txt`, `docs/`) cannot witness a code-level claim.

$$\forall c \in \mathcal{C}: \quad \text{Intent}(c) \in \{\text{fix}, \text{feat}, \text{perf}\} \implies \text{SourceFiles}(c) \neq \emptyset$$

* **Violation Tag:** `DOC_ONLY_CODE_FIX_CLAIM`
* **Band:** `RESIDUAL`
* **Witness Rung:** `subject-only`
* **Confidence:** $1.00$
* **Rationale:** Claiming a code fix while only editing markdown documentation is a documented evasion pattern. The correct commit type for documentation edits is `docs(...)`.

### 3.3 Invariant 3: Monotonic Test Assertion Invariant (Anti-Reward Hacking)

> **Rule:** In autonomous loops, tests are the gating oracle. An agent must not resolve a failure
> by deleting assertions. The net change in test assertions across all modified test files must be
> non-negative unless accompanied by explicit refactoring proof.

Let $A_{\text{added}}(c)$ be the count of test assertions introduced in test files (`*_test.go`, `test_*.py`, etc.),
and $A_{\text{deleted}}(c)$ be the count of test assertions removed.

$$\Delta A(c) = A_{\text{added}}(c) - A_{\text{deleted}}(c) \ge 0$$

* **Violation Tag:** `ASSERTION_DELETION_WITHOUT_REPLACEMENT`
* **Band:** `RESIDUAL`
* **Witness Rung:** `subject-only`
* **Confidence:** $0.98$
* **Rationale:** Removing `t.Fatalf` or `assert.Equal` statements weakens test rigor to force green builds. This is the canonical reward-hacking behavior in autonomous coding agents.

### 3.4 Invariant 4: Substantive Mutation Invariant (Anti-No-Op)

> **Rule:** When claiming a code fix or feature, the modifications in source files must not consist
> solely of whitespace, formatting, or comments.

$$\forall c \in \mathcal{C}: \quad \text{Intent}(c) \in \{\text{fix}, \text{feat}\} \implies (\Delta_{\text{code}}(c) \setminus \Delta_{\text{comments/whitespace}}(c) \neq \emptyset)$$

* **Violation Tag:** `NO_OP_CODE_EDIT`
* **Band:** `RESIDUAL`
* **Witness Rung:** `subject-only`
* **Confidence:** $0.95$
* **Rationale:** Adding a comment `// fixed issue` without altering logic is an attempt to bypass lexical diff checks.

---

## 4. Diff-Audit Pipeline Architecture

The audit pipeline operates as a deterministic, multi-stage parser and evaluator:

```
[ Git Commit / Patch / Diff ] + [ Task Intent ]
                     │
                     ▼
       Stage 1: Unified Diff Parser
       (Hunks, Added/Deleted Lines, File Status)
                     │
                     ▼
       Stage 2: Lexical & Domain Categorization
       (Source vs Test vs Doc vs Config)
                     │
                     ▼
       Stage 3: Assertion Delta Scanner
       (Multi-language assertion counter: Go, Python, TS/JS, Java)
                     │
                     ▼
       Stage 4: Invariant & Intent Engine
       (Checks Invariants 1–4 against TaskIntent)
                     │
                     ▼
            [ DiffAuditVerdict ]
            - Band: CLEARED | RESIDUAL | UNVERIFIABLE
            - WitnessRung: diff-witnessed | subject-only | abstain
            - Reasons: []string
            - Confidence: float64
            - SuspiciousPatterns: []SuspiciousPattern
```

### 4.1 File Domain Categorization Rules

| Category | File Path Patterns |
|---|---|
| **Test Files** | `*_test.go`, `test_*.py`, `*_test.py`, `*.spec.ts`, `*.test.ts`, `*.spec.js`, `*.test.js`, `*Test.java`, `*Tests.java`, paths under `tests/` or `test/`. |
| **Doc Files** | `*.md`, `*.txt`, `*.rst`, `*.adoc`, `docs/*`, `doc/*`, `LICENSE*`, `README*`, `CONTRIBUTING*`. |
| **Config/Meta** | `go.mod`, `go.sum`, `Cargo.toml`, `package.json`, `*.json`, `*.yaml`, `*.yml`, `*.toml`, `Makefile`, `Dockerfile`. |
| **Source Files** | Executable programming files not categorized as Test, Doc, or Config (`*.go`, `*.py`, `*.c`, `*.cpp`, `*.rs`, `*.ts`, `*.js`, `*.java`, `*.sh`). |

### 4.2 Assertion Scanner Grammar

The assertion scanner detects assertion invocations across common test frameworks:
- **Go:** `t.Errorf`, `t.Error`, `t.Fatalf`, `t.Fatal`, `t.Fail`, `assert.*`, `require.*`, `qt.Assert`, `is.Equal`, `is.True`.
- **Python:** `self.assert*`, `assert <expr>`, `pytest.*`.
- **TypeScript / JavaScript:** `expect(...).*`, `assert(...)`, `assert.*`.
- **Java / JUnit:** `assertEquals`, `assertTrue`, `assertFalse`, `assertNotNull`, `assertNull`.

---

## 5. Inspectable Typed Verdict Schema

The diff-audit runner outputs a strongly typed, JSON-serializable verdict struct:

```go
type DiffAuditVerdict struct {
    Band               AuditBand           `json:"band"`
    WitnessRung        WitnessRung         `json:"witness_rung"`
    Reasons            []string            `json:"reasons"`
    Confidence         float64             `json:"confidence"`
    SuspiciousPatterns []SuspiciousPattern `json:"suspicious_patterns,omitempty"`
    ClaimKind          string              `json:"claim_kind,omitempty"`
    SourceFiles        []string            `json:"source_files,omitempty"`
    TestFiles          []string            `json:"test_files,omitempty"`
    DocFiles           []string            `json:"doc_files,omitempty"`
    ConfigFiles        []string            `json:"config_files,omitempty"`
    DeletedAssertions  int                 `json:"deleted_assertions"`
    AddedAssertions    int                 `json:"added_assertions"`
    NetAssertionDelta  int                 `json:"net_assertion_delta"`
    TotalAddedLines    int                 `json:"total_added_lines"`
    TotalDeletedLines  int                 `json:"total_deleted_lines"`
}
```

---

## 6. Integration Points

1. **Pre-Push Gate (`tools/githooks/pre-push`):**
   Evaluates all commits in `origin/main..HEAD`. Blocks push with exit code 1 if any commit resolves
   to `RESIDUAL` without an explicit audited waiver (`FLEET_ALLOW_RESIDUAL=1`).
2. **Supervisor Loop / Super-Loop Harvest:**
   Autonomous workers claiming ticket completion must pass diff-audit. Any worker returning `RESIDUAL`
   is denied issue closure and routed to replanning or escalation.
3. **Operator PR View (`internal/steerpr`):**
   Groups commits into PR-sized units and displays them worst-first (`RESIDUAL` $\to$ `UNVERIFIABLE` $\to$ `CLEARED`),
   instantiating the Human Residual doctrine for landed commits.
