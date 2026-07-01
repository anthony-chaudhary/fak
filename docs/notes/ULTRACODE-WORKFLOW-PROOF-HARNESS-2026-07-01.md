---
title: "Ultracode workflow proof harness: shared path + fak concept embedding"
date: 2026-07-01
issue: 1503
---

# Ultracode Workflow Proof Harness (#1503)

This is the committed C5 proof artifact for #1503. It grades the two load-bearing
premises from a clean `origin/main` archive, using command outputs and hashes rather
than agent self-reports.

Witness tree:

```text
origin/main = 4fe285912b59be3e0925afd5586285f2fe709470
```

## Verdicts

| Premise | Verdict | Evidence |
| --- | --- | --- |
| P1: shared fak self-query path | PROVEN | Three independent reader invocations over the same clean tree returned byte-identical `fak index lane --json` output and byte-identical `fak memory drivers --json` output. |
| P2: workflow embedding | PROVEN | `fak workflow seed` linted `FAK-NATIVE` with `self-index`, `memory`, and `shared-path` present; a stripped generic workflow linted `FAK-BLIND` with all three classes missing. |

## Reader Rows

Each reader used the same two read-only surfaces:

```bash
/tmp/fak-1503 index lane --json cmd/fak/workflow.go internal/workflowlint/workflowlint.go internal/workflowlint/seed.js
/tmp/fak-1503 memory drivers --json
```

| Reader | Index digest | Memory-driver digest |
| --- | --- | --- |
| reader-a | `e086df458d2f2cd42a2c1fe70ee5beea5ccc6d3106c5a9c7f6a0cda6e82b180e` | `c59f3fe0a42cee4b85b8a946e42826c9003ce964d79c6d9abf8f69bfe8206510` |
| reader-b | `e086df458d2f2cd42a2c1fe70ee5beea5ccc6d3106c5a9c7f6a0cda6e82b180e` | `c59f3fe0a42cee4b85b8a946e42826c9003ce964d79c6d9abf8f69bfe8206510` |
| reader-c | `e086df458d2f2cd42a2c1fe70ee5beea5ccc6d3106c5a9c7f6a0cda6e82b180e` | `c59f3fe0a42cee4b85b8a946e42826c9003ce964d79c6d9abf8f69bfe8206510` |

The witnessed index payload was:

```json
[
  {
    "path": "cmd/fak/workflow.go",
    "lane": "cmd",
    "stamp": "(fak cmd)"
  },
  {
    "path": "internal/workflowlint/workflowlint.go",
    "lane": "workflowlint",
    "stamp": "(fak workflowlint)"
  },
  {
    "path": "internal/workflowlint/seed.js",
    "lane": "workflowlint",
    "stamp": "(fak workflowlint)"
  }
]
```

The witnessed memory-driver payload was 7,233 bytes and contains the built-in
strategy names `clean`, `codex-recall`, `compact`, `dream`, `recall`, and `render`.
All three reader rows hashed that payload to the same digest shown above.

## Workflow Embedding Evidence

Generated workflow under test:

```bash
/tmp/fak-1503 workflow seed > /tmp/fak-1503-seed.js
/tmp/fak-1503 workflow lint --json /tmp/fak-1503-seed.js
```

Seed digest:

```text
41e867bc807f59b2d3998f8b2815c1e3305b1b362166b8eec2ca487287314f65  /tmp/fak-1503-seed.js
```

Lint digest:

```text
f667db9bb512d8e15a78be873bed58fef5ce3789e3e44d06e9d4a996f3aacf4d  /tmp/fak-1503-seed-lint.json
```

Captured verdict:

```json
{
  "native": true,
  "verdict": "FAK-NATIVE",
  "classes": [
    {
      "key": "self-index",
      "present": true,
      "matched": ["fak index", "fak_index"]
    },
    {
      "key": "memory",
      "present": true,
      "matched": ["--driver clean", "--driver compact", "--driver recall", "fak memory"]
    },
    {
      "key": "shared-path",
      "present": true,
      "matched": ["collision_risk", "dos_arbitrate", "lease"]
    }
  ]
}
```

Adversarial stripped copy:

```bash
printf "generic workflow no project concepts\nreview changed files\n" > /tmp/fak-1503-blind.js
/tmp/fak-1503 workflow lint --json /tmp/fak-1503-blind.js
```

The stripped-copy lint exited nonzero. Captured digest:

```text
93fa242bb869bc9602fa62b3a43df85c3c150a415e2beb9f7c6031f7757da554  /tmp/fak-1503-blind-lint.json
```

Captured verdict:

```json
{
  "native": false,
  "verdict": "FAK-BLIND",
  "classes": [
    {
      "key": "self-index",
      "present": false
    },
    {
      "key": "memory",
      "present": false
    },
    {
      "key": "shared-path",
      "present": false
    }
  ],
  "missing": ["self-index", "memory", "shared-path"]
}
```

## Reproduction

The witness run used this clean snapshot command sequence:

```bash
rm -rf /tmp/fak-origin-1503 /tmp/fak-1503
mkdir -p /tmp/fak-origin-1503
cd /mnt/c/work/fak
git archive origin/main | tar -xf - -C /tmp/fak-origin-1503
cd /tmp/fak-origin-1503
GOTOOLCHAIN=auto go build -o /tmp/fak-1503 ./cmd/fak

/tmp/fak-1503 index lane --json cmd/fak/workflow.go internal/workflowlint/workflowlint.go internal/workflowlint/seed.js > /tmp/fak-1503-reader-a-index.json
/tmp/fak-1503 memory drivers --json > /tmp/fak-1503-reader-a-memory.json
/tmp/fak-1503 index lane --json cmd/fak/workflow.go internal/workflowlint/workflowlint.go internal/workflowlint/seed.js > /tmp/fak-1503-reader-b-index.json
/tmp/fak-1503 memory drivers --json > /tmp/fak-1503-reader-b-memory.json
/tmp/fak-1503 index lane --json cmd/fak/workflow.go internal/workflowlint/workflowlint.go internal/workflowlint/seed.js > /tmp/fak-1503-reader-c-index.json
/tmp/fak-1503 memory drivers --json > /tmp/fak-1503-reader-c-memory.json

/tmp/fak-1503 workflow seed > /tmp/fak-1503-seed.js
/tmp/fak-1503 workflow lint --json /tmp/fak-1503-seed.js > /tmp/fak-1503-seed-lint.json
printf "generic workflow no project concepts\nreview changed files\n" > /tmp/fak-1503-blind.js
/tmp/fak-1503 workflow lint --json /tmp/fak-1503-blind.js > /tmp/fak-1503-blind-lint.json

sha256sum /tmp/fak-1503-reader-a-index.json /tmp/fak-1503-reader-b-index.json /tmp/fak-1503-reader-c-index.json
sha256sum /tmp/fak-1503-reader-a-memory.json /tmp/fak-1503-reader-b-memory.json /tmp/fak-1503-reader-c-memory.json
sha256sum /tmp/fak-1503-seed.js /tmp/fak-1503-seed-lint.json /tmp/fak-1503-blind-lint.json
```
