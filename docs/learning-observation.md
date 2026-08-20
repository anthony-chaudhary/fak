---
title: "Learning observation lineage in fak"
description: "How fak records content-addressed observations, candidates, replay witnesses, and keep or reject verdicts without bypassing admission policy."
---

# Learning observation lineage

`fak learning-observation` keeps a local, durable, content-addressed graph connecting source observations, candidates, replay witnesses, and explicit keep/reject verdicts. The observation-and-relation mechanism is borrowed from EvoScientist v0.2.6 at commit `12adc6286881e94d23c5711225da883f7a6e3f42`; fak uses its own schema and implementation.

This graph does **not** admit a skill, generate a candidate, run a replay, or decide an outcome. It records provenance supplied by the caller. fak's witness-gated admission remains a separate policy: a `kept` record is evidence about a decision, not authority to bypass the existing admission gate.

Records have four kinds: `observation`, `candidate`, `witness`, and `verdict`. Verdicts require the closed outcome `kept` or `rejected`. Edges use only `observed-from`, `supports`, `contradicts`, `proposes`, `tested-by`, `kept-as`, and `rejected-as`.

```text
fak learning-observation add --kind observation --source trajectory://run/7 --content "retry recovered"
fak learning-observation add --kind candidate --source issue://5982 --content "bound retries"
fak learning-observation link --from lo_... --relation tested-by --to lo_...
fak learning-observation trace --candidate lo_...
```

Use `--store PATH` for an explicit store. Otherwise fak writes `<UserConfigDir>/fak/learning-observations.json`; `FAK_LEARNING_OBSERVATION_STORE` overrides that default. Duplicate normalized kind/source/content is idempotent, while reusing a kind/source pair for conflicting content is denied. Links deny unknown relation names, dangling IDs, and cycles. Trace follows only stored outgoing edges, so it cannot invent lineage.
