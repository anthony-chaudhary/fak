---
title: "Latest-generation-only model evaluation"
description: "fak model acceptance-run spends provider calls only on each declared model family's latest generation. Every model entry must declare:"
---
# Latest-generation-only model evaluation

`fak model acceptance-run` spends provider calls only on each declared model family's
`latest` generation. Every model entry must declare:

- `family`: a stable lineage such as `anthropic/sonnet`;
- `generation`: the exact generation label used to distinguish replacements;
- `lifecycle`: `latest` or `tombstoned`.

## One latest entry per family

A family has exactly one `latest` entry. When a replacement arrives, change the former
entry to `tombstoned`, add the replacement as `latest`, and keep the old row as a durable
record. The runner prints `SKIP` for that tombstone and does not invoke a provider. The
gate records the model verdict as `SKIP`, so missing old-model runs do not hold the current
campaign.

## Older-generation exceptions

Older-generation evaluation is exceptional, not a compatibility matrix. To run one, add
an explicit `eval_exception` with a non-empty named reason; add a `ticket` when the work is
tracked:

```json
{
  "model": "vendor-pro-5",
  "family": "vendor/pro",
  "generation": "5",
  "lifecycle": "tombstoned",
  "requested_tier": 1,
  "eval_exception": {
    "reason": "regression bisect for structured tool calls",
    "ticket": "#7001"
  }
}
```

## Retiring an exception

Remove the exception when that named investigation ends. Do not retain old generations
merely to prove broad backward compatibility; a new ticket or similarly specific reason
must justify each such run.
