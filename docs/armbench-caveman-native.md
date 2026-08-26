---
title: "Caveman native control benchmark"
description: "This command reproduces JuliusBrussee/caveman@c72984e4392c7a154e55c11dbf445f01ce5c35d4's audience benchmark without fak in either inference arm."
---
# Caveman native control benchmark

This command reproduces `JuliusBrussee/caveman@c72984e4392c7a154e55c11dbf445f01ce5c35d4`'s audience benchmark without fak in either inference arm.

```powershell
fak armbench caveman-native --out docs/_witnesses/armbench-caveman-native/exact --base-url https://api.anthropic.com/v1 --api-key-env ANTHROPIC_API_KEY
```

The default exact manifest fixes `claude-sonnet-4-20250514`, temperature 0, 4096 maximum output tokens, three trials, upstream's ten prompts, normal system text, and Caveman skill text. It refuses fixture hash drift. A replacement model must use `--model` and `--label replacement-...`; the resulting manifest says `exact_model: false`.

The control calls one provider endpoint directly through its OpenAI-compatible API and records the complete provider response, output, usage, and deterministic task-specific semantic checks. fak only orchestrates and records the benchmark; it is not in either inference path and no fak performance claim follows from this result.

## Native medium paired arm (#8785)

The existing gate now runs three otherwise-identical arms: `normal` (baseline), `caveman` (the pinned upstream original), and `native_medium`. The third arm is rendered at run time by `internal/syspromptmmu.ResolveStyle("caveman:native:medium")`; the benchmark does not copy the fragment. Each v2 manifest records the canonical profile identity and SHA-256 digest under `Profiles.native_medium`, while retaining the pinned corpus, model, temperature, trial count, semantic gates, and raw provider responses used by both comparators.

Generate the deterministic construction witness without credentials or provider traffic:

```bash
go run ./cmd/fak armbench caveman-native --dry-run \
  --input docs/_witnesses/armbench-caveman-native/inputs \
  --out docs/_witnesses/armbench-caveman-native/no-spend-native-medium \
  --model deterministic-fixture --label replacement-no-spend
```

The checked-in no-spend receipt proves arm selection, canonical identity/digest capture, unchanged semantic-gate plumbing, summary aggregation, and raw-output preservation. Its synthetic outputs and zero token counts are **not effectiveness evidence**. A live receipt, when separately authorized, must report semantic pass rate and safety failures before output-token delta and must bound findings to its measured model and corpus. Response terseness does not establish input/context savings.
