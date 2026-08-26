# Issue #8130 captured witness

This directory turns the research done condition into an executable document contract. It does not claim a hardware result.

## Fail-before / pass-after

At base `d5c84386cf42aa2aaec77de88d5519071f598680`, the map did not exist, so the same command failed while loading `../../research/qwen38-upstream-support-map-2026-08-26.md` (`file not found`). After adding the map:

```text
> go test ./docs/_witnesses/issue-8130 -count=1
ok github.com/anthony-chaudhary/fak/docs/_witnesses/issue-8130
```

The test requires six immutable source pins, all five requested runtimes, architecture-sensitive tool/context/cache/quant/MTP/multimodal facts, at least eight typed decisions, PRESENT/PARTIAL/ABSENT fak evidence, source limitations, stale policy, and the issue link.

Source pins were captured on 2026-08-26 with:

```powershell
$repos = 'QwenLM/Qwen3.8','huggingface/transformers','vllm-project/vllm','ggml-org/llama.cpp','sgl-project/sglang','ml-explore/mlx-lm'
$repos | ForEach-Object { gh api "repos/$_/commits?per_page=1" --jq '.[0] | [.sha,.commit.committer.date] | @tsv' }
```

Hardware-dependent gaps are explicitly follow-ons in the map. They require sanctioned fleet execution and exact fak-native engine/artifact receipts; this documentary witness does not replace them.
