# Caveman native control benchmark

This command reproduces `JuliusBrussee/caveman@c72984e4392c7a154e55c11dbf445f01ce5c35d4`'s audience benchmark without fak in either inference arm.

```powershell
fak armbench caveman-native --out docs/_witnesses/armbench-caveman-native/exact --base-url https://api.anthropic.com/v1 --api-key-env ANTHROPIC_API_KEY
```

The default exact manifest fixes `claude-sonnet-4-20250514`, temperature 0, 4096 maximum output tokens, three trials, upstream's ten prompts, normal system text, and Caveman skill text. It refuses fixture hash drift. A replacement model must use `--model` and `--label replacement-...`; the resulting manifest says `exact_model: false`.

The control calls one provider endpoint directly through its OpenAI-compatible API and records the complete provider response, output, usage, and deterministic task-specific semantic checks. fak only orchestrates and records the benchmark; it is not in either inference path and no fak performance claim follows from this result.
