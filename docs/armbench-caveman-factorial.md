# Caveman context-treatment factorial (#6683)

**Verdict: SHIPPED with bounded evidence.** The harness crosses normal and pinned Caveman style with passthrough, tool-result compression only, context shedding only, both, and the tuned bundle over native one-shot and growing multi-turn tool-result workloads at pressure 1/4/12. Routing, policy, and semantic response reuse are explicitly disabled in every manifest.

Comparator: `JuliusBrussee/caveman@c72984e4392c7a154e55c11dbf445f01ce5c35d4`; the command verifies the imported comparator hashes before running. Live endpoint/model: `http://127.0.0.1:64355/v1`, `gpt-5.6-sol`. Evidence: [deterministic manifest](_witnesses/armbench-caveman-factorial/deterministic/manifest.json) and [live manifest](_witnesses/armbench-caveman-factorial/live-gpt-5.6-sol/manifest.json).

## Evidence boundary

The manifests keep provider input, cache-read, cache-write, and provider output tokens in distinct fields; estimated output tokens are separately labeled. The live run contains real provider receipts: input 93–15131, output 80–452, and 36096 cache-read tokens. Retained facts are a separate semantic quantity. No price table or bill was supplied, so this report makes no cost or net-efficiency claim. Cache observations are provider receipts, not a promise that another provider will cache the same arms.

## Pressure-12 operating point

| Style | Treatment | Provider input | Cache read | Output | Facts | Final bytes | Transform CPU ns |
|---|---|---:|---:|---:|---:|---:|---:|
| normal | passthrough | 15112 | 4864 | 336 | 12/12 | 65856 | 3 |
| normal | tool-result-compression | 12052 | 3840 | 432 | 12/12 | 52248 | 3 |
| normal | context-shedding | 10110 | 0 | 303 | 11/12 | 43986 | 976602 |
| normal | compression+shedding | 11058 | 0 | 420 | 11/12 | 47907 | 518602 |
| normal | tuned-bundle | 11067 | 0 | 452 | 11/12 | 47944 | 1041101 |
| caveman | passthrough | 15131 | 4864 | 336 | 12/12 | 65885 | 3 |
| caveman | tool-result-compression | 12071 | 0 | 276 | 12/12 | 52277 | 3 |
| caveman | context-shedding | 10129 | 0 | 371 | 12/12 | 44015 | 512102 |
| caveman | compression+shedding | 11077 | 0 | 351 | 11/12 | 47936 | 522102 |
| caveman | tuned-bundle | 11086 | 9984 | 302 | 11/12 | 47973 | 1071601 |

All 60 live cells are retained in the manifest, including pressure 1 and 4 and the native one-shot workload. Quality was perfect in 55 cells; 5 cells exposed a frontier tradeoff rather than hiding it. The machine-readable frontier has 35 non-dominated points.

## Interaction effects

Difference-in-differences rows compare each fak treatment with passthrough in normal versus Caveman style. The live table classifies 1 complementary, 23 redundant, and 0 harmful cells under its byte/output/fact rule. This run therefore finds the features mostly **redundant with respect to style interaction**, while the managed-context treatments still change provider input independently of answer terseness. Treat the one complementary cell as workload evidence, not a universal claim.

Each cell records three named stages with CPU nanoseconds and bytes/estimated tokens before and after. Disabled stages remain explicit no-op receipts. Tuned adds stable-prefix cache layout after compression and shedding; its provider cache values remain whatever the provider actually reported.

## Reproduce

```powershell
go build -o $env:TEMP\fak-armbench-factorial.exe ./cmd/fak
& $env:TEMP\fak-armbench-factorial.exe armbench caveman-factorial --out docs/_witnesses/armbench-caveman-factorial/deterministic
& $env:TEMP\fak-armbench-factorial.exe armbench caveman-factorial --out docs/_witnesses/armbench-caveman-factorial/live-gpt-5.6-sol --base-url $env:OPENAI_BASE_URL --model gpt-5.6-sol
```
