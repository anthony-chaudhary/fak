# Production completion and project scope

`complete` means **production complete** unless a ticket explicitly declares a narrower completion standard. A demo, prototype, experiment, research result, staging integration, or toy model bring-up can close under its named standard, but it contributes no production-complete credit.

Every dispatchable ticket declares:

- `## Work estimate` — positive points for the work in that ticket;
- `## Overall completion contribution` — `N/M points`, binding the ticket to its parent's declared production baseline;
- `## Completion standard` — production by default, or an explicit narrower maturity.


## Maturity vocabulary

| Standard | What a close proves | Production credit |
|---|---|---:|
| `research` / `experiment` | A question was investigated or a hypothesis measured. | 0 |
| `prototype` / `demo` | A bounded example works; toy inputs and manual setup are allowed when declared. | 0 |
| `development` / `dev` | The implementation works in a development path but has not crossed integration/operations gates. | 0 |
| `integrated` / `staging` | The real integration path works in a pre-production environment. | 0 |
| `production` | The declared supported path, operating envelope, failure handling, observability, operator/docs path, and matched witness are complete. | Full declared contribution |

Use the maturity word with the claim: `demo complete`, `prototype complete`, or `production complete`. Bare `complete`, `done`, `shipped`, `ready`, and `model ready` are production claims by default.

## Estimation and author checklist

Points are relative work units, not elapsed-time promises. Use one stable local scale (for example 1/2/3/5/8), include uncertainty in prose, and keep estimate points equal to contribution points. Before filing, verify:

1. The parent issue owns a named production scope and point baseline.
2. This ticket states `Estimate: N points`.
3. Its contribution states `N/M points`, where `M` is that parent baseline.
4. Its completion standard is explicit; omit custom wording only when production is intended.
5. Done/witness conditions match the completion standard and target operating envelope.
6. Sibling contributions cover the baseline; unknown or deferred work remains visible rather than guessed away.

Project progress is weighted, not an issue count. Run:

```powershell
gh issue list --state all --limit 1000 --json number,title,body,state,labels,url > issues.json
fak project completion --from-issues issues.json
fak project completion --from-issues issues.json --json
```

The report separates closed maturity buckets from `production_complete_points`. Missing metadata, conflicting parent denominators, multiple parents, an uncovered baseline, or contributions above the baseline set confidence to `incomplete`; the tool does not guess legacy estimates.

Example: a 2-point tokenizer demo and 3-point single-request prototype can both close, while a 15-point production serving path remains open. Against a 20-point parent baseline, the truthful result is **0/20 production-complete points (0%)**, with five closed non-production points still visible. This is the model-bring-up guardrail: bring-up evidence is useful, but “model ready” means production-ready only when the production-scoped work and witness are complete.

Rebaseline by editing the parent scope and every child denominator together, with the reason recorded. Never improve the percentage by silently deleting work from the denominator.
