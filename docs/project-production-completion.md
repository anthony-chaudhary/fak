# Production completion and project scope

`complete` means **production complete** unless a ticket explicitly declares a narrower completion standard. A demo, prototype, experiment, research result, staging integration, or toy model bring-up can close under its named standard, but it contributes no production-complete credit.

Every dispatchable ticket declares:

- `## Work estimate` — positive points for the work in that ticket;
- `## Overall completion contribution` — `N/M points`, binding the ticket to its parent's declared production baseline;
- `## Completion standard` — production by default, or an explicit narrower maturity.

Project progress is weighted, not an issue count. Run:

```powershell
gh issue list --state all --limit 1000 --json number,title,body,state,labels,url > issues.json
fak project completion --from-issues issues.json
fak project completion --from-issues issues.json --json
```

The report separates closed maturity buckets from `production_complete_points`. Missing metadata, conflicting parent denominators, multiple parents, an uncovered baseline, or contributions above the baseline set confidence to `incomplete`; the tool does not guess legacy estimates.

Example: a 2-point tokenizer demo and 3-point single-request prototype can both close, while a 15-point production serving path remains open. Against a 20-point parent baseline, the truthful result is **0/20 production-complete points (0%)**, with five closed non-production points still visible. This is the model-bring-up guardrail: bring-up evidence is useful, but “model ready” means production-ready only when the production-scoped work and witness are complete.

Rebaseline by editing the parent scope and every child denominator together, with the reason recorded. Never improve the percentage by silently deleting work from the denominator.
