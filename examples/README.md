# Policy Manifest Examples

These files are starter floors for the gateway-first adoption path. They are not
universal security policies; each one is a reviewable allow-list with a concrete
deny witness an adopter can run before putting it in front of an agent.

Run checks from `fak/`:

```bash
go run ./cmd/fak policy --check examples/customer-support-readonly-policy.json
go run ./cmd/fak policy --check examples/research-agent-policy.json
go run ./cmd/fak policy --check examples/devops-dryrun-policy.json
go run ./cmd/fak policy --check examples/dev-agent-policy.json
```

## Templates

| File | Intended use | Main boundary | Witness command |
|---|---|---|---|
| `policy.example.json` | General manifest shape | explicit destructive denies + provenance/IFC fields | `go run ./cmd/fak preflight --policy examples/policy.example.json --tool delete_account --args "{}"` |
| `dev-agent-policy.json` | Coding agent in this repo | no shared-history mutations without release discipline | `go run ./cmd/fak preflight --policy examples/dev-agent-policy.json --tool git_push --args "{}"` |
| `customer-support-readonly-policy.json` | Support lookup + ticket handoff | read/customer-ticket workflow; no direct account, refund, or email action | `go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"` |
| `research-agent-policy.json` | Open-web research and note taking | read/search/summarize; no posting, shell, upload, or arbitrary note path | `go run ./cmd/fak preflight --policy examples/research-agent-policy.json --tool send_email --args "{}"` |
| `devops-dryrun-policy.json` | Infra review without execution | plan/diff/template only; no apply, exec, delete, or production deploy | `go run ./cmd/fak preflight --policy examples/devops-dryrun-policy.json --tool terraform_apply --args "{}"` |

The useful adoption pattern is: copy the closest template, delete what you do not
need, run `policy --check`, then run one or two `preflight` calls that prove the
most important dangerous actions are denied with closed reason codes.
