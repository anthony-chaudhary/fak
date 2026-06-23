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
go run ./cmd/fak policy --check examples/flight-booking-agent-policy.json
go run ./cmd/fak policy --check examples/healthcare-phi-policy.json
go run ./cmd/fak policy --check examples/sql-analyst-policy.json
```

## Templates

| File | Intended use | Main boundary | Witness command |
|---|---|---|---|
| `policy.example.json` | General manifest shape | explicit destructive denies + provenance/IFC fields | `go run ./cmd/fak preflight --policy examples/policy.example.json --tool delete_account --args "{}"` |
| `dev-agent-policy.json` | Coding agent in this repo | no shared-history mutations without release discipline | `go run ./cmd/fak preflight --policy examples/dev-agent-policy.json --tool git_push --args "{}"` |
| `customer-support-readonly-policy.json` | Support lookup + ticket handoff | read/customer-ticket workflow; no direct account, refund, or email action | `go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json --tool refund_payment --args "{}"` |
| `research-agent-policy.json` | Open-web research and note taking | read/search/summarize; no posting, shell, upload, or arbitrary note path | `go run ./cmd/fak preflight --policy examples/research-agent-policy.json --tool send_email --args "{}"` |
| `devops-dryrun-policy.json` | Infra review without execution | plan/diff/template only; no apply, exec, delete, or production deploy | `go run ./cmd/fak preflight --policy examples/devops-dryrun-policy.json --tool terraform_apply --args "{}"` |
| `flight-booking-agent-policy.json` | The README's canonical SFO→JFK booking agent (the [live A/B](../docs/benchmarks/LIVE-RESULTS.md) task) | search/book/read; refund, cancel, PNR-export, and fund transfer need a human; `read_policy` is `untrusted` (the booby-trap vector) | `go run ./cmd/fak preflight --policy examples/flight-booking-agent-policy.json --tool refund_payment --args "{}"` |
| `healthcare-phi-policy.json` | HIPAA-style clinical agent — the heavy-PHI `redact_fields` + EHR/inbox provenance showcase | read EHR / search ICD / drug-interaction / note / appointment; export, email-PHI, and record-delete need a human; `read_patient_record` is `trusted_local` while `fetch_medical_literature` and `read_patient_message` are `untrusted` (the prompt-injection vector) | `go run ./cmd/fak preflight --policy examples/healthcare-phi-policy.json --tool export_patient_data --args "{}"` |
| `sql-analyst-policy.json` | Internal-data / BI analyst — the heavy `arg_rules` showcase (SELECT-only SQL, row caps, schema allow-list) | read-only query / list / describe / chart / sanitized-CSV; write-query, DDL (`drop`/`alter`/`create_table`), `copy_to`/`pg_dump`, `shell`, and fund transfer are denied; the `run_read_query.sql` text is constrained to reject DDL/DML even when the tool name clears the allow-list | `go run ./cmd/fak preflight --policy examples/sql-analyst-policy.json --tool run_read_query --args '{"sql":"DROP TABLE customers"}'` |

`refund_payment` returns **`DENY (POLICY_BLOCK)`** — the denied refund is meant to
escalate to the `transfer_to_human_agents` safe sink, not silently fail.
[`escalation-demo/`](escalation-demo/README.md) is the runnable artifact that exercises
that path end-to-end (deny → catch → route to the declared `safe_sink` → redacted human
ticket), so the `safe_sinks` field every template declares is no longer decorative. The
ALLOW side of the same floor:

```bash
go run ./cmd/fak preflight --policy examples/flight-booking-agent-policy.json --tool search_flights --args "{}"
```

returns **`ALLOW`**. [`policy-hot-reload/`](policy-hot-reload/README.md) is the other
runnable lifecycle artifact: it walks the served-gateway operator loop — edit the
floor → `policy --check` → `POST /v1/fak/policy/reload` → re-test — and proves the
verdict swaps **in-process** (same `start_time_unix`, IFC ledger intact, no restart).

A `book_flight` whose `fare_amount` is `$10,000` or more is also
refused (`deny_regex ^[0-9]{5,}` — the manifest's argument matchers are
`allow_glob` / `deny_regex` / `max_bytes`, so the price cap is expressed as a regex
rather than a numeric `max_value`):

```bash
go run ./cmd/fak preflight --policy examples/flight-booking-agent-policy.json --tool book_flight --args '{"fare_amount": 50000}'
```

### `healthcare-phi-policy.json` — PHI redaction + provenance

The witness above returns **`DENY (SECRET_EXFIL)`** for `export_patient_data`. The
ALLOW side and the second deny:

```bash
go run ./cmd/fak preflight --policy examples/healthcare-phi-policy.json --tool read_patient_record --args "{}"
go run ./cmd/fak preflight --policy examples/healthcare-phi-policy.json --tool email_phi --args "{}"
```

return **`ALLOW`** and **`DENY (POLICY_BLOCK)`** respectively — every denied action is
meant to escalate to the `transfer_to_human_clinician` safe sink, not fail silently.
The `schedule_appointment` provider allow-list is an `allow_glob`, so an out-of-network
booking is refused while an in-network one clears:

```bash
go run ./cmd/fak preflight --policy examples/healthcare-phi-policy.json --tool schedule_appointment --args '{"provider":"random-clinic"}'
go run ./cmd/fak preflight --policy examples/healthcare-phi-policy.json --tool schedule_appointment --args '{"provider":"in-network-cardiology"}'
```

> **This is an example allow-list, not HIPAA compliance.** fak is a capability floor,
> not a certification: this manifest demonstrates the *posture* (heavy `redact_fields`,
> trusted-EHR vs untrusted-inbox provenance, fail-closed denies) but a real deployment
> must still own its BAAs, audit logging, encryption, and access controls. Note also
> that `redact_fields` matches arg keys **exactly** (no wildcard), so a PHI vocabulary
> is enumerated explicitly — `phi_notes`, `phi_address`, … — rather than as a `phi_*`
> glob.

### `sql-analyst-policy.json` — the `arg_rules` showcase

This is the heaviest `arg_rules` template: it constrains the **SQL text itself**, not
just the tool name. `run_read_query` clears the name-level allow-list, but four
argument predicates then narrow it — SELECT-only (DDL/DML keywords are denied), a body
size cap, a row cap, and a schema allow-list:

```bash
go run ./cmd/fak preflight --policy examples/sql-analyst-policy.json --tool run_read_query --args '{"sql":"SELECT 1","schema":"public.users","limit":50}'
go run ./cmd/fak preflight --policy examples/sql-analyst-policy.json --tool run_read_query --args '{"sql":"DROP TABLE customers"}'
go run ./cmd/fak preflight --policy examples/sql-analyst-policy.json --tool run_write_query --args '{}'
```

return **`ALLOW`**, **`DENY (POLICY_BLOCK)`** (the `run_read_query.sql` arg-rule fires
on `DROP` even though `run_read_query` itself is allowed), and **`DENY (POLICY_BLOCK)`**
(`run_write_query` is an explicit deny — the tool an `allow run_query` mistake would
have admitted). The row cap and schema allow-list also fire:

```bash
go run ./cmd/fak preflight --policy examples/sql-analyst-policy.json --tool run_read_query --args '{"sql":"SELECT 1","limit":100000}'
go run ./cmd/fak preflight --policy examples/sql-analyst-policy.json --tool run_read_query --args '{"sql":"SELECT 1","schema":"secrets.creds"}'
```

return **`DENY (OVERSIZE)`** and **`DENY (POLICY_BLOCK)`**.

> **The SQL `deny_regex` is a best-effort keyword guard, not a SQL parser.** It is the
> regex-shaped analogue of a structural check: like `adjudication-demo/README.md` is
> honest that `POLICY_BLOCK` regexes are softer than the structural `DEFAULT_DENY`
> floor, this manifest's SELECT-only rule can be defeated by an attacker who hides a
> keyword from the pattern (comments, casing tricks, `/**/` splitting, dialect quoting).
> The load-bearing floor here is the **structural** layer — `run_write_query`,
> `drop_table`, `alter_table`, `create_table`, `copy_to`, and `pg_dump` are denied as
> whole tools, so the dangerous operations have no admitted tool to ride on at all. The
> arg-rule is defense-in-depth on the one tool (`run_read_query`) that *must* take
> free-form SQL. Two of the issue's idealized rules are expressed in the matchers the
> engine actually has (`allow_glob` / `deny_regex` / `max_bytes`): the row cap is a
> `deny_regex ^[0-9]{6,}` on `limit` (denies a six-or-more-digit limit) rather than a
> numeric `max_value`, and the schema allow-list is a single `allow_glob public.*`
> rather than a comma list — `allow_glob` is one `path.Match` glob, so list one schema
> prefix per deployment.

The useful adoption pattern is: copy the closest template, delete what you do not
need, run `policy --check`, then run one or two `preflight` calls that prove the
most important dangerous actions are denied with closed reason codes.
