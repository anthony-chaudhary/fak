# Documentation cohort dispatch

**Primary audience:** documentation dispatcher  
**Lifecycle:** current  
**Generation:** current trunk process  
**Authority:** [documentation audience architecture](../project/DOCUMENTATION-AUDIENCE-ARCHITECTURE-2026-07-15.md)  
**Last verified:** `fak issue contract`, `fak issue cohort`, and `dos arbitrate` CLI contracts on 2026-07-15

Use this playbook to turn documentation findings into disjoint, witnessed work before page edits begin. The default is a small cohort whose issues name exact owned paths, direct navigation consumers, and a read-back witness. This shifts discovery of ambiguous scope, stale routes, and file collisions to contract time instead of merge time.

## Choose the dispatch mode

| Situation | Mode | Default? | Result |
|---|---|---:|---|
| One route or tightly coupled pages | Single issue | **Yes** | One worker owns the route and its direct navigation consumers. |
| Several routes with disjoint files | Priced cohort | No | Independent workers run concurrently in collision-free waves. |
| Shared front door, generated index, or taxonomy redesign | Serial architecture pass | No | One owner reconciles the shared surface before page workers resume. |

Do not split files that must tell one coherent story merely to increase concurrency. A route includes its page, direct inbound navigation entries, and any authority link that the edit changes.

## Shift-left contract

Before a documentation issue is dispatchable, record:

- **Reader job:** one primary audience and the task they need to complete.
- **Owned route:** exact files the worker may edit, including direct navigation consumers.
- **Lifecycle and generation:** `current`, `versioned`, `research`, or `archived`, plus release, backend, or mode context when behavior varies.
- **Bounded change:** the smallest complete route improvement, not an open-ended request to “improve docs.”
- **Done condition:** observable statements about the resulting route.
- **Witness:** changed-link validation plus an independent reader restating the page's job, applicable choice, and next action. Front-door work also captures a render or structured first-screen read-back.
- **Residual policy:** findings outside the owned route become deduplicated issues before the worker stops.

Use this body skeleton when creating or repairing a contract:

```markdown
## Reader job
<primary audience> needs to <complete one task>.

## Owned route
- `path/to/page.md`
- `path/to/direct-navigation-consumer.md`

## Lifecycle and generation
<current|versioned|research|archived>; <release/backend/mode or "generation-independent">

## Bounded change
<one complete route improvement>

## Done condition
- <observable reader outcome>
- Direct navigation and authority links are reconciled.

## Witness
Validate changed local links; capture the front-door/read-back artifact; have an
independent reader state the page job, applicable choice, and next action.
```

Run strict contract review before assigning work:

```powershell
# Export the selected open issues to an OS-temp artifact.
$issues = Join-Path $env:TEMP "fak-doc-cohort-issues.json"
gh issue list --state open --limit 200 --json number,title,body,labels | Set-Content -Encoding utf8 $issues

# A hold is useful feedback: repair the issue contract before launch.
fak issue contract --from-issues $issues --strict-project-work --strict-witness --strict-scale --strict-born-routed --json
```

## Price the cohort before launch

Generate waves from the same reviewed issue set:

```powershell
$plan = Join-Path $env:TEMP "fak-doc-cohort-plan.json"
fak issue cohort --from-issues $issues --strict-project-work --max-wave 8 --json | Set-Content -Encoding utf8 $plan
Get-Content $plan
```

Inspect every proposed wave. Two issues are concurrent only when their complete owned routes are disjoint. Treat these as shared serialization points unless the contracts prove otherwise:

- `README.md`, `START-HERE.md`, `llms.txt`, and `INDEX.md`;
- generated indexes and their generators;
- a common integration landing page;
- the same authority or taxonomy page.

For each worker, ask the repository arbiter before editing and then journal the granted lease so later workers can see it:

```powershell
# Decision-only collision check.
dos arbitrate --lane docs --kind keyword --mode exclusive --tree docs/example.md llms.txt --output json --explain

# Hold the admitted route for the worker. Use a unique issue/run owner.
dos lease-lane acquire --lane docs --kind keyword --mode exclusive --tree docs/example.md llms.txt --owner docs-issue-1234 --pretty
```

A refusal changes the wave: serialize the colliding routes or rewrite their ownership. It is not permission to omit the shared consumer.

## Worker packet

Give each worker only the issue contract plus these invariants:

1. Read the owned route and authority page before editing.
2. Preserve factual claims, provenance, scope, and supported-mode boundaries.
3. Lead with affirmative current behavior and make the default choice explicit.
4. Update direct navigation consumers in the same change.
5. Capture the issue's witness; worker narration is not completion evidence.
6. Commit one issue as one signed, path-scoped commit on `main`.
7. File deduplicated residual findings instead of leaving prose follow-ups.
8. Release the lane lease after the landed effect is independently read back.

## Reconcile the wave

The dispatcher accepts trunk effects, not worker status strings:

1. Read each landed diff and confirm its committed paths equal the contracted route.
2. Re-run changed local links and any page-specific commands from the issue.
3. Read the rendered or structured artifact and independently state:
   - the page's primary job;
   - the applicable mode or default choice;
   - the checkable next action.
4. Confirm direct navigation and authority links agree with the changed page.
5. Verify residual findings are either deduplicated issues or explicitly absent.
6. Release the worker lease and update the cohort ledger or parent issue with the witnessed commit.

```powershell
# Release the held route after witnessing the landed effect.
dos lease-lane release --lane docs --owner docs-issue-1234
```

Use the exact lane and owner returned at acquisition. If an owner has multiple leases on one lane, also pass the recorded `--loop-ts`; the lease journal is the authority for active ownership.

## Completion check

A cohort is complete when every issue has one witnessed trunk effect, every shared navigation consumer has been reconciled once, all leases are released, and no discovered follow-up exists only in a handoff. The next checkable step is to run strict contract review on the candidate issue set before assigning the first worker.

