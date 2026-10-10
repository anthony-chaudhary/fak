# TICKET-04: waived gateway change reverts deadline cache credit for long Halo chats

## Current state

e1850846f91 made deadline admission count only the uncached prompt, so multi-turn Halo pi sessions past 50k context were no longer refused with deadline_infeasible.

Waived commit 1dbc0638aa0 ("reject unproven deadline cache credit") removes about 129 lines of that logic in `internal/gateway/deadline_admission.go` and treats chat preflight residency as unknown. Its own body states that warm long chats may now receive conservative cold refusals.

`go test ./internal/gateway/` is green, so the tests do not catch this behavioral regression.

Also in this group: d2ab953eaf1 malformed schema carriers; 8cb934c8203 schema resource caps; 296ebc3340f planner wire domain.

## Working spine

1. Reproduce on the live Halo route whether a multi-turn session past 50k context is refused.
2. Grant cache credit only on route-bound, eviction-safe residency evidence. Neither unproven credit nor no credit at all is the answer.
3. Add tests for both the warm-long-chat case and the unproven-credit case.
4. Add fail-before/pass-after checks for the schema commits.

## Witness

`go test -count=1 ./internal/gateway/` passes with both new cases, plus one live route trace.

## Done condition

- [ ] A warm long chat is admitted when residency is proven, and unproven credit is refused; both cases are tested.
