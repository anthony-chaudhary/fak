# TICKET-05: pi launcher and garden launchd changes landed without compiler or CLI runs

## Current state

Covered commits:

- Pi launcher/router: 51185bc3243 selected-window propagation; 0c191ba16bc shared router compaction across model windows; c2e64512ad9 retry-policy re-enabling in plans (extends 7e402a72eae); d3b9e84bd60 byte split in activated-ring refusals.
- Garden launchd: 5cc863a9108 argument escaping; a619dd77c0b rendering isolation; 8775571f56c bootstrap guidance quoting; 88c193eda11 unused declarations; 54919cf2b2b reject uncertain probes; ab88159c233 remove forced watchdog restarts; 174f94fb837 test env isolation.
- Test-only: c2de9c56fb9, b2220fbff39, a48f6b04deb, 64359e99166.

The module builds and vets green. On Windows the full `go test ./cmd/fak/` run hits its 20m timeout and has older, unrelated reds, so this group needs targeted runs.

## Working spine

1. Run targeted `go test -run '<Pi|Garden|Launchd|Retry|ActivatedRing>' ./cmd/fak/` for each group.
2. Run `fak pi config --from-router --write` into a temp home and check the retry and window output.
3. Render a garden launchd plist and check its escaping and quoting.
4. Add missing fail-before/pass-after tests.

## Witness

The targeted test runs exit 0, plus captured CLI output.

## Done condition

- [x] Both groups green under targeted tests and CLI execution. Every fix fails before and passes after its test; `fak pi config --from-router --write` (temp HOME) and the garden plist render were run end to end. New tests: 501095431bb (Pi), 8fcf7b7e995 (garden). `fak-sync release verify-queue list --root ../fak` shows all 11 non-test commits `verified full`, each noting what was not run (repo-wide suite, macOS launchd hardware). Follow-up: TICKET-05a (#13778).
- [x] The cmd/fak suite timeout filed separately if it is not already tracked: already tracked as #13441 (TestContractLifecycle passes alone in 22.29s; the cause is cumulative serial suite time).
