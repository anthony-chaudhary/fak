# Public hygiene: scrub scope and deferrals (2026-11-26)

Scope of the scrub in this change:

- Personal Windows home path `C:\Users\USER` (and `C:/Users/USER`, `/Users/USER`) replaced by `%USERPROFILE%` in docs and by `C:\Users\example` in Go and Python test fixtures.
- Tool defaults that embedded the path now derive it at run time (`internal/framevisibility` `-homes` uses the parent of `os.UserHomeDir()`; `tools/install_self_update_schedule.ps1` uses `$env:USERPROFILE`; `tools/popularization_dispatch_loop.sh` uses `${TMPDIR:-/tmp}`).
- Private LAN addresses `192.168.1.x` replaced by `<lan-ip>` in tracked docs and goal files.

Deliberately not changed:

- Detector rule tables and their tests (`tools/scrub_public_copy.py`, `tools/check_secret_shapes.py`, `internal/hooks/gate_publicleak.go`, `internal/hooks/hooks_test.go`, `internal/hooks/scan_outbound_test.go`). These match the placeholder username on purpose; rewriting them would blind the detectors.
- Witness receipts and ledgers (`docs/_witnesses/`, `docs/nightrun/`, `experiments/`, `docs/questions/`). They are recorded evidence; editing them falsifies the record.
- `internal/adjudicator/outoftree_test.go`: hard-self core-locked path. Changing it needs an independent maintenance witness, so it is deferred to a witnessed follow-up.
- Namespace fixtures such as `-Users-USER-Documents-GitHub-fleet` in `internal/sessionaudit` and `tools/fleet_bottleneck_test.py`. They encode the Claude project directory scheme, not a host path.
- Go test fixtures with `192.168.1.x` (`cmd/fak`, `internal/egressfloor`, `internal/gateway`, `pkg/security`, and similar). They exercise LAN-address classification and need real private addresses.

Deferred for size (too large for one night):

- About 218 internal scaffolding files at the repository root and under `internal/`. Moving them needs a planned rename with import-path and architest updates.
- Large corpora (`docs/nightrun`, `experiments/`, `docs/_witnesses/`). Split into a separate archive lane, not mass-moved.
- Root clutter files left by earlier runs. Clean up under a fleet-owned ticket.
