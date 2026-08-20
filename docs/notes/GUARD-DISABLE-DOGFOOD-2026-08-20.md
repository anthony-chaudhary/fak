# Scoped guard-disable dogfood — 2026-08-20

Issue #8203 asked for one real repository run of the child-scoped break-glass launcher added by #8197. The run passed: the repair child operated on the live repository, inherited none of the outer guard's routing or identity state, returned its status, and left the parent environment unchanged.

## Run

The executable was built from committed trunk `695c1aa772d5` to avoid admitting unrelated shared-tree WIP. That tip contains the launcher spine at `cmd/fak r3101+gfbe86e17ac`; the run itself used `cmd/fak r3102+g695c1aa7`.

The operator invoked the new front door with a bounded PowerShell child:

```powershell
fak guard disable --reason "repair guarded launcher" -- pwsh -NoProfile -NonInteractive -File probe.ps1
```

The probe read the live #8203 issue and repository HEAD. The parent supplied synthetic guard marker, loopback URL, placeholder credential, and audit identity values so the child could prove their absence without publishing real configuration.

The child emitted this public-safe receipt:

```json
{"schema":"fak-guard-disable-dogfood/1","child":"powershell","repo_head":"695c1aa772d584276e965529635e62914c40c866","issue_number":8203,"issue_state":"OPEN","guard_active_present":false,"loopback_base_present":false,"placeholder_key_present":false,"audit_identity_present":false,"raw_recovery":"break-glass","direct_continue":"1"}
```

The launcher then printed its terminal boundary:

```text
fak guard disable: BREAK-GLASS raw session ended (exit 0); later launches remain guarded by default.
```

After the child exited, all four synthetic parent values were still present and unchanged. This witnesses child scope in both directions: inherited guard routing did not enter the raw child, and the child did not persist a disabled posture into its parent or a later launch.

## Findings

No product defect surfaced in the current-trunk launcher, so this run filed no defect follow-up. As a control, the older installed executable identified itself as build `a0ded6bda0e3` and reproduced the pre-fix `"disable" is not on your PATH` behavior; it predates #8197 and is not evidence against the shipped trunk implementation.
