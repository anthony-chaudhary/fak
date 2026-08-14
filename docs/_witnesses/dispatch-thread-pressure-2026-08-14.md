# Host-thread admission policy witness — 2026-08-14

## Value frame

- **Problem centrality:** Enabling.
- **For:** operators launching independent guarded workers from a shared Windows control host.
- **Problem:** absolute system thread inventory can collapse worker capacity even when those threads are unrelated and mostly idle.
- **Today:** `cores * threads_per_core - total_system_threads` is a hard min; a normal desktop baseline crossed it and reduced a 32-core / high-free-RAM host to one worker.
- **Better because:** the policy separates visible inventory from a marginal baseline delta or worker-attributed pressure. Inventory alone abstains; measured pressure may only contract an existing structural cap.
- **Witness:** focused policy tests plus a captured live preflight before/after host calibration.

## All-work checks

- **P1 managed context:** one typed result carries signal provenance (`inventory`, `baseline_delta`, `worker_attributed`) instead of relying on an undocumented environment tweak.
- **P2 net-true efficiency:** preserves CPU/RAM/configured/seat ceilings while removing a false one-worker cliff; it does not claim more capacity than those ceilings grant.
- **P3 bounded adaptation:** pressure can only contract an existing cap and cannot raise it; invalid/unmeasured input abstains.
- **P4 integrated operations:** the additive policy is shaped for the existing dispatch preflight result and limiter JSON; live integration remains sequenced behind the peer-owned WIP change and #6779's attributed-footprint wiring.

## Captured reproduction

Default preflight on 2026-08-13:

```text
cores=32
free_ram_mb=159337
total_threads=13933
components={cores:16, ram:106, threads:0}
host_cap=1
binding=threads
```

Temporary calibrated preflight on 2026-08-14 with `FAK_HOST_THREADS_PER_CORE=600` and `--max-workers 4`:

```text
verdict=SPAWN_OK
cap=4
headroom=4
components={cores:16, ram:113, threads:35}
host_cap=16
binding=cores
capacity_limiter=configured_max
```

The user-scoped calibration is reversible and is not the product fix. Issues #6778, #6779, and #6780 record the permanent host policy, live attributed-footprint prerequisite, and effect-class/weighted-maintenance follow-on.

## Code witness

```text
go test ./internal/dispatchtick -run '^TestThreadPressure' -count=1
ok github.com/anthony-chaudhary/fak/internal/dispatchtick

go vet ./internal/dispatchtick
go test ./internal/dispatchtick -run '^TestThreadPressure|^TestHostCapacity' -count=1
ok github.com/anthony-chaudhary/fak/internal/dispatchtick
```

A literal whole-tree build was also attempted and failed exclusively on peer WIP duplicate `serveGGUF*` declarations in `cmd/fak/serve.go` and `cmd/fak/serve_ggufplan.go`; the path-scoped package build/vet above is green.
