---
title: "Shipped `fak harness web` launch witness — 2026-08-15"
description: "Documentation for Shipped `fak harness web` launch witness — 2026-08-15, including the captured behavior, operating context, and reproducible fak evidence."
---

# Shipped `fak harness web` launch witness — 2026-08-15

Issue: #6980.

## Verdict

The browser runtime is shared by the compatibility demo and the shipped `fak` command. A temp-built `fak` binary completed the offline protocol/render selfcheck and its HTTP launch is covered by `TestHarnessWebTempBuiltFakBinary`; no in-tree binary was written.

## Captured command

```powershell
$binary = Join-Path $env:TEMP ("fak-6980-" + [guid]::NewGuid().ToString("N") + ".exe")
go build -o $binary ./cmd/fak
& $binary harness web --selfcheck
```

Captured output:

```text
HARNESS_WEB_SELFCHECK ok protocol=fak.harness.run/v1 normal=8 resumed=2 approval=4 failure=3 skins=2 html_sha256=ff709b8bd6fb1eff1ef52a6e48be97e9ae5722244fc84b0cee2387bd8386a77d
```

## Independent checks

```text
go test ./internal/harnessweb
ok github.com/anthony-chaudhary/fak/internal/harnessweb

go test ./cmd/fak -run TestHarnessWeb -count=1
ok github.com/anthony-chaudhary/fak/cmd/fak

go test ./cmd/harnesswebdemo -count=1
? github.com/anthony-chaudhary/fak/cmd/harnesswebdemo [no test files]

go vet ./internal/harnessweb ./cmd/harnesswebdemo
(exit 0)
```

`TestHarnessWebTempBuiltFakBinary` builds to `t.TempDir()`, runs the command-level selfcheck, starts `fak harness web` on an ephemeral loopback port, and requires an HTTP 200 response. `TestHarnessWebRejectsNonLoopbackBind` preserves the remote-bind refusal. Existing package tests preserve persistence, exclusive resume cursors, approval semantics, typed failures, workspace identity without root disclosure, and native gateway capability projection.

## Launch and rollback

Launch with `fak harness web`; add `--fak-url` and `--workspace` only for an explicitly armed native gateway. Roll back by stopping that loopback process; source-checkout users can continue to use the thin `go run ./cmd/harnesswebdemo` wrapper.
