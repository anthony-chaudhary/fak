# ci.ps1 — the `make ci` equivalent on Windows (unit 12): build + vet + test.
# Exit non-zero on any failure so it is a single mechanical witness.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "== go build =="
go build ./...
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "== go vet =="
go vet ./...
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "== go test =="
go test ./...
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "== claims lint =="
& (Join-Path $PSScriptRoot "claims-lint.ps1")
if ($LASTEXITCODE -ne 0) { exit 1 }

# index-sync (#511): the curated INDEX.md / llms.txt must not drift from the tree.
# Resolve a python interpreter once (Windows ships `python`/`py`, not `python3`).
$py = if (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" }
      elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" }
      else { $null }
if ($null -ne $py) {
    Set-Location (Join-Path $PSScriptRoot "..")

    # salience (dos-kernel docs/391): route CLAIMS.md through the `dos salience` verdict —
    # the first wired consumer of that verb (was built-but-latent). Asserts the no-loss
    # invariant + cross-checks live/parked counts against the ledger. Real gate where the
    # dos kernel is importable; an advisory SKIP (exit 0) where it is not.
    Write-Host "== salience (dos salience consumer) =="
    & $py tools/claims_salience_register.py --check
    if ($LASTEXITCODE -ne 0) { exit 1 }

    Write-Host "== index-sync =="
    & $py tools/check_index_sync.py --audit-tree
    if ($LASTEXITCODE -ne 0) { exit 1 }
    & $py tools/gen_llms_full.py --check
    if ($LASTEXITCODE -ne 0) { exit 1 }

    # repo-hygiene gates: the deterministic, no-network checks ci.yml runs HARD
    # (doc placement, links, hosted-demo URLs, demo-command refs, browser-demo metadata,
    # file admission, secret shapes), mirrored here so a local `scripts/ci.ps1` fails on
    # the same things GH does. Pure-stdlib python, fast, audits the tracked tree. (gofmt is NOT mirrored: a
    # native-Windows checkout under core.autocrlf=true is CRLF, so `gofmt -l`
    # false-positives — the Makefile/WSL `make ci` and CI run the canonical LF gofmt gate.)
    Write-Host "== repo-hygiene gates =="
    foreach ($chk in @("check_doc_placement", "check_links", "check_committed_files", "check_secret_shapes", "check_brand_consistency")) {
        & $py "tools/$chk.py" --audit-tree
        if ($LASTEXITCODE -ne 0) { exit 1 }
    }
    & $py "tools/demo_live_links.py"
    if ($LASTEXITCODE -ne 0) { exit 1 }
    & $py "tools/demo_command_audit.py"
    if ($LASTEXITCODE -ne 0) { exit 1 }
    & $py "tools/demo_browser_contract.py"
    if ($LASTEXITCODE -ne 0) { exit 1 }
    # hardware-name lint: no prose A100/DGX/SXM4 tell in the doc set (--check, not
    # --audit-tree: the scrubber reads its own default doc set off the tracked *.md).
    & $py "tools/scrub_hardware_names.py" --check
    if ($LASTEXITCODE -ne 0) { exit 1 }

    # Demo audit tool unit tests: the static/live-link/command/browser/smoke harnesses
    # are load-bearing demo quality gates, so their own tests run before the scorecards
    # and dynamic witnesses. Scorecard tests remain in the scorecard block below.
    Write-Host "== demo audit tool tests =="
    foreach ($test in @(
        "demo_registry_test",
        "demo_live_links_test",
        "demo_command_audit_test",
        "demo_browser_contract_test",
        "demo_http_smoke_test",
        "demo_headless_smoke_test"
    )) {
        & $py "tools/$test.py"
        if ($LASTEXITCODE -ne 0) { exit 1 }
    }

    # Content-only demo quality gates: no model, no network, no build. These ratchet
    # demo-debt and robustness-debt to zero locally the same way ci.yml does.
    Write-Host "== demo scorecards =="
    & $py "tools/demo_quality_scorecard_test.py"
    if ($LASTEXITCODE -ne 0) { exit 1 }
    & $py "tools/demo_quality_scorecard.py"
    if ($LASTEXITCODE -ne 0) { exit 1 }
    & $py "tools/demo_quality_scorecard.py" --check-doc
    if ($LASTEXITCODE -ne 0) { exit 1 }
    & $py "tools/demo_robustness_scorecard_test.py"
    if ($LASTEXITCODE -ne 0) { exit 1 }
    & $py "tools/demo_robustness_scorecard.py"
    if ($LASTEXITCODE -ne 0) { exit 1 }
    & $py "tools/demo_robustness_scorecard.py" --check-doc
    if ($LASTEXITCODE -ne 0) { exit 1 }

    # Dynamic but hermetic browser-demo smoke: build each browser demo into a temp dir,
    # serve it on loopback under its documented base path, and fetch page + JSON API.
    Write-Host "== demo HTTP smoke =="
    & $py "tools/demo_http_smoke.py" --timeout 60
    if ($LASTEXITCODE -ne 0) { exit 1 }

    # Dynamic but hermetic headless-demo smoke: run the model-free terminal witnesses
    # documented in run-the-demos.md and check their pinned invariant output.
    Write-Host "== demo headless smoke =="
    & $py "tools/demo_headless_smoke.py" --timeout 120
    if ($LASTEXITCODE -ne 0) { exit 1 }

    # tool-test no-blackhole gate: Windows runs the runner's own unit tests + --check
    # (pure-fs manifest consistency) ONLY. The hermetic --run is Linux-authoritative — a
    # few tests are platform-divergent (e.g. repo_guard's OUT_OF_TREE_WRITE deny, the
    # check_index_sync hang on win32), so the full run lives in ci.yml + WSL `make ci`.
    Write-Host "== tool-test no-blackhole gate (--check) =="
    & $py "tools/gated_tool_tests_test.py"
    if ($LASTEXITCODE -ne 0) { exit 1 }
    & $py "tools/gated_tool_tests.py" --check
    if ($LASTEXITCODE -ne 0) { exit 1 }

    # cuda-check: the GPU-free CUDA ABI/header-portability gate. Pure text, no nvcc / no GPU /
    # no cgo, so it runs on the canonical Windows dev host (CGO_ENABLED=0, no CUDA toolkit) —
    # the local mirror of cuda-build.yml's `static` job and `make ci`'s cuda-check. See
    # docs/cuda-dev.md.
    Write-Host "== cuda ABI parity + header portability =="
    & $py "tools/cuda_abi_parity.py" --check
    if ($LASTEXITCODE -ne 0) { exit 1 }
} else {
    Write-Host "== index-sync + repo-hygiene (warn): python not found; gates skipped =="
}

Write-Host "CI OK"
exit 0
