#!/usr/bin/env bash
# fak/test.sh — the canonical way to run this module's Go tests on this host.
#
# WHY THIS EXISTS
# ---------------
# Run natively on Windows, `go test ./...` is unreliable here: a Windows
# Application Control policy (Smart App Control / WDAC) intermittently refuses to
# fork/exec the freshly compiled, *unsigned* per-package test binaries that the
# Go toolchain drops into %TEMP%. The failure looks like:
#
#     fork/exec C:\...\Temp\go-build.../<pkg>.test.exe:
#         An Application Control policy has blocked this file.
#     FAIL  github.com/anthony-chaudhary/fak/internal/<pkg>
#
# It hits a different package each run, so a green run is luck, not a pass. The
# policy can't be changed without admin on this box.
#
# Linux ELF test binaries built and run inside WSL are not subject to Windows
# Application Control, and WSL's default GOCACHE/tmp already live on the
# WSL-native ext4 filesystem (off NTFS, out of %TEMP%). So we run the suite
# there. This is the project default — see fak/test.ps1 for the Windows wrapper.
#
# FILESYSTEM PERFORMANCE (the part that's easy to get wrong)
# ----------------------------------------------------------
# WSL2 exposes two filesystems with wildly different performance:
#   * ext4   — the distro's own virtual disk ($HOME, GOCACHE, /tmp): native
#              Linux speed.
#   * /mnt/c — Windows NTFS reached over the 9p protocol: every file op is a
#              round-trip across the VM boundary, so stat/open-heavy work is
#              ~100x+ slower. (\\wsl$\ from Windows is the same 9p tax in reverse.)
#
# This repo lives on /mnt/c (9p). Measured on this host (best-of-3, warm cache):
#       operation                       /mnt/c (9p)     ext4
#       read all *.go (cat, 176 files)      1.77s       0.01s    (~177x)
#       go list ./...                      28.43s       0.09s    (~300x)
# i.e. Go pays a ~28s "enumerate + parse the source" tax on every invocation,
# *before any test runs*, purely because the source sits on 9p.
#
# What we already do right (and why this script exists): GOCACHE/GOTMPDIR stay on
# ext4 (see below), so compiled objects and the temp test binaries never touch
# 9p — the big, safe win. Note: 9p `msize`/`cache=` mount tuning does NOT fix the
# per-op stat latency above; only ext4-resident *source* does.
#
# To also get the source onto ext4 (the remaining ~100x lever), this script has
# the FAK_FAST mirror below. The Windows wrapper (`test.ps1`) enables it by
# default; call WSL directly with `FAK_FAST=1 bash ./test.sh ...` or set
# `FAK_FAST=0` when you intentionally need the slower /mnt/c checkout path.
#
# USAGE
# -----
#   From WSL:      bash ./test.sh [go-test-args...]   # default target: ./...
#   From Windows:  wsl bash ./fak/test.sh [args...]   # or: .\fak\test.ps1
#
# Examples:
#   ./test.sh                          # whole suite
#   ./test.sh ./internal/ctxmmu/       # one package
#   ./test.sh -run TestEvict ./internal/model/
#   ./test.sh -count=1 ./...           # force a clean (uncached) run
set -euo pipefail

# Module dir = the dir this script lives in. Works regardless of caller CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Non-login `wsl bash test.sh` does not source the profile that puts the Go install
# on PATH, so on a distro where Go lives at the standard /usr/local/go/bin it is not
# found. Prepend the usual locations if `go` is not already resolvable, so test.ps1
# works regardless of which WSL distro (FAK_WSL_DISTRO) is in use.
if ! command -v go >/dev/null 2>&1; then
  export PATH="/usr/local/go/bin:${HOME}/go/bin:${PATH}"
fi

# fak/go.mod requires `go 1.26`; the distro Go may be older. GOTOOLCHAIN=auto
# lets the toolchain fetch the required version into user space (no sudo). The
# first run downloads it (~once); later runs reuse the cached toolchain.
export GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"

# GOCACHE / GOTMPDIR are deliberately left at their WSL defaults
# (~/.cache/go-build and $TMPDIR|/tmp) — both already on ext4, which is the
# whole point. Override them in the environment if you need to.

args=("$@")
if [ "${#args[@]}" -eq 0 ]; then
  args=("./...")
fi

# --- ext4 fast path (FAK_FAST=1; default from test.ps1 on Windows) ------------
# The source tree lives on /mnt/c (9p), which taxes every run with a ~28s
# enumerate+parse phase before a test runs (see the FILESYSTEM PERFORMANCE note
# above). With FAK_FAST=1 we mirror the module to an ext4 scratch tree and run
# `go test` there, so that stat/parse-heavy phase runs at native speed. rsync is
# incremental, so only the *first* run (and changed files after) pays the 9p
# read; repeated inner-loop runs are fast. The ~0.5G generated weight cache
# (internal/model/.cache) is symlinked back to the original rather than copied —
# those reads are throughput-bound (fine on 9p) and the cache stays single-sourced.
#
# Bypassed automatically when the run must WRITE back into the real tree — golden
# regeneration (UPDATE_GOLDEN) or relative-path output (-coverprofile, -o, …) —
# so those still land on /mnt/c. Override the scratch location with FAK_FAST_DIR.
if [ "${FAK_FAST:-}" = "1" ]; then
  bypass=""
  [ -n "${UPDATE_GOLDEN:-}" ] && bypass="UPDATE_GOLDEN set"
  for a in "${args[@]}"; do
    case "$a" in
      -update|-coverprofile=*|-cpuprofile=*|-memprofile=*|-blockprofile=*|-trace=*|-o|-o=*|-c)
        bypass="${bypass:-writes relative output ($a)}";;
    esac
  done
  if [ -n "$bypass" ]; then
    echo "fak/test.sh: FAK_FAST set but bypassed ($bypass) — running on /mnt/c so output lands in the real tree"
  else
    SCRATCH="${FAK_FAST_DIR:-$HOME/.cache/fak-src}"
    CACHE_REL="internal/model/.cache"   # generated weights — symlink, never copy
    mkdir -p "$SCRATCH"
    if command -v rsync >/dev/null 2>&1; then
      # plain --delete (NOT --delete-excluded) so excluded runtime state and the
      # symlinked cache are preserved. The live dogfood fleet mutates these dirs
      # while tests start; copying them can make rsync fail before `go test` runs.
      rsync_args=(
        -a --delete
        --exclude="/$CACHE_REL"
        --exclude="/.git/*.lock"
        --exclude="/.git/**/*.lock"
        --exclude="/.codex-tmp"
        --exclude="/.dispatch-runs"
        --exclude="/.dos/metrics"
        --exclude="/.dos/runs"
        --exclude="/.dos/streams"
        --exclude="/.fak"
        "$SCRIPT_DIR/" "$SCRATCH/"
      )
      rsync_rc=0
      for rsync_attempt in 1 2 3; do
        set +e
        rsync "${rsync_args[@]}"
        rsync_rc=$?
        set -e
        if [ "$rsync_rc" -eq 0 ]; then
          break
        fi
        if [ "$rsync_rc" -eq 23 ] && [ "$rsync_attempt" -lt 3 ]; then
          echo "fak/test.sh: rsync saw concurrent source mutation (exit 23); retrying mirror ($rsync_attempt/3)"
          sleep 0.2
          continue
        fi
        exit "$rsync_rc"
      done
      if [ -d "$SCRATCH/.git" ]; then
        find "$SCRATCH/.git" -type f -name '*.lock' -delete
      fi
    else
      # tar fallback (no pruning of stale files, but correct): copy all but the cache.
      ( cd "$SCRIPT_DIR" && find . -path "./$CACHE_REL" -prune -o -type f -print ) \
        | tar -C "$SCRIPT_DIR" -cf - -T - | tar -C "$SCRATCH" -xf -
    fi
    if [ -e "$SCRIPT_DIR/$CACHE_REL" ] && [ ! -e "$SCRATCH/$CACHE_REL" ]; then
      mkdir -p "$(dirname "$SCRATCH/$CACHE_REL")"
      ln -s "$SCRIPT_DIR/$CACHE_REL" "$SCRATCH/$CACHE_REL"
    fi
    echo "fak/test.sh: FAK_FAST=1 -> ext4 scratch $SCRATCH (source mirrored off 9p; weight cache symlinked)"
    cd "$SCRATCH"
  fi
fi

echo "fak/test.sh: distro go=$(go version | awk '{print $3}'), GOTOOLCHAIN=$GOTOOLCHAIN, target=${args[*]}"
exec go test "${args[@]}"
