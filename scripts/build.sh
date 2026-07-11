#!/bin/sh
# scripts/build.sh - the ONE canonical `fak` build recipe (issues #3709/#3710, epic #3708).
#
# Every shipping consumer routes through this script so the -trimpath / -ldflags /
# version-stamp flags live in a single place and cannot drift apart:
#   - Dockerfile                                (static distroless image)
#   - Dockerfile.cuda                           (CGO + -tags cuda GPU image)
#   - .github/workflows/release-artifacts.yml   (the 5-target release matrix)
#   - Makefile `release:` / `build:` / `build-race:`  (local parity with the shipped build)
# Anti-drift is witnessed by tools/build_entrypoint_test.py, which reds if the stamp
# recipe reappears inline in any consumer instead of routing through here.
#
# PROFILES (#3710) - one flag set per named profile, selected by $PROFILE. The single
# flag-delta table (profile x flag) lives in docs/dev-tooling.md; keep the two in sync.
#   release  -trimpath + strip (-s -w): the SHIPPED binary. Stripped and reproducible-ready
#            (CGO off by default via the caller's env), stamped with BuildVersion. THE default.
#   dev      no -trimpath, no strip: DWARF/symbols kept so Delve can set a breakpoint and
#            step; still stamps BuildVersion so `fak version` is honest. Fast, and its object
#            cache is shared with `go build ./...` / `go test`. `make build`.
#   race     dev + -race. The race detector REQUIRES cgo, so the CALLER must build with
#            CGO_ENABLED=1 (Makefile `build-race:` sets it and preflights a C compiler);
#            the result is instrumented and NOT the static pure-Go binary. Opt-in only.
#
# Inputs are environment variables; each caller sets only what it varies:
#   OUT      output binary path              (default: fak)
#   VERSION  version stamped into the binary (default: ./VERSION, else empty)
#   TAGS     go build tags, space-separated  (default: none; e.g. TAGS=cuda, TAGS=dev)
#   PROFILE  build profile                   (default: release; dev|race for #3710)
#   GCFLAGS  extra -gcflags (dev|race only)  (default: none; set 'all=-N -l' so Delve steps
#                                             pristinely by disabling inlining/optimization)
# GOOS, GOARCH, CGO_ENABLED, CGO_*, GOTOOLCHAIN pass through from the environment
# unchanged - build.sh is deliberately toolchain- and CGO-agnostic: the CUDA build
# sets CGO_ENABLED=1 in Dockerfile.cuda's ENV, the static builds set CGO_ENABLED=0,
# the race profile's caller sets CGO_ENABLED=1, and this script overrides none of them.
set -eu

OUT="${OUT:-fak}"
: "${VERSION:=$(cat VERSION 2>/dev/null || true)}"
TAGS="${TAGS:-}"
PROFILE="${PROFILE:-release}"
GCFLAGS="${GCFLAGS:-}"

# BuildVersion is stamped in EVERY profile so `fak version` never lies about what ran;
# only the strip/trim/DWARF posture differs between the shipped and the debuggable builds.
STAMP="-X github.com/anthony-chaudhary/fak/internal/appversion.BuildVersion=${VERSION}"

case "$PROFILE" in
  release)
    # Shipped: strip symbols + DWARF (-s -w) and trim host paths (-trimpath) for a
    # reproducible-ready static binary. The one recipe every consumer above inherits.
    set -- -trimpath -ldflags "-s -w ${STAMP}"
    ;;
  dev)
    # Debuggable: keep DWARF/symbols (no -s -w) and host paths (no -trimpath) so Delve
    # can set a breakpoint and step; still stamp BuildVersion.
    set -- -ldflags "${STAMP}"
    [ -n "$GCFLAGS" ] && set -- "$@" -gcflags "$GCFLAGS"
    ;;
  race)
    # Debuggable + the race detector. -race REQUIRES cgo (the caller sets CGO_ENABLED=1);
    # the result is instrumented and NOT the static pure-Go binary the other profiles emit.
    set -- -race -ldflags "${STAMP}"
    [ -n "$GCFLAGS" ] && set -- "$@" -gcflags "$GCFLAGS"
    ;;
  *)
    echo "build.sh: PROFILE='$PROFILE' is not a known profile (release|dev|race)" >&2
    exit 2
    ;;
esac

[ -n "$TAGS" ] && set -- "$@" -tags "$TAGS"
set -- "$@" -o "$OUT" ./cmd/fak

exec go build "$@"
