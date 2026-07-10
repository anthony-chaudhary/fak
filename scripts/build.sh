#!/bin/sh
# scripts/build.sh - the ONE canonical `fak` release build recipe (issue #3709, epic #3708).
#
# Every shipping consumer routes through this script so the -trimpath / -ldflags /
# version-stamp flags live in a single place and cannot drift apart:
#   - Dockerfile                                (static distroless image)
#   - Dockerfile.cuda                           (CGO + -tags cuda GPU image)
#   - .github/workflows/release-artifacts.yml   (the 5-target release matrix)
#   - Makefile `release:`                       (local parity with the shipped build)
# Anti-drift is witnessed by tools/build_entrypoint_test.py, which reds if the stamp
# recipe reappears inline in any consumer instead of routing through here.
#
# Inputs are environment variables; each caller sets only what it varies:
#   OUT      output binary path              (default: fak)
#   VERSION  version stamped into the binary (default: ./VERSION, else empty)
#   TAGS     go build tags, space-separated  (default: none; e.g. TAGS=cuda)
#   PROFILE  build profile                   (default: release; dev/race reserved for #3710)
# GOOS, GOARCH, CGO_ENABLED, CGO_*, GOTOOLCHAIN pass through from the environment
# unchanged - build.sh is deliberately toolchain- and CGO-agnostic: the CUDA build
# sets CGO_ENABLED=1 in Dockerfile.cuda's ENV, the static builds set CGO_ENABLED=0,
# and this script overrides neither.
set -eu

OUT="${OUT:-fak}"
: "${VERSION:=$(cat VERSION 2>/dev/null || true)}"
TAGS="${TAGS:-}"
PROFILE="${PROFILE:-release}"

if [ "$PROFILE" != "release" ]; then
  echo "build.sh: PROFILE='$PROFILE' is not implemented; only 'release' ships today" >&2
  echo "         (dev/race profiles are reserved for issue #3710)" >&2
  exit 2
fi

# The single source of truth for the release stamp + strip/trim flags. When this
# recipe needs to change it changes HERE, and every consumer inherits the change.
LDFLAGS="-s -w -X github.com/anthony-chaudhary/fak/internal/appversion.BuildVersion=${VERSION}"

set -- -trimpath
[ -n "$TAGS" ] && set -- "$@" -tags "$TAGS"
set -- "$@" -ldflags "$LDFLAGS" -o "$OUT" ./cmd/fak

exec go build "$@"
