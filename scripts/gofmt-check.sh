#!/bin/sh
# scripts/gofmt-check.sh - the body of `make gofmt-check`, extracted so the gate is one
# independently runnable, testable artefact (#6490). The Makefile target is now a one-line
# delegation to this script, the way `build:` delegates to scripts/build.sh.
#
# WHAT IT CHECKS (unchanged): every .go file git can see - tracked plus untracked and not
# ignored (`git ls-files -co --exclude-standard`) - must be gofmt-clean. The scan stays
# WHOLE-TREE; only the report and the exit condition are scoped.
#
# WHY THE SPLIT: this checkout is routinely shared by several concurrent sessions, so at
# any moment the tree carries in-flight .go files nobody in the current change has
# touched. A whole-tree gofmt gate therefore reds on files the change under test never
# went near, and prints a flat list with no way to tell which is which - the reader has to
# diff it against their own pathspec by hand, and the usual answer is "none of these are
# mine". A gate that reds for reasons unrelated to the change trains people to read past
# it, which is how a real formatting break gets waved through.
#
# So: declare the change under test in GOFMT_OWNED_PATHS (or pass the paths as arguments)
# and the findings are reported in two labelled groups -
#
#   owned         the files the change under test owns -> FAILS the gate (exit 1)
#   pre-existing  everything else                      -> a visible, non-fatal notice
#
# That is the pairing `fak validate` already uses (gofmtOwnedPaths scopes the STYLE check
# to the change; whole-tree `go build ./...` / `go vet ./...` stay unscoped). The
# whole-tree compile gate is what makes scoping the style check safe: scoping can hide a
# stale format, never a real break.
#
# FALLBACK: with no scope declared the whole tree IS the owned set - byte-identical to the
# pre-split gate, so an unscoped `make ci` behaves exactly as it always did.
#
# Scope entries are repo-relative, forward-slash, and match a file exactly or as a
# directory prefix (`internal/foo` owns `internal/foo/bar.go`), matching `fak commit
# --path` semantics. Separate them with whitespace.
#
# LINUX/WSL ONLY, like the target it came from: a native-Windows checkout under
# core.autocrlf=true rewrites .go to CRLF, which `gofmt -l` flags as a false positive
# (.gitattributes pins only *.sh/*.golden to LF), so scripts/ci.ps1 deliberately omits
# this gate and leaves the canonical LF check to WSL `make ci` / CI.
#
# Usage:  sh scripts/gofmt-check.sh [owned-path ...]
#         GOFMT_OWNED_PATHS="internal/foo cmd/bar/baz.go" sh scripts/gofmt-check.sh
set -u

scope="${GOFMT_OWNED_PATHS-}"
if [ "$#" -gt 0 ]; then
	scope="$*"
fi

files="$(git ls-files -co --exclude-standard '*.go')"
unformatted=""
if [ -n "$files" ]; then
	unformatted="$(printf '%s\n' "$files" | xargs gofmt -l)"
fi

if [ -z "$unformatted" ]; then
	echo "gofmt: clean"
	exit 0
fi

# No declared scope: the whole tree is the owned set, and the report is the original
# gate's verbatim - nothing to attribute, so nothing new to say.
if [ -z "$scope" ]; then
	echo "gofmt: not formatted (run 'gofmt -w .' from the repo root):"
	echo "$unformatted"
	echo "gofmt: no change scope declared (set GOFMT_OWNED_PATHS to split owned findings from pre-existing tree debt)"
	exit 1
fi

owned=""
debt=""
# Unquoted expansion splits on whitespace, so a path containing a space would split -- the
# same limit the pre-existing `xargs gofmt -l` above already has, and this tree has no such
# path (see internal/hooks/tree.go on why the ls-files readers use -z).
for f in $unformatted; do
	hit=""
	for p in $scope; do
		p="${p%/}"
		case "$f" in
		"$p" | "$p"/*)
			hit=1
			break
			;;
		esac
	done
	if [ -n "$hit" ]; then
		owned="$owned$f
"
	else
		debt="$debt$f
"
	fi
done

rc=0
if [ -n "$owned" ]; then
	echo "gofmt: not formatted in the change under test (run 'gofmt -w' on these):"
	printf '%s' "$owned" | sed 's/^/  /'
	rc=1
fi
if [ -n "$debt" ]; then
	echo "gofmt: pre-existing tree debt outside the change under test ($(printf '%s' "$debt" | wc -l | tr -d ' ') file(s); reported, not failing this gate):"
	printf '%s' "$debt" | sed 's/^/  /'
fi
if [ "$rc" -eq 0 ]; then
	echo "gofmt: clean (change under test)"
fi
exit "$rc"
