package main

import "strings"

// guardBuildStampUnattested reports whether a guard build-stamp string describes a binary that
// carries NO usable VCS provenance — either "no VCS stamp" at all or "no embedded build info".
// Such a binary literally cannot attest which commit it was built from, so its staleness is
// UNVERIFIABLE: it is the exact "is an old fak guard STILL running?" blind spot (#3306). A
// stamped commit ("abc123 +uncommitted (committed …)") or a released module build ("module vX")
// is attested and returns false.
//
// The decision reads the SAME buildStamp string the banner prints on its `build` row (from
// guardBannerBuildStamp), so the warning is always consistent with what is displayed and the
// check stays a pure function of its input — no git call, no dependence on how a test binary
// happened to be built.
func guardBuildStampUnattested(buildStamp string) bool {
	s := strings.ToLower(buildStamp)
	return strings.Contains(s, "no vcs stamp") || strings.Contains(s, "no embedded build info")
}

// guardUnattestedBuildWarning is the one-line WARN the guard banner prints under its `build`
// row when the running binary cannot attest its commit. It names the defect (staleness is
// UNVERIFIABLE — a stale guard would look identical to a current one) and the two durable
// fixes: a plain in-repo `go build ./cmd/fak` re-stamps the binary, and `fak self-update
// --force` installs a gated, stamped origin/main build over this one. Returns "" when the
// binary IS attested, so callers can emit it unconditionally.
func guardUnattestedBuildWarning(buildStamp string) string {
	if !guardBuildStampUnattested(buildStamp) {
		return ""
	}
	return "  build WARN : no VCS stamp — this guard cannot confirm which commit it is running, so staleness is " +
		"UNVERIFIABLE (a stale guard looks identical). Rebuild in-repo with `go build ./cmd/fak`, or run " +
		"`fak self-update --force` to install a stamped origin/main build.\n"
}

// guardInfoStalenessNote is the info PANE's persistent twin of guardUnattestedBuildWarning. The
// banner prints its `build WARN` row once at startup and it scrolls off, but `fak info` stays on
// screen for the whole session — so the pane header is where an operator will actually SEE that
// the running fak cannot attest its commit. It reuses guardBuildStampUnattested over the SAME
// build-stamp string the banner and version tag read (guardBannerBuildStamp), so pane, banner,
// and the header's version tag never disagree about attestation. Returns "" when the build IS
// attested — the "+"-marked build id in guardInfoVersionTag is then the visible staleness tell,
// and the pane stays uncluttered — so the header can emit this line unconditionally.
func guardInfoStalenessNote(buildStamp string) string {
	if !guardBuildStampUnattested(buildStamp) {
		return ""
	}
	return "stale-build WARN: cannot confirm which commit fak is running (staleness UNVERIFIABLE) — " +
		"rebuild `go build ./cmd/fak`, or `fak self-update --force`, then relaunch"
}
