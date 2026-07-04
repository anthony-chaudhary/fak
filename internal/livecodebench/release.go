package livecodebench

import (
	"fmt"
	"strconv"
	"strings"
)

// lcbReleases mirrors lcb_runner's --release_version values, ordered
// oldest-to-newest. The last entry is what ReleaseLatest resolves to.
var lcbReleases = []string{
	"release_v1", "release_v2", "release_v3",
	"release_v4", "release_v5", "release_v6",
}

// ReleaseLatest is upstream's alias for the newest release. It is always
// resolved to a concrete release and recorded explicitly, never left implicit.
const ReleaseLatest = "release_latest"

// ReleaseSelection is the resolved outcome of a --release-version selector: the
// selector echoed back, the concrete releases it expands to (a single release or
// an inclusive range), and the newest release the selection covers. Releases is
// always non-empty and ordered oldest-to-newest.
type ReleaseSelection struct {
	Selector string   `json:"selector"`
	Releases []string `json:"releases"`
	Resolved string   `json:"resolved"`
}

// ReleaseHeader pins a resolved release alongside the number of problems scored,
// so a report reader knows exactly which dataset version(s) a result covers and
// over how many problems — never an implicit "latest" a reader cannot verify.
type ReleaseHeader struct {
	Selection ReleaseSelection `json:"release"`
	Problems  int              `json:"problems"`
}

// LatestRelease returns the concrete release ReleaseLatest resolves to.
func LatestRelease() string { return lcbReleases[len(lcbReleases)-1] }

// releaseIndex returns the 1-based position of a "release_vN" value in
// lcbReleases, or 0 if it is not a known release.
func releaseIndex(rel string) int {
	for i, r := range lcbReleases {
		if r == rel {
			return i + 1
		}
	}
	return 0
}

// ResolveRelease parses an lcb_runner-style --release-version selector, mirroring
// upstream so results are comparable across dataset versions:
//
//   - "" or "release_latest": the newest known release, recorded explicitly.
//   - "release_vN" (v1..v6):  that single release.
//   - "vA_vB" (A <= B):       the inclusive range release_vA .. release_vB.
//
// An unknown version, a malformed range, or a reversed range is a clear error
// rather than a silent default.
func ResolveRelease(selector string) (ReleaseSelection, error) {
	sel := strings.TrimSpace(selector)
	if sel == "" || sel == ReleaseLatest {
		latest := LatestRelease()
		return ReleaseSelection{Selector: ReleaseLatest, Releases: []string{latest}, Resolved: latest}, nil
	}
	if lo, hi, ok := parseShortRange(sel); ok {
		loIdx, hiIdx := releaseIndex("release_"+lo), releaseIndex("release_"+hi)
		if loIdx == 0 || hiIdx == 0 {
			return ReleaseSelection{}, fmt.Errorf("livecodebench release: range %q references an unknown release (known: release_v1..release_v%d)", sel, len(lcbReleases))
		}
		if loIdx > hiIdx {
			return ReleaseSelection{}, fmt.Errorf("livecodebench release: range %q is reversed (%s comes after %s)", sel, lo, hi)
		}
		rels := append([]string(nil), lcbReleases[loIdx-1:hiIdx]...)
		return ReleaseSelection{Selector: sel, Releases: rels, Resolved: rels[len(rels)-1]}, nil
	}
	if releaseIndex(sel) != 0 {
		return ReleaseSelection{Selector: sel, Releases: []string{sel}, Resolved: sel}, nil
	}
	return ReleaseSelection{}, fmt.Errorf("livecodebench release: unknown --release-version %q (want release_v1..release_v%d, release_latest, or a range like v1_v3)", sel, len(lcbReleases))
}

// PinRelease resolves the selector and stamps it against a problem count for a
// report header, so the resolved release and the count it was scored over are
// recorded together and cannot drift.
func PinRelease(selector string, problems int) (ReleaseHeader, error) {
	sel, err := ResolveRelease(selector)
	if err != nil {
		return ReleaseHeader{}, err
	}
	return ReleaseHeader{Selection: sel, Problems: problems}, nil
}

// parseShortRange splits a compact "vA_vB" selector into its two short tokens.
// It only matches the two-token underscore form where each token is "v" + an
// integer, so a single prefixed release ("release_v2" -> ["release","v2"]) and
// any other junk fall through to be handled elsewhere.
func parseShortRange(sel string) (lo, hi string, ok bool) {
	parts := strings.Split(sel, "_")
	if len(parts) != 2 {
		return "", "", false
	}
	for _, p := range parts {
		if len(p) < 2 || p[0] != 'v' {
			return "", "", false
		}
		if _, err := strconv.Atoi(p[1:]); err != nil {
			return "", "", false
		}
	}
	return parts[0], parts[1], true
}
