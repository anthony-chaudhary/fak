package dojocal

// rewrite.go is the load-bearing PURE core of the dojo-RSI worktree arm (Phase 2
// of docs/fak/dojo-rsi-loop.md, issue #1024): the cell-anchored rewrite of the one
// `claim(<float>,` / `floor(<float>,` literal a RECALIBRATE candidate re-points at
// its corpus mean. It is the dojo twin of rsiloop.rewriteTunable, but the dojo's
// registry is harder than a single `DefaultCacheSize = <int>` const in two ways
// that make a naive global replace WRONG:
//
//  1. The same number recurs across cells — six cells claim 1.0 — so the rewrite
//     MUST be anchored on the (lever, metric) key, not the bare value.
//  2. The number also appears in the PROSE basis ("~85% ... (share = 0.85)"), so a
//     value-only replace would corrupt the human justification.
//
// The fix is one anchored regex per cell: match `{"lever", "metric"}: claim(` (or
// `floor(`) and rewrite only the FIRST numeric argument that immediately follows.
// Everything else — the basis string, a floor's trailing bool, every other cell —
// is byte-untouched. The function is pure (bytes in, bytes out, no I/O, no clock),
// so the worktree harness reads claims.go, calls RewriteClaim, writes the result
// back into the throwaway worktree, and the keep-bit's TruthClean rung can demand
// `treeChangedOnly(claims.go)` — exactly one file, exactly one recalibrated literal.
//
// Floor protection does NOT live here: a floor's claim is pinned and the proposer
// never emits a RECALIBRATE for one (dojocal.ProposeRecals routes it ROUTE_FLOOR;
// dojo.FoldCalibrable folds it by its breach, not its calib_err). So a floor can
// never reach this rewriter through the legitimate path. This stays a pure text op
// on whatever cell it is handed; the structural floor guard is upstream.

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// ClaimsRelPath is the module-relative path of the dojo claim registry — the one
// file the RSI loop's RECALIBRATE arm rewrites. Named here so the worktree harness
// and the rewrite verifier agree on one literal, exactly as rsiloop pins
// tunableRelPath. Forward slashes; the command layer joins it onto the module dir.
const ClaimsRelPath = "internal/dojo/claims.go"

// claimArgRe is the float immediately after `claim(`/`floor(`. Anchored separately
// per cell (see cellClaimRe) so it only ever rewrites the first numeric argument of
// the targeted cell's constructor and never the basis prose or a floor's bool. The
// numeric form matches Go float literals the registry uses: an optional sign, an
// integer part, and an optional fraction (e.g. 1, 1.0, 0.85, -0.5).
const claimArgRe = `(-?\d+(?:\.\d+)?)`

// cellClaimRe builds the anchored rewrite regex for one (lever, metric) cell. The
// lever/metric are QuoteMeta'd so a value containing a regex metacharacter is a
// literal. Whitespace is permissive (\s* matches across a newline too) so the match
// survives any gofmt reflow of the key/constructor, though the registry keeps them
// on one line today. Group 1 is the anchor up to and including the open paren; group
// 2 is the float to rewrite.
func cellClaimRe(lever, metric string) *regexp.Regexp {
	anchor := `\{\s*"` + regexp.QuoteMeta(lever) + `"\s*,\s*"` + regexp.QuoteMeta(metric) + `"\s*\}\s*:\s*(?:claim|floor)\(\s*`
	return regexp.MustCompile(`(` + anchor + `)` + claimArgRe)
}

// RewriteClaim re-points the single registered claim literal for (lever, metric) in
// the claims.go source `src` to newClaimed, returning the rewritten bytes and the
// OLD value it replaced. It is the pure rewrite the worktree arm applies inside a
// throwaway worktree before re-measuring FoldCalibrable.
//
// It fails closed on anything that would make the rewrite ambiguous or empty:
//   - the cell's `claim(`/`floor(` anchor is not found (an unregistered or moved
//     cell — the rewrite contract is broken, never a silent no-op),
//   - the anchor matches more than once (the registry's keys are unique; a double
//     match means the source is not the registry this targets),
//   - the new value formats identically to the old (a no-op is not a real
//     recalibration — the same "changed nothing fails closed" rule treeChangedOnly
//     enforces on the worktree side).
//
// Only the float is rewritten; the basis string (which may quote the same number),
// a floor's trailing bool, and every other cell are byte-identical in the output.
func RewriteClaim(src []byte, lever, metric string, newClaimed float64) ([]byte, float64, error) {
	re := cellClaimRe(lever, metric)
	locs := re.FindAllSubmatchIndex(src, -1)
	switch {
	case len(locs) == 0:
		return nil, 0, fmt.Errorf("dojocal: no claim/floor literal for cell %s/%s in %s (cell unregistered or moved — the rewrite anchor is broken)", lever, metric, ClaimsRelPath)
	case len(locs) > 1:
		return nil, 0, fmt.Errorf("dojocal: claim literal for cell %s/%s is ambiguous (%d matches) — the source is not the canonical registry", lever, metric, len(locs))
	}

	m := locs[0]
	// Submatch indexes: [0:1]=whole, [2:3]=group1 anchor, [4:5]=group2 float.
	valStart, valEnd := m[4], m[5]
	oldText := string(src[valStart:valEnd])
	oldVal, err := strconv.ParseFloat(oldText, 64)
	if err != nil {
		return nil, 0, fmt.Errorf("dojocal: cell %s/%s literal %q is not a float: %w", lever, metric, oldText, err)
	}

	newText := formatClaim(newClaimed)
	if newText == oldText {
		return nil, oldVal, fmt.Errorf("dojocal: rewrite of %s/%s is a no-op (claims.go already at %s) — a recalibration must change the literal", lever, metric, oldText)
	}

	out := make([]byte, 0, len(src)-len(oldText)+len(newText))
	out = append(out, src[:valStart]...)
	out = append(out, newText...)
	out = append(out, src[valEnd:]...)
	return out, oldVal, nil
}

// ReadClaim returns the literal claim value currently registered for (lever, metric)
// in `src`, without rewriting anything. The worktree harness uses it to read the
// pinned-baseline literal a candidate forks from; the preview command uses it to show
// the before value. Same fail-closed anchoring as RewriteClaim (not-found / ambiguous
// are errors, never a silent zero).
func ReadClaim(src []byte, lever, metric string) (float64, error) {
	re := cellClaimRe(lever, metric)
	locs := re.FindAllSubmatchIndex(src, -1)
	switch {
	case len(locs) == 0:
		return 0, fmt.Errorf("dojocal: no claim/floor literal for cell %s/%s in %s", lever, metric, ClaimsRelPath)
	case len(locs) > 1:
		return 0, fmt.Errorf("dojocal: claim literal for cell %s/%s is ambiguous (%d matches)", lever, metric, len(locs))
	}
	m := locs[0]
	return strconv.ParseFloat(string(src[m[4]:m[5]]), 64)
}

// formatClaim renders a float as a Go float literal in the registry's style: the
// shortest exact decimal, but always carrying a decimal point so an integral value
// stays visually a float (1 -> "1.0", 0 -> "0.0", 0.893 -> "0.893"). The proposer's
// NewClaimed is already Round3, so this never emits a long mantissa.
func formatClaim(v float64) string {
	s := strconv.FormatFloat(v, 'g', -1, 64)
	if !strings.ContainsAny(s, ".eE") {
		s += ".0"
	}
	return s
}

// ClaimChangeLine reports the single source line that RewriteClaim would change for
// (lever, metric): its 1-based line number and the before/after text, for a legible
// preview diff. It re-runs RewriteClaim internally so the preview is exactly what an
// apply would write (never a hand-rendered approximation). A rewrite that changes no
// line (impossible once RewriteClaim succeeds) returns ok=false.
func ClaimChangeLine(src []byte, lever, metric string, newClaimed float64) (lineNo int, before, after string, err error) {
	out, _, err := RewriteClaim(src, lever, metric, newClaimed)
	if err != nil {
		return 0, "", "", err
	}
	oldLines := strings.Split(string(src), "\n")
	newLines := strings.Split(string(out), "\n")
	for i := 0; i < len(oldLines) && i < len(newLines); i++ {
		if oldLines[i] != newLines[i] {
			return i + 1, oldLines[i], newLines[i], nil
		}
	}
	return 0, "", "", fmt.Errorf("dojocal: rewrite of %s/%s produced no line change", lever, metric)
}
