package hooks

import (
	"bytes"
	"fmt"
	"go/format"
	"sort"
	"strings"
)

// gate_gofmt.go — the GOFMT gate. It is the commit-boundary sibling of `make ci`'s
// `gofmt-check` target: it flags any staged .go file whose content gofmt would
// reformat, at the commit that introduces it, BEFORE the unformatted file reaches
// the trunk and reds every peer's `make ci` at the gofmt step.
//
// Why a commit-boundary gate: gofmt drift is a RECURRING trunk red — the release
// notes carry repeated "clear the CI gofmt gate" fixes (v0.32.0 alone landed four:
// ctxplan / three test tables / turntaxdemo; v0.34.0 another for gateway/metrics.go).
// The drift lands because gofmt is only checked post-hoc in `make ci`, which a worker
// that commits+pushes without running the full gate never sees, and the remote CI that
// would otherwise catch it is billing-blocked on the private repo (see the note in
// tools/githooks/pre-push). The commit boundary — which git runs below the agent layer,
// binding Claude Code, Codex, and a human alike — is the earliest deterministic place to
// catch it. `fak test gofmt` already exposes the same `gofmt -l .` check as a manual
// runner; this gate is that check wired into the boundary that actually fires unprompted.
//
// ADVISORY BY DEFAULT (DefaultMode "warn"), exactly like BARE_COMMIT_SWEEP /
// E2E_OVER_MOCKS / PRIOR_ART / UNTIERED_LEAF: out of the box it never reds a shared
// trunk — it names the unformatted staged files and the one-line fix (`gofmt -w`). Set
// FLEET_GOFMT_GUARD=block to hard-enforce it (refuse the commit), or ALLOW_GOFMT_DRIFT=1
// to skip it once.

// gofmtListCap bounds how many unformatted paths the finding lists before "(+N more)" —
// enough to see the drift without a wall of text on a large reformat.
const gofmtListCap = 12

// gateGofmt fires ONE GOFMT finding when any staged .go file's content is not gofmt-clean.
// A file gofmt cannot parse (a syntax error the build/vet gate owns anyway) is skipped so
// this gate never fails on unparseable input; a deleted/unreadable path is skipped; a
// staged set with no .go files, or all-gofmt-clean .go files, returns clean.
func gateGofmt(d *StagedDiff) ([]Finding, error) {
	var unformatted []string
	// judged counts the staged .go files this gate actually formatted and compared — after the
	// unreadable and unparseable skips, so the denominator names the set it truly judged rather
	// than the set it was offered (#5602).
	judged := 0
	for _, p := range d.StagedPaths {
		if !strings.HasSuffix(p, ".go") {
			continue
		}
		src, ok := d.FileBytes(p)
		if !ok {
			continue // deleted or unreadable — nothing to format
		}
		formatted, err := format.Source(src)
		if err != nil {
			continue // unparseable — the build/vet gate owns syntax, not gofmt
		}
		judged++
		if !bytes.Equal(formatted, src) {
			unformatted = append(unformatted, strings.ReplaceAll(p, "\\", "/"))
		}
	}
	d.NoteCandidates("GOFMT", judged, "staged .go file(s)")
	if len(unformatted) == 0 {
		return nil, nil
	}
	sort.Strings(unformatted)
	return []Finding{{
		Gate:   "GOFMT",
		File:   unformatted[0],
		Line:   0,
		Detail: gofmtDetail(unformatted),
	}}, nil
}

// gofmtDetail renders the one-line advisory: how many staged .go files are unformatted, a
// capped list of them, and the in-place fix + escape hatches.
func gofmtDetail(paths []string) string {
	shown := paths
	extra := 0
	if len(shown) > gofmtListCap {
		extra = len(shown) - gofmtListCap
		shown = shown[:gofmtListCap]
	}
	list := strings.Join(shown, ", ")
	if extra > 0 {
		list = fmt.Sprintf("%s (+%d more)", list, extra)
	}
	return fmt.Sprintf(
		"%d staged .go file(s) are not gofmt-clean: %s — they would red `make ci`'s "+
			"gofmt-check once on the trunk. Fix in place before committing: `gofmt -w %s`. "+
			"(advisory; FLEET_GOFMT_GUARD=block enforces, ALLOW_GOFMT_DRIFT=1 skips once)",
		len(paths), list, strings.Join(paths, " "),
	)
}
