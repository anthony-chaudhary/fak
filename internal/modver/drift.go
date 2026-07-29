package modver

// Drift is the per-module projection of release staleness. internal/releasestale
// answers "has the WHOLE binary moved far past the tag `go install ...@latest`
// resolves to?" — one number for the trunk. This answers the finer question that
// number cannot: WHICH modules are rotting behind `@latest`, and by how much. It
// joins tag history (the `@latest` boundary) with per-module revs (the count of
// non-merge commits touching a module in `tag..HEAD`), so a release-readiness
// readout can name the modules that carry unpublished work — "internal/gateway is
// 7 revs ahead of the last tag" — instead of only "the binary is 40 commits stale".
//
// The reference point is deliberately the SAME boundary releasestale measures the
// whole-binary lag against: the newest vMAJOR.MINOR.PATCH tag merged into HEAD (see
// latestMergedSemverTag). Sharing that definition is what keeps the per-module drift
// readout and the whole-binary staleness verdict talking about the same `@latest`.

import (
	"context"
	"regexp"
	"sort"
	"strings"
)

// DriftSchema is the control-plane envelope id for a drift readout, mirroring
// releasestale.Schema ("fak-release-staleness/1") so a release-readiness fold can
// consume one typed JSON object. It is a NEW schema for a NEW readout — additive,
// not a change to the fak-module-versions/1 ledger row (a breaking row change would
// be a /2 with its own contract).
const DriftSchema = "fak-module-drift/1"

// DriftRow is one module that moved since the last published tag: its rev count
// WITHIN the tag..HEAD range (not its absolute lifetime rev) plus the last touch in
// that range. RevsSinceTag is exactly "moved N revs since the last tag".
type DriftRow struct {
	Module       string `json:"module"`
	Kind         string `json:"kind"`
	RevsSinceTag int    `json:"revs_since_tag"`
	LastCommit   string `json:"last_commit"`
	LastDate     string `json:"last_date"`
}

// DriftReport is the whole readout: the `@latest` boundary it was measured against,
// how many live modules were scanned, and one row per module that moved since that
// boundary, most-moved-first. Tag == "" means no published semver tag is merged into
// HEAD — there is no `@latest` reference point, so Rows is empty by construction (a
// conservative empty readout, never a false "everything drifted"; mirrors
// releasestale.Unknown).
type DriftReport struct {
	Schema  string     `json:"schema"`
	Tag     string     `json:"tag"`     // the @latest boundary (newest merged vX.Y.Z tag), "" if none
	TagSHA  string     `json:"tag_sha"` // commit the tag points at
	Head    string     `json:"head"`
	Scanned int        `json:"scanned"` // live modules considered (the denominator)
	Moved   int        `json:"moved"`   // == len(Rows)
	Rows    []DriftRow `json:"rows"`
}

// Drift is the pure core: it folds a `tag..HEAD` range log (same
// %x1e%h%x09%cI + --name-only shape parseLog reads) against the live module set into
// the drift readout. It reuses parseLog, so the rev semantics are identical to
// Snapshot's — distinct non-merge commits touching the module, deleted modules
// dropped (a module absent at HEAD is gone, not "rotting behind @latest") — only the
// history is bounded to the post-tag range. No git, no clock, no I/O: the same
// (rangeLog, live, tag) always yields the same report, so the witness is a fixture.
func Drift(rangeLog []byte, live map[string]bool, tag, tagSHA, head string) DriftReport {
	moved := parseLog(rangeLog, live)
	rows := make([]DriftRow, 0, len(moved))
	for _, m := range moved {
		rows = append(rows, DriftRow{
			Module:       m.Name,
			Kind:         m.Kind,
			RevsSinceTag: m.Rev,
			LastCommit:   m.LastCommit,
			LastDate:     m.LastDate,
		})
	}
	// Most-moved-first: the modules carrying the most unpublished work lead the
	// readout (the rows a release-readiness scan wants first). Ties fall back to name
	// so the order is total and deterministic.
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].RevsSinceTag != rows[j].RevsSinceTag {
			return rows[i].RevsSinceTag > rows[j].RevsSinceTag
		}
		return rows[i].Module < rows[j].Module
	})
	return DriftReport{
		Schema:  DriftSchema,
		Tag:     tag,
		TagSHA:  tagSHA,
		Head:    head,
		Scanned: len(live),
		Moved:   len(rows),
		Rows:    rows,
	}
}

// driftSemverRe matches a plain rolling release tag (vMAJOR.MINOR.PATCH) — the same
// definition internal/releasestale.semverRe uses, kept in sync deliberately: both the
// per-module drift readout and the whole-binary staleness verdict resolve `@latest`
// to the newest such tag merged into HEAD, so they must agree on what a tag IS.
var driftSemverRe = regexp.MustCompile(`^v?\d+\.\d+\.\d+$`)

// DriftSnapshot is the impure shell: it resolves the `@latest` boundary and runs the
// bounded range log via the Runner seam, then delegates to Drift. Conservative when
// git is unreadable or no semver tag is merged into HEAD — it returns a well-formed
// empty readout (Head set, no rows) rather than an error or a false drift, matching
// releasestale's "no reference point => Unknown, never a spurious stale" stance.
func DriftSnapshot(ctx context.Context, dir string, run Runner) (DriftReport, error) {
	run = gitRunner(run)
	head, err := headShort(ctx, dir, run)
	if err != nil {
		return DriftReport{}, err
	}

	tag, tagSHA := latestMergedSemverTag(ctx, dir, run)
	if tag == "" {
		return DriftReport{Schema: DriftSchema, Head: head}, nil
	}

	// Same shape and rev semantics as Snapshot's history walk (liveAndLog owns both),
	// but BOUNDED to tag..HEAD so each module's rev IS its commit count since the last
	// tag — the one thing this pass does differently.
	live, logOut, err := liveAndLog(ctx, dir, run, tag+"..HEAD")
	if err != nil {
		return DriftReport{}, err
	}
	return Drift(logOut, live, tag, tagSHA, head), nil
}

// latestMergedSemverTag returns the newest vX.Y.Z tag merged into HEAD and the commit
// it points at — exactly what `go install ...@latest` resolves to, and the same
// boundary internal/releasestale.latestSemverTag resolves for the whole-binary lag.
// "" when no reachable semver tag exists. Any unreadable rung is tolerated (git
// failure => "" tag => empty readout upstream), never fatal.
func latestMergedSemverTag(ctx context.Context, dir string, run Runner) (tag, sha string) {
	out, err := run(ctx, dir, "tag", "--sort=-v:refname", "--merged", "HEAD")
	if err != nil {
		return "", ""
	}
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if driftSemverRe.MatchString(line) {
			tag = line
			break
		}
	}
	if tag == "" {
		return "", ""
	}
	if shaOut, err := run(ctx, dir, "rev-list", "-n1", tag); err == nil {
		sha = strings.TrimSpace(string(shaOut))
	}
	return tag, sha
}
