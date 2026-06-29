// Package releasestale answers one question durably: "is the version that
// `go install github.com/.../cmd/fak@latest` would install actually current, or
// has the trunk moved far past it?"
//
// This is the PUBLISH axis of staleness, and it is distinct from the source axis
// that internal/binstamp + `fak self-update` already cover. Those compare a RUNNING
// binary's embedded VCS revision to origin/main HEAD — keeping a built-from-source
// guard fleet converged. But an external adopter (or the module proxy) does not build
// from HEAD: `@latest` resolves to the newest semver TAG. If no tag is cut as work
// lands, `@latest` silently rots — the trunk can be hundreds of commits ahead of the
// last published tag while every `fak version` line still looks plausible. That is the
// "we did the work but the latest version isn't the one being used" failure, and until
// now nothing measured it as a first-class, gateable number (the Python release-status
// fold knows commits-since-tag but is slow, Python-only, and not surfaced as a loud
// verdict).
//
// The verdict is a PURE function of injected git facts (Compute over Facts), so it is
// deterministic and unit-testable without a repo. Gather is the thin impure shell that
// reads those facts via git. The design mirrors binstamp: conservative — when git is
// unreadable or no semver tag exists, the verdict is Unknown (never a false "stale" that
// would red a gate spuriously); only a tag that provably lags HEAD beyond the thresholds
// reads Stale / VeryStale.
//
// It performs NO release: it is pure observation. Cutting the tag (the fix) lives in the
// /release skill and tools/release_*.py + release-cadence.yml; this package is the signal
// that says when that fix is overdue, and by how much.
package releasestale

import (
	"context"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
)

// Schema is the stable envelope id for the JSON payload, matching the
// schema/ok/verdict/finding/reason/next_action shape every other fak fold emits.
const Schema = "fak-release-staleness/1"

// semverRe matches a plain rolling release tag (vMAJOR.MINOR.PATCH). Pre-release and
// channel tags (stable/...) are deliberately excluded — `@latest` only ever resolves to
// a clean semver tag, so that is the only thing whose lag matters here.
var semverRe = regexp.MustCompile(`^v?(\d+)\.(\d+)\.(\d+)$`)

// Verdict is the publish-freshness judgment of the latest tag against HEAD.
type Verdict int

const (
	// Unknown: git unreadable, or no reachable semver tag exists — cannot prove fresh
	// OR stale. A gate must NOT fail on this (mirrors binstamp.Unknown), but the reason
	// is surfaced so the absence of evidence is itself legible.
	Unknown Verdict = iota
	// Fresh: the latest published tag is at HEAD, or lags by less than the stale floor —
	// `@latest` tracks the trunk closely enough.
	Fresh
	// Stale: the latest tag lags HEAD past the stale threshold (commits OR days) — a
	// release is due so `@latest` stops rotting.
	Stale
	// VeryStale: the lag is past the very-stale threshold — `@latest` is badly behind and
	// adopters/dogfooders are installing an old binary.
	VeryStale
)

func (v Verdict) String() string {
	switch v {
	case Fresh:
		return "fresh"
	case Stale:
		return "stale"
	case VeryStale:
		return "very_stale"
	default:
		return "unknown"
	}
}

// Thresholds gate the Fresh/Stale/VeryStale boundary. A lag at or above EITHER the
// commit or the day bound for a level promotes the verdict to that level. Defaults are
// deliberately lenient on the commit count (you do not cut a tag per commit) but firm on
// elapsed time (a fortnight without a published release is a real lag).
type Thresholds struct {
	StaleCommits     int     `json:"stale_commits"`
	StaleDays        float64 `json:"stale_days"`
	VeryStaleCommits int     `json:"very_stale_commits"`
	VeryStaleDays    float64 `json:"very_stale_days"`
}

// DefaultThresholds returns the built-in lag bounds.
func DefaultThresholds() Thresholds {
	return Thresholds{
		StaleCommits:     20,
		StaleDays:        14,
		VeryStaleCommits: 100,
		VeryStaleDays:    45,
	}
}

// Facts is the injected git/repo evidence the verdict is computed from. Gathering them
// is Gather's job; Compute is a pure function of them so the same Facts always yields the
// same verdict, with no git in the test.
type Facts struct {
	Reachable     bool    // git was readable at all
	LatestTag     string  // newest vX.Y.Z tag MERGED INTO HEAD — what `@latest` resolves to ("" if none)
	TagSHA        string  // the commit the latest tag points at
	HeadSHA       string  // current trunk tip
	CommitsBehind int     // commits on HEAD not reachable from the latest tag (rev-list --count tag..HEAD)
	DaysBehind    float64 // HEAD commit time minus tag commit time, in days (0 if unknown)
	VersionFile   string  // the VERSION marker at the working tree (e.g. "0.34.0"; "" if absent)
}

// Payload is the control-pane record: the schema/ok/verdict/finding/reason/next_action
// envelope plus the raw measurements, so a loop or the cadence fold can consume one JSON
// object instead of re-deriving the lag.
type Payload struct {
	Schema            string     `json:"schema"`
	OK                bool       `json:"ok"`
	Verdict           string     `json:"verdict"`
	Finding           string     `json:"finding"`
	Reason            string     `json:"reason"`
	NextAction        string     `json:"next_action"`
	Workspace         string     `json:"workspace"`
	LatestTag         string     `json:"latest_tag"`
	TagSHA            string     `json:"tag_sha"`
	HeadSHA           string     `json:"head_sha"`
	CommitsBehind     int        `json:"commits_behind"`
	DaysBehind        float64    `json:"days_behind"`
	VersionFile       string     `json:"version_file"`
	VersionAheadOfTag bool       `json:"version_ahead_of_tag"`
	Thresholds        Thresholds `json:"thresholds"`
}

// Compute is the pure core: it turns Facts + Thresholds into the full Payload (verdict,
// gate bit, and the operator next-action). No git, no clock, no I/O.
func Compute(f Facts, t Thresholds, workspace string) Payload {
	p := Payload{
		Schema:        Schema,
		Workspace:     workspace,
		LatestTag:     f.LatestTag,
		TagSHA:        f.TagSHA,
		HeadSHA:       f.HeadSHA,
		CommitsBehind: f.CommitsBehind,
		DaysBehind:    f.DaysBehind,
		VersionFile:   f.VersionFile,
		Thresholds:    t,
	}
	p.VersionAheadOfTag = versionAheadOfTag(f.VersionFile, f.LatestTag)

	v := classify(f, t)
	p.Verdict = v.String()
	// OK is the gate bit: only a proven lag (Stale/VeryStale) fails. Unknown is advisory
	// — a gate must not red on the absence of evidence (e.g. no tag yet, git unreadable).
	p.OK = v != Stale && v != VeryStale

	switch v {
	case Unknown:
		p.Finding = "publish_unknown"
		if !f.Reachable {
			p.Reason = "could not read git — cannot judge whether @latest tracks HEAD"
			p.NextAction = "verify git is readable from the workspace root, then re-run"
		} else {
			p.Reason = "no reachable vX.Y.Z tag is merged into HEAD — nothing has been published for @latest to resolve to"
			p.NextAction = "cut the first release with /release so `go install ...@latest` resolves to a real version"
		}
	case Fresh:
		p.Finding = "publish_fresh"
		p.Reason = freshReason(f)
		p.NextAction = "hold; `@latest` tracks HEAD within the staleness thresholds"
	case Stale, VeryStale:
		p.Finding = "publish_stale"
		if v == VeryStale {
			p.Finding = "publish_very_stale"
		}
		p.Reason = lagReason(f, v)
		p.NextAction = nextAction(p)
	}
	return p
}

// classify is the pure verdict rule. Order matters: very-stale dominates stale dominates
// fresh, and either the commit OR the day bound can promote a level.
func classify(f Facts, t Thresholds) Verdict {
	if !f.Reachable || f.LatestTag == "" {
		return Unknown
	}
	if f.CommitsBehind <= 0 {
		return Fresh
	}
	if atLeast(f.CommitsBehind, t.VeryStaleCommits) || atLeastF(f.DaysBehind, t.VeryStaleDays) {
		return VeryStale
	}
	if atLeast(f.CommitsBehind, t.StaleCommits) || atLeastF(f.DaysBehind, t.StaleDays) {
		return Stale
	}
	return Fresh
}

// atLeast reports n >= bound, treating a non-positive bound as "disabled" (never trips).
func atLeast(n, bound int) bool { return bound > 0 && n >= bound }

func atLeastF(n, bound float64) bool { return bound > 0 && n >= bound }

func freshReason(f Facts) string {
	if f.CommitsBehind <= 0 {
		return "the latest tag " + f.LatestTag + " is at HEAD — `@latest` is current"
	}
	return "the latest tag " + f.LatestTag + " lags HEAD by " + strconv.Itoa(f.CommitsBehind) +
		" commit(s) / " + days(f.DaysBehind) + "d, within the staleness thresholds"
}

func lagReason(f Facts, v Verdict) string {
	level := "stale"
	if v == VeryStale {
		level = "very stale"
	}
	return "`@latest` is " + level + ": the latest published tag " + f.LatestTag + " lags HEAD by " +
		strconv.Itoa(f.CommitsBehind) + " commit(s) / " + days(f.DaysBehind) +
		"d — adopters running `go install ...@latest` get " + f.LatestTag + ", not the work on the trunk"
}

// nextAction points at the real publish levers and distinguishes the two failure shapes:
// an untagged cut that landed (tag it) vs ordinary lag (cut a release).
func nextAction(p Payload) string {
	if p.VersionAheadOfTag {
		return "VERSION (" + p.VersionFile + ") is ahead of the latest tag " + p.LatestTag +
			" — a cut commit landed untagged; publish the tag (release_tag.py / /release step 6) so `@latest` advances"
	}
	return "cut a release (" + strconv.Itoa(p.CommitsBehind) + " commit(s) / " + days(p.DaysBehind) +
		"d since " + p.LatestTag + "): run `/release`, or arm release-cadence.yml (workflow_dispatch dry_run=false / the RELEASE_CADENCE_AUTO opt-in) so `@latest` tracks HEAD"
}

// versionAheadOfTag reports whether the VERSION marker is a higher semver than the latest
// tag — i.e. a release commit bumped VERSION but the tag was never pushed.
func versionAheadOfTag(version, tag string) bool {
	vv, okv := parseSemver(version)
	tv, okt := parseSemver(tag)
	if !okv || !okt {
		return false
	}
	return less(tv, vv)
}

func parseSemver(s string) ([3]int, bool) {
	m := semverRe.FindStringSubmatch(strings.TrimSpace(s))
	if m == nil {
		return [3]int{}, false
	}
	var out [3]int
	for i := 0; i < 3; i++ {
		n, err := strconv.Atoi(m[i+1])
		if err != nil {
			return [3]int{}, false
		}
		out[i] = n
	}
	return out, true
}

func less(a, b [3]int) bool {
	for i := 0; i < 3; i++ {
		if a[i] != b[i] {
			return a[i] < b[i]
		}
	}
	return false
}

// days renders a day count compactly: integral days without a decimal, fractional days to
// one place, so "13" and "0.4" both read cleanly in the human line.
func days(d float64) string {
	if d == float64(int64(d)) {
		return strconv.FormatInt(int64(d), 10)
	}
	return strconv.FormatFloat(d, 'f', 1, 64)
}

// Render is the human one-screen view of a Payload.
func Render(p Payload) string {
	mark := "OK"
	if !p.OK {
		mark = "ACTION"
	}
	lines := []string{
		"release staleness — " + mark + " (" + p.Verdict + ")",
		"  latest published tag: " + dashIfEmpty(p.LatestTag) + "   HEAD: " + shortSHA(p.HeadSHA),
		"  behind: " + strconv.Itoa(p.CommitsBehind) + " commit(s) / " + days(p.DaysBehind) + " day(s)",
	}
	if p.VersionAheadOfTag {
		lines = append(lines, "  note: VERSION ("+p.VersionFile+") is ahead of the tag — a cut landed untagged")
	}
	lines = append(lines, "  "+p.Reason, "", "  -> "+p.NextAction)
	return strings.Join(lines, "\n")
}

func dashIfEmpty(s string) string {
	if strings.TrimSpace(s) == "" {
		return "(none)"
	}
	return s
}

func shortSHA(s string) string {
	s = strings.TrimSpace(s)
	if len(s) > 12 {
		return s[:12]
	}
	if s == "" {
		return "(unknown)"
	}
	return s
}

// ---- the impure shell: gather facts via git --------------------------------------------

// Runner runs a command in dir and returns its trimmed stdout and whether it succeeded.
// Injected so Gather is testable with a canned git transcript (mirrors the
// selfinstall.RealRunner shape used by `fak self-update`).
type Runner func(ctx context.Context, dir, name string, args ...string) (string, bool)

// RealRunner is the production Runner: it execs the command and returns trimmed stdout,
// ok=false on any non-zero exit or exec error. Read-only git only.
func RealRunner(ctx context.Context, dir, name string, args ...string) (string, bool) {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		return "", false
	}
	return strings.TrimSpace(string(out)), true
}

// Gather reads the publish-staleness Facts from the repo at root using git (and the
// VERSION marker passed in by the caller, which already knows how to resolve it). It is
// deliberately tolerant: any unreadable rung leaves its field zero/empty and Reachable
// reflects whether the HEAD read itself worked.
func Gather(ctx context.Context, run Runner, root, versionFile string) Facts {
	f := Facts{VersionFile: strings.TrimSpace(versionFile)}

	head, ok := run(ctx, root, "git", "--no-optional-locks", "rev-parse", "HEAD")
	if !ok || head == "" {
		return f // Reachable stays false: git could not be read
	}
	f.Reachable = true
	f.HeadSHA = head

	f.LatestTag = latestSemverTag(ctx, run, root)
	if f.LatestTag == "" {
		return f
	}

	if sha, ok := run(ctx, root, "git", "--no-optional-locks", "rev-list", "-n1", f.LatestTag); ok {
		f.TagSHA = sha
	}
	if cnt, ok := run(ctx, root, "git", "--no-optional-locks", "rev-list", "--count", f.LatestTag+"..HEAD"); ok {
		if n, err := strconv.Atoi(strings.TrimSpace(cnt)); err == nil {
			f.CommitsBehind = n
		}
	}
	f.DaysBehind = daysBetween(commitEpoch(ctx, run, root, f.LatestTag), commitEpoch(ctx, run, root, "HEAD"))
	return f
}

// latestSemverTag returns the newest vX.Y.Z tag MERGED INTO HEAD — exactly what
// `go install ...@latest` resolves to (the proxy only serves tags reachable on the
// default branch). Mirrors tools/release_status.py's semver_tags(merged=True).
func latestSemverTag(ctx context.Context, run Runner, root string) string {
	out, ok := run(ctx, root, "git", "--no-optional-locks", "tag", "--sort=-v:refname", "--merged", "HEAD")
	if !ok {
		return ""
	}
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if semverRe.MatchString(line) {
			return line
		}
	}
	return ""
}

// commitEpoch returns the committer epoch seconds of ref, or 0 if unreadable.
func commitEpoch(ctx context.Context, run Runner, root, ref string) int64 {
	out, ok := run(ctx, root, "git", "--no-optional-locks", "log", "-1", "--format=%ct", ref)
	if !ok {
		return 0
	}
	n, err := strconv.ParseInt(strings.TrimSpace(out), 10, 64)
	if err != nil {
		return 0
	}
	return n
}

// daysBetween returns (head - tag) in days, never negative, 0 when either epoch is unknown.
func daysBetween(tagEpoch, headEpoch int64) float64 {
	if tagEpoch <= 0 || headEpoch <= 0 || headEpoch <= tagEpoch {
		return 0
	}
	secs := float64(headEpoch - tagEpoch)
	d := secs / 86400.0
	// round to one decimal so the JSON is stable and the human line is tidy
	return float64(int64(d*10+0.5)) / 10
}
