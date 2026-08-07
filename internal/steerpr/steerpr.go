// Package steerpr folds continuous-merge trunk commits into operator-legible,
// PR-sized units and bands each unit by where operator attention is owed.
//
// The fak trunk takes work PR-free: one issue, one commit, one leaf, stamped
// (fak <leaf>). That is optimal for throughput, but it dissolves the unit of
// operator ATTENTION — there is no PR to read, so the only window onto "what is
// the fleet doing, as a coherent unit I can react to" is the raw commit
// firehose. This package rebuilds that unit WITHOUT rebuilding the merge gate:
// it is a read-mostly overlay, and nothing here blocks, delays, or gates a
// commit.
//
// The fold is deterministic over git history — there is no plan file to go
// stale: every stamped commit is already a line item in the unit of the lane
// that owns it. It is the shared primitive behind both `fak release prplan`
// (the release-time promotion plan, ordered biggest-first) and the continuous
// operator view (ordered worst-attention-first via SortWorstFirst).
//
// The package is deliberately PURE and git-free: it takes already-read git log
// text and caller-supplied witness verdicts, so it is unit-testable without a
// repo and imports nothing internal. Verdicts are supplied rather than derived
// so the band reads the kernel's existing witness rungs (internal/dispatchtick)
// without this leaf depending on them — the band is a VIEW over an existing
// oracle, never a second oracle.
package steerpr

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// Schema is the machine payload identifier for the overlay fold.
const Schema = "fak.steerpr.v1"

var (
	leafRE  = regexp.MustCompile(`\(fak ([a-z0-9][a-z0-9-]*)\)\s*$`)
	typeRE  = regexp.MustCompile(`^([a-z]+)[(!:]`)
	issueRE = regexp.MustCompile(`#(\d+)\b`)
)

// Verdict is a per-commit witness verdict, supplied by the caller from the
// kernel's existing witness rungs (internal/dispatchtick). It is an input to
// this package, never something this package decides — that separation is what
// keeps the band a view over the real oracle rather than a competing one.
type Verdict string

const (
	// VerdictWitnessed: the diff proves the claim (the non-forgeable bit).
	VerdictWitnessed Verdict = "CLAIM_WITNESSED"
	// VerdictUnwitnessed: a claim was made and the diff did not prove it.
	VerdictUnwitnessed Verdict = "CLAIM_UNWITNESSED"
	// VerdictAbstain: no checkable claim was made.
	VerdictAbstain Verdict = "ABSTAIN"
	// VerdictNoCommit: the claim resolved to no commit at all.
	VerdictNoCommit Verdict = "CLAIM_NO_COMMIT"
	// VerdictUnknown: the verdict was not supplied (not yet graded).
	VerdictUnknown Verdict = ""
)

// Band is where operator attention is owed on a unit. It is the HUMAN_RESIDUAL
// doctrine ("escalate only what an oracle could not resolve") pointed at landed
// commits instead of harness prompts.
type Band string

const (
	// BandCleared: every member was witnessed. Attention buys nothing here.
	BandCleared Band = "CLEARED"
	// BandUnverifiable: a member made no checkable claim. Reviewable, lower priority.
	BandUnverifiable Band = "UNVERIFIABLE"
	// BandResidual: a member claimed something the machine could not confirm.
	// This is where operator attention buys something.
	BandResidual Band = "RESIDUAL"
)

// bandRank orders bands worst-first. A unit folds to its WORST member, so an
// operator who reads CLEARED may conclude EVERY member was witnessed — not
// most of them. The fold is pessimistic by design, not by accident.
var bandRank = map[Band]int{
	BandResidual:     0,
	BandUnverifiable: 1,
	BandCleared:      2,
}

// Commit is one landed trunk commit, parsed from git log.
type Commit struct {
	SHA      string   `json:"sha"`
	Subject  string   `json:"subject"`
	Leaf     string   `json:"leaf,omitempty"`
	Type     string   `json:"type,omitempty"`
	Resolves []string `json:"resolves,omitempty"` // #N bound in the subject (closure-grade)
	Mentions []string `json:"mentions,omitempty"` // #N only in the body (safe mention)
	Files    []string `json:"files,omitempty"`
	Verdict  Verdict  `json:"verdict,omitempty"`
	Band     Band     `json:"band,omitempty"`
}

// Unit is one operator-legible bundle of trunk commits: an Operator
// Steerability PR. Its Band is the worst of its members'.
type Unit struct {
	// Leaf is the unit's KEY: the (fak <leaf>) ship-stamp when GroupedBy is
	// "leaf", or the wave key ("wave:<n>", see WaveKey) when GroupedBy is "wave".
	// Every operator verb addresses a unit by this string, which is why the two
	// bases share one field — and why the wave key carries a colon the ship-stamp
	// grammar cannot produce, so the two key spaces can never collide.
	Leaf string `json:"leaf"`
	// GroupedBy names WHY this unit holds what it holds: "leaf" (one unit per
	// ship-stamp — the default and the fallback) or "wave" (one unit per `fak
	// issue cohort` wave). It is mandatory on every unit, never omitempty: two
	// grouping bases coexisting silently would make the overlay LESS legible than
	// one basis, which is the #5040 fail condition stated as a field tag.
	GroupedBy string `json:"grouped_by"`
	// Leaves are the distinct member ship-stamps folded into a WAVE unit, so an
	// operator reading a wave unit can still see which lanes it spans. Empty on a
	// leaf unit, where the answer is already Leaf.
	Leaves   []string       `json:"leaves,omitempty"`
	Title    string         `json:"title"`
	Commits  []Commit       `json:"commits"`
	Types    map[string]int `json:"types"`
	Resolves []string       `json:"resolves,omitempty"`
	Mentions []string       `json:"mentions,omitempty"`
	Files    []string       `json:"files"`
	Band     Band           `json:"band"`
	// Curve, when set, is the bound trajctl objective's progress signal carried
	// onto this unit (see curve.go). It is ORTHOGONAL to Band: Band says "was each
	// claim confirmed", Curve says "is the objective progressing". A unit with no
	// bound objective leaves this nil — the common case, and not a warning.
	Curve *Curve `json:"curve,omitempty"`
	// Partial, when set, is this unit's membership completeness: N of M expected
	// commits landed (see partial.go). A third orthogonal axis — Band says "was
	// each claim confirmed", Curve says "is the objective progressing", Partial
	// says "is the intent all here yet". A set Partial whose Expected is nil means
	// the denominator was NOT DERIVABLE, which is explicitly reported and is never
	// rendered complete.
	Partial *Partial `json:"partial,omitempty"`
}

// BandFor maps one witness verdict to the band it implies.
//
// An unsupplied/unknown verdict is UNVERIFIABLE, never CLEARED: "not yet
// graded" is not "confirmed". Defaulting an ungraded commit to CLEARED would
// let unwitnessed work render as safe, which is the exact failure the band
// exists to prevent.
func BandFor(v Verdict) Band {
	switch v {
	case VerdictWitnessed:
		return BandCleared
	case VerdictUnwitnessed:
		return BandResidual
	case VerdictAbstain, VerdictNoCommit, VerdictUnknown:
		return BandUnverifiable
	default:
		return BandUnverifiable
	}
}

// commitBand resolves one commit's band from its two possible inputs, and is
// the single choke point where a supplied band meets its witness verdict.
//
// The two are not peers. A supplied Band is a CACHE of an earlier fold; the
// Verdict is the machine's witness bit. So when both are present they reconcile
// PESSIMISTICALLY — the worse of the two wins — which makes the relation
// one-directional: a supplied band can only ever make a commit read WORSE than
// its witness rung allows, never better.
//
// That asymmetry is what makes the band unforgeable. An operator affordance (an
// ack, a steer) that writes a CLEARED band onto an unwitnessed commit cannot
// launder it into diff-witnessed, because the verdict still floors the result at
// RESIDUAL. A band written the other way — an operator flagging something the
// machine cleared — is allowed to stand: that is real signal, and pessimism is
// always safe.
//
// When no verdict was supplied at all, the caller's band is taken as-is: a
// caller that pre-banded its commits is reporting a fold that already happened,
// not forging one. That path is still safe, because BandFor(VerdictUnknown) is
// UNVERIFIABLE — an ungraded commit never reads as CLEARED on its own.
func commitBand(c Commit) Band {
	if c.Band == "" {
		return BandFor(c.Verdict)
	}
	if c.Verdict == VerdictUnknown {
		return c.Band
	}
	return worseBand(c.Band, BandFor(c.Verdict))
}

// worseBand returns the worse (lower-ranked) of two bands. An unrecognized band
// ranks 0 via the zero value, i.e. as bad as RESIDUAL — an unknown band fails
// safe rather than clearing anything.
func worseBand(a, b Band) Band {
	if bandRank[a] < bandRank[b] {
		return a
	}
	return b
}

// FoldBand folds member commits to the unit's band: the WORST member wins.
//
// An EMPTY unit is UNVERIFIABLE, not CLEARED. "All zero members were witnessed"
// is vacuously true and operationally a lie — nothing was shown to the operator,
// so nothing was cleared.
func FoldBand(commits []Commit) Band {
	if len(commits) == 0 {
		return BandUnverifiable
	}
	worst := BandCleared
	for _, c := range commits {
		if b := commitBand(c); bandRank[b] < bandRank[worst] {
			worst = b
		}
	}
	return worst
}

// ParseLog parses `git log --no-merges --name-only
// --format=%x1e%H%x1f%s%x1f%b%x1f` output: records split on \x1e, fields on
// \x1f, with the touched-file list trailing the final field separator.
func ParseLog(raw string) []Commit {
	var commits []Commit
	for _, record := range strings.Split(raw, "\x1e") {
		if strings.TrimSpace(record) == "" {
			continue
		}
		fields := strings.SplitN(record, "\x1f", 4)
		if len(fields) < 4 {
			continue
		}
		sha := strings.TrimSpace(fields[0])
		subject := strings.TrimSpace(fields[1])
		body := fields[2]
		if sha == "" || subject == "" {
			continue
		}
		var files []string
		for _, line := range strings.Split(fields[3], "\n") {
			if line = strings.TrimSpace(line); line != "" {
				files = append(files, line)
			}
		}
		commits = append(commits, parseCommit(sha, subject, body, files))
	}
	return commits
}

// parseCommit derives ONE commit's overlay facts from its message: the unit's
// ship-stamp leaf, the conventional-commit type, and the closure-grade vs
// mention issue refs. It is the single per-commit parser, shared by ParseLog
// (the tick's git-log path) and AssignLanded (the land-time path, #5026), so the
// unit a commit is assigned the moment it lands and the unit the tick folds it
// into cannot drift — there is no second implementation to drift from.
func parseCommit(sha, subject, body string, files []string) Commit {
	leaf := ""
	if m := leafRE.FindStringSubmatch(subject); m != nil {
		leaf = m[1]
	}
	typ := ""
	if m := typeRE.FindStringSubmatch(subject); m != nil {
		typ = m[1]
	}
	resolves := Issues(subject, nil)
	return Commit{
		SHA: sha, Subject: subject, Leaf: leaf, Type: typ,
		Resolves: resolves, Mentions: Issues(body, resolves), Files: files,
	}
}

// Issues extracts deduplicated #N refs from text, excluding any already present
// in exclude (subject-bound refs outrank body mentions).
func Issues(text string, exclude []string) []string {
	seen := map[string]bool{}
	for _, ref := range exclude {
		seen[ref] = true
	}
	var out []string
	for _, m := range issueRE.FindAllStringSubmatch(text, -1) {
		ref := "#" + m[1]
		if !seen[ref] {
			seen[ref] = true
			out = append(out, ref)
		}
	}
	sort.Strings(out)
	return out
}

// FoldUnits groups commits into one unit per (fak <leaf>) lane, banding each
// unit by its worst member. Commits without a stamp are returned separately:
// they are legibility debt, surfaced rather than dropped — the partition over
// (units, unstamped) is TOTAL and DISJOINT, so no landed commit is ever
// invisible to an operator.
//
// Units are ordered biggest-first, then by leaf (the promotion-plan order);
// the commits inside each unit read oldest-first, the way a PR body should.
// Call SortWorstFirst for the operator view's attention order.
//
// This is the LEAF-grouped fold: every unit it returns reports grouped_by leaf.
// It is `release prplan`'s fold and the overlay's fallback, and it is exactly
// FoldUnitsByWave with no wave bindings — one loop, so the two bases can never
// drift apart in anything but their key (see grouping.go, #5040).
func FoldUnits(commits []Commit) (units []Unit, unstamped []Commit) {
	return FoldUnitsByWave(commits, nil)
}

// sortBiggestFirst is the promotion-plan order: most commits first, then by
// leaf. The leaf tiebreak makes the order TOTAL, so two equal-sized units never
// swap between runs on Go's randomized map iteration.
func sortBiggestFirst(units []Unit) {
	sort.Slice(units, func(i, j int) bool {
		if len(units[i].Commits) != len(units[j].Commits) {
			return len(units[i].Commits) > len(units[j].Commits)
		}
		return units[i].Leaf < units[j].Leaf
	})
}

// SortWorstFirst orders units for the OPERATOR view: worst band first
// (RESIDUAL -> UNVERIFIABLE -> CLEARED), then biggest-first, then by leaf.
// This mirrors the superloop's worst-first walk and dos review's residual-first
// ordering: what owes attention surfaces first. The leaf tiebreak keeps the
// order total and therefore deterministic across re-ticks.
func SortWorstFirst(units []Unit) {
	sort.SliceStable(units, func(i, j int) bool {
		if bandRank[units[i].Band] != bandRank[units[j].Band] {
			return bandRank[units[i].Band] < bandRank[units[j].Band]
		}
		if len(units[i].Commits) != len(units[j].Commits) {
			return len(units[i].Commits) > len(units[j].Commits)
		}
		return units[i].Leaf < units[j].Leaf
	})
}

// Residual counts the units that owe operator attention. It is the overlay's
// headline number (osp_residual) and is deliberately INDEPENDENT of any
// operator ack: the pile falls when work gets WITNESSED, not when a human looks
// at it. Letting an ack deflate this number would launder a self-report into a
// witness.
func Residual(units []Unit) int {
	n := 0
	for _, u := range units {
		if u.Band == BandResidual {
			n++
		}
	}
	return n
}

// MergeRefs unions add into have, deduplicated and sorted.
func MergeRefs(have, add []string) []string {
	seen := map[string]bool{}
	for _, v := range have {
		seen[v] = true
	}
	for _, v := range add {
		if !seen[v] {
			seen[v] = true
			have = append(have, v)
		}
	}
	sort.Strings(have)
	return have
}

// SubtractRefs removes every ref in drop from the from list.
func SubtractRefs(from, drop []string) []string {
	gone := map[string]bool{}
	for _, v := range drop {
		gone[v] = true
	}
	var out []string
	for _, v := range from {
		if !gone[v] {
			out = append(out, v)
		}
	}
	return out
}

// UnitTitle renders a unit's title: a single-commit unit reads as its subject;
// a multi-commit unit summarizes its conventional-commit type histogram.
func UnitTitle(unit Unit) string {
	if len(unit.Commits) == 1 {
		return unit.Commits[0].Subject
	}
	types := make([]string, 0, len(unit.Types))
	for t := range unit.Types {
		types = append(types, t)
	}
	sort.Slice(types, func(i, j int) bool {
		if unit.Types[types[i]] != unit.Types[types[j]] {
			return unit.Types[types[i]] > unit.Types[types[j]]
		}
		return types[i] < types[j]
	})
	parts := make([]string, 0, len(types))
	for _, t := range types {
		parts = append(parts, fmt.Sprintf("%s %d", t, unit.Types[t]))
	}
	detail := strings.Join(parts, ", ")
	if detail == "" {
		detail = "mixed"
	}
	return fmt.Sprintf("%s: %d commits (%s)", unit.Leaf, len(unit.Commits), detail)
}
