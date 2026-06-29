// Package loopmap is the loop-stage -> tool map: the in-loop affordance that answers
// "what tool do I reach for RIGHT NOW?" at each stage of the agentic-coding loop
// (orient -> plan -> act -> verify -> ship -> learn, the six stages owned by
// internal/loopindex). It is the #1153 lever of the 10x dev-experience epic (#1148):
// the loop-index scorecard measures whether the loop is getting faster; this makes the
// RIGHT verb cheap to FIND at the right moment so every other lever gets used by
// DEFAULT instead of by luck.
//
// The map is DATA, not prose — a queryable table the impure shell (cmd/fak/loopmap.go,
// the `fak loop-map` verb) prints, filters by stage, or matches against a free-text
// "what now?" situation. Each fak-kind entry names a REAL `fak` verb; the no-drift
// witness (loopmap_test.go) re-derives fak's verb registry from cmd/fak/main.go and
// fails if the map points at a verb the binary does not have — so the map cannot rot
// away from the tool surface it describes.
package loopmap

import (
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/loopindex"
)

// Kind tags where a verb lives so a reader knows which surface to reach for.
type Kind string

const (
	// KindFak is a `fak` subcommand (always present in this binary; drift-checked).
	KindFak Kind = "fak"
	// KindDos is a `dos` kernel verb (the domain-free trust substrate / MCP tools).
	KindDos Kind = "dos"
	// KindSkill is a paged Claude skill (#1103 loads these on demand).
	KindSkill Kind = "skill"
)

// Entry is one row of the map: at loop Stage, when you are in Situation, reach for
// Verb (which lives on surface Kind), because Why. Keywords drive the free-text Ask
// match; they are the words a mid-tier agent would actually type at that moment.
type Entry struct {
	Stage     string   `json:"stage"`
	Situation string   `json:"situation"`
	Verb      string   `json:"verb"`
	Kind      Kind     `json:"kind"`
	Why       string   `json:"why"`
	Keywords  []string `json:"keywords"`
}

// entries is the curated map, one or more rows per loop stage, in loop order. The
// three rows the issue (#1153) calls out by name — claim-done -> verify, fan-out ->
// arbitrate, trust-a-memory -> recall — are the verify/plan/orient anchors below.
//
// Verb-surface honesty: the issue text says "dos answer" and "dos recall", but the
// live `dos` CLI has neither (it has answer-shape, not answer; review/resume/rewind,
// not recall). So the recall anchor points at `fak recall` — a REAL, drift-checked fak
// verb — and names the dos_recall MCP tool as the kernel-side equivalent in Why,
// rather than shipping a map that sends an agent at a verb that does not exist.
var entries = []Entry{
	{
		Stage:     loopindex.StageOrient,
		Situation: "about to trust a recalled memory or a remembered fact",
		Verb:      "fak recall",
		Kind:      KindFak,
		Why:       "re-verify recalled memory against the tree at read time before you trust it — a memory reflects what was true when written; the kernel-side equivalent is the dos_recall MCP tool.",
		Keywords:  []string{"trust a memory", "recalled memory", "remember", "recall", "memory says", "from memory"},
	},
	{
		Stage:     loopindex.StagePlan,
		Situation: "about to fan out N agents / launch parallel work",
		Verb:      "dos arbitrate",
		Kind:      KindDos,
		Why:       "prove the lanes are disjoint BEFORE launching, so workers do not collide on the same files or needlessly serialize — honor a REFUSE.",
		Keywords:  []string{"fan out", "fanout", "parallel", "multiple agents", "several agents", "dispatch", "collide", "concurrent", "arbitrate"},
	},
	{
		Stage:     loopindex.StageAct,
		Situation: "about to run a tool call (and a repeat read or a malformed call would cost a turn)",
		Verb:      "fak guard",
		Kind:      KindFak,
		Why:       "front the agent with the kernel: it adjudicates each tool call, repairs a malformed call in place, and serves an identical repeat locally instead of spending a turn.",
		Keywords:  []string{"tool call", "malformed", "repair", "run a tool", "execute a tool", "denied", "adjudicate"},
	},
	{
		Stage:     loopindex.StageVerify,
		Situation: "about to claim the work is done",
		Verb:      "dos verify",
		Kind:      KindDos,
		Why:       "confirm the claim landed from git evidence, not a self-report; pair with `fak commit --preview` to pre-check the lane + ship-stamp before you commit.",
		Keywords:  []string{"claim done", "claim it works", "is done", "finished", "verify", "self-report", "did it land", "confirm done"},
	},
	{
		Stage:     loopindex.StageShip,
		Situation: "the tree is green and you are about to commit/ship",
		Verb:      "fak commit",
		Kind:      KindFak,
		Why:       "commit-by-path under the green gate (refuses OFF_TRUNK / PATHSPEC_RACE, never `git add -A`); then `dos commit-audit <sha>` grades the commit diff-witnessed.",
		Keywords:  []string{"tree is green", "ship", "commit", "push", "land it", "green gate", "make ci passed"},
	},
	{
		Stage:     loopindex.StageLearn,
		Situation: "the session is over — capture its cost and outcome",
		Verb:      "fak sessions",
		Kind:      KindFak,
		Why:       "ingest + score this host's transcripts so the session -> outcome loop (helped / wash / hurt) learns from the run instead of forgetting it.",
		Keywords:  []string{"session over", "session is over", "capture", "outcome", "learn", "transcript", "reflect", "what happened"},
	},
}

// Map returns the whole loop-stage -> tool map in loop order (a copy; callers must not
// mutate the package data).
func Map() []Entry {
	out := make([]Entry, len(entries))
	copy(out, entries)
	return out
}

// Stages returns the six canonical loop stages, in loop order — bound to the
// loopindex spine so a renamed stage there fails to compile here.
func Stages() []string {
	return []string{
		loopindex.StageOrient, loopindex.StagePlan, loopindex.StageAct,
		loopindex.StageVerify, loopindex.StageShip, loopindex.StageLearn,
	}
}

// ForStage returns every entry at the given loop stage, in map order. An unknown stage
// returns nil.
func ForStage(stage string) []Entry {
	var out []Entry
	for _, e := range entries {
		if e.Stage == stage {
			out = append(out, e)
		}
	}
	return out
}

// Ask matches a free-text "what tool do I reach for right now?" situation to the best
// map entry by keyword overlap (the thin lexical lookup the issue asks for — a 6-way
// stage classifier, not a new oracle). It returns the highest-scoring entry and true,
// or a zero Entry and false when nothing matches. Ties resolve to loop order, so the
// earliest stage wins a draw.
func Ask(situation string) (Entry, bool) {
	q := strings.ToLower(situation)
	best := -1
	bestIdx := -1
	for i, e := range entries {
		score := 0
		for _, kw := range e.Keywords {
			if strings.Contains(q, strings.ToLower(kw)) {
				score++
			}
		}
		if score > best {
			best = score
			bestIdx = i
		}
	}
	if best <= 0 || bestIdx < 0 {
		return Entry{}, false
	}
	return entries[bestIdx], true
}

// FakVerbs returns the set of bare `fak` subcommand names the map references (the first
// token after "fak " of every KindFak verb). The no-drift witness checks each against
// the real registry parsed from cmd/fak/main.go.
func FakVerbs() []string {
	seen := map[string]struct{}{}
	for _, e := range entries {
		if e.Kind != KindFak {
			continue
		}
		fields := strings.Fields(e.Verb)
		if len(fields) >= 2 && fields[0] == "fak" {
			seen[fields[1]] = struct{}{}
		}
	}
	out := make([]string, 0, len(seen))
	for v := range seen {
		out = append(out, v)
	}
	sort.Strings(out)
	return out
}

// VerifyNudge is the "did you verify?" reminder surfaced at the verify/ship boundary —
// the one-line prompt that turns a silent self-report into a checked claim.
func VerifyNudge() string {
	return "Did you verify? Before you claim done, run `dos verify` (claim from git evidence, not a self-report) and `fak commit --preview` (lane + ship-stamp pre-check)."
}
