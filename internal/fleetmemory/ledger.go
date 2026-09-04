// Package fleetmemory is the cross-agent lessons ledger (#2141) and its
// write-time duplicate guard (#2142).
//
// Invariant: fleet memory injection is fail-closed and deterministic across all session queries.
// Guard: empty candidate facts fail closed and are refused immediately without mutating the ledger.
//
// Auto-memory is per-agent-store: when one agent learns a workaround ("Bash
// `git` hangs here -> use PowerShell"), every OTHER agent re-discovers it the
// hard way. The repo's own memory files (bash_git_gh_hang_use_powershell,
// wsl_go_test_capture_technique, ...) are exactly this re-learned tax, frozen
// per-store. This package makes a lesson publishable ONCE with a trigger context
// and injectable to any peer whose session matches that trigger — before the
// peer hits the wall.
//
// The core is pure and deterministic: New folds a slice of lessons into a
// key-indexed Ledger, Match answers "does a canonical entry already cover this
// fact?", and Inject selects the lessons whose Trigger matches a session
// context. Read-time re-verification (the dos_recall discipline — a stale lesson
// is withheld, not asserted) is the CALLER's job; Inject only selects by
// trigger, it never vouches for freshness. It is stdlib-only and off the hot
// path.
//
// factKey is the dedup key both Match (#2141) and Publish (#2142) share; it
// mirrors the normalized marker-key pattern internal/issuecohort uses for
// duplicate issues, so the fleet gets one dedup notion, not two.
package fleetmemory

import (
	"sort"
	"strings"
)

// Trigger is the context that makes a lesson relevant to a peer. Every field is
// a wildcard when empty: an all-empty Trigger injects into every session (a
// universal lesson), and a field that is set narrows the match. Match/Inject use
// AND across the SET fields only.
type Trigger struct {
	Host         string   `json:"host,omitempty"`          // host the lesson applies to ("" = any host)
	PathGlobs    []string `json:"path_globs,omitempty"`    // repo-relative globs ("" set = any path)
	Tool         string   `json:"tool,omitempty"`          // tool name, e.g. "Bash" ("" = any tool)
	RefusalToken string   `json:"refusal_token,omitempty"` // dos refusal token, e.g. "OFF_TRUNK" ("" = any)
}

// Lesson is one published, cross-agent lesson.
//
// Witness is the honesty bit: a pointer to the evidence behind the lesson (a
// commit SHA, a memory slug, a test). A lesson with no Witness is a claimed-
// without-witness lesson — still stored, but a re-verify caller can weight it
// accordingly, the same split closurerate draws for closes.
type Lesson struct {
	ID         string  `json:"id"`                   // stable identifier / slug
	Fact       string  `json:"fact"`                 // the human-readable lesson
	Trigger    Trigger `json:"trigger"`              // when to inject it
	Witness    string  `json:"witness,omitempty"`    // pointer to evidence (commit/memory/test)
	Generation string  `json:"generation,omitempty"` // optional gen classification
}

// SessionContext is what a starting peer session knows about itself, matched
// against each lesson's Trigger.
type SessionContext struct {
	Host         string
	Paths        []string
	Tool         string
	RefusalToken string
}

// Ledger is a set of canonical lessons indexed by factKey. Build it with New;
// it is read-only after construction except through Publish, which returns a new
// entry or a DUP_LESSON refusal (see dedup.go).
type Ledger struct {
	lessons []Lesson
	byKey   map[string]int // factKey(fact) -> index into lessons (first writer wins)
}

// New folds a slice of lessons into a key-indexed Ledger. The first lesson to
// claim a given factKey is canonical; a later equivalent is not indexed (it
// would have been refused at write time by Publish). The input is not mutated.
func New(lessons []Lesson) *Ledger {
	l := &Ledger{
		lessons: append([]Lesson(nil), lessons...),
		byKey:   make(map[string]int, len(lessons)),
	}
	for i, les := range l.lessons {
		k := factKey(les.Fact)
		if k == "" {
			continue
		}
		if _, seen := l.byKey[k]; !seen {
			l.byKey[k] = i
		}
	}
	return l
}

// Lessons returns a copy of the ledger's canonical entries, sorted by ID for a
// deterministic dump.
func (l *Ledger) Lessons() []Lesson {
	out := append([]Lesson(nil), l.lessons...)
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Len is the number of entries in the ledger.
func (l *Ledger) Len() int { return len(l.lessons) }

// Match returns the canonical lesson whose fact is equivalent to the proposed
// fact, and true, when the ledger already covers it. An empty/whitespace-only
// fact never matches. This is the queryable set #2142 dedupes writes against.
func (l *Ledger) Match(fact string) (Lesson, bool) {
	k := factKey(fact)
	if k == "" {
		return Lesson{}, false
	}
	if i, ok := l.byKey[k]; ok {
		return l.lessons[i], true
	}
	return Lesson{}, false
}

// Inject returns the lessons whose Trigger matches the session context — the
// auto-inject side of #2141 — sorted by ID for determinism. It selects by
// trigger only; the caller is responsible for read-time re-verification before
// asserting any returned lesson (a stale lesson must be withheld, not trusted).
func (l *Ledger) Inject(ctx SessionContext) []Lesson {
	var out []Lesson
	for _, les := range l.lessons {
		if les.Trigger.Matches(ctx) {
			out = append(out, les)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Matches reports whether the trigger fires for the given session context. Every
// unset field is a wildcard; a set field must agree with the session. PathGlobs
// fire when ANY glob matches ANY session path.
func (t Trigger) Matches(ctx SessionContext) bool {
	if t.Host != "" && !strings.EqualFold(t.Host, ctx.Host) {
		return false
	}
	if t.Tool != "" && !strings.EqualFold(t.Tool, ctx.Tool) {
		return false
	}
	if t.RefusalToken != "" && !strings.EqualFold(t.RefusalToken, ctx.RefusalToken) {
		return false
	}
	if len(t.PathGlobs) > 0 && !anyGlobMatches(t.PathGlobs, ctx.Paths) {
		return false
	}
	return true
}

// factKey normalizes a fact to its dedup key: lower-cased, punctuation dropped,
// whitespace collapsed to single spaces. Two facts that differ only in casing,
// spacing, or trailing punctuation share a key and are duplicates — the same
// normalized-marker notion internal/issuecohort uses for duplicate issues.
func factKey(fact string) string {
	var b strings.Builder
	prevSpace := false
	for _, r := range strings.ToLower(strings.TrimSpace(fact)) {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			b.WriteRune(r)
			prevSpace = false
		default:
			// Any non-alphanumeric rune collapses to a single separating space.
			if !prevSpace && b.Len() > 0 {
				b.WriteByte(' ')
				prevSpace = true
			}
		}
	}
	return strings.TrimSpace(b.String())
}

// anyGlobMatches reports whether any glob matches any path. Globs support a
// leading/trailing "*" wildcard (the common "internal/x/**" / "*.go" shapes);
// an exact string matches an exact path.
func anyGlobMatches(globs, paths []string) bool {
	for _, g := range globs {
		g = strings.TrimSpace(g)
		if g == "" {
			continue
		}
		for _, p := range paths {
			if globMatch(g, strings.TrimSpace(p)) {
				return true
			}
		}
	}
	return false
}

// globMatch is a small prefix/suffix/substring glob: "a/*" prefix, "*.go"
// suffix, "*x*" substring, "**" or "*" any, else exact. Deliberately minimal —
// a trigger glob is a coarse relevance filter, not a full path matcher.
func globMatch(glob, path string) bool {
	glob = strings.ReplaceAll(glob, "**", "*")
	switch {
	case glob == "*" || glob == "":
		return true
	case strings.HasPrefix(glob, "*") && strings.HasSuffix(glob, "*") && len(glob) > 1:
		return strings.Contains(path, strings.Trim(glob, "*"))
	case strings.HasSuffix(glob, "*"):
		return strings.HasPrefix(path, strings.TrimSuffix(glob, "*"))
	case strings.HasPrefix(glob, "*"):
		return strings.HasSuffix(path, strings.TrimPrefix(glob, "*"))
	default:
		return glob == path
	}
}
