package commitrollup

import (
	"fmt"
	"path"
	"regexp"
	"sort"
	"strings"
)

// Schema identifies the machine-readable planner envelope.
const Schema = "fak.commit-rollup/v1"

// Reason is a closed refusal token emitted by the pure planner.
type Reason string

const (
	ReasonMissingID         Reason = "MISSING_ID"
	ReasonMissingPathset    Reason = "MISSING_PATHSET"
	ReasonInvalidPath       Reason = "INVALID_PATH"
	ReasonMissingStamp      Reason = "MISSING_STAMP"
	ReasonInvalidStamp      Reason = "INVALID_STAMP"
	ReasonStaleInput        Reason = "STALE_INPUT"
	ReasonRefusedInput      Reason = "REFUSED_INPUT"
	ReasonOverlappingPath   Reason = "OVERLAPPING_PATH"
	ReasonIncompatibleStamp Reason = "INCOMPATIBLE_STAMP"
	ReasonRollupDisabled    Reason = "ROLLUP_DISABLED"
	ReasonPathsetMismatch   Reason = "PATHSET_MISMATCH"
)

// Intent is the minimal local shape the #1788 commit-intent drain can later map
// into. Stamp accepts either a leaf ("gateway") or a trailer ("(fak gateway)").
type Intent struct {
	ID        string   `json:"id"`
	Submitter string   `json:"submitter,omitempty"`
	Paths     []string `json:"paths"`
	Stamp     string   `json:"stamp"`

	Stale         bool     `json:"stale,omitempty"`
	Refused       bool     `json:"refused,omitempty"`
	RefusedReason Reason   `json:"refused_reason,omitempty"`
	Witnesses     []string `json:"witnesses,omitempty"`
}

// Config controls planner behavior. The zero value enables rollup.
type Config struct {
	// DisableRollup keeps one-intent-at-a-time behavior while still returning
	// typed bounces for every intent that was not allowed into the next commit.
	DisableRollup bool
}

// Plan is the pure rollup decision for the next commit.
type Plan struct {
	Schema        string    `json:"schema"`
	OK            bool      `json:"ok"`
	RollupEnabled bool      `json:"rollup_enabled"`
	Stamp         string    `json:"stamp,omitempty"`
	Trailer       string    `json:"trailer,omitempty"`
	Subject       string    `json:"subject,omitempty"`
	IntentIDs     []string  `json:"intent_ids"`
	Submitters    []string  `json:"submitters,omitempty"`
	UnionPaths    []string  `json:"union_paths"`
	Witnesses     []string  `json:"witnesses,omitempty"`
	Refusals      []Refusal `json:"refusals,omitempty"`
}

// Refusal records why a single input intent could not join the planned batch.
type Refusal struct {
	IntentID string   `json:"intent_id,omitempty"`
	Reason   Reason   `json:"reason"`
	Detail   string   `json:"detail,omitempty"`
	Paths    []string `json:"paths,omitempty"`
	Stamp    string   `json:"stamp,omitempty"`
}

// PathsetAssertion is a pure equality witness for the later impure commit step:
// the committed pathset must equal the planner's union, with no hidden expansion.
type PathsetAssertion struct {
	OK       bool     `json:"ok"`
	Reason   Reason   `json:"reason,omitempty"`
	Expected []string `json:"expected"`
	Actual   []string `json:"actual"`
	Missing  []string `json:"missing,omitempty"`
	Extra    []string `json:"extra,omitempty"`
}

var stampLeafRe = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*$`)

// PlanBatch greedily builds the next compatible rollup batch from queued intents.
// It is deterministic, side-effect free, and never reads git.
func PlanBatch(intents []Intent, cfg Config) Plan {
	p := Plan{
		Schema:        Schema,
		RollupEnabled: !cfg.DisableRollup,
		IntentIDs:     []string{},
		UnionPaths:    []string{},
	}

	pathOwners := map[string]string{}
	submitters := map[string]bool{}
	witnesses := map[string]bool{}

	for _, in := range intents {
		intentID := strings.TrimSpace(in.ID)
		stamp, stampRefusal := normalizeStamp(in.Stamp)
		paths, pathRefusal := normalizePaths(in.Paths)

		switch {
		case intentID == "":
			p.Refusals = append(p.Refusals, Refusal{
				Reason: ReasonMissingID,
				Detail: "commit intents must carry a stable id before they can be rolled up",
				Paths:  paths,
				Stamp:  stamp,
			})
			continue
		case in.Stale:
			p.Refusals = append(p.Refusals, Refusal{
				IntentID: intentID,
				Reason:   ReasonStaleInput,
				Detail:   "stale intent refused before rollup compatibility is considered",
				Paths:    paths,
				Stamp:    stamp,
			})
			continue
		case in.Refused:
			reason := in.RefusedReason
			if reason == "" {
				reason = ReasonRefusedInput
			}
			p.Refusals = append(p.Refusals, Refusal{
				IntentID: intentID,
				Reason:   reason,
				Detail:   "input was already refused upstream",
				Paths:    paths,
				Stamp:    stamp,
			})
			continue
		case stampRefusal != "":
			p.Refusals = append(p.Refusals, Refusal{
				IntentID: intentID,
				Reason:   stampRefusal,
				Detail:   "intent has no witness-gradeable (fak <leaf>) stamp",
				Paths:    paths,
				Stamp:    strings.TrimSpace(in.Stamp),
			})
			continue
		case pathRefusal != "":
			p.Refusals = append(p.Refusals, Refusal{
				IntentID: intentID,
				Reason:   pathRefusal,
				Detail:   "intent must name at least one repo-relative path",
				Stamp:    stamp,
			})
			continue
		}

		if len(p.IntentIDs) > 0 && cfg.DisableRollup {
			p.Refusals = append(p.Refusals, Refusal{
				IntentID: intentID,
				Reason:   ReasonRollupDisabled,
				Detail:   "rollup disabled; only the first valid intent may enter this plan",
				Paths:    paths,
				Stamp:    stamp,
			})
			continue
		}
		if p.Stamp != "" && stamp != p.Stamp {
			p.Refusals = append(p.Refusals, Refusal{
				IntentID: intentID,
				Reason:   ReasonIncompatibleStamp,
				Detail:   fmt.Sprintf("intent stamp (fak %s) cannot share batch stamp (fak %s)", stamp, p.Stamp),
				Paths:    paths,
				Stamp:    stamp,
			})
			continue
		}
		if owner, conflictPath, ok := firstPathConflict(paths, pathOwners); ok {
			p.Refusals = append(p.Refusals, Refusal{
				IntentID: intentID,
				Reason:   ReasonOverlappingPath,
				Detail:   fmt.Sprintf("path %q overlaps intent %q", conflictPath, owner),
				Paths:    paths,
				Stamp:    stamp,
			})
			continue
		}

		if p.Stamp == "" {
			p.Stamp = stamp
			p.Trailer = "(fak " + stamp + ")"
		}
		p.IntentIDs = append(p.IntentIDs, intentID)
		for _, path := range paths {
			pathOwners[path] = intentID
		}
		if s := strings.TrimSpace(in.Submitter); s != "" {
			submitters[s] = true
		}
		for _, w := range in.Witnesses {
			if w = strings.TrimSpace(w); w != "" {
				witnesses[w] = true
			}
		}
	}

	p.OK = len(p.IntentIDs) > 0
	p.UnionPaths = sortedStringKeys(pathOwners)
	p.Submitters = sortedKeys(submitters)
	p.Witnesses = sortedKeys(witnesses)
	if p.OK {
		p.Subject = subjectFor(p.Stamp, p.IntentIDs)
	}
	return p
}

// AssertPathset compares the planned union with the paths the commit layer says
// it actually committed.
func AssertPathset(expected, actual []string) PathsetAssertion {
	exp, _ := normalizePaths(expected)
	act, _ := normalizePaths(actual)
	expSet := sliceSet(exp)
	actSet := sliceSet(act)
	missing := difference(expSet, actSet)
	extra := difference(actSet, expSet)
	ok := len(missing) == 0 && len(extra) == 0
	out := PathsetAssertion{
		OK:       ok,
		Expected: sortedKeys(expSet),
		Actual:   sortedKeys(actSet),
		Missing:  missing,
		Extra:    extra,
	}
	if !ok {
		out.Reason = ReasonPathsetMismatch
	}
	return out
}

// AssertPathset compares this plan's union with the actual committed pathset.
func (p Plan) AssertPathset(actual []string) PathsetAssertion {
	return AssertPathset(p.UnionPaths, actual)
}

func subjectFor(stamp string, ids []string) string {
	return fmt.Sprintf("chore(%s): roll up commit intents %s (fak %s)", stamp, strings.Join(ids, ", "), stamp)
}

func normalizeStamp(raw string) (string, Reason) {
	s := strings.TrimSpace(raw)
	if s == "" {
		return "", ReasonMissingStamp
	}
	if strings.HasPrefix(s, "(fak ") && strings.HasSuffix(s, ")") {
		s = strings.TrimSuffix(strings.TrimPrefix(s, "(fak "), ")")
	}
	if strings.HasPrefix(s, "fak ") {
		s = strings.TrimPrefix(s, "fak ")
	}
	s = strings.TrimSpace(s)
	if s == "" {
		return "", ReasonMissingStamp
	}
	if !stampLeafRe.MatchString(s) {
		return s, ReasonInvalidStamp
	}
	return s, ""
}

func normalizePaths(paths []string) ([]string, Reason) {
	seen := map[string]bool{}
	for _, p := range paths {
		n, reason := normalizePath(p)
		if reason != "" {
			return nil, reason
		}
		if n == "" {
			continue
		}
		seen[n] = true
	}
	if len(seen) == 0 {
		return nil, ReasonMissingPathset
	}
	return sortedKeys(seen), ""
}

func normalizePath(p string) (string, Reason) {
	p = strings.TrimSpace(strings.ReplaceAll(p, "\\", "/"))
	if strings.ContainsRune(p, '\x00') || strings.HasPrefix(p, "/") || strings.Contains(p, ":") {
		return "", ReasonInvalidPath
	}
	for strings.HasPrefix(p, "./") {
		p = strings.TrimPrefix(p, "./")
	}
	if p == "" {
		return "", ""
	}
	clean := path.Clean(p)
	if clean == "." {
		return "", ""
	}
	if clean == ".." || strings.HasPrefix(clean, "../") || clean == ".git" || strings.HasPrefix(clean, ".git/") {
		return "", ReasonInvalidPath
	}
	return clean, ""
}

func firstPathConflict(paths []string, owners map[string]string) (owner, conflictPath string, ok bool) {
	plannedPaths := sortedStringKeys(owners)
	for _, candidate := range paths {
		for _, planned := range plannedPaths {
			if pathsOverlap(candidate, planned) {
				return owners[planned], candidate, true
			}
		}
	}
	return "", "", false
}

func pathsOverlap(a, b string) bool {
	return a == b || strings.HasPrefix(a, b+"/") || strings.HasPrefix(b, a+"/")
}

func sliceSet(vals []string) map[string]bool {
	out := make(map[string]bool, len(vals))
	for _, v := range vals {
		out[v] = true
	}
	return out
}

func difference(left, right map[string]bool) []string {
	var out []string
	for k := range left {
		if !right[k] {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func sortedStringKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
