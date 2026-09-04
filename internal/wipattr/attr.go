// Package wipattr is the pure attribution core for #3874 (C2). Given the current
// dirty working-tree hunks and the per-session WIP checkpoints (refs/fak/wip/*,
// minted by #3872/#3873), it decides — for EVERY dirty hunk — whether a session
// OWNS it (that session's checkpoint records the identical edit) or it is an ORPHAN
// (no checkpoint claims it: unattributed WIP that a peer's `git add -A` sweep could
// silently destroy, the failure this epic exists to close).
//
// The fold is:
//   - total       — every input hunk yields exactly one Attribution; nothing is dropped;
//   - deterministic — output is sorted by (file, signature), sessions sorted, so two
//     runs over the same inputs are byte-identical;
//   - fail-safe    — a hunk matching no checkpoint is ORPHAN, never silently attributed;
//     a hunk claimed by >1 session is SHARED (ambiguous — a human decides), never
//     arbitrarily assigned to one.
//
// Attribution keys on the EDIT content (the +/- payload lines) within a file, NOT on
// line-number offsets: a checkpoint snapshot and the live tree can place the same
// edit at different line positions, but the edit itself is what a session owns. This
// package is pure — no git, no I/O; the cmd shell (cmd/fak) parses `git diff` into
// Hunks via ParseHunks and hands them here.
package wipattr

import (
	"crypto/sha256"
	"encoding/hex"
	"path"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// Hunk is one unified-diff hunk scoped to a file. Edit is the ordered list of the
// hunk's payload lines — those beginning with '+' or '-' (their leading sign kept),
// excluding the '+++'/'---' file headers, the '@@' hunk header, and context lines.
// Two hunks in the same file with the same Edit are "the same change".
type Hunk struct {
	File string   `json:"file"`
	Edit []string `json:"edit"`
}

// Signature is the attribution key: file + a hash of the edit payload. Hunks with
// equal Signature are attributed as the same change regardless of where they sit.
func (h Hunk) Signature() string {
	sum := sha256.Sum256([]byte(h.File + "\x00" + strings.Join(h.Edit, "\n")))
	return h.File + "@" + hex.EncodeToString(sum[:8])
}

// AttrState is the closed classification vocabulary. Every dirty hunk lands in
// exactly one state — the totality guarantee callers rely on.
type AttrState string

const (
	// AttrOwned: exactly one session's checkpoint records this edit.
	AttrOwned AttrState = "OWNED"
	// AttrOrphan: no checkpoint records this edit — unattributed, at-risk WIP.
	AttrOrphan AttrState = "ORPHAN"
	// AttrShared: more than one session's checkpoint records this edit — ambiguous
	// ownership a human must resolve; never silently collapsed to one owner.
	AttrShared AttrState = "SHARED"
)

// Attribution is the per-hunk verdict.
type Attribution struct {
	File   string    `json:"file"`
	Edit   []string  `json:"edit"`
	State  AttrState `json:"state"`
	Owner  string    `json:"owner,omitempty"`  // the sole owning session when OWNED
	Owners []string  `json:"owners,omitempty"` // all claimants (sorted) when SHARED
}

// SessionScope conveys declared scope and active lane context for a session.
type SessionScope struct {
	Session  string   `json:"session"`
	Scope    []string `json:"scope,omitempty"`
	Active   bool     `json:"active"`
	Lane     string   `json:"lane,omitempty"`
	Manifest []string `json:"manifest,omitempty"`
}

// AttributeOptions configures scope-aware and lane-aware attribution.
type AttributeOptions struct {
	Sessions      map[string]SessionScope
	LaneManifests map[string][]string
	DOSToml       []byte
}

type AttributeOption func(*AttributeOptions)

// WithSessionScope registers explicit scope and active lane context for one session.
func WithSessionScope(s SessionScope) AttributeOption {
	return func(o *AttributeOptions) {
		if o.Sessions == nil {
			o.Sessions = make(map[string]SessionScope)
		}
		o.Sessions[s.Session] = s
	}
}

// WithSessionScopes registers multiple session scopes.
func WithSessionScopes(scopes []SessionScope) AttributeOption {
	return func(o *AttributeOptions) {
		if o.Sessions == nil {
			o.Sessions = make(map[string]SessionScope)
		}
		for _, s := range scopes {
			o.Sessions[s.Session] = s
		}
	}
}

// WithActiveLane marks a session's lane active with the given manifest patterns.
func WithActiveLane(session, lane string, manifest []string) AttributeOption {
	return func(o *AttributeOptions) {
		if o.Sessions == nil {
			o.Sessions = make(map[string]SessionScope)
		}
		cur := o.Sessions[session]
		cur.Session = session
		cur.Active = true
		cur.Lane = lane
		if len(manifest) > 0 {
			cur.Manifest = manifest
		}
		o.Sessions[session] = cur
	}
}

// WithDOSToml supplies raw dos.toml configuration bytes to resolve lane patterns.
func WithDOSToml(data []byte) AttributeOption {
	return func(o *AttributeOptions) {
		o.DOSToml = data
	}
}

// WithLaneManifests sets the global lane-to-patterns taxonomy.
func WithLaneManifests(manifests map[string][]string) AttributeOption {
	return func(o *AttributeOptions) {
		o.LaneManifests = manifests
	}
}

// Attribute classifies every dirty hunk against the per-session checkpoint hunks.
// checkpoints maps a session id to the hunks that session's WIP checkpoint records.
// The result has one entry per input hunk (totality), sorted by (file, signature)
// for determinism. A nil/empty dirty slice yields an empty (non-nil) result.
// If options are supplied and a dirty hunk has no matching checkpoint signature,
// explicit scope claims and active lane patterns (from dos.toml or lane manifests)
// are consulted: if Scope is empty but session/lane is active and matches the dirty file,
// ownership is inferred (AttrOwned) rather than falling back to AttrOrphan.
func Attribute(dirty []Hunk, checkpoints map[string][]Hunk, opts ...AttributeOption) []Attribution {
	var opt AttributeOptions
	for _, fn := range opts {
		if fn != nil {
			fn(&opt)
		}
	}
	return AttributeWithOptions(dirty, checkpoints, opt)
}

// AttributeWithOptions classifies dirty hunks with explicit AttributeOptions.
func AttributeWithOptions(dirty []Hunk, checkpoints map[string][]Hunk, opts AttributeOptions) []Attribution {
	if len(opts.DOSToml) > 0 {
		parsed := ParseDOSTomlLanes(opts.DOSToml)
		if opts.LaneManifests == nil {
			opts.LaneManifests = make(map[string][]string)
		}
		for l, p := range parsed {
			if len(opts.LaneManifests[l]) == 0 {
				opts.LaneManifests[l] = p
			}
		}
	}

	// Index: signature -> set of sessions whose checkpoint contains that edit.
	claimants := make(map[string]map[string]struct{})
	for session, hunks := range checkpoints {
		for _, h := range hunks {
			sig := h.Signature()
			set := claimants[sig]
			if set == nil {
				set = make(map[string]struct{})
				claimants[sig] = set
			}
			set[session] = struct{}{}
		}
	}

	out := make([]Attribution, 0, len(dirty))
	for _, h := range dirty {
		a := Attribution{File: h.File, Edit: h.Edit}
		owners := sortedKeys(claimants[h.Signature()])
		switch len(owners) {
		case 1:
			a.State, a.Owner = AttrOwned, owners[0]
		default:
			if len(owners) > 1 {
				a.State, a.Owners = AttrShared, owners
				break
			}

			// len(owners) == 0: check Scope and Lane inference.
			normFile := pathutil.NormalizeScope(h.File)
			scopeClaimants := make(map[string]struct{})

			for sessionID, s := range opts.Sessions {
				// 1. Explicit declared / auto-bound scope:
				if len(s.Scope) > 0 {
					if scopeMatches(s.Scope, normFile) {
						scopeClaimants[sessionID] = struct{}{}
					}
					continue
				}

				// 2. If Scope is empty but session/lane is active and dos.toml
				// (or lane manifest / declared scope) matches the dirty files,
				// infer ownership (AttrOwned, owner = session) rather than falling back to AttrOrphan.
				if s.Active {
					var patterns []string
					if len(s.Manifest) > 0 {
						patterns = s.Manifest
					} else if s.Lane != "" && len(opts.LaneManifests[strings.ToLower(s.Lane)]) > 0 {
						patterns = opts.LaneManifests[strings.ToLower(s.Lane)]
					} else if len(opts.LaneManifests[strings.ToLower(sessionID)]) > 0 {
						patterns = opts.LaneManifests[strings.ToLower(sessionID)]
					}
					if patternsMatch(patterns, normFile) {
						scopeClaimants[sessionID] = struct{}{}
					}
				}
			}

			scopeOwners := sortedKeys(scopeClaimants)
			switch len(scopeOwners) {
			case 0:
				a.State = AttrOrphan
			case 1:
				a.State, a.Owner = AttrOwned, scopeOwners[0]
			default:
				a.State, a.Owners = AttrShared, scopeOwners
			}
		}
		out = append(out, a)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].File != out[j].File {
			return out[i].File < out[j].File
		}
		return signatureOf(out[i]) < signatureOf(out[j])
	})
	return out
}

// Orphans returns just the ORPHAN attributions — the at-risk set a sweep-guard
// (#3879) warns on. Convenience over Attribute; preserves its ordering.
func Orphans(as []Attribution) []Attribution {
	out := make([]Attribution, 0)
	for _, a := range as {
		if a.State == AttrOrphan {
			out = append(out, a)
		}
	}
	return out
}

func signatureOf(a Attribution) string {
	return Hunk{File: a.File, Edit: a.Edit}.Signature()
}

func sortedKeys(set map[string]struct{}) []string {
	if len(set) == 0 {
		return nil
	}
	keys := make([]string, 0, len(set))
	for k := range set {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func scopeMatches(scope []string, file string) bool {
	for _, s := range scope {
		norm := pathutil.NormalizeScope(s)
		if norm == "" {
			continue
		}
		if file == norm || strings.HasPrefix(file, norm+"/") {
			return true
		}
		if matched, _ := path.Match(norm, file); matched {
			return true
		}
	}
	return false
}

func patternsMatch(patterns []string, file string) bool {
	for _, p := range patterns {
		p = pathutil.NormalizeScope(p)
		if p == "" {
			continue
		}
		if file == p {
			return true
		}
		if strings.HasSuffix(p, "/**") {
			prefix := strings.TrimSuffix(p, "/**")
			if file == prefix || strings.HasPrefix(file, prefix+"/") {
				return true
			}
		}
		if strings.HasPrefix(file, p+"/") {
			return true
		}
		if matched, _ := path.Match(p, file); matched {
			return true
		}
	}
	return false
}

// ParseDOSTomlLanes parses the [lanes.trees] section of dos.toml bytes.
func ParseDOSTomlLanes(data []byte) map[string][]string {
	lanes := make(map[string][]string)
	if len(data) == 0 {
		return lanes
	}
	section := ""
	for _, raw := range strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n") {
		line := raw
		if i := strings.IndexByte(line, '#'); i >= 0 {
			line = line[:i]
		}
		t := strings.TrimSpace(line)
		if t == "" {
			continue
		}
		if strings.HasPrefix(t, "[") {
			section = strings.Trim(t, "[]")
			continue
		}
		if section == "lanes.trees" {
			eq := strings.IndexByte(t, '=')
			if eq < 0 {
				continue
			}
			lane := strings.ToLower(strings.TrimSpace(t[:eq]))
			if lane == "" {
				continue
			}
			lanes[lane] = append(lanes[lane], parseQuotedStrings(t[eq+1:])...)
		}
	}
	return lanes
}

func parseQuotedStrings(s string) []string {
	var out []string
	inQuote := false
	quoteChar := byte(0)
	start := -1
	for i := 0; i < len(s); i++ {
		c := s[i]
		if !inQuote && (c == '"' || c == '\'') {
			inQuote = true
			quoteChar = c
			start = i + 1
			continue
		}
		if inQuote && c == quoteChar {
			inQuote = false
			out = append(out, s[start:i])
			start = -1
		}
	}
	return out
}
