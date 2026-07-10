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
	"sort"
	"strings"
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

// Attribute classifies every dirty hunk against the per-session checkpoint hunks.
// checkpoints maps a session id to the hunks that session's WIP checkpoint records.
// The result has one entry per input hunk (totality), sorted by (file, signature)
// for determinism. A nil/empty dirty slice yields an empty (non-nil) result.
func Attribute(dirty []Hunk, checkpoints map[string][]Hunk) []Attribution {
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
		case 0:
			a.State = AttrOrphan
		case 1:
			a.State, a.Owner = AttrOwned, owners[0]
		default:
			a.State, a.Owners = AttrShared, owners
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
