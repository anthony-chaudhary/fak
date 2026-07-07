package trajctl

// phasecommits.go — issue #3129, the producer half of the trajectory-control
// spine (epic #2533): assemble the phase→commit bindings a turn-end scoring pass
// needs from the ONE durable, forgery-resistant record a session already writes —
// the commit trailer.
//
// The dogfood (docs/notes/TRAJCTL-DOGFOOD-2026-07-07.md) surfaced the gap: the
// turn-end sampler (turnend.go / trajctlhook.RunTurnEnd) is wired to run every
// turn, but nothing assembles EvidenceWindow.PhaseCommits, so the W3
// witnessed-commit scorer sees an empty map and scores 0 forever. The dogfood
// hand-built the bindings from `git log`; this is the shipped fold that does it.
//
// The binding convention is a dedicated commit trailer:
//
//	Trajctl-Phase: <phase-id>
//
// A commit that lands a plan phase names that phase in this trailer, exactly as
// it already names its leaf with `(fak <leaf>)`. The trailer is the phase→commit
// edge; the commit SHA is the witness the W3 scorer re-resolves through the
// injected git resolver. Keeping the parse pure (pre-read commits in, map out)
// holds this package tier-1: the impure `git log` walk that reads the commits
// lives at the call site (internal/trajctlhook), the same seam GitEvidenceResolver
// shells from.

import "strings"

// PhaseTrailerKey is the commit-trailer key that binds a commit to a plan phase.
// A trailer line `Trajctl-Phase: <phase-id>` in a commit message declares that the
// commit lands the named phase; the turn-end W3 scorer then credits that phase
// once the commit SHA still resolves. The key is matched case-insensitively so a
// hand-typed `trajctl-phase:` still binds.
const PhaseTrailerKey = "Trajctl-Phase"

// TrailerCommit is one pre-read commit fact: its SHA and full message (subject +
// body). The caller reads these via `git log`; this package touches no git
// plumbing of its own, so the fold stays deterministic and tier-1.
type TrailerCommit struct {
	SHA     string
	Message string
}

// PhaseCommitsFromTrailers folds pre-read commits into the phase→commit map an
// EvidenceWindow carries. For each commit it reads every `Trajctl-Phase: <id>`
// trailer line and appends the commit's SHA to that phase's candidate list, in
// commit input order. A commit may name more than one phase (a bundled ship binds
// each named phase); a phase may be named by more than one commit (all candidates
// are kept, and the W3 scorer credits the phase once ANY of them resolves). A
// commit with no phase trailer, an empty SHA, or an empty phase id contributes
// nothing.
//
// The result is exactly the shape EvidenceWindow.PhaseCommits and
// trajctlhook.WindowInput.PhaseCommits expect: map[phaseID][]sha. It is nil when
// no commit binds any phase, so a window with no bindings scores 0 rather than
// crediting unverified work — the fail-closed rung the scorer relies on.
func PhaseCommitsFromTrailers(commits []TrailerCommit) map[string][]string {
	var out map[string][]string
	for _, c := range commits {
		sha := strings.TrimSpace(c.SHA)
		if sha == "" {
			continue
		}
		for _, phase := range phaseIDsFromMessage(c.Message) {
			if out == nil {
				out = map[string][]string{}
			}
			out[phase] = append(out[phase], sha)
		}
	}
	return out
}

// phaseIDsFromMessage returns the phase ids named by every `Trajctl-Phase:`
// trailer line in a commit message, in line order, de-duplicated within the one
// commit (a commit naming the same phase twice binds it once). Matching is
// line-oriented and case-insensitive on the key; the value is the trimmed text
// after the first colon. An empty value is skipped.
func phaseIDsFromMessage(message string) []string {
	var ids []string
	seen := map[string]bool{}
	for _, raw := range strings.Split(message, "\n") {
		line := strings.TrimSpace(raw)
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		if !strings.EqualFold(strings.TrimSpace(key), PhaseTrailerKey) {
			continue
		}
		id := strings.TrimSpace(value)
		if id == "" || seen[id] {
			continue
		}
		seen[id] = true
		ids = append(ids, id)
	}
	return ids
}
