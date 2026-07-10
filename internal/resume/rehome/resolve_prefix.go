package rehome

import (
	"fmt"
	"path/filepath"
	"strings"
)

// looksLikeFullSessionID reports whether sid is long enough to be a full Claude
// session id (a 36-char UUID, or a 32-char hex form) rather than a truncated prefix.
// The resume resolver treats a full-but-absent id as a brand-new session to land
// fresh (PIN_FRESH), but a SHORT id is a likely paste-truncation or typo and must be
// disambiguated against on-disk transcripts instead of silently pinned (#3782). The
// gate is deliberately a length floor, not a UUID-format match: a mis-classified full
// id is self-correcting (its own transcript exists, so the prefix scan resolves it
// back to itself), whereas being strict risks refusing a legitimate new session.
func looksLikeFullSessionID(sid string) bool {
	return len(sid) >= 32
}

// prefixGlobs enumerates every "<home>/.claude*/projects/*/<prefix>*.jsonl" match,
// mirroring LocateMatches' exact-id glob widened to a prefix.
func prefixGlobs(prefix, home string) []string {
	if prefix == "" {
		return nil
	}
	acctDirs, err := filepath.Glob(filepath.Join(home, ".claude*"))
	if err != nil {
		return nil
	}
	var out []string
	for _, acctDir := range acctDirs {
		files, err := filepath.Glob(filepath.Join(acctDir, "projects", "*", prefix+"*.jsonl"))
		if err != nil {
			continue
		}
		out = append(out, files...)
	}
	return out
}

// resolvePrefix scans every ~/.claude* account's projects for transcripts whose file
// name begins with prefix and returns the single matching full session id (empty when
// the count is not exactly one) and the count of DISTINCT ids matched.
func resolvePrefix(prefix, home string) (string, int) {
	seen := map[string]struct{}{}
	var only string
	for _, f := range prefixGlobs(prefix, home) {
		id := strings.TrimSuffix(filepath.Base(f), ".jsonl")
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		only = id
	}
	if len(seen) != 1 {
		only = ""
	}
	return only, len(seen)
}

// prefixCandidates returns up to limit distinct full ids a prefix matched, in glob
// order, for an AMBIGUOUS_PREFIX decision's operator-facing hint.
func prefixCandidates(prefix, home string, limit int) []string {
	seen := map[string]struct{}{}
	var out []string
	for _, f := range prefixGlobs(prefix, home) {
		id := strings.TrimSuffix(filepath.Base(f), ".jsonl")
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	return out
}

// withSID returns a copy of in with its session id replaced — used to re-run Resolve
// against a prefix-resolved full id without mutating the caller's input.
func withSID(in ResolveInput, sid string) ResolveInput {
	in.SID = sid
	return in
}

// ambiguousPrefixDecision refuses a partial id that prefix-matched more than one
// session, listing the colliding full ids so the operator can pick one.
func ambiguousPrefixDecision(in ResolveInput, home string, n int) Decision {
	cands := prefixCandidates(in.SID, home, 5)
	return Decision{
		OK: false, Action: ActionAmbiguousPrefix, Session: in.SID,
		PrefixCandidates: cands,
		Reason: fmt.Sprintf("id prefix %q matches %d sessions; pass the full session id (candidates: %s)",
			in.SID, n, strings.Join(cands, ", ")),
	}
}

// notFullIDDecision refuses a partial id that matched no transcript, rather than
// silently pinning a fresh seat for what is almost certainly a typo.
func notFullIDDecision(sid string) Decision {
	return Decision{
		OK: false, Action: ActionNotFullID, Session: sid,
		Reason: fmt.Sprintf("id %q is not a full session id and matched no transcript; "+
			"refusing to pin a fresh seat for a likely typo (pass the full session id)", sid),
	}
}
