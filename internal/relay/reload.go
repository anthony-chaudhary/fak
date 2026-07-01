// Rung D1 (issue #1877): the reload verifier. When a successor leg picks up a baton, the
// ProgressCursor is the least-trusted half — a set of anchors the successor must re-read
// against ground truth before believing any progress (baton.go's no-`claimed` invariant).
// This file re-checks the cursor against git via the C3 Resolver and reports fresh (the
// anchor still resolves) or stale (it has diverged). It does NOT re-derive a stale cursor
// or emit the typed stale outcome — those are later rungs (#1878 emits RELAY_BATON_STALE;
// D2 re-derives).
package relay

// ReloadVerdict is the two-state outcome of re-checking a ProgressCursor against git.
type ReloadVerdict string

const (
	// ReloadFresh means the cursor's ground-truth anchor still resolves — the successor
	// may trust the cursor and continue.
	ReloadFresh ReloadVerdict = "fresh"
	// ReloadStale means the anchor no longer resolves (diverged), OR could not be verified
	// at all. It is emitted fail-closed: an unverifiable anchor is never reported fresh, so
	// the successor re-derives from durable state rather than trusting a cursor git cannot
	// confirm.
	ReloadStale ReloadVerdict = "stale"
)

// ReloadResult is the typed verdict plus the deciding evidence: which cursor field drove
// the call and a human-readable reason (never consumed as progress).
type ReloadResult struct {
	Verdict ReloadVerdict `json:"verdict"`
	Anchor  string        `json:"anchor"`
	Reason  string        `json:"reason"`
}

// VerifyReload re-checks cur against git through the C3 Resolver and reports fresh|stale.
// The start_sha is the ground-truth anchor: it must still resolve to a commit for the
// cursor to be fresh. A missing anchor, a diverged (dangling) anchor, or an unreachable
// store all yield stale — fail-closed, since none of them proves the cursor still matches
// reality. Deeper re-checks of ledger_ref/held_region are a later rung; this rung pins the
// git anchor that dos_status/dos_verify verify against.
func VerifyReload(cur ProgressCursor, r Resolver) ReloadResult {
	if cur.StartSHA == "" {
		return ReloadResult{Verdict: ReloadStale, Anchor: "start_sha", Reason: "progress_cursor.start_sha is empty; no ground-truth anchor to verify against git"}
	}
	res := r.Resolve(Artifact{Kind: string(ArtifactCommit), Ref: cur.StartSHA})
	switch res.Verdict {
	case ResolveVerified:
		return ReloadResult{Verdict: ReloadFresh, Anchor: "start_sha", Reason: "start_sha resolves in git: " + cur.StartSHA}
	case ResolveDangling:
		return ReloadResult{Verdict: ReloadStale, Anchor: "start_sha", Reason: "start_sha has diverged from git: " + res.Detail}
	default: // ResolveUnknown — store unreachable; fail closed rather than trust it.
		return ReloadResult{Verdict: ReloadStale, Anchor: "start_sha", Reason: "start_sha could not be verified against git: " + res.Detail}
	}
}
