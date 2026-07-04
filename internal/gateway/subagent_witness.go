package gateway

import (
	"regexp"
	"strings"
)

// subagent_witness.go — child of #2438 (harness-native program #2387): extend the
// loop-body witness (adjudicate_proposed.go, ReasonLoopBodyUnwitnessed) to the SUBAGENT
// fold boundary. A subagent's terminal result is the same forgeable self-report one
// level up: a rate-limited or cut-off child can return an empty "success" the parent
// folds clean. So a child result whose prose CLAIMS ship/create/fix must carry an
// artifact witness — a commit SHA (the marker `git commit` prints), a file hash, or a
// registry id — before the parent folds it clean. An unwitnessed 'done' claim is
// admitted as TAINTED: the content is still forwarded (the parent sees it) but its
// ResultAdmission verdict is demoted to RESIDUAL/LOOP_DONE_UNWITNESSED, mirroring
// dos_status's no-claimed-field contract — the claim arrives visibly unverified.
//
// This reuses the witness discipline, not new machinery: the commit-marker pattern is
// the one internal/sessionobs.commitMarker (ingest.go) extracts into a session's
// Evidence.CommitSHAs, and the reason token is the SAME closed-vocabulary
// LOOP_DONE_UNWITNESSED the loop-body witness already cites (dos.toml [reasons.*]).
//
// HONEST LIMIT: this gates on the STRUCTURAL presence of an artifact-witness token in
// the child's result — the same altitude as the loop-body witness (which checks for an
// effect-capable CALL, not that the call succeeded). Deep reachability (that the SHA is
// actually an ancestor in git, dos_verify's registry/grep ladder) is the costlier rung
// the fold references, not one run inline on the proxy hot path.

// subagentFoldBy names the gate that renders a subagent-fold demotion (forensics) — the
// subagent-boundary counterpart of adjudicate_proposed's "loop-body-witness".
const subagentFoldBy = "subagent-fold-witness"

// subagentSpawnShapes are the tool-name shapes of a subagent-spawn tool (case-insensitive
// substring), the boundary across which a child's terminal result is folded into the
// parent. Coarse by design — the same tool-name-shape approach turnHasEffectCapableCall
// uses — because the OpenAI-compatible proxy is provider-agnostic and a child spawn is
// not one fixed tool name. A false positive only demotes an UNBACKED done-claim to
// RETRYABLE (content still forwarded), the conservative direction.
var subagentSpawnShapes = []string{"task", "agent", "subagent", "dispatch", "spawn"}

// subagentSpawnTool reports whether a tool name looks like a subagent-spawn tool.
func subagentSpawnTool(tool string) bool {
	return containsAnySubstring(strings.ToLower(tool), subagentSpawnShapes)
}

// doneClaim pairs a space-padded completion-claim phrase with the coarse claim KIND it
// asserts. An ordered slice (not a map) so a multi-phrase result resolves deterministically.
type doneClaim struct{ phrase, kind string }

// subagentDoneClaims are the "ship/create/fix" completion-claim shapes a child result's
// prose carries when it asserts it finished effect-bearing work. Phrases are space-padded
// so a bare substring (e.g. "fixed" inside "prefixed") does not trip the net.
var subagentDoneClaims = []doneClaim{
	{" shipped ", "ship"},
	{" committed ", "ship"},
	{" merged ", "ship"},
	{" landed ", "ship"},
	{" pushed ", "ship"},
	{" created ", "create"},
	{" wrote ", "create"},
	{" added ", "create"},
	{" generated ", "create"},
	{" fixed ", "fix"},
	{" resolved ", "fix"},
	{" patched ", "fix"},
	{" implemented ", "fix"},
	{" done ", "done"},
	{" completed ", "done"},
}

// claimedDoneKind returns the coarse completion-claim kind a child result asserts, or ""
// if it makes no such claim. Normalized like turnBodyClaimsCompletedEdit: collapse
// whitespace, lowercase, and pad with a space so the phrases match on word boundaries.
func claimedDoneKind(content string) string {
	body := " " + strings.ToLower(strings.Join(strings.Fields(content), " ")) + " "
	if strings.TrimSpace(body) == "" {
		return ""
	}
	for _, dc := range subagentDoneClaims {
		if strings.Contains(body, dc.phrase) {
			return dc.kind
		}
	}
	return ""
}

// commitMarkerPat matches the line `git commit` prints on success — "[main 809c5339]
// subject". Byte-identical to internal/sessionobs.commitMarker (that symbol is
// unexported, so the pattern is mirrored here rather than imported; the canonical source
// is cited in the file header). The bracket+SHA context makes it a specific witness.
var commitMarkerPat = regexp.MustCompile(`\[[^\]]*?\b[0-9a-f]{7,40}\]`)

// fullHashPat matches a standalone full git SHA (40 hex) or a sha256 file hash (64 hex),
// word-bounded so it is not a substring of a longer token. These are the "commit SHA /
// file hash" artifact witnesses at their specific, non-ambiguous lengths.
var fullHashPat = regexp.MustCompile(`\b([0-9a-f]{40}|[0-9a-f]{64})\b`)

// resultCarriesArtifactWitness reports whether a child result carries a non-forgeable
// artifact reference — a git-commit success marker, a full SHA / sha256 file hash, or an
// explicit sha256:/sha1: hash prefix — that backs a completion claim. Present ⇒ the
// parent may fold the claim clean; absent ⇒ the claim is self-report only.
func resultCarriesArtifactWitness(content string) bool {
	if content == "" {
		return false
	}
	if commitMarkerPat.MatchString(content) {
		return true
	}
	lc := strings.ToLower(content)
	if strings.Contains(lc, "sha256:") || strings.Contains(lc, "sha1:") {
		return true
	}
	return fullHashPat.MatchString(lc)
}

// subagentDoneVerdict decides whether a subagent's terminal result must be folded TAINTED
// rather than clean. It returns the demoted verdict and true only when ALL hold: the base
// admission was a clean ALLOW (a QUARANTINE/DENY already held it, nothing to demote), the
// tool is a subagent-spawn shape, the result CLAIMS ship/create/fix, and the result
// carries NO artifact witness. Otherwise it returns the base verdict unchanged and false.
func subagentDoneVerdict(tool, content string, base WireVerdict) (WireVerdict, bool) {
	if base.Kind != "ALLOW" {
		return base, false
	}
	if !subagentSpawnTool(tool) {
		return base, false
	}
	kind := claimedDoneKind(content)
	if kind == "" {
		return base, false
	}
	if resultCarriesArtifactWitness(content) {
		return base, false
	}
	return WireVerdict{
		Kind:        "RESIDUAL",
		Reason:      ReasonLoopBodyUnwitnessed,
		By:          subagentFoldBy,
		Disposition: "RETRYABLE",
		Detail:      map[string]string{"claim": kind},
	}, true
}
