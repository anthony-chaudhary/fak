package gitgate

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/wipattr"
	"github.com/anthony-chaudhary/fak/internal/wipref"
	"github.com/anthony-chaudhary/fak/internal/witness"
)

// The two-session sweep-guard fixture (#3879). sessA is the ACTOR; sessB is a PEER.
// Three dirty hunks live under different files:
//
//   - ownHunk        — authored by sessA, recorded in sessA's own checkpoint;
//   - peerCkptHunk   — authored by sessB, recorded in sessB's checkpoint (recoverable);
//   - peerOrphanHunk — authored by sessB but recorded in NO checkpoint (irrecoverable).
//
// A path-scoped `git commit -- <path>` over each is screened by ToolSweepGuard. The
// attribution fold (internal/wipattr) decides OWNED-by-self / OWNED-by-peer / ORPHAN,
// and gitgate's escalation maps those to ALLOW / ADVISE(defer) / REFUSE(deny). These
// fixtures pin the Done condition: own-path unaffected, checkpointed-peer advisory,
// un-checkpointed-peer refused, naming the peer session throughout.
const (
	sweepSelf = "sessA"
	sweepPeer = "sessB"
)

var (
	sweepOwnHunk        = wipattr.Hunk{File: "internal/actor/mine.go", Edit: []string{"+ actor sessA edit"}}
	sweepPeerCkptHunk   = wipattr.Hunk{File: "internal/peer/checkpointed.go", Edit: []string{"+ peer sessB edit (checkpointed)"}}
	sweepPeerOrphanHunk = wipattr.Hunk{File: "internal/peer/orphan.go", Edit: []string{"+ peer sessB edit (never checkpointed)"}}
)

// sweepCheckpoints is the two-session checkpoint set: sessA owns its edit, sessB owns
// its checkpointed edit, and the orphan hunk is recorded by NObody (at-risk WIP).
func sweepCheckpoints() map[string][]wipattr.Hunk {
	return map[string][]wipattr.Hunk{
		sweepSelf: {sweepOwnHunk},
		sweepPeer: {sweepPeerCkptHunk},
	}
}

// sweepPlan builds a SweepGuardPlan the way cmd/fak would: it runs wipattr.Attribute
// over the op's dirty hunks against the live refs/fak/wip/* checkpoints.
func sweepPlan(op string, dirty []wipattr.Hunk, checkpoints map[string][]wipattr.Hunk) SweepGuardPlan {
	return SweepGuardPlan{
		Op:      op,
		Self:    sweepSelf,
		Live:    []string{sweepSelf, sweepPeer},
		Targets: wipattr.Attribute(dirty, checkpoints),
	}
}

func sweepCall(plan SweepGuardPlan) *abi.ToolCall {
	b, _ := json.Marshal(plan)
	return &abi.ToolCall{Tool: ToolSweepGuard, Args: abi.Ref{Kind: abi.RefInline, Inline: b}}
}

// TestSweepGuardOwnPathUnaffected: `git commit -- <own-path>` sweeps only the actor's
// own OWNED delta, so the guard is CLEAR and the op ALLOWs — the "unaffected" arm of
// the Done condition.
func TestSweepGuardOwnPathUnaffected(t *testing.T) {
	plan := sweepPlan("git commit -- internal/actor/mine.go", []wipattr.Hunk{sweepOwnHunk}, sweepCheckpoints())

	finding := CheckSweepGuard(plan)
	if finding.Disposition != SweepClear {
		t.Fatalf("own-path disposition=%v, want SweepClear (finding=%+v)", finding.Disposition, finding)
	}
	if len(finding.Hazards) != 0 {
		t.Fatalf("own-path hazards=%+v, want none", finding.Hazards)
	}

	v := New().Adjudicate(context.Background(), sweepCall(plan))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("own-path verdict=%+v, want Allow", v)
	}
	if v.By != ToolSweepGuard {
		t.Fatalf("own-path By=%q, want %q", v.By, ToolSweepGuard)
	}
}

// TestSweepGuardCheckpointedPeerRefuses: `git commit -- <peer-path>` over a peer delta
// that IS checkpointed at refs/fak/wip/<peer> is elevated from SweepAdvise to SweepRefuse
// with ReasonPeerWIPCollision under strict file-level attribution (#11232).
func TestSweepGuardCheckpointedPeerRefuses(t *testing.T) {
	plan := sweepPlan("git commit -- internal/peer/checkpointed.go", []wipattr.Hunk{sweepPeerCkptHunk}, sweepCheckpoints())

	finding := CheckSweepGuard(plan)
	if finding.Disposition != SweepRefuse {
		t.Fatalf("checkpointed-peer disposition=%v, want SweepRefuse (finding=%+v)", finding.Disposition, finding)
	}
	if finding.Reason != ReasonPeerWIPCollision {
		t.Fatalf("checkpointed-peer reason=%q, want %q", finding.Reason, ReasonPeerWIPCollision)
	}
	if !strings.Contains(finding.Claim, sweepPeer) {
		t.Fatalf("refuse claim does not name peer %q: %q", sweepPeer, finding.Claim)
	}
	if ref := wipref.SessionRef(sweepPeer); !strings.Contains(finding.Claim, ref) {
		t.Fatalf("refuse claim does not cite checkpoint ref %q: %q", ref, finding.Claim)
	}

	g := New()
	rec, captured := decisionRecorder(t)
	g.SetRecorder(rec)

	v := g.Adjudicate(context.Background(), sweepCall(plan))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("checkpointed-peer verdict=%+v, want Deny POLICY_BLOCK", v)
	}
	if v.By != ToolSweepGuard {
		t.Fatalf("checkpointed-peer By=%q, want %q", v.By, ToolSweepGuard)
	}
	wp, ok := v.Payload.(abi.WitnessPayload)
	if !ok || !strings.Contains(wp.Claim, sweepPeer) {
		t.Fatalf("checkpointed-peer payload=%v, want a refusal claim naming peer %q", v.Payload, sweepPeer)
	}
	if len(*captured) != 1 {
		t.Fatalf("expected one recorded refusal, got %d: %+v", len(*captured), *captured)
	}
	got := (*captured)[0]
	if got.Op != ToolSweepGuard || got.Verdict != witness.VerdictRefuse || got.ReasonClass != ReasonPeerWIPCollision {
		t.Fatalf("recorded decision=%+v, want sweep_guard/refuse/%s", got, ReasonPeerWIPCollision)
	}
}

// TestSweepGuardSharedAdvises: a delta claimed by MORE THAN ONE peer checkpoint is
// ambiguous but still checkpointed (recoverable), so it ADVISEs and names every
// claimant + their refs — the SHARED escalation branch.
func TestSweepGuardSharedAdvises(t *testing.T) {
	const peerC = "sessC"
	shared := wipattr.Hunk{File: "internal/peer/shared.go", Edit: []string{"+ edit claimed by two peers"}}
	checkpoints := map[string][]wipattr.Hunk{
		sweepPeer: {shared},
		peerC:     {shared},
	}
	plan := SweepGuardPlan{
		Op:      "git commit -- internal/peer/shared.go",
		Self:    sweepSelf,
		Live:    []string{sweepSelf, sweepPeer, peerC},
		Targets: wipattr.Attribute([]wipattr.Hunk{shared}, checkpoints),
	}

	finding := CheckSweepGuard(plan)
	if finding.Disposition != SweepAdvise {
		t.Fatalf("shared disposition=%v, want SweepAdvise (finding=%+v)", finding.Disposition, finding)
	}
	if !strings.Contains(finding.Claim, "shared across") {
		t.Fatalf("shared claim missing 'shared across': %q", finding.Claim)
	}
	if !strings.Contains(finding.Claim, sweepPeer) || !strings.Contains(finding.Claim, peerC) {
		t.Fatalf("shared claim must name both claimants %q,%q: %q", sweepPeer, peerC, finding.Claim)
	}

	if v := New().Adjudicate(context.Background(), sweepCall(plan)); v.Kind != abi.VerdictDefer {
		t.Fatalf("shared verdict=%+v, want Defer (advisory)", v)
	}
}

// TestSweepGuardUncheckpointedPeerRefuses is the refuse arm + the captured readout the
// witness asks for: `git commit -- <peer-path>` over an ORPHAN (un-checkpointed) peer
// delta has no refs/fak/wip/* safety net, so it is irrecoverable if swept — the one
// case that escalates past advisory to a provable POLICY_BLOCK deny, recorded to the
// witness sink.
func TestSweepGuardUncheckpointedPeerRefuses(t *testing.T) {
	plan := sweepPlan("git commit -- internal/peer/orphan.go", []wipattr.Hunk{sweepPeerOrphanHunk}, sweepCheckpoints())

	finding := CheckSweepGuard(plan)
	if finding.Disposition != SweepRefuse {
		t.Fatalf("orphan disposition=%v, want SweepRefuse (finding=%+v)", finding.Disposition, finding)
	}
	if !strings.Contains(finding.Claim, "UN-CHECKPOINTED") || !strings.Contains(finding.Claim, sweepPeerOrphanHunk.File) {
		t.Fatalf("orphan claim must flag UN-CHECKPOINTED WIP naming %q: %q", sweepPeerOrphanHunk.File, finding.Claim)
	}
	// The captured gitgate readout on the refuse path (issue Witness).
	t.Logf("sweep-guard refuse readout: %s", finding.Claim)

	g := New()
	rec, captured := decisionRecorder(t)
	g.SetRecorder(rec)

	v := g.Adjudicate(context.Background(), sweepCall(plan))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonPolicyBlock {
		t.Fatalf("orphan verdict=%+v, want Deny POLICY_BLOCK", v)
	}
	if v.By != ToolSweepGuard {
		t.Fatalf("orphan By=%q, want %q", v.By, ToolSweepGuard)
	}
	wp, ok := v.Payload.(abi.WitnessPayload)
	if !ok || !strings.Contains(wp.Claim, "UN-CHECKPOINTED") {
		t.Fatalf("orphan payload=%v, want an UN-CHECKPOINTED refusal claim", v.Payload)
	}

	if len(*captured) != 1 {
		t.Fatalf("expected one recorded refusal, got %d: %+v", len(*captured), *captured)
	}
	got := (*captured)[0]
	if got.Op != ToolSweepGuard || got.Verdict != witness.VerdictRefuse || got.ReasonClass != "POLICY_BLOCK" {
		t.Fatalf("recorded decision=%+v, want sweep_guard/refuse/POLICY_BLOCK", got)
	}
	if !containsString(got.Tree, sweepPeerOrphanHunk.File) {
		t.Fatalf("recorded Tree=%+v, want it to include the orphan file %q", got.Tree, sweepPeerOrphanHunk.File)
	}
}

// TestSweepGuardOrphanDominatesMixedSweep: when a single pathspec sweeps the actor's
// own delta, a checkpointed peer delta, AND an orphan together, the irrecoverable
// ORPHAN dominates — the whole op REFUSES. Proves refuse > advise > clear precedence.
func TestSweepGuardOrphanDominatesMixedSweep(t *testing.T) {
	dirty := []wipattr.Hunk{sweepOwnHunk, sweepPeerCkptHunk, sweepPeerOrphanHunk}
	plan := sweepPlan("git commit -- internal/", dirty, sweepCheckpoints())

	finding := CheckSweepGuard(plan)
	if finding.Disposition != SweepRefuse {
		t.Fatalf("mixed disposition=%v, want SweepRefuse (orphan dominates) (finding=%+v)", finding.Disposition, finding)
	}
	// The actor's own SAFE hunk is never a hazard; both peer hunks are.
	if len(finding.Hazards) != 2 {
		t.Fatalf("mixed hazards=%+v, want 2 (checkpointed peer + orphan), own excluded", finding.Hazards)
	}

	if v := New().Adjudicate(context.Background(), sweepCall(plan)); v.Kind != abi.VerdictDeny {
		t.Fatalf("mixed verdict=%+v, want Deny (orphan present)", v)
	}
}

// TestSweepGuardMalformedArgsDeny: a sweep-guard call with no JSON args is a malformed
// plan, not a silent allow — it Denies ReasonMalformed (fail-closed).
func TestSweepGuardMalformedArgsDeny(t *testing.T) {
	call := &abi.ToolCall{Tool: ToolSweepGuard, Args: abi.Ref{Kind: abi.RefInline}}
	if v := New().Adjudicate(context.Background(), call); v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonMalformed {
		t.Fatalf("malformed sweep-guard verdict=%+v, want Deny MALFORMED", v)
	}
}

// TestSweepGuardOrphanAllowedWhenQuarantined proves that when an orphan dirty hunk
// is backed by a quarantine ref archive (Issue #11238), it is recoverable and allowed.
func TestSweepGuardOrphanAllowedWhenQuarantined(t *testing.T) {
	plan := sweepPlan("git clean -- internal/peer/orphan.go", []wipattr.Hunk{sweepPeerOrphanHunk}, sweepCheckpoints())
	plan.QuarantineRef = "refs/fak/quarantine/sessA/20260904-120000"

	finding := CheckSweepGuard(plan)
	if finding.Disposition != SweepClear {
		t.Fatalf("quarantined orphan disposition=%v, want SweepClear (finding=%+v)", finding.Disposition, finding)
	}

	v := New().Adjudicate(context.Background(), sweepCall(plan))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("quarantined orphan verdict=%+v, want Allow", v)
	}
}

func containsString(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
