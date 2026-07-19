package sessionctl

import (
	"context"
	"strings"
	"testing"
	"time"
)

// parkOutcome carries one ParkGatedAction return across the goroutine boundary.
type parkOutcome struct {
	verdict ParkVerdict
	refusal *ParkRefusal
}

// parkInBackground parks the action on its own goroutine (ParkGatedAction
// blocks the calling loop by design) and returns the outcome channel.
func parkInBackground(ctx context.Context, trace string, action GatedAction) chan parkOutcome {
	done := make(chan parkOutcome, 1)
	go func() {
		v, ref := ParkGatedAction(ctx, trace, action)
		done <- parkOutcome{verdict: v, refusal: ref}
	}()
	return done
}

// waitPending polls the addressable queue until one action is pending — the
// operator's list-then-resolve read.
func waitPending(t *testing.T, trace string) PendingAction {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if pending := PendingGatedActions(trace); len(pending) == 1 {
			return pending[0]
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatal("parked action never appeared on the pending queue")
	return PendingAction{}
}

func awaitOutcome(t *testing.T, done chan parkOutcome) parkOutcome {
	t.Helper()
	select {
	case out := <-done:
		return out
	case <-time.After(5 * time.Second):
		t.Fatal("parked loop never returned")
		return parkOutcome{}
	}
}

// TestParkedActionProceedsOnApprove is the #2757 done-condition witness, approve
// half: an operator lists the pending gated action on a running (parked) session
// and approves it out of band; the parked loop wakes with the approve verdict —
// the proceed the loop honors — and the audit journal records the resolution.
func TestParkedActionProceedsOnApprove(t *testing.T) {
	trace := "trace-park-approve"
	defer ClearParked(trace)
	if ref := EnableGateParking(trace, time.Minute); ref != nil {
		t.Fatalf("enable parking: %v", ref)
	}

	action := GatedAction{
		Tool:    "Bash",
		Args:    `{"command":"git push origin main"}`,
		Reason:  "REQUIRE_WITNESS",
		Preview: "outward-facing push; confirm out of band",
	}
	done := parkInBackground(context.Background(), trace, action)

	// The operator READS the pending queue: the parked call is addressable with
	// its gate context intact.
	pending := waitPending(t, trace)
	if pending.ID == "" || pending.Trace != trace {
		t.Fatalf("pending action not addressable: %+v", pending)
	}
	if pending.Action.Tool != "Bash" || pending.Action.Reason != "REQUIRE_WITNESS" || pending.Action.Preview == "" {
		t.Fatalf("pending action lost its gate context: %+v", pending.Action)
	}

	// The operator APPROVES out of band.
	if ref := ResolveGatedAction(trace, pending.ID, ParkVerdict{Kind: ParkApprove, Note: "reviewed the preview"}); ref != nil {
		t.Fatalf("approve refused: %v", ref)
	}

	out := awaitOutcome(t, done)
	if out.refusal != nil {
		t.Fatalf("approved park refused: %v", out.refusal)
	}
	if out.verdict.Kind != ParkApprove {
		t.Fatalf("loop woke with verdict %q, want %q", out.verdict.Kind, ParkApprove)
	}
	if got := PendingGatedActions(trace); len(got) != 0 {
		t.Fatalf("resolved action still pending: %+v", got)
	}

	// Audit journal: the resolution is witnessed as an APPLIED proceed.
	records := ReadParkNextRecords(trace)
	if len(records) != 1 {
		t.Fatalf("audit records = %d, want 1: %+v", len(records), records)
	}
	rec := records[0]
	if !rec.Applied || rec.Refusal != "" {
		t.Fatalf("approve audit row not applied: %+v", rec)
	}
	if rec.Move.Gate != "sessionctl-park" || !strings.Contains(rec.Move.Payload, "approve") || !strings.Contains(rec.Move.Payload, "Bash") {
		t.Fatalf("approve audit row lost its payload: %+v", rec.Move)
	}
	if !strings.Contains(rec.Move.Payload, "reviewed the preview") {
		t.Fatalf("operator note missing from audit row: %q", rec.Move.Payload)
	}
	if extra := ReadParkNextRecords(trace); len(extra) != 0 {
		t.Fatalf("audit read did not clear: %+v", extra)
	}
}

// TestParkedActionAbortsOnDeny is the #2757 done-condition witness, deny half:
// an external deny wakes the parked loop with the deny verdict — the abort the
// loop honors — and the audit journal records the closed PARK_OPERATOR_DENIED
// outcome.
func TestParkedActionAbortsOnDeny(t *testing.T) {
	trace := "trace-park-deny"
	defer ClearParked(trace)
	if ref := EnableGateParking(trace, time.Minute); ref != nil {
		t.Fatalf("enable parking: %v", ref)
	}

	done := parkInBackground(context.Background(), trace, GatedAction{
		Tool: "Bash", Args: `{"command":"rm -rf build"}`, Reason: "REQUIRE_WITNESS",
	})
	pending := waitPending(t, trace)

	if ref := ResolveGatedAction(trace, pending.ID, ParkVerdict{Kind: ParkDeny, Note: "not during the freeze"}); ref != nil {
		t.Fatalf("deny refused: %v", ref)
	}

	out := awaitOutcome(t, done)
	if out.refusal != nil {
		t.Fatalf("denied park returned a machinery refusal, want the deny verdict: %v", out.refusal)
	}
	if out.verdict.Kind != ParkDeny {
		t.Fatalf("loop woke with verdict %q, want %q", out.verdict.Kind, ParkDeny)
	}

	records := ReadParkNextRecords(trace)
	if len(records) != 1 {
		t.Fatalf("audit records = %d, want 1: %+v", len(records), records)
	}
	rec := records[0]
	if rec.Applied {
		t.Fatalf("deny audit row claims applied: %+v", rec)
	}
	if !strings.Contains(rec.Refusal, string(ParkOperatorDenied)) {
		t.Fatalf("deny audit row lost the closed reason: %+v", rec)
	}
	if !strings.Contains(rec.Move.Payload, "deny") || !strings.Contains(rec.Move.Payload, "not during the freeze") {
		t.Fatalf("deny audit row lost its payload: %+v", rec.Move)
	}
}

// TestParkedActionApproveWithModifiedArgs: argument-modify is an approve
// carrying replacement args — the loop receives them for a fresh adjudication,
// and the audit row names the modification.
func TestParkedActionApproveWithModifiedArgs(t *testing.T) {
	trace := "trace-park-modify"
	defer ClearParked(trace)
	if ref := EnableGateParking(trace, time.Minute); ref != nil {
		t.Fatalf("enable parking: %v", ref)
	}

	done := parkInBackground(context.Background(), trace, GatedAction{
		Tool: "Bash", Args: `{"command":"rm -rf data"}`, Reason: "REQUIRE_WITNESS",
	})
	pending := waitPending(t, trace)

	modified := `{"command":"rm -rf build/tmp"}`
	if ref := ResolveGatedAction(trace, pending.ID, ParkVerdict{Kind: ParkApprove, Args: modified}); ref != nil {
		t.Fatalf("approve-with-modify refused: %v", ref)
	}

	out := awaitOutcome(t, done)
	if out.refusal != nil || out.verdict.Kind != ParkApprove {
		t.Fatalf("modify outcome = %+v / %v", out.verdict, out.refusal)
	}
	if out.verdict.Args != modified {
		t.Fatalf("modified args lost: %q", out.verdict.Args)
	}
	records := ReadParkNextRecords(trace)
	if len(records) != 1 || !strings.Contains(records[0].Move.Payload, "args modified") {
		t.Fatalf("modify audit row missing: %+v", records)
	}
}

// TestParkTimeoutIsExplicit: a parked action nobody resolves aborts with the
// closed PARK_TIMEOUT reason at the park window — witnessed, never silently
// dropped — and leaves the pending queue.
func TestParkTimeoutIsExplicit(t *testing.T) {
	trace := "trace-park-timeout"
	defer ClearParked(trace)
	if ref := EnableGateParking(trace, 20*time.Millisecond); ref != nil {
		t.Fatalf("enable parking: %v", ref)
	}

	done := parkInBackground(context.Background(), trace, GatedAction{Tool: "Bash", Reason: "REQUIRE_WITNESS"})
	out := awaitOutcome(t, done)
	if out.refusal == nil || out.refusal.Reason != ParkTimeout {
		t.Fatalf("timeout outcome = %+v / %v, want %s", out.verdict, out.refusal, ParkTimeout)
	}
	if got := PendingGatedActions(trace); len(got) != 0 {
		t.Fatalf("timed-out action still pending: %+v", got)
	}
	records := ReadParkNextRecords(trace)
	if len(records) != 1 || records[0].Applied || !strings.Contains(records[0].Refusal, string(ParkTimeout)) {
		t.Fatalf("timeout not witnessed: %+v", records)
	}
	// The stale id is no longer resolvable: the abort was terminal.
	if ref := ResolveGatedAction(trace, "park-1", ParkVerdict{Kind: ParkApprove}); ref == nil || ref.Reason != ParkUnknownAction {
		t.Fatalf("stale resolve = %v, want %s", ref, ParkUnknownAction)
	}
}

// TestParkAbortsOnContextCancel: a cancelled park context releases the loop
// with the closed PARK_ABORTED reason.
func TestParkAbortsOnContextCancel(t *testing.T) {
	trace := "trace-park-cancel"
	defer ClearParked(trace)
	if ref := EnableGateParking(trace, time.Minute); ref != nil {
		t.Fatalf("enable parking: %v", ref)
	}

	ctx, cancel := context.WithCancel(context.Background())
	done := parkInBackground(ctx, trace, GatedAction{Tool: "Bash", Reason: "REQUIRE_WITNESS"})
	waitPending(t, trace)
	cancel()
	out := awaitOutcome(t, done)
	if out.refusal == nil || out.refusal.Reason != ParkAborted {
		t.Fatalf("cancel outcome = %+v / %v, want %s", out.verdict, out.refusal, ParkAborted)
	}
	if records := ReadParkNextRecords(trace); len(records) != 1 || !strings.Contains(records[0].Refusal, string(ParkAborted)) {
		t.Fatalf("abort not witnessed: %+v", records)
	}
}

// TestParkClearedAtTeardownReleasesWaiter: ClearParked never leaves a parked
// loop blocking — the waiter is released with PARK_ABORTED.
func TestParkClearedAtTeardownReleasesWaiter(t *testing.T) {
	trace := "trace-park-teardown"
	if ref := EnableGateParking(trace, time.Minute); ref != nil {
		t.Fatalf("enable parking: %v", ref)
	}
	done := parkInBackground(context.Background(), trace, GatedAction{Tool: "Bash"})
	waitPending(t, trace)
	ClearParked(trace)
	out := awaitOutcome(t, done)
	if out.refusal == nil || out.refusal.Reason != ParkAborted {
		t.Fatalf("teardown outcome = %+v / %v, want %s", out.verdict, out.refusal, ParkAborted)
	}
	if GateParkingEnabled(trace) {
		t.Fatal("teardown left the inbox open")
	}
}

// TestParkResolveRefusalsAreClosed: the malformed / unknown-action edges refuse
// synchronously with their closed reasons, and a resolve is never
// double-delivered.
func TestParkResolveRefusalsAreClosed(t *testing.T) {
	trace := "trace-park-refusals"
	defer ClearParked(trace)
	if ref := EnableGateParking(trace, time.Minute); ref != nil {
		t.Fatalf("enable parking: %v", ref)
	}

	if _, ref := ParkGatedAction(context.Background(), "", GatedAction{Tool: "Bash"}); ref == nil || ref.Reason != ParkMalformed {
		t.Fatalf("empty-trace park = %v, want %s", ref, ParkMalformed)
	}
	if _, ref := ParkGatedAction(context.Background(), trace, GatedAction{}); ref == nil || ref.Reason != ParkMalformed {
		t.Fatalf("toolless park = %v, want %s", ref, ParkMalformed)
	}
	if ref := EnableGateParking("", time.Minute); ref == nil || ref.Reason != ParkMalformed {
		t.Fatalf("empty-trace enable = %v, want %s", ref, ParkMalformed)
	}

	if ref := ResolveGatedAction(trace, "park-999", ParkVerdict{Kind: ParkApprove}); ref == nil || ref.Reason != ParkUnknownAction {
		t.Fatalf("unknown-id resolve = %v, want %s", ref, ParkUnknownAction)
	}

	done := parkInBackground(context.Background(), trace, GatedAction{Tool: "Bash", Reason: "REQUIRE_WITNESS"})
	pending := waitPending(t, trace)

	if ref := ResolveGatedAction(trace, pending.ID, ParkVerdict{Kind: "shrug"}); ref == nil || ref.Reason != ParkMalformed {
		t.Fatalf("unknown-kind resolve = %v, want %s", ref, ParkMalformed)
	}
	if ref := ResolveGatedAction(trace, pending.ID, ParkVerdict{Kind: ParkDeny, Args: `{"command":"x"}`}); ref == nil || ref.Reason != ParkMalformed {
		t.Fatalf("deny-with-modify resolve = %v, want %s", ref, ParkMalformed)
	}
	if ref := ResolveGatedAction(trace, pending.ID, ParkVerdict{Kind: ParkApprove, Args: `not json`}); ref == nil || ref.Reason != ParkMalformed {
		t.Fatalf("non-object-modify resolve = %v, want %s", ref, ParkMalformed)
	}
	// A malformed resolve consumed nothing: the action is still pending and one
	// legal resolve still lands.
	if got := PendingGatedActions(trace); len(got) != 1 {
		t.Fatalf("malformed resolves consumed the pending action: %+v", got)
	}
	if ref := ResolveGatedAction(trace, pending.ID, ParkVerdict{Kind: ParkApprove}); ref != nil {
		t.Fatalf("legal resolve refused after malformed attempts: %v", ref)
	}
	// The second delivery refuses: the verdict was consumed exactly once.
	if ref := ResolveGatedAction(trace, pending.ID, ParkVerdict{Kind: ParkDeny}); ref == nil || ref.Reason != ParkUnknownAction {
		t.Fatalf("double resolve = %v, want %s", ref, ParkUnknownAction)
	}
	out := awaitOutcome(t, done)
	if out.refusal != nil || out.verdict.Kind != ParkApprove {
		t.Fatalf("outcome after refusal gauntlet = %+v / %v", out.verdict, out.refusal)
	}
}

// TestParkVerdictBeatsDeadlineRace: a verdict an operator delivered is honored
// even when it races the park window — an operator resolve is never converted
// into a timeout abort.
func TestParkVerdictBeatsDeadlineRace(t *testing.T) {
	trace := "trace-park-race"
	defer ClearParked(trace)
	if ref := EnableGateParking(trace, 30*time.Millisecond); ref != nil {
		t.Fatalf("enable parking: %v", ref)
	}
	done := parkInBackground(context.Background(), trace, GatedAction{Tool: "Bash", Reason: "REQUIRE_WITNESS"})
	pending := waitPending(t, trace)
	// Resolve inside the window; the parked loop must observe the verdict, not
	// the deadline, regardless of scheduling.
	if ref := ResolveGatedAction(trace, pending.ID, ParkVerdict{Kind: ParkDeny}); ref != nil {
		t.Fatalf("resolve refused: %v", ref)
	}
	out := awaitOutcome(t, done)
	if out.refusal != nil || out.verdict.Kind != ParkDeny {
		t.Fatalf("raced outcome = %+v / %v, want the delivered deny", out.verdict, out.refusal)
	}
}
