package toolprocgate

// reusearm_test.go — the arming witness for #5407. Three failure classes, each
// of which was silently reachable while repeatreason.go had no consumer:
//
//  1. NOBODY REGISTERED. abi.ReasonName falls back to REASON_<n> for an
//     unregistered code, so forgetting the registration produces a wire that
//     still looks well-formed and a verdict nobody can read. Asserted through
//     the seam, NOT by registering the pairs here first — a test that registers
//     its own vocabulary proves the registry works, never that the consumer
//     armed it. Negative witness: drop reusearm.go's init body and
//     TestServedRepeatCitesItsRegisteredToken reds on the first verdict it
//     renders (`name="REASON_1092"`, the fresh-fetch code), with
//     TestEveryReuseVerdictNameRoundTrips reporting all six as REASON_<n>.
//
//  2. THE CONSUMER FABRICATES. ReuseReasonCode is fail-closed for a reason
//     outside the closed set; that only matters if the call site honours it.
//     Negative witness: make ReuseVerdictFor name an unmapped receipt anyway
//     and TestUnmappedReceiptYieldsNoVerdictAtTheConsumer reds.
//
//  3. UNBLESSED BYTES SERVE. An unnamed verdict must not answer a call locally
//     even when the store holds the payload, because "no verdict" and "a
//     verdict that permits reuse" are the two things a fail-closed seam may
//     never conflate.

import (
	"bytes"
	"context"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

func readCall(raw string, atMS int64, n int) toolproc.CallRecord {
	return toolproc.CallRecord{Tool: "shell_command", Raw: raw, AtMS: atMS, OutputBytes: int64(n), Digest: "d1"}
}

// TestServedRepeatCitesItsRegisteredToken is the arming assertion: a repeat this
// seam actually serves renders its stable REUSE_* token, on the same Result.Meta
// surface a quarantined completion uses for kill_reason. This is the test the
// missing registration reds.
func TestServedRepeatCitesItsRegisteredToken(t *testing.T) {
	arm := NewReuseArm(toolproc.ArmedConfig{})
	body := []byte("skill file content")

	first := arm.Serve(context.Background(), nil, readCall("cat C:/x/SKILL.md", 0, len(body)))
	if first.Served() {
		t.Fatalf("first read must not be answered locally, got %+v", first)
	}
	if !first.Named || first.Name() != toolproc.ReasonReuseFirstFetchName {
		t.Fatalf("a fresh fetch must still cite its verdict: got named=%v name=%q",
			first.Named, first.Name())
	}
	if !arm.Offer(readCall("cat C:/x/SKILL.md", 0, len(body)), body) {
		t.Fatal("offer of an immutable-read body must be retained")
	}

	hit := arm.Serve(context.Background(), &abi.ToolCall{TraceID: "call-1"}, readCall(`Get-Content -Raw C:\x\SKILL.md`, 20, len(body)))
	if !hit.Served() {
		t.Fatalf("an equivalent re-read must be answered locally, got %+v", hit)
	}
	if hit.Reason != toolproc.ReasonReuseKeyedHit {
		t.Errorf("served repeat cites code %d, want %d", hit.Reason, toolproc.ReasonReuseKeyedHit)
	}
	// The round trip that IS the point of repeatreason.go's bridge. Compared
	// against the fallback spelling explicitly so the failure message names the
	// cause rather than just a string mismatch.
	fallback := "REASON_" + strconv.Itoa(int(toolproc.ReasonReuseKeyedHit))
	if got := hit.Name(); got != toolproc.ReasonReuseKeyedHitName {
		if got == fallback {
			t.Fatalf("served repeat renders %q: nothing registered the reuse pairs, so the "+
				"verdict is unreadable while the wire still looks well-formed", got)
		}
		t.Fatalf("served repeat renders %q, want %q", got, toolproc.ReasonReuseKeyedHitName)
	}
	if got := hit.Result.Meta[ReuseReasonMetaKey]; got != toolproc.ReasonReuseKeyedHitName {
		t.Errorf("Result.Meta[%s] = %q, want %q — the verdict must be visible where the "+
			"other kernel refusals are", ReuseReasonMetaKey, got, toolproc.ReasonReuseKeyedHitName)
	}
	if got := hit.Result.Meta[ReuseSourceMetaKey]; got != string(toolproc.SourceImmutable) {
		t.Errorf("Result.Meta[%s] = %q, want %q", ReuseSourceMetaKey, got, toolproc.SourceImmutable)
	}
	if hit.Result.Meta["toolprocgate"] != ReuseServedMetaValue {
		t.Errorf("a locally answered repeat must carry the toolprocgate marker, got %v", hit.Result.Meta)
	}
	if !bytes.Equal(hit.Result.Payload.Inline, body) {
		t.Errorf("served payload = %q, want %q", hit.Result.Payload.Inline, body)
	}
	if hit.Result.Payload.Taint != abi.TaintTainted || hit.Result.Payload.Scope != abi.ScopeAgent {
		t.Errorf("reuse must not widen trust: taint=%d scope=%d", hit.Result.Payload.Taint, hit.Result.Payload.Scope)
	}
}

// TestEveryReuseVerdictNameRoundTrips widens the arming assertion past the one
// token the served path happens to cite: registration is for-loop-shaped, and a
// pair dropped from the loop's data would leave five of six readable.
func TestEveryReuseVerdictNameRoundTrips(t *testing.T) {
	pairs := toolproc.ReuseReasonPairs()
	if len(pairs) == 0 {
		t.Fatal("no reuse pairs to register: the vocabulary is gone, not armed")
	}
	for _, p := range pairs {
		if got := abi.ReasonName(p.Code); got != p.Name {
			t.Errorf("abi.ReasonName(%d) = %q, want %q: this leaf's init did not register the pair",
				p.Code, got, p.Name)
		}
	}
}

// TestUnmappedReceiptYieldsNoVerdictAtTheConsumer pins the fail-closed half at
// the call site. An empty Receipt.Reason is the realistic case — a zero-valued
// receipt that never ran the fold — and an invented token is the careless one.
// Neither may acquire a code, a name, or a REASON_<n> stand-in.
func TestUnmappedReceiptYieldsNoVerdictAtTheConsumer(t *testing.T) {
	for _, r := range []toolproc.ReuseReason{"", "keyed_miss", "KEYED_HIT", " keyed_hit"} {
		v := ReuseVerdictFor(toolproc.Receipt{Reason: r})
		if v.Named {
			t.Errorf("ReuseVerdictFor(%q) named the verdict %d (%q): a reason outside the closed "+
				"set has none, and the consumer must not fabricate one", r, v.Reason, v.Name())
		}
		if v.Reason != 0 {
			t.Errorf("ReuseVerdictFor(%q) carried code %d, want none", r, v.Reason)
		}
		if got := v.Name(); got != "" {
			t.Errorf("ReuseVerdictFor(%q).Name() = %q, want \"\": not even the REASON_<n> fallback, "+
				"which would read as a registered verdict on the wire", r, got)
		}
		if v.Served() {
			t.Errorf("ReuseVerdictFor(%q) reports a local answer with no verdict behind it", r)
		}
	}
}

// TestZeroValuedReceiptNeverServesBytes is the third class: the fail-closed
// branch has to gate the SERVE, not only the rendering. Driven through Serve so
// it pins the seam's own ordering (name first, answer only if named).
func TestZeroValuedReceiptNeverServesBytes(t *testing.T) {
	v := ReuseVerdictFor(toolproc.Receipt{Served: true, Source: toolproc.SourceImmutable})
	if v.Served() || v.Result != nil {
		t.Fatalf("a receipt whose reason is unmapped must never carry a served result, got %+v", v)
	}
	// And the live path agrees: a write is fail-closed at the leaf, so it is a
	// named refusal, never an answer.
	arm := NewReuseArm(toolproc.ArmedConfig{})
	w := readCall("git push origin main", 0, 10)
	arm.Offer(w, []byte("pushed"))
	got := arm.Serve(context.Background(), nil, w)
	if got.Served() {
		t.Fatalf("a write must never be answered from the reuse store, got %+v", got)
	}
	if got.Name() != toolproc.ReasonReuseNeverReusedName {
		t.Errorf("a write's refusal cites %q, want %q", got.Name(), toolproc.ReasonReuseNeverReusedName)
	}
}
