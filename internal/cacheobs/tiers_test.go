package cacheobs

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

// access builds one fully-witnessed TierAccess (bytes and latency both measured).
func access(tier CacheTier, op TierOp, outcome TierOutcome, backend BackendClass, bytes int64, latency time.Duration) Event {
	return Event{Kind: EventTierAccess, Access: TierAccess{
		Tier: tier, Op: op, Outcome: outcome, Backend: backend,
		Bytes: bytes, BytesKnown: true,
		Latency: latency, LatencyKnown: true,
	}}
}

// unwitnessedAccess builds an access from a tap that measures NEITHER bytes nor latency —
// the provider-relayed case, where fak learns the outcome but never the payload size or the
// upstream's internal timing.
func unwitnessedAccess(tier CacheTier, op TierOp, outcome TierOutcome, backend BackendClass) Event {
	return Event{Kind: EventTierAccess, Access: TierAccess{Tier: tier, Op: op, Outcome: outcome, Backend: backend}}
}

// tierWorkload is the fixed multi-tier workload both arms of the A/B witness consume: a fast
// memory-resident prefix tier, a slow remote shared store (whose write and error both land on
// it), and a provider-managed tier fak only observes. The tiers deliberately differ by orders
// of magnitude in latency and payload — that is the whole point: one blended mean describes
// none of them.
func tierWorkload() []Event {
	const kb = 1024
	return []Event{
		access(TierLocalPrefix, OpRead, OutcomeHit, BackendMemory, 4*kb, 200*time.Microsecond),
		access(TierLocalPrefix, OpRead, OutcomeHit, BackendMemory, 6*kb, 300*time.Microsecond),
		access(TierLocalPrefix, OpRead, OutcomeMiss, BackendMemory, 0, 100*time.Microsecond),
		access(TierSharedStore, OpRead, OutcomeHit, BackendRemote, 64*kb, 12*time.Millisecond),
		access(TierSharedStore, OpRead, OutcomeError, BackendRemote, 0, 30*time.Millisecond),
		access(TierSharedStore, OpWrite, OutcomeHit, BackendRemote, 64*kb, 20*time.Millisecond),
		unwitnessedAccess(TierProviderManaged, OpRead, OutcomeHit, BackendRemote),
	}
}

// observeWorkload books a workload's tier accesses into a fresh observer.
func observeWorkload(events []Event) *Observer {
	o := New()
	for _, e := range events {
		if e.Kind == EventTierAccess {
			o.ObserveTier(e.Access)
		}
	}
	return o
}

// formatTierWitness renders a report as the CAPTURED witness line-set: one deterministic line
// per tier plus the grand total. Capturing the rendering (not just spot-checked fields) is
// what makes a silent shape change — a dropped tier row, a renamed status, a counter that
// stops accumulating — fail rather than pass unnoticed.
func formatTierWitness(rep TierReport) string {
	var b strings.Builder
	for _, ts := range rep.Tiers {
		fmt.Fprintf(&b, "tier=%s status=%s requests=%d hits=%d misses=%d errors=%d bytes=%d sized=%d latency_ns=%d timed=%d\n",
			ts.Tier, ts.Status, ts.Requests, ts.Hits, ts.Misses, ts.Errors, ts.Bytes, ts.SizedAccesses, ts.LatencyNanos, ts.TimedAccesses)
	}
	t := rep.Total
	fmt.Fprintf(&b, "total requests=%d hits=%d misses=%d errors=%d bytes=%d sized=%d latency_ns=%d timed=%d\n",
		t.Requests, t.Hits, t.Misses, t.Errors, t.Bytes, t.SizedAccesses, t.LatencyNanos, t.TimedAccesses)
	return b.String()
}

// capturedTierWitness is the golden split of tierWorkload. Read it as the evidence the issue
// asks for: the same seven accesses, attributed to the three tiers they actually hit.
const capturedTierWitness = `tier=local_prefix status=supported requests=3 hits=2 misses=1 errors=0 bytes=10240 sized=3 latency_ns=600000 timed=3
tier=shared_store status=supported requests=3 hits=2 misses=0 errors=1 bytes=131072 sized=3 latency_ns=62000000 timed=3
tier=provider_managed status=supported requests=1 hits=1 misses=0 errors=0 bytes=0 sized=0 latency_ns=0 timed=0
total requests=7 hits=5 misses=1 errors=1 bytes=141312 sized=6 latency_ns=62600000 timed=6
`

// TestTierWitnessSplitsWorkloadAndReconcilesToAggregate is the A/B witness the issue names.
//
// Arm A is the tier-BLIND aggregate: the same access stream reduced flat through the
// declarative registry, with no tier attribution anywhere in the path — this is what the
// cache reported before the tier axis existed.
// Arm B is the tier-SPLIT report: the imperative per-tier accumulator the production taps
// feed, read back through TierSnapshot.
//
// The two must agree on every total (nothing is lost or invented by splitting), while arm B
// additionally shows the per-tier truth arm A cannot express — and the test pins that the
// blended mean latency arm A reports describes NEITHER tier, which is exactly the
// conflation the split exists to prevent.
func TestTierWitnessSplitsWorkloadAndReconcilesToAggregate(t *testing.T) {
	events := tierWorkload()

	// Arm B: the split.
	rep := observeWorkload(events).TierSnapshot()
	if got := formatTierWitness(rep); got != capturedTierWitness {
		t.Fatalf("captured tier witness drifted:\n got:\n%s\nwant:\n%s", got, capturedTierWitness)
	}

	// Arm A: the flat aggregate, reduced through the shared registry with no tier map.
	flat := SpecFold(TierSpecs(), events)
	for _, tc := range []struct {
		field string
		got   float64
	}{
		{FieldTierRequests, float64(rep.Total.Requests)},
		{FieldTierHits, float64(rep.Total.Hits)},
		{FieldTierMisses, float64(rep.Total.Misses)},
		{FieldTierErrors, float64(rep.Total.Errors)},
		{FieldTierBytes, float64(rep.Total.Bytes)},
		{FieldTierLatencyNanos, float64(rep.Total.LatencyNanos)},
	} {
		if flat[tc.field] != tc.got {
			t.Fatalf("tier totals must reconcile with the flat aggregate on %s: split=%v flat=%v", tc.field, tc.got, flat[tc.field])
		}
	}

	// The per-tier rows must sum to the grand total (the parts==total invariant).
	var sum TierCounters
	for _, ts := range rep.Tiers {
		sum.Requests += ts.Requests
		sum.Hits += ts.Hits
		sum.Misses += ts.Misses
		sum.Errors += ts.Errors
		sum.Bytes += ts.Bytes
		sum.SizedAccesses += ts.SizedAccesses
		sum.LatencyNanos += ts.LatencyNanos
		sum.TimedAccesses += ts.TimedAccesses
	}
	if sum != (TierCounters{
		Requests: rep.Total.Requests, Hits: rep.Total.Hits, Misses: rep.Total.Misses, Errors: rep.Total.Errors,
		Bytes: rep.Total.Bytes, SizedAccesses: rep.Total.SizedAccesses,
		LatencyNanos: rep.Total.LatencyNanos, TimedAccesses: rep.Total.TimedAccesses,
	}) {
		t.Fatalf("per-tier rows must sum to the grand total: rows=%+v total=%+v", sum, rep.Total)
	}

	// The conflation the split prevents: the blended mean latency belongs to no tier.
	local, ok := rep.Tier(TierLocalPrefix)
	if !ok {
		t.Fatal("local_prefix row must be present")
	}
	shared, ok := rep.Tier(TierSharedStore)
	if !ok {
		t.Fatal("shared_store row must be present")
	}
	blended := rep.Total.MeanLatencyNanos
	if !(local.MeanLatencyNanos < blended && blended < shared.MeanLatencyNanos) {
		t.Fatalf("blended mean latency %v must sit between the tier means (local %v, shared %v) — the tiers are not separated",
			blended, local.MeanLatencyNanos, shared.MeanLatencyNanos)
	}
	if ratio := shared.MeanLatencyNanos / local.MeanLatencyNanos; ratio < 10 {
		t.Fatalf("the two tiers must differ by an order of magnitude for this witness to mean anything, got %v", ratio)
	}
	// Hit ratio is per tier, not one flattering aggregate: the remote tier's error is kept
	// out of its denominator, so it reads 2/2 while the local tier reads 2/3.
	if local.HitRatio != 2.0/3.0 {
		t.Fatalf("local_prefix hit ratio = %v, want 2/3", local.HitRatio)
	}
	if shared.HitRatio != 1.0 {
		t.Fatalf("shared_store hit ratio = %v, want 1.0 (its error is not a cache miss)", shared.HitRatio)
	}
}

// TestExistingTurnTapEmitsLocalPrefixTierRecord proves the tier record is emitted by the
// EXISTING cache path rather than only by a new opt-in call: every depth-axis tap
// (Observe / ObserveSplit / ObservePreempted / ObserveLabeled) books one local-prefix read.
// The tier totals then reconcile exactly with the aggregate they decompose —
// requests == Stats.Turns — which is the reconciliation the issue asks for on the live path.
func TestExistingTurnTapEmitsLocalPrefixTierRecord(t *testing.T) {
	o := New()
	o.Observe(1000, 900)                                    // hit
	o.ObserveSplit(500, 300, 0)                             // miss: nothing served
	o.ObserveLabeled(Labels{Model: "m"}, 200, 100, 50, 200) // hit
	o.ObservePreempted(400, 200, 0, 100)                    // miss (self-inflicted, still a miss)
	o.Observe(0, 0)                                         // ignored: no turn to attribute

	snap := o.Snapshot()
	rep := o.TierSnapshot()
	local, ok := rep.Tier(TierLocalPrefix)
	if !ok {
		t.Fatal("local_prefix row must be present")
	}
	if uint64(local.Requests) != snap.Turns {
		t.Fatalf("local_prefix requests %d must reconcile with the aggregate turns %d", local.Requests, snap.Turns)
	}
	if local.Hits != 2 || local.Misses != 2 || local.Errors != 0 {
		t.Fatalf("local_prefix verdicts = (hits %d, misses %d, errors %d), want (2, 2, 0)", local.Hits, local.Misses, local.Errors)
	}
	if local.Hits+local.Misses != local.Requests {
		t.Fatalf("every turn must produce exactly one verdict: %d + %d != %d", local.Hits, local.Misses, local.Requests)
	}
	// The planner measures tokens, not bytes or lookup latency. That must read as "never
	// measured" (zero SIZED/TIMED accesses), never as "measured zero bytes in zero time".
	if local.SizedAccesses != 0 || local.TimedAccesses != 0 {
		t.Fatalf("the turn tap witnesses neither bytes nor latency, got sized=%d timed=%d", local.SizedAccesses, local.TimedAccesses)
	}
	if local.MeanBytes != 0 || local.MeanLatencyNanos != 0 {
		t.Fatalf("means over an empty witnessed denominator must be 0, got bytes=%v latency=%v", local.MeanBytes, local.MeanLatencyNanos)
	}
	// The tap is a memory-backed read, and it is the ONLY row the turn path opens.
	if len(local.Ops) != 1 || local.Ops[0].Op != "read" || local.Ops[0].Backend != "memory" {
		t.Fatalf("turn tap must open exactly one (read, memory) row, got %+v", local.Ops)
	}
	// The other tiers are untouched by the turn path and stay explicitly unsupported.
	for _, tier := range []CacheTier{TierSharedStore, TierProviderManaged} {
		row, ok := rep.Tier(tier)
		if !ok {
			t.Fatalf("%s row must be present", tier)
		}
		if row.Requests != 0 || row.Status != TierStatusUnsupported.String() {
			t.Fatalf("%s must stay an explicit unsupported zero, got %+v", tier, row)
		}
	}
	if rep.Total.Requests != local.Requests {
		t.Fatalf("grand total %d must equal the only fed tier %d", rep.Total.Requests, local.Requests)
	}
}

// TestUnsupportedTierIsExplicitNotSilentZero pins the distinction a bare zero row destroys:
// a tier with no collector in this build ("unsupported" — its zeros are structural) must be
// distinguishable from a tier that IS collected and simply earned nothing ("supported"), and
// from one nobody declared at all ("undeclared"). All three appear in the report; none is
// omitted.
func TestUnsupportedTierIsExplicitNotSilentZero(t *testing.T) {
	o := New()
	rep := o.TierSnapshot()
	if len(rep.Tiers) != len(AllTiers()) {
		t.Fatalf("every tier in the vocabulary must be reported, got %d of %d", len(rep.Tiers), len(AllTiers()))
	}
	want := map[string]string{
		"local_prefix":     "supported",   // fed by the turn tap in this build
		"shared_store":     "unsupported", // no collector yet: the zeros are structural
		"provider_managed": "unsupported",
	}
	for _, ts := range rep.Tiers {
		if ts.Requests != 0 {
			t.Fatalf("a fresh observer must report no accesses, got %+v", ts)
		}
		if got := want[ts.Tier]; ts.Status != got {
			t.Fatalf("%s status = %q, want %q", ts.Tier, ts.Status, got)
		}
	}

	// Declaring a tier supported flips the reading of the SAME zero row: idle, not absent.
	o.DeclareTier(TierSharedStore, TierStatusSupported)
	if row, _ := o.TierSnapshot().Tier(TierSharedStore); row.Status != "supported" || row.Requests != 0 {
		t.Fatalf("a declared-supported idle tier must report supported with zero requests, got %+v", row)
	}

	// Evidence beats declaration: a tier declared unsupported that nonetheless serves an
	// access is reported supported — a stale declaration must not label real traffic as
	// structurally impossible.
	o.DeclareTier(TierSharedStore, TierStatusUnsupported)
	o.ObserveTier(TierAccess{Tier: TierSharedStore, Op: OpRead, Outcome: OutcomeHit, Backend: BackendRemote})
	if row, _ := o.TierSnapshot().Tier(TierSharedStore); row.Status != "supported" || row.Requests != 1 {
		t.Fatalf("an observed tier must report supported whatever the declaration says, got %+v", row)
	}

	// A zero-value observer declares nothing, so its tiers are undeclared — "unknown", which
	// is neither "idle" nor "absent".
	var bare Observer
	for _, ts := range bare.TierSnapshot().Tiers {
		if ts.Status != TierStatusUndeclared.String() {
			t.Fatalf("an undeclared observer must report %q, got %q for %s", TierStatusUndeclared, ts.Status, ts.Tier)
		}
	}
}

// tierJSONKeys / tierJSONValues are the CLOSED sets the exported report may contain. Every
// string that reaches the wire must be a schema field name or a vocabulary spelling — there
// is no third category, which is what makes a leak structurally impossible rather than
// merely absent today.
var tierJSONKeys = map[string]bool{
	"tiers": true, "total": true, "tier": true, "status": true, "ops": true, "op": true, "backend": true,
	"requests": true, "hits": true, "misses": true, "errors": true,
	"bytes": true, "sized_accesses": true, "latency_ns": true, "timed_accesses": true,
	"hit_ratio": true, "mean_bytes": true, "mean_latency_ns": true, "rejected_tier_accesses": true,
}

func closedVocabularyValues() map[string]bool {
	out := map[string]bool{}
	for _, t := range AllTiers() {
		out[t.String()] = true
	}
	for _, op := range AllOps() {
		out[op.String()] = true
	}
	for _, b := range AllBackends() {
		out[b.String()] = true
	}
	for _, s := range []TierStatus{TierStatusUndeclared, TierStatusSupported, TierStatusUnsupported} {
		out[s.String()] = true
	}
	return out
}

// walkJSONStrings collects every key and every string value of a decoded JSON document.
func walkJSONStrings(v any, keys, values *[]string) {
	switch t := v.(type) {
	case map[string]any:
		for k, sub := range t {
			*keys = append(*keys, k)
			walkJSONStrings(sub, keys, values)
		}
	case []any:
		for _, sub := range t {
			walkJSONStrings(sub, keys, values)
		}
	case string:
		*values = append(*values, t)
	}
}

// TestTierReportLeaksNothingBeyondClosedVocabulary is the privacy proof the issue asks for.
// The report is generated from a workload whose surrounding session carries exactly the
// things that must never reach telemetry — a cache key, a prompt fragment, a tenant/account
// identifier, a provider response body — and the emitted JSON is then checked two ways:
// none of those strings appears, AND every string in the document is drawn from the closed
// schema/vocabulary sets. The second check is the durable one: a future free-form field
// would fail it even if this test's secrets were never routed through it.
func TestTierReportLeaksNothingBeyondClosedVocabulary(t *testing.T) {
	secrets := []string{
		"sha256:5f2b0c1d9e",                      // a cache key
		"You are a helpful assistant. The API",   // a prompt fragment
		"tenant-acme-prod",                       // a tenant identifier
		"acct_9931",                              // an account identifier
		`{"choices":[{"text":"the answer is"}]}`, // a provider response body
	}
	// The tap "knows" all of the above while it books its accesses; the record simply has
	// nowhere to put any of it.
	o := observeWorkload(tierWorkload())
	for range secrets {
		o.ObserveTier(TierAccess{Tier: TierSharedStore, Op: OpRead, Outcome: OutcomeHit, Backend: BackendRemote, Bytes: 128, BytesKnown: true})
	}

	blob, err := json.Marshal(o.TierSnapshot())
	if err != nil {
		t.Fatalf("marshal tier report: %v", err)
	}
	doc := string(blob)
	for _, s := range secrets {
		if strings.Contains(doc, s) {
			t.Fatalf("tier report leaked %q:\n%s", s, doc)
		}
	}

	var decoded any
	if err := json.Unmarshal(blob, &decoded); err != nil {
		t.Fatalf("unmarshal tier report: %v", err)
	}
	var keys, values []string
	walkJSONStrings(decoded, &keys, &values)
	for _, k := range keys {
		if !tierJSONKeys[k] {
			t.Fatalf("unexpected field %q in the tier report — the schema must stay closed:\n%s", k, doc)
		}
	}
	vocab := closedVocabularyValues()
	for _, v := range values {
		if !vocab[v] {
			t.Fatalf("string value %q is outside the closed vocabulary — a free-form field can carry a key, a prompt, or a tenant:\n%s", v, doc)
		}
	}
	if len(values) == 0 {
		t.Fatal("expected the report to carry vocabulary spellings; got none")
	}
}

// TestTierAccessHasNoFreeFormField is the structural half of the privacy proof: the record
// itself carries no string, byte slice, map, or interface field, so a caller CANNOT attach a
// cache key, a prompt fragment, a tenant identifier, or a provider payload to a tier
// observation — not even by mistake. A future field that could would fail here at once.
func TestTierAccessHasNoFreeFormField(t *testing.T) {
	rt := reflect.TypeOf(TierAccess{})
	for i := 0; i < rt.NumField(); i++ {
		f := rt.Field(i)
		switch f.Type.Kind() {
		case reflect.String, reflect.Slice, reflect.Array, reflect.Map, reflect.Interface, reflect.Pointer, reflect.Chan, reflect.Struct:
			t.Fatalf("TierAccess.%s is %s — the record must carry only numbers, closed-vocabulary enums, and their witness flags", f.Name, f.Type.Kind())
		}
	}
}

// TestObserveTierRejectsOutOfVocabularyAccess pins the closed-vocabulary rule: an access with
// any dimension outside its vocabulary is ignored WHOLE, never bucketed to a catch-all row.
// The report may therefore under-count, but it can never attribute one tier's value to
// another — the failure mode a catch-all bucket would introduce.
func TestObserveTierRejectsOutOfVocabularyAccess(t *testing.T) {
	o := New()
	o.ObserveTier(TierAccess{Tier: CacheTier(99), Op: OpRead, Outcome: OutcomeHit, Backend: BackendMemory})
	o.ObserveTier(TierAccess{Tier: TierSharedStore, Op: TierOp(7), Outcome: OutcomeHit, Backend: BackendMemory})
	o.ObserveTier(TierAccess{Tier: TierSharedStore, Op: OpRead, Outcome: TierOutcome(-1), Backend: BackendMemory})
	o.ObserveTier(TierAccess{Tier: TierSharedStore, Op: OpRead, Outcome: OutcomeHit, Backend: BackendClass(42)})
	o.ObserveTier(TierAccess{Tier: numCacheTiers, Op: OpRead, Outcome: OutcomeHit, Backend: BackendMemory})

	rep := o.TierSnapshot()
	snap := o.Snapshot()
	if rep.Total.Requests != 0 || rep.RejectedTierAccesses != 5 {
		t.Fatalf("five out-of-vocabulary accesses must be rejected whole, got %+v", rep)
	}
	if snap.RejectedTierAccesses != rep.RejectedTierAccesses {
		t.Fatalf("snapshots disagree on rejected tier accesses: stats=%d tier=%d", snap.RejectedTierAccesses, rep.RejectedTierAccesses)
	}
	encoded, err := json.Marshal(rep)
	if err != nil || !strings.Contains(string(encoded), `"rejected_tier_accesses":5`) {
		t.Fatalf("tier report JSON does not expose the rejected aggregate: %s (err=%v)", encoded, err)
	}
	for _, ts := range rep.Tiers {
		if len(ts.Ops) != 0 {
			t.Fatalf("no row may be opened by a rejected access, got %s -> %+v", ts.Tier, ts.Ops)
		}
	}

	// Stringifying a raw out-of-range value renders "unknown" rather than panicking.
	if CacheTier(99).String() != "unknown" || TierOp(7).String() != "unknown" ||
		TierOutcome(-1).String() != "unknown" || BackendClass(42).String() != "unknown" ||
		TierStatus(9).String() != "unknown" {
		t.Fatal("out-of-range vocabulary values must render \"unknown\"")
	}

	// A nil observer is safe, and reports the whole vocabulary as undeclared zeros rather
	// than a phantom split.
	var nilObs *Observer
	nilObs.ObserveTier(TierAccess{Tier: TierLocalPrefix, Op: OpRead, Outcome: OutcomeHit, Backend: BackendMemory})
	nilObs.DeclareTier(TierLocalPrefix, TierStatusSupported)
	nilRep := nilObs.TierSnapshot()
	nilSnap := nilObs.Snapshot()
	if len(nilRep.Tiers) != len(AllTiers()) || nilRep.Total.Requests != 0 || nilRep.RejectedTierAccesses != 0 || nilSnap.RejectedTierAccesses != 0 {
		t.Fatalf("a nil observer must report the vocabulary with zero traffic, got %+v", nilRep)
	}
	for _, ts := range nilRep.Tiers {
		if ts.Status != TierStatusUndeclared.String() {
			t.Fatalf("nil observer tier %s status = %q, want undeclared", ts.Tier, ts.Status)
		}
	}

	// An out-of-vocabulary declaration is ignored too.
	o.DeclareTier(CacheTier(99), TierStatusSupported)
	o.DeclareTier(TierSharedStore, TierStatus(9))
	if row, _ := o.TierSnapshot().Tier(TierSharedStore); row.Status != TierStatusUnsupported.String() {
		t.Fatalf("an invalid status must not overwrite the build declaration, got %q", row.Status)
	}
	if _, ok := o.TierSnapshot().Tier(CacheTier(99)); ok {
		t.Fatal("Tier() must not resolve a value outside the vocabulary")
	}

	// A subsequent valid access is accepted without changing the rejection aggregate.
	o.ObserveTier(TierAccess{Tier: TierSharedStore, Op: OpRead, Outcome: OutcomeHit, Backend: BackendRemote})
	rep = o.TierSnapshot()
	snap = o.Snapshot()
	if rep.Total.Requests != 1 || rep.Total.Hits != 1 || rep.RejectedTierAccesses != 5 || snap.RejectedTierAccesses != 5 {
		t.Fatalf("valid access must book once without changing rejected count: stats=%+v tier=%+v", snap, rep)
	}

	// The internal turn tap books a known-valid access directly and does not increment the
	// public-admission rejection aggregate.
	internal := New()
	internal.Observe(100, 50)
	if internal.Snapshot().RejectedTierAccesses != 0 || internal.TierSnapshot().RejectedTierAccesses != 0 {
		t.Fatal("internal valid tier observation incremented the rejection aggregate")
	}

	// Rejection accounting saturates rather than wrapping to zero.
	saturated := New()
	saturated.rejectedTierAccesses = math.MaxUint64 - 1
	invalid := TierAccess{Tier: CacheTier(99), Op: OpRead, Outcome: OutcomeHit, Backend: BackendMemory}
	saturated.ObserveTier(invalid)
	saturated.ObserveTier(invalid)
	if got := saturated.Snapshot().RejectedTierAccesses; got != math.MaxUint64 {
		t.Fatalf("rejected count wrapped instead of saturating: %d", got)
	}
	if got := saturated.TierSnapshot().RejectedTierAccesses; got != math.MaxUint64 {
		t.Fatalf("tier report rejected count wrapped instead of saturating: %d", got)
	}
}

// TestUnmeasuredDimensionIsNotAMeasuredZero pins the "not silently zero" rule on the byte and
// latency dimensions: an access that did not measure them books into Requests only, so 0
// bytes over 0 sized accesses ("never measured") stays distinguishable from 0 bytes over N
// sized accesses ("measured, and nothing moved"). A negative reading is treated as no
// measurement at all rather than as a witnessed zero.
func TestUnmeasuredDimensionIsNotAMeasuredZero(t *testing.T) {
	unmeasured := New()
	unmeasured.ObserveTier(TierAccess{Tier: TierSharedStore, Op: OpRead, Outcome: OutcomeMiss, Backend: BackendRemote})
	uRow, _ := unmeasured.TierSnapshot().Tier(TierSharedStore)

	measured := New()
	measured.ObserveTier(TierAccess{
		Tier: TierSharedStore, Op: OpRead, Outcome: OutcomeMiss, Backend: BackendRemote,
		Bytes: 0, BytesKnown: true, Latency: 0, LatencyKnown: true,
	})
	mRow, _ := measured.TierSnapshot().Tier(TierSharedStore)

	if uRow.Bytes != 0 || mRow.Bytes != 0 || uRow.Requests != mRow.Requests {
		t.Fatalf("both arms must book one zero-byte request: unmeasured=%+v measured=%+v", uRow, mRow)
	}
	if uRow.SizedAccesses != 0 || uRow.TimedAccesses != 0 {
		t.Fatalf("an unmeasured access must carry no witnessed denominator, got %+v", uRow)
	}
	if mRow.SizedAccesses != 1 || mRow.TimedAccesses != 1 {
		t.Fatalf("a measured zero must carry its witnessed denominator, got %+v", mRow)
	}

	// A negative reading is a bug, not a measurement: it is dropped rather than booked.
	neg := New()
	neg.ObserveTier(TierAccess{
		Tier: TierSharedStore, Op: OpWrite, Outcome: OutcomeHit, Backend: BackendDisk,
		Bytes: -4096, BytesKnown: true, Latency: -time.Second, LatencyKnown: true,
	})
	nRow, _ := neg.TierSnapshot().Tier(TierSharedStore)
	if nRow.Requests != 1 {
		t.Fatalf("the access itself still counts, got %+v", nRow)
	}
	if nRow.Bytes != 0 || nRow.SizedAccesses != 0 || nRow.LatencyNanos != 0 || nRow.TimedAccesses != 0 {
		t.Fatalf("negative readings must be dropped, not booked, got %+v", nRow)
	}
}

// TestTierSpecsCoverTierFieldsExactlyOnce runs the same structural coverage guard the depth
// axis uses (LMCache mp_continuous.py:99-112) over the tier registry: every tier field
// declared once, none missing, none rival.
func TestTierSpecsCoverTierFieldsExactlyOnce(t *testing.T) {
	if err := CheckCoverage(TierSpecs(), TierFields()); err != nil {
		t.Fatalf("tier specs must cover the tier schema exactly once: %v", err)
	}
	// The two registries stay disjoint: a tier field must not collide with a turn field.
	turn := map[string]bool{}
	for _, f := range SnapshotFields() {
		turn[f] = true
	}
	for _, f := range TierFields() {
		if turn[f] {
			t.Fatalf("tier field %q collides with the turn schema", f)
		}
	}
}

// TestTierReconcileAgreesAcrossCollectionPaths is the A/B parity guarantee at the registry
// level: the declarative reduction of the access stream and the imperative per-tier
// accumulator produce the same value for every covered field, so Reconcile finds nothing.
func TestTierReconcileAgreesAcrossCollectionPaths(t *testing.T) {
	specs := TierSpecs()
	events := tierWorkload()
	diffs := Reconcile(specs, events,
		NamedReporter{Name: "spec", Fold: SpecFold},
		NamedReporter{Name: "observer", Fold: TierObserverFold},
	)
	if len(diffs) != 0 {
		t.Fatalf("the tier split must reconcile with the flat reduction, got %+v", diffs)
	}
	// Spot-check the reference so "agreement" is not agreement on wrong numbers.
	r := SpecFold(specs, events)
	if r[FieldTierRequests] != 7 || r[FieldTierHits] != 5 || r[FieldTierMisses] != 1 || r[FieldTierErrors] != 1 {
		t.Fatalf("reference reduction = %+v, want 7 requests / 5 hits / 1 miss / 1 error", r)
	}
	if r[FieldTierBytes] != 141312 || r[FieldTierLatencyNanos] != float64(62600000) {
		t.Fatalf("reference reduction bytes/latency = (%v, %v), want (141312, 62600000)", r[FieldTierBytes], r[FieldTierLatencyNanos])
	}
}

// TestTierReconcileCatchesDroppedAccessDrift is the loud-failure case: an access outside the
// closed vocabulary is counted by the naive declarative reduction but REJECTED by the
// observer. The two paths therefore disagree, and Reconcile must name the fields — a real
// divergence between real collection paths, surfaced rather than silently absorbed.
func TestTierReconcileCatchesDroppedAccessDrift(t *testing.T) {
	events := append(tierWorkload(),
		Event{Kind: EventTierAccess, Access: TierAccess{Tier: CacheTier(99), Op: OpRead, Outcome: OutcomeHit, Backend: BackendMemory}})
	diffs := Reconcile(TierSpecs(), events,
		NamedReporter{Name: "spec", Fold: SpecFold},
		NamedReporter{Name: "observer", Fold: TierObserverFold},
	)
	byField := map[string]Disagreement{}
	for _, d := range diffs {
		byField[d.Field] = d
	}
	req, ok := byField[FieldTierRequests]
	if !ok {
		t.Fatalf("a rejected access must show up as a requests divergence, got %+v", diffs)
	}
	if req.Reporter != "observer" || req.Reference != 8 || req.Got != 7 {
		t.Fatalf("divergence = %+v, want observer tier_requests reference=8 got=7", req)
	}
	if _, ok := byField[FieldTierHits]; !ok {
		t.Fatalf("the rejected access's hit must also diverge, got %+v", diffs)
	}
}

// TestObserveTierConcurrent exercises the tier map under -race: many goroutines booking
// accesses and reading snapshots at once, with the totals still exact at the end. The turn
// tap runs alongside, since it books a tier record from inside the aggregate's own critical
// section — the path where a lock mistake would show up first.
func TestObserveTierConcurrent(t *testing.T) {
	o := New()
	const workers, perWorker = 8, 200
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < perWorker; i++ {
				o.ObserveTier(TierAccess{Tier: TierSharedStore, Op: OpRead, Outcome: OutcomeHit, Backend: BackendRemote, Bytes: 10, BytesKnown: true, Latency: time.Microsecond, LatencyKnown: true})
				o.Observe(100, 50) // the turn tap: books a local-prefix read under the same lock
				_ = o.TierSnapshot()
			}
		}()
	}
	wg.Wait()

	rep := o.TierSnapshot()
	shared, _ := rep.Tier(TierSharedStore)
	if shared.Requests != workers*perWorker || shared.Hits != workers*perWorker {
		t.Fatalf("shared_store = %+v, want %d hit requests", shared, workers*perWorker)
	}
	if shared.Bytes != 10*workers*perWorker || shared.LatencyNanos != uint64(time.Microsecond)*workers*perWorker {
		t.Fatalf("shared_store bytes/latency = (%d, %d), want (%d, %d)", shared.Bytes, shared.LatencyNanos, 10*workers*perWorker, uint64(time.Microsecond)*workers*perWorker)
	}
	local, _ := rep.Tier(TierLocalPrefix)
	snap := o.Snapshot()
	if uint64(local.Requests) != snap.Turns || local.Requests != workers*perWorker {
		t.Fatalf("local_prefix requests %d must equal turns %d (= %d)", local.Requests, snap.Turns, workers*perWorker)
	}
}

func TestPublicVocabularyExcludesBoundSentinels(t *testing.T) {
	if got := AllTiers(); len(got) != int(numCacheTiers) {
		t.Fatalf("AllTiers length = %d, want bound %d", len(got), numCacheTiers)
	}
	if got := AllOps(); len(got) != int(numTierOps) {
		t.Fatalf("AllOps length = %d, want bound %d", len(got), numTierOps)
	}
	if got := AllBackends(); len(got) != int(numBackendClasses) {
		t.Fatalf("AllBackends length = %d, want bound %d", len(got), numBackendClasses)
	}
	if _, ok := defaultTierStatus()[numCacheTiers]; ok {
		t.Fatal("default status includes exclusive tier bound sentinel")
	}
	if numTierOutcomes.valid() {
		t.Fatal("exclusive outcome bound sentinel is valid")
	}
}
