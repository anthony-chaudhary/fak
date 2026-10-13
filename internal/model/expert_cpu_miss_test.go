// fak-test:runtime fast est=1s lane=default
package model

import (
	"math"
	"os"
	"testing"
)

// expert_cpu_miss_test.go — adversarial tests authored independently from the
// ticket 3343 (DS4.1 P7) spec + exported API contract. These probe omitted edges
// on purpose: disabled-path neutrality, per-layer cap/overflow, quality-label
// monotonicity, ledger reconciliation, unknown-before-measurement, nil safety,
// UMA validation + arithmetic, and NLL/perplexity delta finiteness.

const ecm3343Tol = 1e-9

func ecm3343Approx(t *testing.T, name string, got, want float64) {
	t.Helper()
	if got == want {
		return
	}
	if math.IsNaN(got) || math.IsInf(got, 0) || math.IsNaN(want) || math.IsInf(want, 0) {
		t.Errorf("%s = %v, want finite %v", name, got, want)
		return
	}
	if math.Abs(got-want) > ecm3343Tol*math.Max(1, math.Abs(want)) {
		t.Errorf("%s = %v, want %v (tol %v)", name, got, want, ecm3343Tol)
	}
}

func ecm3343NoPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("%s panicked: %v", name, r)
		}
	}()
	fn()
}

func ecm3343RecencySet(c []MoEMissCandidate) map[int]int64 {
	m := make(map[int]int64, len(c))
	for _, x := range c {
		m[x.ExpertID] = x.Recency
	}
	return m
}

// --- mode parsing / string ---------------------------------------------------

func TestExpertCPUMissModeParsing3343(t *testing.T) {
	cases := []struct {
		in   string
		want ExpertCPUMissMode
		ok   bool
	}{
		{"", ExpertCPUMissOff, true},
		{"off", ExpertCPUMissOff, true},
		{"OFF", ExpertCPUMissOff, true},
		{"Off", ExpertCPUMissOff, true},
		// Leading/trailing whitespace is trimmed by design (strings.TrimSpace,
		// matching the repo's env-var convention), so these are accepted.
		{" exact", ExpertCPUMissExact, true},
		{"exact", ExpertCPUMissExact, true},
		{"EXACT", ExpertCPUMissExact, true},
		{"Exact", ExpertCPUMissExact, true},
		{"exact ", ExpertCPUMissExact, true},
		{"on", ExpertCPUMissOff, false},
		{"0", ExpertCPUMissOff, false},
		{"1", ExpertCPUMissOff, false},
		{"cpu", ExpertCPUMissOff, false},
		{"exactly", ExpertCPUMissOff, false},
	}
	for _, tc := range cases {
		got, ok := ParseExpertCPUMissMode(tc.in)
		if ok != tc.ok || got != tc.want {
			t.Errorf("ParseExpertCPUMissMode(%q) = (%v,%v), want (%v,%v)", tc.in, got, ok, tc.want, tc.ok)
		}
	}

	var zero ExpertCPUMissMode
	if zero != ExpertCPUMissOff {
		t.Errorf("zero ExpertCPUMissMode = %v, want Off", zero)
	}
	if got := zero.String(); got != "off" {
		t.Errorf("Off.String() = %q, want off", got)
	}
	if got := ExpertCPUMissExact.String(); got != "exact" {
		t.Errorf("Exact.String() = %q, want exact", got)
	}

	if DefaultExpertCPUMissPerLayer != 3 {
		t.Errorf("DefaultExpertCPUMissPerLayer = %d, want 3", DefaultExpertCPUMissPerLayer)
	}
}

func TestExpertMissQualityString3343(t *testing.T) {
	var zero ExpertMissQuality
	if zero != ExpertMissQualityExact {
		t.Errorf("zero ExpertMissQuality = %v, want Exact", zero)
	}
	cases := []struct {
		q    ExpertMissQuality
		want string
	}{
		{ExpertMissQualityExact, "exact"},
		{ExpertMissQualityNonExact, "non-exact"},
	}
	for _, tc := range cases {
		if got := tc.q.String(); got != tc.want {
			t.Errorf("ExpertMissQuality(%d).String() = %q, want %q", tc.q, got, tc.want)
		}
	}
}

// --- disabled-path neutrality -----------------------------------------------

func TestExpertCPUMissDisabledNeutrality3343(t *testing.T) {
	// Deliberately invalid profile: the disabled path must not validate it.
	badProfile := MoEBandwidthProfile{
		PCIeBandwidthBytesPerSec:      -1,
		CPUBandwidthBytesPerSec:       -2,
		ExpertSizeBytes:               -3,
		GPUComputeLatencySecPerExpert: -4,
	}
	misses := []MoEMissCandidate{
		{ExpertID: 7, Recency: 1},
		{ExpertID: 2, Recency: 9},
		{ExpertID: 5, Recency: 5},
	}
	for _, mode := range []ExpertCPUMissMode{ExpertCPUMissOff, ExpertCPUMissMode(0)} {
		p := ExpertCPUMissPolicy{Mode: mode, Profile: badProfile}
		if p.Enabled() {
			t.Fatalf("policy with mode %v reported Enabled() = true", mode)
		}
		plan, err := p.PlanExpertCPUMiss(11, misses)
		if err != nil {
			t.Fatalf("disabled PlanExpertCPUMiss returned error %v; must not validate profile", err)
		}
		if plan.Mode != ExpertCPUMissOff {
			t.Errorf("disabled plan.Mode = %v, want Off", plan.Mode)
		}
		if plan.Layer != 11 {
			t.Errorf("disabled plan.Layer = %d, want 11", plan.Layer)
		}
		if len(plan.CPUExperts) != 0 {
			t.Errorf("disabled plan CPUExperts = %v, want empty", plan.CPUExperts)
		}
		if len(plan.StreamExperts) != len(misses) {
			t.Fatalf("disabled plan StreamExperts len = %d, want %d", len(plan.StreamExperts), len(misses))
		}
		for i := range misses {
			if plan.StreamExperts[i] != misses[i] {
				t.Errorf("disabled StreamExperts[%d] = %+v, want %+v (order must be preserved)", i, plan.StreamExperts[i], misses[i])
			}
		}
		if got := plan.QualityLabel(); got != ExpertMissQualityExact {
			t.Errorf("disabled plan.QualityLabel() = %v, want Exact", got)
		}
	}
}

func TestExpertCPUMissEnabledValidatesProfile3343(t *testing.T) {
	negatives := []MoEBandwidthProfile{
		{PCIeBandwidthBytesPerSec: -1, HostCPUEnabled: true},
		{CPUBandwidthBytesPerSec: -1, HostCPUEnabled: true},
		{ExpertSizeBytes: -1, HostCPUEnabled: true},
		{GPUComputeLatencySecPerExpert: -1, HostCPUEnabled: true},
	}
	for i, prof := range negatives {
		p := ExpertCPUMissPolicy{Mode: ExpertCPUMissExact, Profile: prof}
		if _, err := p.PlanExpertCPUMiss(0, []MoEMissCandidate{{ExpertID: 1, Recency: 1}}); err == nil {
			t.Errorf("negative profile case %d: enabled PlanExpertCPUMiss returned nil error, want error", i)
		}
	}
}

// --- per-layer cap and exact partition --------------------------------------

func TestExpertCPUMissCapPartition3343(t *testing.T) {
	// PCIe bandwidth is intentionally tiny and CPU bandwidth huge so the solver
	// routes every miss to the CPU; the cap is then the only thing shaping the
	// realized partition. Overflow (lowest recency) must return to the stream.
	prof := MoEBandwidthProfile{
		PCIeBandwidthBytesPerSec: 1.0,
		CPUBandwidthBytesPerSec:  1e9,
		ExpertSizeBytes:          1e9,
		HostCPUEnabled:           true,
	}
	misses := []MoEMissCandidate{
		{ExpertID: 1, Recency: 50},
		{ExpertID: 2, Recency: 40},
		{ExpertID: 3, Recency: 30},
		{ExpertID: 4, Recency: 20},
		{ExpertID: 5, Recency: 10},
	}
	cases := []struct {
		name       string
		cap        int
		wantCPU    int
		wantStream int
	}{
		{"defaultCap", 0, 3, 2},
		{"explicitCap2", 2, 2, 3},
		{"explicitCap10", 10, 5, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := ExpertCPUMissPolicy{Mode: ExpertCPUMissExact, MaxCPUMissesPerLayer: tc.cap, Profile: prof}
			plan, err := p.PlanExpertCPUMiss(3, misses)
			if err != nil {
				t.Fatalf("PlanExpertCPUMiss: %v", err)
			}
			if len(plan.CPUExperts) != tc.wantCPU {
				t.Fatalf("CPUExperts len = %d, want %d (%+v)", len(plan.CPUExperts), tc.wantCPU, plan.CPUExperts)
			}
			if len(plan.StreamExperts) != tc.wantStream {
				t.Fatalf("StreamExperts len = %d, want %d (%+v)", len(plan.StreamExperts), tc.wantStream, plan.StreamExperts)
			}

			// Exact partition: union is a multiset equal to the input, no duplicates.
			cpu := ecm3343RecencySet(plan.CPUExperts)
			stream := ecm3343RecencySet(plan.StreamExperts)
			if len(cpu) != len(plan.CPUExperts) || len(stream) != len(plan.StreamExperts) {
				t.Fatalf("duplicate expert IDs in partition: cpu=%+v stream=%+v", plan.CPUExperts, plan.StreamExperts)
			}
			wantIDs := ecm3343RecencySet(misses)
			for id := range wantIDs {
				_, inCPU := cpu[id]
				_, inStream := stream[id]
				if inCPU == inStream {
					t.Fatalf("expert %d appears in CPU=%v stream=%v; want exactly one", id, inCPU, inStream)
				}
			}

			// Cap enforced -> overflow keeps the highest-recency CPU experts.
			if tc.wantStream > 0 {
				minCPU := int64(math.MaxInt64)
				for _, c := range plan.CPUExperts {
					if c.Recency < minCPU {
						minCPU = c.Recency
					}
				}
				for _, s := range plan.StreamExperts {
					if s.Recency >= minCPU {
						t.Errorf("overflow stream expert %+v recency >= min CPU recency %d; overflow must be lowest-recency", s, minCPU)
					}
				}
			}

			// EstimatedCompletionSec must be recomputed at the realized split.
			wantCompletion := math.Max(plan.EstimatedCPUTimeSec, plan.EstimatedStreamTimeSec)
			ecm3343Approx(t, "EstimatedCompletionSec", plan.EstimatedCompletionSec, wantCompletion)

			wantQ := prof.PCIeBandwidthBytesPerSec / (prof.PCIeBandwidthBytesPerSec + prof.CPUBandwidthBytesPerSec)
			ecm3343Approx(t, "QStar", plan.QStar, wantQ)
		})
	}
}

// --- quality labels: exact default, non-exact propagation, never upgrade ----

func TestExpertCPUMissQualityLabels3343(t *testing.T) {
	prof := MoEBandwidthProfile{
		PCIeBandwidthBytesPerSec: 8e9,
		CPUBandwidthBytesPerSec:  8e9,
		ExpertSizeBytes:          1e6,
		HostCPUEnabled:           true,
	}
	misses := []MoEMissCandidate{{ExpertID: 1, Recency: 3}, {ExpertID: 2, Recency: 2}, {ExpertID: 3, Recency: 1}}

	p := ExpertCPUMissPolicy{Mode: ExpertCPUMissExact, Profile: prof}
	plan, err := p.PlanExpertCPUMiss(0, misses)
	if err != nil {
		t.Fatalf("exact plan: %v", err)
	}
	if plan.NonExact {
		t.Errorf("exact plan.NonExact = true, want false")
	}
	if got := plan.QualityLabel(); got != ExpertMissQualityExact {
		t.Errorf("exact plan.QualityLabel() = %v, want Exact", got)
	}

	pNonExact := ExpertCPUMissPolicy{Mode: ExpertCPUMissExact, NonExact: true, Profile: prof}
	planNE, err := pNonExact.PlanExpertCPUMiss(0, misses)
	if err != nil {
		t.Fatalf("non-exact plan: %v", err)
	}
	if !planNE.NonExact {
		t.Errorf("non-exact plan.NonExact = false, want true")
	}
	if got := planNE.QualityLabel(); got != ExpertMissQualityNonExact {
		t.Errorf("non-exact plan.QualityLabel() = %v, want NonExact", got)
	}

	// Controller Stats reports non-exact iff Policy.NonExact; label must never
	// be silently upgraded back to exact.
	cExact := &ExpertCPUMissController{Policy: ExpertCPUMissPolicy{Mode: ExpertCPUMissExact, Profile: prof}}
	cExact.Observe(ExpertCPUMissExact, 1000, 0)
	if got := cExact.Stats().QualityLabel; got != "exact" {
		t.Errorf("exact controller Stats().QualityLabel = %q, want %q", got, "exact")
	}
	cNE := &ExpertCPUMissController{Policy: ExpertCPUMissPolicy{Mode: ExpertCPUMissExact, NonExact: true, Profile: prof}}
	cNE.Observe(ExpertCPUMissExact, 1000, 0)
	if got := cNE.Stats().QualityLabel; got != "non-exact" {
		t.Errorf("non-exact controller Stats().QualityLabel = %q, want %q", got, "non-exact")
	}
}

// --- ledger reconciliation ---------------------------------------------------

func TestExpertCPUMissLedgerReconciles3343(t *testing.T) {
	led := &ExpertCPUMissLedger{}
	if !led.Reconciles() {
		t.Errorf("fresh ledger Reconciles() = false, want true")
	}

	// exactly one accepted outcome each, plus every documented refusal.
	led.Observe(ExpertCPUMissExact, 100, 0)  // CPUComputed
	led.Observe(ExpertCPUMissOff, 0, 200)    // Streamed
	led.Observe(ExpertCPUMissExact, 0, 0)    // Hits
	led.Observe(ExpertCPUMissMode(99), 1, 0) // Refused: unknown mode
	led.Observe(ExpertCPUMissExact, -1, 0)   // Refused: negative nanos
	led.Observe(ExpertCPUMissExact, 0, -5)   // Refused: negative nanos
	led.Observe(ExpertCPUMissExact, 5, 5)    // Refused: both channels ran

	wantLookups, wantHits, wantCPU, wantStream, wantRefused := 7, 1, 1, 1, 4
	if led.Lookups != wantLookups {
		t.Errorf("Lookups = %d, want %d", led.Lookups, wantLookups)
	}
	if led.Hits != wantHits {
		t.Errorf("Hits = %d, want %d", led.Hits, wantHits)
	}
	if led.CPUComputed != wantCPU {
		t.Errorf("CPUComputed = %d, want %d", led.CPUComputed, wantCPU)
	}
	if led.Streamed != wantStream {
		t.Errorf("Streamed = %d, want %d", led.Streamed, wantStream)
	}
	if led.Refused != wantRefused {
		t.Errorf("Refused = %d, want %d", led.Refused, wantRefused)
	}
	if !led.Reconciles() {
		t.Errorf("Reconciles() = false: hits=%d cpu=%d stream=%d refused=%d lookups=%d",
			led.Hits, led.CPUComputed, led.Streamed, led.Refused, led.Lookups)
	}
	if got := led.Hits + led.CPUComputed + led.Streamed + led.Refused; got != led.Lookups {
		t.Errorf("outcome sum = %d, Lookups = %d", got, led.Lookups)
	}
}

func TestExpertCPUMissLedgerMillisPerToken3343(t *testing.T) {
	led := &ExpertCPUMissLedger{}

	if _, ok := led.MissPathMillisPerToken(2); ok {
		t.Errorf("MillisPerToken known before any observation, want unknown")
	}
	if _, ok := led.MissPathMillisPerToken(0); ok {
		t.Errorf("MillisPerToken with tokens=0 known, want unknown")
	}
	if _, ok := led.MissPathMillisPerToken(-1); ok {
		t.Errorf("MillisPerToken with negative tokens known, want unknown")
	}

	// 3 accepted CPU exact observations of 2ms each = 6ms total, token=2 => 3ms.
	led.Observe(ExpertCPUMissExact, 2_000_000, 0)
	led.Observe(ExpertCPUMissExact, 2_000_000, 0)
	led.Observe(ExpertCPUMissExact, 2_000_000, 0)
	got, ok := led.MissPathMillisPerToken(2)
	if !ok {
		t.Fatalf("MillisPerToken known = false after observations, want true")
	}
	ecm3343Approx(t, "MillisPerToken", got, 3.0)

	// tokens>0 but Lookups==0 stays unknown.
	if _, ok := (&ExpertCPUMissLedger{}).MissPathMillisPerToken(5); ok {
		t.Errorf("MillisPerToken known with Lookups==0, want unknown")
	}
}

// --- nil safety --------------------------------------------------------------

func TestExpertCPUMissNilSafety3343(t *testing.T) {
	var c *ExpertCPUMissController
	if c.Enabled() {
		t.Errorf("nil controller Enabled() = true, want false")
	}
	ecm3343NoPanic(t, "nil controller Stats", func() { _ = c.Stats() })
	ecm3343NoPanic(t, "nil controller Observe", func() { c.Observe(ExpertCPUMissExact, 1, 0) })
	ecm3343NoPanic(t, "nil controller Plan", func() { _, _ = c.Plan(0, []MoEMissCandidate{{ExpertID: 1}}) })

	// Observe has a pointer receiver and is expected to tolerate nil.
	var led *ExpertCPUMissLedger
	ecm3343NoPanic(t, "nil ledger Observe", func() { led.Observe(ExpertCPUMissExact, 1, 0) })
}

// --- UMA bandwidth sharing ---------------------------------------------------

func TestUMABandwidthSharingValidation3343(t *testing.T) {
	cases := []struct {
		name    string
		samples []UMABandwidthSharingSample
	}{
		{"nil", nil},
		{"empty", []UMABandwidthSharingSample{}},
		{"one", []UMABandwidthSharingSample{{CPUBytes: 1e9, CPUNanos: 1e9}}},
		{"negativeBytes", []UMABandwidthSharingSample{
			{CPUBytes: -1, CPUNanos: 1e9},
			{GPUBytes: 1e9, GPUNanos: 1e9},
		}},
		{"negativeNanos", []UMABandwidthSharingSample{
			{CPUBytes: 1e9, CPUNanos: -1},
			{GPUBytes: 1e9, GPUNanos: 1e9},
		}},
		{"bytesWithoutNanos", []UMABandwidthSharingSample{
			{CPUBytes: 1e9, CPUNanos: 0},
			{GPUBytes: 1e9, GPUNanos: 1e9},
		}},
		{"gpuBytesWithoutNanos", []UMABandwidthSharingSample{
			{CPUBytes: 1e9, CPUNanos: 1e9},
			{GPUBytes: 1e9, GPUNanos: 0},
		}},
		{"allZeroNanos", []UMABandwidthSharingSample{
			{CPUBytes: 0, CPUNanos: 0, GPUBytes: 0, GPUNanos: 0},
			{CPUBytes: 0, CPUNanos: 0, GPUBytes: 0, GPUNanos: 0},
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := MeasureUMABandwidthSharing(tc.samples); err == nil {
				t.Errorf("MeasureUMABandwidthSharing(%s) returned nil error, want error", tc.name)
			}
		})
	}
}

func TestUMABandwidthSharingArithmetic3343(t *testing.T) {
	samples := []UMABandwidthSharingSample{
		{Label: "solo-cpu", CPUBytes: 50e9, CPUNanos: 1e9, Concurrent: false},
		{Label: "solo-gpu", GPUBytes: 20e9, GPUNanos: 1e9, Concurrent: false},
		{Label: "conc-1", CPUBytes: 25e9, CPUNanos: 1e9, GPUBytes: 10e9, GPUNanos: 1e9, Concurrent: true},
		{Label: "conc-2", CPUBytes: 30e9, CPUNanos: 1e9, Concurrent: true},
	}
	// Hand-computed:
	//   soloCPU = max(50, 0) = 50 GB/s
	//   soloGPU = max(0, 20) = 20 GB/s
	//   concCPU = max(25, 30) = 30 GB/s
	//   concGPU = max(10, 0) = 10 GB/s
	//   ratios  = 30/50 = 0.6 ; 10/20 = 0.5
	//   shared  = (25+10+30)e9 bytes / (1+1)e9 ns * 1e9 = 32.5 GB/s
	receipt, err := MeasureUMABandwidthSharing(samples)
	if err != nil {
		t.Fatalf("MeasureUMABandwidthSharing: %v", err)
	}
	if receipt.Schema != UMABandwidthSharingSchema {
		t.Errorf("Schema = %q, want %q", receipt.Schema, UMABandwidthSharingSchema)
	}
	if receipt.Samples != len(samples) {
		t.Errorf("Samples = %d, want %d", receipt.Samples, len(samples))
	}
	if receipt.Measurement != ExpertCacheMeasurementSimulated {
		t.Errorf("Measurement = %q, want %q", receipt.Measurement, ExpertCacheMeasurementSimulated)
	}
	ecm3343Approx(t, "SoloCPUGbps", receipt.SoloCPUGbps, 50)
	ecm3343Approx(t, "SoloGPUGbps", receipt.SoloGPUGbps, 20)
	ecm3343Approx(t, "ConcurrentCPUGbps", receipt.ConcurrentCPUGbps, 30)
	ecm3343Approx(t, "ConcurrentGPUGbps", receipt.ConcurrentGPUGbps, 10)
	ecm3343Approx(t, "CPUContentionRatio", receipt.CPUContentionRatio, 0.6)
	ecm3343Approx(t, "GPUContentionRatio", receipt.GPUContentionRatio, 0.5)
	ecm3343Approx(t, "SharedBandwidthGbps", receipt.SharedBandwidthGbps, 32.5)
}

func TestUMABandwidthSharingZeroSolo3343(t *testing.T) {
	// No solo samples: solo rates are 0 and ratios must be 0 rather than NaN/Inf.
	samples := []UMABandwidthSharingSample{
		{Label: "conc-a", CPUBytes: 10e9, CPUNanos: 1e9, GPUBytes: 5e9, GPUNanos: 1e9, Concurrent: true},
		{Label: "conc-b", CPUBytes: 20e9, CPUNanos: 1e9, GPUBytes: 8e9, GPUNanos: 1e9, Concurrent: true},
	}
	receipt, err := MeasureUMABandwidthSharing(samples)
	if err != nil {
		t.Fatalf("MeasureUMABandwidthSharing: %v", err)
	}
	if receipt.SoloCPUGbps != 0 || receipt.SoloGPUGbps != 0 {
		t.Errorf("solo rates = (%v,%v), want (0,0)", receipt.SoloCPUGbps, receipt.SoloGPUGbps)
	}
	if receipt.CPUContentionRatio != 0 || receipt.GPUContentionRatio != 0 {
		t.Errorf("ratios = (%v,%v), want (0,0)", receipt.CPUContentionRatio, receipt.GPUContentionRatio)
	}
	// max over concurrent: cpu=20, gpu=8; max nanos both 1e9.
	ecm3343Approx(t, "ConcurrentCPUGbps", receipt.ConcurrentCPUGbps, 20)
	ecm3343Approx(t, "ConcurrentGPUGbps", receipt.ConcurrentGPUGbps, 8)
	// shared = sum(conc bytes)/sum(max(cpu,gpu) nanos) = (10+5+20+8)e9 / 2e9 = 21.5.
	ecm3343Approx(t, "SharedBandwidthGbps", receipt.SharedBandwidthGbps, 21.5)
}

// --- NLL / perplexity quality panel -----------------------------------------

func TestMissQualityPanelDelta3343(t *testing.T) {
	known := MissQualityPanel{BaselineNLL: 10, BaselineTokens: 5, CandidateNLL: 12, CandidateTokens: 4}
	// NLL/token: 12/4 - 10/5 = 3 - 2 = 1.
	dNLL, ok := known.NLLDeltaPerToken()
	if !ok {
		t.Fatalf("NLLDeltaPerToken known = false, want true")
	}
	ecm3343Approx(t, "NLLDeltaPerToken", dNLL, 1.0)

	// perplexity delta = exp(3) - exp(2).
	dPPL, ok := known.PerplexityDelta()
	if !ok {
		t.Fatalf("PerplexityDelta known = false, want true")
	}
	wantPPL := math.Exp(3) - math.Exp(2)
	ecm3343Approx(t, "PerplexityDelta", dPPL, wantPPL)

	unknown := []struct {
		name string
		p    MissQualityPanel
	}{
		{"zeroBaseTokens", MissQualityPanel{BaselineNLL: 1, BaselineTokens: 0, CandidateNLL: 1, CandidateTokens: 1}},
		{"zeroCandTokens", MissQualityPanel{BaselineNLL: 1, BaselineTokens: 1, CandidateNLL: 1, CandidateTokens: 0}},
		{"negBaseTokens", MissQualityPanel{BaselineNLL: 1, BaselineTokens: -1, CandidateNLL: 1, CandidateTokens: 1}},
		{"negCandTokens", MissQualityPanel{BaselineNLL: 1, BaselineTokens: 1, CandidateNLL: 1, CandidateTokens: -1}},
	}
	for _, tc := range unknown {
		if _, ok := tc.p.NLLDeltaPerToken(); ok {
			t.Errorf("%s: NLLDeltaPerToken known = true, want false", tc.name)
		}
		if _, ok := tc.p.PerplexityDelta(); ok {
			t.Errorf("%s: PerplexityDelta known = true, want false", tc.name)
		}
	}

	nonFinite := []struct {
		name string
		p    MissQualityPanel
	}{
		{"infCandNLL", MissQualityPanel{BaselineNLL: 0, BaselineTokens: 1, CandidateNLL: math.Inf(1), CandidateTokens: 1}},
		{"nanCandNLL", MissQualityPanel{BaselineNLL: 0, BaselineTokens: 1, CandidateNLL: math.NaN(), CandidateTokens: 1}},
		{"overflowExp", MissQualityPanel{BaselineNLL: 0, BaselineTokens: 1, CandidateNLL: 1e308, CandidateTokens: 1}},
		{"infBaseNLL", MissQualityPanel{BaselineNLL: math.Inf(1), BaselineTokens: 1, CandidateNLL: 1, CandidateTokens: 1}},
	}
	for _, tc := range nonFinite {
		if _, ok := tc.p.PerplexityDelta(); ok {
			t.Errorf("%s: PerplexityDelta known = true, want false for non-finite", tc.name)
		}
	}
}

func TestQualityPanelLabelAndConstruction3343(t *testing.T) {
	// NLL = -sumLogprob; tokens must be clamped at 0; label carried verbatim.
	q := NewMissQualityPanel(-10, 5, -12, 4, ExpertMissQualityExact)
	if q.Panel.BaselineNLL != 10 || q.Panel.CandidateNLL != 12 {
		t.Errorf("constructed NLL = (%v,%v), want (10,12)", q.Panel.BaselineNLL, q.Panel.CandidateNLL)
	}
	if q.Panel.BaselineTokens != 5 || q.Panel.CandidateTokens != 4 {
		t.Errorf("constructed tokens = (%d,%d), want (5,4)", q.Panel.BaselineTokens, q.Panel.CandidateTokens)
	}
	if q.Label != ExpertMissQualityExact {
		t.Errorf("label = %v, want exact", q.Label)
	}
	if !q.Exact() || !q.Verdict() {
		t.Errorf("exact panel Exact=%v Verdict=%v, want true/true", q.Exact(), q.Verdict())
	}

	// Negative tokens clamp to 0.
	clamped := NewMissQualityPanel(-10, -5, -12, 3, ExpertMissQualityExact)
	if clamped.Panel.BaselineTokens != 0 {
		t.Errorf("clamped BaselineTokens = %d, want 0", clamped.Panel.BaselineTokens)
	}
	if clamped.Panel.CandidateTokens != 3 {
		t.Errorf("clamped CandidateTokens = %d, want 3", clamped.Panel.CandidateTokens)
	}
	if _, ok := clamped.Panel.NLLDeltaPerToken(); ok {
		t.Errorf("clamped NLLDeltaPerToken known = true, want false (base tokens 0)")
	}

	// Label never upgraded: a non-exact panel stays non-exact and inadmissible.
	ne := NewMissQualityPanel(-10, 5, -12, 4, ExpertMissQualityNonExact)
	if ne.Label != ExpertMissQualityNonExact {
		t.Errorf("non-exact label = %v, want non-exact", ne.Label)
	}
	if ne.Exact() {
		t.Errorf("non-exact panel Exact() = true, want false")
	}
	if ne.Verdict() {
		t.Errorf("non-exact panel Verdict() = true, want false (not admissible as exact evidence)")
	}
}

// --- env-flag binding (DoD item 2) -----------------------------------------

func TestExpertCPUMissEnvFlag3343(t *testing.T) {
	cases := []struct {
		value   string
		want    ExpertCPUMissMode
		wantOK  bool
		unsetIt bool
	}{
		{"off", ExpertCPUMissOff, true, false},
		{"OFF", ExpertCPUMissOff, true, false},
		{"", ExpertCPUMissOff, true, false},
		{"exact", ExpertCPUMissExact, true, false},
		{"EXACT", ExpertCPUMissExact, true, false},
		{" exact ", ExpertCPUMissExact, true, false},
		{"bogus", ExpertCPUMissOff, false, false},
		{"", ExpertCPUMissOff, true, true}, // truly unset
	}
	for _, tc := range cases {
		name := tc.value
		if tc.unsetIt {
			name = "<unset>"
		}
		t.Run(name, func(t *testing.T) {
			if tc.unsetIt {
				t.Setenv(ExpertCPUMissEnv, "x")
				os.Unsetenv(ExpertCPUMissEnv)
			} else {
				t.Setenv(ExpertCPUMissEnv, tc.value)
			}
			got, ok := ExpertCPUMissModeFromEnv()
			if got != tc.want || ok != tc.wantOK {
				t.Errorf("ExpertCPUMissModeFromEnv() = (%v,%v), want (%v,%v)", got, ok, tc.want, tc.wantOK)
			}
		})
	}

	profile := MoEBandwidthProfile{PCIeBandwidthBytesPerSec: 2e9, CPUBandwidthBytesPerSec: 50e9, ExpertSizeBytes: 1 << 20, HostCPUEnabled: true}

	t.Setenv(ExpertCPUMissEnv, "off")
	if p, err := ExpertCPUMissPolicyFromEnv(profile); err != nil || p.Enabled() {
		t.Errorf("off: policy=%+v err=%v, want disabled+nil", p, err)
	}

	t.Setenv(ExpertCPUMissEnv, "exact")
	if p, err := ExpertCPUMissPolicyFromEnv(profile); err != nil || !p.Enabled() {
		t.Errorf("exact: policy=%+v err=%v, want enabled+nil", p, err)
	}

	t.Setenv(ExpertCPUMissEnv, "bogus")
	if _, err := ExpertCPUMissPolicyFromEnv(profile); err == nil {
		t.Errorf("bogus: err=nil, want fail-closed error")
	}
}

// --- the swapped-channel regression guard ------------------------------------

func TestExpertCPUMissChannelTimesNotSwapped3343(t *testing.T) {
	profile := MoEBandwidthProfile{
		PCIeBandwidthBytesPerSec: 2e9,
		CPUBandwidthBytesPerSec:  50e9,
		ExpertSizeBytes:          1 << 30,
		HostCPUEnabled:           true,
	}
	policy := ExpertCPUMissPolicy{Mode: ExpertCPUMissExact, MaxCPUMissesPerLayer: 100, Profile: profile}
	misses := make([]MoEMissCandidate, 8)
	for i := range misses {
		misses[i] = MoEMissCandidate{ExpertID: i, Recency: int64(i * 10)}
	}
	plan, err := policy.PlanExpertCPUMiss(0, misses)
	if err != nil {
		t.Fatalf("PlanExpertCPUMiss: %v", err)
	}
	if len(plan.CPUExperts)+len(plan.StreamExperts) != len(misses) {
		t.Fatalf("split %d+%d != %d", len(plan.CPUExperts), len(plan.StreamExperts), len(misses))
	}
	seen := map[int]bool{}
	for _, m := range append(append([]MoEMissCandidate{}, plan.CPUExperts...), plan.StreamExperts...) {
		if seen[m.ExpertID] {
			t.Errorf("expert %d appears in both arms", m.ExpertID)
		}
		seen[m.ExpertID] = true
	}
	// Independent arithmetic from the profile and the realized counts. A swap of the
	// two plan fields fails both of these.
	wantStream := float64(len(plan.StreamExperts)) * float64(profile.ExpertSizeBytes) / profile.PCIeBandwidthBytesPerSec
	wantCPU := float64(len(plan.CPUExperts)) * float64(profile.ExpertSizeBytes) / profile.CPUBandwidthBytesPerSec
	ecm3343Approx(t, "EstimatedStreamTimeSec", plan.EstimatedStreamTimeSec, wantStream)
	ecm3343Approx(t, "EstimatedCPUTimeSec", plan.EstimatedCPUTimeSec, wantCPU)
	if len(plan.StreamExperts) != 0 && len(plan.CPUExperts) != 0 && wantStream == wantCPU {
		t.Fatalf("degenerate asymmetric profile: times equal, cannot detect a swap")
	}
}

// --- refused rows contribute nothing to the ms/token numerator ---------------

func TestExpertCPUMissRefusedDoesNotContaminate3343(t *testing.T) {
	var refusedOnly ExpertCPUMissLedger
	refusedOnly.Observe(ExpertCPUMissMode(99), 100, 0) // unknown mode -> refused
	refusedOnly.Observe(ExpertCPUMissOff, 5, 5)        // both channels -> refused
	if refusedOnly.Refused != 2 || !refusedOnly.Reconciles() {
		t.Fatalf("refused-only ledger: refused=%d reconciles=%v, want 2/true", refusedOnly.Refused, refusedOnly.Reconciles())
	}
	if refusedOnly.CPUNanos != 0 || refusedOnly.StreamNanos != 0 {
		t.Errorf("refused rows added nanos: cpu=%d stream=%d, want 0/0", refusedOnly.CPUNanos, refusedOnly.StreamNanos)
	}
	if ms, known := refusedOnly.MissPathMillisPerToken(1); known {
		t.Errorf("refused-only MissPathMillisPerToken known=true (%.6f), want false", ms)
	}

	var accepted ExpertCPUMissLedger
	accepted.Observe(ExpertCPUMissExact, 1000, 0)
	ms, known := accepted.MissPathMillisPerToken(1)
	if !known {
		t.Fatalf("accepted ledger MissPathMillisPerToken known=false, want true")
	}
	ecm3343Approx(t, "MissPathMillisPerToken", ms, 1000.0/1e6)
}

// --- Verdict requires an actually measured panel ------------------------------

func TestQualityPanelVerdictRequiresMeasurement3343(t *testing.T) {
	var zero QualityPanel
	if zero.Exact() != true {
		t.Errorf("zero panel Exact()=false, want true (label is the zero value)")
	}
	if zero.Verdict() {
		t.Errorf("unmeasured zero panel Verdict()=true, want false")
	}

	unmeasured := NewMissQualityPanel(0, 0, 0, 0, ExpertMissQualityExact)
	if unmeasured.Verdict() {
		t.Errorf("unmeasured panel Verdict()=true, want false")
	}

	measured := NewMissQualityPanel(-10, 5, -12, 4, ExpertMissQualityExact)
	if !measured.Verdict() {
		t.Errorf("measured exact panel Verdict()=false, want true")
	}

	lossy := NewMissQualityPanel(-10, 5, -12, 4, ExpertMissQualityNonExact)
	if lossy.Verdict() {
		t.Errorf("measured non-exact panel Verdict()=true, want false")
	}
}

func TestExpertMissQualityUnknownString3343(t *testing.T) {
	if got := ExpertMissQuality(99).String(); got != "unknown" {
		t.Errorf("ExpertMissQuality(99).String() = %q, want \"unknown\"", got)
	}
	if got := ExpertMissQualityExact.String(); got != "exact" {
		t.Errorf("Exact.String() = %q, want \"exact\"", got)
	}
	if got := ExpertMissQualityNonExact.String(); got != "non-exact" {
		t.Errorf("NonExact.String() = %q, want \"non-exact\"", got)
	}
}
