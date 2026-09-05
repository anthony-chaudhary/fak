package metrics

import (
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestTrustViolationSubtypesClosedAndNormalized(t *testing.T) {
	subtypes := Subtypes()
	if len(subtypes) != 5 {
		t.Fatalf("expected 5 subtypes, got %d", len(subtypes))
	}

	expected := []TrustViolationSubtype{
		SubtypeInjectionQuarantine,
		SubtypeIFCSink,
		SubtypeWitnessRefuted,
		SubtypeResidencyMismatch,
		SubtypeGeneric,
	}
	for i, exp := range expected {
		if subtypes[i] != exp {
			t.Errorf("subtype[%d] = %q, want %q", i, subtypes[i], exp)
		}
		if !exp.Valid() {
			t.Errorf("subtype %q should be valid", exp)
		}
	}

	// Test normalization and cardinality control.
	cases := []struct {
		input string
		want  TrustViolationSubtype
	}{
		{"injection_quarantine", SubtypeInjectionQuarantine},
		{"INJECTION_QUARANTINE", SubtypeInjectionQuarantine},
		{" ifc_sink ", SubtypeIFCSink},
		{"witness_refuted", SubtypeWitnessRefuted},
		{"residency_mismatch", SubtypeResidencyMismatch},
		{"generic", SubtypeGeneric},
		// Invalid / high-cardinality values must fold to generic
		{"", SubtypeGeneric},
		{"unknown_subtype", SubtypeGeneric},
		{"user_session_12345", SubtypeGeneric},
		{"prompt_injection_variant_4", SubtypeGeneric},
		{"malicious_code_exfil", SubtypeGeneric},
	}

	for _, tc := range cases {
		got := NormalizeTrustViolationSubtype(tc.input)
		if got != tc.want {
			t.Errorf("NormalizeTrustViolationSubtype(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestClassifyTrustViolationSubtypes(t *testing.T) {
	tests := []struct {
		name    string
		verdict abi.Verdict
		want    TrustViolationSubtype
	}{
		{
			name: "injection quarantine via VerdictQuarantine",
			verdict: abi.Verdict{
				Kind:   abi.VerdictQuarantine,
				Reason: abi.ReasonTrustViolation,
				By:     "ctxmmu",
			},
			want: SubtypeInjectionQuarantine,
		},
		{
			name: "injection quarantine via normgate",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonPromptInjection,
				By:     "normgate",
			},
			want: SubtypeInjectionQuarantine,
		},
		{
			name: "injection quarantine via wirescreen",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonTrustViolation,
				By:     "wirescreen",
			},
			want: SubtypeInjectionQuarantine,
		},
		{
			name: "injection quarantine via semantic screen",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonTrustViolation,
				By:     "semantic:model_screener",
			},
			want: SubtypeInjectionQuarantine,
		},
		{
			name: "injection quarantine via meta quarantine_id",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonTrustViolation,
				By:     "custom_gate",
				Meta:   map[string]string{"quarantine_id": "q42"},
			},
			want: SubtypeInjectionQuarantine,
		},
		{
			name: "injection quarantine via meta threat",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonTrustViolation,
				By:     "custom_gate",
				Meta:   map[string]string{"threat": "prompt_injection"},
			},
			want: SubtypeInjectionQuarantine,
		},
		{
			name: "ifc sink via ifc-sink adjudicator",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonTrustViolation,
				By:     "ifc-sink",
				Meta:   map[string]string{"sink": "send_email"},
			},
			want: SubtypeIFCSink,
		},
		{
			name: "ifc sink via ifc-stamp",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonTrustViolation,
				By:     "ifc-stamp",
			},
			want: SubtypeIFCSink,
		},
		{
			name: "ifc sink via ReasonTaintEgress",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonTaintEgress,
				By:     "egress_monitor",
			},
			want: SubtypeIFCSink,
		},
		{
			name: "ifc sink via meta taint_source_tool",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonTrustViolation,
				By:     "custom_gate",
				Meta:   map[string]string{"taint_source_tool": "fetch_url", "ifc_sink": "network"},
			},
			want: SubtypeIFCSink,
		},
		{
			name: "witness refuted via ReasonIntegrityRefuted",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonIntegrityRefuted,
				By:     "kernel",
			},
			want: SubtypeWitnessRefuted,
		},
		{
			name: "witness refuted via require-witness with refuted claim",
			verdict: abi.Verdict{
				Kind:    abi.VerdictDeny,
				Reason:  abi.ReasonTrustViolation,
				By:      "require-witness",
				Payload: abi.WitnessPayload{Claim: "the declared witness refuted the assumption"},
			},
			want: SubtypeWitnessRefuted,
		},
		{
			name: "witness refuted via witness-gate",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonTrustViolation,
				By:     "witness-gate",
			},
			want: SubtypeWitnessRefuted,
		},
		{
			name: "witness refuted via meta witness_verdict=refuted",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonTrustViolation,
				By:     "custom_oracle",
				Meta:   map[string]string{"witness_verdict": "refuted"},
			},
			want: SubtypeWitnessRefuted,
		},
		{
			name: "residency mismatch via engine-residency",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonTrustViolation,
				By:     "engine-residency",
			},
			want: SubtypeResidencyMismatch,
		},
		{
			name: "residency mismatch via ReasonScopeCrossing",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonScopeCrossing,
				By:     "pdp",
			},
			want: SubtypeResidencyMismatch,
		},
		{
			name: "residency mismatch via meta engine_route and scope",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonTrustViolation,
				By:     "router",
				Meta:   map[string]string{"engine_route": "remote:gpt-5", "scope": "tenant", "remote": "true"},
			},
			want: SubtypeResidencyMismatch,
		},
		{
			name: "generic fallback on unclassified trust violation",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonTrustViolation,
				By:     "unclassified_adjudicator",
			},
			want: SubtypeGeneric,
		},
		{
			name: "explicit refusal_subtype in meta wins",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonTrustViolation,
				By:     "some_tool",
				Meta:   map[string]string{"refusal_subtype": "ifc_sink"},
			},
			want: SubtypeIFCSink,
		},
		{
			name: "explicit invalid refusal_subtype in meta falls back safely",
			verdict: abi.Verdict{
				Kind:   abi.VerdictDeny,
				Reason: abi.ReasonTrustViolation,
				By:     "unknown",
				Meta:   map[string]string{"refusal_subtype": "invalid_cardinality_exploit"},
			},
			want: SubtypeGeneric,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ClassifyTrustViolation(tc.verdict)
			if got != tc.want {
				t.Fatalf("ClassifyTrustViolation() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestClassifyTrustViolationDetailsAndAttributes(t *testing.T) {
	// Test detail-based classification.
	d1 := ClassifyTrustViolationDetails("custom", nil, "poisoned tool result quarantined")
	if d1 != SubtypeInjectionQuarantine {
		t.Fatalf("expected injection_quarantine, got %q", d1)
	}

	d2 := ClassifyTrustViolationDetails("custom", nil, "IFC egress block: sink fed tainted data")
	if d2 != SubtypeIFCSink {
		t.Fatalf("expected ifc_sink, got %q", d2)
	}

	d3 := ClassifyTrustViolationDetails("custom", nil, "witness actively refuted the commit")
	if d3 != SubtypeWitnessRefuted {
		t.Fatalf("expected witness_refuted, got %q", d3)
	}

	d4 := ClassifyTrustViolationDetails("custom", nil, "tenant payload routed to remote engine violates residency")
	if d4 != SubtypeResidencyMismatch {
		t.Fatalf("expected residency_mismatch, got %q", d4)
	}

	d5 := ClassifyTrustViolationDetails("custom", nil, "generic policy block")
	if d5 != SubtypeGeneric {
		t.Fatalf("expected generic, got %q", d5)
	}

	// Test attribute map classification.
	a1 := ClassifyTrustViolationAttributes(map[string]string{
		"by":     "ifc-sink",
		"detail": "taint flow violation",
	})
	if a1 != SubtypeIFCSink {
		t.Fatalf("expected ifc_sink, got %q", a1)
	}

	aNil := ClassifyTrustViolationAttributes(nil)
	if aNil != SubtypeGeneric {
		t.Fatalf("expected generic for nil attributes, got %q", aNil)
	}
}

func TestTrustViolationMetricLabelEmission(t *testing.T) {
	recorder := NewTrustViolationRecorder()

	// Record occurrences for all subtypes.
	recorder.Record(abi.Verdict{
		Kind:   abi.VerdictQuarantine,
		Reason: abi.ReasonTrustViolation,
		By:     "ctxmmu",
	})
	recorder.Record(abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonPromptInjection,
		By:     "normgate",
	})
	recorder.Record(abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonTrustViolation,
		By:     "ifc-sink",
	})
	recorder.Record(abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonIntegrityRefuted,
		By:     "kernel",
	})
	recorder.Record(abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonScopeCrossing,
		By:     "engine-residency",
	})
	recorder.Record(abi.Verdict{
		Kind:   abi.VerdictDeny,
		Reason: abi.ReasonTrustViolation,
		By:     "unclassified_component",
	})

	rep := recorder.Report()
	if rep.Total != 6 {
		t.Fatalf("expected Total = 6, got %d", rep.Total)
	}

	if rep.Tally(SubtypeInjectionQuarantine) != 2 {
		t.Errorf("injection_quarantine tally = %d, want 2", rep.Tally(SubtypeInjectionQuarantine))
	}
	if rep.Tally(SubtypeIFCSink) != 1 {
		t.Errorf("ifc_sink tally = %d, want 1", rep.Tally(SubtypeIFCSink))
	}
	if rep.Tally(SubtypeWitnessRefuted) != 1 {
		t.Errorf("witness_refuted tally = %d, want 1", rep.Tally(SubtypeWitnessRefuted))
	}
	if rep.Tally(SubtypeResidencyMismatch) != 1 {
		t.Errorf("residency_mismatch tally = %d, want 1", rep.Tally(SubtypeResidencyMismatch))
	}
	if rep.Tally(SubtypeGeneric) != 1 {
		t.Errorf("generic tally = %d, want 1", rep.Tally(SubtypeGeneric))
	}

	// Verify Prometheus text emission.
	promText := rep.Prometheus()
	expectedLines := []string{
		`fak_trust_violations_total{refusal_subtype="injection_quarantine"} 2`,
		`fak_trust_violations_total{refusal_subtype="ifc_sink"} 1`,
		`fak_trust_violations_total{refusal_subtype="witness_refuted"} 1`,
		`fak_trust_violations_total{refusal_subtype="residency_mismatch"} 1`,
		`fak_trust_violations_total{refusal_subtype="generic"} 1`,
	}
	for _, line := range expectedLines {
		if !strings.Contains(promText, line) {
			t.Errorf("Prometheus text missing line %q; got:\n%s", line, promText)
		}
	}

	// Verify OpenMetricFamilies rendering.
	families := rep.OpenMetricFamilies()
	if len(families) != 1 {
		t.Fatalf("expected 1 OpenMetricFamily, got %d", len(families))
	}
	if families[0].Name != "fak_trust_violations_total" {
		t.Errorf("family name = %q, want fak_trust_violations_total", families[0].Name)
	}
	if families[0].Type != OpenMetricCounter {
		t.Errorf("family type = %q, want counter", families[0].Type)
	}
	if len(families[0].Samples) != 5 {
		t.Errorf("expected 5 samples, got %d", len(families[0].Samples))
	}

	// Render through RenderOpenMetricsText to verify spec compliance.
	rendered, err := RenderOpenMetricsText(families)
	if err != nil {
		t.Fatalf("RenderOpenMetricsText failed: %v", err)
	}
	renderedStr := string(rendered)
	if !strings.Contains(renderedStr, "# TYPE fak_trust_violations_total counter") {
		t.Errorf("missing TYPE declaration in rendered OpenMetrics text")
	}
	if !strings.Contains(renderedStr, "# EOF") {
		t.Errorf("missing EOF in rendered OpenMetrics text")
	}

	// Verify OpenMetricFamiliesWithReason includes reason="TRUST_VIOLATION".
	reasonFamilies := rep.OpenMetricFamiliesWithReason()
	renderedReason, err := RenderOpenMetricsText(reasonFamilies)
	if err != nil {
		t.Fatalf("RenderOpenMetricsText(WithReason) failed: %v", err)
	}
	renderedReasonStr := string(renderedReason)
	if !strings.Contains(renderedReasonStr, `reason="TRUST_VIOLATION"`) {
		t.Errorf("rendered OpenMetrics text missing reason=\"TRUST_VIOLATION\" label: %s", renderedReasonStr)
	}
	if !strings.Contains(renderedReasonStr, `refusal_subtype="injection_quarantine"`) {
		t.Errorf("rendered OpenMetrics text missing refusal_subtype=\"injection_quarantine\" label: %s", renderedReasonStr)
	}
}

func TestTrustViolationRecorderSkipsAllowAndDefer(t *testing.T) {
	recorder := NewTrustViolationRecorder()

	subAllow := recorder.Record(abi.Verdict{Kind: abi.VerdictAllow})
	if subAllow != "" {
		t.Errorf("expected empty string on VerdictAllow, got %q", subAllow)
	}

	subDefer := recorder.Record(abi.Verdict{Kind: abi.VerdictDefer})
	if subDefer != "" {
		t.Errorf("expected empty string on VerdictDefer, got %q", subDefer)
	}

	rep := recorder.Report()
	if rep.Total != 0 {
		t.Errorf("expected Total = 0 after allow/defer, got %d", rep.Total)
	}
	if len(rep.BySubtype) != 0 {
		t.Errorf("expected 0 subtype tallies, got %d", len(rep.BySubtype))
	}
}

func TestTrustViolationRecorderConcurrency(t *testing.T) {
	recorder := NewTrustViolationRecorder()
	var wg sync.WaitGroup
	workers := 10
	iterations := 100

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			sub := Subtypes()[workerID%len(Subtypes())]
			for j := 0; j < iterations; j++ {
				recorder.RecordSubtype(sub, 1)
			}
		}(i)
	}

	wg.Wait()

	rep := recorder.Report()
	expectedTotal := uint64(workers * iterations)
	if rep.Total != expectedTotal {
		t.Fatalf("concurrent total = %d, want %d", rep.Total, expectedTotal)
	}
}

func TestFoldTrustViolations(t *testing.T) {
	events := []TrustViolationEvent{
		{Subtype: SubtypeInjectionQuarantine, Count: 3},
		{Subtype: SubtypeIFCSink, Count: 2},
		{Subtype: TrustViolationSubtype("invalid_subtype_attack"), Count: 5},
	}

	rep := FoldTrustViolations(events)
	if rep.Total != 10 {
		t.Fatalf("expected total = 10, got %d", rep.Total)
	}
	if rep.Tally(SubtypeInjectionQuarantine) != 3 {
		t.Errorf("injection_quarantine = %d, want 3", rep.Tally(SubtypeInjectionQuarantine))
	}
	if rep.Tally(SubtypeIFCSink) != 2 {
		t.Errorf("ifc_sink = %d, want 2", rep.Tally(SubtypeIFCSink))
	}
	// Invalid subtype must have folded to generic.
	if rep.Tally(SubtypeGeneric) != 5 {
		t.Errorf("generic = %d, want 5", rep.Tally(SubtypeGeneric))
	}
}

func TestTrustViolationLabelsHelpers(t *testing.T) {
	labels := TrustViolationLabels(SubtypeIFCSink)
	if len(labels) != 1 || labels[0].Name != "refusal_subtype" || labels[0].Value != "ifc_sink" {
		t.Fatalf("TrustViolationLabels unexpected: %+v", labels)
	}

	mapLabels := TrustViolationMetricLabels(SubtypeWitnessRefuted)
	if mapLabels[ReasonLabel] != "TRUST_VIOLATION" || mapLabels[RefusalSubtypeLabelKey] != "witness_refuted" {
		t.Fatalf("TrustViolationMetricLabels unexpected: %+v", mapLabels)
	}
}
