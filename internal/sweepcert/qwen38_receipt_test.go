package sweepcert

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// Synthetic contract witness only; these values are not hardware measurements.
func TestQwen38PrefillReportRetainsReceiptContract(t *testing.T) {
	zero := 0
	throughput := func(v float64) *float64 { return &v }
	receipt := Qwen38PrefillReceipt{
		Provenance: qwen38FixtureProvenance(), BinaryDigest: "sha256:binary", ShaderDigest: "sha256:shader",
		ChunkTokens: 512, AdmissionEvidence: "fixture admission receipt", AdmittedPromptTokens: 4,
		Samples: []Qwen38PrefillSample{
			{Source: "run-2", PromptTokens: 2, TokenIDs: []int{11, 12}, ExecutedTokenIDs: []int{11, 12}, Renderer: "chat-template-v1", RenderedDigest: "sha256:render-2", ReusedTokens: &zero, ContextTokens: 8, ReservedOutputTokens: 2, Throughput: throughput(10)},
			{Source: "run-4", PromptTokens: 4, TokenIDs: []int{21, 22, 23, 24}, ExecutedTokenIDs: []int{21, 22, 23, 24}, Renderer: "chat-template-v1", RenderedDigest: "sha256:render-4", ReusedTokens: &zero, ContextTokens: 8, ReservedOutputTokens: 2, Throughput: throughput(20)},
			{Source: "run-25000", PromptTokens: 25000, Failure: "allocation failed at 25k"},
		},
	}
	report, err := Qwen38PrefillReport(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if got := report.Evidence.Envelope.Axis.Coordinates; !reflect.DeepEqual(got, []float64{2, 4, 25000}) {
		t.Fatalf("coordinates=%v", got)
	}
	if got := report.SampleCounts; !reflect.DeepEqual(got, []int{1, 1, 0}) {
		t.Fatalf("sample counts=%v", got)
	}
	failed := report.Evidence.Points[2]
	if failed.Status != PointNotMeasured || failed.Observations["prefill_throughput"].Value != nil || failed.Observations["prefill_throughput"].Reason == "" {
		t.Fatalf("failed 25k point=%+v", failed)
	}
	raw, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), `"value":null`) || strings.Contains(string(raw), "p50") || strings.Contains(string(raw), "p90") {
		t.Fatalf("failed value must be omitted and N=1 must not promote percentiles: %s", raw)
	}
	if report.Finding.Status != FindingNotIdentifiable {
		t.Fatalf("gap finding=%+v", report.Finding)
	}

	originalDigest := report.Evidence.EnvelopeDigest
	for name, mutate := range map[string]func(*Qwen38PrefillReceipt){
		"source":    func(r *Qwen38PrefillReceipt) { r.Samples[0].Source += "-other" },
		"binary":    func(r *Qwen38PrefillReceipt) { r.BinaryDigest += "-other" },
		"shader":    func(r *Qwen38PrefillReceipt) { r.ShaderDigest += "-other" },
		"chunk":     func(r *Qwen38PrefillReceipt) { r.ChunkTokens++ },
		"admission": func(r *Qwen38PrefillReceipt) { r.AdmissionEvidence += "-other" },
	} {
		t.Run("binds_"+name, func(t *testing.T) {
			changed := receipt
			changed.Samples = append([]Qwen38PrefillSample(nil), receipt.Samples...)
			mutate(&changed)
			got, err := Qwen38PrefillReport(changed)
			if err != nil || got.Evidence.EnvelopeDigest == originalDigest {
				t.Fatalf("digest=%q err=%v", got.Evidence.EnvelopeDigest, err)
			}
		})
	}
	for name, mutate := range map[string]func(*Qwen38PrefillReceipt){
		"encoded count":    func(r *Qwen38PrefillReceipt) { r.Samples[0].TokenIDs = []int{11} },
		"executed order":   func(r *Qwen38PrefillReceipt) { r.Samples[0].ExecutedTokenIDs = []int{12, 11} },
		"unobserved reuse": func(r *Qwen38PrefillReceipt) { r.Samples[0].ReusedTokens = nil },
		"nonzero reuse":    func(r *Qwen38PrefillReceipt) { one := 1; r.Samples[0].ReusedTokens = &one },
		"memory admission": func(r *Qwen38PrefillReceipt) { r.AdmittedPromptTokens = 3 },
		"output headroom":  func(r *Qwen38PrefillReceipt) { r.Samples[1].ReservedOutputTokens = 5 },
		"failure value":    func(r *Qwen38PrefillReceipt) { r.Samples[2].Throughput = throughput(1) },
	} {
		t.Run("rejects_"+name, func(t *testing.T) {
			changed := receipt
			changed.Samples = append([]Qwen38PrefillSample(nil), receipt.Samples...)
			mutate(&changed)
			if _, err := Qwen38PrefillReport(changed); err == nil {
				t.Fatal("invalid receipt accepted")
			}
		})
	}

	receipt.Samples[0].TokenIDs[0], receipt.Samples[0].ExecutedTokenIDs[0], receipt.Samples[0].Source = 99, 99, "mutated"
	if !reflect.DeepEqual(report.Receipt.Samples[0].TokenIDs, []int{11, 12}) || report.Receipt.Samples[0].Source != "run-2" {
		t.Fatalf("caller mutation escaped into report: %+v", report.Receipt.Samples[0])
	}

	closedTail := report.Receipt
	closedTail.AdmittedPromptTokens = 6
	closedTail.Samples[2] = Qwen38PrefillSample{Source: "run-6", PromptTokens: 6, TokenIDs: []int{31, 32, 33, 34, 35, 36}, ExecutedTokenIDs: []int{31, 32, 33, 34, 35, 36}, Renderer: "chat-template-v1", RenderedDigest: "sha256:render-6", ReusedTokens: &zero, ContextTokens: 10, ReservedOutputTokens: 2, Throughput: throughput(30)}
	tailReport, err := Qwen38PrefillReport(closedTail)
	if err != nil {
		t.Fatal(err)
	}
	if tailReport.Finding.Status != FindingRightCensored || tailReport.Finding.PointID != "prompt-6" {
		t.Fatalf("terminal success finding=%+v", tailReport.Finding)
	}
}
