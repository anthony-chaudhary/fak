package gateway

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

// TestPrefillRouteFromReceiptRecordsDeclineReason pins that a declined
// whole-sequence prefill reaches the perf row with its stable decline reason, so
// a scalar token-replay turn is attributable from the ledger alone.
func TestPrefillRouteFromReceiptRecordsDeclineReason(t *testing.T) {
	receipt := &model.NativeInferenceReceipt{
		Qwen35SequencePrefillRoute: &model.NativeSequencePrefillRouteReceipt{
			PrefillCalls:  1,
			ObservedCalls: 1,
			Complete:      true,
			Status: &model.Qwen35SequencePrefillRouteStatus{
				RequestedPath:  compute.Qwen35SequencePrefillPath,
				EffectivePath:  model.Qwen35SequencePrefillFallbackPath,
				DeclineReason:  model.Qwen35SequencePrefillDeclineEmbeddingCap,
				FallbackActive: true,
			},
		},
	}
	got := prefillRouteFromReceipt(receipt)
	if got == nil {
		t.Fatal("declined route produced no perf-row prefill route")
	}
	if got.EffectivePath != model.Qwen35SequencePrefillFallbackPath || got.DeclineReason != model.Qwen35SequencePrefillDeclineEmbeddingCap || !got.FallbackActive || got.NativePerformanceQualifying {
		t.Fatalf("perf-row route=%+v, want the recorded decline", got)
	}
}

// TestPrefillRouteFromReceiptOmitsAbsentRoute pins that a receipt with no route
// block yields no perf-row field, rather than a fabricated route.
func TestPrefillRouteFromReceiptOmitsAbsentRoute(t *testing.T) {
	for name, receipt := range map[string]*model.NativeInferenceReceipt{
		"nil-receipt": nil,
		"no-route":    {},
		"no-calls":    {Qwen35SequencePrefillRoute: &model.NativeSequencePrefillRouteReceipt{}},
		"status-without-calls": {Qwen35SequencePrefillRoute: &model.NativeSequencePrefillRouteReceipt{
			Status: &model.Qwen35SequencePrefillRouteStatus{NativePerformanceQualifying: true},
		}},
	} {
		if got := prefillRouteFromReceipt(receipt); got != nil {
			t.Fatalf("%s: perf-row route=%+v, want nil", name, got)
		}
		data, err := json.Marshal(perfledger.Record{PrefillRoute: prefillRouteFromReceipt(receipt)})
		if err != nil {
			t.Fatal(err)
		}
		var wire map[string]json.RawMessage
		if err := json.Unmarshal(data, &wire); err != nil {
			t.Fatal(err)
		}
		if _, present := wire["prefill_route"]; present {
			t.Fatalf("%s: fabricated route in %s", name, data)
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestPrefillRouteFromReceiptAggregateEvidence(t *testing.T) {
	qualifying := &model.Qwen35SequencePrefillRouteStatus{
		RequestedPath:               compute.Qwen35SequencePrefillPath,
		EffectivePath:               compute.Qwen35SequencePrefillPath,
		NativePerformanceQualifying: true,
	}
	declined := &model.Qwen35SequencePrefillRouteStatus{
		RequestedPath:  compute.Qwen35SequencePrefillPath,
		EffectivePath:  model.Qwen35SequencePrefillFallbackPath,
		DeclineReason:  model.Qwen35SequencePrefillDeclineEmbeddingCap,
		FallbackActive: true,
	}
	for _, tc := range []struct {
		name      string
		aggregate model.NativeSequencePrefillRouteReceipt
	}{
		// Matches the producer regression's qualifying T32 followed by an
		// unobserved T1. The representative status cannot qualify both calls.
		{"mixed_observed_and_unobserved", model.NativeSequencePrefillRouteReceipt{
			PrefillCalls: 2, ObservedCalls: 1, Status: qualifying,
		}},
		{"wholly_unobserved", model.NativeSequencePrefillRouteReceipt{PrefillCalls: 2}},
		{"complete_qualifying", model.NativeSequencePrefillRouteReceipt{
			PrefillCalls: 2, ObservedCalls: 2, Complete: true, NativePerformanceQualifying: true, Status: qualifying,
		}},
		{"complete_declined", model.NativeSequencePrefillRouteReceipt{
			PrefillCalls: 1, ObservedCalls: 1, Complete: true, Status: declined,
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := prefillRouteFromReceipt(&model.NativeInferenceReceipt{Qwen35SequencePrefillRoute: &tc.aggregate})
			if got == nil {
				t.Fatal("executed prefill aggregate disappeared")
			}
			if got.NativePerformanceQualifying != tc.aggregate.NativePerformanceQualifying {
				t.Fatalf("representative status overrode aggregate qualification: %+v", got)
			}
			if status := tc.aggregate.Status; status != nil {
				if got.RequestedPath != status.RequestedPath || got.EffectivePath != status.EffectivePath ||
					got.DeclineReason != status.DeclineReason || got.FallbackActive != status.FallbackActive {
					t.Fatalf("representative status changed: %+v", got)
				}
			} else if got.RequestedPath != "" || got.EffectivePath != "" || got.DeclineReason != "" || got.FallbackActive {
				t.Fatalf("unobserved request acquired fabricated status: %+v", got)
			}
			data, err := json.Marshal(perfledger.Record{PrefillRoute: got})
			if err != nil {
				t.Fatal(err)
			}
			var wire struct {
				Route map[string]json.RawMessage `json:"prefill_route"`
			}
			if err := json.Unmarshal(data, &wire); err != nil {
				t.Fatal(err)
			}
			for name, want := range map[string]any{
				"prefill_calls":                 tc.aggregate.PrefillCalls,
				"observed_calls":                tc.aggregate.ObservedCalls,
				"complete":                      tc.aggregate.Complete,
				"native_performance_qualifying": tc.aggregate.NativePerformanceQualifying,
			} {
				encoded, err := json.Marshal(want)
				if err != nil {
					t.Fatal(err)
				}
				if string(wire.Route[name]) != string(encoded) {
					t.Fatalf("wire %s=%s, want explicit %s in %s", name, wire.Route[name], encoded, data)
				}
			}
			if tc.aggregate.Status == nil {
				for _, name := range []string{"requested_path", "effective_path", "decline_reason", "fallback_active"} {
					if _, present := wire.Route[name]; present {
						t.Fatalf("unobserved status fabricated %s in %s", name, data)
					}
				}
			} else {
				for name, want := range map[string]string{
					"requested_path": tc.aggregate.Status.RequestedPath,
					"effective_path": tc.aggregate.Status.EffectivePath,
				} {
					encoded, err := json.Marshal(want)
					if err != nil {
						t.Fatal(err)
					}
					if string(wire.Route[name]) != string(encoded) {
						t.Fatalf("representative wire %s=%s, want %s", name, wire.Route[name], encoded)
					}
				}
			}
		})
	}
}
