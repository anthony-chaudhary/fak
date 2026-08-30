package main

import (
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

func TestQwen38PagedSwapControlsBindExactNativeEnvelope(t *testing.T) {
	env := map[string]string{
		"FAK_METAL_STREAM_Q4K":        "1",
		"FAK_Q4K":                     "1",
		nativeControlGGUFMMap:         "1",
		nativeProfileSequenceSelector: nativeProfileSelectorOff,
		"FAK_NATIVE_MAX_RUNNING":      "2",
		"FAK_NATIVE_KV_BLOCK_TOKENS":  "16",
		"FAK_NATIVE_KV_PREEMPT_MODE":  "swap",
		"FAK_NATIVE_KV_MAX_BLOCKS":    "3",
		"FAK_NATIVE_KV_VICTIM_RULE":   "newest",
	}
	declarations := make([]string, 0, len(env))
	for key, value := range env {
		declarations = append(declarations, key+"="+value)
	}
	lookup := func(key string) (string, bool) {
		value, ok := env[key]
		return value, ok
	}
	controls, maxBlocks, err := qwen38PagedSwapControls(lookup, declarations, 0)
	if err != nil {
		t.Fatalf("exact paged-swap controls rejected: %v", err)
	}
	if maxBlocks != 3 || controls["FAK_NATIVE_KV_VICTIM_RULE"] != "most-recent" {
		t.Fatalf("controls=%v max_blocks=%d", controls, maxBlocks)
	}
	if err := validateQwen38PagedSwapControls(controls); err != nil {
		t.Fatalf("captured controls failed readback validation: %v", err)
	}

	for _, test := range []struct {
		name   string
		mutate func(map[string]string)
	}{
		{name: "under one-request floor", mutate: func(values map[string]string) { values["FAK_NATIVE_KV_MAX_BLOCKS"] = "1" }},
		{name: "no two-request pressure", mutate: func(values map[string]string) { values["FAK_NATIVE_KV_MAX_BLOCKS"] = "4" }},
		{name: "recompute", mutate: func(values map[string]string) { values["FAK_NATIVE_KV_PREEMPT_MODE"] = "recompute" }},
		{name: "wrong concurrency", mutate: func(values map[string]string) { values["FAK_NATIVE_MAX_RUNNING"] = "1" }},
		{name: "selector on", mutate: func(values map[string]string) { values[nativeProfileSequenceSelector] = nativeProfileSelectorOn }},
		{name: "unknown control", mutate: func(values map[string]string) { values["FAK_FUTURE_SWAP_SWITCH"] = "1" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			bad := make(map[string]string, len(env)+1)
			for key, value := range env {
				bad[key] = value
			}
			test.mutate(bad)
			decls := make([]string, 0, len(bad))
			for key, value := range bad {
				decls = append(decls, key+"="+value)
			}
			lookup := func(key string) (string, bool) {
				value, ok := bad[key]
				return value, ok
			}
			if _, _, err := qwen38PagedSwapControls(lookup, decls, 0); err == nil {
				t.Fatal("behavior-changing control was accepted")
			}
		})
	}
}

func TestQwen38PagedSwapFlagGateRequiresExactNativeMetalPath(t *testing.T) {
	valid := testCompleteBenchFlags()
	*valid.qwenSwapOut = "receipt.json"
	*valid.gguf = "qwen.gguf"
	*valid.q4k = true
	*valid.metal = true
	*valid.name = "qwen38:27b"
	if err := validateFlagCombinations(valid); err != nil {
		t.Fatalf("exact paged-swap flags rejected: %v", err)
	}
	for _, test := range []struct {
		name   string
		mutate func(*benchFlags)
	}{
		{name: "missing q4k", mutate: func(f *benchFlags) { *f.q4k = false }},
		{name: "missing metal", mutate: func(f *benchFlags) { *f.metal = false }},
		{name: "wrong model", mutate: func(f *benchFlags) { *f.name = "qwen36:27b" }},
		{name: "wrong backend", mutate: func(f *benchFlags) { *f.backendName = "cpu-ref" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			f := testCompleteBenchFlags()
			*f.qwenSwapOut = "receipt.json"
			*f.gguf = "qwen.gguf"
			*f.q4k = true
			*f.metal = true
			*f.name = "qwen38:27b"
			test.mutate(f)
			if err := validateFlagCombinations(f); err == nil {
				t.Fatal("wrong engine/model/backend flag set was accepted")
			}
		})
	}
}

func TestQwen38PagedSwapPairFailsClosedBeforeAcceptingInvalidRuntime(t *testing.T) {
	validSwap := qwen38PagedSwapArm{
		Pressure:        "ON",
		SwapTotal:       1,
		ReadmittedTotal: 1,
		SwapBytes:       64,
		RestoredBytes:   64,
		KVMaxBlocks:     3,
		PeakRunning:     2,
		PeakUsedBlocks:  4,
		SwapUsage: []modelperfobs.QwenSwapWeeklyUsage{{
			Invocations: 2,
			SwapOut:     1,
			RestoreIn:   1,
			Succeeded:   2,
		}},
	}
	tests := []struct {
		name   string
		mutate func(*qwen38PagedSwapPair)
		want   string
	}{
		{name: "zero swap", mutate: func(pair *qwen38PagedSwapPair) { pair.On.SwapTotal = 0 }, want: "swap/readmission"},
		{name: "zero readmit", mutate: func(pair *qwen38PagedSwapPair) { pair.On.ReadmittedTotal = 0 }, want: "swap/readmission"},
		{name: "byte mismatch", mutate: func(pair *qwen38PagedSwapPair) { pair.On.RestoredBytes++ }, want: "byte-exact"},
		{name: "recompute", mutate: func(pair *qwen38PagedSwapPair) { pair.On.RecomputeTotal = 1 }, want: "recompute"},
		{name: "no durable usage", mutate: func(pair *qwen38PagedSwapPair) { pair.On.SwapUsage = nil }, want: "durable"},
		{name: "fallback", mutate: func(pair *qwen38PagedSwapPair) { pair.Off.FallbackTotal = 1 }, want: "fallback"},
		{name: "error", mutate: func(pair *qwen38PagedSwapPair) { pair.Off.ErrorTotal = 1 }, want: "error"},
		{name: "refusal", mutate: func(pair *qwen38PagedSwapPair) { pair.Off.RefusalTotal = 1 }, want: "refusal"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pair := qwen38PagedSwapPair{
				Off: qwen38PagedSwapArm{
					Pressure:                 "OFF",
					PeakRSSBytes:             1,
					AggregateTokensPerSecond: 1,
					TTFTP50Milliseconds:      1,
					ITLP50Milliseconds:       1,
					TeardownMilliseconds:     1,
					TeardownComplete:         true,
				},
				On: validSwap,
			}
			pair.On.PeakRSSBytes = 1
			pair.On.AggregateTokensPerSecond = 1
			pair.On.TTFTP50Milliseconds = 1
			pair.On.ITLP50Milliseconds = 1
			pair.On.TeardownMilliseconds = 1
			pair.On.TeardownComplete = true
			test.mutate(&pair)
			err := validateQwen38PagedSwapPair(pair)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error=%v, want substring %q", err, test.want)
			}
		})
	}
}

func TestQwen38PagedSwapOutputAndStateEqualityIgnoreOnlyNonSemanticIdentity(t *testing.T) {
	off := []qwen38PagedSwapRequest{{
		Request: "request-1", TokenIDs: []int{1, 2}, OutputSHA256: strings.Repeat("a", 64),
		TTFTMilliseconds: 1, ITLMilliseconds: []float64{2},
	}}
	on := []qwen38PagedSwapRequest{{
		Request: "request-1", TokenIDs: []int{1, 2}, OutputSHA256: strings.Repeat("a", 64),
		TTFTMilliseconds: 99, ITLMilliseconds: []float64{88},
	}}
	if !qwen38PagedSwapOutputsEqual(off, on) {
		t.Fatal("timing-only differences changed output equality")
	}
	on[0].TokenIDs[1]++
	if qwen38PagedSwapOutputsEqual(off, on) {
		t.Fatal("token mismatch was accepted")
	}
	on[0].TokenIDs[1]--
	on[0].OutputSHA256 = strings.Repeat("b", 64)
	if qwen38PagedSwapOutputsEqual(off, on) {
		t.Fatal("payload digest mismatch was accepted")
	}

	stateA := []model.Qwen35MetalStateIdentityReceipt{{
		OwnerGeneration:    strings.Repeat("a", 64),
		TokenLineageSHA256: strings.Repeat("c", 64),
		States:             []model.Qwen35MetalStateDigest{{Layer: 1, Role: model.Qwen35MetalStateRoleKRaw, SHA256: strings.Repeat("d", 64)}},
	}}
	stateB := []model.Qwen35MetalStateIdentityReceipt{{
		OwnerGeneration:    strings.Repeat("b", 64),
		TokenLineageSHA256: strings.Repeat("c", 64),
		States:             append([]model.Qwen35MetalStateDigest(nil), stateA[0].States...),
	}}
	if !qwen38PagedSwapStatesEqual(stateA, stateB) {
		t.Fatal("opaque owner generation should not change state equality")
	}
	stateB[0].States[0].SHA256 = strings.Repeat("e", 64)
	if qwen38PagedSwapStatesEqual(stateA, stateB) {
		t.Fatal("state digest mismatch was accepted")
	}
	if reflect.DeepEqual(stateA, stateB) {
		t.Fatal("test fixture did not diverge")
	}
}

func TestQwen38PagedSwapReceiptRejectsWrongEngineOrBackend(t *testing.T) {
	for _, mutate := range []func(*qwen38PagedSwapReceipt){
		func(receipt *qwen38PagedSwapReceipt) { receipt.Engine = "llama.cpp" },
		func(receipt *qwen38PagedSwapReceipt) { receipt.Backend = "cpu" },
	} {
		receipt := qwen38PagedSwapReceipt{
			Schema: qwen38PagedSwapSchema, Verdict: "KEEP",
			Engine: qwen38PagedSwapEngine, Backend: qwen38PagedSwapBackend, ForwardPath: qwen38PagedSwapForwardPath,
		}
		mutate(&receipt)
		if err := validateQwen38PagedSwapReceipt(receipt); err == nil {
			t.Fatal("wrong engine/backend receipt was accepted")
		}
	}
}

func TestQwen38PagedSwapLifecycleNamesAreStable(t *testing.T) {
	if modelengine.NativeSessionFresh != "fresh" || modelengine.NativeSessionRestored != "restored" {
		t.Fatalf("unexpected lifecycle names: %q %q", modelengine.NativeSessionFresh, modelengine.NativeSessionRestored)
	}
}
