package modelengine

import (
	"reflect"
	"strings"
	"testing"
)

func TestQwenPagedSwapReceiptExecutionAndOutputEquality(t *testing.T) {
	receipt, err := RunQwenPagedSwapReceipt(nil)
	if err != nil {
		t.Fatalf("RunQwenPagedSwapReceipt failed: %v", err)
	}

	if receipt.Schema != QwenPagedSwapSchema {
		t.Fatalf("receipt.Schema = %q, want %q", receipt.Schema, QwenPagedSwapSchema)
	}
	if receipt.Verdict != QwenPagedSwapVerdictKeep {
		t.Fatalf("receipt.Verdict = %q, want %q", receipt.Verdict, QwenPagedSwapVerdictKeep)
	}
	if receipt.Engine != QwenPagedSwapFakNative {
		t.Fatalf("receipt.Engine = %q, want %q", receipt.Engine, QwenPagedSwapFakNative)
	}
	if receipt.Backend != QwenPagedSwapBackend {
		t.Fatalf("receipt.Backend = %q, want %q", receipt.Backend, QwenPagedSwapBackend)
	}
	if receipt.ArtifactSHA256 != QwenPagedSwapArtifactSHA256 {
		t.Fatalf("receipt.ArtifactSHA256 = %q, want %q", receipt.ArtifactSHA256, QwenPagedSwapArtifactSHA256)
	}
	if len(receipt.ArrivalTrace) != 2 {
		t.Fatalf("len(receipt.ArrivalTrace) = %d, want 2", len(receipt.ArrivalTrace))
	}
	for i, arrival := range receipt.ArrivalTrace {
		if arrival.Tokens <= 0 || arrival.PromptSHA256 == "" || arrival.Request == "" {
			t.Fatalf("arrival %d is incomplete: %+v", i, arrival)
		}
	}

	// OFF arm: baseline without swap pressure
	if receipt.Off.Pressure != "OFF" {
		t.Fatalf("receipt.Off.Pressure = %q, want OFF", receipt.Off.Pressure)
	}
	if receipt.Off.SwapTotal != 0 || receipt.Off.ReadmittedTotal != 0 || receipt.Off.SwapBytes != 0 || receipt.Off.RestoredBytes != 0 {
		t.Fatalf("receipt.Off observed swap: %+v", receipt.Off)
	}
	if receipt.Off.RecomputeTotal != 0 {
		t.Fatalf("receipt.Off.RecomputeTotal = %d, want 0", receipt.Off.RecomputeTotal)
	}
	if receipt.Off.FallbackTotal != 0 || receipt.Off.ErrorTotal != 0 {
		t.Fatalf("receipt.Off had fallbacks or errors: fallback=%d error=%d", receipt.Off.FallbackTotal, receipt.Off.ErrorTotal)
	}

	// ON arm: forced paged-swap pressure
	if receipt.On.Pressure != "ON" {
		t.Fatalf("receipt.On.Pressure = %q, want ON", receipt.On.Pressure)
	}
	if receipt.On.SwapTotal <= 0 {
		t.Fatalf("receipt.On.SwapTotal = %d, want > 0", receipt.On.SwapTotal)
	}
	if receipt.On.ReadmittedTotal <= 0 {
		t.Fatalf("receipt.On.ReadmittedTotal = %d, want > 0", receipt.On.ReadmittedTotal)
	}
	if receipt.On.SwapBytes <= 0 {
		t.Fatalf("receipt.On.SwapBytes = %d, want > 0", receipt.On.SwapBytes)
	}
	if receipt.On.RestoredBytes != receipt.On.SwapBytes {
		t.Fatalf("receipt.On byte mismatch: swapped %d, restored %d", receipt.On.SwapBytes, receipt.On.RestoredBytes)
	}
	if receipt.On.RecomputeTotal != 0 {
		t.Fatalf("receipt.On.RecomputeTotal = %d, want 0", receipt.On.RecomputeTotal)
	}
	if receipt.On.FallbackTotal != 0 || receipt.On.ErrorTotal != 0 {
		t.Fatalf("receipt.On had fallbacks or errors: fallback=%d error=%d", receipt.On.FallbackTotal, receipt.On.ErrorTotal)
	}

	// Output and state equality between OFF and ON arms
	if len(receipt.Off.Requests) != 2 || len(receipt.On.Requests) != 2 {
		t.Fatalf("request count mismatch: OFF=%d, ON=%d", len(receipt.Off.Requests), len(receipt.On.Requests))
	}
	for i := range receipt.Off.Requests {
		offReq := receipt.Off.Requests[i]
		onReq := receipt.On.Requests[i]
		if offReq.Request != onReq.Request {
			t.Fatalf("request name mismatch at %d: OFF=%q, ON=%q", i, offReq.Request, onReq.Request)
		}
		if len(offReq.TokenIDs) == 0 {
			t.Fatalf("request %q produced no tokens", offReq.Request)
		}
		if !reflect.DeepEqual(offReq.TokenIDs, onReq.TokenIDs) {
			t.Fatalf("token ID mismatch for %q:\n OFF: %v\n  ON: %v", offReq.Request, offReq.TokenIDs, onReq.TokenIDs)
		}
		if offReq.OutputSHA256 != onReq.OutputSHA256 {
			t.Fatalf("output SHA256 mismatch for %q: OFF=%s, ON=%s", offReq.Request, offReq.OutputSHA256, onReq.OutputSHA256)
		}
		if offReq.StateDigest == "" || onReq.StateDigest == "" {
			t.Fatalf("empty state digest for %q", offReq.Request)
		}
		if offReq.StateDigest != onReq.StateDigest {
			t.Fatalf("state digest mismatch for %q: OFF=%s, ON=%s", offReq.Request, offReq.StateDigest, onReq.StateDigest)
		}
	}
	if !reflect.DeepEqual(receipt.Off.StateDigests, receipt.On.StateDigests) {
		t.Fatalf("state digests mismatch: OFF=%v, ON=%v", receipt.Off.StateDigests, receipt.On.StateDigests)
	}

	// Memory, latency, throughput, and teardown accounting
	for _, arm := range []QwenPagedSwapArm{receipt.Off, receipt.On} {
		if arm.PeakRunning <= 0 {
			t.Fatalf("%s arm.PeakRunning = %d, want > 0", arm.Pressure, arm.PeakRunning)
		}
		if arm.PeakRSSBytes == 0 {
			t.Fatalf("%s arm.PeakRSSBytes = 0, want > 0", arm.Pressure)
		}
		if arm.TTFTP50Milliseconds <= 0 || arm.TTFTP95Milliseconds <= 0 {
			t.Fatalf("%s arm TTFT percentiles non-positive: p50=%f, p95=%f", arm.Pressure, arm.TTFTP50Milliseconds, arm.TTFTP95Milliseconds)
		}
		if arm.ITLP50Milliseconds <= 0 || arm.ITLP95Milliseconds <= 0 {
			t.Fatalf("%s arm ITL percentiles non-positive: p50=%f, p95=%f", arm.Pressure, arm.ITLP50Milliseconds, arm.ITLP95Milliseconds)
		}
		if arm.AggregateTokensPerSecond <= 0 {
			t.Fatalf("%s arm.AggregateTokensPerSecond = %f, want > 0", arm.Pressure, arm.AggregateTokensPerSecond)
		}
		if !arm.TeardownComplete {
			t.Fatalf("%s arm.TeardownComplete = false", arm.Pressure)
		}
	}

	if err := ValidateQwenPagedSwapReceipt(receipt); err != nil {
		t.Fatalf("ValidateQwenPagedSwapReceipt rejected green receipt: %v", err)
	}
}

func TestQwenPagedSwapReceiptFailClosedValidation(t *testing.T) {
	base, err := RunQwenPagedSwapReceipt(nil)
	if err != nil {
		t.Fatalf("RunQwenPagedSwapReceipt failed: %v", err)
	}

	cloneReceipt := func() *QwenPagedSwapReceipt {
		data, err := MarshalQwenPagedSwapReceipt(base)
		if err != nil {
			t.Fatalf("marshal base: %v", err)
		}
		r, err := UnmarshalQwenPagedSwapReceipt(data)
		if err != nil {
			t.Fatalf("unmarshal base: %v", err)
		}
		return r
	}

	cases := []struct {
		name        string
		mutate      func(*QwenPagedSwapReceipt)
		errContains string
	}{
		{
			name: "unequal token IDs in first request",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.On.Requests[0].TokenIDs[0]++
			},
			errContains: "accepted token IDs mismatch",
		},
		{
			name: "extra token in ON arm request",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.On.Requests[0].TokenIDs = append(r.On.Requests[0].TokenIDs, 99999)
			},
			errContains: "accepted token IDs mismatch",
		},
		{
			name: "unequal output SHA256",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.On.Requests[0].OutputSHA256 = "0000000000000000000000000000000000000000000000000000000000000000"
			},
			errContains: "output sha256 mismatch",
		},
		{
			name: "unequal state digest",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.On.Requests[0].StateDigest = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
			},
			errContains: "state digest mismatch",
		},
		{
			name: "unequal arm state digests array",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.On.StateDigests[0] = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
			},
			errContains: "state digests mismatch",
		},
		{
			name: "nonzero fallback in OFF arm",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.Off.FallbackTotal = 1
			},
			errContains: "fallback",
		},
		{
			name: "nonzero fallback in ON arm",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.On.FallbackTotal = 1
			},
			errContains: "fallback",
		},
		{
			name: "missing swap in ON arm",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.On.SwapTotal = 0
			},
			errContains: "missing swap",
		},
		{
			name: "missing readmission in ON arm",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.On.ReadmittedTotal = 0
			},
			errContains: "missing readmission",
		},
		{
			name: "swap bytes non-positive in ON arm",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.On.SwapBytes = 0
				r.On.RestoredBytes = 0
			},
			errContains: "swap_bytes must be > 0",
		},
		{
			name: "swapped and restored byte mismatch in ON arm",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.On.RestoredBytes = r.On.SwapBytes + 1
			},
			errContains: "swapped and restored bytes mismatch",
		},
		{
			name: "nonzero recompute in ON arm",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.On.RecomputeTotal = 1
			},
			errContains: "recompute",
		},
		{
			name: "nonzero error in ON arm",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.On.ErrorTotal = 1
			},
			errContains: "error",
		},
		{
			name: "swap observed in OFF arm",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.Off.SwapTotal = 1
				r.Off.ReadmittedTotal = 1
				r.Off.SwapBytes = 64
				r.Off.RestoredBytes = 64
			},
			errContains: "OFF arm observed swap",
		},
		{
			name: "disqualifying verdict REJECT",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.Verdict = QwenPagedSwapVerdictReject
			},
			errContains: "invalid verdict",
		},
		{
			name: "wrong engine identity",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.Engine = "llama.cpp"
			},
			errContains: "invalid engine",
		},
		{
			name: "wrong backend identity",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.Backend = "cpu"
			},
			errContains: "invalid backend",
		},
		{
			name: "wrong artifact sha256",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.ArtifactSHA256 = "0000000000000000000000000000000000000000000000000000000000000000"
			},
			errContains: "invalid artifact sha256",
		},
		{
			name: "incomplete arrival trace",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.ArrivalTrace = r.ArrivalTrace[:1]
			},
			errContains: "at least 2 sessions",
		},
		{
			name: "incomplete teardown",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.Off.TeardownComplete = false
			},
			errContains: "teardown is incomplete",
		},
		{
			name: "zero peak RSS",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.Off.PeakRSSBytes = 0
			},
			errContains: "peak_rss_bytes",
		},
		{
			name: "tampered binding sha256",
			mutate: func(r *QwenPagedSwapReceipt) {
				r.BindingSHA256 = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
			},
			errContains: "receipt binding mismatch",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := cloneReceipt()
			tc.mutate(r)
			err := ValidateQwenPagedSwapReceipt(r)
			if err == nil {
				t.Fatalf("validation unexpectedly succeeded for case %q", tc.name)
			}
			if !strings.Contains(err.Error(), tc.errContains) {
				t.Fatalf("error %q does not contain expected substring %q", err.Error(), tc.errContains)
			}
		})
	}
}

func TestQwenPagedSwapReceiptSerializationRoundtrip(t *testing.T) {
	receipt, err := RunQwenPagedSwapReceipt(nil)
	if err != nil {
		t.Fatalf("RunQwenPagedSwapReceipt failed: %v", err)
	}

	data, err := MarshalQwenPagedSwapReceipt(receipt)
	if err != nil {
		t.Fatalf("MarshalQwenPagedSwapReceipt failed: %v", err)
	}

	unmarshaled, err := UnmarshalQwenPagedSwapReceipt(data)
	if err != nil {
		t.Fatalf("UnmarshalQwenPagedSwapReceipt failed: %v", err)
	}

	if !reflect.DeepEqual(receipt, unmarshaled) {
		t.Fatalf("unmarshaled receipt does not match original")
	}

	if err := ValidateQwenPagedSwapReceipt(unmarshaled); err != nil {
		t.Fatalf("ValidateQwenPagedSwapReceipt rejected unmarshaled receipt: %v", err)
	}
	if err := unmarshaled.CheckBinding(); err != nil {
		t.Fatalf("CheckBinding failed on roundtrip receipt: %v", err)
	}
}
