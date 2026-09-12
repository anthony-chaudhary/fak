package kernel

import (
	"crypto/sha256"
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// FieldCompletionTiming carries an opaque kernel-created measurement. Plain
// elapsed_ns fields and result metadata cannot manufacture this provenance.
const FieldCompletionTiming = "kernel_completion_timing"

type completionTiming struct {
	call       *abi.ToolCall
	result     *abi.Result
	callHash   [32]byte
	resultHash [32]byte
	elapsedNS  int64
}

func completionTimingHash(value any) ([32]byte, bool) {
	body, err := json.Marshal(value)
	if err != nil {
		return [32]byte{}, false
	}
	return sha256.Sum256(body), true
}

func attestCompletionTiming(call *abi.ToolCall, result *abi.Result, elapsedNS int64) *completionTiming {
	if call == nil || result == nil || result.Status != abi.StatusOK || elapsedNS <= 0 {
		return nil
	}
	ch, cok := completionTimingHash(call)
	rh, rok := completionTimingHash(result)
	if !cok || !rok {
		return nil
	}
	return &completionTiming{call: call, result: result, callHash: ch, resultHash: rh, elapsedNS: elapsedNS}
}

// CompletionTimingNanos reads an actual engine span only while the completion
// event still names the same call and result with the same contents. Admission
// rewrites, fabricated events, and copied result envelopes provide no evidence.
// The attestation is transient; consumers retain only the validated scalar.
func CompletionTimingNanos(ev abi.Event) (int64, bool) {
	value, ok := ev.Fields[FieldCompletionTiming].(*completionTiming)
	if !ok || value == nil || ev.Kind != abi.EvComplete || ev.Call != value.call || ev.Result != value.result ||
		ev.Call == nil || ev.Result == nil || ev.Result.Status != abi.StatusOK || value.elapsedNS <= 0 {
		return 0, false
	}
	ch, cok := completionTimingHash(ev.Call)
	rh, rok := completionTimingHash(ev.Result)
	if !cok || !rok || ch != value.callHash || rh != value.resultHash {
		return 0, false
	}
	return value.elapsedNS, true
}
