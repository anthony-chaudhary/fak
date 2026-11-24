package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
)

const codeLoopsOutOfRange = "loops_out_of_range"

// Loop-depth sources reported on LoopDepthReceipt.Source.
const (
	LoopDepthSourceRequest = "request"
	LoopDepthSourceDefault = "default"
)

// LoopDepthReceipt reports the recurrent depth selected for a looped model.
type LoopDepthReceipt struct {
	Loops    int    `json:"loops"`
	MaxLoops int    `json:"max_loops"`
	Source   string `json:"source"`
}

// modelMaxLoops is the per-model loop capability. No looped model is served yet,
// so every model reports depth 1 and only "loops":1 is admissible.
var modelMaxLoops = func(model string) int { return 1 }

// requestedLoops reads the per-request loop count from the two wire shapes vLLM
// accepts: top-level "loops" and vllm_xargs.loops. Both present must agree.
func requestedLoops(req ChatRequest) (*int, string) {
	var xargsLoops *int
	raw := bytes.TrimSpace(req.VLLMXArgs)
	if len(raw) > 0 && !bytes.Equal(raw, []byte("null")) {
		if raw[0] != '{' {
			return nil, "vllm_xargs must be a JSON object"
		}
		var xargs struct {
			Loops *int `json:"loops"`
		}
		if err := json.Unmarshal(raw, &xargs); err != nil {
			return nil, "vllm_xargs.loops must be an integer"
		}
		xargsLoops = xargs.Loops
	}
	switch {
	case req.Loops != nil && xargsLoops != nil && *req.Loops != *xargsLoops:
		return nil, fmt.Sprintf("loops (%d) and vllm_xargs.loops (%d) disagree", *req.Loops, *xargsLoops)
	case req.Loops != nil:
		return req.Loops, ""
	default:
		return xargsLoops, ""
	}
}

// resolveLoopDepth selects the loop count for req against maxLoops. A nil
// receipt with an empty message means the request does not touch loop depth and
// the response stays byte-identical to a non-looped turn.
func resolveLoopDepth(req ChatRequest, maxLoops int) (*LoopDepthReceipt, string) {
	if maxLoops < 1 {
		maxLoops = 1
	}
	explicit, msg := requestedLoops(req)
	if msg != "" {
		return nil, msg
	}
	if explicit != nil {
		if *explicit < 1 || *explicit > maxLoops {
			return nil, fmt.Sprintf("loops must be in [1, %d] for model %q, got %d", maxLoops, req.Model, *explicit)
		}
		return &LoopDepthReceipt{Loops: *explicit, MaxLoops: maxLoops, Source: LoopDepthSourceRequest}, ""
	}
	if maxLoops == 1 {
		return nil, ""
	}
	// A looped model without an explicit count runs its native full depth.
	return &LoopDepthReceipt{Loops: maxLoops, MaxLoops: maxLoops, Source: LoopDepthSourceDefault}, ""
}

// resolveChatLoopDepth writes a 400 and returns ok=false on an inadmissible loop count.
func resolveChatLoopDepth(w http.ResponseWriter, req ChatRequest) (*LoopDepthReceipt, bool) {
	receipt, msg := resolveLoopDepth(req, modelMaxLoops(req.Model))
	if msg != "" {
		writeErrCode(w, http.StatusBadRequest, codeLoopsOutOfRange, msg)
		return nil, false
	}
	return receipt, true
}

func stampLoopDepthReceipt(resp *ChatResponse, receipt *LoopDepthReceipt) {
	if receipt == nil {
		return
	}
	if resp.Fak == nil {
		resp.Fak = &FakExt{}
	}
	resp.Fak.LoopDepth = receipt
}
