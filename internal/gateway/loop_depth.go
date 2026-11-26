package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	capmatrix "github.com/anthony-chaudhary/fak/internal/capabilitymatrix"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

const codeLoopsOutOfRange = "loops_out_of_range"

// Loop-depth sources reported on LoopDepthReceipt.Source.
const (
	LoopDepthSourceRequest     = "request"
	LoopDepthSourceDefault     = "default"
	LoopDepthSourceEffort      = "effort"
	LoopDepthSourcePolicyClamp = "policy_clamp"
)

// LoopDepthReceipt reports the recurrent depth selected for a looped model.
type LoopDepthReceipt struct {
	Loops    int    `json:"loops"`
	MaxLoops int    `json:"max_loops"`
	Source   string `json:"source"`
}

// modelMaxLoops is the per-model loop capability from the capability matrix
// (max_loops, overridable via FAK_MODEL_CAPABILITIES). Unknown models are depth 1.
var modelMaxLoops = func(model string) int { return max(1, capmatrix.Lookup(model).MaxLoops) }

// modelTrainedLoops is the checkpoint's trained recurrence count (GGUF loop.count
// via the capability matrix). Zero means unknown, which the resolver treats as
// fixed-depth at maxLoops.
var modelTrainedLoops = func(model string) int { return capmatrix.Lookup(model).TrainedLoops }

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

// resolveLoopDepth selects the loop count for req against maxLoops, then clamps
// it under rule. A nil receipt with an empty message means the request does not
// touch loop depth and the response stays byte-identical to a non-looped turn.
func resolveLoopDepth(req ChatRequest, maxLoops, trainedLoops int, rule *policy.LoopDepthRule) (*LoopDepthReceipt, string) {
	receipt, msg := selectLoopDepth(req, maxLoops, trainedLoops)
	if receipt == nil || receipt.MaxLoops <= 1 {
		return receipt, msg
	}
	// The policy cap has no tool scope here, so only the rule's default cap applies.
	if granted, clamped := rule.ClampLoopDepth("", receipt.Loops); clamped {
		receipt.Loops = min(max(1, granted), receipt.MaxLoops)
		receipt.Source = LoopDepthSourcePolicyClamp
	}
	// The floor runs after the clamp so a cap cannot drop irreversible work
	// below the trained pass count.
	if requestIrreversible(req) {
		receipt.Loops = min(agent.ResolveIrreversibleLoopFloor(receipt.Loops, trainedLoops, true), receipt.MaxLoops)
	}
	return receipt, ""
}

// shellToolNames are tools that execute arbitrary commands. Their destructive
// reach is only known per call, so the scope counts as irreversible up front.
var shellToolNames = map[string]bool{"bash": true, "shell": true, "sh": true, "exec": true, "run_command": true, "execute_command": true, "terminal": true}

// requestIrreversible reports whether the request's tool scope can take an
// irreversible action. Per-call arguments do not exist at depth-selection time, so
// a tool counts when its name is irreversible or it is a shell that can run any command.
func requestIrreversible(req ChatRequest) bool {
	for _, t := range req.Tools {
		name := strings.ToLower(t.Function.Name)
		if shellToolNames[name] || adjudicator.ClassifyReversibility(name, nil).Class == adjudicator.ReversibilityIrreversible {
			return true
		}
	}
	return false
}

func selectLoopDepth(req ChatRequest, maxLoops, trainedLoops int) (*LoopDepthReceipt, string) {
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
	if req.ReasoningEffort != "" {
		// Effort may raise depth toward maxLoops but never lowers it below the
		// trained count; an unknown trained count is fixed-depth at maxLoops.
		return &LoopDepthReceipt{Loops: agent.ResolveTrainedLoopDepth(req.ReasoningEffort, maxLoops, trainedLoops), MaxLoops: maxLoops, Source: LoopDepthSourceEffort}, ""
	}
	// A looped model without an explicit count or effort runs its native full depth.
	return &LoopDepthReceipt{Loops: maxLoops, MaxLoops: maxLoops, Source: LoopDepthSourceDefault}, ""
}

// SetLoopDepthRule installs the policy cap on looped-model depth; nil removes it.
// Safe to call while the server is handling requests.
func (s *Server) SetLoopDepthRule(rule *policy.LoopDepthRule) error {
	compiled, err := policy.CompileLoopDepth(rule)
	if err != nil {
		return err
	}
	s.loopDepthRule.Store(compiled)
	return nil
}

// resolveChatLoopDepth writes a 400 and returns ok=false on an inadmissible loop count.
func resolveChatLoopDepth(w http.ResponseWriter, req ChatRequest, rule *policy.LoopDepthRule) (*LoopDepthReceipt, bool) {
	receipt, msg := resolveLoopDepth(req, modelMaxLoops(req.Model), modelTrainedLoops(req.Model), rule)
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

// loopDepthSampleOpt forwards the resolved depth upstream only for a looped
// model, so non-looped traffic keeps its pre-seam request body.
func loopDepthSampleOpt(receipt *LoopDepthReceipt) agent.SampleOpt {
	if receipt == nil || receipt.MaxLoops <= 1 {
		return agent.WithLoops(0)
	}
	return agent.WithLoops(receipt.Loops)
}

// loopDepthClass quantizes depth onto the shallow/default/deep operating points
// so arbitrary loop counts cannot fragment affinity into one key per depth.
func loopDepthClass(loops int) int {
	switch {
	case loops <= 1:
		return 1
	case loops <= 3:
		return 3
	default:
		return 5
	}
}

// depthAffinityKey salts a session affinity key with the depth class, since a
// prefix computed at one depth is not reusable at another.
func depthAffinityKey(key string, receipt *LoopDepthReceipt) string {
	if key == "" || receipt == nil || receipt.MaxLoops <= 1 {
		return key
	}
	return key + "@d=" + strconv.Itoa(loopDepthClass(receipt.Loops))
}
