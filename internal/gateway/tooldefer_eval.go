package gateway

// tooldefer_eval.go — the #3533 held-accuracy fault-in-recall eval (epic #3229, QA gate
// #2 of 3 blocking the default-on flip #3537). Deferral only pays off if a cold tool the
// model needs can still be FAULTED BACK IN via the injected tool_search_tool; the risk the
// ticket names is SILENT CAPABILITY LOSS — a task needs a deferred fak_*/dos_*/built-in
// tool that the deferral quietly dropped or mangled, so the model can never reach it.
//
// This eval runs a FIXED task set (one per representative cold category) through the
// PRODUCTION transform (deferColdToolsInBody + the production defaultHotToolSet) ARMED vs
// ABLATED and scores, per task, the MECHANICAL fault-in chain the model would depend on:
//
//	ablated_pass — the required tool is present in the untouched (eager) body.
//	armed_pass   — after deferral the required tool is STILL faultable-in: present in
//	               tools[], marked defer_loading (or kept eager if hot), a tool_search_tool
//	               is injected as the search means, AND its input_schema is byte-identical
//	               to the eager schema (a faulted-in tool is the SAME tool, not a mangled one).
//
// The gate is armed_pass >= ablated_pass: deferral must lose NO capability. A drop names
// the offending tool + the failing step.
//
// HONESTY (Mode = deterministic-faultin-sim, LiveAccuracyClaimAllowed = false): this
// WITNESSES the mechanical recall property — the deferred tool is never silently lost —
// not live-model task completion. Whether the model actually SEARCHES for and calls the
// tool is a live-run number (#3536), never claimed here. The mechanism, determinism, and
// no-bypass adjudication are witnessed by messages_tooldefer_test.go and
// tooldefer_no_bypass_test.go; this file is the recall witness the #3537 flip is gated on.

import (
	"bytes"
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// HeldAccuracyTask is one eval task: a required tool that starts COLD (not in the hot set),
// so deferral must keep it faultable-in for the task to remain completable.
type HeldAccuracyTask struct {
	Name     string `json:"name"`
	Tool     string `json:"required_tool"`
	Category string `json:"category"` // fak_self_query | dos_verdict | cold_builtin
}

// HeldAccuracyResult is one task's armed-vs-ablated outcome.
type HeldAccuracyResult struct {
	Task        HeldAccuracyTask `json:"task"`
	AblatedPass bool             `json:"ablated_pass"`
	ArmedPass   bool             `json:"armed_pass"`
	Reason      string           `json:"reason,omitempty"` // why armed failed, naming the failing step
}

// HeldAccuracyReport is the #3533 scorecard: the held-accuracy PAIR (#3537 gate input) plus
// the per-task detail and the offender list when the gate does not hold.
type HeldAccuracyReport struct {
	Mode                     string               `json:"mode"`       // deterministic-faultin-sim
	Provenance               string               `json:"provenance"` // WITNESSED (mechanical fault-in recall)
	LiveAccuracyClaimAllowed bool                 `json:"live_accuracy_claim_allowed"`
	Results                  []HeldAccuracyResult `json:"results"`
	AblatedPass              int                  `json:"ablated_pass"`
	ArmedPass                int                  `json:"armed_pass"`
	Total                    int                  `json:"total"`
	GateHolds                bool                 `json:"gate_holds"` // armed_pass >= ablated_pass
	Offenders                []string             `json:"offenders,omitempty"`
}

// defaultHeldAccuracyTasks is the FIXED eval set: one representative cold tool per category
// on fak's own surface. Each MUST be absent from defaultHotToolSet (asserted by test) so the
// transform genuinely defers it — otherwise the eval would score a tool deferral never touched.
var defaultHeldAccuracyTasks = []HeldAccuracyTask{
	{Name: "fak self-query fault-in", Tool: "mcp__fak__fak_index_docs", Category: "fak_self_query"},
	{Name: "dos verdict fault-in", Tool: "mcp__dos__dos_verify", Category: "dos_verdict"},
	{Name: "cold built-in fault-in", Tool: "BashOutput", Category: "cold_builtin"},
}

// DeferHeldAccuracyReport runs the fixed eval set and returns the #3533 held-accuracy pair.
func DeferHeldAccuracyReport() HeldAccuracyReport { return heldAccuracy(defaultHeldAccuracyTasks) }

func heldAccuracy(tasks []HeldAccuracyTask) HeldAccuracyReport {
	results := make([]HeldAccuracyResult, 0, len(tasks))
	for _, task := range tasks {
		results = append(results, scoreHeldAccuracyTask(task))
	}
	return foldHeldAccuracyResults(results)
}

// foldHeldAccuracyResults folds per-task outcomes into the held-accuracy pair, the gate
// verdict (armed_pass >= ablated_pass), and the offender list (a tool the eager arm reached
// but deferral did not). Split out so the fold is testable on synthetic results.
func foldHeldAccuracyResults(results []HeldAccuracyResult) HeldAccuracyReport {
	rep := HeldAccuracyReport{
		Mode:                     "deterministic-faultin-sim",
		Provenance:               "WITNESSED (mechanical fault-in recall; live task-completion accuracy is #3536)",
		LiveAccuracyClaimAllowed: false,
		Results:                  results,
		Total:                    len(results),
	}
	for _, res := range results {
		if res.AblatedPass {
			rep.AblatedPass++
		}
		if res.ArmedPass {
			rep.ArmedPass++
		}
		if res.AblatedPass && !res.ArmedPass {
			rep.Offenders = append(rep.Offenders, res.Task.Tool)
		}
	}
	rep.GateHolds = rep.ArmedPass >= rep.AblatedPass
	return rep
}

// scoreHeldAccuracyTask builds a task body, runs the production deferral transform, and
// scores the mechanical fault-in chain against the eager (ablated) baseline.
func scoreHeldAccuracyTask(task HeldAccuracyTask) HeldAccuracyResult {
	body := heldAccuracyBody(task.Tool)
	r := HeldAccuracyResult{Task: task}

	origSchema, present := toolInputSchema(body, task.Tool)
	r.AblatedPass = present // eager: the tool is directly available.

	res := deferColdToolsInBody(body, defaultHotToolSet, func(b []byte) error {
		_, err := agent.DecodeAnthropicMessagesRequest(b)
		return err
	})
	armed := body
	if res.Changed {
		armed = res.Body
	}
	ok, reason := toolFaultableIn(armed, task.Tool, origSchema)
	r.ArmedPass = ok
	if !ok {
		r.Reason = reason
	}
	return r
}

// heldAccuracyBody is a minimal Claude-Code-shaped body: a hot tool (Read, kept eager), the
// task's required cold tool with a non-trivial input_schema (so the schema-intact check has
// teeth), and one filler cold tool so the deferred tail is non-degenerate.
func heldAccuracyBody(requiredTool string) []byte {
	tools := []map[string]any{
		{"name": "Read", "description": "read a file", "input_schema": map[string]any{"type": "object"}},
		{"name": requiredTool, "description": "the task-required cold tool", "input_schema": map[string]any{
			"type": "object", "properties": map[string]any{"query": map[string]any{"type": "string"}}, "required": []any{"query"},
		}},
		{"name": "mcp__filler__other", "description": "another cold tool", "input_schema": map[string]any{"type": "object"}},
	}
	raw, _ := json.Marshal(map[string]any{
		"model": "claude-x", "max_tokens": 64,
		"messages": []map[string]any{{"role": "user", "content": "complete the task"}},
		"tools":    tools,
	})
	return raw
}

// toolInputSchema returns the byte-exact input_schema of the named tool in body, and
// whether the tool is present at all.
func toolInputSchema(body []byte, tool string) (schema []byte, present bool) {
	for _, m := range decodeToolElems(body) {
		if rawStringField(m, "name") == tool {
			return append([]byte(nil), m["input_schema"]...), true
		}
	}
	return nil, false
}

// toolFaultableIn reports whether the named tool survives deferral in a form the model can
// still fault in and call: present, deferred (or eagerly hot), searchable, schema-intact.
func toolFaultableIn(armed []byte, tool string, eagerSchema []byte) (bool, string) {
	elems := decodeToolElems(armed)
	var def map[string]json.RawMessage
	searchInjected := false
	for _, m := range elems {
		name := rawStringField(m, "name")
		if name == tool {
			def = m
		}
		if name == toolSearchToolName || rawStringField(m, "type") == toolSearchToolType {
			searchInjected = true
		}
	}
	if def == nil {
		return false, "dropped from tools[] under deferral — silent capability loss"
	}
	deferred := string(bytes.TrimSpace(def["defer_loading"])) == "true"
	if deferred && !searchInjected {
		return false, "deferred but no tool_search_tool injected — unsearchable, cannot fault in"
	}
	if !deferred && !defaultHotToolSet[tool] {
		return false, "a cold tool left neither deferred nor eager — unreachable"
	}
	if !bytes.Equal(bytes.TrimSpace(def["input_schema"]), bytes.TrimSpace(eagerSchema)) {
		return false, "input_schema mangled under deferral — a faulted-in tool would differ from the eager one"
	}
	return true, ""
}

// decodeToolElems decodes body's tools[] into per-element objects; nil on any parse error.
func decodeToolElems(body []byte) []map[string]json.RawMessage {
	var obj map[string]json.RawMessage
	if json.Unmarshal(body, &obj) != nil {
		return nil
	}
	var raws []json.RawMessage
	if json.Unmarshal(obj["tools"], &raws) != nil {
		return nil
	}
	out := make([]map[string]json.RawMessage, 0, len(raws))
	for _, r := range raws {
		var m map[string]json.RawMessage
		if json.Unmarshal(r, &m) == nil {
			out = append(out, m)
		}
	}
	return out
}
