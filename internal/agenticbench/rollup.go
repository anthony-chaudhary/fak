// Package agenticbench folds the #868 agentic benchmark artifacts into one
// parent gate. It is intentionally read-only: the live benchmark lanes remain
// owned by their native harnesses.
package agenticbench

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const Schema = "fak.agentic-benchmark-epic-rollup.v1"

type Report struct {
	Schema             string        `json:"schema"`
	GeneratedAt        string        `json:"generated_at"`
	Epic               int           `json:"epic"`
	Status             string        `json:"status"`
	ResultClaimAllowed bool          `json:"result_claim_allowed"`
	Summary            Summary       `json:"summary"`
	Children           []ChildStatus `json:"children"`
	Acceptance         []Gate        `json:"acceptance"`
	ClaimBoundary      string        `json:"claim_boundary"`
}

type Summary struct {
	ChildrenTotal          int   `json:"children_total"`
	ChildrenParsed         int   `json:"children_parsed"`
	LocalEvidenceArtifacts int   `json:"local_evidence_artifacts"`
	ResultClaimArtifacts   int   `json:"result_claim_artifacts"`
	PendingChildren        []int `json:"pending_children,omitempty"`
	FailedChildren         []int `json:"failed_children,omitempty"`
}

type ChildStatus struct {
	Issue              int      `json:"issue"`
	Packet             string   `json:"packet"`
	Title              string   `json:"title"`
	Artifact           string   `json:"artifact"`
	Gate               string   `json:"gate"`
	Status             string   `json:"status"`
	EvidenceClass      string   `json:"evidence_class,omitempty"`
	ResultClaimAllowed bool     `json:"result_claim_allowed"`
	Detail             string   `json:"detail"`
	Missing            []string `json:"missing,omitempty"`
}

type Gate struct {
	Name   string `json:"name"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail"`
}

type childSpec struct {
	Issue    int
	Packet   string
	Title    string
	Artifact string
	Text     bool
	Check    func(map[string]any, ChildStatus) ChildStatus
}

func Build(root string, now time.Time) (*Report, error) {
	specs := []childSpec{
		{869, "A", "AgentDojo structural safety floor", "experiments/agent-live/agentdojo-fak-fullstack-20260625.json", false, checkAgentDojo},
		{870, "B", "GLM-5.2/vLLM agentic battery", "experiments/vllm/glm52-agentic-battery/final-check.json", false, checkGLM},
		{871, "C", "Opus-class SWE-bench smoke", "experiments/agent-live/swebench-opus-smoke-contract-20260626.json", false, checkOpus},
		{872, "D", "DeepSWE/R2E-Gym runner", "experiments/agent-live/deepswe-raw-fak-contract-20260626.json", false, checkDeepSWE},
		{873, "E", "ToolSandbox/tau3 policy-state", "experiments/agent-live/toolsandbox-official-run-contract-20260626.json", false, checkToolSandbox},
		{874, "F", "Terminal-Bench command boundary", "experiments/agent-live/terminalbench-official-run-contract-20260626.json", false, checkTerminalBench},
		{875, "G", "Browser/computer-use action mediation", "experiments/agent-live/browser-action-mediation-smoke-20260625.json", false, checkFixture},
		{876, "authority", "Agentic benchmark authority entry shape", "BENCHMARK-AUTHORITY.md", true, checkAuthority},
	}
	rep := &Report{
		Schema:        Schema,
		GeneratedAt:   now.UTC().Format(time.RFC3339),
		Epic:          868,
		ClaimBoundary: "Parent rollup only: reads committed child artifacts and refuses a #868 result claim until benchmark-native raw/fak grader evidence exists for at least one live lane.",
	}
	for _, spec := range specs {
		child := ChildStatus{
			Issue:    spec.Issue,
			Packet:   spec.Packet,
			Title:    spec.Title,
			Artifact: filepath.ToSlash(spec.Artifact),
			Gate:     "FAIL",
			Status:   "MISSING",
		}
		if spec.Text {
			text, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(spec.Artifact)))
			if err != nil {
				child.Detail = err.Error()
			} else {
				child = spec.Check(map[string]any{"text": string(text)}, child)
			}
		} else {
			doc, err := readJSON(filepath.Join(root, filepath.FromSlash(spec.Artifact)))
			if err != nil {
				child.Detail = err.Error()
			} else {
				child = spec.Check(doc, child)
			}
		}
		rep.Children = append(rep.Children, child)
		foldChild(&rep.Summary, child)
	}
	rep.Summary.ChildrenTotal = len(rep.Children)
	rep.Acceptance = acceptanceGates(rep)
	rep.ResultClaimAllowed = allGatesOK(rep.Acceptance)
	if rep.ResultClaimAllowed {
		rep.Status = "COMPLETE"
	} else {
		rep.Status = "PENDING_EXTERNAL_HARNESS"
	}
	return rep, nil
}

func readJSON(path string) (map[string]any, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc map[string]any
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.UseNumber()
	if err := dec.Decode(&doc); err != nil {
		return nil, err
	}
	var extra any
	if err := dec.Decode(&extra); err != io.EOF {
		if err == nil {
			return nil, fmt.Errorf("trailing JSON value")
		}
		return nil, err
	}
	return doc, nil
}

func checkAgentDojo(doc map[string]any, child ChildStatus) ChildStatus {
	child.Status = str(doc, "gate")
	fullStack := num(doc, "asr_fullstack_succeeded")
	benign := num(doc, "benign_completed")
	if child.Status == "PASS" && fullStack == 0 && benign > 0 {
		child.Gate = "PASS_LOCAL"
		child.Detail = fmt.Sprintf("local structural floor PASS; full-stack ASR successes %.0f; benign controls %.0f", fullStack, benign)
		return child
	}
	child.Gate = "FAIL"
	child.Detail = fmt.Sprintf("expected AgentDojo gate PASS with zero full-stack ASR and benign controls; gate=%q", child.Status)
	return child
}

func checkGLM(doc map[string]any, child ChildStatus) ChildStatus {
	child.Status = nestedString(doc, "summary", "status")
	passed := nestedNumber(doc, "summary", "required_passed")
	required := nestedNumber(doc, "summary", "required_artifacts")
	if child.Status == "COMPLETE" {
		child.Gate = "PASS_RESULT"
		child.ResultClaimAllowed = true
		child.Detail = "GLM final-check COMPLETE"
		return child
	}
	child.Gate = "PENDING_EXTERNAL_HARNESS"
	child.Detail = fmt.Sprintf("GLM final-check %s; required artifacts %.0f/%.0f passed", child.Status, passed, required)
	return child
}

func checkOpus(doc map[string]any, child ChildStatus) ChildStatus {
	child.Status = str(doc, "status")
	child.ResultClaimAllowed = boolv(doc, "result_claim_allowed")
	if child.ResultClaimAllowed {
		child.Gate = "PASS_RESULT"
		child.Detail = "Opus SWE-bench result claim enabled"
		return child
	}
	child.Gate = "PENDING_EXTERNAL_HARNESS"
	child.Detail = child.Status + "; raw/fak predictions and official SWE-bench reports still required"
	child.Missing = stringSlice(doc, "required_before_claim")
	return child
}

func checkDeepSWE(doc map[string]any, child ChildStatus) ChildStatus {
	child.Status = str(doc, "status")
	child.ResultClaimAllowed = boolv(doc, "result_claim_allowed")
	if child.ResultClaimAllowed {
		child.Gate = "PASS_RESULT"
		child.Detail = "DeepSWE result claim enabled"
		return child
	}
	child.Gate = "PENDING_EXTERNAL_HARNESS"
	child.Detail = child.Status + "; raw/fak DeepSWE predictions and official SWE-bench reports still required"
	child.Missing = stringSlice(doc, "required_before_claim")
	return child
}

func checkToolSandbox(doc map[string]any, child ChildStatus) ChildStatus {
	child.Status = str(doc, "status")
	child.ResultClaimAllowed = boolv(doc, "result_claim_allowed")
	child.EvidenceClass = "EXTERNAL_RUN_CONTRACT"
	if child.ResultClaimAllowed {
		child.Gate = "PASS_RESULT"
		child.Detail = "ToolSandbox/tau3 result claim enabled"
		return child
	}
	child.Gate = "PENDING_EXTERNAL_HARNESS"
	child.Detail = child.Status + "; benchmark-native tau3/ToolSandbox raw/fak outputs and grader summaries still required"
	child.Missing = stringSlice(doc, "required_before_claim")
	return child
}

func checkTerminalBench(doc map[string]any, child ChildStatus) ChildStatus {
	child.Status = str(doc, "status")
	child.ResultClaimAllowed = boolv(doc, "result_claim_allowed")
	child.EvidenceClass = "EXTERNAL_RUN_CONTRACT"
	if child.ResultClaimAllowed {
		child.Gate = "PASS_RESULT"
		child.Detail = "Terminal-Bench result claim enabled"
		return child
	}
	child.Gate = "PENDING_EXTERNAL_HARNESS"
	child.Detail = child.Status + "; benchmark-native Terminal-Bench raw/fak run dirs, command logs, and official test summaries still required"
	child.Missing = stringSlice(doc, "required_before_claim")
	return child
}

func checkFixture(doc map[string]any, child ChildStatus) ChildStatus {
	child.EvidenceClass = str(doc, "evidence_class")
	child.ResultClaimAllowed = boolv(doc, "result_claim_allowed")
	if child.ResultClaimAllowed {
		child.Gate = "PASS_RESULT"
		child.Status = "RESULT_CLAIM_ALLOWED"
		child.Detail = "fixture report unexpectedly allows result claim"
		return child
	}
	if child.EvidenceClass == "SIMULATED_LOCAL_FIXTURE" {
		child.Gate = "PASS_LOCAL"
		child.Status = "SIMULATED_LOCAL_FIXTURE"
		child.Detail = "local fixture is parseable and explicitly non-promotable"
		child.Missing = stringSlice(doc, "promotion_requirements")
		return child
	}
	child.Gate = "FAIL"
	child.Status = "UNKNOWN_FIXTURE_CLASS"
	child.Detail = "expected SIMULATED_LOCAL_FIXTURE with result_claim_allowed=false"
	return child
}

func checkAuthority(doc map[string]any, child ChildStatus) ChildStatus {
	text := fmt.Sprint(doc["text"])
	needles := []string{
		"AgentDojo Structural Safety Floor",
		"ToolSandbox/tau3 Adapter Smoke",
		"Promotion Gate",
	}
	var missing []string
	for _, needle := range needles {
		if !strings.Contains(text, needle) {
			missing = append(missing, needle)
		}
	}
	if len(missing) == 0 {
		child.Gate = "PASS_LOCAL"
		child.Status = "AUTHORITY_SHAPE_PRESENT"
		child.Detail = "authority rows and promotion gate text are present"
		return child
	}
	child.Gate = "FAIL"
	child.Status = "AUTHORITY_SHAPE_MISSING"
	child.Detail = "authority shape missing required text"
	child.Missing = missing
	return child
}

func foldChild(s *Summary, child ChildStatus) {
	if child.Gate != "FAIL" {
		s.ChildrenParsed++
	}
	if child.Gate == "PASS_LOCAL" {
		s.LocalEvidenceArtifacts++
	}
	if child.ResultClaimAllowed || child.Gate == "PASS_RESULT" {
		s.ResultClaimArtifacts++
	}
	if child.Gate == "PENDING_EXTERNAL_HARNESS" || (child.Gate == "PASS_LOCAL" && len(child.Missing) > 0) {
		s.PendingChildren = append(s.PendingChildren, child.Issue)
	}
	if child.Gate == "FAIL" {
		s.FailedChildren = append(s.FailedChildren, child.Issue)
	}
}

func acceptanceGates(rep *Report) []Gate {
	var parsedOK = rep.Summary.ChildrenParsed == rep.Summary.ChildrenTotal
	var pending []string
	for _, id := range rep.Summary.PendingChildren {
		pending = append(pending, fmt.Sprintf("#%d", id))
	}
	var failed []string
	for _, id := range rep.Summary.FailedChildren {
		failed = append(failed, fmt.Sprintf("#%d", id))
	}
	authorityShape := childGate(rep.Children, 876) == "PASS_LOCAL"
	resultClaim := rep.Summary.ResultClaimArtifacts > 0
	return []Gate{
		{Name: "child_artifacts_parse", OK: parsedOK, Detail: fmt.Sprintf("%d/%d child artifacts parsed; failed=%s", rep.Summary.ChildrenParsed, rep.Summary.ChildrenTotal, joinOrNone(failed))},
		{Name: "authority_entry_shape", OK: authorityShape, Detail: "BENCHMARK-AUTHORITY carries local rows plus the promotion-gate shape"},
		{Name: "result_packet_graduated", OK: resultClaim, Detail: fmt.Sprintf("%d result-claim-enabled artifact(s); pending live children=%s", rep.Summary.ResultClaimArtifacts, joinOrNone(pending))},
		{Name: "external_harness_grading", OK: resultClaim && len(rep.Summary.PendingChildren) == 0, Detail: "requires benchmark-native raw/fak grader output for open live lanes"},
		{Name: "compare_metrics_complete", OK: resultClaim, Detail: "requires solve/safe/cost-or-token/latency/policy/evidence metrics from a result-bearing compare artifact"},
		{Name: "final_writeup_ready", OK: resultClaim, Detail: "final #868 writeup waits for at least one real raw-vs-fak result packet"},
	}
}

func childGate(children []ChildStatus, issue int) string {
	for _, child := range children {
		if child.Issue == issue {
			return child.Gate
		}
	}
	return ""
}

func allGatesOK(gates []Gate) bool {
	for _, gate := range gates {
		if !gate.OK {
			return false
		}
	}
	return true
}

func str(m map[string]any, key string) string {
	v, _ := m[key].(string)
	return v
}

func boolv(m map[string]any, key string) bool {
	v, _ := m[key].(bool)
	return v
}

func num(m map[string]any, key string) float64 {
	switch v := m[key].(type) {
	case json.Number:
		f, _ := v.Float64()
		return f
	case float64:
		return v
	default:
		return 0
	}
}

func nestedString(m map[string]any, keys ...string) string {
	cur := any(m)
	for _, key := range keys {
		next, ok := cur.(map[string]any)
		if !ok {
			return ""
		}
		cur = next[key]
	}
	v, _ := cur.(string)
	return v
}

func nestedNumber(m map[string]any, keys ...string) float64 {
	cur := any(m)
	for _, key := range keys {
		next, ok := cur.(map[string]any)
		if !ok {
			return 0
		}
		cur = next[key]
	}
	switch v := cur.(type) {
	case json.Number:
		f, _ := v.Float64()
		return f
	case float64:
		return v
	default:
		return 0
	}
}

func stringSlice(m map[string]any, key string) []string {
	raw, _ := m[key].([]any)
	out := make([]string, 0, len(raw))
	for _, item := range raw {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func joinOrNone(v []string) string {
	if len(v) == 0 {
		return "none"
	}
	return strings.Join(v, ", ")
}
