// Command resultbudgetdemo demonstrates deterministic tool-result request shaping.
package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

const (
	knownTool      = "github.search_issues"
	resultArgument = "/per_page"
)

type operatorMode string

const (
	modeEnforce operatorMode = "enforce"
	modeObserve operatorMode = "observe"
)

type resultBudget struct {
	Items int64 `json:"items"`
}

type argumentContract struct {
	Tool         string `json:"tool"`
	Argument     string `json:"argument"`
	Dimension    string `json:"dimension"`
	Maximum      int64  `json:"maximum"`
	Minimum      int64  `json:"minimum"`
	SafeToReduce bool   `json:"safe_to_reduce"`
}

type shapingConfig struct {
	Kind          string             `json:"kind"`
	Version       string             `json:"version"`
	Name          string             `json:"name"`
	Mode          operatorMode       `json:"mode"`
	DefaultBudget resultBudget       `json:"default_budget"`
	Contracts     []argumentContract `json:"contracts"`
}

type structuredIntent struct {
	Exhaustive bool `json:"exhaustive"`
}

type toolCall struct {
	Tool   string
	Args   json.RawMessage
	Intent structuredIntent
}

type budgetChange struct {
	Path      string `json:"path"`
	From      int64  `json:"from"`
	To        int64  `json:"to"`
	Dimension string `json:"dimension"`
}

type shapingIdentity struct {
	Name    string       `json:"name"`
	Version string       `json:"version"`
	SHA256  string       `json:"sha256"`
	Mode    operatorMode `json:"mode"`
}

type actualResult struct {
	Items int64 `json:"items"`
}

type continuation struct {
	Kind      string `json:"kind"`
	Available bool   `json:"available"`
}

type executionReceipt struct {
	Outcome             string          `json:"decision"`
	Reason              string          `json:"reason"`
	Tool                string          `json:"tool"`
	OriginalArgsSHA256  string          `json:"original_args_sha256"`
	EffectiveArgsSHA256 string          `json:"effective_args_sha256"`
	Changes             []budgetChange  `json:"changes,omitempty"`
	ProposedChanges     []budgetChange  `json:"proposed_changes,omitempty"`
	Policy              shapingIdentity `json:"policy"`
	ModelRoundTrips     int             `json:"model_round_trips"`
	Actual              actualResult    `json:"actual"`
	Continuation        *continuation   `json:"continuation,omitempty"`
}

type caseResult struct {
	Name         string           `json:"name"`
	Requested    int64            `json:"requested_items"`
	ToolObserved int64            `json:"tool_observed_items"`
	Receipt      executionReceipt `json:"receipt"`
}

type demoReport struct {
	Schema    string       `json:"schema"`
	Verdict   string       `json:"verdict"`
	ToolCalls int          `json:"tool_calls"`
	Cases     []caseResult `json:"cases"`
	SelfCheck string       `json:"self_check"`
}

type budgetAdapter struct {
	policy shapingConfig
}

func (a budgetAdapter) adapt(call toolCall) ([]byte, executionReceipt, error) {
	original := append([]byte(nil), call.Args...)
	receipt := executionReceipt{
		Outcome:             "pass",
		Reason:              "within_budget",
		Tool:                call.Tool,
		OriginalArgsSHA256:  digest(original),
		EffectiveArgsSHA256: digest(original),
		Policy:              identity(a.policy),
		ModelRoundTrips:     0,
	}

	contract, ok := exactContract(a.policy.Contracts, call.Tool)
	if !ok {
		receipt.Reason = "unknown_tool_contract"
		return original, receipt, nil
	}
	if call.Intent.Exhaustive {
		receipt.Reason = "structured_exhaustive_intent"
		return original, receipt, nil
	}
	key, ok := pointerKey(contract.Argument)
	if !ok {
		receipt.Reason = "argument_path_unavailable"
		return original, receipt, nil
	}
	args, err := decodeArgs(original)
	if err != nil {
		return nil, receipt, err
	}
	requested, ok := integerArgument(args[key])
	if !ok || requested < contract.Minimum {
		receipt.Reason = "unshapable_argument"
		return original, receipt, nil
	}
	if requested <= contract.Maximum || !contract.SafeToReduce {
		return original, receipt, nil
	}

	change := budgetChange{Path: contract.Argument, From: requested, To: contract.Maximum, Dimension: contract.Dimension}
	if a.policy.Mode == modeObserve {
		receipt.Outcome = "observe"
		receipt.Reason = "clamp_proposed_above_maximum"
		receipt.ProposedChanges = []budgetChange{change}
		return original, receipt, nil
	}
	if a.policy.Mode != modeEnforce {
		return nil, receipt, fmt.Errorf("unsupported policy mode %q", a.policy.Mode)
	}

	args[key] = contract.Maximum
	effective, err := json.Marshal(args)
	if err != nil {
		return nil, receipt, fmt.Errorf("encode effective arguments: %w", err)
	}
	receipt.Outcome = "clamp"
	receipt.Reason = "requested_items_above_maximum"
	receipt.Changes = []budgetChange{change}
	receipt.EffectiveArgsSHA256 = digest(effective)
	receipt.Continuation = &continuation{Kind: "rerun", Available: true}
	return effective, receipt, nil
}

type fakeTool struct {
	calls []toolCall
}

func (f *fakeTool) execute(call toolCall) (int64, error) {
	args, err := decodeArgs(call.Args)
	if err != nil {
		return 0, err
	}
	key, ok := pointerKey(resultArgument)
	if !ok {
		return 0, errors.New("invalid demo argument pointer")
	}
	observed, ok := integerArgument(args[key])
	if !ok {
		return 0, errors.New("fake tool requires an integer per_page")
	}
	f.calls = append(f.calls, call)
	return observed, nil
}

func executeCase(name string, policy shapingConfig, call toolCall, tool *fakeTool) (caseResult, error) {
	requestedArgs, err := decodeArgs(call.Args)
	if err != nil {
		return caseResult{}, err
	}
	key, _ := pointerKey(resultArgument)
	requested, ok := integerArgument(requestedArgs[key])
	if !ok {
		return caseResult{}, errors.New("demo request has no integer per_page")
	}

	effective, receipt, err := (budgetAdapter{policy: policy}).adapt(call)
	if err != nil {
		return caseResult{}, err
	}
	call.Args = effective
	observed, err := tool.execute(call)
	if err != nil {
		return caseResult{}, err
	}
	receipt.Actual = actualResult{Items: observed}
	return caseResult{Name: name, Requested: requested, ToolObserved: observed, Receipt: receipt}, nil
}

func runSelfcheck() (demoReport, error) {
	policy := shapingConfig{
		Kind:          "fak/tool-result-budget",
		Version:       "1.0.0",
		Name:          "thimble/default",
		Mode:          modeEnforce,
		DefaultBudget: resultBudget{Items: 10},
		Contracts: []argumentContract{{
			Tool: knownTool, Argument: resultArgument, Dimension: "items",
			Maximum: 10, Minimum: 1, SafeToReduce: true,
		}},
	}
	request := json.RawMessage(`{"query":"open","per_page":500}`)
	tool := &fakeTool{}
	cases := make([]caseResult, 0, 4)

	add := func(name string, p shapingConfig, call toolCall) error {
		got, err := executeCase(name, p, call, tool)
		if err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		cases = append(cases, got)
		return nil
	}
	if err := add("enforce", policy, toolCall{Tool: knownTool, Args: request}); err != nil {
		return demoReport{}, err
	}
	observe := policy
	observe.Mode = modeObserve
	if err := add("observe", observe, toolCall{Tool: knownTool, Args: request}); err != nil {
		return demoReport{}, err
	}
	if err := add("exhaustive-intent", policy, toolCall{Tool: knownTool, Args: request, Intent: structuredIntent{Exhaustive: true}}); err != nil {
		return demoReport{}, err
	}
	if err := add("unknown-tool", policy, toolCall{Tool: "github.unknown_search", Args: request}); err != nil {
		return demoReport{}, err
	}

	report := demoReport{Schema: "fak-result-budget-demo/1", Verdict: "PASS", ToolCalls: len(tool.calls), Cases: cases}
	if err := validateReport(report); err != nil {
		report.Verdict = "FAIL"
		return report, err
	}
	report.SelfCheck = "PASS"
	return report, nil
}

func validateReport(report demoReport) error {
	if report.Schema != "fak-result-budget-demo/1" || report.ToolCalls != 4 || len(report.Cases) != 4 {
		return fmt.Errorf("unexpected report envelope: schema=%q calls=%d cases=%d", report.Schema, report.ToolCalls, len(report.Cases))
	}
	byName := make(map[string]caseResult, len(report.Cases))
	for _, result := range report.Cases {
		byName[result.Name] = result
		if result.Requested != 500 || result.Receipt.ModelRoundTrips != 0 {
			return fmt.Errorf("%s: requested=%d model_round_trips=%d", result.Name, result.Requested, result.Receipt.ModelRoundTrips)
		}
	}
	enforce := byName["enforce"]
	if enforce.ToolObserved != 10 || enforce.Receipt.Outcome != "clamp" || enforce.Receipt.OriginalArgsSHA256 == enforce.Receipt.EffectiveArgsSHA256 || len(enforce.Receipt.Changes) != 1 || enforce.Receipt.Changes[0].From != 500 || enforce.Receipt.Changes[0].To != 10 || enforce.Receipt.Continuation == nil || !enforce.Receipt.Continuation.Available {
		return fmt.Errorf("enforce witness mismatch: %+v", enforce)
	}
	observe := byName["observe"]
	if observe.ToolObserved != 500 || observe.Receipt.Outcome != "observe" || observe.Receipt.OriginalArgsSHA256 != observe.Receipt.EffectiveArgsSHA256 || len(observe.Receipt.ProposedChanges) != 1 || observe.Receipt.ProposedChanges[0].To != 10 {
		return fmt.Errorf("observe witness mismatch: %+v", observe)
	}
	exhaustive := byName["exhaustive-intent"]
	if exhaustive.ToolObserved != 500 || exhaustive.Receipt.Outcome != "pass" || exhaustive.Receipt.Reason != "structured_exhaustive_intent" {
		return fmt.Errorf("exhaustive-intent witness mismatch: %+v", exhaustive)
	}
	unknown := byName["unknown-tool"]
	if unknown.ToolObserved != 500 || unknown.Receipt.Outcome != "pass" || unknown.Receipt.Reason != "unknown_tool_contract" {
		return fmt.Errorf("unknown-tool witness mismatch: %+v", unknown)
	}
	return nil
}

func exactContract(contracts []argumentContract, tool string) (argumentContract, bool) {
	for _, contract := range contracts {
		if contract.Tool == tool && contract.Argument == resultArgument {
			return contract, true
		}
	}
	return argumentContract{}, false
}

func pointerKey(pointer string) (string, bool) {
	if !strings.HasPrefix(pointer, "/") || strings.Contains(pointer[1:], "/") {
		return "", false
	}
	key := strings.ReplaceAll(strings.ReplaceAll(pointer[1:], "~1", "/"), "~0", "~")
	return key, key != ""
}

func decodeArgs(raw []byte) (map[string]any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var args map[string]any
	if err := decoder.Decode(&args); err != nil {
		return nil, fmt.Errorf("decode arguments: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, errors.New("decode arguments: trailing JSON value")
	}
	return args, nil
}

func integerArgument(value any) (int64, bool) {
	number, ok := value.(json.Number)
	if !ok {
		return 0, false
	}
	parsed, err := strconv.ParseInt(number.String(), 10, 64)
	return parsed, err == nil
}

func identity(policy shapingConfig) shapingIdentity {
	encoded, err := json.Marshal(policy)
	if err != nil {
		panic(err)
	}
	return shapingIdentity{Name: policy.Name, Version: policy.Version, SHA256: digest(encoded), Mode: policy.Mode}
}

func digest(value []byte) string {
	sum := sha256.Sum256(value)
	return hex.EncodeToString(sum[:])
}

func main() {
	selfcheck := flag.Bool("selfcheck", false, "run the deterministic result-budget witness")
	pretty := flag.Bool("pretty", false, "indent JSON output")
	flag.Parse()
	if !*selfcheck {
		fmt.Fprintln(os.Stderr, "usage: resultbudgetdemo -selfcheck [-pretty]")
		os.Exit(2)
	}

	report, err := runSelfcheck()
	if err != nil {
		fmt.Fprintln(os.Stderr, "resultbudgetdemo:", err)
		os.Exit(1)
	}
	encoder := json.NewEncoder(os.Stdout)
	if *pretty {
		encoder.SetIndent("", "  ")
	}
	if err := encoder.Encode(report); err != nil {
		fmt.Fprintln(os.Stderr, "resultbudgetdemo:", err)
		os.Exit(1)
	}
}
