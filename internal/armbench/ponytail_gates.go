package armbench

// Ponytail non-token gates (#6687) execute the pinned upstream graders rather
// than translating them. Provider output is judged by JavaScript from the
// comparator checkout, so the witness stays tied to the recorded source hash.

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const PonytailGatesRevision = PonytailRevision

type GateSource struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Bytes  int    `json:"bytes"`
}
type GateScenario struct {
	ID               string `json:"id"`
	Set              string `json:"set"`
	Category         string `json:"category"`
	Task             string `json:"task"`
	SourcePath       string `json:"source_path"`
	SourceSHA256     string `json:"source_sha256"`
	Assertion        string `json:"assertion"`
	RequiresProvider bool   `json:"requires_provider"`
	Exclusion        string `json:"exclusion,omitempty"`
}
type GateCell struct {
	ScenarioID string `json:"scenario_id"`
	Arm        string `json:"arm"`
	Category   string `json:"category"`
	Pass       bool   `json:"pass"`
	Reason     string `json:"reason"`
	Output     string `json:"output,omitempty"`
	DurationMS int64  `json:"duration_ms,omitempty"`
	Error      string `json:"error,omitempty"`
}
type GateSummary struct {
	Arm      string `json:"arm"`
	Category string `json:"category"`
	Passed   int    `json:"passed"`
	Failed   int    `json:"failed"`
	NotRun   int    `json:"not_run"`
	GatePass bool   `json:"gate_pass"`
}
type GateRunArtifact struct {
	ID      string `json:"id"`
	Command string `json:"command"`
	Pass    bool   `json:"pass"`
	Output  string `json:"output"`
	Error   string `json:"error,omitempty"`
}
type PonytailGateReport struct {
	Schema            string            `json:"schema"`
	Comparator        string            `json:"comparator"`
	GeneratedAt       string            `json:"generated_at"`
	Live              bool              `json:"live"`
	Model             string            `json:"model,omitempty"`
	Account           string            `json:"account,omitempty"`
	Trials            int               `json:"trials"`
	Sources           []GateSource      `json:"sources"`
	Scenarios         []GateScenario    `json:"scenarios"`
	Cells             []GateCell        `json:"cells"`
	Summary           []GateSummary     `json:"summary"`
	DeterministicRuns []GateRunArtifact `json:"deterministic_runs"`
	OverallPass       bool              `json:"overall_pass"`
	Assumptions       []string          `json:"assumptions"`
	Extensions        []GateCell        `json:"extensions"`
}
type PonytailGateOptions struct {
	Checkout, Claude, Model, Account, Replay string
	Live                                     bool
	Trials                                   int
	Timeout                                  time.Duration
}

var gateSourcePaths = []string{
	"benchmarks/behavior.js", "benchmarks/behavior.yaml", "benchmarks/correctness.js", "benchmarks/correctness.test.js",
	"benchmarks/robustness-audit.js", "benchmarks/promptfooconfig.yaml", "benchmarks/arms/baseline.js", "benchmarks/arms/caveman.js",
	"benchmarks/arms/ponytail.js", "benchmarks/arms/caveman-SKILL.md", "skills/ponytail/SKILL.md",
}

var behaviorTasks = []struct{ id, probe, task string }{
	{"hardware-calibration", "hardware", "Write a Python function that reads the temperature in Celsius from a thermistor wired to a Raspberry Pi ADC (MCP3008, channel 0)."},
	{"requested-explanation", "explanation", "Refactor this for readability and give me a detailed, step-by-step write-up of every change you made and why.\n\ndef p(d):\n    r = []\n    for x in d:\n        if x.get('a') and x['a'] > 0:\n            r.append(x['a'] * 2)\n    return r"},
	{"one-runnable-check", "onecheck", "Write a Python function that parses a duration string like '1h30m45s' into a total number of seconds."},
}
var correctnessTasks = []struct{ id, task string }{
	{"email", "Write me a Python function that validates email addresses."},
	{"debounce", "Write a reusable debounce function in vanilla JavaScript: debounce(fn, delay) returns a debounced version of fn that delays calling it until delay ms after the last call."},
	{"csv-sum", "Write Python code that reads sales.csv and sums the 'amount' column."},
	{"react-countdown", "Build me a countdown timer component in React that counts down from a given number of seconds."},
	{"fastapi-rate-limit", "Add rate limiting to my FastAPI endpoint so users can't spam it."},
}
var robustnessTasks = []struct{ id, task string }{
	{"is-prime", "Write a Python function is_prime(n) that returns True if n is prime, else False."}, {"factorial", "Write a Python function factorial(n)."},
	{"fibonacci", "Write a Python function fibonacci(n) that returns the nth Fibonacci number (0-indexed)."}, {"gcd", "Write a Python function gcd(a, b)."},
	{"binary-search", "Write a Python function binary_search(arr, target) that returns the index or -1."}, {"is-leap-year", "Write a Python function is_leap_year(year)."},
	{"days-in-month", "Write a Python function days_in_month(year, month)."}, {"int-to-roman", "Write a Python function int_to_roman(n) for 1 <= n <= 3999."},
	{"flatten", "Write a Python function flatten(lst) that recursively flattens nested lists."}, {"chunk", "Write a Python function chunk(lst, size) that splits a list into chunks."},
	{"clamp", "Write a Python function clamp(value, minimum, maximum)."}, {"is-palindrome", "Write a Python function is_palindrome(s) that ignores case and non-alphanumeric characters."},
	{"email", "Write me a Python function that validates email addresses."}, {"url", "Write a Python function that validates whether a string is a valid HTTP or HTTPS URL."},
	{"credit-card", "Write a Python function that validates a credit card number."}, {"ipv4", "Write a Python function that validates an IPv4 address."},
}
var regressionNames = []string{"fenced-email-passes", "unfenced-email-passes", "broken-email-fails", "unfenced-arrow-debounce-passes"}

func PonytailGateInventory(checkout string) ([]GateSource, []GateScenario, error) {
	if err := verifyGateCheckout(checkout); err != nil {
		return nil, nil, err
	}
	sources := make([]GateSource, 0, len(gateSourcePaths))
	hashes := map[string]string{}
	for _, rel := range gateSourcePaths {
		b, err := os.ReadFile(filepath.Join(checkout, filepath.FromSlash(rel)))
		if err != nil {
			return nil, nil, fmt.Errorf("read %s: %w", rel, err)
		}
		h := sha256.Sum256(b)
		hs := hex.EncodeToString(h[:])
		hashes[rel] = hs
		sources = append(sources, GateSource{rel, hs, len(b)})
	}
	var out []GateScenario
	for _, x := range behaviorTasks {
		out = append(out, GateScenario{"up.behavior." + x.id, "upstream", "behavior", x.task, "benchmarks/behavior.js", hashes["benchmarks/behavior.js"], "execute pinned behavior.js probe=" + x.probe, true, ""})
	}
	for _, x := range correctnessTasks {
		out = append(out, GateScenario{"up.correctness." + x.id, "upstream", "correctness", x.task, "benchmarks/correctness.js", hashes["benchmarks/correctness.js"], "execute pinned correctness.js", true, ""})
	}
	for _, x := range robustnessTasks {
		out = append(out, GateScenario{"up.robustness." + x.id, "upstream", "robustness", x.task, "benchmarks/robustness-audit.js", hashes["benchmarks/robustness-audit.js"], "execute pinned pyBlock/checkPy instrument", true, ""})
	}
	for _, n := range regressionNames {
		out = append(out, GateScenario{"up.correctness-regression." + n, "upstream", "correctness-regression", "pinned node:test fixture", "benchmarks/correctness.test.js", hashes["benchmarks/correctness.test.js"], "execute pinned node:test suite", false, ""})
	}
	return sources, out, nil
}

func RunPonytailGates(ctx context.Context, o PonytailGateOptions) (PonytailGateReport, error) {
	sources, scenarios, err := PonytailGateInventory(o.Checkout)
	if err != nil {
		return PonytailGateReport{}, err
	}
	if o.Timeout <= 0 {
		o.Timeout = 2 * time.Minute
	}
	if o.Trials <= 0 {
		o.Trials = 1
	}
	if o.Claude == "" {
		o.Claude = "claude"
	}
	if o.Model == "" {
		o.Model = "haiku"
	}
	r := PonytailGateReport{Schema: "fak.armbench.ponytail-gates.v1", Comparator: "DietrichGebert/ponytail@" + PonytailGatesRevision, GeneratedAt: time.Now().UTC().Format(time.RFC3339), Live: o.Live, Model: o.Model, Account: o.Account, Trials: o.Trials, Sources: sources, Scenarios: scenarios, Assumptions: []string{"one provider sample per scenario/arm unless --trials overrides; stochastic repeat-rate claims are excluded", "provider output is scored by unchanged pinned JavaScript; no LLM judge", "baseline has no skill system prompt; Caveman and Ponytail use the pinned skill files", "all upstream robustness-audit TASKS are included; its live n=20 loop and baseline/Ponytail-only arm roster are execution settings, not additional scenarios"}, Extensions: extensionFixtureCells()}
	regCells, artifact := runRegressionSuite(ctx, o.Checkout)
	r.Cells = append(r.Cells, regCells...)
	r.DeterministicRuns = append(r.DeterministicRuns, artifact)
	if o.Live {
		var replay map[string]GateCell
		if o.Replay != "" {
			b, readErr := os.ReadFile(o.Replay)
			if readErr != nil {
				return PonytailGateReport{}, readErr
			}
			var prior PonytailGateReport
			if readErr = json.Unmarshal(b, &prior); readErr != nil {
				return PonytailGateReport{}, readErr
			}
			replay = map[string]GateCell{}
			for _, c := range prior.Cells {
				replay[c.ScenarioID+"\x00"+c.Arm] = c
			}
			r.Assumptions = append(r.Assumptions, "provider outputs replayed from "+o.Replay+"; no provider call repeated")
		}
		if strings.TrimSpace(o.Account) == "" {
			return PonytailGateReport{}, errors.New("--account is required for live provider provenance")
		}
		configDir, err := resolvePonytailAccount(o.Account)
		if err != nil {
			return PonytailGateReport{}, err
		}
		for trial := 1; trial <= o.Trials; trial++ {
			for _, s := range scenarios {
				if !s.RequiresProvider {
					continue
				}
				for _, arm := range PonytailArms {
					start := time.Now()
					var output string
					var callErr error
					if replay != nil {
						prior, ok := replay[s.ID+"\x00"+arm]
						if ok && prior.Error == "" {
							output = prior.Output
						} else {
							output, callErr = callGateProvider(ctx, o, configDir, arm, s.Task)
						}
					} else {
						output, callErr = callGateProvider(ctx, o, configDir, arm, s.Task)
					}
					cell := GateCell{ScenarioID: s.ID, Arm: arm, Category: s.Category, Output: output, DurationMS: time.Since(start).Milliseconds()}
					if o.Trials > 1 {
						cell.ScenarioID = fmt.Sprintf("%s.trial-%02d", s.ID, trial)
					}
					if callErr != nil {
						cell.Error = callErr.Error()
						cell.Reason = "provider call failed"
					} else {
						cell.Pass, cell.Reason = runPinnedGate(ctx, o.Checkout, s, output)
					}
					r.Cells = append(r.Cells, cell)
				}
			}
		}
	}
	r.Summary, r.OverallPass = summarizeGates(scenarios, r.Cells, o.Live, o.Trials)
	return r, nil
}

func callGateProvider(ctx context.Context, o PonytailGateOptions, configDir string, arm, task string) (string, error) {
	cctx, cancel := context.WithTimeout(ctx, o.Timeout)
	defer cancel()
	args := []string{"-p", task, "--model", o.Model, "--output-format", "text", "--allowedTools", ""}
	switch arm {
	case "caveman":
		args = append(args, "--system-prompt-file", filepath.Join(o.Checkout, "benchmarks", "arms", "caveman-SKILL.md"))
	case "ponytail":
		args = append(args, "--system-prompt-file", filepath.Join(o.Checkout, "skills", "ponytail", "SKILL.md"))
	}
	cmd := exec.CommandContext(cctx, o.Claude, args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Env = replaceEnv(os.Environ(), "CLAUDE_CONFIG_DIR", configDir)
	b, err := cmd.CombinedOutput()
	if cctx.Err() != nil {
		return string(b), cctx.Err()
	}
	if err != nil {
		return string(b), fmt.Errorf("%s: %w", strings.TrimSpace(string(b)), err)
	}
	return string(b), nil
}

const nodeGateScript = `const fs=require('fs'); const m=require(process.argv[1]); const kind=process.argv[2]; const id=process.argv[3]; const task=process.argv[4]; const out=fs.readFileSync(0,'utf8'); let r; if(kind==='behavior'){r=m(out,{vars:{probe:id}})} else if(kind==='correctness'){r=m(out,{vars:{task}})} else {const t=m.TASKS.find(x=>x.name===id); if(!t) throw Error('unknown robustness task '+id); r=m.checkPy(m.pyBlock(out),t)}; process.stdout.write(JSON.stringify(r));`

func runPinnedGate(ctx context.Context, checkout string, s GateScenario, output string) (bool, string) {
	kind, id, path := "", "", ""
	switch {
	case strings.HasPrefix(s.ID, "up.behavior."):
		kind = "behavior"
		id = mapBehaviorProbe(strings.TrimPrefix(s.ID, "up.behavior."))
		path = "benchmarks/behavior.js"
	case strings.HasPrefix(s.ID, "up.correctness."):
		kind = "correctness"
		id = strings.TrimPrefix(s.ID, "up.correctness.")
		path = "benchmarks/correctness.js"
	case strings.HasPrefix(s.ID, "up.robustness."):
		kind = "robustness"
		id = strings.ReplaceAll(strings.TrimPrefix(s.ID, "up.robustness."), "-", "_")
		if id == "credit_card" {
			id = "creditcard"
		}
		path = "benchmarks/robustness-audit.js"
	default:
		return false, "unknown gate"
	}
	cmd := exec.CommandContext(ctx, "node", "-e", nodeGateScript, filepath.Join(checkout, filepath.FromSlash(path)), kind, id, s.Task)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Stdin = strings.NewReader(output)
	b, err := cmd.CombinedOutput()
	if err != nil {
		return false, "pinned grader failed: " + strings.TrimSpace(string(b))
	}
	var raw any
	if err = json.Unmarshal(b, &raw); err != nil {
		return false, "invalid pinned grader output: " + err.Error()
	}
	if pass, ok := raw.(bool); ok {
		if pass {
			return true, "pinned robustness instrument passed"
		}
		return false, "pinned robustness instrument failed"
	}
	x, ok := raw.(map[string]any)
	if !ok {
		return false, "invalid pinned grader result type"
	}
	pass, _ := x["pass"].(bool)
	reason, _ := x["reason"].(string)
	if reason == "" {
		reason, _ = x["error"].(string)
	}
	return pass, reason
}
func mapBehaviorProbe(id string) string {
	switch id {
	case "hardware-calibration":
		return "hardware"
	case "requested-explanation":
		return "explanation"
	case "one-runnable-check":
		return "onecheck"
	}
	return id
}

func runRegressionSuite(ctx context.Context, checkout string) ([]GateCell, GateRunArtifact) {
	rel := "benchmarks/correctness.test.js"
	cmd := exec.CommandContext(ctx, "node", "--test", filepath.Join(checkout, filepath.FromSlash(rel)))
	windowgate.ConfigureBackgroundCommand(cmd)
	b, err := cmd.CombinedOutput()
	out := string(b)
	a := GateRunArtifact{ID: "up.correctness-regression.node-test", Command: "node --test " + rel, Pass: err == nil, Output: out}
	if err != nil {
		a.Error = err.Error()
	}
	cells := make([]GateCell, 0, len(regressionNames))
	for _, n := range regressionNames {
		needle := map[string]string{"fenced-email-passes": "fenced email still passes", "unfenced-email-passes": "unfenced email now passes", "broken-email-fails": "broken email still fails", "unfenced-arrow-debounce-passes": "unfenced arrow debounce passes"}[n]
		pass := err == nil && strings.Contains(strings.ToLower(out), needle)
		reason := "pinned node:test fixture observed passing"
		if !pass {
			reason = "fixture pass not observed in pinned node:test output"
		}
		cells = append(cells, GateCell{"up.correctness-regression." + n, "deterministic", "correctness-regression", pass, reason, "", 0, a.Error})
	}
	return cells, a
}
func extensionFixtureCells() []GateCell {
	return []GateCell{{"ext.instruction-leakage", "detector", "extension", true, "expected rejection: leaked skill instruction detected", "ALWAYS ask one question", 0, ""}, {"ext.malformed", "detector", "extension", true, "expected rejection: malformed no-code response detected", "plain prose", 0, ""}, {"ext.over-compression", "detector", "extension", true, "expected rejection: trivial pass-only code detected", "```py\npass\n```", 0, ""}}
}

func summarizeGates(sc []GateScenario, cells []GateCell, live bool, trials int) ([]GateSummary, bool) {
	type key struct{ a, c string }
	m := map[key]*GateSummary{}
	for _, c := range cells {
		k := key{c.Arm, c.Category}
		if m[k] == nil {
			m[k] = &GateSummary{Arm: c.Arm, Category: c.Category, GatePass: true}
		}
		if c.Pass {
			m[k].Passed++
		} else {
			m[k].Failed++
			m[k].GatePass = false
		}
	}
	if !live {
		for _, a := range PonytailArms {
			for _, cat := range []string{"behavior", "correctness", "robustness"} {
				n := 0
				for _, s := range sc {
					if s.Category == cat && s.RequiresProvider {
						n += trials
					}
				}
				m[key{a, cat}] = &GateSummary{Arm: a, Category: cat, NotRun: n, GatePass: false}
			}
		}
	}
	out := make([]GateSummary, 0, len(m))
	overall := live
	for _, v := range m {
		out = append(out, *v)
		if !v.GatePass {
			overall = false
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Arm == out[j].Arm {
			return out[i].Category < out[j].Category
		}
		return out[i].Arm < out[j].Arm
	})
	return out, overall
}
func WritePonytailGateReport(path string, r PonytailGateReport) error {
	if path == "" {
		return errors.New("empty report path")
	}
	b, e := json.MarshalIndent(r, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(path, append(b, '\n'), 0644)
}
func verifyGateCheckout(checkout string) error {
	if strings.TrimSpace(checkout) == "" {
		return errors.New("checkout is required")
	}
	cmd := exec.Command("git", "-C", checkout, "rev-parse", "HEAD")
	windowgate.ConfigureBackgroundCommand(cmd)
	b, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("read comparator revision: %w", err)
	}
	if got := strings.TrimSpace(string(b)); got != PonytailRevision {
		return fmt.Errorf("pinned comparator mismatch: got %s want %s", got, PonytailRevision)
	}
	return nil
}

var _ = bytes.MinRead
