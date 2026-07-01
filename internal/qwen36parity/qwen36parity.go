package qwen36parity

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const (
	Schema      = "fak.qwen36-parity-witness-gate.v1"
	WitnessDir  = "experiments/agent-live"
	WitnessGlob = "qwen36-mac-parity-gate-*.json"
	BarPrefill  = 51.55
	BarDecode   = 7.29
)

var (
	OracleIDs            = []int{248068, 198, 90700}
	KnownFakIDs          = []int{248068, 198, 8160}
	KnownDivergenceIndex = 2
	IssueLinks           = map[string][]int{
		"correctness":   {64, 1458},
		"metal_gate":    {71, 1458},
		"speed":         {64, 1382},
		"q6k_fused_mlp": {1381},
	}
	scrubPlaceholderHosts = map[string]bool{"node-macos-a.local": true, "node-macos-a": true}
	ipv4RE                = regexp.MustCompile(`\b(?:\d{1,3}\.){3}\d{1,3}\b`)
	sshUserHostRE         = regexp.MustCompile(`\b[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\b`)
	tailnetRE             = regexp.MustCompile(`\b[a-zA-Z0-9-]+\.ts\.net\b`)
	dotLocalRE            = regexp.MustCompile(`\b([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*\.local)\b`)
)

type Correctness struct {
	Verdict                 string `json:"verdict"`
	FakIDs                  []int  `json:"fak_ids"`
	OracleIDs               []int  `json:"oracle_ids"`
	FirstDivergenceIndex    *int   `json:"first_divergence_index"`
	ReportedDivergenceIndex any    `json:"reported_divergence_index"`
	KnownDivergenceIndex    int    `json:"known_divergence_index"`
	Regression              bool   `json:"regression"`
	Note                    string `json:"note"`
}

type Speed struct {
	PrefillTokS    *float64 `json:"prefill_tok_s"`
	DecodeTokS     *float64 `json:"decode_tok_s"`
	BarPrefillTokS float64  `json:"bar_prefill_tok_s"`
	BarDecodeTokS  float64  `json:"bar_decode_tok_s"`
	PrefillRatio   float64  `json:"prefill_ratio"`
	DecodeRatio    float64  `json:"decode_ratio"`
	MinRatio       float64  `json:"min_ratio"`
	Gated          bool     `json:"gated"`
	Failures       []string `json:"failures"`
}

type Report struct {
	Schema      string           `json:"schema"`
	Witness     string           `json:"witness"`
	Status      string           `json:"status,omitempty"`
	Issues      map[string][]int `json:"issues"`
	Commit      any              `json:"commit,omitempty"`
	CapturedAt  any              `json:"captured_at,omitempty"`
	Host        any              `json:"host,omitempty"`
	Correctness Correctness      `json:"correctness,omitempty"`
	MetalGate   map[string]any   `json:"metal_gate,omitempty"`
	Speed       Speed            `json:"speed,omitempty"`
	Scrub       map[string]any   `json:"scrub,omitempty"`
	Passed      bool             `json:"passed"`
	Failures    []string         `json:"failures"`
	Note        string           `json:"note,omitempty"`
}

func Rel(path, root string) string {
	absPath, err1 := filepath.Abs(path)
	absRoot, err2 := filepath.Abs(root)
	if err1 == nil && err2 == nil {
		if rel, err := filepath.Rel(absRoot, absPath); err == nil && !strings.HasPrefix(rel, "..") {
			return filepath.ToSlash(rel)
		}
	}
	return filepath.ToSlash(path)
}

func FindLatestWitness(directory string) (string, bool) {
	hits, _ := filepath.Glob(filepath.Join(directory, WitnessGlob))
	sort.Strings(hits)
	if len(hits) == 0 {
		return "", false
	}
	return hits[len(hits)-1], true
}

func FirstDivergence(a, b []int) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) != len(b) {
		return n
	}
	return -1
}

func GradeCorrectness(witness map[string]any) Correctness {
	fak, fakOK := asIntList(witness["fak_token_ids"])
	oracle, oracleOK := asIntList(witness["llamacpp_token_ids"])
	reported := witness["first_divergence_index"]
	if !fakOK || !oracleOK {
		return corr("MALFORMED", fak, oracle, nil, reported, true, "fak_token_ids or llamacpp_token_ids missing / not an int array")
	}
	if !intPrefixEqual(oracle, OracleIDs) {
		div := FirstDivergence(fak, oracle)
		return corr("ORACLE_DRIFT", fak, oracle, &div, reported, true, fmt.Sprintf("llama.cpp oracle prefix changed from the frozen %v -> %v; the reference moved (setup/measurement bug), not a fak result", OracleIDs, prefix(oracle, len(OracleIDs))))
	}
	div := FirstDivergence(fak, oracle)
	if ri, ok := asStrictInt(reported); ok && ri != div {
		return corr("MALFORMED", fak, oracle, &div, reported, true, fmt.Sprintf("witness first_divergence_index=%d disagrees with the ids (computed %d)", ri, div))
	}
	if div == -1 {
		return corr("PARITY", fak, oracle, &div, reported, false, "fak reproduces the llama.cpp token stream through the compared window - the token-3 drift is CLOSED")
	}
	if div < KnownDivergenceIndex {
		return corr("REGRESSION", fak, oracle, &div, reported, true, fmt.Sprintf("fak diverges at index %d, EARLIER than the known index %d - strictly worse than the recorded state", div, KnownDivergenceIndex))
	}
	if div > KnownDivergenceIndex {
		return corr("PROGRESS", fak, oracle, &div, reported, false, fmt.Sprintf("fak diverges at index %d, LATER than the known index %d - partial improvement; confirm it is not a fluke", div, KnownDivergenceIndex))
	}
	if intPrefixEqual(fak, KnownFakIDs) {
		return corr("KNOWN_DRIFT", fak, oracle, &div, reported, false, "the exact recorded token-3 near-tie drift, unchanged (expected today)")
	}
	return corr("DRIFT_CHANGED", fak, oracle, &div, reported, false, fmt.Sprintf("diverges at the known index %d but on different ids (got %v, recorded %v) - not worse, but the signature moved", div, fak, KnownFakIDs))
}

func ScanForLeaks(witness map[string]any) []string {
	data, _ := json.Marshal(witness)
	blob := string(data)
	seen := map[string]bool{}
	var leaks []string
	add := func(s string) {
		if !seen[s] {
			seen[s] = true
			leaks = append(leaks, s)
		}
	}
	for _, ip := range ipv4RE.FindAllString(blob, -1) {
		add("ipv4-address:" + ip)
	}
	for _, uh := range sshUserHostRE.FindAllString(blob, -1) {
		add("ssh-user@host:" + uh)
	}
	for _, tn := range tailnetRE.FindAllString(blob, -1) {
		add("tailnet-host:" + tn)
	}
	for _, m := range dotLocalRE.FindAllStringSubmatch(blob, -1) {
		if len(m) > 1 && !scrubPlaceholderHosts[m[1]] {
			add("non-placeholder-host:" + m[1])
		}
	}
	sort.Strings(leaks)
	return leaks
}

func GradeSpeed(witness map[string]any, minRatio float64) Speed {
	pref := asFloatPtr(witness["fak_prefill_tok_s"])
	dec := asFloatPtr(witness["fak_decode_tok_s"])
	barP := asFloat(witness["bar_prefill_tok_s"], BarPrefill)
	barD := asFloat(witness["bar_decode_tok_s"], BarDecode)
	pratio, dratio := 0.0, 0.0
	if pref != nil && *pref != 0 && barP != 0 {
		pratio = *pref / barP
	}
	if dec != nil && *dec != 0 && barD != 0 {
		dratio = *dec / barD
	}
	var failures []string
	if minRatio > 0 {
		if pref != nil && *pref != 0 && pratio < minRatio {
			failures = append(failures, fmt.Sprintf("prefill %.4g tok/s < %gx bar %.4g", *pref, minRatio, barP))
		}
		if dec != nil && *dec != 0 && dratio < minRatio {
			failures = append(failures, fmt.Sprintf("decode %.4g tok/s < %gx bar %.4g", *dec, minRatio, barD))
		}
	}
	return Speed{
		PrefillTokS: pref, DecodeTokS: dec, BarPrefillTokS: barP, BarDecodeTokS: barD,
		PrefillRatio: round4(pratio), DecodeRatio: round4(dratio), MinRatio: minRatio,
		Gated: minRatio > 0, Failures: failures,
	}
}

func GradeWitness(witness map[string]any, witnessPath string, minRatio float64) Report {
	correctness := GradeCorrectness(witness)
	speed := GradeSpeed(witness, minRatio)
	leaks := ScanForLeaks(witness)
	var failures []string
	if correctness.Regression {
		failures = append(failures, fmt.Sprintf("correctness %s: %s", correctness.Verdict, correctness.Note))
	}
	if len(leaks) > 0 {
		failures = append(failures, "scrub leak(s): "+strings.Join(leaks, ", "))
	}
	failures = append(failures, speed.Failures...)
	return Report{
		Schema: Schema, Witness: witnessPath, Issues: IssueLinks,
		Commit: witness["commit"], CapturedAt: witness["captured_at"], Host: witness["host"],
		Correctness: correctness,
		MetalGate:   map[string]any{"pass": asBool(witness["metal_gate_pass"]), "gated": false, "note": "recorded; gated by the gate script itself, not re-gated here"},
		Speed:       speed,
		Scrub:       map[string]any{"clean": len(leaks) == 0, "leaks": leaks},
		Passed:      len(failures) == 0, Failures: failures,
	}
}

func NoWitnessReport(require bool) Report {
	note := fmt.Sprintf("no Mac parity witness found - a SKIP is not a PASS (%s/%s)", WitnessDir, WitnessGlob)
	failures := []string(nil)
	if require {
		failures = []string{note}
	}
	return Report{Schema: Schema, Witness: "<none>", Status: "NO_WITNESS", Issues: IssueLinks, Passed: !require, Failures: failures, Note: note}
}

func IssueLinksMarkdown(issues map[string][]int) string {
	keys := []string{"correctness", "metal_gate", "speed", "q6k_fused_mlp"}
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		var nums []string
		for _, n := range issues[key] {
			nums = append(nums, fmt.Sprintf("#%d", n))
		}
		parts = append(parts, strings.ReplaceAll(key, "_", " ")+": "+strings.Join(nums, "/"))
	}
	return strings.Join(parts, "; ")
}

func RenderMarkdown(report Report) string {
	if report.Status == "NO_WITNESS" {
		return fmt.Sprintf("# Qwen3.6 Parity Witness Gate\n\n- Verdict: NO_WITNESS\n- Issues: %s\n- %s\n", IssueLinksMarkdown(report.Issues), report.Note)
	}
	c, s := report.Correctness, report.Speed
	metalPass := false
	if report.MetalGate != nil {
		metalPass, _ = report.MetalGate["pass"].(bool)
	}
	scrubClean := false
	var scrubLeaks string
	if report.Scrub != nil {
		scrubClean, _ = report.Scrub["clean"].(bool)
		if leaks, ok := report.Scrub["leaks"].([]string); ok {
			scrubLeaks = strings.Join(leaks, ", ")
		} else if leaks, ok := report.Scrub["leaks"].([]any); ok {
			var parts []string
			for _, leak := range leaks {
				parts = append(parts, fmt.Sprint(leak))
			}
			scrubLeaks = strings.Join(parts, ", ")
		}
	}
	metal := "not-yet"
	if metalPass {
		metal = "PASS"
	}
	scrub := "clean"
	if !scrubClean {
		scrub = "LEAK -> " + scrubLeaks
	}
	lines := []string{
		"# Qwen3.6 Parity Witness Gate",
		"",
		fmt.Sprintf("- Verdict: %s", passFail(report.Passed)),
		fmt.Sprintf("- Witness: `%s`  (commit `%v`)", report.Witness, report.Commit),
		fmt.Sprintf("- Issues: %s", IssueLinksMarkdown(report.Issues)),
		fmt.Sprintf("- Correctness: **%s** -- %s", c.Verdict, c.Note),
		fmt.Sprintf("  - fak ids `%v`  vs oracle `%v`  (first divergence index %s, known %d)", c.FakIDs, c.OracleIDs, divString(c.FirstDivergenceIndex), c.KnownDivergenceIndex),
		fmt.Sprintf("- Metal hybrid-prefill gate (#71): %s (recorded, not gated here)", metal),
		fmt.Sprintf("- Speed (recorded): prefill %s tok/s (%v x bar %v), decode %s tok/s (%v x bar %v)", floatPtrString(s.PrefillTokS), s.PrefillRatio, s.BarPrefillTokS, floatPtrString(s.DecodeTokS), s.DecodeRatio, s.BarDecodeTokS),
		"- Scrub: " + scrub,
	}
	if len(report.Failures) > 0 {
		lines = append(lines, "", "Failures (fail-closed):")
		for _, f := range report.Failures {
			lines = append(lines, "- "+f)
		}
	}
	return strings.Join(lines, "\n") + "\n"
}

func LoadAndGrade(root, witnessPath string, require bool, minRatio float64) (Report, error) {
	path := witnessPath
	if path == "" {
		found, ok := FindLatestWitness(filepath.Join(root, WitnessDir))
		if !ok {
			return NoWitnessReport(require), nil
		}
		path = found
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return Report{Schema: Schema, Witness: Rel(path, root), Status: "UNREADABLE", Passed: false, Failures: []string{"cannot read witness: " + err.Error()}, Note: err.Error()}, nil
	}
	var witness map[string]any
	if err := json.Unmarshal(data, &witness); err != nil {
		return Report{Schema: Schema, Witness: Rel(path, root), Status: "UNREADABLE", Passed: false, Failures: []string{"cannot read witness: " + err.Error()}, Note: err.Error()}, nil
	}
	return GradeWitness(witness, Rel(path, root), minRatio), nil
}

func MarshalJSON(report Report) ([]byte, error) {
	return json.MarshalIndent(report, "", "  ")
}

func WriteJSON(path string, report Report) error {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

func WriteMarkdown(path string, report Report) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(RenderMarkdown(report)), 0o644)
}

func corr(verdict string, fak, oracle []int, div *int, reported any, regression bool, note string) Correctness {
	return Correctness{Verdict: verdict, FakIDs: fak, OracleIDs: oracle, FirstDivergenceIndex: div, ReportedDivergenceIndex: reported, KnownDivergenceIndex: KnownDivergenceIndex, Regression: regression, Note: note}
}

func asIntList(v any) ([]int, bool) {
	arr, ok := v.([]any)
	if !ok {
		return nil, false
	}
	out := make([]int, 0, len(arr))
	for _, item := range arr {
		i, ok := asStrictInt(item)
		if !ok {
			return nil, false
		}
		out = append(out, i)
	}
	return out, true
}

func asStrictInt(v any) (int, bool) {
	switch x := v.(type) {
	case bool:
		return 0, false
	case float64:
		if math.Trunc(x) != x {
			return 0, false
		}
		return int(x), true
	case int:
		return x, true
	case json.Number:
		i, err := x.Int64()
		return int(i), err == nil
	default:
		return 0, false
	}
}

func asFloatPtr(v any) *float64 {
	if b, ok := v.(bool); ok && b {
		return nil
	}
	switch x := v.(type) {
	case float64:
		return &x
	case int:
		f := float64(x)
		return &f
	case json.Number:
		if f, err := x.Float64(); err == nil {
			return &f
		}
	}
	return nil
}

func asFloat(v any, fallback float64) float64 {
	if p := asFloatPtr(v); p != nil && *p != 0 {
		return *p
	}
	return fallback
}

func asBool(v any) bool {
	b, _ := v.(bool)
	return b
}

func intPrefixEqual(got, want []int) bool {
	if len(got) < len(want) {
		return false
	}
	for i := range want {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func prefix[T any](items []T, n int) []T {
	if len(items) < n {
		n = len(items)
	}
	return items[:n]
}

func round4(x float64) float64 { return math.Round(x*10_000) / 10_000 }

func passFail(ok bool) string {
	if ok {
		return "PASS"
	}
	return "FAIL"
}

func divString(v *int) string {
	if v == nil {
		return "<nil>"
	}
	return fmt.Sprint(*v)
}

func floatPtrString(v *float64) string {
	if v == nil {
		return "<nil>"
	}
	return fmt.Sprint(*v)
}
