package freshstatus

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	// Schema is the canonical schema identifier for fresh-status payloads.
	Schema = "fak-fresh-status/1"

	// CatalogRel is the repo-relative path to the benchmark catalog.
	CatalogRel = "experiments/benchmark/catalog.json"

	// AuthorityRel is the repo-relative path to benchmark authority doc.
	AuthorityRel = "BENCHMARK-AUTHORITY.md"

	// StaleDays is the benchmark staleness threshold.
	StaleDays = 30

	// DefaultPushLagActionSeconds is the default threshold for unpushed commit age (45m).
	DefaultPushLagActionSeconds = 45 * 60

	// DefaultDirtyLagActionSeconds is the default threshold for uncommitted dirty file age (45m).
	DefaultDirtyLagActionSeconds = 45 * 60
)

// Provenance tags in closed vocabulary.
var Tags = []string{"measured", "modeled", "functional", "unknown"}

var (
	engineMeasured   = []string{"fak-in-kernel", "radixbench", "sessionbench", "decode", "prefill"}
	engineFunctional = []string{"fak model load"}
	engineModeled    = []string{"geometry", "projection", "work floor"}

	throughputFields = []string{"peak_tok_per_sec"}

	functionalTags = map[string]bool{
		"agent-live": true, "parity": true, "subsystem-checks": true, "permission-systems": true,
		"api-host-bridge": true, "safetensors-load-rss": true, "recall": true, "contextq": true,
		"csv": true, "visual-gen": true, "rsi": true,
	}
	modeledTags = map[string]bool{
		"fan-benchmark": true, "fanout": true, "value-sweep": true, "turn-tax": true, "fleet": true,
	}
	measuredTags = map[string]bool{
		"radix-benchmark": true, "model-benchmark": true, "gpu-benchmark": true, "cuda": true,
		"gpu": true, "ada": true, "cpu-q8-parity": true, "decode": true, "cpu-forward": true,
		"engine-fak-cpu": true, "engine-fak-cuda": true, "engine-llama": true, "headtohead": true,
		"kernel": true, "batch": true, "phase0": true,
	}
	weakMeasuredTags = map[string]bool{
		"model-benchmark": true, "gpu-benchmark": true,
	}

	runIDFunctional = []string{
		"agent-live", "parity-", "subsystem-checks", "permission-systems", "api-host",
		"safetensors-load-rss", "recall", "contextq", "visualgen", "surface-smoke", "dogfood",
	}
	runIDModeled = []string{
		"webvoyager-geometry", "ultra-long-context-floor", "projection", "fanbench",
		"fanout", "value-sweep", "turn-tax", "fleet-writeheavy",
	}
	runIDMeasured = []string{
		"cpu-q8-parity", "radixbench-smollm2", "radix-", "smollm2-135m-q8-batch",
		"gpu-qwen2.5", "gcp-g2-l4", "mac-battery",
	}
)

// Payload is the machine-readable fresh-status envelope.
type Payload struct {
	Schema      string                    `json:"schema"`
	OK          bool                      `json:"ok"`
	Verdict     string                    `json:"verdict"`
	Finding     string                    `json:"finding"`
	Reason      string                    `json:"reason"`
	NextAction  string                    `json:"next_action"`
	Workspace   string                    `json:"workspace"`
	Commit      string                    `json:"commit"`
	GeneratedAt string                    `json:"generated_at"`
	Panes       map[string]map[string]any `json:"panes"`
	PaneOrder   []string                  `json:"pane_order"`
}

// GitRunner executes git commands in the given root.
type GitRunner func(root string, args ...string) (string, error)

// ToolRunner runs an external script (e.g. tools/plan_audit.py).
type ToolRunner func(root string, script string, args ...string) (map[string]any, string)

// CollectOptions parameterizes Collect.
type CollectOptions struct {
	Now                   time.Time
	Timeout               time.Duration
	PushLagActionSeconds  int
	DirtyLagActionSeconds int
	GitRunner             GitRunner
	ToolRunner            ToolRunner
}

func hasAny(substrs []string, s string) bool {
	for _, sub := range substrs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

func toInt(v any) (int, bool) {
	switch val := v.(type) {
	case int:
		return val, true
	case int64:
		return int(val), true
	case float64:
		return int(val), true
	case float32:
		return int(val), true
	case string:
		if i, err := strconv.Atoi(strings.TrimSpace(val)); err == nil {
			return i, true
		}
	case *int:
		if val != nil {
			return *val, true
		}
	case *int64:
		if val != nil {
			return int(*val), true
		}
	}
	return 0, false
}

func toFloat(v any) (float64, bool) {
	switch val := v.(type) {
	case float64:
		return val, true
	case float32:
		return float64(val), true
	case int:
		return float64(val), true
	case int64:
		return float64(val), true
	case string:
		if f, err := strconv.ParseFloat(strings.TrimSpace(val), 64); err == nil {
			return f, true
		}
	case *float64:
		if val != nil {
			return *val, true
		}
	}
	return 0, false
}

func formatFloat(v float64) string {
	s := strconv.FormatFloat(v, 'f', -1, 64)
	if !strings.Contains(s, ".") {
		s += ".0"
	}
	return s
}

// ParseRunTS parses a benchmark catalog timestamp (ISO or compact) to UTC time.
func ParseRunTS(ts string) (time.Time, bool) {
	if ts == "" {
		return time.Time{}, false
	}
	raw := strings.TrimSpace(ts)
	formats := []string{
		"20060102T150405Z",
		time.RFC3339,
		"2006-01-02T15:04:05Z",
		time.RFC3339Nano,
		"2006-01-02T15:04:05",
		"2006-01-02 15:04:05",
	}
	for _, fmtStr := range formats {
		if dt, err := time.Parse(fmtStr, raw); err == nil {
			return dt.UTC(), true
		}
	}
	return time.Time{}, false
}

// Classify determines the 4-way provenance tag for a catalog run via the verified priority ladder.
func Classify(run map[string]any, artifactEngines ...[]string) string {
	if pre, ok := run["provenance"].(string); ok {
		for _, t := range Tags {
			if pre == t {
				return pre
			}
		}
	}

	// Rule 1: artifact engine field (highest trust).
	var engines []string
	if len(artifactEngines) > 0 && artifactEngines[0] != nil {
		engines = artifactEngines[0]
	} else if rawEngines, ok := run["artifact_engines"].([]string); ok {
		engines = rawEngines
	} else if rawEnginesAny, ok := run["artifact_engines"].([]any); ok {
		for _, e := range rawEnginesAny {
			if s, ok := e.(string); ok {
				engines = append(engines, s)
			}
		}
	}

	if len(engines) > 0 {
		blob := strings.ToLower(strings.Join(engines, " "))
		if hasAny(engineModeled, blob) {
			return "modeled"
		}
		if hasAny(engineMeasured, blob) {
			return "measured"
		}
		if hasAny(engineFunctional, blob) {
			return "functional"
		}
	}

	// Rule 2: timed throughput field.
	for _, field := range throughputFields {
		if val, ok := run[field]; ok && val != nil {
			if f, ok := toFloat(val); ok && f > 0 {
				return "measured"
			}
		}
	}

	// Rule 3: catalog tags.
	var tags []string
	if rawTags, ok := run["tags"].([]string); ok {
		tags = rawTags
	} else if rawTagsAny, ok := run["tags"].([]any); ok {
		for _, t := range rawTagsAny {
			if s, ok := t.(string); ok {
				tags = append(tags, s)
			}
		}
	}
	tagSet := make(map[string]bool)
	for _, t := range tags {
		tagSet[strings.ToLower(t)] = true
	}

	byTag := ""
	for t := range tagSet {
		if functionalTags[t] {
			byTag = "functional"
			break
		}
	}
	if byTag == "" {
		for t := range tagSet {
			if modeledTags[t] {
				byTag = "modeled"
				break
			}
		}
	}
	if byTag == "" {
		for t := range tagSet {
			if measuredTags[t] && !weakMeasuredTags[t] {
				byTag = "measured"
				break
			}
		}
	}

	// Rule 4: run_id substrings.
	runID, _ := run["run_id"].(string)
	low := strings.ToLower(runID)
	byRunID := ""
	if hasAny(runIDFunctional, low) {
		byRunID = "functional"
	} else if hasAny(runIDModeled, low) {
		byRunID = "modeled"
	} else if hasAny(runIDMeasured, low) {
		byRunID = "measured"
	}

	// Rule 5: resolving tag wins over run_id lift.
	if byTag != "" {
		return byTag
	}
	if byRunID != "" {
		return byRunID
	}

	// Rule 6: fail-closed.
	return "unknown"
}

// ClassifyAll computes provenance counts over a slice of runs.
func ClassifyAll(runs []map[string]any) map[string]int {
	counts := map[string]int{
		"measured":   0,
		"modeled":    0,
		"functional": 0,
		"unknown":    0,
	}
	for _, r := range runs {
		counts[Classify(r)]++
	}
	return counts
}

// SummaryLine renders the one-line provenance readout.
func SummaryLine(counts map[string]int) string {
	return fmt.Sprintf("%d measured / %d modeled / %d functional / %d unknown",
		counts["measured"], counts["modeled"], counts["functional"], counts["unknown"])
}

// SummaryLineFromAny formats summary from map[string]int or map[string]any.
func SummaryLineFromAny(prov any) string {
	counts := map[string]int{"measured": 0, "modeled": 0, "functional": 0, "unknown": 0}
	if m, ok := prov.(map[string]int); ok {
		for k, v := range m {
			counts[k] = v
		}
	} else if m, ok := prov.(map[string]any); ok {
		for k, v := range m {
			if n, ok := toInt(v); ok {
				counts[k] = n
			}
		}
	}
	return SummaryLine(counts)
}

// EnrichCatalogEngines reads local artifact engine fields for runs without mutating the input catalog.
func EnrichCatalogEngines(root string, catalog map[string]any) map[string]any {
	if catalog == nil {
		return nil
	}
	out := make(map[string]any, len(catalog))
	for k, v := range catalog {
		out[k] = v
	}

	var runs []map[string]any
	if rawRuns, ok := catalog["runs"].([]any); ok {
		for _, item := range rawRuns {
			if m, ok := item.(map[string]any); ok {
				runs = append(runs, m)
			}
		}
	} else if typedRuns, ok := catalog["runs"].([]map[string]any); ok {
		runs = typedRuns
	}

	var newRuns []map[string]any
	for _, r := range runs {
		rCopy := make(map[string]any, len(r))
		for k, v := range r {
			rCopy[k] = v
		}
		pathVal, _ := r["path"].(string)
		rel := strings.ReplaceAll(pathVal, "\\", "/")
		if rel != "" {
			runDir := filepath.Join(root, filepath.FromSlash(rel))
			if fi, err := os.Stat(runDir); err == nil && fi.IsDir() {
				entries, err := os.ReadDir(runDir)
				if err == nil {
					var jsonFiles []string
					for _, e := range entries {
						if !e.IsDir() && strings.HasSuffix(e.Name(), ".json") {
							jsonFiles = append(jsonFiles, e.Name())
						}
					}
					sort.Strings(jsonFiles)
					var engines []string
					for _, jf := range jsonFiles {
						b, err := os.ReadFile(filepath.Join(runDir, jf))
						if err != nil {
							continue
						}
						var m map[string]any
						if err := json.Unmarshal(b, &m); err != nil {
							continue
						}
						eng, _ := m["engine"].(string)
						if eng == "" {
							eng, _ = m["generated_by"].(string)
						}
						if eng != "" {
							engines = append(engines, eng)
						}
					}
					if len(engines) > 0 {
						rCopy["artifact_engines"] = engines
					}
				}
			}
		}
		newRuns = append(newRuns, rCopy)
	}
	out["runs"] = newRuns
	return out
}

// LoadCatalog reads and parses experiments/benchmark/catalog.json.
func LoadCatalog(root string) (map[string]any, error) {
	p := filepath.Join(root, filepath.FromSlash(CatalogRel))
	data, err := os.ReadFile(p)
	if err != nil {
		return nil, err
	}
	var cat map[string]any
	if err := json.Unmarshal(data, &cat); err != nil {
		return nil, err
	}
	return cat, nil
}

// FoldBenchmarks folds catalog.json into the benchmark pane.
func FoldBenchmarks(catalog map[string]any, now time.Time, staleDays int) map[string]any {
	emptyProv := func() map[string]int {
		return map[string]int{
			"measured":   0,
			"modeled":    0,
			"functional": 0,
			"unknown":    0,
		}
	}
	if catalog == nil {
		return map[string]any{
			"key":        "benchmarks",
			"label":      "benchmarks",
			"ok":         false,
			"verdict":    "ERROR",
			"reason":     "catalog.json missing or unreadable — no benchmark rollup",
			"runs":       nil,
			"machines":   nil,
			"newest":     nil,
			"age_days":   nil,
			"provenance": emptyProv(),
		}
	}

	var runs []map[string]any
	if rawRuns, ok := catalog["runs"].([]any); ok {
		for _, r := range rawRuns {
			if m, ok := r.(map[string]any); ok {
				runs = append(runs, m)
			}
		}
	} else if typedRuns, ok := catalog["runs"].([]map[string]any); ok {
		runs = typedRuns
	}

	nMachines := 0
	if rawMachines, ok := catalog["machines"].(map[string]any); ok {
		nMachines = len(rawMachines)
	}

	prov := ClassifyAll(runs)
	if len(runs) == 0 {
		return map[string]any{
			"key":        "benchmarks",
			"label":      "benchmarks",
			"ok":         false,
			"verdict":    "ACTION",
			"reason":     fmt.Sprintf("catalog has 0 runs across %d machine(s) — nothing benchmarked", nMachines),
			"runs":       0,
			"machines":   nMachines,
			"newest":     nil,
			"age_days":   nil,
			"stale":      false,
			"provenance": prov,
		}
	}

	var newestDT time.Time
	newestTS := ""
	for _, r := range runs {
		tsStr, _ := r["timestamp"].(string)
		if dt, ok := ParseRunTS(tsStr); ok {
			if newestDT.IsZero() || dt.After(newestDT) {
				newestDT = dt
				newestTS = tsStr
			}
		}
	}

	var ageDays *float64
	stale := false
	if !newestDT.IsZero() {
		days := math.Round((now.Sub(newestDT).Seconds()/86400.0)*10) / 10
		ageDays = &days
		stale = days > float64(staleDays)
	}

	verdict := "OK"
	if stale {
		verdict = "ACTION"
	}
	ok := !stale
	provLine := SummaryLine(prov)

	ageDaysStr := "unknown"
	if ageDays != nil {
		ageDaysStr = formatFloat(*ageDays)
	}
	reason := fmt.Sprintf("%d runs / %d machines; newest %s (%sd ago); %s", len(runs), nMachines, newestTS, ageDaysStr, provLine)
	if stale {
		reason += fmt.Sprintf(" — STALE (>%dd since newest run; catalog may have stopped being fed)", staleDays)
	}

	var newestVal any = newestTS
	if newestTS == "" {
		newestVal = nil
	}

	return map[string]any{
		"key":        "benchmarks",
		"label":      "benchmarks",
		"ok":         ok,
		"verdict":    verdict,
		"reason":     reason,
		"runs":       len(runs),
		"machines":   nMachines,
		"newest":     newestVal,
		"age_days":   ageDays,
		"stale":      stale,
		"provenance": prov,
	}
}

// FoldWork folds plan_audit payload into the work pane.
func FoldWork(planPayload map[string]any, errStr string) map[string]any {
	if errStr != "" || planPayload == nil {
		desc := "plan_audit unavailable"
		if errStr != "" {
			desc = errStr
		}
		return map[string]any{
			"key":         "work",
			"label":       "work",
			"ok":          true,
			"verdict":     "SKIP",
			"reason":      fmt.Sprintf("no plan surface (%s); work pane skipped (not a failure)", desc),
			"total_plans": nil,
			"shipped":     nil,
			"remaining":   nil,
		}
	}

	counts, _ := planPayload["counts"].(map[string]any)
	var totalPlans *int
	var shipped *int
	var remaining *int

	if counts != nil {
		if tp, ok := toInt(counts["total_plans"]); ok {
			totalPlans = &tp
		}
		if s, ok := toInt(counts["shipped"]); ok {
			shipped = &s
		}
		if r, ok := toInt(counts["remaining"]); ok {
			remaining = &r
		}
	}

	if totalPlans == nil || *totalPlans == 0 {
		var tpVal any = 0
		if totalPlans != nil {
			tpVal = *totalPlans
		}
		return map[string]any{
			"key":         "work",
			"label":       "work",
			"ok":          true,
			"verdict":     "SKIP",
			"reason":      "0 phased plans tracked in this clone (valid zero-state)",
			"total_plans": tpVal,
			"shipped":     shipped,
			"remaining":   remaining,
		}
	}

	parts := []string{fmt.Sprintf("%d plans", *totalPlans)}
	if shipped != nil {
		parts = append(parts, fmt.Sprintf("%d shipped", *shipped))
	}
	if remaining != nil {
		parts = append(parts, fmt.Sprintf("%d remaining", *remaining))
	}

	return map[string]any{
		"key":         "work",
		"label":       "work",
		"ok":          true,
		"verdict":     "OK",
		"reason":      strings.Join(parts, ", "),
		"total_plans": *totalPlans,
		"shipped":     shipped,
		"remaining":   remaining,
	}
}

// FoldIndustry folds industry_scorecard payload into the industry pane.
func FoldIndustry(indPayload map[string]any, errStr string) map[string]any {
	if errStr != "" || indPayload == nil {
		desc := "no payload"
		if errStr != "" {
			desc = errStr
		}
		return map[string]any{
			"key":         "industry",
			"label":       "industry",
			"ok":          true,
			"verdict":     "SKIP",
			"reason":      fmt.Sprintf("industry scorecard unavailable (%s); skipped", desc),
			"parity_debt": nil,
			"grade":       nil,
		}
	}

	corpus, _ := indPayload["corpus"].(map[string]any)
	var parityDebt *int
	var grade *string

	if corpus != nil {
		if pd, ok := toInt(corpus["parity_debt"]); ok {
			parityDebt = &pd
		}
		if g, ok := corpus["grade"].(string); ok {
			grade = &g
		}
	}

	if parityDebt == nil {
		return map[string]any{
			"key":         "industry",
			"label":       "industry",
			"ok":          true,
			"verdict":     "SKIP",
			"reason":      "industry scorecard reported no parity_debt; skipped",
			"parity_debt": nil,
			"grade":       grade,
		}
	}

	gradeStr := "?"
	if grade != nil && *grade != "" {
		gradeStr = *grade
	}

	return map[string]any{
		"key":         "industry",
		"label":       "industry",
		"ok":          true,
		"verdict":     "OK",
		"reason":      fmt.Sprintf("parity-debt %d vs SOTA, grade %s", *parityDebt, gradeStr),
		"parity_debt": *parityDebt,
		"grade":       grade,
	}
}

// DefaultGitRunner runs git CLI with a 30s timeout.
func DefaultGitRunner(root string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = root
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	if err := cmd.Run(); err != nil {
		return "", err
	}
	return strings.TrimSpace(stdout.String()), nil
}

// HeadCommit returns the short HEAD commit SHA or "unknown".
func HeadCommit(root string, runner GitRunner) string {
	if runner == nil {
		runner = DefaultGitRunner
	}
	sha, err := runner(root, "rev-parse", "--short", "HEAD")
	if err != nil || sha == "" {
		return "unknown"
	}
	return sha
}

// DirtyPathsFromPorcelain parses git status --porcelain output into repository relative paths.
func DirtyPathsFromPorcelain(porcelain string) []string {
	var paths []string
	for _, raw := range strings.Split(porcelain, "\n") {
		trimmed := strings.TrimRight(raw, "\r\n")
		if strings.TrimSpace(trimmed) == "" || len(trimmed) < 4 {
			continue
		}
		path := strings.TrimSpace(trimmed[3:])
		if idx := strings.Index(path, " -> "); idx != -1 {
			path = strings.TrimSpace(path[idx+4:])
		}
		path = strings.Trim(path, "\"")
		if path != "" {
			paths = append(paths, path)
		}
	}
	return paths
}

// OldestDirtyPath returns the relative path and mtime unix timestamp of the oldest dirty path.
func OldestDirtyPath(root string, paths []string) (*string, *int64) {
	var oldestPath *string
	var oldestTS *int64
	for _, rel := range paths {
		info, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			continue
		}
		ts := info.ModTime().Unix()
		if oldestTS == nil || ts < *oldestTS {
			p := rel
			oldestPath = &p
			oldestTS = &ts
		}
	}
	return oldestPath, oldestTS
}

// GitPaneWithRunner computes the git pane using the provided GitRunner.
func GitPaneWithRunner(root string, runner GitRunner, now time.Time, pushLagActionSeconds, dirtyLagActionSeconds int) map[string]any {
	if runner == nil {
		runner = DefaultGitRunner
	}
	if pushLagActionSeconds <= 0 {
		pushLagActionSeconds = DefaultPushLagActionSeconds
	}
	if dirtyLagActionSeconds <= 0 {
		dirtyLagActionSeconds = DefaultDirtyLagActionSeconds
	}
	if now.IsZero() {
		now = time.Now().UTC()
	}

	sha, _ := runner(root, "rev-parse", "--short", "HEAD")
	branch, _ := runner(root, "rev-parse", "--abbrev-ref", "HEAD")
	if sha == "" {
		return map[string]any{
			"key":                "git",
			"label":              "git",
			"ok":                 false,
			"verdict":            "ERROR",
			"reason":             "git rev-parse HEAD failed — not a repo or git unavailable",
			"sha":                nil,
			"branch":             nil,
			"dirty":              nil,
			"ahead":              nil,
			"behind":             nil,
			"push_lag_seconds":   nil,
			"oldest_unpushed_ts": nil,
			"dirty_lag_seconds":  nil,
			"oldest_dirty_path":  nil,
			"oldest_dirty_mtime": nil,
			"push_lag_stale":     false,
			"dirty_lag_stale":    false,
		}
	}

	porcelain, _ := runner(root, "status", "--porcelain")
	dirtyPaths := DirtyPathsFromPorcelain(porcelain)
	dirty := len(dirtyPaths)
	oldestDirtyPath, oldestDirtyMtime := OldestDirtyPath(root, dirtyPaths)

	var dirtyLagSeconds *int
	if oldestDirtyMtime != nil {
		lag := int(now.Unix() - *oldestDirtyMtime)
		if lag < 0 {
			lag = 0
		}
		dirtyLagSeconds = &lag
	}

	var ahead *int
	var behind *int
	counts, _ := runner(root, "rev-list", "--left-right", "--count", "@{upstream}...HEAD")
	if counts != "" {
		parts := strings.Fields(counts)
		if len(parts) == 2 {
			if b, ok := toInt(parts[0]); ok {
				if a, ok2 := toInt(parts[1]); ok2 {
					behind = &b
					ahead = &a
				}
			}
		}
	}

	var pushLagSeconds *int
	var oldestUnpushedTS *int64
	if ahead != nil && *ahead > 0 {
		cts, _ := runner(root, "log", "--format=%ct", "@{upstream}..HEAD")
		var stamps []int64
		for _, s := range strings.Fields(cts) {
			if ts, err := strconv.ParseInt(s, 10, 64); err == nil {
				stamps = append(stamps, ts)
			}
		}
		if len(stamps) > 0 {
			minTS := stamps[0]
			for _, ts := range stamps[1:] {
				if ts < minTS {
					minTS = ts
				}
			}
			oldestUnpushedTS = &minTS
			lag := int(now.Unix() - minTS)
			if lag < 0 {
				lag = 0
			}
			pushLagSeconds = &lag
		}
	}

	stalePush := pushLagSeconds != nil && *pushLagSeconds > pushLagActionSeconds
	staleDirty := dirtyLagSeconds != nil && *dirtyLagSeconds > dirtyLagActionSeconds

	branchLabel := branch
	if branchLabel == "" {
		branchLabel = "detached"
	}
	bits := []string{fmt.Sprintf("%s (%s)", sha, branchLabel)}

	if dirty > 0 {
		bits = append(bits, fmt.Sprintf("%d dirty", dirty))
		if dirtyLagSeconds != nil {
			mins := *dirtyLagSeconds / 60
			path := "unknown"
			if oldestDirtyPath != nil && *oldestDirtyPath != "" {
				path = *oldestDirtyPath
			}
			if staleDirty {
				bits = append(bits, fmt.Sprintf("oldest dirty %dm old at %s — run fak sweep --json", mins, path))
			} else {
				bits = append(bits, fmt.Sprintf("oldest dirty %dm at %s", mins, path))
			}
		}
	} else {
		bits = append(bits, "clean tree")
	}

	if ahead != nil {
		bits = append(bits, fmt.Sprintf("+%d/-%d vs upstream", *ahead, *behind))
	}

	if pushLagSeconds != nil {
		mins := *pushLagSeconds / 60
		if stalePush {
			bits = append(bits, fmt.Sprintf("%d unpushed, oldest %dm old — push to origin", *ahead, mins))
		} else {
			bits = append(bits, fmt.Sprintf("%d unpushed, oldest %dm", *ahead, mins))
		}
	}

	verdict := "OK"
	if stalePush || staleDirty {
		verdict = "ACTION"
	}
	ok := !(stalePush || staleDirty)

	var branchVal any = branch
	if branch == "" {
		branchVal = nil
	}

	return map[string]any{
		"key":                "git",
		"label":              "git",
		"ok":                 ok,
		"verdict":            verdict,
		"reason":             strings.Join(bits, ", "),
		"sha":                sha,
		"branch":             branchVal,
		"dirty":              dirty,
		"ahead":              ahead,
		"behind":             behind,
		"push_lag_seconds":   pushLagSeconds,
		"oldest_unpushed_ts": oldestUnpushedTS,
		"dirty_lag_seconds":  dirtyLagSeconds,
		"oldest_dirty_path":  oldestDirtyPath,
		"oldest_dirty_mtime": oldestDirtyMtime,
		"push_lag_stale":     stalePush,
		"dirty_lag_stale":    staleDirty,
	}
}

// GitPane computes the git pane using DefaultGitRunner.
func GitPane(root string, now time.Time, pushLagActionSeconds, dirtyLagActionSeconds int) map[string]any {
	return GitPaneWithRunner(root, DefaultGitRunner, now, pushLagActionSeconds, dirtyLagActionSeconds)
}

// Fold aggregates the domain panes into one rollup payload.
func Fold(panes []map[string]any, workspace, commit, generatedAt string) Payload {
	var actionable []map[string]any
	var skipped []map[string]any
	var live []map[string]any

	panesMap := make(map[string]map[string]any)
	var paneOrder []string

	for _, p := range panes {
		key, _ := p["key"].(string)
		panesMap[key] = p
		paneOrder = append(paneOrder, key)

		verdict, _ := p["verdict"].(string)
		if verdict == "ERROR" || verdict == "ACTION" {
			actionable = append(actionable, p)
		} else if verdict == "SKIP" {
			skipped = append(skipped, p)
		} else if verdict == "OK" {
			live = append(live, p)
		}
	}

	var ok bool
	var verdict, finding, reason, nextAction string

	if len(actionable) > 0 {
		ok = false
		verdict = "ACTION"
		finding = "needs_attention"

		var parts []string
		for _, p := range actionable {
			label, _ := p["label"].(string)
			r, _ := p["reason"].(string)
			parts = append(parts, fmt.Sprintf("%s: %s", label, r))
		}
		reason = strings.Join(parts, "; ")

		first := actionable[0]
		firstKey, _ := first["key"].(string)
		firstVerdict, _ := first["verdict"].(string)
		pushLagStale, _ := first["push_lag_stale"].(bool)
		dirtyLagStale, _ := first["dirty_lag_stale"].(bool)
		benchStale, _ := first["stale"].(bool)
		firstReason, _ := first["reason"].(string)
		firstLabel, _ := first["label"].(string)

		if firstKey == "git" && firstVerdict == "ERROR" {
			nextAction = "fix the git context (not a repo / git unavailable) before trusting any other pane"
		} else if firstKey == "git" && pushLagStale {
			mins := 0
			if s, ok := toInt(first["push_lag_seconds"]); ok {
				mins = s / 60
			}
			ahead := 0
			if a, ok := toInt(first["ahead"]); ok {
				ahead = a
			}
			nextAction = fmt.Sprintf("push to origin — %d commit(s) have been unpushed for %dm; committed work is not reaching the remote", ahead, mins)
		} else if firstKey == "git" && dirtyLagStale {
			mins := 0
			if s, ok := toInt(first["dirty_lag_seconds"]); ok {
				mins = s / 60
			}
			dirty := 0
			if d, ok := toInt(first["dirty"]); ok {
				dirty = d
			}
			path := "unknown path"
			if p, ok := first["oldest_dirty_path"].(string); ok && p != "" {
				path = p
			} else if pPtr, ok := first["oldest_dirty_path"].(*string); ok && pPtr != nil && *pPtr != "" {
				path = *pPtr
			}
			nextAction = fmt.Sprintf("run `fak sweep --json` — %d dirty path(s), oldest %dm old at %s; inspect by lane before committing", dirty, mins, path)
		} else if firstKey == "git" {
			nextAction = fmt.Sprintf("resolve the git pane: %s", firstReason)
		} else if firstKey == "benchmarks" && benchStale {
			nextAction = "refresh the benchmark catalog — run the relevant bench + `python tools/bench_catalog.py build`"
		} else if firstKey == "benchmarks" {
			nextAction = "populate experiments/benchmark/catalog.json (no runs registered)"
		} else {
			nextAction = fmt.Sprintf("resolve the %s pane: %s", firstLabel, firstReason)
		}
	} else {
		ok = true
		verdict = "OK"
		finding = "all_green"

		var parts []string
		for _, p := range live {
			label, _ := p["label"].(string)
			r, _ := p["reason"].(string)
			parts = append(parts, fmt.Sprintf("%s: %s", label, r))
		}
		reason = strings.Join(parts, "; ")
		if len(skipped) > 0 {
			var skippedLabels []string
			for _, p := range skipped {
				label, _ := p["label"].(string)
				skippedLabels = append(skippedLabels, label)
			}
			reason += fmt.Sprintf(" (%d pane(s) skipped: %s)", len(skipped), strings.Join(skippedLabels, ", "))
		}
		nextAction = "rollup is green; nothing required"
	}

	return Payload{
		Schema:      Schema,
		OK:          ok,
		Verdict:     verdict,
		Finding:     finding,
		Reason:      reason,
		NextAction:  nextAction,
		Workspace:   workspace,
		Commit:      commit,
		GeneratedAt: generatedAt,
		Panes:       panesMap,
		PaneOrder:   paneOrder,
	}
}

// DefaultToolRunner executes tools/<script> using python.
func DefaultToolRunner(timeout time.Duration) ToolRunner {
	return func(root, script string, args ...string) (map[string]any, string) {
		scriptPath := filepath.Join(root, "tools", script)
		if _, err := os.Stat(scriptPath); err != nil {
			return nil, fmt.Sprintf("missing tool: tools/%s", script)
		}
		py := "python"
		if _, err := exec.LookPath("python3"); err == nil {
			py = "python3"
		}
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		cmdArgs := append([]string{scriptPath}, args...)
		cmd := exec.CommandContext(ctx, py, cmdArgs...)
		cmd.Dir = root
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		err := cmd.Run()
		if ctx.Err() == context.DeadlineExceeded {
			return nil, fmt.Sprintf("timed out after %ds", int(timeout.Seconds()))
		}
		if err != nil && stdout.Len() == 0 {
			return nil, err.Error()
		}
		var payload map[string]any
		if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
			tail := ""
			lines := strings.Split(strings.TrimSpace(stderr.String()+"\n"+stdout.String()), "\n")
			if len(lines) > 0 && lines[len(lines)-1] != "" {
				tail = lines[len(lines)-1]
			}
			if len(tail) > 160 {
				tail = tail[:160]
			}
			exitCode := 1
			var exitErr *exec.ExitError
			if errors.As(err, &exitErr) {
				exitCode = exitErr.ExitCode()
			}
			return nil, fmt.Sprintf("non-JSON output (exit %d): %s", exitCode, tail)
		}
		return payload, ""
	}
}

// Collect gathers all four domain panes from the live tree.
func Collect(root string, opts CollectOptions) []map[string]any {
	now := opts.Now
	if now.IsZero() {
		now = time.Now().UTC()
	}
	pushLag := opts.PushLagActionSeconds
	if pushLag <= 0 {
		pushLag = DefaultPushLagActionSeconds
	}
	dirtyLag := opts.DirtyLagActionSeconds
	if dirtyLag <= 0 {
		dirtyLag = DefaultDirtyLagActionSeconds
	}
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}

	git := GitPaneWithRunner(root, opts.GitRunner, now, pushLag, dirtyLag)

	cat, _ := LoadCatalog(root)
	bench := FoldBenchmarks(EnrichCatalogEngines(root, cat), now, StaleDays)

	toolRun := opts.ToolRunner
	if toolRun == nil {
		toolRun = DefaultToolRunner(timeout)
	}

	planPayload, planErr := toolRun(root, "plan_audit.py", "--json")
	work := FoldWork(planPayload, planErr)

	indPayload, indErr := toolRun(root, "industry_scorecard.py", "--json")
	ind := FoldIndustry(indPayload, indErr)

	return []map[string]any{git, bench, work, ind}
}

// Render formats a Payload into the CLI human rollup text.
func Render(payload Payload) string {
	lines := []string{
		fmt.Sprintf("fresh status — %s (%s)  @%s", payload.Verdict, payload.Finding, payload.Commit),
		fmt.Sprintf("  generated %s", payload.GeneratedAt),
		"",
	}
	for _, key := range payload.PaneOrder {
		p := payload.Panes[key]
		verdict, _ := p["verdict"].(string)
		mark := "?"
		switch verdict {
		case "OK":
			mark = "✓"
		case "SKIP":
			mark = "·"
		case "ACTION", "ERROR":
			mark = "✗"
		case "WARN":
			mark = "!"
		}
		label, _ := p["label"].(string)
		reason, _ := p["reason"].(string)
		lines = append(lines, fmt.Sprintf("  %s %-11s %s", mark, label, reason))
	}
	lines = append(lines, "", fmt.Sprintf("  → %s", payload.NextAction))
	return strings.Join(lines, "\n")
}

// RenderDoc formats a Payload into Markdown snapshot documentation.
func RenderDoc(payload Payload, date string) string {
	g := payload.Panes["git"]
	b := payload.Panes["benchmarks"]

	gitLine := fmt.Sprintf("- **git:** HEAD `%v` on `%v`, %v dirty file(s)", g["sha"], g["branch"], g["dirty"])
	if ahead, ok := toInt(g["ahead"]); ok {
		behind, _ := toInt(g["behind"])
		gitLine += fmt.Sprintf(", +%d/-%d vs upstream", ahead, behind)
	}

	ageDaysStr := ""
	if ad, ok := toFloat(b["age_days"]); ok {
		ageDaysStr = formatFloat(ad)
	}
	provLine := SummaryLineFromAny(b["provenance"])
	bLine := fmt.Sprintf("- **benchmarks:** %v runs / %v machines; newest %v (%sd ago); provenance %s",
		b["runs"], b["machines"], b["newest"], ageDaysStr, provLine)

	var rows []string
	for _, key := range payload.PaneOrder {
		p := payload.Panes[key]
		rows = append(rows, fmt.Sprintf("| %v | %v | %v |", p["label"], p["verdict"], p["reason"]))
	}

	lines := []string{
		fmt.Sprintf("# Fresh status snapshot (%s)", date),
		"",
		"> Regenerated by `python tools/fresh_status.py --write-doc`. This is a",
		"> committed front door for the cross-domain rollup that folds git +",
		"> benchmarks + work + industry into one control-pane payload. Re-run the",
		"> tool for the live state; this note is the last pinned snapshot.",
		"",
		fmt.Sprintf("**Overall:** %s (%s) — %s", payload.Verdict, payload.Finding, payload.NextAction),
		"",
		gitLine,
		bLine,
		"",
		"## Panes",
		"",
		"| Pane | Verdict | Detail |",
		"|---|---|---|",
	}
	lines = append(lines, rows...)
	lines = append(lines,
		"",
		"_Provenance discipline (`tools/bench_provenance.py`, authority-grounded + adversarially verified): **measured** = a real wall-clock; **modeled** = a closed-form work floor; **functional** = a correctness / agent-live / load-only witness that is NOT a throughput number; **unknown** = the fail-closed residue, surfaced loudly and never silently counted as measured — the same honesty floor `tools/check_provenance_labels.py` enforces._",
		"",
	)
	return strings.Join(lines, "\n")
}

// RepoRoot discovers the workspace root starting from start or cwd.
func RepoRoot(start string) string {
	if start != "" {
		abs, err := filepath.Abs(start)
		if err == nil {
			return abs
		}
		return start
	}
	cwd, err := os.Getwd()
	if err != nil {
		return "."
	}
	dir := cwd
	for {
		if fi, err := os.Stat(filepath.Join(dir, ".git")); err == nil && (fi.IsDir() || fi.Mode().IsRegular()) {
			return dir
		}
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return cwd
}

// CollectFn is the collect function used by Run (can be swapped in tests).
var CollectFn = Collect

// Run executes the fresh-status CLI with the provided stdout, stderr, and arguments.
func Run(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("fresh-status", flag.ContinueOnError)
	fs.SetOutput(stderr)

	jsonOut := fs.Bool("json", false, "emit machine-readable JSON")
	check := fs.Bool("check", false, "advisory gate: exit non-zero only on an ACTION pane")
	writeDoc := fs.Bool("write-doc", false, "regenerate the committed snapshot note under docs/notes/")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	date := fs.String("date", "", "snapshot date YYYY-MM-DD for --write-doc (default: today UTC)")
	timeout := fs.Int("timeout", 60, "per-sub-tool timeout seconds")
	pushLagMins := fs.Int("push-lag-mins", DefaultPushLagActionSeconds/60, "trip the git pane to ACTION when the oldest unpushed commit is older than this many minutes")
	dirtyLagMins := fs.Int("dirty-lag-mins", DefaultDirtyLagActionSeconds/60, "trip the git pane to ACTION when the oldest dirty path mtime is older than this many minutes")

	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() > 0 {
		fmt.Fprintf(stderr, "fresh-status: unexpected arguments: %v\n", fs.Args())
		return 2
	}

	root := RepoRoot(*workspace)
	now := time.Now().UTC()

	panes := CollectFn(root, CollectOptions{
		Now:                   now,
		Timeout:               time.Duration(*timeout) * time.Second,
		PushLagActionSeconds:  *pushLagMins * 60,
		DirtyLagActionSeconds: *dirtyLagMins * 60,
	})

	commit := HeadCommit(root, nil)
	payload := Fold(panes, root, commit, now.Format(time.RFC3339))

	if *writeDoc {
		snapDate := *date
		if snapDate == "" {
			snapDate = now.Format("2006-01-02")
		}
		docPath := filepath.Join(root, "docs", "notes", fmt.Sprintf("FRESH-STATUS-%s.md", snapDate))
		if err := os.MkdirAll(filepath.Dir(docPath), 0755); err != nil {
			fmt.Fprintf(stderr, "failed to create directory for doc: %v\n", err)
			return 2
		}
		docContent := RenderDoc(payload, snapDate)
		if err := os.WriteFile(docPath, []byte(docContent), 0644); err != nil {
			fmt.Fprintf(stderr, "failed to write snapshot: %v\n", err)
			return 2
		}
		if !*jsonOut {
			rel, _ := filepath.Rel(root, docPath)
			fmt.Fprintf(stdout, "wrote snapshot -> %s\n", filepath.ToSlash(rel))
		}
	}

	if *check {
		if *jsonOut {
			b, _ := json.MarshalIndent(payload, "", "  ")
			fmt.Fprintln(stdout, string(b))
		} else {
			fmt.Fprintln(stdout, Render(payload))
		}
		if payload.Verdict == "ACTION" {
			return 1
		}
		return 0
	}

	if *jsonOut {
		b, _ := json.MarshalIndent(payload, "", "  ")
		fmt.Fprintln(stdout, string(b))
	} else if !*writeDoc {
		fmt.Fprintln(stdout, Render(payload))
	}

	if payload.OK {
		return 0
	}
	return 1
}
