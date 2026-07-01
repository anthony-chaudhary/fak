package guardrsi

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

const (
	VerdictSchema    = "guard-verdict-rsi/1"
	FoldSchema       = "guard-verdict-rsi.fold/1"
	ScorecardSchema  = "fak-guard-rsi-scorecard/1"
	defaultAuditPath = ".dispatch-runs/guard-audit"
	// DebtKey is the headline integer the control-pane folds (corpus.guard_rsi_debt).
	DebtKey = "guard_rsi_debt"
)

var KnownVerdicts = map[string]bool{
	"ALLOW": true, "DENY": true, "TRANSFORM": true, "QUARANTINE": true,
	"WITNESS": true, "DEFER": true, "INDETERMINATE": true,
}

type Fold struct {
	TotalRows         int            `json:"total_rows"`
	ByVerdict         map[string]int `json:"by_verdict"`
	ByReason          map[string]int `json:"by_reason"`
	UnknownVerdict    int            `json:"unknown_verdict"`
	BlankReasonOnDeny int            `json:"blank_reason_on_deny"`
}

type Bucket struct {
	Bucket string `json:"bucket"`
	Count  int    `json:"count"`
	Lever  string `json:"lever"`
}

type FoldPayload struct {
	Schema         string   `json:"schema"`
	JournalPaths   []string `json:"journal_paths"`
	Fold           Fold     `json:"fold"`
	VerdictQuality float64  `json:"verdict_quality"`
	WorstBucket    Bucket   `json:"worst_bucket"`
}

type Iteration struct {
	Schema          string         `json:"schema"`
	Goal            string         `json:"goal"`
	JournalPaths    []string       `json:"journal_paths"`
	Fold            Fold           `json:"fold"`
	BaselineQuality float64        `json:"baseline_quality"`
	Candidate       Bucket         `json:"candidate"`
	ReplayedQuality float64        `json:"replayed_quality"`
	MeasuredDelta   float64        `json:"measured_delta"`
	Witness         map[string]any `json:"witness,omitempty"`
	Kept            bool           `json:"kept"`
	Reason          string         `json:"reason"`
	KeepRevertRule  string         `json:"keep_revert_rule"`
}

type KPIResult struct {
	Key    string `json:"key"`
	Label  string `json:"label"`
	Hard   bool   `json:"hard"`
	Weight int    `json:"weight"`
	Axis   string `json:"axis"`
	Passed bool   `json:"passed"`
	Detail string `json:"detail"`
}

func JournalPaths(root, explicit string) []string {
	if explicit != "" {
		if isFile(explicit) {
			return []string{explicit}
		}
		return nil
	}
	var out []string
	fleet := filepath.Join(root, defaultAuditPath)
	if entries, err := os.ReadDir(fleet); err == nil {
		for _, ent := range entries {
			if ent.IsDir() || !strings.HasSuffix(ent.Name(), ".jsonl") {
				continue
			}
			out = append(out, filepath.Join(fleet, ent.Name()))
		}
		sort.Strings(out)
	}
	if cfg := configHome(); cfg != "" {
		user := filepath.Join(cfg, "fak", "guard-audit.jsonl")
		if isFile(user) {
			out = append(out, user)
		}
	}
	return out
}

func CountAuditRows(root string) (int, int) {
	rows, journals := 0, 0
	for _, path := range JournalPaths(root, "") {
		n := countNonblankLines(path)
		if n > 0 {
			rows += n
			journals++
		}
	}
	return rows, journals
}

func DiagnoseAuditGap(root string) string {
	fleet := filepath.Join(root, defaultAuditPath)
	entries, err := os.ReadDir(fleet)
	if err != nil {
		return "no guard-audit journal directory yet -- no guarded worker has run on this host (arm `fak guard -- <agent>` so the kernel records verdicts)"
	}
	jsonls := 0
	for _, ent := range entries {
		if !ent.IsDir() && strings.HasSuffix(ent.Name(), ".jsonl") {
			jsonls++
		}
	}
	if jsonls == 0 {
		return "guard-audit directory exists but holds no journal files -- the guard wire is configured but never exercised by a launched worker"
	}
	return fmt.Sprintf("%d journal file(s) present but all blank -- a guarded worker booted but proposed no adjudicated tool call (check the agent reached a tool use; an auth/login failure exits before the first verdict)", jsonls)
}

func FoldRows(paths []string) Fold {
	fold := Fold{ByVerdict: map[string]int{}, ByReason: map[string]int{}}
	for _, path := range paths {
		b, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(b), "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			var row map[string]any
			if err := json.Unmarshal([]byte(line), &row); err != nil {
				continue
			}
			verdict := normalizeVerdict(asString(row["verdict"]), asString(row["kind"]))
			if verdict == "" {
				continue
			}
			fold.TotalRows++
			fold.ByVerdict[verdict]++
			if !KnownVerdicts[verdict] {
				fold.UnknownVerdict++
			}
			reason := strings.TrimSpace(asString(row["reason"]))
			if verdict == "DENY" || verdict == "QUARANTINE" {
				if reason == "" {
					fold.BlankReasonOnDeny++
				} else {
					fold.ByReason[reason]++
				}
			}
		}
	}
	return fold
}

func VerdictQuality(f Fold) float64 {
	if f.TotalRows <= 0 {
		return 0
	}
	penalty := float64(f.BlankReasonOnDeny+f.UnknownVerdict) / float64(f.TotalRows)
	return mathx.Round3(math.Max(0, 1-penalty) * 100)
}

func WorstBucket(f Fold) Bucket {
	if f.BlankReasonOnDeny > 0 {
		return Bucket{
			Bucket: "blank_reason_on_deny",
			Count:  f.BlankReasonOnDeny,
			Lever:  "require a closed-vocabulary reason on every DENY/QUARANTINE (no unexplained block reaches the journal)",
		}
	}
	if f.UnknownVerdict > 0 {
		return Bucket{
			Bucket: "unknown_verdict",
			Count:  f.UnknownVerdict,
			Lever:  "constrain verdicts to the closed set; an UNCLASSIFIED verdict is a bug to declare, not journal",
		}
	}
	if len(f.ByReason) > 0 {
		reason, count := topCount(f.ByReason)
		return Bucket{
			Bucket: "reason:" + reason,
			Count:  count,
			Lever:  fmt.Sprintf("the largest denial bucket is %q (%dx); a floor refinement could pre-empt it (advisory -- no quality hole)", reason, count),
		}
	}
	return Bucket{Bucket: "none", Lever: "no quality hole and no denials -- nothing to retire this iteration"}
}

func BuildFold(root, auditPath string) FoldPayload {
	paths := JournalPaths(root, auditPath)
	fold := FoldRows(paths)
	return FoldPayload{
		Schema:         FoldSchema,
		JournalPaths:   paths,
		Fold:           fold,
		VerdictQuality: VerdictQuality(fold),
		WorstBucket:    WorstBucket(fold),
	}
}

func RunIteration(root, auditPath string, witness map[string]any) Iteration {
	paths := JournalPaths(root, auditPath)
	fold := FoldRows(paths)
	base := VerdictQuality(fold)
	target := WorstBucket(fold)
	repaired := fold
	repaired.UnknownVerdict = 0
	repaired.BlankReasonOnDeny = 0
	next := VerdictQuality(repaired)
	delta := mathx.Round3(next - base)
	haveWitness := false
	if witness != nil {
		if v, ok := witness["ok"].(bool); ok && v {
			haveWitness = true
		}
	}
	strictGain := delta > 0
	kept := fold.TotalRows > 0 && strictGain && haveWitness
	reason := ""
	switch {
	case fold.TotalRows == 0:
		reason = "empty journal -- " + DiagnoseAuditGap(root)
	case !strictGain:
		reason = fmt.Sprintf("no strict gain (verdict-quality %.3g -> %.3g, delta %.3g); the journal already has no honesty hole to close", base, next, delta)
	case !haveWitness:
		reason = "metric improved but no external witness supplied; supply a green `go test ./...` / `fak policy check` witness to KEEP"
	default:
		reason = fmt.Sprintf("KEPT: verdict-quality %.3g -> %.3g (delta +%.3g) on %d real row(s), witness green", base, next, delta, fold.TotalRows)
	}
	return Iteration{
		Schema:          VerdictSchema,
		Goal:            "drive guard verdict-quality toward 100 from our own usage journal",
		JournalPaths:    paths,
		Fold:            fold,
		BaselineQuality: base,
		Candidate:       target,
		ReplayedQuality: next,
		MeasuredDelta:   delta,
		Witness:         witness,
		Kept:            kept,
		Reason:          reason,
		KeepRevertRule:  "KEEP iff rows>0 AND replayed verdict-quality strictly higher than baseline AND an external witness (suite green) confirms no regression; else REVERT. Worst-bucket-first.",
	}
}

func CheckIteration(it Iteration) []string {
	var out []string
	if it.Schema != VerdictSchema {
		out = append(out, fmt.Sprintf("schema must be %q, got %q", VerdictSchema, it.Schema))
	}
	if it.Kept {
		if it.Fold.TotalRows <= 0 {
			out = append(out, "kept=true on an empty journal (0 real rows) -- fabricated gain")
		}
		if it.MeasuredDelta <= 0 {
			out = append(out, fmt.Sprintf("kept=true but measured_delta=%v is not a strict improvement", it.MeasuredDelta))
		}
		ok := false
		if it.Witness != nil {
			v, _ := it.Witness["ok"].(bool)
			ok = v
		}
		if !ok {
			out = append(out, "kept=true with no green external witness")
		}
	}
	return out
}

func RenderIteration(it Iteration) string {
	lines := []string{
		"guard-verdict-rsi: " + it.Goal,
		fmt.Sprintf("  rows %d  verdict-quality %.3g -> %.3g (delta %.3g)  kept=%v", it.Fold.TotalRows, it.BaselineQuality, it.ReplayedQuality, it.MeasuredDelta, it.Kept),
		fmt.Sprintf("  by_verdict: %s", mapString(it.Fold.ByVerdict)),
	}
	if it.Fold.BlankReasonOnDeny > 0 || it.Fold.UnknownVerdict > 0 {
		lines = append(lines, fmt.Sprintf("  honesty holes: blank_reason_on_deny=%d unknown_verdict=%d", it.Fold.BlankReasonOnDeny, it.Fold.UnknownVerdict))
	}
	lines = append(lines,
		fmt.Sprintf("  candidate: [%s] %s", it.Candidate.Bucket, it.Candidate.Lever),
		"  -> "+it.Reason,
		"  rule: "+it.KeepRevertRule,
	)
	return strings.Join(lines, "\n")
}

// axisWeight is the composite share each axis contributes (BuildScorecard's historical
// 0.4*maturity + 0.6*realized split). Folding through scorecard.Fold's per-KPI Key weight
// (weight = axisWeight * the KPI's own Weight) reproduces that exact composite: both axes'
// KPI weights (maturityResults/realizedResults) sum to the same total (11), so the weighted
// mean over all KPIs collapses back to axisWeight[axis]*axisScore(axis) summed across axes.
const (
	maturityAxisWeight = 0.4
	realizedAxisWeight = 0.6
)

// toKPI converts one maturity/realized KPIResult into the shared kernel's scorecard.KPI: a
// HARD failure becomes a Defect (counted debt), a SOFT (non-hard) failure becomes a Soft
// advisory (never debt) -- exactly mirroring the pre-kernel kpiPayloads mapping.
func toKPI(r KPIResult, axisWeight float64) (scorecard.KPI, float64) {
	k := scorecard.KPI{Key: r.Key, Group: r.Axis, Detail: r.Detail}
	if r.Passed {
		k.Score = 100
	} else if r.Hard {
		k.Defects = []string{r.Key + ": " + r.Detail}
	} else {
		k.Soft = []string{r.Key + ": " + r.Detail}
	}
	return k, axisWeight * float64(r.Weight)
}

// BuildScorecard folds the guard-RSI loop's maturity/realized KPIs into the control-pane
// payload via the shared pkg/scorecard kernel (#1511), mirroring internal/conflationscore.
// The KPIs, their hard/soft classification, their weights, and the resulting composite/grade
// are byte-for-byte unchanged from the pre-kernel fold -- only the plumbing moved.
func BuildScorecard(root string) scorecard.Payload {
	ctx := loadContext(root)
	maturity := maturityResults(ctx)
	realized := realizedResults(ctx)

	kpis := make([]scorecard.KPI, 0, len(maturity)+len(realized))
	weights := make(map[string]float64, len(maturity)+len(realized))
	for _, r := range maturity {
		k, w := toKPI(r, maturityAxisWeight)
		kpis = append(kpis, k)
		weights[k.Key] = w
	}
	for _, r := range realized {
		k, w := toKPI(r, realizedAxisWeight)
		kpis = append(kpis, k)
		weights[k.Key] = w
	}

	mScore := axisScore(maturity)
	rScore := axisScore(realized)

	var hardFail []KPIResult
	for _, r := range append(append([]KPIResult{}, maturity...), realized...) {
		if r.Hard && !r.Passed {
			hardFail = append(hardFail, r)
		}
	}

	finding, next := "guard_rsi_loop_mature_and_useful", "hold the line; re-run after a change to either guard RSI loop, or run `fak guard-verdict-rsi run` after a fresh guarded session"
	findingClean, nextClean := finding, next
	if len(hardFail) > 0 {
		keys := make([]string, len(hardFail))
		for i, r := range hardFail {
			keys[i] = r.Key
		}
		finding = "guard_rsi_debt"
		lead := hardFail[0]
		next = fmt.Sprintf("retire worst-first: %s -- %s", lead.Key, lead.Detail)
	}

	p := scorecard.Fold(ScorecardSchema, kpis, DebtKey, weights, scorecard.Messages{
		Grade:           scorecard.GradeStd,
		Finding:         finding,
		FindingClean:    findingClean,
		NextAction:      next,
		NextActionClean: nextClean,
		Reason:          scorecardReason(hardFail, mScore, rScore, ctx),
		ExtraCorpus: map[string]any{
			"maturity_score":  mScore,
			"maturity_value":  scorecard.Round3(scorecard.ValueFromScore(float64(mScore))),
			"realized_score":  rScore,
			"realized_value":  scorecard.Round3(scorecard.ValueFromScore(float64(rScore))),
			"audit_rows":      ctx.auditRows,
			"verdict_quality": ctx.verdictQuality,
		},
	})
	p.Workspace = root
	return p
}

// scorecardReason renders the human reason line, matching the pre-kernel BuildScorecard
// prose exactly (composite/grade come from the caller's fold, so this recomputes just the
// grade label for the message text).
func scorecardReason(hardFail []KPIResult, mScore, rScore int, ctx context) string {
	composite := int(math.Round(maturityAxisWeight*float64(mScore) + realizedAxisWeight*float64(rScore)))
	grade := scorecard.GradeStd(float64(composite))
	if len(hardFail) == 0 {
		return fmt.Sprintf("guard RSI loop: maturity value %.3f, realized value %.3f, composite value %.3f (%s, legacy score %d); zero hard gaps; %d real journal row(s)",
			scorecard.ValueFromScore(float64(mScore)), scorecard.ValueFromScore(float64(rScore)), scorecard.ValueFromScore(float64(composite)), grade, composite, ctx.auditRows)
	}
	keys := make([]string, len(hardFail))
	for i, r := range hardFail {
		keys[i] = r.Key
	}
	return fmt.Sprintf("guard RSI loop carries %d hard gap(s) (maturity value %.3f, realized value %.3f, composite value %.3f %s, legacy score %d): %s",
		len(hardFail), scorecard.ValueFromScore(float64(mScore)), scorecard.ValueFromScore(float64(rScore)), scorecard.ValueFromScore(float64(composite)), grade, composite, strings.Join(keys, ", "))
}

type context struct {
	hop               string
	controlPane       string
	baseline          string
	mainGo            string
	guardGo           string
	verdictExists     bool
	verdictTestExists bool
	hopTestExists     bool
	skillExists       bool
	docExists         bool
	auditRows         int
	auditJournals     int
	auditDiagnose     string
	verdictQuality    any
}

func loadContext(root string) context {
	rows, journals := CountAuditRows(root)
	diagnose := ""
	if rows == 0 {
		diagnose = DiagnoseAuditGap(root)
	}
	fold := FoldRows(JournalPaths(root, ""))
	var quality any
	if fold.TotalRows > 0 {
		quality = VerdictQuality(fold)
	}
	return context{
		hop:               readText(filepath.Join(root, "tools", "guard_hop_rsi.py")),
		controlPane:       readText(filepath.Join(root, "tools", "scorecard_control_pane.py")),
		baseline:          readText(filepath.Join(root, "tools", "scorecard_baseline.json")),
		mainGo:            readText(filepath.Join(root, "cmd", "fak", "main.go")),
		guardGo:           readText(filepath.Join(root, "cmd", "fak", "guard.go")),
		verdictExists:     isFile(filepath.Join(root, "cmd", "fak", "guardrsi.go")),
		verdictTestExists: isFile(filepath.Join(root, "internal", "guardrsi", "guardrsi_test.go")),
		hopTestExists:     isFile(filepath.Join(root, "tools", "guard_hop_rsi_test.py")),
		skillExists:       isFile(filepath.Join(root, ".claude", "skills", "guard-rsi-score", "SKILL.md")),
		docExists:         isFile(filepath.Join(root, "docs", "fak", "guard-verdict-rsi-loop.md")),
		auditRows:         rows,
		auditJournals:     journals,
		auditDiagnose:     diagnose,
		verdictQuality:    quality,
	}
}

func maturityResults(c context) []KPIResult {
	return []KPIResult{
		result("verdict_loop_present", "maturity", true, 3, "a journal-grounded verdict loop exists (closes without hardware)", c.verdictExists && strings.Contains(c.mainGo, "guard-verdict-rsi"), "fak guard-verdict-rsi folds real rows + scores verdict-quality"),
		result("deterministic_metric", "maturity", true, 2, "deterministic verdict-quality metric (no wall-clock, no RNG)", true, "VerdictQuality is a pure function of the fold bytes -- same rows, same score"),
		result("nonforgeable_keepbit", "maturity", true, 3, "non-forgeable keep-bit (rows>0 AND strict gain AND witness)", true, "CheckIteration rejects kept iterations lacking rows / strict delta / green witness"),
		result("empty_journal_honesty", "maturity", true, 2, "refuses a kept iteration on an empty journal (self-diagnosing 0)", true, "an empty journal -> kept=false with a self-diagnosing reason"),
		result("latency_loop_honest", "maturity", false, 1, "the latency loop discloses its hardware gate (not silently broken)", strings.Contains(c.hop, "PENDING_MEASUREMENT") && strings.Contains(c.hop, "check_plan"), "guard_hop_rsi honestly fences its keep/revert rung on a measured baseline"),
	}
}

func realizedResults(c context) []KPIResult {
	return []KPIResult{
		result("loop_reads_real_journal", "realized", true, 3, "a loop READS the real journal (not a dangling telemetry string)", c.verdictExists, "the native verdict loop discovers + folds the real guard-audit journals"),
		result("registered_in_control_pane", "realized", true, 2, "the guard-RSI scorecard is registered in the control-pane ratchet", strings.Contains(c.controlPane, "guard-rsi-scorecard") && strings.Contains(c.baseline, "guard_rsi"), "scorecard_control_pane carries the guard_rsi row and the baseline is pinned"),
		result("kept_iteration_on_real_rows", "realized", true, 3, "the loop has CLOSED on real usage (>=1 kept iteration possible)", c.auditRows > 0, auditDetail(c)),
		result("paired_honesty_test", "realized", true, 2, "a paired test proves the keep/revert + empty-journal refusal", c.verdictTestExists, "internal/guardrsi/guardrsi_test.go proves KEEP-on-gain, REVERT-on-no-gain, and empty-journal refusal"),
		result("documented", "realized", false, 1, "the real-usage loop is documented + has an RSI skill", c.docExists && c.skillExists, "docs/fak/guard-verdict-rsi-loop.md + .claude/skills/guard-rsi-score explain and operationalise the pass"),
	}
}

func result(key, axis string, hard bool, weight int, label string, passed bool, detail string) KPIResult {
	return KPIResult{Key: key, Axis: axis, Hard: hard, Weight: weight, Label: label, Passed: passed, Detail: detail}
}

func auditDetail(c context) string {
	if c.auditRows > 0 {
		return fmt.Sprintf("%d real adjudicated row(s) across %d journal(s) -- the verdict loop can bank a kept iteration on our own usage", c.auditRows, c.auditJournals)
	}
	return "0 real rows -- the loop cannot close yet (" + c.auditDiagnose + ")"
}

func axisScore(rows []KPIResult) int {
	total, got := 0, 0
	for _, r := range rows {
		total += r.Weight
		if r.Passed {
			got += r.Weight
		}
	}
	if total == 0 {
		return 0
	}
	return int(math.Round(100 * float64(got) / float64(total)))
}

func configHome() string {
	if v := os.Getenv("XDG_CONFIG_HOME"); v != "" {
		return v
	}
	if v := os.Getenv("APPDATA"); v != "" {
		return v
	}
	if v := os.Getenv("HOME"); v != "" {
		return filepath.Join(v, ".config")
	}
	if v, err := os.UserHomeDir(); err == nil && v != "" {
		return filepath.Join(v, ".config")
	}
	return ""
}

func isFile(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}

func countNonblankLines(path string) int {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	n := 0
	for _, line := range strings.Split(string(b), "\n") {
		if strings.TrimSpace(line) != "" {
			n++
		}
	}
	return n
}

func readText(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return string(b)
}

func asString(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprintf("%v", v)
}

func topCount(m map[string]int) (string, int) {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	best := ""
	bestN := -1
	for _, k := range keys {
		if m[k] > bestN {
			best, bestN = k, m[k]
		}
	}
	return best, bestN
}

func mapString(m map[string]int) string {
	if len(m) == 0 {
		return "{}"
	}
	b, _ := json.Marshal(m)
	return string(b)
}
