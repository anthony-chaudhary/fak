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
)

const (
	VerdictSchema    = "guard-verdict-rsi/1"
	FoldSchema       = "guard-verdict-rsi.fold/1"
	ScorecardSchema  = "fak-guard-rsi-scorecard/1"
	defaultAuditPath = ".dispatch-runs/guard-audit"
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

type KPIPayload struct {
	KPI     string   `json:"kpi"`
	Group   string   `json:"group"`
	Score   int      `json:"score"`
	Detail  string   `json:"detail"`
	Defects []string `json:"defects"`
	Soft    []string `json:"soft"`
}

type ScorecardPayload struct {
	Schema     string         `json:"schema"`
	OK         bool           `json:"ok"`
	Verdict    string         `json:"verdict"`
	Finding    string         `json:"finding"`
	Reason     string         `json:"reason"`
	NextAction string         `json:"next_action"`
	Workspace  string         `json:"workspace"`
	Corpus     map[string]any `json:"corpus"`
	KPIs       []KPIPayload   `json:"kpis"`
	Maturity   []KPIResult    `json:"maturity"`
	Realized   []KPIResult    `json:"realized"`
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
			verdict := strings.ToUpper(strings.TrimSpace(asString(row["verdict"])))
			kind := strings.ToUpper(strings.TrimSpace(asString(row["kind"])))
			if verdict == "" {
				switch kind {
				case "DENY", "RESULT_DENY":
					verdict = "DENY"
				case "QUARANTINE":
					verdict = "QUARANTINE"
				case "DECIDE", "VDSO_HIT":
					verdict = "ALLOW"
				default:
					verdict = kind
				}
			}
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

func BuildScorecard(root string) ScorecardPayload {
	ctx := loadContext(root)
	maturity := maturityResults(ctx)
	realized := realizedResults(ctx)
	all := append(append([]KPIResult{}, maturity...), realized...)
	mScore := axisScore(maturity)
	rScore := axisScore(realized)
	composite := int(math.Round(0.4*float64(mScore) + 0.6*float64(rScore)))
	var hardFail []KPIResult
	for _, r := range all {
		if r.Hard && !r.Passed {
			hardFail = append(hardFail, r)
		}
	}
	debt := len(hardFail)
	grade := GradeLetter(composite)
	ok := debt == 0
	verdict, finding, reason, next := "OK", "guard_rsi_loop_mature_and_useful", "", ""
	if ok {
		reason = fmt.Sprintf("guard RSI loop: maturity %d/100, realized %d/100, composite %d/100 (%s); zero hard gaps; %d real journal row(s)", mScore, rScore, composite, grade, ctx.auditRows)
		next = "hold the line; re-run after a change to either guard RSI loop, or run `fak guard-verdict-rsi run` after a fresh guarded session"
	} else {
		verdict, finding = "ACTION", "guard_rsi_debt"
		keys := make([]string, len(hardFail))
		for i, r := range hardFail {
			keys[i] = r.Key
		}
		reason = fmt.Sprintf("guard RSI loop carries %d hard gap(s) (maturity %d/100, realized %d/100, composite %d/100 %s): %s", debt, mScore, rScore, composite, grade, strings.Join(keys, ", "))
		lead := hardFail[0]
		next = fmt.Sprintf("retire worst-first: %s -- %s", lead.Key, lead.Detail)
	}
	return ScorecardPayload{
		Schema:     ScorecardSchema,
		OK:         ok,
		Verdict:    verdict,
		Finding:    finding,
		Reason:     reason,
		NextAction: next,
		Workspace:  root,
		Corpus: map[string]any{
			"guard_rsi_debt":  debt,
			"score":           composite,
			"grade":           grade,
			"maturity_score":  mScore,
			"realized_score":  rScore,
			"audit_rows":      ctx.auditRows,
			"verdict_quality": ctx.verdictQuality,
		},
		KPIs:     kpiPayloads(all),
		Maturity: maturity,
		Realized: realized,
	}
}

func RenderScorecard(p ScorecardPayload) string {
	c := p.Corpus
	vq := ""
	if c["verdict_quality"] != nil {
		vq = fmt.Sprintf("   verdict-quality %.3g", c["verdict_quality"])
	}
	lines := []string{
		fmt.Sprintf("guard RSI loop -- %s (%s)", p.Verdict, p.Finding),
		fmt.Sprintf("  guard_rsi_debt: %v   composite %v/100 [%v]   (maturity %v; realized %v)", c["guard_rsi_debt"], c["score"], c["grade"], c["maturity_score"], c["realized_score"]),
		fmt.Sprintf("  real journal: %v row(s)%s", c["audit_rows"], vq),
		"",
		"  MATURITY (can the loop honestly close?):",
	}
	for _, r := range p.Maturity {
		lines = append(lines, scorecardLine(r))
		if !r.Passed {
			lines = append(lines, "           -> "+r.Detail)
		}
	}
	lines = append(lines, "  REALIZED (operationalised on our usage?):")
	for _, r := range p.Realized {
		lines = append(lines, scorecardLine(r))
		if !r.Passed {
			lines = append(lines, "           -> "+r.Detail)
		}
	}
	lines = append(lines, "", "  -> "+p.NextAction)
	return strings.Join(lines, "\n")
}

func Markdown(p ScorecardPayload) string {
	c := p.Corpus
	var b strings.Builder
	fmt.Fprintln(&b, "---")
	fmt.Fprintln(&b, `title: "fak guard RSI loop scorecard"`)
	fmt.Fprintln(&b, `description: "How mature and realized the RSI loop(s) for fak guard are, scored from the tree plus the real decision journal."`)
	fmt.Fprintln(&b, "---")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "# fak guard RSI loop scorecard")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "**guard_rsi_debt: %v**; composite **%v/100 (%v)**; maturity %v/100; realized %v/100; real journal rows %v\n\n", c["guard_rsi_debt"], c["score"], c["grade"], c["maturity_score"], c["realized_score"], c["audit_rows"])
	fmt.Fprintf(&b, "> %s\n\n", p.Reason)
	fmt.Fprintln(&b, "## Maturity -- can the loop honestly close?")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "| ok | criterion |")
	fmt.Fprintln(&b, "|---|---|")
	for _, r := range p.Maturity {
		fmt.Fprintf(&b, "| %s | %s |\n", passMark(r.Passed), r.Label)
	}
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "## Realized -- does it run on our own usage?")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "| ok | criterion | detail |")
	fmt.Fprintln(&b, "|---|---|---|")
	for _, r := range p.Realized {
		fmt.Fprintf(&b, "| %s | %s | %s |\n", passMark(r.Passed), r.Label, r.Detail)
	}
	fmt.Fprintf(&b, "\n**Next:** %s\n", p.NextAction)
	return b.String()
}

func Compare(current ScorecardPayload, baseline map[string]any) string {
	bc, _ := baseline["corpus"].(map[string]any)
	if bc == nil {
		bc = baseline
	}
	bDebt := anyInt(bc["guard_rsi_debt"])
	if bDebt == 0 {
		bDebt = anyInt(bc["score"])
	}
	cDebt := anyInt(current.Corpus["guard_rsi_debt"])
	delta := bDebt - cDebt
	lines := []string{
		"guard-rsi compare:",
		fmt.Sprintf("  guard_rsi_debt: %d -> %d  (retired %d)", bDebt, cDebt, delta),
		fmt.Sprintf("  composite: %v -> %v  grade %v -> %v", bc["score"], current.Corpus["score"], bc["grade"], current.Corpus["grade"]),
		fmt.Sprintf("  real journal rows: %v -> %v", bc["audit_rows"], current.Corpus["audit_rows"]),
	}
	switch {
	case bDebt > 0 && cDebt*3 <= bDebt:
		lines = append(lines, fmt.Sprintf("  VERDICT: >=3x improvement (debt %d -> %d, <= 1/3 of baseline)", bDebt, cDebt))
	case bDebt > 0 && cDebt*2 <= bDebt:
		lines = append(lines, fmt.Sprintf("  VERDICT: >=2x improvement (debt %d -> %d)", bDebt, cDebt))
	case bDebt > 0 && cDebt < bDebt:
		lines = append(lines, fmt.Sprintf("  VERDICT: improved but < 2x (debt %d -> %d)", bDebt, cDebt))
	case bDebt > 0:
		lines = append(lines, fmt.Sprintf("  VERDICT: no improvement (debt %d -> %d)", bDebt, cDebt))
	}
	return strings.Join(lines, "\n")
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

func GradeLetter(score int) string {
	switch {
	case score >= 90:
		return "A"
	case score >= 80:
		return "B"
	case score >= 70:
		return "C"
	case score >= 60:
		return "D"
	default:
		return "F"
	}
}

func kpiPayloads(rows []KPIResult) []KPIPayload {
	out := make([]KPIPayload, 0, len(rows))
	for _, r := range rows {
		k := KPIPayload{KPI: r.Key, Group: r.Axis, Detail: r.Detail}
		if r.Passed {
			k.Score = 100
		} else if r.Hard {
			k.Defects = []string{r.Key + ": " + r.Detail}
		} else {
			k.Soft = []string{r.Key + ": " + r.Detail}
		}
		out = append(out, k)
	}
	return out
}

func scorecardLine(r KPIResult) string {
	mark := "PASS"
	if !r.Passed {
		if r.Hard {
			mark = "FAIL"
		} else {
			mark = "----"
		}
	}
	return fmt.Sprintf("    [%s] %s", mark, r.Label)
}

func passMark(ok bool) string {
	if ok {
		return "yes"
	}
	return "no"
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

func anyInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	case json.Number:
		i, _ := n.Int64()
		return int(i)
	default:
		return 0
	}
}
