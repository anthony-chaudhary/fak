package productscorecard

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

const (
	Schema      = "fak-product-scorecard/1"
	DataDirRel  = "tools/product_scorecard.data"
	ClaimsRel   = "CLAIMS.md"
	CLIRefRel   = "docs/cli-reference.md"
	DefaultDocs = "docs/product-scorecard"
)

type Row map[string]any
type Data struct {
	Meta               map[string]any
	Categories         []map[string]any
	Rows               []Row
	ManagedContextSLOs []ManagedContextSLO
}

type ManagedContextSLO struct {
	ID         string `json:"id"`
	Name       string `json:"name"`
	Area       string `json:"area"`
	Status     string `json:"status"`
	Hard       bool   `json:"hard"`
	Source     string `json:"source,omitempty"`
	Detail     string `json:"detail,omitempty"`
	NextAction string `json:"next_action,omitempty"`
}

type Section struct {
	Section string `json:"section"`
	Norm    string `json:"norm"`
}

type Tree struct {
	Catalog     []Section
	SectionTags map[string]map[string]bool
	CmdDirs     map[string]bool
	DocVerbs    map[string]bool
	Exists      func(string) bool
}

type KPI struct {
	KPI     string   `json:"kpi"`
	Group   string   `json:"group"`
	Value   float64  `json:"value"`
	Score   int      `json:"score"`
	Detail  string   `json:"detail"`
	Defects []string `json:"defects"`
	Soft    []string `json:"soft"`
}

type Payload struct {
	Schema     string         `json:"schema"`
	OK         bool           `json:"ok"`
	Verdict    string         `json:"verdict"`
	Finding    string         `json:"finding"`
	Reason     string         `json:"reason"`
	NextAction string         `json:"next_action"`
	Workspace  string         `json:"workspace"`
	Corpus     map[string]any `json:"corpus"`
	KPIs       []KPI          `json:"kpis"`
	Data       map[string]any `json:"_data,omitempty"`
}

var (
	maturities                 = set("shipped", "simulated", "stub", "concept")
	claimsTags                 = set("SHIPPED", "SIMULATED", "STUB")
	audiences                  = set("buyer", "platform", "developer", "researcher")
	surfaces                   = set("product", "benchmark", "subsystem", "seam")
	verdicts                   = []string{"durable-product", "usable-today", "real-not-easy", "honest-stub", "concept-only"}
	groups                     = []string{"well-formed", "honesty", "usefulness", "durability", "managed-context"}
	nonConcepts                = set("what fak is not", "prior-art posture")
	requiredManagedContextSLOs = []ManagedContextSLO{
		{ID: "assumption_effect_safety", Name: "Assumption effect safety", Area: "assumption", Hard: true},
		{ID: "context_visibility", Name: "Context visibility", Area: "visibility", Hard: true},
		{ID: "pinned_objective_reconciliation", Name: "Pinned-objective reconciliation", Area: "objective", Hard: true},
		{ID: "deterministic_resets", Name: "Deterministic resets", Area: "reset", Hard: true},
		{ID: "budget_compliance", Name: "Budget compliance", Area: "budget", Hard: true},
		{ID: "query_correctness", Name: "Query correctness", Area: "query", Hard: true},
		{ID: "cache_preservation", Name: "Cache preservation", Area: "cache", Hard: true},
		{ID: "memory_promotion_safety", Name: "Memory-promotion safety", Area: "memory", Hard: true},
	}

	maturityTag = map[string]string{"shipped": "SHIPPED", "simulated": "SIMULATED", "stub": "STUB"}
	kpiGroup    = map[string]string{
		"well_formed":         "well-formed",
		"claim_honest":        "honesty",
		"verdict_consistency": "honesty",
		"command_resolves":    "usefulness",
		"witnessed":           "durability",
		"discoverable":        "durability",
	}
	kpiWeight = map[string]float64{
		"well_formed": 0.12, "claim_honest": 0.22, "verdict_consistency": 0.22,
		"command_resolves": 0.14, "witnessed": 0.16, "discoverable": 0.14,
	}
	kpiPenalty = map[string]int{
		"well_formed": 12, "claim_honest": 20, "verdict_consistency": 25,
		"command_resolves": 15, "witnessed": 15, "discoverable": 15,
	}
	requiredFields = []string{
		"id", "concept", "category", "surface", "what_you_get", "audience", "maturity",
		"claims_section", "claims_tag", "first_command", "first_command_verb",
		"needs_gpu", "needs_key", "witness_path", "witness", "entry_doc",
		"verdict", "gaps", "durability_note",
	}
	verdictRank = rank(verdicts)
)

func set(vals ...string) map[string]bool {
	out := map[string]bool{}
	for _, v := range vals {
		out[v] = true
	}
	return out
}

func rank(vals []string) map[string]int {
	out := map[string]int{}
	for i, v := range vals {
		out[v] = i
	}
	return out
}

func stringValue(r Row, key string) string {
	if v, ok := r[key].(string); ok {
		return v
	}
	return ""
}

// rowID returns the row's declared id, falling back to its zero-based index
// when the id is absent. It is the shared preamble every per-row KPI loop uses
// to label a defect.
func rowID(r Row, i int) string {
	rid := stringValue(r, "id")
	if rid == "" {
		rid = fmt.Sprint(i)
	}
	return rid
}

func boolValue(r Row, key string) bool {
	if v, ok := r[key].(bool); ok {
		return v
	}
	return false
}

func listValue(r Row, key string) []string {
	raw, ok := r[key].([]any)
	if !ok {
		if ss, ok := r[key].([]string); ok {
			return ss
		}
		return nil
	}
	out := make([]string, 0, len(raw))
	for _, v := range raw {
		if s, ok := v.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func NonEmpty(v any) bool {
	s, ok := v.(string)
	return ok && strings.TrimSpace(s) != ""
}

func clamp(score float64) int {
	if score < 0 {
		return 0
	}
	if score > 100 {
		return 100
	}
	return int(math.Round(score))
}

func GradeLetter(score float64) string {
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

func NormSection(s any) string {
	text, ok := s.(string)
	if !ok {
		return ""
	}
	text = strings.TrimSpace(text)
	for strings.HasPrefix(text, "#") {
		text = strings.TrimSpace(strings.TrimPrefix(text, "#"))
	}
	cut := len(text)
	for _, sep := range []string{"(", "\u2014", ":"} {
		if idx := strings.Index(text, sep); idx >= 0 && idx < cut {
			cut = idx
		}
	}
	text = strings.ToLower(strings.TrimSpace(text[:cut]))
	return strings.Join(strings.Fields(text), " ")
}

func SectionMatch(rowSection, catalogNorm string) bool {
	a, b := NormSection(rowSection), catalogNorm
	if a == "" || b == "" {
		return false
	}
	if a == b {
		return true
	}
	return len(a) >= 6 && len(b) >= 6 && (strings.Contains(a, b) || strings.Contains(b, a))
}

var cmdRe = regexp.MustCompile(`\./cmd/([\w-]+)`)

func ParseCommand(cmd any) (string, string) {
	if !NonEmpty(cmd) {
		return "", ""
	}
	text := cmd.(string)
	m := cmdRe.FindStringSubmatchIndex(text)
	if m == nil {
		return "", ""
	}
	dir := text[m[2]:m[3]]
	rest := strings.TrimSpace(text[m[1]:])
	verb := ""
	for _, tok := range strings.Fields(rest) {
		if strings.HasPrefix(tok, "-") {
			break
		}
		verb = tok
		break
	}
	return dir, verb
}

func makeKPI(name string, defects []string, okDetail string, soft []string, badDetail string) KPI {
	detail := okDetail
	if len(defects) > 0 {
		if badDetail != "" {
			detail = badDetail
		} else {
			detail = fmt.Sprintf("%d defect(s)", len(defects))
		}
	}
	score := clamp(float64(100-kpiPenalty[name]*len(defects)) - math.Min(10, float64(2*len(soft))))
	return KPI{KPI: name, Group: kpiGroup[name], Value: productValue(float64(score)), Score: score, Detail: detail, Defects: defects, Soft: soft}
}

func productValue(score float64) float64 {
	return scorecard.Round3(scorecard.ValueFromScore(score))
}

func productValueFor(value, legacyScore float64) float64 {
	if value != 0 || legacyScore == 0 {
		return scorecard.Round3(value)
	}
	return productValue(legacyScore)
}

func KPIWellFormed(rows []Row, categories map[string]bool) KPI {
	var defects []string
	seen := map[string]bool{}
	for i, r := range rows {
		rid := stringValue(r, "id")
		if strings.TrimSpace(rid) == "" {
			rid = fmt.Sprintf("row[%d]", i)
		}
		for _, f := range requiredFields {
			if _, ok := r[f]; !ok {
				defects = append(defects, fmt.Sprintf("%s: missing field '%s'", rid, f))
			}
		}
		if !NonEmpty(r["id"]) {
			defects = append(defects, fmt.Sprintf("%s: missing id", rid))
		} else if seen[stringValue(r, "id")] {
			defects = append(defects, fmt.Sprintf("%s: duplicate id", rid))
		} else {
			seen[stringValue(r, "id")] = true
		}
		if len(categories) > 0 && !categories[stringValue(r, "category")] {
			defects = append(defects, fmt.Sprintf("%s: category %q not declared in _meta.json", rid, stringValue(r, "category")))
		}
		if !surfaces[stringValue(r, "surface")] {
			defects = append(defects, fmt.Sprintf("%s: surface %q not in %v", rid, stringValue(r, "surface"), sortedKeys(surfaces)))
		}
		if !maturities[stringValue(r, "maturity")] {
			defects = append(defects, fmt.Sprintf("%s: maturity %q not in %v", rid, stringValue(r, "maturity"), sortedKeys(maturities)))
		}
		if !claimsTags[stringValue(r, "claims_tag")] {
			defects = append(defects, fmt.Sprintf("%s: claims_tag %q not in %v", rid, stringValue(r, "claims_tag"), sortedKeys(claimsTags)))
		}
		if !audiences[stringValue(r, "audience")] {
			defects = append(defects, fmt.Sprintf("%s: audience %q not in %v", rid, stringValue(r, "audience"), sortedKeys(audiences)))
		}
		if _, ok := verdictRank[stringValue(r, "verdict")]; !ok {
			defects = append(defects, fmt.Sprintf("%s: verdict %q not in %v", rid, stringValue(r, "verdict"), verdicts))
		}
		for _, b := range []string{"needs_gpu", "needs_key"} {
			if _, ok := r[b].(bool); !ok {
				defects = append(defects, fmt.Sprintf("%s: %s must be a bool", rid, b))
			}
		}
		if !NonEmpty(r["what_you_get"]) {
			defects = append(defects, fmt.Sprintf("%s: missing what_you_get (one plain sentence)", rid))
		}
		if _, ok := r["gaps"].([]any); !ok {
			if _, ok2 := r["gaps"].([]string); !ok2 {
				defects = append(defects, fmt.Sprintf("%s: gaps must be a list", rid))
			}
		}
	}
	return makeKPI("well_formed", defects, fmt.Sprintf("all %d rows well-formed", len(rows)), nil, fmt.Sprintf("%d malformed field(s)", len(defects)))
}

func KPIClaimHonest(rows []Row, sectionTags map[string]map[string]bool) KPI {
	var defects, soft []string
	for i, r := range rows {
		rid := rowID(r, i)
		mat, tag := stringValue(r, "maturity"), stringValue(r, "claims_tag")
		if expected, ok := maturityTag[mat]; ok && claimsTags[tag] && expected != tag {
			defects = append(defects, fmt.Sprintf("%s: maturity '%s' disagrees with claims_tag '%s' (expected %s)", rid, mat, tag, expected))
		}
		var tags map[string]bool
		for norm, tagset := range sectionTags {
			if SectionMatch(stringValue(r, "claims_section"), norm) {
				tags = tagset
				break
			}
		}
		switch {
		case tags == nil:
			if mat != "concept" {
				soft = append(soft, fmt.Sprintf("%s: claims_section %q did not match a CLAIMS.md section - cannot cross-check the tag", rid, stringValue(r, "claims_section")))
			}
		case claimsTags[tag] && !tags[tag]:
			defects = append(defects, fmt.Sprintf("%s: claims_tag '%s' but CLAIMS.md section carries only %v - overclaim vs the honesty ledger", rid, tag, sortedKeys(tags)))
		}
	}
	return makeKPI("claim_honest", defects, fmt.Sprintf("every claimed maturity matches CLAIMS.md (%d unmatched section)", len(soft)), soft, fmt.Sprintf("%d maturity overclaim(s) vs CLAIMS.md", len(defects)))
}

func KPICommandResolves(rows []Row, cmdDirs, docVerbs map[string]bool) KPI {
	var defects []string
	for i, r := range rows {
		rid := rowID(r, i)
		cmd := stringValue(r, "first_command")
		verbField := stringValue(r, "first_command_verb")
		if strings.TrimSpace(cmd) == "" {
			if strings.TrimSpace(verbField) != "" {
				defects = append(defects, fmt.Sprintf("%s: first_command_verb set but first_command is empty", rid))
			}
			continue
		}
		cmdDir, verb := ParseCommand(cmd)
		if cmdDir == "" {
			defects = append(defects, fmt.Sprintf("%s: first_command has no `./cmd/<dir>` invocation - cannot verify it runs", rid))
			continue
		}
		if !cmdDirs[cmdDir] {
			defects = append(defects, fmt.Sprintf("%s: first_command runs ./cmd/%s which does not exist", rid, cmdDir))
			continue
		}
		if cmdDir == "fak" && verb != "" && !docVerbs[strings.ToLower(verb)] {
			defects = append(defects, fmt.Sprintf("%s: fak verb '%s' is not documented in docs/cli-reference.md", rid, verb))
		}
	}
	return makeKPI("command_resolves", defects, "every first command resolves to a real cmd dir + documented verb", nil, fmt.Sprintf("%d unrunnable first command(s)", len(defects)))
}

func KPIWitnessed(rows []Row, exists func(string) bool) KPI {
	var defects []string
	for i, r := range rows {
		rid := rowID(r, i)
		mat := stringValue(r, "maturity")
		if mat == "shipped" || mat == "simulated" {
			wp := stringValue(r, "witness_path")
			if strings.TrimSpace(wp) == "" {
				defects = append(defects, fmt.Sprintf("%s: %s but no witness_path - name the test dir / results doc that proves it", rid, mat))
			} else if !exists(wp) {
				defects = append(defects, fmt.Sprintf("%s: witness_path '%s' does not exist in the tree", rid, wp))
			}
		}
	}
	return makeKPI("witnessed", defects, "every shipped/simulated concept is witnessed by a real path", nil, fmt.Sprintf("%d unproven product claim(s)", len(defects)))
}

func KPIDiscoverable(rows []Row, exists func(string) bool) KPI {
	var defects []string
	for i, r := range rows {
		rid := rowID(r, i)
		if stringValue(r, "maturity") == "concept" {
			continue
		}
		ed := stringValue(r, "entry_doc")
		if strings.TrimSpace(ed) == "" {
			defects = append(defects, fmt.Sprintf("%s: no entry_doc - name where a person learns it", rid))
		} else if !exists(ed) {
			defects = append(defects, fmt.Sprintf("%s: entry_doc '%s' does not exist in the tree", rid, ed))
		}
	}
	return makeKPI("discoverable", defects, "every usable concept has a real entry doc", nil, fmt.Sprintf("%d undiscoverable concept(s)", len(defects)))
}

func ExpectedVerdict(row Row) (string, string) {
	mat := stringValue(row, "maturity")
	switch mat {
	case "concept":
		return "concept-only", "a roadmap idea (maturity=concept)"
	case "stub", "simulated":
		return "honest-stub", fmt.Sprintf("a labeled seam (maturity=%s)", mat)
	}
	surface := stringValue(row, "surface")
	if surface == "subsystem" || surface == "seam" || !NonEmpty(row["first_command"]) {
		if surface == "" {
			surface = "subsystem"
		}
		return "real-not-easy", fmt.Sprintf("shipped %s with no product surface a person runs directly", surface)
	}
	if surface == "benchmark" {
		return "usable-today", "shipped benchmark/demo a person runs today to see or reproduce a result"
	}
	if boolValue(row, "needs_gpu") || boolValue(row, "needs_key") {
		return "usable-today", "shipped product surface, but its first command needs a GPU / key / network"
	}
	return "durable-product", "shipped product surface + an OFFLINE first command a person runs today"
}

func KPIVerdictConsistency(rows []Row) KPI {
	var defects []string
	for i, r := range rows {
		rid := rowID(r, i)
		exp, why := ExpectedVerdict(r)
		if stringValue(r, "verdict") != exp {
			defects = append(defects, fmt.Sprintf("%s: claims '%s' but evidence implies '%s' - %s", rid, stringValue(r, "verdict"), exp, why))
		}
	}
	return makeKPI("verdict_consistency", defects, "every verdict matches its evidence", nil, fmt.Sprintf("%d verdict overclaim(s)", len(defects)))
}

func CoverageReport(catalog []Section, rows []Row) map[string]any {
	var uncovered []Section
	covered := 0
	for _, sec := range catalog {
		hit := false
		for _, r := range rows {
			if SectionMatch(stringValue(r, "claims_section"), sec.Norm) {
				hit = true
				break
			}
		}
		if hit {
			covered++
		} else {
			uncovered = append(uncovered, sec)
		}
	}
	total := len(catalog)
	pct := 100.0
	if total > 0 {
		pct = math.Round((1000.0*float64(covered))/float64(total)) / 10
	}
	return map[string]any{
		"catalog_total": total,
		"covered":       covered,
		"coverage_pct":  pct,
		"coverage_debt": total - covered,
		"uncovered":     uncovered,
	}
}

func RunKPIs(rows []Row, categories map[string]bool, tree Tree) []KPI {
	exists := tree.Exists
	if exists == nil {
		exists = func(string) bool { return false }
	}
	return []KPI{
		KPIWellFormed(rows, categories),
		KPIClaimHonest(rows, tree.SectionTags),
		KPIVerdictConsistency(rows),
		KPICommandResolves(rows, tree.CmdDirs, tree.DocVerbs),
		KPIWitnessed(rows, exists),
		KPIDiscoverable(rows, exists),
	}
}

func ManagedContextSLOReport(slos []ManagedContextSLO) map[string]any {
	if len(slos) == 0 {
		return nil
	}
	byID := map[string]ManagedContextSLO{}
	for _, slo := range slos {
		id := strings.TrimSpace(slo.ID)
		if id == "" {
			continue
		}
		if strings.TrimSpace(slo.Status) == "" {
			slo.Status = "unknown"
		}
		if strings.TrimSpace(slo.Area) == "" {
			for _, req := range requiredManagedContextSLOs {
				if req.ID == id {
					slo.Area = req.Area
					break
				}
			}
		}
		if strings.TrimSpace(slo.Name) == "" {
			for _, req := range requiredManagedContextSLOs {
				if req.ID == id {
					slo.Name = req.Name
					break
				}
			}
		}
		byID[id] = slo
	}
	rows := make([]map[string]any, 0, len(requiredManagedContextSLOs))
	debt, passed := 0, 0
	for _, req := range requiredManagedContextSLOs {
		slo, ok := byID[req.ID]
		if !ok {
			slo = req
			slo.Status = "missing"
			slo.Detail = "no managed-context SLO fixture declared"
			slo.NextAction = "declare the fixture and witness before calling this SLO green"
		}
		hard := slo.Hard || req.Hard
		status := strings.ToLower(strings.TrimSpace(slo.Status))
		if status == "" {
			status = "unknown"
		}
		failing := hard && status != "pass"
		if failing {
			debt++
		} else if status == "pass" {
			passed++
		}
		rows = append(rows, map[string]any{
			"id": slo.ID, "name": slo.Name, "area": slo.Area, "status": status,
			"hard": hard, "debt": boolToInt(failing), "source": slo.Source,
			"detail": slo.Detail, "next_action": slo.NextAction,
		})
	}
	score := 100.0
	total := len(requiredManagedContextSLOs)
	if total > 0 {
		score = math.Round((100.0*float64(total-debt)/float64(total))*10) / 10
	}
	value := productValue(score)
	return map[string]any{
		"schema": "fak-managed-context-slos/1", "debt": debt,
		"value": value, "value_unit": "quality_ratio", "score": score, "legacy_score": score, "legacy_score_scale": 100,
		"passed": passed, "total": total, "rows": rows,
	}
}

func BuildPayload(workspace string, data *Data, tree Tree, readErr string) Payload {
	if readErr != "" || data == nil {
		reason := readErr
		if reason == "" {
			reason = "no data"
		}
		return Payload{
			Schema: Schema, OK: false, Verdict: "AUDIT_ERROR", Finding: "tooling_error",
			Reason: reason, NextAction: fmt.Sprintf("fix the read (run from repo ROOT; check %s/), then re-run", DataDirRel),
			Workspace: workspace, Corpus: map[string]any{}, KPIs: []KPI{},
		}
	}
	categories := map[string]bool{}
	for _, c := range data.Categories {
		if id, ok := c["id"].(string); ok && strings.TrimSpace(id) != "" {
			categories[id] = true
		}
	}
	rows := data.Rows
	kpis := RunKPIs(rows, categories, tree)
	byName := map[string]KPI{}
	for i := range kpis {
		kpis[i].Value = productValueFor(kpis[i].Value, float64(kpis[i].Score))
		byName[kpis[i].KPI] = kpis[i]
	}
	honestyScore := 0.0
	for name, w := range kpiWeight {
		if k, ok := byName[name]; ok {
			honestyScore += w * float64(k.Score)
		}
	}
	honestyScore = math.Round(honestyScore*10) / 10
	honestyDefects, softSignals := 0, 0
	for _, k := range kpis {
		honestyDefects += len(k.Defects)
		softSignals += len(k.Soft)
	}
	cov := CoverageReport(tree.Catalog, rows)
	coverageDebt := intValue(cov["coverage_debt"])
	managedContext := ManagedContextSLOReport(data.ManagedContextSLOs)
	managedContextDebt := intValue(managedContext["debt"])
	productDebt := honestyDefects + coverageDebt + managedContextDebt
	covPct := floatValue(cov["coverage_pct"])
	score := math.Round((0.60*honestyScore+0.40*covPct)*10) / 10
	if len(managedContext) > 0 {
		score = math.Round((0.50*honestyScore+0.30*covPct+0.20*floatValue(managedContext["score"]))*10) / 10
	}
	value := productValue(score)
	grade := GradeLetter(score)
	debtByGroup := map[string]any{}
	for _, g := range groups {
		debtByGroup[g] = 0
	}
	for _, k := range kpis {
		debtByGroup[k.Group] = intValue(debtByGroup[k.Group]) + len(k.Defects)
	}
	debtByGroup["managed-context"] = managedContextDebt
	breakdown := make([]map[string]any, 0, len(kpis))
	for _, k := range kpis {
		breakdown = append(breakdown, map[string]any{
			"kpi": k.KPI, "group": k.Group, "value": k.Value, "score": k.Score, "debt": len(k.Defects), "detail": k.Detail,
		})
	}
	sort.SliceStable(breakdown, func(i, j int) bool {
		di, dj := intValue(breakdown[i]["debt"]), intValue(breakdown[j]["debt"])
		if di != dj {
			return di > dj
		}
		return intValue(breakdown[i]["score"]) < intValue(breakdown[j]["score"])
	})
	pos := Standing(rows)
	rowDebt := PerRowDebt(rows, kpis)
	crit := CriticalBacklog(rows, rowDebt)
	nDurable := pos["durable-product"]
	kpiScores, kpiValues, debtByKPI := map[string]any{}, map[string]any{}, map[string]any{}
	for _, k := range kpis {
		kpiScores[k.KPI] = k.Score
		kpiValues[k.KPI] = k.Value
		debtByKPI[k.KPI] = len(k.Defects)
	}
	corpus := map[string]any{
		"value": value, "value_unit": "quality_ratio", "score": score, "legacy_score": score, "legacy_score_scale": 100,
		"grade": grade, "honesty_value": productValue(honestyScore), "honesty_score": honestyScore,
		"coverage_value": productValue(covPct),
		"product_debt":   productDebt, "honesty_defects": honestyDefects,
		"coverage_debt": coverageDebt, "managed_context_debt": managedContextDebt,
		"managed_context": managedContext, "coverage": cov, "soft_signals": softSignals,
		"rows": len(rows), "durable_products": nDurable, "as_of": stringMapValue(data.Meta, "as_of"),
		"fak_version": stringMapValue(data.Meta, "fak_version"), "standing": anyIntMap(pos),
		"debt_by_group": debtByGroup, "kpi_scores": kpiScores, "kpi_values": kpiValues, "debt_by_kpi": debtByKPI,
		"breakdown": breakdown, "leaderboard": Leaderboard(rows), "critical": crit,
	}
	standingLine := fmt.Sprintf("%d durable - %d usable-today - %d real-not-easy - %d honest-stub - %d concept",
		pos["durable-product"], pos["usable-today"], pos["real-not-easy"], pos["honest-stub"], pos["concept-only"])
	covLine := fmt.Sprintf("coverage %.1f%% (%d/%d concept sections positioned)", covPct, intValue(cov["covered"]), intValue(cov["catalog_total"]))
	ok, verdict, finding := false, "ACTION", "product_debt"
	var reason, next string
	switch {
	case productDebt == 0:
		ok, verdict, finding = true, "OK", "product_map_complete_and_honest"
		reason = fmt.Sprintf("product map complete + honest: value %.3f (grade %s, legacy score %.1f); %s; zero product-debt across %d KPIs over %d concepts (%s; %d advisory)", value, grade, score, covLine, len(kpis), len(rows), standingLine, softSignals)
		next = "hold the line; when CLAIMS.md adds a concept section, coverage drops - position it; when a stub ships, raise its verdict; re-run to keep debt at 0"
	case honestyDefects == 0 && managedContextDebt > 0:
		finding = "managed_context_debt"
		reason = fmt.Sprintf("%d managed-context SLO fixture(s) failing + %d coverage gap(s) = product-debt %d; %s; value %.3f (grade %s, legacy score %.1f); standing %s",
			managedContextDebt, coverageDebt, productDebt, covLine, value, grade, score, standingLine)
		next = "retire managed-context SLO debt: make each hard fixture pass with a witness for visibility, resets, budgets, queries, cache preservation, and memory-promotion safety; re-run"
	case honestyDefects == 0 && coverageDebt > 0:
		finding = "coverage_debt"
		reason = fmt.Sprintf("%d concept section(s) not yet positioned; %s; value %.3f (grade %s, legacy score %.1f); rows are honest (0 honesty-debt); standing %s", coverageDebt, covLine, value, grade, score, standingLine)
		next = "close coverage (see --gaps): add an honest product row for each uncovered CLAIMS.md concept section (a real-not-easy / honest-stub row is valid); re-run"
	default:
		worst := breakdown[0]
		reason = fmt.Sprintf("%d honesty/quality defect(s) + %d coverage gap(s) + %d managed-context SLO defect(s) = product-debt %d; value %.3f (grade %s, legacy score %.1f); heaviest KPI: %s (%d); %s; standing %s",
			honestyDefects, coverageDebt, managedContextDebt, productDebt, value, grade, score, worst["kpi"], intValue(worst["debt"]), covLine, standingLine)
		next = "retire product-debt worst-first (--critical + per-KPI defects): fix overclaims, command targets, witnesses, entry docs, then close coverage (--gaps); re-run to prove the drop"
	}
	return Payload{
		Schema: Schema, OK: ok, Verdict: verdict, Finding: finding, Reason: reason, NextAction: next,
		Workspace: workspace, Corpus: corpus, KPIs: kpis,
		Data: map[string]any{"rows": rows, "categories": data.Categories, "catalog": tree.Catalog},
	}
}

func Standing(rows []Row) map[string]int {
	out := map[string]int{}
	for _, v := range verdicts {
		out[v] = 0
	}
	for _, r := range rows {
		if _, ok := out[stringValue(r, "verdict")]; ok {
			out[stringValue(r, "verdict")]++
		}
	}
	return out
}

func PerRowDebt(rows []Row, kpis []KPI) map[string]int {
	out := map[string]int{}
	for i, r := range rows {
		id := stringValue(r, "id")
		if id == "" {
			id = fmt.Sprintf("row[%d]", i)
		}
		out[id] = 0
	}
	for _, k := range kpis {
		for _, d := range k.Defects {
			rid := strings.SplitN(d, ":", 2)[0]
			if _, ok := out[rid]; ok {
				out[rid]++
			}
		}
	}
	return out
}

func Leaderboard(rows []Row) []map[string]any {
	out := make([]map[string]any, 0, len(rows))
	for _, r := range rows {
		exp, _ := ExpectedVerdict(r)
		out = append(out, map[string]any{
			"id": stringValue(r, "id"), "concept": stringValue(r, "concept"), "category": stringValue(r, "category"),
			"surface": stringValue(r, "surface"), "maturity": stringValue(r, "maturity"), "verdict": stringValue(r, "verdict"),
			"expected_verdict": exp, "what_you_get": stringValue(r, "what_you_get"), "first_command": stringValue(r, "first_command"),
			"offline":      NonEmpty(r["first_command"]) && !boolValue(r, "needs_gpu") && !boolValue(r, "needs_key"),
			"witness_path": stringValue(r, "witness_path"), "entry_doc": stringValue(r, "entry_doc"),
		})
	}
	return out
}

func CriticalBacklog(rows []Row, rowDebt map[string]int) []map[string]any {
	out := make([]map[string]any, 0, len(rows))
	for _, r := range rows {
		id := stringValue(r, "id")
		out = append(out, map[string]any{
			"id": id, "concept": stringValue(r, "concept"), "category": stringValue(r, "category"),
			"verdict": stringValue(r, "verdict"), "debt": rowDebt[id],
			"distance": rankOf(stringValue(r, "verdict")), "gaps": listValue(r, "gaps"),
		})
	}
	sort.SliceStable(out, func(i, j int) bool {
		di, dj := intValue(out[i]["debt"]), intValue(out[j]["debt"])
		if di != dj {
			return di > dj
		}
		ri, rj := intValue(out[i]["distance"]), intValue(out[j]["distance"])
		if ri != rj {
			return ri > rj
		}
		return stringValue(Row{"id": out[i]["id"]}, "id") < stringValue(Row{"id": out[j]["id"]}, "id")
	})
	return out
}

func LoadDataDir(dir string) (*Data, string) {
	metaDoc, err := readJSON(filepath.Join(dir, "_meta.json"))
	if err != "" {
		return nil, err
	}
	metaObj, ok := metaDoc.(map[string]any)
	if !ok {
		return nil, "_meta.json is not a JSON object"
	}
	out := &Data{
		Meta:               mapValue(metaObj["meta"]),
		Categories:         mapList(metaObj["categories"]),
		Rows:               []Row{},
		ManagedContextSLOs: managedContextSLOList(metaObj["managed_context_slos"]),
	}
	entries, e := os.ReadDir(dir)
	if e != nil {
		return nil, fmt.Sprintf("cannot read data directory: %v", e)
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	for _, ent := range entries {
		name := ent.Name()
		if ent.IsDir() || !strings.HasSuffix(name, ".json") || strings.HasPrefix(name, "_") {
			continue
		}
		doc, err := readJSON(filepath.Join(dir, name))
		if err != "" {
			return nil, err
		}
		for _, r := range rowList(mapValue(doc).Get("rows")) {
			if _, ok := r["_source_file"]; !ok {
				r["_source_file"] = name
			}
			out.Rows = append(out.Rows, r)
		}
	}
	return out, ""
}

func LoadData(path string) (*Data, string) {
	info, err := os.Stat(path)
	if err == nil && info.IsDir() {
		return LoadDataDir(path)
	}
	return nil, fmt.Sprintf("missing data directory: %s", path)
}

func readJSON(path string) (any, string) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Sprintf("cannot parse %s: %v", filepath.Base(path), err)
	}
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return nil, fmt.Sprintf("cannot parse %s: %v", filepath.Base(path), err)
	}
	return v, ""
}

type object map[string]any

func (o object) Get(k string) any { return o[k] }

func mapValue(v any) object {
	if m, ok := v.(map[string]any); ok {
		return object(m)
	}
	return object{}
}

func mapList(v any) []map[string]any {
	raw, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]map[string]any, 0, len(raw))
	for _, it := range raw {
		if m, ok := it.(map[string]any); ok {
			out = append(out, m)
		}
	}
	return out
}

func rowList(v any) []Row {
	raw, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]Row, 0, len(raw))
	for _, it := range raw {
		if m, ok := it.(map[string]any); ok {
			out = append(out, Row(m))
		}
	}
	return out
}

func managedContextSLOList(v any) []ManagedContextSLO {
	raw, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]ManagedContextSLO, 0, len(raw))
	for _, it := range raw {
		m, ok := it.(map[string]any)
		if !ok {
			continue
		}
		slo := ManagedContextSLO{
			ID:         stringMapValue(m, "id"),
			Name:       stringMapValue(m, "name"),
			Area:       stringMapValue(m, "area"),
			Status:     stringMapValue(m, "status"),
			Source:     stringMapValue(m, "source"),
			Detail:     stringMapValue(m, "detail"),
			NextAction: stringMapValue(m, "next_action"),
		}
		if hard, ok := m["hard"].(bool); ok {
			slo.Hard = hard
		}
		out = append(out, slo)
	}
	return out
}

func boolToInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func ParseClaimsCatalog(text string) ([]Section, map[string]map[string]bool) {
	var catalog []Section
	tags := map[string]map[string]bool{}
	cur := ""
	tagRe := regexp.MustCompile(`^\s*-\s*\[(SHIPPED|SIMULATED|STUB)\]`)
	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(line, "## ") {
			header := strings.TrimSpace(line[3:])
			n := NormSection(header)
			cur = n
			if n != "" && !nonConcepts[n] && !containsSection(catalog, n) {
				catalog = append(catalog, Section{Section: header, Norm: n})
			}
			continue
		}
		if cur != "" {
			if m := tagRe.FindStringSubmatch(line); m != nil {
				if tags[cur] == nil {
					tags[cur] = map[string]bool{}
				}
				tags[cur][m[1]] = true
			}
		}
	}
	return catalog, tags
}

func LoadTree(root string) Tree {
	claimsText := readText(filepath.Join(root, ClaimsRel))
	catalog, tags := ParseClaimsCatalog(claimsText)
	cmdDirs := map[string]bool{}
	if entries, err := os.ReadDir(filepath.Join(root, "cmd")); err == nil {
		for _, ent := range entries {
			if ent.IsDir() {
				cmdDirs[ent.Name()] = true
			}
		}
	}
	return Tree{
		Catalog: catalog, SectionTags: tags, CmdDirs: cmdDirs,
		DocVerbs: wordSet(readText(filepath.Join(root, CLIRefRel))),
		Exists: func(p string) bool {
			if strings.TrimSpace(p) == "" {
				return false
			}
			_, err := os.Stat(filepath.Join(root, filepath.FromSlash(p)))
			return err == nil
		},
	}
}

func Collect(workspace string, dataPath string) Payload {
	root, err := filepath.Abs(workspace)
	if err == nil {
		workspace = root
	}
	if dataPath == "" {
		dataPath = filepath.Join(workspace, DataDirRel)
	}
	data, loadErr := LoadData(dataPath)
	return BuildPayload(workspace, data, LoadTree(workspace), loadErr)
}

func readText(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return string(b)
}

func wordSet(text string) map[string]bool {
	re := regexp.MustCompile(`[a-z0-9-]+`)
	out := map[string]bool{}
	for _, w := range re.FindAllString(strings.ToLower(text), -1) {
		out[w] = true
	}
	return out
}

func containsSection(catalog []Section, norm string) bool {
	for _, c := range catalog {
		if c.Norm == norm {
			return true
		}
	}
	return false
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func intValue(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	case float32:
		return int(n)
	default:
		return 0
	}
}

func floatValue(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case float32:
		return float64(n)
	case int:
		return float64(n)
	case int64:
		return float64(n)
	default:
		return 0
	}
}

func stringMapValue(m map[string]any, k string) string {
	if s, ok := m[k].(string); ok {
		return s
	}
	return ""
}

func anyIntMap(m map[string]int) map[string]any {
	out := map[string]any{}
	for k, v := range m {
		out[k] = v
	}
	return out
}

func rankOf(v string) int {
	if r, ok := verdictRank[v]; ok {
		return r
	}
	return 9
}
