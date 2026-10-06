package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/projectassets"
	"github.com/anthony-chaudhary/fak/pkg/fakclient"
)

// `fak pi config --from-router` sources provider "fak"'s model catalog from the
// router's GET /v1/models instead of one hand-named id, so models.json lists
// exactly what the router serves at the window it serves it.
//
// Alias rule (the contract the tests pin):
//  1. A row whose `alias_of` (or OpenAI-style `root`) names another advertised id
//     collapses onto that id, followed transitively.
//  2. An unmarked row collapses onto an earlier unmarked row with the same
//     `owned_by` backend AND the same model leaf (case-insensitive text after the
//     last '/'), so `org/model` and `model` on one backend are one entry.
//  3. Every other id is canonical and kept in router order. owned_by alone never
//     merges rows: one backend legitimately serves several models.
//
// A canonical model's served window is the smallest positive window advertised by
// it or any id collapsed onto it (context_window, context_length,
// max_context_length). An unadvertised window falls back to the smallest window
// the catalog advertises, never the larger default prior. The written
// contextWindow is the safe 50% resident target of that window, the same doctrine
// every other Pi writer follows (projectassets.PiSafeContextBudget).

var (
	errPiRouterEmptyCatalog  = errors.New("router advertised no models")
	errPiRouterUnauthorized  = errors.New("router refused the catalog request (gateway key required or rejected)")
	errPiRouterCatalog       = errors.New("router catalog unavailable")
	errPiConfigMalformed     = errors.New("existing Pi config is not a JSON object")
	errPiConfigProviderShape = errors.New("existing Pi provider block is not a JSON object")
)

// piConfigRouterKey resolves the gateway credential for the router origin. It is a
// variable so tests drive the auth header without reading operator credentials.
var piConfigRouterKey = func(origin string) string {
	key, _ := piRouterCredential(origin)
	return key
}

// piConfigNow stamps backup file names; a variable so tests can pin it.
var piConfigNow = time.Now

const piRouterCatalogTimeout = 5 * time.Second

// piFromRouterFlag is an optional-value flag: bare `--from-router` uses the
// default router URL, `--from-router=URL` (or `--from-router URL`, folded by
// foldPiFromRouterArg) names one.
type piFromRouterFlag struct {
	set bool
	url string
}

func (f *piFromRouterFlag) String() string {
	if f == nil {
		return ""
	}
	return f.url
}

func (f *piFromRouterFlag) Set(v string) error {
	v = strings.TrimSpace(v)
	switch v {
	case "false":
		f.set, f.url = false, ""
		return nil
	case "true":
		v = ""
	}
	f.set, f.url = true, v
	return nil
}

func (f *piFromRouterFlag) IsBoolFlag() bool { return true }

// foldPiFromRouterArg rewrites `--from-router URL` to `--from-router=URL` so the
// bool-style flag can take an optional separate value without stopping flag parsing.
func foldPiFromRouterArg(argv []string) []string {
	out := make([]string, 0, len(argv))
	for i := 0; i < len(argv); i++ {
		a := argv[i]
		if a == "--" {
			return append(out, argv[i:]...)
		}
		if (a == "--from-router" || a == "-from-router") && i+1 < len(argv) && !strings.HasPrefix(argv[i+1], "-") {
			out = append(out, a+"="+argv[i+1])
			i++
			continue
		}
		out = append(out, a)
	}
	return out
}

type piRouterRow struct {
	ID      string
	OwnedBy string
	Window  int
	AliasOf string
	Default bool
}

type piRouterModel struct {
	ID      string
	OwnedBy string
	Window  int
	Aliases []string
	Default bool
}

func fetchPiRouterCatalog(baseURL string, timeout time.Duration) ([]piRouterRow, string, error) {
	client := &http.Client{Timeout: timeout}
	rows, err := fetchPiRouterCatalogOnce(client, baseURL)
	if err == nil {
		return rows, baseURL, nil
	}
	if !errors.Is(err, errPiRouterUnauthorized) && !errors.Is(err, errPiRouterEmptyCatalog) {
		if fallback, ok := fakclient.LoopbackFallbackURL(baseURL); ok {
			if fRows, fErr := fetchPiRouterCatalogOnce(client, fallback); fErr == nil {
				return fRows, fallback, nil
			}
		}
	}
	return nil, baseURL, err
}

func fetchPiRouterCatalogOnce(client *http.Client, baseURL string) ([]piRouterRow, error) {
	modelsURL := strings.TrimRight(baseURL, "/") + "/models"
	req, err := http.NewRequest(http.MethodGet, modelsURL, nil)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", errPiRouterCatalog, err)
	}
	if key := strings.TrimSpace(piConfigRouterKey(baseURL)); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%w: GET %s: %v", errPiRouterCatalog, modelsURL, err)
	}
	defer resp.Body.Close()
	switch {
	case resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden:
		return nil, fmt.Errorf("%w: GET %s: HTTP %d", errPiRouterUnauthorized, modelsURL, resp.StatusCode)
	case resp.StatusCode != http.StatusOK:
		return nil, fmt.Errorf("%w: GET %s: HTTP %d", errPiRouterCatalog, modelsURL, resp.StatusCode)
	}
	var doc struct {
		Data []struct {
			ID               string `json:"id"`
			OwnedBy          string `json:"owned_by"`
			ContextWindow    int    `json:"context_window"`
			ContextLength    int    `json:"context_length"`
			MaxContextLength int    `json:"max_context_length"`
			AliasOf          string `json:"alias_of"`
			Root             string `json:"root"`
			Default          bool   `json:"default"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 8<<20)).Decode(&doc); err != nil {
		return nil, fmt.Errorf("%w: decode %s: %v", errPiRouterCatalog, modelsURL, err)
	}
	rows := make([]piRouterRow, 0, len(doc.Data))
	for _, d := range doc.Data {
		id := strings.TrimSpace(d.ID)
		if id == "" {
			continue
		}
		alias := strings.TrimSpace(d.AliasOf)
		if alias == "" {
			alias = strings.TrimSpace(d.Root)
		}
		rows = append(rows, piRouterRow{
			ID:      id,
			OwnedBy: strings.TrimSpace(d.OwnedBy),
			Window:  minPositive(minPositive(d.ContextWindow, d.ContextLength), d.MaxContextLength),
			AliasOf: alias,
			Default: d.Default,
		})
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("%w: GET %s", errPiRouterEmptyCatalog, modelsURL)
	}
	return rows, nil
}

func minPositive(a, b int) int {
	switch {
	case a <= 0:
		return max(b, 0)
	case b <= 0:
		return a
	default:
		return min(a, b)
	}
}

// collapsePiRouterCatalog applies the alias rule documented at the top of this file.
func collapsePiRouterCatalog(rows []piRouterRow) []piRouterModel {
	index := map[string]int{}
	uniq := make([]piRouterRow, 0, len(rows))
	for _, r := range rows {
		r.ID = strings.TrimSpace(r.ID)
		if r.ID == "" {
			continue
		}
		if _, dup := index[r.ID]; dup {
			continue
		}
		index[r.ID] = len(uniq)
		uniq = append(uniq, r)
	}
	marker := func(i int) (int, bool) {
		t := strings.TrimSpace(uniq[i].AliasOf)
		if t == "" || t == uniq[i].ID {
			return 0, false
		}
		j, ok := index[t]
		return j, ok
	}
	canon := make([]string, len(uniq))
	byLeaf := map[string]string{}
	for i, r := range uniq {
		if _, marked := marker(i); marked {
			continue
		}
		key := strings.ToLower(r.OwnedBy) + "\x00" + strings.ToLower(piModelLeaf(r.ID))
		if first, ok := byLeaf[key]; ok {
			canon[i] = first
			continue
		}
		byLeaf[key] = r.ID
		canon[i] = r.ID
	}
	// Follow alias markers; a marker cycle resolves to its earliest-listed member.
	for i := range uniq {
		if canon[i] != "" {
			continue
		}
		var path []int
		onPath := map[int]bool{}
		cur, result := i, ""
		for {
			if canon[cur] != "" {
				result = canon[cur]
				break
			}
			if onPath[cur] {
				k := 0
				for path[k] != cur {
					k++
				}
				first := cur
				for _, p := range path[k:] {
					first = min(first, p)
				}
				result = uniq[first].ID
				break
			}
			onPath[cur] = true
			path = append(path, cur)
			j, ok := marker(cur)
			if !ok {
				result = uniq[cur].ID
				break
			}
			cur = j
		}
		for _, p := range path {
			canon[p] = result
		}
	}
	groups := map[string]*piRouterModel{}
	var order []string
	group := func(id string) *piRouterModel {
		if g, ok := groups[id]; ok {
			return g
		}
		r := uniq[index[id]]
		g := &piRouterModel{ID: id, OwnedBy: r.OwnedBy}
		groups[id] = g
		order = append(order, id)
		return g
	}
	for i, r := range uniq {
		if canon[i] == r.ID {
			group(r.ID)
		}
	}
	for i, r := range uniq {
		g := group(canon[i])
		if canon[i] != r.ID {
			g.Aliases = append(g.Aliases, r.ID)
		}
		g.Window = minPositive(g.Window, r.Window)
		g.Default = g.Default || r.Default
	}
	out := make([]piRouterModel, 0, len(order))
	for _, id := range order {
		out = append(out, *groups[id])
	}
	return out
}

func piModelLeaf(id string) string {
	if i := strings.LastIndex(id, "/"); i >= 0 {
		return id[i+1:]
	}
	return id
}

// piRouterBudget derives the safe envelope for one router model. An unadvertised
// window uses the smallest window the catalog advertises (floor) when that is
// below the default prior, so the default is never larger than what the router
// advertises.
func piRouterBudget(window, floor int) projectassets.PiContextBudget {
	served := window
	if served <= 0 {
		served = projectassets.DefaultPiServedWindow
		if floor > 0 && floor < served {
			served = floor
		}
	}
	b := projectassets.PiSafeContextBudget(served)
	if b.ServedWindow > served {
		target := max(served/2, 1)
		b = projectassets.PiContextBudget{
			ServedWindow:     served,
			ResidentTarget:   target,
			OutputReserve:    max(target/4, 1),
			KeepRecentTokens: max(target/2, 1),
			Provenance:       projectassets.PiBudgetProvenance,
		}
	}
	return b
}

type piRouterChange struct {
	ID           string
	FromWindow   int
	ToWindow     int
	FromMaxToken int
	ToMaxToken   int
}

type piRouterDefaultPlan struct {
	SettingsPath string
	Provider     string
	Model        string
	Skipped      string
	Served       bool
	Pick         string
	Reason       string
}

type piRouterPlan struct {
	RouterURL  string
	ConfigPath string
	Models     []piRouterModel
	Budgets    map[string]projectassets.PiContextBudget
	Added      []string
	Removed    []string
	Changed    []piRouterChange
	Kept       []string
	Exists     bool
	Original   []byte
	Rendered   []byte
	Default    piRouterDefaultPlan
	Warnings   []string
}

func (p *piRouterPlan) configChanged() bool {
	return !p.Exists || !bytes.Equal(p.Original, p.Rendered)
}

func buildPiRouterPlan(rows []piRouterRow, routerURL, configTarget, settingsTarget string) (*piRouterPlan, error) {
	models := collapsePiRouterCatalog(rows)
	if len(models) == 0 {
		return nil, errPiRouterEmptyCatalog
	}
	floor := 0
	for _, m := range models {
		floor = minPositive(floor, m.Window)
	}
	plan := &piRouterPlan{
		RouterURL:  routerURL,
		ConfigPath: projectassets.ResolvePiConfigPath(configTarget),
		Models:     models,
		Budgets:    make(map[string]projectassets.PiContextBudget, len(models)),
	}
	for _, m := range models {
		plan.Budgets[m.ID] = piRouterBudget(m.Window, floor)
	}
	data, err := os.ReadFile(plan.ConfigPath)
	switch {
	case os.IsNotExist(err):
		fresh, genErr := renderFreshPiRouterConfig(plan)
		if genErr != nil {
			return nil, genErr
		}
		plan.Rendered = fresh
		for _, m := range models {
			plan.Added = append(plan.Added, m.ID)
		}
	case err != nil:
		return nil, fmt.Errorf("read %s: %w", plan.ConfigPath, err)
	default:
		plan.Exists = true
		plan.Original = data
		if err := renderPiRouterConfig(plan, data); err != nil {
			return nil, fmt.Errorf("%s: %w", plan.ConfigPath, err)
		}
	}
	def, warnings, err := planPiRouterDefault(settingsTarget, plan)
	if err != nil {
		return nil, err
	}
	plan.Default = def
	plan.Warnings = warnings
	return plan, nil
}

func piRouterProviderBlock(plan *piRouterPlan) map[string]interface{} {
	entries := make([]interface{}, 0, len(plan.Models))
	for _, m := range plan.Models {
		entries = append(entries, projectassets.PiModelEntry(m.ID, plan.Budgets[m.ID]))
	}
	return map[string]interface{}{
		"baseUrl": plan.RouterURL,
		"apiKey":  "fak",
		"api":     "openai-completions",
		"models":  entries,
	}
}

func renderFreshPiRouterConfig(plan *piRouterPlan) ([]byte, error) {
	out, err := json.MarshalIndent(map[string]interface{}{
		"providers": map[string]interface{}{projectassets.DefaultPiProviderID: piRouterProviderBlock(plan)},
	}, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(out, '\n'), nil
}

// renderPiRouterConfig edits only providers.fak.models (and a missing baseUrl) in
// place, so every other byte of the file — other providers, unknown fields, key
// order, formatting — is preserved. An unchanged catalog leaves Rendered == data.
func renderPiRouterConfig(plan *piRouterPlan, data []byte) error {
	bom := []byte{0xEF, 0xBB, 0xBF}
	hasBOM := bytes.HasPrefix(data, bom)
	doc := bytes.TrimPrefix(data, bom)
	root, ok := jsonRootObject(doc)
	if !ok {
		return errPiConfigMalformed
	}
	unit := detectJSONIndentUnit(doc)
	fakID := projectassets.DefaultPiProviderID

	provSpan, found, err := jsonMemberSpan(doc, root, "providers")
	if err != nil {
		return err
	}
	if !found {
		for _, m := range plan.Models {
			plan.Added = append(plan.Added, m.ID)
		}
		value, err := marshalPiValue(map[string]interface{}{fakID: piRouterProviderBlock(plan)}, memberIndentFor(doc, root, unit), unit)
		if err != nil {
			return err
		}
		plan.Rendered = withBOM(hasBOM, jsonInsertMember(doc, root, "providers", value, unit))
		return nil
	}
	if doc[provSpan.start] != '{' {
		return errPiConfigProviderShape
	}
	fakSpan, found, err := jsonMemberSpan(doc, provSpan, fakID)
	if err != nil {
		return err
	}
	if !found {
		for _, m := range plan.Models {
			plan.Added = append(plan.Added, m.ID)
		}
		value, err := marshalPiValue(piRouterProviderBlock(plan), memberIndentFor(doc, provSpan, unit), unit)
		if err != nil {
			return err
		}
		plan.Rendered = withBOM(hasBOM, jsonInsertMember(doc, provSpan, fakID, value, unit))
		return nil
	}
	if doc[fakSpan.start] != '{' {
		return errPiConfigProviderShape
	}

	modelsSpan, hasModels, err := jsonMemberSpan(doc, fakSpan, "models")
	if err != nil {
		return err
	}
	memberIndent := memberIndentFor(doc, fakSpan, unit)
	elemIndent := memberIndent + unit
	var elems []jsonSpan
	if hasModels && doc[modelsSpan.start] == '[' {
		elems, err = jsonArrayElements(doc, modelsSpan)
		if err != nil {
			return err
		}
		if len(elems) > 0 && lineStartsAt(doc, elems[0].start) {
			elemIndent = lineIndent(doc, elems[0].start)
		}
	}

	wanted := make(map[string]bool, len(plan.Models))
	for _, m := range plan.Models {
		wanted[m.ID] = true
	}
	seen := map[string]bool{}
	var entries [][]byte
	for _, e := range elems {
		raw := doc[e.start:e.end]
		var head struct {
			ID string `json:"id"`
		}
		_ = json.Unmarshal(raw, &head)
		id := head.ID
		if !wanted[id] || seen[id] || raw[0] != '{' {
			if id == "" {
				id = "(entry without id)"
			}
			plan.Removed = append(plan.Removed, id)
			continue
		}
		seen[id] = true
		patched, change, err := patchPiModelBudget(raw, plan.Budgets[id], elemIndent+unit, unit)
		if err != nil {
			return err
		}
		if change != nil {
			change.ID = id
			plan.Changed = append(plan.Changed, *change)
		} else {
			plan.Kept = append(plan.Kept, id)
		}
		entries = append(entries, patched)
	}
	for _, m := range plan.Models {
		if seen[m.ID] {
			continue
		}
		value, err := marshalPiValue(projectassets.PiModelEntry(m.ID, plan.Budgets[m.ID]), elemIndent, unit)
		if err != nil {
			return err
		}
		plan.Added = append(plan.Added, m.ID)
		entries = append(entries, value)
	}

	out := doc
	if len(plan.Added) > 0 || len(plan.Removed) > 0 || len(plan.Changed) > 0 {
		var arr bytes.Buffer
		arr.WriteString("[")
		for i, e := range entries {
			if i > 0 {
				arr.WriteString(",")
			}
			arr.WriteString("\n" + elemIndent)
			arr.Write(e)
		}
		arr.WriteString("\n" + memberIndent + "]")
		if hasModels {
			out = jsonReplace(out, modelsSpan, arr.Bytes())
		} else {
			out = jsonInsertMember(out, fakSpan, "models", arr.Bytes(), unit)
		}
	}
	// A provider block with no baseUrl cannot reach the router; fill only that gap.
	root, _ = jsonRootObject(out)
	provSpan, _, _ = jsonMemberSpan(out, root, "providers")
	fakSpan, _, _ = jsonMemberSpan(out, provSpan, fakID)
	if _, hasBase, err := jsonMemberSpan(out, fakSpan, "baseUrl"); err == nil && !hasBase {
		q, _ := json.Marshal(plan.RouterURL)
		out = jsonInsertMember(out, fakSpan, "baseUrl", q, unit)
	}
	plan.Rendered = withBOM(hasBOM, out)
	return nil
}

func withBOM(has bool, doc []byte) []byte {
	if !has {
		return doc
	}
	return append([]byte{0xEF, 0xBB, 0xBF}, doc...)
}

// patchPiModelBudget rewrites only contextWindow/maxTokens inside one existing
// model entry, with the same direction rules as projectassets' budget repair:
// contextWindow must equal the target; maxTokens must be in (0, OutputReserve].
func patchPiModelBudget(entry []byte, budget projectassets.PiContextBudget, memberIndent, unit string) ([]byte, *piRouterChange, error) {
	obj := jsonSpan{0, len(entry)}
	change := &piRouterChange{ToWindow: budget.ResidentTarget}
	changed := false
	out := entry
	cwSpan, hasCW, err := jsonMemberSpan(out, obj, "contextWindow")
	if err != nil {
		return nil, nil, err
	}
	cur, numeric := 0, false
	if hasCW {
		cur, numeric = piJSONInt(out[cwSpan.start:cwSpan.end])
	}
	change.FromWindow = cur
	if !numeric || cur != budget.ResidentTarget {
		val := []byte(strconv.Itoa(budget.ResidentTarget))
		if hasCW {
			out = jsonReplace(out, cwSpan, val)
		} else {
			out = jsonInsertMemberIndent(out, jsonSpan{0, len(out)}, "contextWindow", val, memberIndent)
		}
		changed = true
	}
	obj = jsonSpan{0, len(out)}
	mtSpan, hasMT, err := jsonMemberSpan(out, obj, "maxTokens")
	if err != nil {
		return nil, nil, err
	}
	mt, mtNumeric := 0, false
	if hasMT {
		mt, mtNumeric = piJSONInt(out[mtSpan.start:mtSpan.end])
	}
	change.FromMaxToken, change.ToMaxToken = mt, mt
	if !mtNumeric || mt <= 0 || mt > budget.OutputReserve {
		val := []byte(strconv.Itoa(budget.OutputReserve))
		if hasMT {
			out = jsonReplace(out, mtSpan, val)
		} else {
			out = jsonInsertMemberIndent(out, jsonSpan{0, len(out)}, "maxTokens", val, memberIndent)
		}
		change.ToMaxToken = budget.OutputReserve
		changed = true
	}
	if !changed {
		return entry, nil, nil
	}
	return out, change, nil
}

func piJSONInt(raw []byte) (int, bool) {
	var f float64
	if err := json.Unmarshal(raw, &f); err != nil {
		return 0, false
	}
	return int(f), true
}

func planPiRouterDefault(settingsTarget string, plan *piRouterPlan) (piRouterDefaultPlan, []string, error) {
	path := projectassets.ResolvePiSettingsPath(settingsTarget)
	def := piRouterDefaultPlan{SettingsPath: path}
	raw := map[string]interface{}{}
	data, err := os.ReadFile(path)
	switch {
	case os.IsNotExist(err):
	case err != nil:
		return def, nil, fmt.Errorf("read %s: %w", path, err)
	default:
		if err := json.Unmarshal(bytes.TrimPrefix(data, []byte{0xEF, 0xBB, 0xBF}), &raw); err != nil {
			return def, nil, fmt.Errorf("parse existing %s: %w", path, err)
		}
	}
	def.Provider, _ = raw["defaultProvider"].(string)
	def.Model, _ = raw["defaultModel"].(string)
	def.Provider, def.Model = strings.TrimSpace(def.Provider), strings.TrimSpace(def.Model)

	var warnings []string
	if block, ok := raw["compaction"].(map[string]interface{}); ok {
		smallest := 0
		for _, b := range plan.Budgets {
			smallest = minPositive(smallest, b.ResidentTarget)
		}
		if keep, ok := block["keepRecentTokens"].(float64); ok && smallest > 0 && int(keep) >= smallest {
			warnings = append(warnings, fmt.Sprintf("%s compaction.keepRecentTokens %d >= smallest planned contextWindow %d; lower it so compaction can make room", path, int(keep), smallest))
		}
	}

	if def.Provider != "" && def.Provider != projectassets.DefaultPiProviderID {
		def.Skipped = fmt.Sprintf("defaultProvider is %q, not %q", def.Provider, projectassets.DefaultPiProviderID)
		return def, warnings, nil
	}
	canonicalOf := map[string]string{}
	for _, m := range plan.Models {
		canonicalOf[m.ID] = m.ID
		for _, a := range m.Aliases {
			canonicalOf[a] = m.ID
		}
	}
	switch c, ok := canonicalOf[def.Model]; {
	case ok && c == def.Model:
		def.Served = true
	case ok:
		def.Pick, def.Reason = c, fmt.Sprintf("defaultModel %q is a router alias of %q, which is the id the catalog lists", def.Model, c)
	case def.Model == "":
		def.Pick, def.Reason = piRouterDefaultModel(plan.Models), "no defaultModel is set"
	default:
		def.Pick, def.Reason = piRouterDefaultModel(plan.Models), fmt.Sprintf("defaultModel %q is not served by the router", def.Model)
	}
	return def, warnings, nil
}

// piRouterDefaultModel is the router's default route: a row it marks `default`,
// else the first canonical id (a routing table lists its declared models before
// aliases, in declaration order, so the first one is its primary route).
func piRouterDefaultModel(models []piRouterModel) string {
	for _, m := range models {
		if m.Default {
			return m.ID
		}
	}
	return models[0].ID
}

func runPiConfigFromRouter(stdout, stderr io.Writer, fs *flag.FlagSet, explicitURL, addr, configTarget, settingsTarget string, write bool) int {
	conflict := ""
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "model" || f.Name == "window" {
			conflict = f.Name
		}
	})
	if conflict != "" {
		fmt.Fprintf(stderr, "fak pi config: --%s conflicts with --from-router (the router catalog names the models and windows)\n", conflict)
		return 2
	}
	routerURL := strings.TrimSpace(explicitURL)
	if routerURL == "" {
		routerURL = configuredPiProviderBaseURL(configTarget)
	}
	if routerURL == "" {
		routerURL = addr
	}
	routerURL = projectassets.NormalizePiBaseURL(routerURL)

	rows, resolved, err := fetchPiRouterCatalog(routerURL, piRouterCatalogTimeout)
	if err != nil {
		fmt.Fprintf(stderr, "fak pi config: %v\n", err)
		return 1
	}
	plan, err := buildPiRouterPlan(rows, resolved, configTarget, settingsTarget)
	if err != nil {
		fmt.Fprintf(stderr, "fak pi config: %v\n", err)
		return 1
	}
	printPiRouterPlan(stdout, plan, len(rows), write)
	if !write {
		return 0
	}
	if plan.configChanged() {
		backup, err := writePiConfigWithBackup(plan.ConfigPath, plan.Original, plan.Exists, plan.Rendered)
		if err != nil {
			fmt.Fprintf(stderr, "fak pi config: %v\n", err)
			return 1
		}
		if backup != "" {
			fmt.Fprintf(stdout, "fak pi config: wrote %s (backup %s)\n", plan.ConfigPath, backup)
		} else {
			fmt.Fprintf(stdout, "fak pi config: wrote %s\n", plan.ConfigPath)
		}
	}
	if plan.Default.Pick != "" {
		backup := ""
		if orig, err := os.ReadFile(plan.Default.SettingsPath); err == nil {
			backup, err = writePiBackup(plan.Default.SettingsPath, orig)
			if err != nil {
				fmt.Fprintf(stderr, "fak pi config: %v\n", err)
				return 1
			}
		}
		dPath, _, err := projectassets.EnsurePiDefaultProviderModel(plan.Default.SettingsPath, projectassets.DefaultPiProviderID, plan.Default.Pick)
		if err != nil {
			fmt.Fprintf(stderr, "fak pi config: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "fak pi config: pinned Pi default to provider %q model %q in %s", projectassets.DefaultPiProviderID, plan.Default.Pick, dPath)
		if backup != "" {
			fmt.Fprintf(stdout, " (backup %s)", backup)
		}
		fmt.Fprintln(stdout)
	}
	return 0
}

func printPiRouterPlan(w io.Writer, plan *piRouterPlan, advertised int, write bool) {
	fmt.Fprintf(w, "fak pi config: router %s advertises %d ids -> %d models for provider %q (%s)\n", plan.RouterURL, advertised, len(plan.Models), projectassets.DefaultPiProviderID, plan.ConfigPath)
	for _, m := range plan.Models {
		b := plan.Budgets[m.ID]
		src := "router window"
		if m.Window <= 0 {
			src = "no router window; conservative"
		}
		fmt.Fprintf(w, "  model   %s  owned_by=%s  contextWindow=%d (%s %d)\n", m.ID, m.OwnedBy, b.ResidentTarget, src, b.ServedWindow)
		if len(m.Aliases) > 0 {
			aliases := append([]string(nil), m.Aliases...)
			sort.Strings(aliases)
			fmt.Fprintf(w, "          aliases collapsed: %s\n", strings.Join(aliases, ", "))
		}
	}
	for _, id := range plan.Added {
		fmt.Fprintf(w, "  + add     %s  contextWindow=%d\n", id, plan.Budgets[id].ResidentTarget)
	}
	for _, id := range plan.Removed {
		fmt.Fprintf(w, "  - remove  %s  (not a canonical router model)\n", id)
	}
	for _, c := range plan.Changed {
		line := fmt.Sprintf("  ~ change  %s  contextWindow %d -> %d", c.ID, c.FromWindow, c.ToWindow)
		if c.FromMaxToken != c.ToMaxToken {
			line += fmt.Sprintf(", maxTokens %d -> %d", c.FromMaxToken, c.ToMaxToken)
		}
		fmt.Fprintln(w, line)
	}
	d := plan.Default
	switch {
	case d.Skipped != "":
		fmt.Fprintf(w, "  default %s: not checked (%s)\n", d.SettingsPath, d.Skipped)
	case d.Served:
		fmt.Fprintf(w, "  default %s: defaultModel %q is served\n", d.SettingsPath, d.Model)
	default:
		action := "rerun with --write to pin it"
		if write {
			action = "pinning"
		}
		fmt.Fprintf(w, "  default %s: %s -> %q (%s)\n", d.SettingsPath, d.Reason, d.Pick, action)
	}
	for _, warn := range plan.Warnings {
		fmt.Fprintf(w, "  warning %s\n", warn)
	}
	if !plan.configChanged() {
		fmt.Fprintf(w, "fak pi config: no model changes; %s already matches the router\n", plan.ConfigPath)
		return
	}
	verb := "rerun with --write to apply"
	if write {
		verb = "applying"
	}
	fmt.Fprintf(w, "fak pi config: plan: %d added, %d removed, %d changed; %s\n", len(plan.Added), len(plan.Removed), len(plan.Changed), verb)
}

// writePiConfigWithBackup backs up the existing file next to itself, then
// replaces it through a temp file + rename so a crash never leaves it half written.
func writePiConfigWithBackup(path string, original []byte, exists bool, rendered []byte) (string, error) {
	backup := ""
	if exists {
		b, err := writePiBackup(path, original)
		if err != nil {
			return "", err
		}
		backup = b
	} else if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", fmt.Errorf("mkdir %s: %w", filepath.Dir(path), err)
	}
	tmp := path + ".tmp-" + strconv.Itoa(os.Getpid())
	if err := os.WriteFile(tmp, rendered, 0o644); err != nil {
		return backup, fmt.Errorf("write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return backup, fmt.Errorf("replace %s: %w", path, err)
	}
	return backup, nil
}

func writePiBackup(path string, original []byte) (string, error) {
	stamp := piConfigNow().UTC().Format("20060102T150405Z")
	backup := path + ".bak-" + stamp
	for n := 1; ; n++ {
		if _, err := os.Stat(backup); os.IsNotExist(err) {
			break
		}
		backup = path + ".bak-" + stamp + "-" + strconv.Itoa(n)
	}
	if err := os.WriteFile(backup, original, 0o644); err != nil {
		return "", fmt.Errorf("backup %s: %w", backup, err)
	}
	return backup, nil
}

// --- byte-preserving JSON editing -------------------------------------------

type jsonSpan struct{ start, end int }

func jsonRootObject(doc []byte) (jsonSpan, bool) {
	if !json.Valid(doc) {
		return jsonSpan{}, false
	}
	start := len(doc) - len(bytes.TrimLeft(doc, " \t\r\n"))
	end := len(bytes.TrimRight(doc, " \t\r\n"))
	if start >= end || doc[start] != '{' {
		return jsonSpan{}, false
	}
	return jsonSpan{start, end}, true
}

type jsonMember struct {
	keyStart int
	key      string
	value    jsonSpan
}

// jsonObjectMembers lists the members of the object at obj, with absolute offsets.
func jsonObjectMembers(doc []byte, obj jsonSpan) ([]jsonMember, error) {
	sub := doc[obj.start:obj.end]
	dec := json.NewDecoder(bytes.NewReader(sub))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('{') {
		return nil, errPiConfigMalformed
	}
	var members []jsonMember
	for dec.More() {
		before := int(dec.InputOffset())
		tok, err := dec.Token()
		if err != nil {
			return nil, err
		}
		key, _ := tok.(string)
		keyStart := before + bytes.IndexByte(sub[before:], '"')
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			return nil, err
		}
		end := int(dec.InputOffset())
		start := end - len(raw)
		if start < 0 || !bytes.Equal(sub[start:end], raw) {
			return nil, fmt.Errorf("%w: cannot locate member %q", errPiConfigMalformed, key)
		}
		members = append(members, jsonMember{keyStart: obj.start + keyStart, key: key, value: jsonSpan{obj.start + start, obj.start + end}})
	}
	return members, nil
}

// jsonMemberSpan returns the value span of key (the last occurrence, as JSON.parse
// resolves duplicates).
func jsonMemberSpan(doc []byte, obj jsonSpan, key string) (jsonSpan, bool, error) {
	members, err := jsonObjectMembers(doc, obj)
	if err != nil {
		return jsonSpan{}, false, err
	}
	for i := len(members) - 1; i >= 0; i-- {
		if members[i].key == key {
			return members[i].value, true, nil
		}
	}
	return jsonSpan{}, false, nil
}

func jsonArrayElements(doc []byte, arr jsonSpan) ([]jsonSpan, error) {
	sub := doc[arr.start:arr.end]
	dec := json.NewDecoder(bytes.NewReader(sub))
	if tok, err := dec.Token(); err != nil || tok != json.Delim('[') {
		return nil, errPiConfigMalformed
	}
	var out []jsonSpan
	for dec.More() {
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			return nil, err
		}
		end := int(dec.InputOffset())
		start := end - len(raw)
		if start < 0 || !bytes.Equal(sub[start:end], raw) {
			return nil, fmt.Errorf("%w: cannot locate array element", errPiConfigMalformed)
		}
		out = append(out, jsonSpan{arr.start + start, arr.start + end})
	}
	return out, nil
}

func jsonReplace(doc []byte, span jsonSpan, value []byte) []byte {
	out := make([]byte, 0, len(doc)-(span.end-span.start)+len(value))
	out = append(out, doc[:span.start]...)
	out = append(out, value...)
	return append(out, doc[span.end:]...)
}

// memberIndentFor is the indentation of obj's members: the first member's line
// indent when it starts its own line, else the object's line indent plus unit.
func memberIndentFor(doc []byte, obj jsonSpan, unit string) string {
	members, err := jsonObjectMembers(doc, obj)
	if err == nil && len(members) > 0 && lineStartsAt(doc, members[0].keyStart) {
		return lineIndent(doc, members[0].keyStart)
	}
	return lineIndent(doc, obj.start) + unit
}

func jsonInsertMember(doc []byte, obj jsonSpan, key string, value []byte, unit string) []byte {
	return jsonInsertMemberIndent(doc, obj, key, value, memberIndentFor(doc, obj, unit))
}

func jsonInsertMemberIndent(doc []byte, obj jsonSpan, key string, value []byte, indent string) []byte {
	q, _ := json.Marshal(key)
	members, _ := jsonObjectMembers(doc, obj)
	if len(members) == 0 {
		body := "{\n" + indent + string(q) + ": " + string(value) + "\n" + lineIndent(doc, obj.start) + "}"
		return jsonReplace(doc, obj, []byte(body))
	}
	last := members[len(members)-1].value.end
	insert := ",\n" + indent + string(q) + ": " + string(value)
	return jsonReplace(doc, jsonSpan{last, last}, []byte(insert))
}

func marshalPiValue(v interface{}, prefix, unit string) ([]byte, error) {
	return json.MarshalIndent(v, prefix, unit)
}

func lineStart(doc []byte, pos int) int {
	return bytes.LastIndexByte(doc[:pos], '\n') + 1
}

// lineStartsAt reports whether only whitespace precedes pos on its line.
func lineStartsAt(doc []byte, pos int) bool {
	return len(bytes.Trim(doc[lineStart(doc, pos):pos], " \t")) == 0
}

func lineIndent(doc []byte, pos int) string {
	ls := lineStart(doc, pos)
	i := ls
	for i < len(doc) && (doc[i] == ' ' || doc[i] == '\t') {
		i++
	}
	return string(doc[ls:i])
}

func detectJSONIndentUnit(doc []byte) string {
	for _, line := range bytes.Split(doc, []byte("\n")) {
		trimmed := bytes.TrimLeft(line, " \t")
		if n := len(line) - len(trimmed); n > 0 && len(bytes.TrimSpace(trimmed)) > 0 {
			return string(line[:n])
		}
	}
	return "  "
}
