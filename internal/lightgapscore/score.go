// Package lightgapscore implements fak's deterministic, per-use-case lightgap scorecard.
package lightgapscore

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	Schema      = "fak-lightgap-scorecard/1"
	DebtKey     = "lightgap_debt"
	horizonBeta = 0.999
	epsilon     = 0.001
)

var Horizon = math.Atanh(horizonBeta)

type rung struct {
	Label string
	Floor float64
	Blurb string
}

var ladder = []rung{
	{"AT-CEILING", 3.80, "at the limit -- nothing better can exist on this axis"},
	{"NEAR-C", 2.00, "category-defining; the alternative is not in the running"},
	{"RELATIVISTIC", 1.00, "a real, large, net win worth restructuring for"},
	{"CRUISE", 0.50, "a solid net win; adopt if this facet is what you came for"},
	{"DRIFT", 0.10, "marginally better once effort is counted; easy to regret"},
	{"REST", -0.10, "indistinguishable from doing nothing"},
	{"DRAG", -1.00, "the alternative wins once you count the effort"},
	{"REGRESSIVE", math.Inf(-1), "actively worse than what you already have"},
}
var order = []string{"AT-CEILING", "NEAR-C", "RELATIVISTIC", "CRUISE", "DRIFT", "REST", "DRAG", "REGRESSIVE"}
var provenanceCap = map[string]string{"MODELED": "CRUISE", "PROJECTED": "CRUISE", "OBSERVED": "RELATIVISTIC"}
var ceilingKindCap = map[string]string{"lower-bound": "RELATIVISTIC"}
var validProvenance = set("MEASURED", "WITNESSED", "OBSERVED", "MODELED", "PROJECTED", "NONE")
var validCeilingKinds = set("physical", "definitional", "lower-bound")

func set(xs ...string) map[string]bool {
	m := map[string]bool{}
	for _, x := range xs {
		m[x] = true
	}
	return m
}
func clamp(x float64) float64       { return math.Max(-horizonBeta, math.Min(horizonBeta, x)) }
func Rapidity(beta float64) float64 { return math.Atanh(clamp(beta)) }
func rungFloor(label string) float64 {
	for _, r := range ladder {
		if r.Label == label {
			return r.Floor
		}
	}
	return 0
}
func capValue(cap string) float64 {
	if cap == "" {
		return math.Inf(1)
	}
	for i, v := range order {
		if v == cap {
			if i == 0 {
				return math.Inf(1)
			}
			return rungFloor(order[i-1]) - epsilon
		}
	}
	return math.Inf(1)
}
func verdictFor(w float64) (string, string) {
	for _, r := range ladder {
		if w >= r.Floor {
			return r.Label, r.Blurb
		}
	}
	return ladder[len(ladder)-1].Label, ladder[len(ladder)-1].Blurb
}
func round4(x float64) float64 { return math.Round(x*10000) / 10000 }
func asFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case int:
		return float64(x), true
	case json.Number:
		f, e := x.Float64()
		return f, e == nil
	}
	return 0, false
}
func str(v any) string { s, _ := v.(string); return s }

type Defect struct {
	Code       string `json:"code"`
	Where      string `json:"where"`
	Detail     string `json:"detail"`
	NextAction string `json:"next_action"`
}
type Cell struct {
	Segment      string   `json:"segment"`
	Facet        string   `json:"facet"`
	Weight       float64  `json:"weight"`
	Shape        string   `json:"shape"`
	Mode         string   `json:"mode"`
	Beta         float64  `json:"beta"`
	W            float64  `json:"w"`
	Load         float64  `json:"load"`
	Tau          float64  `json:"tau"`
	WNet         float64  `json:"w_net"`
	WEff         float64  `json:"w_eff"`
	Cap          string   `json:"cap"`
	CapReason    string   `json:"cap_reason"`
	Verdict      string   `json:"verdict"`
	Why          string   `json:"why"`
	FakValue     float64  `json:"fak_value"`
	FakSource    string   `json:"fak_source"`
	Provenance   string   `json:"provenance"`
	AltName      string   `json:"alt_name"`
	AltID        string   `json:"alt_id"`
	AltValue     float64  `json:"alt_value"`
	AltSource    string   `json:"alt_source"`
	AltCostHours float64  `json:"alt_cost_hours"`
	Ceiling      float64  `json:"ceiling"`
	CeilingKind  string   `json:"ceiling_kind"`
	CostHours    float64  `json:"cost_hours"`
	CostBasis    []any    `json:"cost_basis"`
	Note         string   `json:"note"`
	Fence        string   `json:"fence"`
	Defects      []Defect `json:"defects"`
}

func (c Cell) rounded() Cell {
	c.Beta = round4(c.Beta)
	c.W = round4(c.W)
	c.Load = round4(c.Load)
	c.Tau = round4(c.Tau)
	c.WNet = round4(c.WNet)
	c.WEff = round4(c.WEff)
	return c
}

type metaFile struct {
	Schema       string           `json:"schema"`
	Subject      map[string]any   `json:"subject"`
	Bands        []map[string]any `json:"bands"`
	Facets       []map[string]any `json:"facets"`
	Segments     []map[string]any `json:"segments"`
	VerdictRules map[string]any   `json:"verdict_rules"`
}
type alternativesFile struct {
	Alternatives []map[string]any `json:"alternatives"`
}
type ceilingsFile struct {
	Ceilings []map[string]any `json:"ceilings"`
}
type cellsFile struct {
	Cells []map[string]any `json:"cells"`
}

type Scorecard struct {
	Root         string
	Meta         metaFile
	Ceilings     map[string]map[string]any
	Alternatives map[string]map[string]any
	Facets       map[string]map[string]any
	Segments     map[string]map[string]any
	Bands        map[string]map[string]any
	Cells        []Cell
	Defects      []Defect
}

func load(path string, out any) error {
	b, e := os.ReadFile(path)
	if e != nil {
		return e
	}
	if e = json.Unmarshal(b, out); e != nil {
		return fmt.Errorf("%s: %w", filepath.Base(path), e)
	}
	return nil
}
func New(root string) (*Scorecard, error) {
	dir := filepath.Join(root, "tools", "lightgap_scorecard.data")
	s := &Scorecard{Root: root, Ceilings: map[string]map[string]any{}, Alternatives: map[string]map[string]any{}, Facets: map[string]map[string]any{}, Segments: map[string]map[string]any{}, Bands: map[string]map[string]any{}}
	var cf ceilingsFile
	var af alternativesFile
	if e := load(filepath.Join(dir, "_meta.json"), &s.Meta); e != nil {
		return nil, e
	}
	if e := load(filepath.Join(dir, "_ceilings.json"), &cf); e != nil {
		return nil, e
	}
	if e := load(filepath.Join(dir, "_alternatives.json"), &af); e != nil {
		return nil, e
	}
	for _, x := range cf.Ceilings {
		s.Ceilings[str(x["id"])] = x
	}
	for _, x := range af.Alternatives {
		s.Alternatives[str(x["id"])] = x
	}
	for _, x := range s.Meta.Facets {
		s.Facets[str(x["id"])] = x
	}
	for _, x := range s.Meta.Segments {
		s.Segments[str(x["id"])] = x
	}
	for _, x := range s.Meta.Bands {
		s.Bands[str(x["id"])] = x
	}
	if e := s.build(dir); e != nil {
		return nil, e
	}
	return s, nil
}
func (s *Scorecard) add(code, where, detail, next string) {
	s.Defects = append(s.Defects, Defect{code, where, detail, next})
}
func (s *Scorecard) build(dir string) error {
	for _, seg := range s.Meta.Segments {
		sid := str(seg["id"])
		var raw cellsFile
		if e := load(filepath.Join(dir, "cells-"+sid+".json"), &raw); e != nil {
			return e
		}
		seen := map[string]bool{}
		for _, r := range raw.Cells {
			fid := str(r["facet"])
			seen[fid] = true
			s.Cells = append(s.Cells, s.buildCell(sid, fid, r))
		}
		if unrun, ok := seg["unrun"].(map[string]any); ok {
			keys := make([]string, 0, len(unrun))
			for fid := range unrun {
				keys = append(keys, fid)
			}
			sort.Strings(keys)
			for _, fid := range keys {
				u, _ := unrun[fid].(map[string]any)
				seen[fid] = true
				weights, _ := seg["weights"].(map[string]any)
				weight, _ := asFloat(weights[fid])
				d := Defect{"UNCOVERED", sid + "/" + fid, str(u["why"]), str(u["next_action"])}
				s.Defects = append(s.Defects, d)
				s.Cells = append(s.Cells, Cell{Segment: sid, Facet: fid, Weight: weight, Shape: str(s.Facets[fid]["shape"]), Mode: "unrun", Provenance: "NONE", Verdict: "NEVER-MEASURED", Why: "the deciding comparison has not been run", Note: str(u["why"]), Defects: []Defect{d}, CostBasis: []any{}})
			}
		}
		weights, _ := seg["weights"].(map[string]any)
		for _, f := range s.Meta.Facets {
			fid := str(f["id"])
			weight, _ := asFloat(weights[fid])
			if weight > 0 && !seen[fid] {
				s.add("UNCOVERED", sid+"/"+fid, "material facet has no cell", "run the head-to-head, then add the cell")
			}
		}
	}
	s.validate()
	return nil
}
func (s *Scorecard) buildCell(sid, fid string, r map[string]any) Cell {
	where := sid + "/" + fid
	defects := []Defect{}
	add := func(code, detail, next string) {
		d := Defect{code, where, detail, next}
		defects = append(defects, d)
		s.Defects = append(s.Defects, d)
	}
	seg, ok := s.Segments[sid]
	if !ok {
		add("UNKNOWN_SEGMENT", sid, "")
		seg = map[string]any{}
	}
	facet, ok := s.Facets[fid]
	if !ok {
		add("UNKNOWN_FACET", fid, "")
		facet = map[string]any{}
	}
	weights, _ := seg["weights"].(map[string]any)
	weight, _ := asFloat(weights[fid])
	shape := str(r["shape"])
	if shape == "" {
		shape = str(facet["shape"])
	}
	mode := str(r["mode"])
	if mode == "" {
		mode = "scored"
	}
	if mode == "immaterial" {
		return Cell{Segment: sid, Facet: fid, Weight: weight, Shape: shape, Mode: mode, Provenance: str(r["provenance"]), Note: str(r["note"]), Defects: defects, CostBasis: []any{}}
	}
	fak, _ := r["fak"].(map[string]any)
	alternative, _ := r["alt"].(map[string]any)
	costData, _ := r["cost"].(map[string]any)
	ceilID := fid
	ceil, ok := s.Ceilings[ceilID]
	if !ok {
		add("UNKNOWN_CEILING", ceilID, "declare it in _ceilings.json")
		ceil = map[string]any{}
	}
	altID := str(alternative["id"])
	alt, ok := s.Alternatives[altID]
	if !ok {
		add("UNKNOWN_ALTERNATIVE", altID, "declare it in _alternatives.json")
		alt = map[string]any{}
	}
	if str(alt["class"]) == "naive" {
		add("NAIVE_BASELINE", "the alternative is explicitly classed naive", "compare against a tuned or SOTA alternative")
	}
	fakValue, fakOK := asFloat(fak["value"])
	altValue, altOK := asFloat(alternative["value"])
	ceiling, ceilOK := asFloat(ceil["c"])
	kindOverride := ""
	if override, ok := r["ceiling"].(map[string]any); ok {
		if v, yes := asFloat(override["c"]); yes {
			ceiling = v
			ceilOK = true
		}
		kindOverride = str(override["kind"])
		if str(override["derivation"]) == "" {
			add("NO_DERIVATION", "cell overrides c with no derivation", "")
		}
	}
	prov := str(fak["provenance"])
	if !validProvenance[prov] {
		add("BAD_PROVENANCE", fmt.Sprintf("%q", prov), "use a closed provenance token")
	}
	if !fakOK || !altOK || !ceilOK {
		add("MISSING_NUMBER", "fak, alternative, and ceiling must all be numeric", "supply or explicitly mark the cell never-measured")
		fakValue = 0
		altValue = 0
		ceiling = 1
	}
	den := ceiling - altValue
	beta := 0.0
	if pure, _ := r["pure_tax"].(bool); pure {
		mode = "pure_tax"
		if altValue == 0 {
			add("DEGENERATE_CEILING", "pure_tax with N = 0", "")
		} else {
			beta = (fakValue - altValue) / math.Abs(altValue)
		}
		if beta > 0 {
			add("CEILING_BREACHED", "pure_tax cell scores above its ceiling", "")
		}
	} else if math.Abs(den) < 1e-12 {
		add("DEGENERATE_CEILING", "ceiling equals the alternative -- mark pure_tax or fix the anchor", "")
	} else {
		beta = (fakValue - altValue) / den
	}
	w := Rapidity(beta)
	altCost, _ := asFloat(costData["alt_hours"])
	cost, _ := asFloat(costData["hours"])
	tolerance, _ := asFloat(seg["tolerance_hours"])
	if tolerance <= 0 {
		tolerance = 1
		add("BAD_TOLERANCE", "segment tolerance_hours must be positive", "")
	}
	load := (cost - altCost) / tolerance
	tau := Rapidity(load)
	wNet := w - tau
	cap := ""
	capReason := ""
	kind := str(ceil["kind"])
	if kindOverride != "" {
		kind = kindOverride
	}
	if x := provenanceCap[prov]; x != "" {
		cap = x
		capReason = prov + " evidence cannot support a verdict above " + x
	}
	if x := ceilingKindCap[kind]; x != "" && (cap == "" || index(order, x) > index(order, cap)) {
		cap = x
		capReason = "a " + kind + " ceiling cannot support a verdict above " + x
	}
	wEff := math.Min(wNet, capValue(cap))
	verdict, why := verdictFor(wEff)
	if prov == "NONE" {
		add("NEVER_MEASURED", str(r["note"]), str(r["next_action"]))
		verdict = "NEVER-MEASURED"
		why = "the deciding comparison has not been run"
		wEff = 0
	}
	costBasis, _ := costData["basis"].([]any)
	if costBasis == nil {
		costBasis = []any{}
	}
	altSource := str(alternative["source"])
	if altSource == "" {
		altSource = str(alt["source"])
	}
	return Cell{sid, fid, weight, shape, mode, beta, w, load, tau, wNet, wEff, cap, capReason, verdict, why, fakValue, str(fak["source"]), prov, str(alt["name"]), altID, altValue, altSource, altCost, ceiling, kind, cost, costBasis, str(r["note"]), str(r["fence"]), defects}
}
func index(xs []string, x string) int {
	for i, v := range xs {
		if v == x {
			return i
		}
	}
	return len(xs)
}
func (s *Scorecard) validate() {
	for _, c := range s.Cells {
		if c.Mode == "immaterial" {
			continue
		}
		where := c.Segment + "/" + c.Facet
		if !validCeilingKinds[c.CeilingKind] {
			s.add("BAD_CEILING_KIND", where, fmt.Sprintf("%q", c.CeilingKind), "use physical, definitional, or lower-bound")
		}
		if c.CeilingKind == "physical" && c.Provenance == "MEASURED" && !strings.Contains(strings.ToLower(c.FakSource), "bench") {
			s.add("SUSPICIOUS_PHYSICAL_CLAIM", where, "MEASURED physical-ceiling claim has no benchmark-like source", "")
		}
		if c.Weight > 0 && c.Provenance == "NONE" && c.Note == "" {
			s.add("SILENT_UNKNOWN", where, "material never-measured cell has no explanation", "say what comparison is absent")
		}
		if c.Provenance == "MODELED" && c.WEff >= rungFloor("RELATIVISTIC") {
			s.add("MODEL_ESCAPED_CAP", where, "modeled evidence reached RELATIVISTIC", "")
		}
		if c.CeilingKind == "lower-bound" && c.WEff >= rungFloor("NEAR-C") {
			s.add("LOWER_BOUND_ESCAPED_CAP", where, "lower-bound ceiling reached NEAR-C", "")
		}
	}
	for _, seg := range s.Meta.Segments {
		sid := str(seg["id"])
		weights, _ := seg["weights"].(map[string]any)
		total := 0.0
		for _, v := range weights {
			f, _ := asFloat(v)
			total += f
		}
		if math.Abs(total-1) > 1e-6 {
			s.add("WEIGHTS_NOT_ONE", sid, fmt.Sprintf("sum is %.6f", total), "normalize the segment weights")
		}
	}
}
func (s *Scorecard) Debt() int {
	n := 0
	for _, d := range s.Defects {
		if d.Code == "UNCOVERED" {
			n++
		}
	}
	return n
}
func (s *Scorecard) bySegment(sid string) []Cell {
	out := []Cell{}
	for _, c := range s.Cells {
		if c.Segment == sid {
			out = append(out, c)
		}
	}
	return out
}
func (s *Scorecard) byFacet(fid string) []Cell {
	out := []Cell{}
	for _, c := range s.Cells {
		if c.Facet == fid {
			out = append(out, c)
		}
	}
	return out
}
func scored(cs []Cell) []Cell {
	out := []Cell{}
	for _, c := range cs {
		if c.Mode != "immaterial" {
			out = append(out, c)
		}
	}
	return out
}
func known(cs []Cell) []Cell {
	out := []Cell{}
	for _, c := range cs {
		if c.Mode != "immaterial" && c.Provenance != "NONE" {
			out = append(out, c)
		}
	}
	return out
}
func (s *Scorecard) SegmentProfile(sid string) map[string]any {
	seg := s.Segments[sid]
	cs := s.bySegment(sid)
	ks := known(cs)
	sort.Slice(ks, func(i, j int) bool { return ks[i].WEff > ks[j].WEff })
	unknown := []Cell{}
	for _, c := range cs {
		if c.Mode != "immaterial" && c.Provenance == "NONE" {
			unknown = append(unknown, c)
		}
	}
	unknownWeight := 0.0
	for _, c := range unknown {
		unknownWeight += c.Weight
	}
	critical := []string{}
	if a, ok := seg["critical_facets"].([]any); ok {
		for _, x := range a {
			critical = append(critical, str(x))
		}
	}
	vr, _ := s.Meta.VerdictRules["switch_bar"].(map[string]any)
	switchBar, _ := asFloat(vr["w_eff"])
	maxUnknown, _ := asFloat(vr["max_unknown_weight"])
	blockFloor, _ := asFloat(vr["critical_block_floor"])
	blocked := []Cell{}
	for _, c := range ks {
		if contains(critical, c.Facet) && c.Weight > 0 && c.WEff < blockFloor {
			blocked = append(blocked, c)
		}
	}
	verdict := "HOLD"
	reason := "no measured facet clears the switch bar"
	if len(blocked) > 0 {
		verdict = "BLOCKED"
		c := blocked[0]
		reason = fmt.Sprintf("%s is %s at weight %.2f -- a material axis goes backwards, and no pull elsewhere buys it back", c.Facet, c.Verdict, c.Weight)
	} else if unknownWeight > maxUnknown {
		verdict = "UNDECIDABLE"
		ids := []string{}
		for _, c := range unknown {
			ids = append(ids, c.Facet)
		}
		reason = fmt.Sprintf("%.0f%% of what this buyer weights has never been measured against their actual alternative (%s)", unknownWeight*100, strings.Join(ids, ", "))
	} else if len(ks) > 0 && ks[0].WEff >= switchBar {
		drag := ks[len(ks)-1]
		if drag.WEff < rungFloor("DRAG") {
			verdict = "ADOPT-WITH-SCARS"
			reason = fmt.Sprintf("%s carries it; you eat %s to get it", ks[0].Facet, drag.Facet)
		} else {
			verdict = "ADOPT"
			reason = fmt.Sprintf("%s clears the switch bar (%+.2f)", ks[0].Facet, switchBar)
		}
	}
	var pull, drag any
	if len(ks) > 0 {
		pull = ks[0].rounded()
		drag = ks[len(ks)-1].rounded()
	}
	return map[string]any{"segment": sid, "name": str(seg["name"]), "verdict": verdict, "reason": reason, "unknown_weight": round4(unknownWeight), "pull": pull, "drag": drag, "cells": roundedCells(cs)}
}
func contains(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}
func roundedCells(cs []Cell) []Cell {
	out := make([]Cell, len(cs))
	for i, c := range cs {
		out[i] = c.rounded()
	}
	return out
}
func (s *Scorecard) Hull() map[string]any {
	ks := known(s.Cells)
	sort.Slice(ks, func(i, j int) bool { return ks[i].WEff > ks[j].WEff })
	neg := 0
	for _, c := range ks {
		if c.WEff < 0 {
			neg++
		}
	}
	if len(ks) == 0 {
		return map[string]any{"peak": nil, "dent": nil, "eccentricity": 0.0, "scored_cells": 0, "negative_cells": 0}
	}
	return map[string]any{"peak": ks[0].rounded(), "dent": ks[len(ks)-1].rounded(), "eccentricity": round4(ks[0].WEff - ks[len(ks)-1].WEff), "scored_cells": len(ks), "negative_cells": neg}
}
func (s *Scorecard) Payload() map[string]any {
	segs := make([]map[string]any, 0, len(s.Meta.Segments))
	for _, seg := range s.Meta.Segments {
		segs = append(segs, s.SegmentProfile(str(seg["id"])))
	}
	facets := make([]map[string]any, 0, len(s.Meta.Facets))
	for _, f := range s.Meta.Facets {
		facets = append(facets, map[string]any{"facet": str(f["id"]), "name": str(f["name"]), "band": str(f["band"]), "shape": str(f["shape"]), "cells": roundedCells(s.byFacet(str(f["id"])))})
	}
	return map[string]any{"schema": Schema, "subject": s.Meta.Subject, "horizon_nats": round4(Horizon), "ladder": ladderPayload(), "bands": s.Meta.Bands, "facets": facets, "segments": segs, "hull": s.Hull(), DebtKey: s.Debt(), "defects": s.Defects}
}
func ladderPayload() []map[string]any {
	out := []map[string]any{}
	for _, r := range ladder {
		floor := any(r.Floor)
		if math.IsInf(r.Floor, -1) {
			floor = "-inf"
		}
		out = append(out, map[string]any{"label": r.Label, "floor": floor, "meaning": r.Blurb})
	}
	return out
}

func (s *Scorecard) RequireSegment(id string) error {
	if _, ok := s.Segments[id]; !ok {
		return fmt.Errorf("unknown segment %q; have: %s", id, strings.Join(ids(s.Meta.Segments), ", "))
	}
	return nil
}
func (s *Scorecard) RequireFacet(id string) error {
	if _, ok := s.Facets[id]; !ok {
		return fmt.Errorf("unknown facet %q; have: %s", id, strings.Join(ids(s.Meta.Facets), ", "))
	}
	return nil
}
func ids(xs []map[string]any) []string {
	out := []string{}
	for _, x := range xs {
		out = append(out, str(x["id"]))
	}
	return out
}
func (s *Scorecard) Check() string {
	var b strings.Builder
	fmt.Fprintf(&b, "lightgap_debt = %d\n", s.Debt())
	if len(s.Defects) > 0 {
		b.WriteString("\nSTRUCTURAL DEFECTS\n")
		for _, d := range s.Defects {
			fmt.Fprintf(&b, "  [%s] %s: %s\n", d.Code, d.Where, d.Detail)
			if d.NextAction != "" {
				fmt.Fprintf(&b, "      next -> %s\n", d.NextAction)
			}
		}
	}
	unknown := []Cell{}
	for _, c := range s.Cells {
		if c.Mode != "immaterial" && c.Provenance == "NONE" {
			unknown = append(unknown, c)
		}
	}
	if len(unknown) > 0 {
		b.WriteString("\nNEVER MEASURED\n")
		for _, c := range unknown {
			fmt.Fprintf(&b, "  %s x %s (weight %.2f): %s\n", c.Segment, c.Facet, c.Weight, c.Note)
			for _, d := range c.Defects {
				if d.Code == "NEVER_MEASURED" && d.NextAction != "" {
					fmt.Fprintf(&b, "      next -> %s\n", d.NextAction)
				}
			}
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

var ErrUnsupported = errors.New("unsupported")
