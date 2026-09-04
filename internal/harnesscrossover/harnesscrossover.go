package harnesscrossover

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// Schema defines the expected JSON schema identifier for harness crossover studies.
const Schema = "fak.harness-crossover-study/v1alpha1"

// ReportSchema defines the JSON schema identifier for evaluation reports.
const ReportSchema = "fak.harness-crossover-report/v1alpha1"

// Study models an empirical crossover trial across tasks, evaluation weights, and candidate systems.
type Study struct {
	Schema       string        `json:"schema"`
	ID           string        `json:"id"`
	Tasks        []Task        `json:"tasks"`
	Weights      Weights       `json:"weights"`
	Alternatives []Alternative `json:"alternatives"`
}

// Task defines an evaluation task associated with a specific operational domain.
type Task struct {
	ID     string `json:"id"`
	Domain string `json:"domain"`
}

// Weights parameterizes time costs assigned to context switching, layering errors, and token overhead.
type Weights struct {
	SwitchActionSeconds float64 `json:"switch_action_seconds"`
	WrongLayerSeconds   float64 `json:"wrong_layer_seconds"`
	ContextTokenSeconds float64 `json:"context_token_seconds"`
}

// Alternative describes a candidate architecture (native profile or contextual harness) evaluated over all tasks.
type Alternative struct {
	ID            string   `json:"id"`
	Kind          string   `json:"kind"`
	Documentation []Source `json:"documentation"`
	Setup         Seconds  `json:"setup"`
	Maintenance   Seconds  `json:"maintenance"`
	Runs          []Run    `json:"runs"`
}

// Source records provenance or documentation metadata supporting an alternative's parameters.
type Source struct {
	URL       string `json:"url"`
	Retrieved string `json:"retrieved"`
	Note      string `json:"note"`
}

// Seconds captures an elapsed duration in seconds along with its evidence classification.
type Seconds struct {
	Value      float64 `json:"value"`
	Provenance string  `json:"provenance"`
}

// Run records observed or modeled performance metrics for executing a single task under an alternative.
type Run struct {
	TaskID              string  `json:"task_id"`
	SwitchActions       int     `json:"switch_actions"`
	WrongLayerIncidents int     `json:"wrong_layer_incidents"`
	Explanation         Seconds `json:"explanation"`
	ContextTokens       int     `json:"context_tokens"`
	Provenance          string  `json:"provenance"`
}

// Cost summarizes the aggregate time breakdown incurred by an alternative.
type Cost struct {
	SetupSeconds       float64 `json:"setup_seconds"`
	MaintenanceSeconds float64 `json:"maintenance_seconds"`
	SwitchSeconds      float64 `json:"switch_seconds"`
	WrongLayerSeconds  float64 `json:"wrong_layer_seconds"`
	ExplanationSeconds float64 `json:"explanation_seconds"`
	ContextSeconds     float64 `json:"context_seconds"`
	TotalSeconds       float64 `json:"total_seconds"`
}

// Result aggregates the evaluated cost and provenance tiers for a single alternative.
type Result struct {
	ID         string   `json:"id"`
	Kind       string   `json:"kind"`
	Cost       Cost     `json:"cost"`
	Provenance []string `json:"provenance"`
}

// Crossover captures the break-even delta and domain-switch threshold where a contextual harness overtakes a native profile.
type Crossover struct {
	ContextualID          string   `json:"contextual_id"`
	NativeID              string   `json:"native_id"`
	FixedDeltaSeconds     float64  `json:"fixed_delta_seconds"`
	PerSwitchDeltaSeconds float64  `json:"per_switch_delta_seconds"`
	BreakEvenSwitches     *float64 `json:"break_even_switches,omitempty"`
	Interpretation        string   `json:"interpretation"`
}

// Report presents the evaluation findings, winning alternative, and crossover economics for a study.
type Report struct {
	Schema      string     `json:"schema"`
	StudyID     string     `json:"study_id"`
	TaskDomains []string   `json:"task_domains"`
	Switches    int        `json:"switches"`
	Winner      string     `json:"winner"`
	Results     []Result   `json:"results"`
	Crossover   *Crossover `json:"crossover,omitempty"`
	Verdict     string     `json:"verdict"`
}

// Parse decodes and validates a JSON study definition, checking required domains, tasks, and provenance levels.
func Parse(raw []byte) (Study, error) {
	var s Study
	d := json.NewDecoder(strings.NewReader(string(raw)))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		return s, fmt.Errorf("parse study: %w", err)
	}
	if s.Schema != Schema {
		return s, fmt.Errorf("schema must be %q", Schema)
	}
	if s.ID == "" || len(s.Tasks) < 2 || len(s.Alternatives) < 2 {
		return s, fmt.Errorf("id, at least two tasks, and at least two alternatives are required")
	}
	task := map[string]bool{}
	domains := map[string]bool{}
	for _, t := range s.Tasks {
		if t.ID == "" || t.Domain == "" || task[t.ID] {
			return s, fmt.Errorf("tasks require unique id and domain")
		}
		task[t.ID] = true
		domains[t.Domain] = true
	}
	for _, domain := range []string{"coding", "legal", "integrated"} {
		if !domains[domain] {
			return s, fmt.Errorf("tasks must include %s", domain)
		}
	}
	ids := map[string]bool{}
	for _, a := range s.Alternatives {
		if a.ID == "" || ids[a.ID] || len(a.Documentation) == 0 {
			return s, fmt.Errorf("alternatives require unique id and documentation")
		}
		ids[a.ID] = true
		if a.Kind != "native-profile" && a.Kind != "contextual-harness" {
			return s, fmt.Errorf("%s has invalid kind", a.ID)
		}
		if !validProv(a.Setup.Provenance) || !validProv(a.Maintenance.Provenance) {
			return s, fmt.Errorf("%s setup/maintenance provenance invalid", a.ID)
		}
		seen := map[string]bool{}
		for _, r := range a.Runs {
			if !task[r.TaskID] || seen[r.TaskID] || !validProv(r.Provenance) || !validProv(r.Explanation.Provenance) {
				return s, fmt.Errorf("%s has invalid run", a.ID)
			}
			seen[r.TaskID] = true
		}
		if len(seen) != len(task) {
			return s, fmt.Errorf("%s must cover every task", a.ID)
		}
	}
	return s, nil
}

// Evaluate computes the cost breakdown for each alternative and calculates the crossover break-even point.
func Evaluate(s Study) Report {
	switches := 0
	domains := make([]string, len(s.Tasks))
	for i, t := range s.Tasks {
		domains[i] = t.Domain
		if i > 0 && t.Domain != s.Tasks[i-1].Domain {
			switches++
		}
	}
	results := make([]Result, 0, len(s.Alternatives))
	for _, a := range s.Alternatives {
		c := Cost{SetupSeconds: a.Setup.Value, MaintenanceSeconds: a.Maintenance.Value}
		prov := map[string]bool{a.Setup.Provenance: true, a.Maintenance.Provenance: true}
		for _, r := range a.Runs {
			c.SwitchSeconds += float64(r.SwitchActions) * s.Weights.SwitchActionSeconds
			c.WrongLayerSeconds += float64(r.WrongLayerIncidents) * s.Weights.WrongLayerSeconds
			c.ExplanationSeconds += r.Explanation.Value
			c.ContextSeconds += float64(r.ContextTokens) * s.Weights.ContextTokenSeconds
			prov[r.Provenance] = true
			prov[r.Explanation.Provenance] = true
		}
		c.TotalSeconds = c.SetupSeconds + c.MaintenanceSeconds + c.SwitchSeconds + c.WrongLayerSeconds + c.ExplanationSeconds + c.ContextSeconds
		ps := make([]string, 0, len(prov))
		for p := range prov {
			ps = append(ps, p)
		}
		sort.Strings(ps)
		results = append(results, Result{ID: a.ID, Kind: a.Kind, Cost: c, Provenance: ps})
	}
	sort.Slice(results, func(i, j int) bool {
		if results[i].Cost.TotalSeconds != results[j].Cost.TotalSeconds {
			return results[i].Cost.TotalSeconds < results[j].Cost.TotalSeconds
		}
		return results[i].ID < results[j].ID
	})
	winner := results[0].ID
	verdict := "native-profile wins under declared costs"
	if results[0].Kind == "contextual-harness" {
		verdict = "contextual harness wins under declared costs"
	}
	return Report{Schema: ReportSchema, StudyID: s.ID, TaskDomains: domains, Switches: switches, Winner: winner, Results: results, Crossover: crossover(results, switches), Verdict: verdict}
}
func crossover(results []Result, switches int) *Crossover {
	if switches == 0 {
		return nil
	}
	var ctx, native *Result
	for i := range results {
		r := &results[i]
		if r.Kind == "contextual-harness" && (ctx == nil || r.Cost.TotalSeconds < ctx.Cost.TotalSeconds) {
			ctx = r
		}
		if r.Kind == "native-profile" && (native == nil || r.Cost.TotalSeconds < native.Cost.TotalSeconds) {
			native = r
		}
	}
	if ctx == nil || native == nil {
		return nil
	}
	fixed := func(c Cost) float64 { return c.SetupSeconds + c.MaintenanceSeconds + c.ExplanationSeconds }
	variable := func(c Cost) float64 {
		return (c.SwitchSeconds + c.WrongLayerSeconds + c.ContextSeconds) / float64(switches)
	}
	fd := fixed(ctx.Cost) - fixed(native.Cost)
	pd := variable(ctx.Cost) - variable(native.Cost)
	out := &Crossover{ContextualID: ctx.ID, NativeID: native.ID, FixedDeltaSeconds: fd, PerSwitchDeltaSeconds: pd}
	if pd < 0 {
		n := fd / -pd
		if n < 0 {
			n = 0
		}
		out.BreakEvenSwitches = &n
		out.Interpretation = fmt.Sprintf("contextual harness becomes cheaper after %.2f domain switches", n)
	} else if fd < 0 {
		out.Interpretation = "contextual harness is cheaper before and after additional switches"
	} else {
		out.Interpretation = "contextual harness does not cross below the best native profile under declared costs"
	}
	return out
}
func validProv(p string) bool { return p == "witnessed" || p == "observed" || p == "modeled" }
