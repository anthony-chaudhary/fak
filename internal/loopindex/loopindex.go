package loopindex

import (
	"fmt"
	"io"
	"sort"
)

// schema is the stable control-pane schema identifier this scorecard emits.
const schema = "fak.loopindex.v1"

// Stage names — the six rungs of the agentic-coding loop, in loop order. The order
// is load-bearing: the worst-first picker walks the slice and the lowest unbuilt
// rung is the one a fleet should build next.
const (
	StageOrient = "orient"
	StagePlan   = "plan"
	StageAct    = "act"
	StageVerify = "verify"
	StageShip   = "ship"
	StageLearn  = "learn"
)

// stageOrder is the canonical loop order; Score validates the input against it so a
// caller cannot silently drop or reorder a stage.
var stageOrder = []string{StageOrient, StagePlan, StageAct, StageVerify, StageShip, StageLearn}

// Probe is one structural witness for a stage: a named check the impure shell ran
// against the TRACKED tree, with whether it passed. Keystone probes decide whether
// the stage is WIRED (a stage with no keystone witness is still a vibe, not a
// measured signal); ALL probes decide the stage's HEALTH fraction. Keeping the
// probe list in the payload lets a child issue point at exactly which witness it
// flipped when it reports its before/after delta.
type Probe struct {
	Name     string `json:"name"`
	Detail   string `json:"detail"`
	Keystone bool   `json:"keystone"`
	Pass     bool   `json:"pass"`
}

// Stage is one rung of the loop. Signal names the witnessed sub-signal; Floor is the
// health threshold below which a wired stage is still in debt; Probes are the
// structural witnesses that derive Wired + Health.
type Stage struct {
	Name   string  `json:"name"`
	Signal string  `json:"signal"`
	Floor  float64 `json:"floor"`
	Probes []Probe `json:"probes"`
}

// wired reports whether the stage has a load-bearing witnessed signal: every
// keystone probe passes. A stage declaring no keystone probe is never wired (it has
// asserted no witness to stand on).
func (s Stage) wired() bool {
	anyKeystone := false
	for _, p := range s.Probes {
		if p.Keystone {
			anyKeystone = true
			if !p.Pass {
				return false
			}
		}
	}
	return anyKeystone
}

// health is the fraction of the stage's probes that pass (0..1). An empty probe set
// reads as 0 — a stage with no witnesses has no health to claim.
func (s Stage) health() float64 {
	if len(s.Probes) == 0 {
		return 0
	}
	pass := 0
	for _, p := range s.Probes {
		if p.Pass {
			pass++
		}
	}
	return float64(pass) / float64(len(s.Probes))
}

// satisfied reports whether the stage is fully built: wired AND at or above its
// floor. A stage that is not satisfied contributes one unit of loopindex_debt.
func (s Stage) satisfied() bool { return s.wired() && s.health() >= s.Floor }

// Loop is the whole input: exactly the six stages in loop order. The impure shell
// builds it from the tracked tree; a test supplies fixtures. Passing it in is what
// keeps Score a pure function.
type Loop struct {
	Stages []Stage `json:"stages"`
}

// KPI is one graded stage rung. Score is 0..100; Debt is 0 or 1 (each stage is one
// rung). Detail is a human one-liner — on a failing rung it carries the fix hint.
type KPI struct {
	Name   string `json:"kpi"`
	Group  string `json:"group"`
	Signal string `json:"signal"`
	Wired  bool   `json:"wired"`
	Score  int    `json:"score"`
	Debt   int    `json:"debt"`
	Detail string `json:"detail"`
}

// Corpus is the headline summary block. The control pane reads
// corpus.loopindex_debt and corpus.grade; the keys are stable.
type Corpus struct {
	Stages         int     `json:"stages"`          // always 6 when valid
	WiredStages    int     `json:"wired_stages"`    // stages with a load-bearing witness
	LoopIndex      int     `json:"loop_index"`      // 0..100 over ALL stages (unwired contributes 0) — the "how close to 10x" headline
	WitnessedIndex int     `json:"witnessed_index"` // 0..100 over only the WIRED stages (health of what we can see)
	Score          int     `json:"score"`           // alias of LoopIndex for the control-pane fold
	Grade          string  `json:"grade"`           // A..F from LoopIndex
	LoopIndexDebt  int     `json:"loopindex_debt"`  // the headline integer: stages not yet witnessed at their floor
	WiredFrac      float64 `json:"wired_frac"`      // WiredStages / Stages
}

// Report is the control-pane envelope every fak scorecard emits, specialized to the
// loop-index surface.
type Report struct {
	Schema      string  `json:"schema"`
	OK          bool    `json:"ok"`
	Verdict     string  `json:"verdict"`
	Finding     string  `json:"finding"`
	Reason      string  `json:"reason"`
	NextAction  string  `json:"next_action"`
	Corpus      Corpus  `json:"corpus"`
	KPIs        []KPI   `json:"kpis"`
	StageDetail []Stage `json:"stage_detail"`
}

// Score is the whole scorecard: a pure, deterministic fold from a Loop to the
// control-pane Report. Same inputs -> identical output, always. An out-of-shape Loop
// (wrong stage count or order) is reported as a single capture-rung failure rather
// than panicking, so a mis-wired shell reds the gate honestly instead of crashing.
func Score(loop Loop) Report {
	if !validShape(loop) {
		return malformed(loop)
	}

	var kpis []KPI
	debt := 0
	wired := 0
	var sumAll, sumWired float64
	for _, st := range loop.Stages {
		w := st.wired()
		h := st.health()
		if w {
			wired++
			sumWired += h
		}
		contrib := 0.0
		if w {
			contrib = h
		}
		sumAll += contrib

		k := KPI{
			Name:   st.Name,
			Group:  st.Name,
			Signal: st.Signal,
			Wired:  w,
			Score:  int(round(100 * contrib)),
		}
		if st.satisfied() {
			k.Detail = stagePassDetail(st, h)
		} else {
			k.Debt = 1
			debt++
			k.Detail = stageFailDetail(st, w, h)
		}
		kpis = append(kpis, k)
	}

	n := len(loop.Stages)
	loopIndex := int(round(100 * sumAll / float64(n)))
	witnessed := 0
	if wired > 0 {
		witnessed = int(round(100 * sumWired / float64(wired)))
	}
	grade := gradeLetter(loopIndex)

	rep := Report{
		Schema: schema,
		OK:     debt == 0,
		Corpus: Corpus{
			Stages:         n,
			WiredStages:    wired,
			LoopIndex:      loopIndex,
			WitnessedIndex: witnessed,
			Score:          loopIndex,
			Grade:          grade,
			LoopIndexDebt:  debt,
			WiredFrac:      round3(float64(wired) / float64(n)),
		},
		KPIs:        kpis,
		StageDetail: loop.Stages,
	}
	rep.Verdict, rep.Finding, rep.Reason, rep.NextAction = verdict(debt, loopIndex, kpis)
	return rep
}

// validShape requires exactly the six canonical stages, in loop order. The spine is
// a fixed contract — a child wires a stage's PROBES, never adds or removes a stage.
func validShape(loop Loop) bool {
	if len(loop.Stages) != len(stageOrder) {
		return false
	}
	for i, st := range loop.Stages {
		if st.Name != stageOrder[i] {
			return false
		}
	}
	return true
}

// malformed reports a mis-shaped Loop as a maximal-debt capture failure: the index
// is 0, every stage counts as unbuilt, and the next action says to fix the shell.
func malformed(loop Loop) Report {
	return Report{
		Schema:     schema,
		OK:         false,
		Verdict:    "ACTION",
		Finding:    "loop-index input is malformed — not the six canonical stages in order",
		Reason:     fmt.Sprintf("got %d stages; want orient,plan,act,verify,ship,learn", len(loop.Stages)),
		NextAction: "fix the impure shell to emit exactly the six canonical stages in loop order",
		Corpus: Corpus{
			Stages:        len(loop.Stages),
			LoopIndexDebt: len(stageOrder),
			Grade:         "F",
		},
	}
}

// verdict picks the one-line finding + the worst-first next action.
func verdict(debt, loopIndex int, kpis []KPI) (verdict, finding, reason, next string) {
	if debt == 0 {
		return "OK",
			fmt.Sprintf("every loop stage is witnessed at its floor (loop-index %d/100)", loopIndex),
			"orient→plan→act→verify→ship→learn each have a load-bearing witnessed signal",
			"hold the index; raise a stage's floor or deepen its probes to push toward 10x"
	}
	worst := worstStage(kpis)
	return "ACTION",
		fmt.Sprintf("loop not fully witnessed: %d/%d stage(s) below floor (loop-index %d/100)", debt, len(kpis), loopIndex),
		fmt.Sprintf("worst-first stage: %s — %s", worst.Name, worst.Detail),
		worst.Detail
}

// worstStage returns the first failing stage in loop order (the lowest unbuilt rung).
func worstStage(kpis []KPI) KPI {
	for _, k := range kpis {
		if k.Debt > 0 {
			return k
		}
	}
	return kpis[0]
}

func stagePassDetail(st Stage, h float64) string {
	return fmt.Sprintf("%s wired (%s); health %.0f%% ≥ floor %.0f%%", st.Signal, witnessList(st, true), 100*h, 100*st.Floor)
}

func stageFailDetail(st Stage, wired bool, h float64) string {
	if !wired {
		return fmt.Sprintf("%s NOT wired — build the keystone witness: %s", st.Signal, witnessList(st, false))
	}
	return fmt.Sprintf("%s wired but health %.0f%% < floor %.0f%% — deepen: %s", st.Signal, 100*h, 100*st.Floor, witnessList(st, false))
}

// witnessList names the probes (passing=true lists what holds; passing=false lists
// what is still missing) so the detail line is actionable.
func witnessList(st Stage, passing bool) string {
	var names []string
	for _, p := range st.Probes {
		if p.Pass == passing {
			names = append(names, p.Name)
		}
	}
	if len(names) == 0 {
		if passing {
			return "none"
		}
		return "all probes pass"
	}
	out := names[0]
	for _, nm := range names[1:] {
		out += ", " + nm
	}
	return out
}

func gradeLetter(score int) string {
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

func round(x float64) float64 {
	if x < 0 {
		return -round(-x)
	}
	return float64(int64(x + 0.5))
}

func round3(x float64) float64 { return round(1000*x) / 1000 }

// Render writes the human work-list — the headline, then the stage rungs worst-first
// (failing before passing, in loop order among equals). It is the terminal view.
func Render(w io.Writer, rep Report) {
	c := rep.Corpus
	fmt.Fprintf(w, "loop-index scorecard (agentic-coding loop): %s grade %s, loopindex_debt=%d\n",
		rep.Verdict, c.Grade, c.LoopIndexDebt)
	fmt.Fprintf(w, "  loop-index=%d/100  witnessed=%d/100  wired=%d/%d stages\n",
		c.LoopIndex, c.WitnessedIndex, c.WiredStages, c.Stages)
	fmt.Fprintf(w, "  finding: %s\n", rep.Finding)
	fmt.Fprintf(w, "  next: %s\n", rep.NextAction)

	ranked := append([]KPI(nil), rep.KPIs...)
	// Stable sort: failing rungs first; otherwise preserve loop order (the slice is
	// already in loop order, so a stable sort keeps it).
	sort.SliceStable(ranked, func(i, j int) bool {
		fi, fj := ranked[i].Debt > 0, ranked[j].Debt > 0
		if fi != fj {
			return fi
		}
		return false
	})
	for _, k := range ranked {
		mark := "ok  "
		if k.Debt > 0 {
			mark = "DEBT"
		}
		wire := "vibe "
		if k.Wired {
			wire = "wired"
		}
		fmt.Fprintf(w, "  [%s] %-7s %-6s %3d  %s\n", mark, k.Name, wire, k.Score, k.Detail)
	}
}
