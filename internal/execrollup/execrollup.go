package execrollup

// execrollup.go is the pure, unit-tested surface of the executive activity
// roll-up: the one read-only fold that turns the firehose of agentic-fleet
// signals into a single signal-dense page a human can read in a glance.
//
// The problem it solves (the "city of agents" problem): a fleet of autonomous
// agents emits more volume than a person can track — hundreds of "closed"
// issues, dozens of commits, many loops, several scorecards — and most of that
// volume is NOISE to a decision-maker. The few things that are SIGNAL are: how
// much of the volume is genuinely real (witnessed vs merely claimed), what is
// trending the wrong way, and the short list of things that actually need a
// human right now.
//
// This fold encodes signal-to-noise as a design rule, not a slogan:
//
//   - A QUIET plane contributes nothing. Only deviations surface in Attention;
//     an all-green plane adds no line. Silence is the absence of signal, so it
//     is the absence of output.
//   - UNMEASURED is never GREEN. A plane whose collector failed becomes a WATCH
//     gap, never a silent pass — a missing witness is honest, a fake green is a
//     defect.
//   - One always-on marquee: the closure-honesty ratio (witnessed-resolved /
//     claimed-closed) — the literal signal-to-noise ratio of the fleet's "done".
//   - Every surfaced number carries a PROVENANCE label (witnessed / observed /
//     claimed / unverified), matching the EXECUTIVE-ROLLUP discipline.
//
// The fold is PURE: it takes the already-emitted JSON payloads of the per-plane
// folds (fak cadence, fak loop health, fak fleet status, tools/dispatch_status.py)
// as typed inputs and returns one envelope. It runs no tool, reads no clock, and
// touches no disk — the live collectors live in cmd/fak/rollup.go (the impure
// shell), exactly the cadencereport.go / collect.go split. That keeps the
// verdict logic, the JSON shape, and the tests sharing one deterministic surface.

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// Schema is the stable control-pane schema identifier for the roll-up envelope.
const Schema = "fak.exec-rollup/v1"

// Provenance labels — the honesty discipline carried onto every surfaced number.
// WITNESSED = proven from git/tests/a committed artifact. OBSERVED = a reading
// relayed from a live source (a box, a loop tick). CLAIMED = self-reported done,
// no witness yet. UNVERIFIED = asserted, no witness and no source.
const (
	Witnessed  = "WITNESSED"
	Observed   = "OBSERVED"
	Claimed    = "CLAIMED"
	Unverified = "UNVERIFIED"
)

// Item levels — the rank that drives "what needs you" ordering. OK items are
// computed but suppressed from the rendered Attention list by design (an OK line
// is noise); they remain available in the JSON for a consumer that wants them.
const (
	LevelCrit = "crit"
	LevelWarn = "warn"
	LevelOK   = "ok"
)

// Fleet verdicts — the one-glance state word. RED: at least one crit. WATCH: at
// least one warn OR an unmeasured plane. GREEN: every plane measured and clean.
const (
	VerdictGreen = "GREEN"
	VerdictWatch = "WATCH"
	VerdictRed   = "RED"
)

// Plane verdict words (per-plane, distinct from the fleet verdict so the table
// reads cleanly): OK | WATCH | CRIT | UNMEASURED.
const (
	PlaneOK         = "OK"
	PlaneWatch      = "WATCH"
	PlaneCrit       = "CRIT"
	PlaneUnmeasured = "UNMEASURED"
)

// Item is one ranked "what needs you now" entry. crit before warn; OK suppressed
// from render. Every item names the plane it came from and a provenance label so
// a reader can weigh it without chasing the source.
type Item struct {
	Level  string `json:"level"`
	Plane  string `json:"plane"`
	Title  string `json:"title"`
	Detail string `json:"detail,omitempty"`
	Prov   string `json:"prov,omitempty"`
}

// PlaneStatus is the per-plane coverage row: was it measured, what is its local
// verdict, and a one-line summary. Measured=false is the honest gap that forces
// the fleet verdict to WATCH.
type PlaneStatus struct {
	Name     string `json:"name"`
	Measured bool   `json:"measured"`
	Verdict  string `json:"verdict"`
	Summary  string `json:"summary,omitempty"`
	Err      string `json:"err,omitempty"`
}

// SignalNoise is the marquee: how much of the fleet's volume is real. A ratio of
// -1 means unmeasured (never rendered as a number).
type SignalNoise struct {
	// ClosureHonest is the witnessed-resolved share of "closed" issues — the
	// canonical "is the fleet's 'done' actually done" ratio.
	ClosureHonest   float64 `json:"closure_honest"`
	ClosureMeasured bool    `json:"closure_measured"`
	TrueResolved    int     `json:"true_resolved,omitempty"`
	ClaimedClosed   int     `json:"claimed_closed,omitempty"`
	// ShipStampRate is the share of trailing-window commits whose subject carries
	// a real ship-stamp — how much committed work is attributed, not anonymous.
	ShipStampRate float64 `json:"ship_stamp_rate"`
	ShipMeasured  bool    `json:"ship_measured"`
	Ships         int     `json:"ships,omitempty"`
	Commits       int     `json:"commits,omitempty"`
	WindowDays    int     `json:"window_days,omitempty"`
}

// Rollup is one folded executive-activity control-pane envelope. It carries the
// same schema/ok/verdict/finding/reason/next_action spine the rest of the fak
// control panes use, plus the marquee S/N, the ranked Attention list, useful
// next-work seeds, and the per-plane coverage table.
type Rollup struct {
	Schema      string        `json:"schema"`
	OK          bool          `json:"ok"`
	Verdict     string        `json:"verdict"`
	Finding     string        `json:"finding"`
	Reason      string        `json:"reason"`
	NextAction  string        `json:"next_action"`
	Headline    string        `json:"headline"`
	Workspace   string        `json:"workspace,omitempty"`
	Commit      string        `json:"commit,omitempty"`
	GeneratedAt string        `json:"generated_at,omitempty"`
	SignalNoise SignalNoise   `json:"signal_noise"`
	Attention   []Item        `json:"attention"`
	NextWork    []Item        `json:"next_work,omitempty"`
	Planes      []PlaneStatus `json:"planes"`
	Unmeasured  int           `json:"unmeasured"`
}

// PlaneInput is the raw payload of one per-plane fold plus the collector error.
// A non-empty Err (or a nil Payload) marks the plane unmeasured — the fold turns
// that into a WATCH gap, never a silent pass.
type PlaneInput struct {
	Payload map[string]any
	Err     string
}

// Inputs is the typed bundle the live collectors hand the fold. Each field is one
// per-plane fold's JSON payload; missing planes degrade, they do not panic.
type Inputs struct {
	Dispatch    PlaneInput // tools/dispatch_status.py --json
	Loops       PlaneInput // fak loop health --json
	Cadence     PlaneInput // fak cadence --json (scores + maturity + work-done)
	Fleet       PlaneInput // fak fleet status --json
	Workspace   string
	Commit      string
	GeneratedAt string
}

// Fold turns the per-plane payloads into one executive roll-up. Deterministic and
// side-effect free.
func Fold(in Inputs) Rollup {
	planeStatuses := make([]PlaneStatus, 0, 4)
	var items []Item

	for _, p := range []struct {
		fn func(PlaneInput) (PlaneStatus, []Item)
		in PlaneInput
	}{
		{interpretDispatch, in.Dispatch},
		{interpretLoops, in.Loops},
		{interpretCadence, in.Cadence},
		{interpretFleet, in.Fleet},
	} {
		st, its := p.fn(p.in)
		planeStatuses = append(planeStatuses, st)
		items = append(items, its...)
	}

	sn := signalNoise(in.Dispatch, in.Cadence)

	unmeasured := 0
	for _, st := range planeStatuses {
		if !st.Measured {
			unmeasured++
		}
	}

	// Rank: crit before warn. OK items are productive next-work seeds, not
	// attention alarms, so they are rendered separately.
	attention := make([]Item, 0, len(items))
	nextWork := make([]Item, 0, len(items))
	for _, it := range items {
		switch it.Level {
		case LevelCrit, LevelWarn:
			attention = append(attention, it)
		case LevelOK:
			nextWork = append(nextWork, it)
		}
	}
	sort.SliceStable(attention, func(i, j int) bool {
		return levelRank(attention[i].Level) < levelRank(attention[j].Level)
	})
	if len(nextWork) > 3 {
		nextWork = nextWork[:3]
	}

	nCrit, nWarn := 0, 0
	for _, it := range attention {
		switch it.Level {
		case LevelCrit:
			nCrit++
		case LevelWarn:
			nWarn++
		}
	}

	verdict := VerdictGreen
	switch {
	case nCrit > 0:
		verdict = VerdictRed
	case nWarn > 0 || unmeasured > 0:
		verdict = VerdictWatch
	}

	r := Rollup{
		Schema:      Schema,
		OK:          verdict == VerdictGreen,
		Verdict:     verdict,
		Workspace:   in.Workspace,
		Commit:      in.Commit,
		GeneratedAt: in.GeneratedAt,
		SignalNoise: sn,
		Attention:   attention,
		NextWork:    nextWork,
		Planes:      planeStatuses,
		Unmeasured:  unmeasured,
	}
	r.Headline = headline(r, nCrit, nWarn)
	r.Finding, r.Reason, r.NextAction = narrate(r, nCrit, nWarn)
	return r
}

func levelRank(level string) int {
	switch level {
	case LevelCrit:
		return 0
	case LevelWarn:
		return 1
	default:
		return 2
	}
}

// planeVerdict folds a plane's items into its local verdict word.
func planeVerdict(items []Item) string {
	v := PlaneOK
	for _, it := range items {
		switch it.Level {
		case LevelCrit:
			return PlaneCrit
		case LevelWarn:
			v = PlaneWatch
		}
	}
	return v
}

// --- per-plane interpreters ------------------------------------------------

// interpretDispatch reads a tools/dispatch_status.py --json payload: closure
// honesty (the marquee S/N), throughput vs target, dead backends, silent workers,
// and witnessed-closable issues a human could close now.
func interpretDispatch(in PlaneInput) (PlaneStatus, []Item) {
	if in.Err != "" || in.Payload == nil {
		return unmeasuredPlane("dispatch", in.Err), nil
	}
	var items []Item

	if cl, ok := in.Payload["closure"].(map[string]any); ok && !asBool(cl["na"]) {
		if rate, ok := asFloatOK(cl["closure_rate"]); ok {
			counts, _ := cl["counts"].(map[string]any)
			tr, cc := asInt(counts["TRUE_RESOLVED"]), asInt(counts["CLAIMED_CLOSED"])
			detail := fmt.Sprintf("%d witnessed-resolved vs %d claimed-closed", tr, cc)
			switch {
			case rate < 0.5:
				items = append(items, Item{LevelCrit, "dispatch",
					fmt.Sprintf("Closure honesty %.0f%% — most 'closed' work is unwitnessed", rate*100),
					detail, Witnessed})
			case rate < 0.85:
				items = append(items, Item{LevelWarn, "dispatch",
					fmt.Sprintf("Closure honesty %.0f%% — some 'closed' work is unwitnessed", rate*100),
					detail, Witnessed})
			}
		}
		if closable := asInt(cl["open_witnessed_closable"]); closable > 0 {
			items = append(items, Item{LevelWarn, "dispatch",
				fmt.Sprintf("%d issue(s) closable now — witnessed-resolved but still open", closable),
				"a referee already proved these; close them to clear false backlog", Witnessed})
		}
	}

	if tp, ok := in.Payload["throughput"].(map[string]any); ok && !asBool(tp["na"]) {
		if v := asString(tp["verdict"]); v == "BELOW_TARGET" || v == "AUDIT_ERROR" {
			rate, _ := asFloatOK(tp["completed_rate_per_hour"])
			target, _ := asFloatOK(tp["target_per_hour"])
			win := asInt(tp["primary_window_hours"])
			items = append(items, Item{LevelWarn, "dispatch",
				fmt.Sprintf("Throughput below target — %.2g/h vs %.2g/h", rate, target),
				fmt.Sprintf("completed over the trailing %dh window", win), Witnessed})
		}
	}

	if bh, ok := in.Payload["backend_health"].(map[string]any); ok {
		if dead := asInt(bh["dead_count"]); dead > 0 {
			items = append(items, Item{LevelCrit, "dispatch",
				fmt.Sprintf("%d backend(s) held dead", dead),
				"a model/account backend is failing and was taken out of rotation", Observed})
		}
	}

	if wk, ok := in.Payload["workers"].(map[string]any); ok {
		if silent := asInt(wk["silent_count"]); silent > 0 {
			items = append(items, Item{LevelWarn, "dispatch",
				fmt.Sprintf("%d worker(s) produced nothing", silent),
				"spawned but emitted no output — wasted spend", Observed})
		}
	}

	st := PlaneStatus{Name: "dispatch", Measured: true, Summary: dispatchSummary(in.Payload)}
	st.Verdict = planeVerdict(items)
	return st, items
}

func dispatchSummary(p map[string]any) string {
	var b strings.Builder
	if bl, ok := p["backlog"].(map[string]any); ok {
		if open := bl["open_issues"]; open != nil {
			fmt.Fprintf(&b, "backlog %d", asInt(open))
		}
	}
	if tp, ok := p["throughput"].(map[string]any); ok && !asBool(tp["na"]) {
		if rate, ok := asFloatOK(tp["completed_rate_per_hour"]); ok {
			if b.Len() > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(&b, "%.2g/h done", rate)
		}
	}
	if b.Len() == 0 {
		return "measured"
	}
	return b.String()
}

// interpretLoops reads a fak loop health --json payload. A DARK loop — scheduled
// agentic automation that has silently stopped ticking — is the canonical "the
// human thinks an agent is running but it is not" failure, so it is surfaced.
func interpretLoops(in PlaneInput) (PlaneStatus, []Item) {
	if in.Err != "" || in.Payload == nil {
		return unmeasuredPlane("loops", in.Err), nil
	}
	var items []Item
	roll, _ := in.Payload["rollup"].(map[string]any)
	dark := asInt(roll["dark"])
	live := asInt(roll["live"])
	total := asInt(roll["loops"])
	if dark > 0 {
		level := LevelWarn
		if dark >= 3 {
			level = LevelCrit
		}
		items = append(items, Item{level, "loops",
			fmt.Sprintf("%d loop(s) DARK — registered automation not ticking", dark),
			fmt.Sprintf("%d of %d loops have stopped within cadence", dark, total), Observed})
	}
	st := PlaneStatus{Name: "loops", Measured: true,
		Summary: fmt.Sprintf("%d live, %d dark of %d", live, dark, total)}
	st.Verdict = planeVerdict(items)
	return st, items
}

// interpretCadence reads a fak cadence --json payload. The WORK-DONE dimension
// (commits + ship-stamped subset) is the always-cheap part that feeds the S/N
// marquee; the SCORES quality-debt trend is an optional overlay — present only
// when the (~4-minute) scorecard pane was run. A missing scores block is "not
// run", not a regression, so it never fabricates a warn.
func interpretCadence(in PlaneInput) (PlaneStatus, []Item) {
	if in.Err != "" || in.Payload == nil {
		return unmeasuredPlane("cadence", in.Err), nil
	}
	var items []Item
	work, _ := in.Payload["work"].(map[string]any)
	commits, ships := asInt(work["commits"]), asInt(work["ships"])
	summary := fmt.Sprintf("%d/%d ship-stamped", ships, commits)

	if scores, ok := in.Payload["scores"].(map[string]any); ok && len(scores) > 0 {
		if asString(scores["trend_direction"]) == "regressed" {
			items = append(items, Item{LevelWarn, "cadence",
				"Quality debt regressed vs the last tick",
				asString(scores["trend_summary"]), Witnessed})
		}
		if e := asString(scores["err"]); e != "" {
			items = append(items, Item{LevelWarn, "cadence",
				"Scorecard portfolio only partially measured", e, Observed})
		}
		summary += fmt.Sprintf(", debt %d (%s)",
			asInt(scores["debt"]), dashIfEmpty(asString(scores["trend_direction"])))
	} else {
		summary += " (scores not run)"
	}
	if maturity, ok := in.Payload["maturity"].(map[string]any); ok && len(maturity) > 0 {
		if e := asString(maturity["err"]); e != "" {
			items = append(items, Item{LevelWarn, "cadence",
				"Maturity scorecard did not measure", e, Observed})
		}
		debt := asInt(maturity["debt"])
		if debt > 0 {
			items = append(items, Item{LevelWarn, "cadence",
				fmt.Sprintf("Maturity ladder-skip debt %d", debt),
				"run `fak maturity next` and retire the skip rows first", Witnessed})
		}
		routeLane := asString(maturity["route_lane"])
		routeItem := asString(maturity["route_item"])
		routeKey := asString(maturity["route_key"])
		skipped := asInt(maturity["route_skipped_private"])
		if routeLane != "" {
			summary += ", route " + routeLane
			detail := fmt.Sprintf("run `fak maturity route --fetch-existing --limit 3`; %s", dashIfEmpty(routeKey))
			if routeItem != "" {
				detail += ": " + routeItem
			}
			if skipped > 0 {
				detail += fmt.Sprintf("; %d private-boundary row(s) skipped", skipped)
			}
			items = append(items, Item{LevelOK, "cadence",
				"Seed maturity dispatch with " + routeLane, detail, Witnessed})
		} else if skipped > 0 {
			summary += fmt.Sprintf(", %d private maturity skip(s)", skipped)
		}
	}
	st := PlaneStatus{Name: "cadence", Measured: true, Summary: summary}
	st.Verdict = planeVerdict(items)
	return st, items
}

// interpretFleet reads a fak fleet status --json payload. The fleet fold already
// emits a ranked "what needs me now" Attention list (box liveness / GPU waste);
// this lifts its crit/warn rows into the unified list, dropping its OK noise.
func interpretFleet(in PlaneInput) (PlaneStatus, []Item) {
	if in.Err != "" || in.Payload == nil {
		return unmeasuredPlane("fleet", in.Err), nil
	}
	var items []Item
	if att, ok := in.Payload["attention"].([]any); ok {
		for _, raw := range att {
			m, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			level := asString(m["level"])
			if level != LevelCrit && level != LevelWarn {
				continue
			}
			items = append(items, Item{level, "fleet",
				asString(m["title"]), asString(m["detail"]), Observed})
		}
	}
	total := asInt(in.Payload["total"])
	reachable := asInt(in.Payload["reachable"])
	summary := fmt.Sprintf("%d/%d boxes reachable", reachable, total)
	if total == 0 {
		summary = "no boxes reporting"
	}
	st := PlaneStatus{Name: "fleet", Measured: true, Summary: summary}
	st.Verdict = planeVerdict(items)
	return st, items
}

func unmeasuredPlane(name, err string) PlaneStatus {
	return PlaneStatus{Name: name, Measured: false, Verdict: PlaneUnmeasured, Err: orNoPayload(err)}
}

// signalNoise builds the marquee from the dispatch (closure) and cadence (work)
// payloads. A ratio stays -1 / Measured=false whenever its source is absent.
func signalNoise(dispatch, cadence PlaneInput) SignalNoise {
	sn := SignalNoise{ClosureHonest: -1, ShipStampRate: -1}
	if dispatch.Err == "" && dispatch.Payload != nil {
		if cl, ok := dispatch.Payload["closure"].(map[string]any); ok && !asBool(cl["na"]) {
			if r, ok := asFloatOK(cl["closure_rate"]); ok {
				sn.ClosureHonest = r
				sn.ClosureMeasured = true
			}
			if counts, ok := cl["counts"].(map[string]any); ok {
				sn.TrueResolved = asInt(counts["TRUE_RESOLVED"])
				sn.ClaimedClosed = asInt(counts["CLAIMED_CLOSED"])
			}
		}
	}
	if cadence.Err == "" && cadence.Payload != nil {
		if w, ok := cadence.Payload["work"].(map[string]any); ok && asString(w["err"]) == "" {
			commits := asInt(w["commits"])
			sn.Commits = commits
			sn.Ships = asInt(w["ships"])
			sn.WindowDays = asInt(w["window_days"])
			if commits > 0 {
				sn.ShipStampRate = float64(sn.Ships) / float64(commits)
				sn.ShipMeasured = true
			}
		}
	}
	return sn
}

// headline is the one-paragraph "city of agents in one line".
func headline(r Rollup, nCrit, nWarn int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Fleet %s", r.Verdict)
	if r.SignalNoise.ClosureMeasured {
		fmt.Fprintf(&b, ": %.0f%% of 'done' is witnessed-real (%d resolved / %d claimed)",
			r.SignalNoise.ClosureHonest*100, r.SignalNoise.TrueResolved, r.SignalNoise.ClaimedClosed)
	}
	if r.SignalNoise.ShipMeasured {
		fmt.Fprintf(&b, "; %d/%d commits ship-stamped over %dd",
			r.SignalNoise.Ships, r.SignalNoise.Commits, r.SignalNoise.WindowDays)
	}
	fmt.Fprintf(&b, ". %d need-now (%d crit / %d watch)", nCrit+nWarn, nCrit, nWarn)
	if len(r.NextWork) > 0 {
		fmt.Fprintf(&b, "; %d next-work seed(s)", len(r.NextWork))
	}
	if r.Unmeasured > 0 {
		fmt.Fprintf(&b, "; %d plane(s) unmeasured", r.Unmeasured)
	}
	b.WriteString(".")
	return b.String()
}

// narrate fills the control-pane finding/reason/next_action triple from the top
// of the ranked Attention list — the single most important thing a human owes.
func narrate(r Rollup, nCrit, nWarn int) (finding, reason, next string) {
	switch r.Verdict {
	case VerdictGreen:
		if len(r.NextWork) > 0 {
			top := r.NextWork[0]
			finding = "Fleet healthy — next work is ready."
			reason = "every plane measured and clean; " + top.Title
			next = top.Title
			if top.Detail != "" {
				next = top.Title + " — " + top.Detail
			}
			return
		}
		finding = "Fleet healthy — no item needs a human now."
		reason = "every plane measured and clean"
		next = "none — keep shipping"
		return
	case VerdictWatch:
		if nWarn > 0 {
			finding = fmt.Sprintf("%d item(s) worth a glance; nothing critical.", nWarn)
		} else {
			finding = fmt.Sprintf("No deviations seen, but %d plane(s) unmeasured — can't certify GREEN.", r.Unmeasured)
		}
	case VerdictRed:
		finding = fmt.Sprintf("%d critical item(s) need a human now.", nCrit)
	}
	if len(r.Attention) > 0 {
		top := r.Attention[0]
		reason = fmt.Sprintf("%s [%s]", top.Title, top.Plane)
		next = top.Title
		if top.Detail != "" {
			next = top.Title + " — " + top.Detail
		}
	} else if r.Unmeasured > 0 {
		reason = fmt.Sprintf("%d plane(s) failed to measure", r.Unmeasured)
		next = "re-run the unmeasured collector(s) before trusting GREEN"
	}
	return
}

// Render produces the human markdown body for the roll-up doc. Deterministic;
// it takes no clock (GeneratedAt is carried on the Rollup).
func Render(r Rollup) string {
	var b strings.Builder
	emoji := map[string]string{VerdictGreen: "🟢", VerdictWatch: "🟡", VerdictRed: "🔴"}[r.Verdict]
	fmt.Fprintf(&b, "## %s Fleet state: %s\n\n", emoji, r.Verdict)
	fmt.Fprintf(&b, "%s\n\n", r.Headline)

	// Marquee: the signal-to-noise ratio, front and center.
	b.WriteString("### Signal-to-noise\n\n")
	if r.SignalNoise.ClosureMeasured {
		fmt.Fprintf(&b, "- **Closure honesty: %.0f%%** — %d witnessed-resolved vs %d claimed-closed. _[WITNESSED]_\n",
			r.SignalNoise.ClosureHonest*100, r.SignalNoise.TrueResolved, r.SignalNoise.ClaimedClosed)
	} else {
		b.WriteString("- **Closure honesty: unmeasured** — the dispatch audit did not report. _[gap]_\n")
	}
	if r.SignalNoise.ShipMeasured {
		fmt.Fprintf(&b, "- **Ship-stamp rate: %.0f%%** — %d of %d commits over %dd carry a real ship-stamp. _[WITNESSED]_\n",
			r.SignalNoise.ShipStampRate*100, r.SignalNoise.Ships, r.SignalNoise.Commits, r.SignalNoise.WindowDays)
	}
	b.WriteString("\n")

	// The signal: what needs a human, ranked.
	b.WriteString("### What needs you\n\n")
	if len(r.Attention) == 0 {
		b.WriteString("_Nothing — every measured plane is clean._\n\n")
	} else {
		for _, it := range r.Attention {
			mark := "⚠️"
			if it.Level == LevelCrit {
				mark = "🔴"
			}
			fmt.Fprintf(&b, "- %s **%s** `%s`", mark, it.Title, it.Plane)
			if it.Prov != "" {
				fmt.Fprintf(&b, " _[%s]_", it.Prov)
			}
			b.WriteString("\n")
			if it.Detail != "" {
				fmt.Fprintf(&b, "  - %s\n", it.Detail)
			}
		}
		b.WriteString("\n")
	}

	// Productive work: not an alarm, but useful for humans/agents deciding what to pick next.
	if len(r.NextWork) > 0 {
		b.WriteString("### Useful next work\n\n")
		for _, it := range r.NextWork {
			fmt.Fprintf(&b, "- **%s** `%s`", it.Title, it.Plane)
			if it.Prov != "" {
				fmt.Fprintf(&b, " _[%s]_", it.Prov)
			}
			b.WriteString("\n")
			if it.Detail != "" {
				fmt.Fprintf(&b, "  - %s\n", it.Detail)
			}
		}
		b.WriteString("\n")
	}

	// Coverage: the folded plane table, so a reader sees what was and wasn't seen.
	b.WriteString("### Plane coverage\n\n")
	b.WriteString("| plane | measured | verdict | summary |\n|---|---|---|---|\n")
	for _, p := range r.Planes {
		measured := "yes"
		summary := p.Summary
		if !p.Measured {
			measured = "**NO**"
			summary = "unmeasured: " + p.Err
		}
		fmt.Fprintf(&b, "| %s | %s | %s | %s |\n", p.Name, measured, p.Verdict, dashIfEmpty(summary))
	}
	b.WriteString("\n")
	fmt.Fprintf(&b, "> **next:** %s\n", r.NextAction)
	return b.String()
}

// MarshalJSON is provided implicitly via struct tags; this helper keeps the CLI
// terse and the indentation consistent with the other panes.
func JSON(r Rollup) ([]byte, error) { return json.MarshalIndent(r, "", "  ") }

// --- coercion helpers (self-contained; mirror cadencereport idioms) --------

func orNoPayload(runErr string) string {
	if runErr != "" {
		return runErr
	}
	return "no payload"
}

func dashIfEmpty(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}

func asBool(v any) bool {
	b, ok := v.(bool)
	return ok && b
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

func asInt(v any) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	case json.Number:
		i, _ := n.Int64()
		return int(i)
	case string:
		var i int
		_, _ = fmt.Sscanf(strings.TrimSpace(n), "%d", &i)
		return i
	default:
		return 0
	}
}

// asFloatOK coerces a JSON number to float64, reporting whether a real number was
// present (so a missing rate stays unmeasured rather than collapsing to 0.0).
func asFloatOK(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}
