package operatorbrief

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cadencereport"
	"github.com/anthony-chaudhary/fak/internal/choicetriage"
	"github.com/anthony-chaudhary/fak/internal/loopfleet"
	"github.com/anthony-chaudhary/fak/internal/milestonereport"
	"github.com/anthony-chaudhary/fak/internal/programreport"
	"github.com/anthony-chaudhary/fak/internal/stallscan"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

// Schema is the stable JSON contract for the operator brief.
const Schema = "fak-operator-brief/1"

// OpsInfo carries host operations health and autonomous maintenance telemetry.
type OpsInfo struct {
	Status          string `json:"status"`
	FreeDiskBytes   uint64 `json:"free_disk_bytes,omitempty"`
	BytesReclaimed  int64  `json:"bytes_reclaimed,omitempty"`
	ProcessesReaped int    `json:"processes_reaped,omitempty"`
	DeadLocksClean  int    `json:"dead_locks_clean,omitempty"`
}

// Inputs are the report envelopes the brief summarizes. Nil reports are treated
// as missing sources, because an operator cannot reason about the whole fleet
// from a partial pane without knowing which panes are absent.
type Inputs struct {
	Workspace   string
	Commit      string
	GeneratedAt string
	Date        string
	Cadence     *cadencereport.Report
	Program     *programreport.Report
	Milestone   *milestonereport.Report
	Heaviness   *scorecard.Payload
	Fleet       *loopfleet.Report
	OSP         *OSP
	Previous    *Report

	// DebtWitnesses carries the normalized task/session witness projection. The
	// brief separates debt caught at origin (ordinary cleanup the fleet retires)
	// from debt first found later (a control gap: the origin check that should
	// have refused the artifact is missing), because the two need different
	// operator responses.
	DebtWitnesses []DebtWitnessRecord

	// Reboot carries per-sample host-reboot advice from stallscan. Advised
	// crossing surfaces as human-authority pages (approve and schedule a reboot),
	// deduplicated by (axis, process) so one sustained crossing pages once.
	Reboot []stallscan.RebootAdvice

	// Ops carries host operations telemetry and background health.
	Ops *OpsInfo

	// TriageGate selects the decenter-the-human paging policy applied during
	// Fold: "enforce" re-partitions the human bucket through choicetriage so the
	// gate pages only on genuine human-residual items; "warn" or "" (default)
	// leaves paging unchanged so the change can soak. The standalone
	// `fak operator triage` lens always enforces, independent of this field.
	TriageGate string
}

// Report is one operator-facing control-pane envelope. It deliberately separates
// human work from agent work so a busy operator sees which items need judgment
// and which items can be delegated back to the fleet.
type Report struct {
	Schema      string         `json:"schema"`
	OK          bool           `json:"ok"`
	Verdict     string         `json:"verdict"`
	Finding     string         `json:"finding"`
	Reason      string         `json:"reason"`
	NextAction  string         `json:"next_action"`
	Pace        string         `json:"pace"` // intervene | delegate | review | monitor
	Workspace   string         `json:"workspace,omitempty"`
	Commit      string         `json:"commit,omitempty"`
	GeneratedAt string         `json:"generated_at,omitempty"`
	Date        string         `json:"date,omitempty"`
	Sources     []SourceState  `json:"sources"`
	Counts      Counts         `json:"counts"`
	State       State          `json:"state"`
	Attention   Attention      `json:"attention"`
	HumanUse    HumanUse       `json:"human_use"`
	Generation  *Generation    `json:"generation,omitempty"`
	Coherence   Coherence      `json:"coherence"`
	Delta       *Delta         `json:"since_previous,omitempty"`
	Strengths   []Strength     `json:"strengths,omitempty"`
	Choices     []Choice       `json:"choices,omitempty"`
	Challenges  []Challenge    `json:"challenges,omitempty"`
	Agenda      LearningAgenda `json:"learning_agenda"`
	Learning    []Learning     `json:"learning,omitempty"`
	Human       []Item         `json:"human,omitempty"`
	Agent       []Item         `json:"agent,omitempty"`
	Watch       []Item         `json:"watch,omitempty"`
	Background  []Item         `json:"background,omitempty"`

	// OriginDebt is debt the origin control caught before handoff; LateFoundDebt
	// is debt that reached a later stage before anyone noticed. Keeping them
	// apart is the point: the first is cleanup, the second names a missing root
	// control the operator has to install.
	OriginDebt    []DebtWitnessRecord `json:"origin_debt,omitempty"`
	LateFoundDebt []DebtWitnessRecord `json:"late_found_debt,omitempty"`

	GateExit    *int   `json:"gate_exit,omitempty"`
	GateMessage string `json:"gate_message,omitempty"`
}

// SourceState records whether each upstream pane was present and measured.
type SourceState struct {
	Name    string `json:"name"`
	Schema  string `json:"schema,omitempty"`
	Status  string `json:"status"` // ok | advisory | action | unmeasured | missing
	Verdict string `json:"verdict,omitempty"`
	Finding string `json:"finding,omitempty"`
	Date    string `json:"date,omitempty"`
	Commit  string `json:"commit,omitempty"`
}

// Counts is the load summary an operator can scan before reading item detail.
type Counts struct {
	Human      int `json:"human"`
	Agent      int `json:"agent"`
	Watch      int `json:"watch"`
	Background int `json:"background"`
}

// State is the short answer an operator reads before detail.
type State struct {
	Mode        string `json:"mode"` // intervene | delegate | review | monitor
	Summary     string `json:"summary"`
	OperatorUse string `json:"operator_use"`
}

// Coherence states whether the folded source reports describe one consistent
// snapshot. A mixed snapshot is still useful, but operators should read it as a
// review signal rather than a clean whole-system state.
type Coherence struct {
	Status  string        `json:"status"` // coherent | mixed | partial
	Summary string        `json:"summary"`
	Action  string        `json:"action,omitempty"`
	Stamps  []SourceStamp `json:"stamps,omitempty"`
}

// SourceStamp is the compact evidence behind Coherence.
type SourceStamp struct {
	Name   string `json:"name"`
	Status string `json:"status"`
	Date   string `json:"date,omitempty"`
	Commit string `json:"commit,omitempty"`
}

// Delta is the temporal compression layer for operators who have already read
// the previous brief. It compares only attention-bearing buckets, not background
// state, so the operator can scan what changed without rereading the fleet.
type Delta struct {
	Status          string      `json:"status"` // changed | unchanged
	Summary         string      `json:"summary"`
	PaceFrom        string      `json:"pace_from,omitempty"`
	PaceTo          string      `json:"pace_to,omitempty"`
	PaceChanged     bool        `json:"pace_changed,omitempty"`
	NewCount        int         `json:"new_count"`
	ResolvedCount   int         `json:"resolved_count"`
	PersistentCount int         `json:"persistent_count"`
	New             []DeltaItem `json:"new,omitempty"`
	Resolved        []DeltaItem `json:"resolved,omitempty"`
	Persistent      []DeltaItem `json:"persistent,omitempty"`
}

// DeltaItem is a compact identity for an item that changed across briefs.
type DeltaItem struct {
	Bucket string `json:"bucket"`
	Source string `json:"source"`
	Title  string `json:"title"`
	Detail string `json:"detail,omitempty"`
	Action string `json:"action,omitempty"`
}

// Attention is the pacing contract for humans who cannot read every agent log.
// It says how urgently to look, how long to spend, and which sections to read
// first before delegating back to the fleet.
type Attention struct {
	Level         string   `json:"level"` // interrupt | delegate | review | none
	BudgetMinutes int      `json:"budget_minutes"`
	Cadence       string   `json:"cadence"`
	ReadOrder     []string `json:"read_order,omitempty"`
	Summary       string   `json:"summary"`
}

// HumanUse states the operating contract between people and agents for this
// snapshot. It keeps the brief from accidentally turning every agent detail into
// human work.
type HumanUse struct {
	UseHumanFor  string `json:"use_human_for"`
	LetAgentsDo  string `json:"let_agents_do"`
	Avoid        string `json:"avoid"`
	EscalateWhen string `json:"escalate_when"`
}

// Generation is the compact "what ships now vs later" readout for the operator.
// It is derived from the milestone report's generation lanes, not from issue-body
// prose, so the answer stays tied to the same witnessed roadmap fold.
type Generation struct {
	Summary             string           `json:"summary"`
	Attention           string           `json:"attention"`
	Heat                string           `json:"heat,omitempty"`
	HottestGeneration   string           `json:"hottest_generation,omitempty"`
	PromotionCandidates int              `json:"promotion_candidates,omitempty"`
	BlockedAssumptions  int              `json:"blocked_assumptions,omitempty"`
	Lanes               []GenerationLane `json:"lanes,omitempty"`
}

// GenerationLane is the operator-brief projection of one product horizon.
type GenerationLane struct {
	Generation          string  `json:"generation"`
	Tracked             int     `json:"tracked"`
	Measured            int     `json:"measured"`
	Programs            int     `json:"programs"`
	Discrete            int     `json:"discrete"`
	Closed              int     `json:"closed,omitempty"`
	Total               int     `json:"total,omitempty"`
	OpenDiscrete        int     `json:"open_discrete"`
	OverallPct          float64 `json:"overall_pct"`
	Errored             int     `json:"errored,omitempty"`
	DebtScore           int     `json:"debt_score,omitempty"`
	HeatScore           int     `json:"heat_score,omitempty"`
	HeatReason          string  `json:"heat_reason,omitempty"`
	StaleIssues         int     `json:"stale_issues,omitempty"`
	StaleAge            string  `json:"stale_age,omitempty"`
	MissingWitnesses    int     `json:"missing_witnesses,omitempty"`
	UnpromotedBets      int     `json:"unpromoted_bets,omitempty"`
	PromotionCandidates int     `json:"promotion_candidates,omitempty"`
	BlockedAssumptions  int     `json:"blocked_assumptions,omitempty"`
	ShipVelocity        string  `json:"ship_velocity,omitempty"`
	LabelShipMismatches int     `json:"label_ship_mismatches,omitempty"`
	DebtReason          string  `json:"debt_reason,omitempty"`
}

// Strength is evidence-backed work the operator can trust or delegate. It keeps
// "what is working" visible beside challenges, so the brief does not train
// humans to look only for red lights.
type Strength struct {
	Source string `json:"source"`
	Kind   string `json:"kind"` // delegable | measured | advancing
	Title  string `json:"title"`
	Detail string `json:"detail,omitempty"`
	Use    string `json:"use"`
}

// Choice is a surfaced decision point — but "surfaced" does not mean "for a
// human". Every choice is first folded through choicetriage, which decides
// whether it is really one obvious action an agent takes, an evaluation for a
// fresh context window, a ticket-sized piece of work, or the small irreducible
// remainder that genuinely needs a person. Disposition/Resolve carry that fold
// and are the load-bearing fields; Default/Options are the legacy "pick one"
// framing kept for back-compat (retirement tracked in the decenter-the-human
// epic) — read them as advisory, not as the answer.
type Choice struct {
	Source      string                   `json:"source"`
	Question    string                   `json:"question"`
	Disposition choicetriage.Disposition `json:"disposition"`
	Resolve     string                   `json:"resolve,omitempty"`
	NeedsHuman  bool                     `json:"needs_human"`
	Default     string                   `json:"default"`
	Options     []string                 `json:"options"`
	Why         string                   `json:"why,omitempty"`
	Action      string                   `json:"action,omitempty"`
}

// triage folds one surfaced choice through choicetriage and stamps the verdict
// onto it. optionCount is the number of legacy Options offered — 1 means the
// choice is fake on its face.
func (c Choice) triage(severity string, optionCount int) Choice {
	v := choicetriage.Triage(choicetriage.Signal{
		Severity:    severity,
		Source:      c.Source,
		Question:    c.Question,
		Detail:      c.Why,
		Action:      c.Action,
		OptionCount: optionCount,
	})
	c.Disposition = v.Disposition
	c.Resolve = v.Resolve
	c.NeedsHuman = v.NeedsHuman
	return c
}

// Challenge is measured friction the operator should understand even when it is
// not a page. It is the "what is hard right now" complement to Choice.
type Challenge struct {
	Source string `json:"source"`
	Kind   string `json:"kind"`
	Title  string `json:"title"`
	Detail string `json:"detail,omitempty"`
	Action string `json:"action,omitempty"`
}

// Learning is a short interpretation note for operators. It turns a brief into
// a feedback surface instead of only a status surface.
type Learning struct {
	Topic  string `json:"topic"`
	Lesson string `json:"lesson"`
	Apply  string `json:"apply,omitempty"`
}

// LearningAgenda is the "what should the operator learn now" spine. It keeps
// learning bounded to the current state instead of asking a person to absorb the
// whole fleet whenever many agents are active.
type LearningAgenda struct {
	Focus     string   `json:"focus"`
	Reason    string   `json:"reason"`
	Practice  string   `json:"practice"`
	Skip      string   `json:"skip"`
	DrillDown []string `json:"drill_down,omitempty"`
}

// Item is one piece of operator/agent work or context. Bucket is repeated in the
// item for downstream renderers that flatten the arrays.
type Item struct {
	Bucket   string `json:"bucket"`
	Source   string `json:"source"`
	Severity string `json:"severity"` // page | decision | action | watch | info
	Title    string `json:"title"`
	Detail   string `json:"detail,omitempty"`
	Action   string `json:"action,omitempty"`
}

// Fold builds the operator brief from the existing report envelopes.
func Fold(in Inputs) Report {
	r := Report{
		Schema:      Schema,
		Workspace:   strmatch.FirstNonBlank(in.Workspace, stampWorkspace(in)),
		Commit:      strmatch.FirstNonBlank(in.Commit, stampCommit(in)),
		GeneratedAt: strmatch.FirstNonBlank(in.GeneratedAt, stampGeneratedAt(in)),
		Date:        strmatch.FirstNonBlank(in.Date, stampDate(in)),
	}

	r.Sources = append(r.Sources,
		cadenceState(in.Cadence),
		programState(in.Program),
		milestoneState(in.Milestone),
	)
	if in.Heaviness != nil {
		r.Sources = append(r.Sources, heavinessState(in.Heaviness))
	}
	if in.Fleet != nil {
		r.Sources = append(r.Sources, fleetState(in.Fleet))
	}
	if in.OSP != nil {
		r.Sources = append(r.Sources, ospState(in.OSP))
	}

	// Host-reboot advice leads the human bucket: a crossed reboot high-water is the
	// most time-critical operator page (the box is on a curve to a hard freeze), so it
	// must drive NextAction ahead of the missing-report pages that empty sources add.
	addReboot(&r, in.Reboot)

	if in.Cadence == nil {
		r.addHuman("cadence", "page", "cadence report missing", "scores, maturity, work, and releases are not in this brief", "generate `fak cadence --json` and pass it with --cadence")
	} else {
		addCadence(&r, *in.Cadence)
	}
	if in.Program == nil {
		r.addHuman("program", "page", "program report missing", "ongoing optimization frontiers are not in this brief", "generate `fak program report --json` and pass it with --program")
	} else {
		addProgram(&r, *in.Program)
	}
	if in.Milestone == nil {
		r.addHuman("milestone", "page", "milestone report missing", "discrete epic and support-maturity progress are not in this brief", "generate `fak milestone report --json` and pass it with --milestone")
	} else {
		addMilestone(&r, *in.Milestone)
	}
	if in.Heaviness != nil {
		addHeaviness(&r, *in.Heaviness)
	}
	if in.Fleet != nil {
		addFleet(&r, *in.Fleet)
	}
	if in.OSP != nil {
		addOSP(&r, *in.OSP)
	}
	if in.Ops != nil {
		addOps(&r, *in.Ops)
	}
	addDebtWitnesses(&r, in.DebtWitnesses)
	r.Coherence = sourceCoherence(r.Sources)
	if r.Coherence.Status == "mixed" {
		r.addWatch("sources", "source snapshots differ", r.Coherence.Summary, r.Coherence.Action)
	}

	r.Human = dedupe(r.Human)
	r.Agent = dedupe(r.Agent)
	r.Watch = dedupe(r.Watch)
	r.Background = dedupe(r.Background)
	r.Counts = Counts{
		Human:      len(r.Human),
		Agent:      len(r.Agent),
		Watch:      len(r.Watch),
		Background: len(r.Background),
	}
	if triageEnforced(in.TriageGate) {
		// Decenter the human: fold the human bucket through choicetriage so the
		// gate pages only on genuine authority decisions. TriageHumanBucket
		// recomputes Counts and runs finalize() itself.
		r, _ = TriageHumanBucket(r)
	} else {
		r.finalize()
	}
	if in.Previous != nil {
		r.Delta = deltaFrom(*in.Previous, r)
	}
	r.deriveHumanReadout()
	return r
}

func addCadence(r *Report, c cadencereport.Report) {
	if unmeasured(c.Finding) {
		r.addHuman("cadence", "page", "cadence incomplete", c.Reason, c.NextAction)
		return
	}
	if strings.EqualFold(c.Scores.TrendDirection, "regressed") {
		r.addAgent("scores", "retire score regression", c.Scores.TrendSummary, "work the regressed scorecard worst-first")
	}
	if c.Maturity.Debt > 0 {
		detail := fmt.Sprintf("maturity debt %d, backlog %d", c.Maturity.Debt, c.Maturity.Backlog)
		if c.Maturity.RouteLane != "" && c.Maturity.RouteItem != "" {
			detail += "; route " + c.Maturity.RouteLane + ": " + c.Maturity.RouteItem
		}
		r.addAgent("maturity", "retire maturity debt", detail, c.NextAction)
	} else if c.Maturity.RouteLane != "" && c.Maturity.RouteItem != "" {
		r.addAgent("maturity", "next maturity route ready", c.Maturity.RouteLane+": "+c.Maturity.RouteItem, c.NextAction)
	}
	if releaseNeedsHuman(c.Releases) {
		r.addHuman("release", "decision", "release decision needed", releaseDetail(c.Releases), c.Releases.ActionDetail)
	} else if releaseNeedsAgent(c.Releases) {
		r.addAgent("release", "release work pending", releaseDetail(c.Releases), c.Releases.ActionDetail)
	}
	if c.Releases.CommitsBehind > 0 {
		r.addWatch("release", "published tag is behind", releaseDetail(c.Releases), "check `fak release-staleness --check` before claiming @latest is fresh")
	}
	if c.Trend != nil && c.Trend.Direction == "regressed" {
		r.addWatch("cadence", "cadence trend regressed", c.Trend.Summary, "inspect the regressed dimension before increasing fleet pace")
	}
	if c.OK && c.Finding == "cadence_recorded" {
		r.addBackground("cadence", "cadence measured", c.Reason, "keep the scheduled cadence tick")
	}
}

func addProgram(r *Report, p programreport.Report) {
	if unmeasured(p.Finding) {
		r.addHuman("program", "page", "program frontier unreadable", p.Reason, p.NextAction)
		return
	}
	if p.Programs.PartialNote != "" {
		r.addWatch("program", "partial program signal", p.Programs.PartialNote, "repair the unreadable program signal when it repeats")
	}
	for _, s := range p.Programs.Signals {
		switch {
		case s.Err != "":
			r.addWatch("program", s.Label+" signal unreadable", s.Err, "repair the source signal before trusting this frontier")
		case s.Direction == "regressed":
			r.addWatch("program", s.Label+" frontier regressed", signalDetail(s), "investigate the program's frontier witness")
		case s.Direction == "advancing":
			r.addBackground("program", s.Label+" advancing", signalDetail(s), "keep recording frontier movement")
		default:
			r.addBackground("program", s.Label+" holding", signalDetail(s), "watch trend; no operator decision")
		}
	}
	if p.Trend != nil && p.Trend.Direction == "regressed" {
		r.addWatch("program", "program trend regressed", p.Trend.Summary, p.NextAction)
	}
}

func addMilestone(r *Report, m milestonereport.Report) {
	if unmeasured(m.Finding) {
		r.addHuman("milestone", "page", "milestone roadmap unreadable", m.Reason, m.NextAction)
		return
	}
	if m.Epics.PartialNote != "" {
		r.addWatch("milestone", "partial epic signal", m.Epics.PartialNote, "repair the unreadable epic signal when it repeats")
	}
	if m.Trend != nil && m.Trend.Direction == "regressed" {
		r.addWatch("milestone", "milestone trend regressed", m.Trend.Summary, "inspect climb and roadmap deltas before changing priorities")
	}
	if len(m.Epics.Generations) > 0 {
		r.Generation = generationReadout(m.Epics.Generations)
	}
	if m.Epics.OK && m.Epics.Total > m.Epics.Closed {
		detail := fmt.Sprintf("roadmap %.1f%% across %d discrete epic(s)", m.Epics.OverallPct, m.Epics.Discrete)
		r.addAgent("milestone", "roadmap work remains", detail, m.NextAction)
	}
	if m.Maturity.OK && len(m.Maturity.Worst) > 0 {
		r.addAgent("support", "lowest support cells are visible", strings.Join(m.Maturity.Worst, "; "), "advance the lowest support cell with a witness")
	}
	if m.OK && m.Finding == "milestone_recorded" && m.Epics.Total == m.Epics.Closed {
		r.addBackground("milestone", "milestone measured", m.Reason, "keep the scheduled milestone tick")
	}
}

func addHeaviness(r *Report, h scorecard.Payload) {
	debt := corpusInt(h.Corpus, "heaviness_debt")
	pressure := corpusInt(h.Corpus, "heaviness_pressure")
	detail := heavinessDetail(h, pressure)
	switch {
	case debt > 0 || !h.OK:
		r.addAgent("heaviness", "retire operator-heaviness debt", strmatch.FirstNonBlank(h.Reason, h.Finding, detail), strmatch.FirstNonBlank(h.NextAction, "fix the operator-heaviness scorecard defects"))
	case pressure > 0:
		r.addWatch("heaviness", "operator surface pressure", detail, strmatch.FirstNonBlank(h.NextAction, "consolidate verbs/flags only when pressure keeps rising"))
	default:
		r.addBackground("heaviness", "operator surface light", detail, "keep the operator surface flat as agents add capabilities")
	}
}

// addReboot surfaces host-reboot advice as operator pages. A reboot is a genuine
// operator-authority decision, so each advised crossing is a read-only recommendation
// (approve and schedule a reboot) — never an automatic reboot. Advice is deduplicated
// by (axis, process): the Item's Title carries only those two, so two samples of one
// sustained crossing (a fresh PID, a higher count) collapse to a single page under the
// human-bucket dedupe. The "approve" in the action keeps the page HUMAN_RESIDUAL under
// the decenter-the-human triage — a host reboot is authority a person holds.
func addReboot(r *Report, advice []stallscan.RebootAdvice) {
	for _, a := range advice {
		if !a.Advised {
			continue
		}
		title := fmt.Sprintf("%s %s crossed", strmatch.FirstNonBlank(a.Process, "host"), humanAxis(a.Axis))
		r.addHuman("host-reboot", "decision", title, a.Reason,
			"approve and schedule a host reboot at a safe checkpoint; do not reboot the host automatically")
	}
}

// humanAxis renders a stallscan reboot axis token as operator-facing prose:
// "handle_high_water" -> "handle high-water". Unknown axes fall back to a plain
// underscore-to-space rewrite.
func humanAxis(axis string) string {
	a := strings.ReplaceAll(axis, "_high_water", " high-water")
	return strings.ReplaceAll(a, "_", " ")
}

func addOps(r *Report, ops OpsInfo) {
	detail := fmt.Sprintf("status=%s (reclaimed: %d bytes, reaped: %d procs, locks: %d clean)",
		ops.Status, ops.BytesReclaimed, ops.ProcessesReaped, ops.DeadLocksClean)
	r.addBackground("host-ops", "host operations "+ops.Status, detail, "keep autonomous host operations ticking")
}

func (r *Report) finalize() {
	switch {
	case len(r.Human) > 0:
		r.OK, r.Verdict, r.Finding, r.Pace = false, "ACTION", "operator_input_needed", "intervene"
		r.Reason = fmt.Sprintf("%d human item(s), %d agent item(s), %d watch item(s)", len(r.Human), len(r.Agent), len(r.Watch))
		r.NextAction = strmatch.FirstNonBlank(r.Human[0].Action, r.Human[0].Title)
	case len(r.Agent) > 0:
		r.OK, r.Verdict, r.Finding, r.Pace = true, "OK", "agent_work_ready", "delegate"
		r.Reason = fmt.Sprintf("no immediate human decision; %d agent item(s), %d watch item(s)", len(r.Agent), len(r.Watch))
		r.NextAction = strmatch.FirstNonBlank(r.Agent[0].Action, r.Agent[0].Title)
	case len(r.Watch) > 0:
		r.OK, r.Verdict, r.Finding, r.Pace = true, "OK", "operator_watchlist", "review"
		r.Reason = fmt.Sprintf("no immediate human decision; %d watch item(s)", len(r.Watch))
		r.NextAction = "review watchlist; keep agents on already-delegated work"
	default:
		r.OK, r.Verdict, r.Finding, r.Pace = true, "OK", "brief_clear", "monitor"
		r.Reason = "no immediate human decision or watch item"
		r.NextAction = "keep the scheduled report cadence"
	}
}

func (r *Report) deriveHumanReadout() {
	r.State = State{
		Mode:        r.Pace,
		Summary:     r.Reason,
		OperatorUse: operatorUse(r),
	}
	r.Attention = attentionFor(r)
	r.HumanUse = humanUseFor(r)
	r.Strengths = strengthsFor(r)
	r.Choices = choicesFor(r)
	r.Challenges = challengesFor(r)
	r.Agenda = learningAgendaFor(r)
	r.Learning = learningFor(r)
}

func operatorUse(r *Report) string {
	switch r.Pace {
	case "intervene":
		return "make the top human decision or restore the missing witness before agents proceed on that branch"
	case "delegate":
		return "let agents take the listed work; reserve human attention for policy or priority changes"
	case "review":
		return "scan the watchlist and decide whether to slow, redirect, or keep fleet pace"
	default:
		return "stay out of the loop unless a source report changes state"
	}
}

func humanUseFor(r *Report) HumanUse {
	switch r.Pace {
	case "intervene":
		return HumanUse{
			UseHumanFor:  "restore missing evidence or make the explicit policy/auth/release decision",
			LetAgentsDo:  "resume routeable work after the named witness or decision lands",
			Avoid:        "do not infer fleet health from a partial pane or manually inspect unrelated transcripts",
			EscalateWhen: "the same witness stays missing after rerun, or the top choice changes policy/priority",
		}
	case "delegate":
		return HumanUse{
			UseHumanFor:  "confirm the default delegation still matches current priorities",
			LetAgentsDo:  "work the agent bucket and produce the next witness",
			Avoid:        "do not hand-drive agent steps that already have a route and next action",
			EscalateWhen: "the default choice is wrong, priority changed, or an agent item repeats without witness",
		}
	case "review":
		return HumanUse{
			UseHumanFor:  "decide whether measured friction should slow or redirect dispatch",
			LetAgentsDo:  "continue already-delegated work while the watch signal is investigated",
			Avoid:        "do not convert every watch item into a page; watchlist is attention shaping, not interruption",
			EscalateWhen: "watch items regress across repeated briefs or threaten a release/security boundary",
		}
	default:
		return HumanUse{
			UseHumanFor:  "stay available for new decisions, not routine transcript review",
			LetAgentsDo:  "continue scheduled cadence and already-delegated work",
			Avoid:        "do not spend attention just because many agents are active",
			EscalateWhen: "a source report changes to human, watch, or unmeasured",
		}
	}
}

func deltaFrom(prev, cur Report) *Delta {
	prevItems := attentionItems(prev)
	curItems := attentionItems(cur)
	prevSeen := itemKeySet(prevItems)
	curSeen := itemKeySet(curItems)

	d := &Delta{
		PaceFrom: reportPace(prev),
		PaceTo:   reportPace(cur),
	}
	d.PaceChanged = d.PaceFrom != "" && d.PaceTo != "" && d.PaceFrom != d.PaceTo
	for _, it := range curItems {
		di := deltaItemFrom(it)
		if prevSeen[deltaKey(it)] {
			d.Persistent = append(d.Persistent, di)
			continue
		}
		d.New = append(d.New, di)
	}
	for _, it := range prevItems {
		if curSeen[deltaKey(it)] {
			continue
		}
		d.Resolved = append(d.Resolved, deltaItemFrom(it))
	}
	d.NewCount = len(d.New)
	d.ResolvedCount = len(d.Resolved)
	d.PersistentCount = len(d.Persistent)
	if d.PaceChanged || d.NewCount > 0 || d.ResolvedCount > 0 {
		d.Status = "changed"
	} else {
		d.Status = "unchanged"
	}
	d.Summary = deltaSummary(*d)
	d.New = capDeltaItems(d.New, 6)
	d.Resolved = capDeltaItems(d.Resolved, 6)
	d.Persistent = capDeltaItems(d.Persistent, 6)
	return d
}

func attentionItems(r Report) []Item {
	out := make([]Item, 0, len(r.Human)+len(r.Agent)+len(r.Watch))
	out = append(out, r.Human...)
	out = append(out, r.Agent...)
	out = append(out, r.Watch...)
	return out
}

func itemKeySet(items []Item) map[string]bool {
	out := make(map[string]bool, len(items))
	for _, it := range items {
		out[deltaKey(it)] = true
	}
	return out
}

func deltaKey(it Item) string {
	return strings.Join([]string{it.Bucket, it.Source, it.Title, it.Action}, "\x00")
}

func deltaItemFrom(it Item) DeltaItem {
	return DeltaItem{
		Bucket: it.Bucket,
		Source: it.Source,
		Title:  it.Title,
		Detail: it.Detail,
		Action: it.Action,
	}
}

func reportPace(r Report) string {
	if r.Pace != "" {
		return r.Pace
	}
	return r.State.Mode
}

func deltaSummary(d Delta) string {
	base := fmt.Sprintf("%d new, %d resolved, %d still present", d.NewCount, d.ResolvedCount, d.PersistentCount)
	if d.PaceChanged {
		return fmt.Sprintf("%s; pace %s -> %s", base, d.PaceFrom, d.PaceTo)
	}
	if d.Status == "unchanged" {
		if d.PaceTo != "" {
			return fmt.Sprintf("no new or resolved human/agent/watch items; pace stays %s", d.PaceTo)
		}
		return "no new or resolved human/agent/watch items"
	}
	return base
}

func capDeltaItems(items []DeltaItem, n int) []DeltaItem {
	if len(items) <= n {
		return items
	}
	return items[:n]
}

func attentionFor(r *Report) Attention {
	switch r.Pace {
	case "intervene":
		a := Attention{
			Level:         "interrupt",
			BudgetMinutes: boundedAttentionMinutes(15, 5, len(r.Human), 30),
			Cadence:       "now",
			ReadOrder:     []string{"human", "choices", "challenges", "sources"},
			Summary:       "resolve the top human item or restore its witness before widening dispatch",
		}
		return withDeltaReadOrder(a, r.Delta)
	case "delegate":
		a := Attention{
			Level:         "delegate",
			BudgetMinutes: 5,
			Cadence:       "at dispatch boundary",
			ReadOrder:     []string{"choices", "agent", "strengths", "learning"},
			Summary:       "confirm the default delegation choice, then let agents work the listed items",
		}
		return withDeltaReadOrder(a, r.Delta)
	case "review":
		a := Attention{
			Level:         "review",
			BudgetMinutes: boundedAttentionMinutes(10, 3, len(r.Watch), 20),
			Cadence:       "next operator review",
			ReadOrder:     []string{"challenges", "watch", "choices", "learning"},
			Summary:       "scan measured friction and adjust pace only if the default review choice is wrong",
		}
		return withDeltaReadOrder(a, r.Delta)
	default:
		a := Attention{
			Level:         "none",
			BudgetMinutes: 0,
			Cadence:       "scheduled cadence only",
			ReadOrder:     []string{"state", "sources", "background"},
			Summary:       "do not inspect every transcript; wait for a source report to change",
		}
		return withDeltaReadOrder(a, r.Delta)
	}
}

func withDeltaReadOrder(a Attention, d *Delta) Attention {
	if d == nil || d.Status != "changed" {
		return a
	}
	a.ReadOrder = prependUnique("since_previous", a.ReadOrder)
	if a.Level == "none" {
		a.Level = "review"
		a.BudgetMinutes = 3
		a.Cadence = "next operator review"
		a.Summary = "scan what changed since the previous brief before staying out of the loop"
	}
	return a
}

func prependUnique(v string, vals []string) []string {
	if v == "" || containsString(vals, v) {
		return vals
	}
	out := make([]string, 0, len(vals)+1)
	out = append(out, v)
	out = append(out, vals...)
	return out
}

func boundedAttentionMinutes(base, perExtra, items, max int) int {
	if items <= 0 {
		return base
	}
	n := base + perExtra*(items-1)
	if n > max {
		return max
	}
	return n
}

func strengthsFor(r *Report) []Strength {
	var out []Strength
	for _, it := range r.Agent {
		out = append(out, Strength{
			Source: it.Source,
			Kind:   "delegable",
			Title:  it.Title,
			Detail: it.Detail,
			Use:    "delegate this to agents unless priority or policy changed",
		})
	}
	for _, it := range r.Background {
		kind := "measured"
		if strings.Contains(strings.ToLower(it.Title), "advancing") {
			kind = "advancing"
		}
		out = append(out, Strength{
			Source: it.Source,
			Kind:   kind,
			Title:  it.Title,
			Detail: it.Detail,
			Use:    "do not spend operator attention here unless the source changes",
		})
	}
	return dedupeStrengths(out)
}

func choicesFor(r *Report) []Choice {
	var out []Choice
	for _, it := range r.Human {
		out = append(out, Choice{
			Source:   it.Source,
			Question: it.Title,
			Default:  "intervene",
			Options:  []string{"intervene now", "delegate after evidence lands", "hold this branch"},
			Why:      it.Detail,
			Action:   it.Action,
		}.triage(it.Severity, 3))
	}
	if len(out) > 0 {
		return out
	}
	if len(r.Agent) > 0 {
		top := r.Agent[0]
		return []Choice{Choice{
			Source:   top.Source,
			Question: "let agents continue with " + top.Title + "?",
			Default:  "delegate",
			Options:  []string{"delegate to agents", "pause for operator review"},
			Why:      top.Detail,
			Action:   top.Action,
		}.triage(top.Severity, 2)}
	}
	if len(r.Watch) > 0 {
		top := r.Watch[0]
		return []Choice{Choice{
			Source:   top.Source,
			Question: "keep fleet pace while watching " + top.Title + "?",
			Default:  "review",
			Options:  []string{"keep pace", "slow dispatch", "investigate now"},
			Why:      top.Detail,
			Action:   top.Action,
		}.triage(top.Severity, 3)}
	}
	return nil
}

func learningFor(r *Report) []Learning {
	var out []Learning
	if len(r.Human) > 0 {
		out = append(out, Learning{
			Topic:  "witness before judgment",
			Lesson: "a missing or unmeasured pane is not a system verdict; it means the operator is looking at an incomplete witness set",
			Apply:  "restore the named source report before changing fleet direction",
		})
	}
	if len(r.Agent) > 0 {
		out = append(out, Learning{
			Topic:  "delegation boundary",
			Lesson: "agent bucket items are routeable work; a human is useful for priorities and policy, not for hand-driving each step",
			Apply:  "let agents work the top agent item unless the default choice is wrong",
		})
	}
	if len(r.Watch) > 0 {
		out = append(out, Learning{
			Topic:  "pace control",
			Lesson: "watchlist items are measured friction; they should tune attention and dispatch pace without automatically paging a human",
			Apply:  "review the watch choice before speeding up or widening dispatch",
		})
	}
	if len(r.Human) == 0 && len(r.Agent) == 0 && len(r.Watch) == 0 {
		out = append(out, Learning{
			Topic:  "negative signal",
			Lesson: "a clear brief is a signal to stay out of the loop, not an invitation to inspect every agent transcript",
			Apply:  "keep the scheduled cadence and wait for a source report to change",
		})
	}
	return out
}

func learningAgendaFor(r *Report) LearningAgenda {
	switch r.Pace {
	case "intervene":
		top := firstItem(r.Human)
		return LearningAgenda{
			Focus:     "witness before judgment",
			Reason:    "the brief has a human bucket, so at least one source is missing, unmeasured, or asking for an explicit decision",
			Practice:  strmatch.FirstNonBlank(top.Action, "restore the missing source report, then rerun `fak operator brief --collect`"),
			Skip:      "skip unrelated transcript review until the top human item has a witness",
			DrillDown: agendaDrillDown(top, "human", "sources"),
		}
	case "delegate":
		top := firstItem(r.Agent)
		return LearningAgenda{
			Focus:     "delegation boundary",
			Reason:    "the top work is routeable agent work, not a scarce human judgment",
			Practice:  strmatch.FirstNonBlank(top.Action, "let agents work the top agent item and ask only for the next witness"),
			Skip:      "skip hand-driving steps that already have an agent-owned next action",
			DrillDown: agendaDrillDown(top, "agent", "strengths"),
		}
	case "review":
		top := firstItem(r.Watch)
		return LearningAgenda{
			Focus:     "watchlist vs page",
			Reason:    "the brief has measured friction but no immediate human decision",
			Practice:  strmatch.FirstNonBlank(top.Action, "compare the watch item against the next brief before slowing dispatch"),
			Skip:      "skip converting every watch item into an interruption",
			DrillDown: agendaDrillDown(top, "watch", "challenges"),
		}
	default:
		return LearningAgenda{
			Focus:     "negative signal discipline",
			Reason:    "the brief has no human, agent, or watch item",
			Practice:  "keep the scheduled report cadence and wait for a source report to change",
			Skip:      "skip reading agent transcripts just because the fleet is busy",
			DrillDown: []string{"state", "sources", "background"},
		}
	}
}

func challengesFor(r *Report) []Challenge {
	var out []Challenge
	for _, it := range r.Human {
		out = append(out, Challenge{
			Source: it.Source,
			Kind:   challengeKind(it),
			Title:  it.Title,
			Detail: it.Detail,
			Action: it.Action,
		})
	}
	for _, it := range r.Watch {
		out = append(out, Challenge{
			Source: it.Source,
			Kind:   "watch",
			Title:  it.Title,
			Detail: it.Detail,
			Action: it.Action,
		})
	}
	return out
}

func firstItem(items []Item) Item {
	if len(items) == 0 {
		return Item{}
	}
	return items[0]
}

func agendaDrillDown(top Item, fallback ...string) []string {
	out := []string{}
	if top.Bucket != "" {
		out = append(out, top.Bucket)
	}
	if top.Source != "" {
		out = append(out, top.Source)
	}
	for _, v := range fallback {
		if v != "" && !containsString(out, v) {
			out = append(out, v)
		}
	}
	return out
}

func dedupeStrengths(items []Strength) []Strength {
	seen := map[string]bool{}
	out := make([]Strength, 0, len(items))
	for _, it := range items {
		key := strings.Join([]string{it.Source, it.Kind, it.Title, it.Use}, "\x00")
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, it)
	}
	return out
}

func challengeKind(it Item) string {
	if it.Severity == "decision" {
		return "operator_decision"
	}
	if it.Severity == "page" {
		return "missing_or_unmeasured_signal"
	}
	return it.Severity
}

func (r *Report) addHuman(source, severity, title, detail, action string) {
	r.Human = append(r.Human, Item{Bucket: "human", Source: source, Severity: severity, Title: title, Detail: detail, Action: action})
}

func (r *Report) addAgent(source, title, detail, action string) {
	r.Agent = append(r.Agent, Item{Bucket: "agent", Source: source, Severity: "action", Title: title, Detail: detail, Action: action})
}

func (r *Report) addWatch(source, title, detail, action string) {
	r.Watch = append(r.Watch, Item{Bucket: "watch", Source: source, Severity: "watch", Title: title, Detail: detail, Action: action})
}

func (r *Report) addBackground(source, title, detail, action string) {
	r.Background = append(r.Background, Item{Bucket: "background", Source: source, Severity: "info", Title: title, Detail: detail, Action: action})
}

// CheckGate is the paging gate for an operator brief. It fails only when the
// brief found work that needs a human operator; agent work and watchlist items
// remain successful measured reports.
func CheckGate(r Report) (int, string) {
	if len(r.Human) > 0 {
		return 1, "OPERATOR ACTION: " + r.Reason + " - " + r.NextAction
	}
	return 0, "OPERATOR OK: " + r.Reason
}

// triageEnforced reports whether the decenter-the-human gate should re-partition
// the human bucket during Fold. Only "enforce" flips paging; "warn"/"" soak.
func triageEnforced(mode string) bool {
	return strings.EqualFold(strings.TrimSpace(mode), "enforce")
}

// itemSignal maps an operator Item onto the choicetriage Signal, using the same
// field correspondence as Choice.triage (Title->Question, Detail, Action) so the
// gate and the surfaced Choice agree on every item's disposition.
func itemSignal(it Item) choicetriage.Signal {
	return choicetriage.Signal{
		Severity: it.Severity,
		Source:   it.Source,
		Question: it.Title,
		Detail:   it.Detail,
		Action:   it.Action,
	}
}

// Reassignment records one human-bucket item that choicetriage judged does not
// truly need a person, and where the fold routed it instead. Reason carries the
// deciding rung of the choicetriage verdict — the one-line "why this is not a
// person's call" (obvious action / knowable evaluation / oversized scope) — so
// an operator reading the lens sees the ground for each reassignment, not just
// its destination.
type Reassignment struct {
	Source      string                   `json:"source"`
	Title       string                   `json:"title"`
	Disposition choicetriage.Disposition `json:"disposition"`
	Reason      string                   `json:"reason,omitempty"`
	Resolve     string                   `json:"resolve"`
}

// TriageHumanBucket re-partitions r.Human through choicetriage so the paging gate
// pages only on genuine human-residual decisions. Every human item whose
// disposition is not HumanResidual is moved to the agent bucket, carrying its
// triage Resolve as the next action; Counts and the finalize() verdict are
// recomputed so CheckGate follows. Pure: state in, state out. This is the
// mechanism behind both the enforce-mode Fold and the `fak operator triage` lens.
func TriageHumanBucket(r Report) (Report, []Reassignment) {
	kept := make([]Item, 0, len(r.Human))
	var moved []Reassignment
	for _, it := range r.Human {
		v := choicetriage.Triage(itemSignal(it))
		if v.NeedsHuman {
			kept = append(kept, it)
			continue
		}
		// Not a person's call: route it back to the fleet. FRESH_CONTEXT items
		// carry "open a fresh context window..." as the action; TAKE_OBVIOUS and
		// FILE_TICKET carry their concrete next move.
		it.Bucket = "agent"
		it.Severity = "action"
		if strings.TrimSpace(v.Resolve) != "" {
			it.Action = v.Resolve
		}
		r.Agent = append(r.Agent, it)
		moved = append(moved, Reassignment{
			Source:      it.Source,
			Title:       it.Title,
			Disposition: v.Disposition,
			Reason:      v.Reason,
			Resolve:     v.Resolve,
		})
	}
	r.Human = kept
	r.Counts = Counts{Human: len(r.Human), Agent: len(r.Agent), Watch: len(r.Watch), Background: len(r.Background)}
	r.finalize()
	return r, moved
}

// Reconcile recomputes the human-readout projections (State, Attention, Choices,
// Challenges, Agenda, Learning) from the current buckets. Callers that mutate the
// buckets after Fold — e.g. the triage lens applying TriageHumanBucket to a loaded
// brief — call this so the derived views match the reconciled buckets.
func (r Report) Reconcile() Report {
	r.deriveHumanReadout()
	return r
}

// TriageSelfcheck proves the decenter-the-human gate with no I/O, no clock, and no
// network. It folds a synthetic brief carrying two human-bucket items — a genuine
// authority decision (a release approval) and a knowable evaluation (an unclear
// score dimension) — through TriageHumanBucket and asserts the load-bearing
// invariants: the authority item still pages and stays in Human; the evaluation is
// routed to the agent bucket as FRESH_CONTEXT and stops paging. A brief whose only
// human item is that evaluation must gate clean after triage.
func TriageSelfcheck() error {
	r := Report{Schema: Schema}
	r.addHuman("release", "decision", "release decision needed", "operator must approve the tagged build before publish", "approve the release")
	r.addHuman("cadence", "page", "score dimension unclear", "one dimension moved but the cause is not obvious", "")

	if code, _ := CheckGate(r); code == 0 {
		return fmt.Errorf("pre-triage gate should page on 2 human items, got exit 0")
	}

	triaged, moved := TriageHumanBucket(r)
	if len(triaged.Human) != 1 {
		return fmt.Errorf("want 1 residual human item after triage, got %d", len(triaged.Human))
	}
	if triaged.Human[0].Source != "release" {
		return fmt.Errorf("want the release authority decision to remain human, got source %q", triaged.Human[0].Source)
	}
	if len(moved) != 1 {
		return fmt.Errorf("want exactly 1 item routed to the fleet, got %d", len(moved))
	}
	if moved[0].Source != "cadence" {
		return fmt.Errorf("want the unclear score dimension routed to the fleet, got source %q", moved[0].Source)
	}
	if moved[0].Disposition != choicetriage.FreshContext {
		return fmt.Errorf("want the evaluation to route as FRESH_CONTEXT, got %s", moved[0].Disposition)
	}
	if strings.TrimSpace(moved[0].Reason) == "" {
		return fmt.Errorf("a reassignment must carry the deciding choicetriage rung, got empty Reason")
	}
	if code, _ := CheckGate(triaged); code != 1 {
		return fmt.Errorf("post-triage gate should still page on the residual authority decision, got exit %d", code)
	}

	only := Report{Schema: Schema}
	only.addHuman("cadence", "page", "score dimension unclear", "one dimension moved but the cause is not obvious", "")
	onlyTriaged, _ := TriageHumanBucket(only)
	if code, _ := CheckGate(onlyTriaged); code != 0 {
		return fmt.Errorf("a brief whose only human item is a knowable evaluation should gate clean after triage, got exit %d", code)
	}
	return nil
}

// WithGate returns a copy reconciled to a CheckGate decision.
func (r Report) WithGate(code int, message string) Report {
	q := r
	q.OK = code == 0
	if code == 0 {
		q.Verdict = "OK"
	} else {
		q.Verdict = "ACTION"
	}
	c := code
	q.GateExit = &c
	q.GateMessage = message
	return q
}

func sourceCoherence(srcs []SourceState) Coherence {
	c := Coherence{Status: "coherent"}
	var missing []string
	dates := map[string]bool{}
	commits := map[string]bool{}
	for _, s := range srcs {
		c.Stamps = append(c.Stamps, SourceStamp{Name: s.Name, Status: s.Status, Date: s.Date, Commit: s.Commit})
		if s.Status == "missing" {
			missing = append(missing, s.Name)
			continue
		}
		if strings.TrimSpace(s.Date) != "" {
			dates[s.Date] = true
		}
		if strings.TrimSpace(s.Commit) != "" {
			commits[s.Commit] = true
		}
	}
	switch {
	case len(missing) > 0:
		c.Status = "partial"
		c.Summary = "missing " + strings.Join(missing, ", ")
		c.Action = "provide every source report or use --collect for one folded snapshot"
	case len(dates) > 1 || len(commits) > 1:
		c.Status = "mixed"
		c.Summary = fmt.Sprintf("%d date stamp(s), %d commit stamp(s)", len(dates), len(commits))
		c.Action = "regenerate source reports together or use --collect before treating the brief as one snapshot"
	default:
		c.Summary = "source reports share one snapshot stamp"
	}
	return c
}

func cadenceState(r *cadencereport.Report) SourceState {
	if r == nil {
		return SourceState{Name: "cadence", Status: "missing"}
	}
	return SourceState{Name: "cadence", Schema: r.Schema, Status: reportStatus(r.OK, r.Finding), Verdict: r.Verdict, Finding: r.Finding, Date: r.Date, Commit: r.Commit}
}

func programState(r *programreport.Report) SourceState {
	if r == nil {
		return SourceState{Name: "program", Status: "missing"}
	}
	return SourceState{Name: "program", Schema: r.Schema, Status: reportStatus(r.OK, r.Finding), Verdict: r.Verdict, Finding: r.Finding, Date: r.Date, Commit: r.Commit}
}

func milestoneState(r *milestonereport.Report) SourceState {
	if r == nil {
		return SourceState{Name: "milestone", Status: "missing"}
	}
	return SourceState{Name: "milestone", Schema: r.Schema, Status: reportStatus(r.OK, r.Finding), Verdict: r.Verdict, Finding: r.Finding, Date: r.Date, Commit: r.Commit}
}

func heavinessState(r *scorecard.Payload) SourceState {
	if r == nil {
		return SourceState{Name: "heaviness", Status: "missing"}
	}
	return SourceState{Name: "heaviness", Schema: r.Schema, Status: reportStatus(r.OK, r.Finding), Verdict: r.Verdict, Finding: r.Finding}
}

func reportStatus(ok bool, finding string) string {
	switch {
	case unmeasured(finding):
		return "unmeasured"
	case strings.Contains(finding, "advisory"):
		return "advisory"
	case !ok:
		return "action"
	default:
		return "ok"
	}
}

func unmeasured(finding string) bool {
	return strings.Contains(finding, "unmeasured")
}

func releaseNeedsHuman(r cadencereport.Releases) bool {
	if r.Err != "" {
		return true
	}
	kind := strings.ToLower(strings.TrimSpace(r.ActionKind))
	if kind == "" || kind == "wait" || kind == "none" {
		return false
	}
	return strings.Contains(kind, "confirm") ||
		strings.Contains(kind, "manual") ||
		strings.Contains(kind, "hold") ||
		strings.Contains(kind, "decision") ||
		strings.Contains(kind, "auth")
}

func releaseNeedsAgent(r cadencereport.Releases) bool {
	kind := strings.ToLower(strings.TrimSpace(r.ActionKind))
	if kind == "" || kind == "wait" || kind == "none" || releaseNeedsHuman(r) {
		return false
	}
	return !r.OK || kind != ""
}

func releaseDetail(r cadencereport.Releases) string {
	parts := []string{"version " + strmatch.DashIfBlank(r.Version), "next " + strmatch.DashIfBlank(r.ActionKind)}
	if r.CommitsBehind > 0 {
		parts = append(parts, fmt.Sprintf("@latest %d commit(s) behind", r.CommitsBehind))
	}
	if r.PublishVerdict != "" {
		parts = append(parts, "publish "+r.PublishVerdict)
	}
	if r.Err != "" {
		parts = append(parts, "error "+r.Err)
	}
	return strings.Join(parts, "; ")
}

func signalDetail(s programreport.Signal) string {
	parts := []string{"frontier " + strmatch.DashIfBlank(s.Frontier), s.Direction}
	if s.Activity != 0 {
		parts = append(parts, fmt.Sprintf("%d shipped move(s)", s.Activity))
	}
	if s.Window != "" {
		parts = append(parts, "window "+s.Window)
	}
	if s.Note != "" {
		parts = append(parts, "fence "+s.Note)
	}
	return strings.Join(parts, "; ")
}

func heavinessDetail(h scorecard.Payload, pressure int) string {
	parts := []string{fmt.Sprintf("heaviness_pressure %d", pressure)}
	for _, key := range []string{"verbs", "front_door_flags", "refusal_reasons"} {
		if v, ok := h.Corpus[key]; ok {
			parts = append(parts, fmt.Sprintf("%s %d", key, anyInt(v)))
		}
	}
	if h.Finding != "" {
		parts = append(parts, h.Finding)
	}
	return strings.Join(parts, "; ")
}

func corpusInt(c map[string]any, key string) int {
	if c == nil {
		return 0
	}
	return anyInt(c[key])
}

func anyInt(v any) int {
	switch x := v.(type) {
	case int:
		return x
	case int64:
		return int(x)
	case float64:
		if x < 0 {
			return int(x - 0.5)
		}
		return int(x + 0.5)
	case float32:
		if x < 0 {
			return int(x - 0.5)
		}
		return int(x + 0.5)
	default:
		return 0
	}
}

func dedupe(items []Item) []Item {
	seen := map[string]bool{}
	out := make([]Item, 0, len(items))
	for _, it := range items {
		key := strings.Join([]string{it.Bucket, it.Source, it.Title, it.Action}, "\x00")
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, it)
	}
	return out
}

func stampWorkspace(in Inputs) string {
	for _, v := range []string{
		stringOrEmpty(in.Cadence, func(r *cadencereport.Report) string { return r.Workspace }),
		stringOrEmpty(in.Program, func(r *programreport.Report) string { return r.Workspace }),
		stringOrEmpty(in.Milestone, func(r *milestonereport.Report) string { return r.Workspace }),
		stringOrEmpty(in.Heaviness, func(r *scorecard.Payload) string { return r.Workspace }),
	} {
		if v != "" {
			return v
		}
	}
	return ""
}

func stampCommit(in Inputs) string {
	for _, v := range []string{
		stringOrEmpty(in.Cadence, func(r *cadencereport.Report) string { return r.Commit }),
		stringOrEmpty(in.Program, func(r *programreport.Report) string { return r.Commit }),
		stringOrEmpty(in.Milestone, func(r *milestonereport.Report) string { return r.Commit }),
	} {
		if v != "" {
			return v
		}
	}
	return ""
}

func stampGeneratedAt(in Inputs) string {
	for _, v := range []string{
		stringOrEmpty(in.Cadence, func(r *cadencereport.Report) string { return r.GeneratedAt }),
		stringOrEmpty(in.Program, func(r *programreport.Report) string { return r.GeneratedAt }),
		stringOrEmpty(in.Milestone, func(r *milestonereport.Report) string { return r.GeneratedAt }),
	} {
		if v != "" {
			return v
		}
	}
	return ""
}

func stampDate(in Inputs) string {
	for _, v := range []string{
		stringOrEmpty(in.Cadence, func(r *cadencereport.Report) string { return r.Date }),
		stringOrEmpty(in.Program, func(r *programreport.Report) string { return r.Date }),
		stringOrEmpty(in.Milestone, func(r *milestonereport.Report) string { return r.Date }),
	} {
		if v != "" {
			return v
		}
	}
	return ""
}

func stringOrEmpty[T any](ptr *T, f func(*T) string) string {
	if ptr == nil {
		return ""
	}
	return f(ptr)
}

func containsString(vals []string, want string) bool {
	for _, v := range vals {
		if v == want {
			return true
		}
	}
	return false
}
