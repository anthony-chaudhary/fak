package operatorbrief

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cadencereport"
	"github.com/anthony-chaudhary/fak/internal/milestonereport"
	"github.com/anthony-chaudhary/fak/internal/programreport"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema is the stable JSON contract for the operator brief.
const Schema = "fak-operator-brief/1"

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
	Previous    *Report
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
	GateExit    *int           `json:"gate_exit,omitempty"`
	GateMessage string         `json:"gate_message,omitempty"`
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
	Summary   string           `json:"summary"`
	Attention string           `json:"attention"`
	Lanes     []GenerationLane `json:"lanes,omitempty"`
}

// GenerationLane is the operator-brief projection of one product horizon.
type GenerationLane struct {
	Generation          string  `json:"generation"`
	Tracked             int     `json:"tracked"`
	Measured            int     `json:"measured"`
	Programs            int     `json:"programs"`
	Discrete            int     `json:"discrete"`
	OpenDiscrete        int     `json:"open_discrete"`
	OverallPct          float64 `json:"overall_pct"`
	Errored             int     `json:"errored,omitempty"`
	DebtScore           int     `json:"debt_score,omitempty"`
	StaleIssues         int     `json:"stale_issues,omitempty"`
	MissingWitnesses    int     `json:"missing_witnesses,omitempty"`
	UnpromotedBets      int     `json:"unpromoted_bets,omitempty"`
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

// Choice is a concrete operator decision point. The brief only creates choices
// when the human can actually decide something; agent-work details stay in Agent.
type Choice struct {
	Source   string   `json:"source"`
	Question string   `json:"question"`
	Default  string   `json:"default"`
	Options  []string `json:"options"`
	Why      string   `json:"why,omitempty"`
	Action   string   `json:"action,omitempty"`
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
		Workspace:   firstNonEmpty(in.Workspace, stampWorkspace(in)),
		Commit:      firstNonEmpty(in.Commit, stampCommit(in)),
		GeneratedAt: firstNonEmpty(in.GeneratedAt, stampGeneratedAt(in)),
		Date:        firstNonEmpty(in.Date, stampDate(in)),
	}

	r.Sources = append(r.Sources,
		cadenceState(in.Cadence),
		programState(in.Program),
		milestoneState(in.Milestone),
	)
	if in.Heaviness != nil {
		r.Sources = append(r.Sources, heavinessState(in.Heaviness))
	}

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
	r.finalize()
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

func generationReadout(rows []milestonereport.GenerationRow) *Generation {
	if len(rows) == 0 {
		return nil
	}
	out := &Generation{}
	var nowOpen, laterTracked, unreadable, debt int
	for _, row := range rows {
		lane := GenerationLane{
			Generation:          row.Generation,
			Tracked:             row.Tracked,
			Measured:            row.Measured,
			Programs:            row.Programs,
			Discrete:            row.Discrete,
			OpenDiscrete:        maxInt(0, row.Total-row.Closed),
			OverallPct:          row.OverallPct,
			Errored:             row.Errored,
			DebtScore:           row.DebtScore,
			StaleIssues:         row.StaleIssues,
			MissingWitnesses:    row.MissingWitnesses,
			UnpromotedBets:      row.UnpromotedBets,
			LabelShipMismatches: row.LabelShipMismatches,
			DebtReason:          row.DebtReason,
		}
		out.Lanes = append(out.Lanes, lane)
		if row.Generation == "now" {
			nowOpen = lane.OpenDiscrete
		} else if row.Generation != "unclassified" {
			laterTracked += lane.Tracked
		}
		unreadable += lane.Errored
		debt += lane.DebtScore
	}
	switch {
	case unreadable > 0:
		out.Summary = fmt.Sprintf("generation lanes have %d unreadable item(s), debt %d; do not promote or demote from this readout yet", unreadable, debt)
		out.Attention = "repair the unreadable generation lane signal before changing dispatch focus"
	case nowOpen > 0:
		out.Summary = fmt.Sprintf("ship-now lane has %d open discrete item(s); %d later-horizon item(s) stay visible; generation debt %d", nowOpen, laterTracked, debt)
		out.Attention = "delegate from the now lane first; review later lanes only when promotion evidence changes"
	case laterTracked > 0:
		out.Summary = fmt.Sprintf("ship-now lane is clear; %d later-horizon item(s) remain as bets or foundations; generation debt %d", laterTracked, debt)
		out.Attention = "no extra human attention unless a later item asks for promotion into now"
	default:
		out.Summary = "generation lanes are clear or unclassified only"
		out.Attention = "preserve generation labels and project fields; no attention budget needed"
	}
	return out
}

func addHeaviness(r *Report, h scorecard.Payload) {
	debt := corpusInt(h.Corpus, "heaviness_debt")
	pressure := corpusInt(h.Corpus, "heaviness_pressure")
	detail := heavinessDetail(h, pressure)
	switch {
	case debt > 0 || !h.OK:
		r.addAgent("heaviness", "retire operator-heaviness debt", firstNonEmpty(h.Reason, h.Finding, detail), firstNonEmpty(h.NextAction, "fix the operator-heaviness scorecard defects"))
	case pressure > 0:
		r.addWatch("heaviness", "operator surface pressure", detail, firstNonEmpty(h.NextAction, "consolidate verbs/flags only when pressure keeps rising"))
	default:
		r.addBackground("heaviness", "operator surface light", detail, "keep the operator surface flat as agents add capabilities")
	}
}

func (r *Report) finalize() {
	switch {
	case len(r.Human) > 0:
		r.OK, r.Verdict, r.Finding, r.Pace = false, "ACTION", "operator_input_needed", "intervene"
		r.Reason = fmt.Sprintf("%d human item(s), %d agent item(s), %d watch item(s)", len(r.Human), len(r.Agent), len(r.Watch))
		r.NextAction = firstNonEmpty(r.Human[0].Action, r.Human[0].Title)
	case len(r.Agent) > 0:
		r.OK, r.Verdict, r.Finding, r.Pace = true, "OK", "agent_work_ready", "delegate"
		r.Reason = fmt.Sprintf("no immediate human decision; %d agent item(s), %d watch item(s)", len(r.Agent), len(r.Watch))
		r.NextAction = firstNonEmpty(r.Agent[0].Action, r.Agent[0].Title)
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
		})
	}
	if len(out) > 0 {
		return out
	}
	if len(r.Agent) > 0 {
		top := r.Agent[0]
		return []Choice{{
			Source:   top.Source,
			Question: "let agents continue with " + top.Title + "?",
			Default:  "delegate",
			Options:  []string{"delegate to agents", "pause for operator review"},
			Why:      top.Detail,
			Action:   top.Action,
		}}
	}
	if len(r.Watch) > 0 {
		top := r.Watch[0]
		return []Choice{{
			Source:   top.Source,
			Question: "keep fleet pace while watching " + top.Title + "?",
			Default:  "review",
			Options:  []string{"keep pace", "slow dispatch", "investigate now"},
			Why:      top.Detail,
			Action:   top.Action,
		}}
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
			Practice:  firstNonEmpty(top.Action, "restore the missing source report, then rerun `fak operator brief --collect`"),
			Skip:      "skip unrelated transcript review until the top human item has a witness",
			DrillDown: agendaDrillDown(top, "human", "sources"),
		}
	case "delegate":
		top := firstItem(r.Agent)
		return LearningAgenda{
			Focus:     "delegation boundary",
			Reason:    "the top work is routeable agent work, not a scarce human judgment",
			Practice:  firstNonEmpty(top.Action, "let agents work the top agent item and ask only for the next witness"),
			Skip:      "skip hand-driving steps that already have an agent-owned next action",
			DrillDown: agendaDrillDown(top, "agent", "strengths"),
		}
	case "review":
		top := firstItem(r.Watch)
		return LearningAgenda{
			Focus:     "watchlist vs page",
			Reason:    "the brief has measured friction but no immediate human decision",
			Practice:  firstNonEmpty(top.Action, "compare the watch item against the next brief before slowing dispatch"),
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

// Render produces the human snapshot.
func Render(r Report) string {
	lines := []string{
		fmt.Sprintf("operator brief - %s (%s)  @%s  %s", r.Verdict, r.Finding, dashIfEmpty(r.Commit), dashIfEmpty(r.Date)),
		"",
		fmt.Sprintf("  pace       %s; human %d, agent %d, watch %d, background %d",
			r.Pace, r.Counts.Human, r.Counts.Agent, r.Counts.Watch, r.Counts.Background),
		"  state      " + dashIfEmpty(r.State.OperatorUse),
		"  attention  " + renderAttention(r.Attention),
		"  human use  " + renderHumanUse(r.HumanUse),
		"  coherence  " + renderCoherence(r.Coherence),
		"  sources    " + renderSources(r.Sources),
	}
	lines = appendGeneration(lines, r.Generation)
	lines = appendDelta(lines, r.Delta)
	lines = appendStrengths(lines, r.Strengths)
	lines = appendChoices(lines, r.Choices)
	lines = appendChallenges(lines, r.Challenges)
	lines = appendLearningAgenda(lines, r.Agenda)
	lines = appendLearning(lines, r.Learning)
	lines = appendSection(lines, "human", r.Human)
	lines = appendSection(lines, "agent", r.Agent)
	lines = appendSection(lines, "watch", r.Watch)
	lines = appendSection(lines, "background", capItems(r.Background, 4))
	lines = append(lines, "", "  -> "+r.NextAction)
	return strings.Join(lines, "\n")
}

func appendGeneration(lines []string, g *Generation) []string {
	if g == nil {
		return lines
	}
	lines = append(lines, "  generation "+g.Summary)
	if g.Attention != "" {
		lines = append(lines, "              "+g.Attention)
	}
	for _, lane := range g.Lanes {
		if lane.Tracked == 0 {
			lines = append(lines, fmt.Sprintf("              %s: 0 tracked", lane.Generation))
			continue
		}
		parts := []string{fmt.Sprintf("%s: %d tracked", lane.Generation, lane.Tracked)}
		if lane.OpenDiscrete > 0 || lane.Discrete > 0 {
			parts = append(parts, fmt.Sprintf("%d open discrete", lane.OpenDiscrete))
			parts = append(parts, fmt.Sprintf("%.1f%% discrete", lane.OverallPct))
		}
		if lane.Programs > 0 {
			parts = append(parts, fmt.Sprintf("%d ongoing", lane.Programs))
		}
		if lane.Errored > 0 {
			parts = append(parts, fmt.Sprintf("%d unreadable", lane.Errored))
		}
		if lane.DebtScore > 0 {
			if lane.DebtReason != "" {
				parts = append(parts, fmt.Sprintf("debt %d (%s)", lane.DebtScore, lane.DebtReason))
			} else {
				parts = append(parts, fmt.Sprintf("debt %d", lane.DebtScore))
			}
		}
		lines = append(lines, "              "+strings.Join(parts, "; "))
	}
	return lines
}

func appendDelta(lines []string, d *Delta) []string {
	if d == nil {
		return lines
	}
	lines = append(lines, "", "  since previous:")
	line := fmt.Sprintf("    - %s: %s", d.Status, d.Summary)
	if d.PaceChanged {
		line += fmt.Sprintf(" | pace: %s -> %s", d.PaceFrom, d.PaceTo)
	}
	lines = append(lines, line)
	lines = appendDeltaItems(lines, "new", d.New)
	lines = appendDeltaItems(lines, "resolved", d.Resolved)
	lines = appendDeltaItems(lines, "still present", d.Persistent)
	return lines
}

func appendDeltaItems(lines []string, label string, items []DeltaItem) []string {
	if len(items) == 0 {
		return lines
	}
	for _, it := range items {
		line := fmt.Sprintf("    - %s: [%s] %s: %s", label, it.Bucket, it.Source, it.Title)
		if it.Detail != "" {
			line += " - " + it.Detail
		}
		if it.Action != "" {
			line += " | " + it.Action
		}
		lines = append(lines, line)
	}
	return lines
}

func appendStrengths(lines []string, strengths []Strength) []string {
	if len(strengths) == 0 {
		return lines
	}
	lines = append(lines, "", "  strengths:")
	for _, st := range strengths {
		line := fmt.Sprintf("    - %s: %s (%s)", st.Source, st.Title, st.Kind)
		if st.Detail != "" {
			line += " - " + st.Detail
		}
		if st.Use != "" {
			line += " | " + st.Use
		}
		lines = append(lines, line)
	}
	return lines
}

func appendChoices(lines []string, choices []Choice) []string {
	if len(choices) == 0 {
		return lines
	}
	lines = append(lines, "", "  choices:")
	for _, ch := range choices {
		line := fmt.Sprintf("    - %s: %s [default: %s]", ch.Source, ch.Question, ch.Default)
		if ch.Why != "" {
			line += " - " + ch.Why
		}
		if ch.Action != "" {
			line += " | " + ch.Action
		}
		lines = append(lines, line)
	}
	return lines
}

func appendChallenges(lines []string, challenges []Challenge) []string {
	if len(challenges) == 0 {
		return lines
	}
	lines = append(lines, "", "  challenges:")
	for _, ch := range challenges {
		line := fmt.Sprintf("    - %s: %s (%s)", ch.Source, ch.Title, ch.Kind)
		if ch.Detail != "" {
			line += " - " + ch.Detail
		}
		if ch.Action != "" {
			line += " | " + ch.Action
		}
		lines = append(lines, line)
	}
	return lines
}

func appendLearningAgenda(lines []string, agenda LearningAgenda) []string {
	if agenda.Focus == "" {
		return lines
	}
	line := fmt.Sprintf("    - focus: %s", agenda.Focus)
	if agenda.Reason != "" {
		line += " - " + agenda.Reason
	}
	if agenda.Practice != "" {
		line += " | practice: " + agenda.Practice
	}
	if agenda.Skip != "" {
		line += " | skip: " + agenda.Skip
	}
	if len(agenda.DrillDown) > 0 {
		line += " | drill: " + strings.Join(agenda.DrillDown, " -> ")
	}
	return append(lines, "", "  learning agenda:", line)
}

func appendLearning(lines []string, lessons []Learning) []string {
	if len(lessons) == 0 {
		return lines
	}
	lines = append(lines, "", "  learning:")
	for _, l := range lessons {
		line := fmt.Sprintf("    - %s: %s", l.Topic, l.Lesson)
		if l.Apply != "" {
			line += " | " + l.Apply
		}
		lines = append(lines, line)
	}
	return lines
}

func appendSection(lines []string, name string, items []Item) []string {
	if len(items) == 0 {
		return lines
	}
	lines = append(lines, "", "  "+name+":")
	for _, it := range items {
		line := fmt.Sprintf("    - %s: %s", it.Source, it.Title)
		if it.Detail != "" {
			line += " - " + it.Detail
		}
		if it.Action != "" {
			line += " | " + it.Action
		}
		lines = append(lines, line)
	}
	return lines
}

func renderAttention(a Attention) string {
	parts := []string{
		dashIfEmpty(a.Level),
		fmt.Sprintf("%d min", a.BudgetMinutes),
		dashIfEmpty(a.Cadence),
	}
	if len(a.ReadOrder) > 0 {
		parts = append(parts, strings.Join(a.ReadOrder, " -> "))
	}
	if a.Summary != "" {
		parts = append(parts, a.Summary)
	}
	return strings.Join(parts, "; ")
}

func renderHumanUse(h HumanUse) string {
	parts := []string{
		"human: " + dashIfEmpty(h.UseHumanFor),
		"agents: " + dashIfEmpty(h.LetAgentsDo),
		"avoid: " + dashIfEmpty(h.Avoid),
		"escalate: " + dashIfEmpty(h.EscalateWhen),
	}
	return strings.Join(parts, "; ")
}

func renderCoherence(c Coherence) string {
	parts := []string{dashIfEmpty(c.Status), dashIfEmpty(c.Summary)}
	if c.Action != "" {
		parts = append(parts, c.Action)
	}
	return strings.Join(parts, "; ")
}

func capItems(items []Item, n int) []Item {
	if len(items) <= n {
		return items
	}
	return items[:n]
}

func renderSources(srcs []SourceState) string {
	parts := make([]string, 0, len(srcs))
	for _, s := range srcs {
		parts = append(parts, s.Name+"="+s.Status)
	}
	return strings.Join(parts, ", ")
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
	parts := []string{"version " + dashIfEmpty(r.Version), "next " + dashIfEmpty(r.ActionKind)}
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
	parts := []string{"frontier " + dashIfEmpty(s.Frontier), s.Direction}
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

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func containsString(vals []string, want string) bool {
	for _, v := range vals {
		if v == want {
			return true
		}
	}
	return false
}

func dashIfEmpty(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}
