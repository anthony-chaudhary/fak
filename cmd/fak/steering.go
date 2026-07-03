package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dogfoodscore"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// steeringChannelDefault is #steering-guard in the scoreboard Slack workspace
// (team T0BDEJF1HGB). It is a PUBLIC channel id (not a secret): the @agent bot is a
// member and posts here with FAK_SCOREBOARD_TOKEN. Override with --channel or
// FAK_STEERING_CHANNEL to point the surface elsewhere — NOT FAK_SCOREBOARD_CHANNEL,
// which is the scoreboard CLI's own default (#scoreboard).
const steeringChannelDefault = "C0BD5J4ERL7"

// steeringBaselineRel is the committed alert ratchet floor (separate from the
// unified scorecard_baseline.json, which tracks only the hard `steer` debt integer,
// not the index or soft-signal count the alert gate compares against).
const steeringBaselineRel = "tools/steering_baseline.json"

// cmdSteering drives the steerability surface in Slack #steering-guard: status,
// alert (regression-gated), report (full snapshot), and pin (re-baseline). It runs
// tools/steerability_scorecard.py --json and posts through internal/scoreboard, so
// it never disturbs the lab SLACK_BOT_TOKEN — only FAK_SCOREBOARD_TOKEN.
//
//	fak steering status            # post the current index card (always)
//	fak steering report            # post the full per-KPI snapshot (always)
//	fak steering alert             # post ONLY on a regression vs the pinned floor
//	fak steering alert --pin       # ...and ratchet the floor down on an improvement
//	fak steering pin               # re-baseline the floor from the current scorecard
func cmdSteering(argv []string) {
	mode := func(m string) func(stdout, stderr io.Writer, a []string) int {
		return func(stdout, stderr io.Writer, a []string) int { return runSteering(stdout, stderr, m, a) }
	}
	dispatchSubcommands("steering", "status | alert | report | pin", argv,
		subcommand{"status", mode("status")},
		subcommand{"report", mode("report")},
		subcommand{"alert", mode("alert")},
		subcommand{"pin", runSteeringPin},
	)
}

// steeringSnapshot is the slice of the steerability payload the alert gate and the
// card need: the headline corpus integers plus the soft per-KPI breakdown for the
// actionable "do this next" pointer.
type steeringSnapshot struct {
	payload    scorecard.Payload
	index      float64
	debt       int
	pressure   int
	softSignal int
	// drift is the per-KPI soft-signal detail, worst-first (most soft signals first),
	// used to point the heaviest drift at the skill that retires it.
	drift []steeringDrift
}

type steeringDrift struct {
	KPI      string
	Group    string
	Score    int
	Soft     int
	Pressure int
	Gain     float64
	Detail   string
}

type steeringDogfood struct {
	Verdict         string
	Grade           string
	Score           string
	Debt            int
	RecentWedged    int
	RecentMarkers   int
	StopMarkers     int
	ConflationTurns int
	TranscriptsSeen int
	ChainReports    int
	ChainAgeHours   float64
	ReceiptMode     string
	PendingCreates  int
	NextAction      string
}

func runSteering(stdout, stderr io.Writer, mode string, argv []string) int {
	fs := flag.NewFlagSet("fak steering "+mode, flag.ContinueOnError)
	fs.SetOutput(stderr)
	channel := fs.String("channel", "", "target channel id (default: "+steeringChannelDefault+" #steering-guard, or $FAK_STEERING_CHANNEL)")
	token := fs.String("token", "", "override bot token (default: $FAK_SCOREBOARD_TOKEN / .env.slack.local)")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	scorecardJSON := fs.String("scorecard-json", "", "read the steerability payload from this file instead of running the scorecard (- for stdin)")
	dogfoodJSON := fs.String("dogfood-json", "", "read the dogfood-score payload from this file instead of collecting live actuals (- for stdin)")
	indexDelta := fs.Float64("index-delta", 2.0, "alert: minimum index drop vs the pinned floor to fire")
	pin := fs.Bool("pin", false, "alert: ratchet the floor down when the read is an improvement")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not post to Slack")
	if !parseFlags(fs, argv) {
		return 2
	}

	snap, err := loadSteeringSnapshot(*scorecardJSON)
	if err != nil {
		fmt.Fprintf(stderr, "fak steering %s: %v\n", mode, err)
		return 2
	}

	src := *source
	if src == "" {
		src = defaultSource()
	}

	// The alert path is regression-gated: decide BEFORE building the card whether to
	// post at all. status/report always post.
	if mode == "alert" {
		base, _ := readSteeringBaseline(steeringBaselineRel) // missing floor -> first run fires
		fire, reason := shouldAlert(snap, base, *indexDelta)
		if !fire {
			fmt.Fprintf(stdout, "fak steering alert: no regression vs %s (%s); nothing posted\n", steeringBaselineRel, reason)
			if *pin && isImprovement(snap, base) {
				if err := writeSteeringBaseline(steeringBaselineRel, snap); err != nil {
					fmt.Fprintf(stderr, "fak steering alert: --pin: %v\n", err)
					return 1
				}
				fmt.Fprintf(stdout, "fak steering alert: ratcheted floor in %s (index %.1f, debt %d, signals %d)\n",
					steeringBaselineRel, snap.index, snap.debt, snap.softSignal)
			}
			return 0
		}
		dog, err := loadSteeringDogfood(*dogfoodJSON)
		if err != nil {
			fmt.Fprintf(stderr, "fak steering %s: %v\n", mode, err)
			return 2
		}
		up := buildSteeringUpdate(snap, dog, "alert", src, reason)
		return postSteering(stdout, stderr, up, *channel, *token, *dryRun)
	}

	dog, err := loadSteeringDogfood(*dogfoodJSON)
	if err != nil {
		fmt.Fprintf(stderr, "fak steering %s: %v\n", mode, err)
		return 2
	}
	up := buildSteeringUpdate(snap, dog, mode, src, "")
	return postSteering(stdout, stderr, up, *channel, *token, *dryRun)
}

func runSteeringPin(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak steering pin", flag.ContinueOnError)
	fs.SetOutput(stderr)
	scorecardJSON := fs.String("scorecard-json", "", "read the steerability payload from this file instead of running the scorecard (- for stdin)")
	if !parseFlags(fs, argv) {
		return 2
	}
	snap, err := loadSteeringSnapshot(*scorecardJSON)
	if err != nil {
		fmt.Fprintf(stderr, "fak steering pin: %v\n", err)
		return 2
	}
	if err := writeSteeringBaseline(steeringBaselineRel, snap); err != nil {
		fmt.Fprintf(stderr, "fak steering pin: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "fak steering pin: pinned %s (index %.1f, debt %d, signals %d)\n",
		steeringBaselineRel, snap.index, snap.debt, snap.softSignal)
	return 0
}

// loadSteeringSnapshot produces the snapshot from a --scorecard-json file or by
// running the Python steerability scorecard. The scorecard tool stays the single
// source of truth for the numbers; this command only folds + routes them.
func loadSteeringSnapshot(path string) (steeringSnapshot, error) {
	var raw []byte
	var err error
	if path != "" {
		raw, err = readFromFile(path)
	} else {
		raw, err = runSteerabilityScorecard()
	}
	if err != nil {
		return steeringSnapshot{}, err
	}
	return parseSteeringSnapshot(raw)
}

// runSteerabilityScorecard invokes `python tools/steerability_scorecard.py --json`
// and returns its stdout. It tries the FAK_PYTHON override, then python3, then
// python — matching how the rest of the repo shells to Python across OSes.
//
// A scorecard's EXIT CODE is its verdict, not a run failure: it exits non-zero
// precisely when the verdict is ACTION (there is steerability-debt) while still
// printing the full, valid JSON payload to stdout — and that payload is exactly
// what the steering surface folds. So a non-zero exit with JSON-shaped stdout is a
// SUCCESS here (the same tolerance runScorecardJSON gives the product/persona
// scorecards); only a missing interpreter or empty/non-JSON stdout is a real error.
// Without this, `fak steering status/report/alert` died ("exit status 1") the
// moment any steerability-debt existed — i.e. almost always.
func runSteerabilityScorecard() ([]byte, error) {
	interps := []string{}
	if p := strings.TrimSpace(os.Getenv("FAK_PYTHON")); p != "" {
		interps = append(interps, p)
	}
	interps = append(interps, "python3", "python")
	var lastErr error
	for _, py := range interps {
		cmd := exec.Command(py, "tools/steerability_scorecard.py", "--json")
		windowgate.ConfigureBackgroundCommand(cmd)
		var out, errb bytes.Buffer
		cmd.Stdout, cmd.Stderr = &out, &errb
		runErr := cmd.Run()
		if out.Len() > 0 && bytes.HasPrefix(bytes.TrimSpace(out.Bytes()), []byte("{")) {
			// Valid-looking JSON on stdout: the verdict (exit code) is the payload, not a failure.
			return out.Bytes(), nil
		}
		if runErr != nil {
			lastErr = fmt.Errorf("%s tools/steerability_scorecard.py --json: %w (%s)", py, runErr, strings.TrimSpace(errb.String()))
			continue
		}
		lastErr = fmt.Errorf("%s tools/steerability_scorecard.py --json produced no JSON output", py)
	}
	return nil, lastErr
}

// parseSteeringSnapshot folds the payload's corpus into the alert-gate slice. The
// corpus carries index/steerability_debt/soft_signals and a worst-first breakdown.
func parseSteeringSnapshot(raw []byte) (steeringSnapshot, error) {
	var p scorecard.Payload
	if err := json.Unmarshal(raw, &p); err != nil {
		return steeringSnapshot{}, fmt.Errorf("parse steerability payload: %w", err)
	}
	snap := steeringSnapshot{payload: p}
	if p.Corpus != nil {
		snap.index = corpusFloat(p.Corpus, "index")
		snap.debt = int(corpusFloat(p.Corpus, "steerability_debt"))
		snap.pressure = int(corpusFloat(p.Corpus, "steering_pressure"))
		snap.softSignal = int(corpusFloat(p.Corpus, "soft_signals"))
		snap.drift = corpusDrift(p.Corpus)
	}
	return snap, nil
}

func loadSteeringDogfood(path string) (*steeringDogfood, error) {
	var p dogfoodscore.ScorecardPayload
	if path != "" {
		raw, err := readFromFile(path)
		if err != nil {
			return nil, err
		}
		if err := json.Unmarshal(raw, &p); err != nil {
			return nil, fmt.Errorf("parse dogfood-score payload: %w", err)
		}
	} else {
		p = dogfoodscore.Build(dogfoodscore.Options{Root: repoRoot()})
	}
	return dogfoodFromPayload(p), nil
}

func dogfoodFromPayload(p dogfoodscore.ScorecardPayload) *steeringDogfood {
	d := &steeringDogfood{
		Verdict:         firstNonEmpty(p.Verdict, "UNKNOWN"),
		Grade:           scoreValueString(p.Corpus["grade"]),
		Score:           scoreValueString(p.Corpus["score"]),
		Debt:            int(corpusFloat(p.Corpus, "dogfood_debt")),
		RecentWedged:    int(corpusFloat(p.Corpus, "recent_wedged")),
		StopMarkers:     int(corpusFloat(p.Corpus, "stop_markers")),
		ConflationTurns: int(corpusFloat(p.Corpus, "conflation_turns")),
		TranscriptsSeen: int(corpusFloat(p.Corpus, "transcripts_seen")),
		NextAction:      p.NextAction,
	}
	d.RecentMarkers = p.Evidence.RecentMarkers
	d.ChainReports = p.Evidence.Chain.Reports
	d.ChainAgeHours = p.Evidence.Chain.NewestAgeHours
	d.ReceiptMode = p.Evidence.Chain.ReceiptMode
	d.PendingCreates = p.Evidence.Chain.PendingCreates
	return d
}

// corpusFloat reads a numeric corpus field tolerant of int/float JSON decoding.
func corpusFloat(c map[string]any, key string) float64 {
	switch v := c[key].(type) {
	case float64:
		return v
	case int:
		return float64(v)
	case json.Number:
		f, _ := v.Float64()
		return f
	default:
		return 0
	}
}

// corpusDrift extracts the per-KPI soft-signal rows from corpus.breakdown, keeping
// only KPIs with at least one soft signal, sorted worst-first (most soft, then by
// lowest score) so the heaviest drift drives the action pointer.
func corpusDrift(c map[string]any) []steeringDrift {
	rows, ok := c["breakdown"].([]any)
	if !ok {
		return nil
	}
	var out []steeringDrift
	for _, r := range rows {
		m, ok := r.(map[string]any)
		if !ok {
			continue
		}
		soft := int(toFloat(m["soft"]))
		pressure := int(toFloat(m["pressure"]))
		gain := toFloat(m["index_gain_to_clean"])
		if soft <= 0 && pressure <= 0 && gain <= 0 {
			continue
		}
		out = append(out, steeringDrift{
			KPI:      toString(m["kpi"]),
			Group:    toString(m["group"]),
			Score:    int(toFloat(m["score"])),
			Soft:     soft,
			Pressure: pressure,
			Gain:     gain,
			Detail:   toString(m["detail"]),
		})
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Gain != out[j].Gain {
			return out[i].Gain > out[j].Gain
		}
		if out[i].Pressure != out[j].Pressure {
			return out[i].Pressure > out[j].Pressure
		}
		if out[i].Soft != out[j].Soft {
			return out[i].Soft > out[j].Soft
		}
		return out[i].KPI < out[j].KPI
	})
	return out
}

func toFloat(v any) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case int:
		return float64(t)
	case json.Number:
		f, _ := t.Float64()
		return f
	default:
		return 0
	}
}

func toString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func scoreValueString(v any) string {
	switch x := v.(type) {
	case string:
		return strings.TrimSpace(x)
	case float64:
		if x == float64(int(x)) {
			return fmt.Sprintf("%d", int(x))
		}
		return fmt.Sprintf("%.1f", x)
	case int:
		return fmt.Sprintf("%d", x)
	default:
		return ""
	}
}

// shouldAlert is the pure regression gate: fire when the hard debt is non-zero, the
// index dropped by at least indexDelta vs the floor, or a NEW drift signal appeared
// (soft count rose). A missing/empty floor fires on first run so the channel learns
// the current state. Returns the deciding reason for the alert card.
func shouldAlert(cur steeringSnapshot, base *steeringBaseline, indexDelta float64) (bool, string) {
	if cur.debt > 0 {
		return true, fmt.Sprintf("hard steerability-debt %d > 0", cur.debt)
	}
	if base == nil {
		return true, "no pinned floor yet — establishing the baseline"
	}
	if drop := base.Index - cur.index; drop >= indexDelta {
		return true, fmt.Sprintf("index dropped %.1f → %.1f (≥ %.1f vs floor)", base.Index, cur.index, indexDelta)
	}
	if cur.softSignal > base.SoftSignals {
		return true, fmt.Sprintf("drift signals rose %d → %d (a new soft signal)", base.SoftSignals, cur.softSignal)
	}
	return false, fmt.Sprintf("index %.1f ≥ floor %.1f, debt 0, signals %d ≤ %d", cur.index, base.Index, cur.softSignal, base.SoftSignals)
}

// isImprovement is true when the current read is strictly better than the floor on
// any axis and no worse on the others — the condition for ratcheting the floor down.
func isImprovement(cur steeringSnapshot, base *steeringBaseline) bool {
	if base == nil {
		return true
	}
	noWorse := cur.index >= base.Index && cur.debt <= base.Debt && cur.softSignal <= base.SoftSignals
	better := cur.index > base.Index || cur.debt < base.Debt || cur.softSignal < base.SoftSignals
	return noWorse && better
}

// buildSteeringUpdate folds the snapshot into a scoreboard Update. mode tailors the
// card: status = compact guard state; report = full actuals + top moves; alert =
// regression reason + actionable "do this next" buttons for the worst drift. The
// card deliberately carries BOTH the bounded grade/index and the unbounded pressure
// count, because a green 0-100 letter can hide growing live work.
func buildSteeringUpdate(snap steeringSnapshot, dog *steeringDogfood, mode, source, reason string) scoreboard.Update {
	title := "steering guard"
	switch mode {
	case "report":
		title = "steering guard report"
	case "alert":
		title = "steering guard alert"
	}
	up := scoreboard.FromPayload(title, snap.payload, "steerability_debt")
	up.Source = source
	up.Verdict = steeringOverallVerdict(snap, dog)
	up.Detail = steeringStatusSummary(snap, dog)
	up.Notes = steeringNotes(snap, dog, mode)
	up.Lines = nil

	if mode == "alert" && reason != "" {
		up.Verdict = "ACTION"
		up.Detail = "alert: " + reason + " — " + steeringStatusSummary(snap, dog)
		up.Actions = steeringActions(snap)
	}

	if mode == "report" {
		up.Actions = steeringActions(snap)
	}
	return up
}

func steeringDriftLine(d steeringDrift) string {
	if d.Gain > 0 {
		return fmt.Sprintf("%s (%s): score %d, pressure %d, +%.1f index pts if clean - %s",
			d.KPI, d.Group, d.Score, d.Pressure, d.Gain, d.Detail)
	}
	return fmt.Sprintf("%s (%s): score %d, pressure %d - %s", d.KPI, d.Group, d.Score, d.Pressure, d.Detail)
}

func steeringOverallVerdict(snap steeringSnapshot, dog *steeringDogfood) string {
	if snap.debt > 0 {
		return "ACTION"
	}
	if dog != nil && (dog.Debt > 0 || strings.EqualFold(dog.Verdict, "ACTION")) {
		return "ACTION"
	}
	return firstNonEmpty(snap.payload.Verdict, "OK")
}

func steeringBand(snap steeringSnapshot, dog *steeringDogfood) string {
	if steeringOverallVerdict(snap, dog) == "ACTION" {
		return "RED"
	}
	if snap.pressure > 0 || snap.softSignal > 0 || snap.index < 95 {
		return "YELLOW"
	}
	return "GREEN"
}

func steeringStatusSummary(snap steeringSnapshot, dog *steeringDogfood) string {
	band := steeringBand(snap, dog)
	switch band {
	case "RED":
		if dog != nil && (dog.Debt > 0 || strings.EqualFold(dog.Verdict, "ACTION")) {
			return fmt.Sprintf("RED — dogfood ACTION: %d/%d recent sessions wedged; steerability index %.1f, pressure %d",
				dog.RecentWedged, dog.RecentMarkers, snap.index, snap.pressure)
		}
		return fmt.Sprintf("RED — steerability debt %d; index %.1f, pressure %d", snap.debt, snap.index, snap.pressure)
	case "YELLOW":
		return fmt.Sprintf("YELLOW — steerability is passing but has pressure %d and %d drift signal(s)", snap.pressure, snap.softSignal)
	default:
		return fmt.Sprintf("GREEN — steerability clean: index %.1f, pressure %d", snap.index, snap.pressure)
	}
}

func steeringNotes(snap steeringSnapshot, dog *steeringDogfood, mode string) string {
	lines := []string{
		"*Color code:* " + steeringColorCodeLine(snap, dog),
		fmt.Sprintf("*Steering actuals:* index %.1f/100; pressure %d (unbounded, lower is better); hard debt %d; soft signals %d",
			snap.index, snap.pressure, snap.debt, snap.softSignal),
	}
	if p := pressureByGroupLine(snap.payload); p != "" {
		lines = append(lines, "*Pressure by group:* "+p)
	}
	if g := groupLine(snap.payload); g != "" {
		lines = append(lines, "*Index by group:* "+g)
	}
	if dog != nil {
		lines = append(lines, "*Dogfood actuals:* "+dogfoodActualsLine(dog))
	}
	if mode != "status" && len(snap.drift) > 0 {
		lines = append(lines, "*Top steering moves:*")
		for i, d := range snap.drift {
			if i == 3 {
				break
			}
			lines = append(lines, "• "+steeringDriftLine(d))
		}
	}
	return strings.Join(lines, "\n")
}

func steeringColorCodeLine(snap steeringSnapshot, dog *steeringDogfood) string {
	band := steeringBand(snap, dog)
	reason := "clean"
	switch band {
	case "RED":
		if dog != nil && (dog.Debt > 0 || strings.EqualFold(dog.Verdict, "ACTION")) {
			reason = "dogfood ACTION/debt"
		} else {
			reason = "hard steering debt"
		}
	case "YELLOW":
		reason = "passing, but pressure or drift remains"
	}
	return fmt.Sprintf("%s = %s; RED = action required, YELLOW = passing with pressure, GREEN = clean", band, reason)
}

func dogfoodActualsLine(d *steeringDogfood) string {
	if d == nil {
		return "unavailable"
	}
	parts := []string{
		fmt.Sprintf("%s grade %s score %s, debt %d", firstNonEmpty(d.Verdict, "UNKNOWN"), firstNonEmpty(d.Grade, "?"), firstNonEmpty(d.Score, "?"), d.Debt),
		fmt.Sprintf("%d/%d recent sessions wedged", d.RecentWedged, d.RecentMarkers),
		fmt.Sprintf("%d conflation turn(s) / %d transcript(s)", d.ConflationTurns, d.TranscriptsSeen),
		fmt.Sprintf("%d stop marker(s)", d.StopMarkers),
	}
	if d.ChainReports > 0 {
		parts = append(parts, fmt.Sprintf("packet %.0fh old, receipt %s, pending creates %d",
			d.ChainAgeHours, firstNonEmpty(d.ReceiptMode, "unknown"), d.PendingCreates))
	}
	if d.NextAction != "" && d.Debt > 0 {
		parts = append(parts, "next: "+d.NextAction)
	}
	return strings.Join(parts, "; ")
}

func pressureByGroupLine(p scorecard.Payload) string {
	if p.Corpus == nil {
		return ""
	}
	m, ok := p.Corpus["pressure_by_group"].(map[string]any)
	if !ok {
		return ""
	}
	order := []string{"modularity", "coupling", "navigability", "correction"}
	var parts []string
	for _, g := range order {
		if v, ok := m[g]; ok {
			parts = append(parts, fmt.Sprintf("%s %d", g, int(toFloat(v))))
		}
	}
	return strings.Join(parts, " · ")
}

// groupLine renders the per-group index from corpus.index_by_group, e.g.
// "modularity 81.5 · coupling 99.0 · navigability 68.0 · correction 97.3".
func groupLine(p scorecard.Payload) string {
	if p.Corpus == nil {
		return ""
	}
	m, ok := p.Corpus["index_by_group"].(map[string]any)
	if !ok {
		return ""
	}
	order := []string{"modularity", "coupling", "navigability", "correction"}
	var parts []string
	for _, g := range order {
		if v, ok := m[g]; ok {
			parts = append(parts, fmt.Sprintf("%s %.1f", g, toFloat(v)))
		}
	}
	return strings.Join(parts, " · ")
}

// steeringSkillByKPI maps the heaviest drift KPI to the repeatable pass that retires
// it, mirroring tools/score_signal.py's SKILL_BY_KEY but at per-KPI granularity. The
// generic /steerability-score conductor is the fallback for anything unmapped.
var steeringSkillByKPI = map[string]string{
	"god_file_rate":       "/modularize",
	"god_func_rate":       "/modularize",
	"func_size_dist":      "/modularize",
	"package_doc_frac":    "/curate-cluster",
	"hub_share":           "/steerability-score",
	"churn_concentration": "/steerability-score",
}

const steeringSkillsBase = "https://github.com/anthony-chaudhary/fak/tree/main/.claude/skills"

// steeringActions turns the worst drift signals into "do this next" link-buttons
// pointing at the owning skill, plus a regenerate button. Bounded to the top few so
// the actions block stays readable (the Text() fallback still lists them all).
func steeringActions(snap steeringSnapshot) []scoreboard.Action {
	var actions []scoreboard.Action
	seen := map[string]bool{}
	for _, d := range snap.drift {
		skill := steeringSkillByKPI[d.KPI]
		if skill == "" {
			skill = "/steerability-score"
		}
		if seen[skill] {
			continue
		}
		seen[skill] = true
		name := strings.TrimPrefix(skill, "/")
		actions = append(actions, scoreboard.Action{
			Label: fmt.Sprintf("Run %s (%s)", skill, d.KPI),
			URL:   steeringSkillsBase + "/" + name,
		})
		if len(actions) == 3 {
			break
		}
	}
	// Always offer the re-measure affordance.
	actions = append(actions, scoreboard.Action{
		Label: "Re-measure",
		URL:   steeringSkillsBase + "/steerability-score",
	})
	return actions
}

// postSteering resolves the channel + token and posts (or dry-runs) the card.
func postSteering(stdout, stderr io.Writer, up scoreboard.Update, channel, token string, dryRun bool) int {
	if dryRun {
		fmt.Fprintln(stdout, up.Text())
		return 0
	}
	ch := resolveSteeringChannel(channel)
	client, err := scoreboard.NewClient(token)
	if err != nil {
		fmt.Fprintf(stderr, "fak steering: %v\n", err)
		return 2
	}
	ts, err := client.Post(ctx(), ch, up.Text(), up.Blocks())
	if err != nil {
		fmt.Fprintf(stderr, "fak steering: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "posted to %s ts=%s\n", ch, ts)
	return 0
}

// resolveSteeringChannel applies: --channel, then the steering-specific
// FAK_STEERING_CHANNEL, then the #steering-guard built-in default. It deliberately
// does NOT fall through to the generic FAK_SCOREBOARD_CHANNEL — that env var is the
// scoreboard CLI's default target (#scoreboard), so reusing it here would misroute
// the steering surface to #scoreboard whenever an operator has sourced
// .env.slack.local. Steering owns its own default, so the surface lands in
// #steering-guard with zero config; redirect it only via --channel or
// FAK_STEERING_CHANNEL.
func resolveSteeringChannel(flagVal string) string {
	if flagVal != "" {
		return flagVal
	}
	if v := strings.TrimSpace(os.Getenv("FAK_STEERING_CHANNEL")); v != "" {
		return v
	}
	return steeringChannelDefault
}

// ----- the committed alert ratchet floor -----

type steeringBaseline struct {
	Schema      string  `json:"schema"`
	Commit      string  `json:"commit,omitempty"`
	Index       float64 `json:"index"`
	Debt        int     `json:"steerability_debt"`
	SoftSignals int     `json:"soft_signals"`
	Stamp       string  `json:"stamp,omitempty"`
	Doc         string  `json:"_doc,omitempty"`
}

// readSteeringBaseline loads the pinned floor; a missing file is not an error (the
// alert gate treats a nil floor as "first run, establish the baseline").
func readSteeringBaseline(path string) (*steeringBaseline, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var b steeringBaseline
	if err := json.Unmarshal(raw, &b); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return &b, nil
}

// writeSteeringBaseline pins the current snapshot as the new floor. The stamp is RFC
// 3339 UTC so a re-pin is auditable; commit is left to git (the file is committed).
func writeSteeringBaseline(path string, snap steeringSnapshot) error {
	b := steeringBaseline{
		Schema:      "fak-steering-baseline/1",
		Index:       snap.index,
		Debt:        snap.debt,
		SoftSignals: snap.softSignal,
		Stamp:       time.Now().UTC().Format(time.RFC3339),
		Doc:         "Pinned steerability floor for `fak steering alert`. Re-pin after an improvement: fak steering pin",
	}
	raw, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	return os.WriteFile(path, raw, 0o644)
}
