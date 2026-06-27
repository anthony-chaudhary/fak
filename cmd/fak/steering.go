package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
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
	if len(argv) == 0 {
		fmt.Fprintln(os.Stderr, "fak steering: missing subcommand (status | alert | report | pin)")
		os.Exit(2)
	}
	switch argv[0] {
	case "status":
		os.Exit(runSteering(os.Stdout, os.Stderr, "status", argv[1:]))
	case "report":
		os.Exit(runSteering(os.Stdout, os.Stderr, "report", argv[1:]))
	case "alert":
		os.Exit(runSteering(os.Stdout, os.Stderr, "alert", argv[1:]))
	case "pin":
		os.Exit(runSteeringPin(os.Stdout, os.Stderr, argv[1:]))
	default:
		fmt.Fprintf(os.Stderr, "fak steering: unknown subcommand %q (want: status | alert | report | pin)\n", argv[0])
		os.Exit(2)
	}
}

// steeringSnapshot is the slice of the steerability payload the alert gate and the
// card need: the headline corpus integers plus the soft per-KPI breakdown for the
// actionable "do this next" pointer.
type steeringSnapshot struct {
	payload    scorecard.Payload
	index      float64
	debt       int
	softSignal int
	// drift is the per-KPI soft-signal detail, worst-first (most soft signals first),
	// used to point the heaviest drift at the skill that retires it.
	drift []steeringDrift
}

type steeringDrift struct {
	KPI    string
	Group  string
	Soft   int
	Detail string
}

func runSteering(stdout, stderr io.Writer, mode string, argv []string) int {
	fs := flag.NewFlagSet("fak steering "+mode, flag.ContinueOnError)
	fs.SetOutput(stderr)
	channel := fs.String("channel", "", "target channel id (default: "+steeringChannelDefault+" #steering-guard, or $FAK_STEERING_CHANNEL)")
	token := fs.String("token", "", "override bot token (default: $FAK_SCOREBOARD_TOKEN / .env.slack.local)")
	source := fs.String("source", "", "who is posting: ci | agent | <hostname> (default: $FAK_SCOREBOARD_SOURCE or hostname)")
	scorecardJSON := fs.String("scorecard-json", "", "read the steerability payload from this file instead of running the scorecard (- for stdin)")
	indexDelta := fs.Float64("index-delta", 2.0, "alert: minimum index drop vs the pinned floor to fire")
	pin := fs.Bool("pin", false, "alert: ratchet the floor down when the read is an improvement")
	dryRun := fs.Bool("dry-run", false, "render the card and print it; do not post to Slack")
	if err := fs.Parse(argv); err != nil {
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
		up := buildSteeringUpdate(snap, "alert", src, reason)
		return postSteering(stdout, stderr, up, *channel, *token, *dryRun)
	}

	up := buildSteeringUpdate(snap, mode, src, "")
	return postSteering(stdout, stderr, up, *channel, *token, *dryRun)
}

func runSteeringPin(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak steering pin", flag.ContinueOnError)
	fs.SetOutput(stderr)
	scorecardJSON := fs.String("scorecard-json", "", "read the steerability payload from this file instead of running the scorecard (- for stdin)")
	if err := fs.Parse(argv); err != nil {
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
func runSteerabilityScorecard() ([]byte, error) {
	interps := []string{}
	if p := strings.TrimSpace(os.Getenv("FAK_PYTHON")); p != "" {
		interps = append(interps, p)
	}
	interps = append(interps, "python3", "python")
	var lastErr error
	for _, py := range interps {
		cmd := exec.Command(py, "tools/steerability_scorecard.py", "--json")
		cmd.Stderr = os.Stderr
		out, err := cmd.Output()
		if err == nil {
			return out, nil
		}
		lastErr = err
	}
	return nil, fmt.Errorf("run steerability scorecard (tried %s): %w", strings.Join(interps, ", "), lastErr)
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
		snap.softSignal = int(corpusFloat(p.Corpus, "soft_signals"))
		snap.drift = corpusDrift(p.Corpus)
	}
	return snap, nil
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
		if soft <= 0 {
			continue
		}
		out = append(out, steeringDrift{
			KPI:    toString(m["kpi"]),
			Group:  toString(m["group"]),
			Soft:   soft,
			Detail: toString(m["detail"]),
		})
	}
	sort.SliceStable(out, func(i, j int) bool {
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
// card: status = headline only; report = headline + per-group index + breakdown;
// alert = headline + reason + actionable "do this next" buttons for the worst drift.
func buildSteeringUpdate(snap steeringSnapshot, mode, source, reason string) scoreboard.Update {
	title := "steerability"
	switch mode {
	case "report":
		title = "steerability report"
	case "alert":
		title = "steerability alert"
	}
	up := scoreboard.FromPayload(title, snap.payload, "steerability_debt")
	up.Source = source

	if mode == "alert" && reason != "" {
		up.Verdict = "ACTION"
		if up.Detail != "" {
			up.Detail = reason + " — " + up.Detail
		} else {
			up.Detail = reason
		}
		up.Actions = steeringActions(snap)
	}

	if mode == "report" {
		// Replace the bare KPI score lines with a richer snapshot: per-group index
		// first, then the worst drift details. FromPayload already sorted KPI lines.
		var lines []string
		if g := groupLine(snap.payload); g != "" {
			lines = append(lines, g)
		}
		for _, d := range snap.drift {
			lines = append(lines, fmt.Sprintf("%s (%s): %s", d.KPI, d.Group, d.Detail))
		}
		if len(lines) > 0 {
			up.Lines = lines
		}
		up.Actions = steeringActions(snap)
	}
	return up
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
