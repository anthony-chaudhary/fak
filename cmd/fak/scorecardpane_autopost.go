package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

const (
	scoreboardAutoPostEnv         = "FAK_SCOREBOARD_AUTOPOST"
	scoreboardAutoPostStateSchema = "fak-scoreboard-autopost-state/1"
	scoreboardAutoPostStateRel    = ".fak/scoreboard-autopost-state.json"
)

type scoreboardAutoPostState struct {
	Schema  string            `json:"schema"`
	Updates map[string]string `json:"updates"`
}

func scorecardAutoPostEnabled(explicit bool) bool {
	if explicit {
		return true
	}
	switch strings.ToLower(strings.TrimSpace(os.Getenv(scoreboardAutoPostEnv))) {
	case "", "0", "false", "no", "off":
		return false
	default:
		return true
	}
}

func scoreboardUpdateFromResult(res scorecardpane.Result) (scoreboard.Update, error) {
	var payload scorecard.Payload
	if err := json.Unmarshal(res.Raw, &payload); err != nil {
		return scoreboard.Update{}, fmt.Errorf("parse %s scorecard payload: %w", res.Card.Key, err)
	}
	title := strings.TrimSpace(res.Card.Label)
	if title == "" {
		title = strings.TrimSpace(res.Card.Key)
	}
	if title == "" {
		title = payload.Schema
	}
	return scoreboardPayloadUpdateFrom(payload, res.Card.Debt, title, defaultSource(), "scorecard"), nil
}

func scoreboardUpdateDigest(up scoreboard.Update) string {
	payload := struct {
		Title   string `json:"title"`
		Grade   string `json:"grade"`
		Score   string `json:"score"`
		Debt    string `json:"debt"`
		Verdict string `json:"verdict"`
	}{
		Title:   up.Title,
		Grade:   up.Grade,
		Score:   up.Score,
		Debt:    up.Debt,
		Verdict: up.Verdict,
	}
	raw, _ := json.Marshal(payload)
	sum := sha256.Sum256(raw)
	return fmt.Sprintf("%x", sum[:])
}

// autopostControlPane posts the freshly-regenerated portfolio payload to #scoreboard
// when the operator opted in (--post, or FAK_SCOREBOARD_AUTOPOST=1); a no-op otherwise.
// This is the LOCAL producer side of the two-workspace status feed (#998, follow-on to
// the 52ed934b publisher): a score an agent moves on a dev box shows up in the channel
// the moment the scorecard is regenerated, with no second manual `fak scoreboard post`.
// It reuses the persistent dedup state so a routine local regen that moved no debt is
// silent, and routes notices (and the posted ts) to stderr so a --json stdout stays clean.
func autopostControlPane(stderr io.Writer, root string, p scorecardpane.Payload, explicit bool) int {
	if !scorecardAutoPostEnabled(explicit) {
		return 0
	}
	up := scoreboardControlPaneUpdate(p, defaultSource())
	return postScoreboardUpdates(stderr, stderr, root, []scoreboard.Update{up})
}

// scoreboardControlPaneUpdate folds the portfolio control-pane payload into one
// scoreboard Update. The headline is the raw-unit ratchet number (total_debt);
// grade_debt rides in Score as the scale-invariant severity companion so the dedupe
// digest also fires on a grade regression that holds total_debt flat. The human
// summary (Reason, else the Finding token) becomes the card detail.
func scoreboardControlPaneUpdate(p scorecardpane.Payload, source string) scoreboard.Update {
	detail := p.Reason
	if detail == "" {
		detail = p.Finding
	}
	return scoreboard.Update{
		Title:   "scorecard portfolio",
		DebtKey: "total_debt",
		Debt:    fmt.Sprintf("%d", p.TotalDebt),
		Score:   fmt.Sprintf("%d", p.GradeDebt),
		Verdict: p.Verdict,
		Detail:  detail,
		Source:  source,
	}
}

// postScorecardResults posts each collected per-card scorecard Result behind the
// persistent dedup state. It parses every payload first (a bad payload fails the
// whole batch before any post) then delegates to the shared post loop.
func postScorecardResults(stdout, stderr io.Writer, root string, results []scorecardpane.Result) int {
	ups := make([]scoreboard.Update, 0, len(results))
	for _, res := range results {
		up, err := scoreboardUpdateFromResult(res)
		if err != nil {
			fmt.Fprintf(stderr, "scorecard autopost: %v\n", err)
			return 2
		}
		ups = append(ups, up)
	}
	return postScoreboardUpdates(stdout, stderr, root, ups)
}

// postScoreboardUpdates is the shared change-gated post loop: each update whose score
// fields differ from the last posted digest is published (via the durable post flow),
// and its digest is banked in the persistent state file so a rerun that moved nothing
// is silent. Any post-flow failure short-circuits with its exit code.
func postScoreboardUpdates(stdout, stderr io.Writer, root string, ups []scoreboard.Update) int {
	statePath := filepath.Join(root, filepath.FromSlash(scoreboardAutoPostStateRel))
	state := loadScoreboardAutoPostState(statePath)
	if state.Updates == nil {
		state.Updates = map[string]string{}
	}
	changed := false
	for _, up := range ups {
		digest := scoreboardUpdateDigest(up)
		key := up.ChangeKey()
		if state.Updates[key] == digest {
			fmt.Fprintf(stderr, "skipped %s: no change\n", up.Title)
			continue
		}
		code := scoreboardPostFlow(stdout, stderr, up, scoreboardPostOpts{
			prefix:        "fak scorecard autopost",
			resolveChan:   scoreboard.ResolveChannel,
			resolveToken:  scoreboard.ResolveToken,
			noChannelHint: "fak scorecard autopost: no channel: pass --channel, set FAK_SCOREBOARD_CHANNEL, or add it to .env.slack.local",
			dedupe:        true,
		})
		if code != 0 {
			return code
		}
		state.Updates[key] = digest
		changed = true
	}
	if changed {
		if err := saveScoreboardAutoPostState(statePath, state); err != nil {
			fmt.Fprintf(stderr, "scorecard autopost: save state: %v\n", err)
			return 1
		}
	}
	return 0
}

func loadScoreboardAutoPostState(path string) scoreboardAutoPostState {
	raw, err := os.ReadFile(path)
	if err != nil {
		return scoreboardAutoPostState{Schema: scoreboardAutoPostStateSchema, Updates: map[string]string{}}
	}
	var state scoreboardAutoPostState
	if err := json.Unmarshal(raw, &state); err != nil {
		return scoreboardAutoPostState{Schema: scoreboardAutoPostStateSchema, Updates: map[string]string{}}
	}
	if state.Schema == "" {
		state.Schema = scoreboardAutoPostStateSchema
	}
	if state.Updates == nil {
		state.Updates = map[string]string{}
	}
	return state
}

func saveScoreboardAutoPostState(path string, state scoreboardAutoPostState) error {
	if state.Schema == "" {
		state.Schema = scoreboardAutoPostStateSchema
	}
	if state.Updates == nil {
		state.Updates = map[string]string{}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	raw = append(raw, '\n')
	return os.WriteFile(path, raw, 0o600)
}
