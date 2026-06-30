package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

func commitReviewOptions(model, objective, endpoint, apiKeyEnv string) *safecommit.ReviewOptions {
	model = strings.TrimSpace(model)
	if model == "" {
		return nil
	}
	apiKey := ""
	if apiKeyEnv = strings.TrimSpace(apiKeyEnv); apiKeyEnv != "" {
		apiKey = strings.TrimSpace(os.Getenv(apiKeyEnv))
	}
	client := agent.NewHTTPPlanner(endpoint, model, apiKey)
	client.MaxTokens = 256
	client.Temperature = 0
	temp := 0.0
	classifier := modelroute.ClassifierFunc(func(ctx context.Context, s modelroute.Subject) (modelroute.ScoutLabel, error) {
		comp, err := client.Complete(ctx, []agent.Message{
			{Role: agent.RoleSystem, Content: commitReviewSystemPrompt},
			{Role: agent.RoleUser, Content: commitReviewPrompt(s.Labels["objective"], s.Labels["diff"])},
		}, nil, agent.WithMaxTokens(256), agent.WithTemperature(&temp))
		if err != nil {
			return modelroute.ScoutLabel{}, err
		}
		if comp == nil {
			return modelroute.ScoutLabel{}, fmt.Errorf("review model returned nil completion")
		}
		return parseCommitReviewScoutLabel(comp.Message.Content)
	})
	return &safecommit.ReviewOptions{
		Model:     model,
		Objective: objective,
		Reviewer: func(ctx context.Context, req modelroute.ReviewRequest) (modelroute.ReviewResult, error) {
			return modelroute.ReviewDiffWithScout(ctx, classifier, req)
		},
	}
}

const commitReviewSystemPrompt = "You are a cheap scout code reviewer. Decide whether the diff should pass or be refuted before commit. Return only JSON: {\"verdict\":\"pass|refute\",\"reason\":\"short reason\"}."

func commitReviewPrompt(objective, diff string) string {
	return "Objective:\n" + strings.TrimSpace(objective) + "\n\nDiff:\n```diff\n" + diff + "\n```\n\nReturn only JSON with verdict pass or refute and a short reason."
}

func parseCommitReviewScoutLabel(text string) (modelroute.ScoutLabel, error) {
	var raw struct {
		Verdict string `json:"verdict"`
		Reason  string `json:"reason"`
	}
	body := []byte(stripJSONFence(text))
	if err := json.Unmarshal(body, &raw); err != nil {
		return modelroute.ScoutLabel{}, err
	}
	return modelroute.ScoutLabel{Labels: map[string]string{
		"verdict": raw.Verdict,
		"reason":  raw.Reason,
	}}, nil
}

func stripJSONFence(text string) string {
	text = strings.TrimSpace(text)
	if !strings.HasPrefix(text, "```") {
		return text
	}
	if i := strings.IndexByte(text, '\n'); i >= 0 {
		text = text[i+1:]
	}
	text = strings.TrimSpace(text)
	text = strings.TrimSuffix(text, "```")
	return strings.TrimSpace(text)
}

func recordCommitReviewForLoop(res safecommit.Result) error {
	if res.Review == nil {
		return nil
	}
	loopID := firstNonEmpty(os.Getenv("FAK_GOAL_LOOP"), os.Getenv("FAK_LOOP_ID"))
	if strings.TrimSpace(loopID) == "" {
		return nil
	}
	ledger := firstNonEmpty(os.Getenv("FAK_LOOP_LEDGER"), defaultLoopLedger())
	if strings.TrimSpace(ledger) == "" {
		return nil
	}

	review := res.Review
	reason := commitReviewReason(review.Verdict)
	summary := commitReviewSummary(*review)
	metrics := map[string]int64{}
	if review.ScoutCalls > 0 {
		metrics["scout_calls"] = int64(review.ScoutCalls)
	}
	_, err := loopmgr.Append(ledger, loopmgr.Event{
		LoopID:  loopID,
		RunID:   firstNonEmpty(os.Getenv("FAK_GOAL_RUN"), os.Getenv("FAK_LOOP_RUN_ID"), commitReviewRunID()),
		Kind:    loopmgr.EventHeartbeat,
		Source:  "fak commit",
		Reason:  reason,
		Summary: summary,
		EvidenceRefs: []loopmgr.EvidenceRef{{
			Kind:    "review",
			Ref:     string(review.Verdict),
			Summary: summary,
			SHA256:  review.DiffSHA256,
		}},
		Metrics: metrics,
	})
	return err
}

func appendCommitReviewRefusalToGoal(res safecommit.Result) error {
	if res.Review == nil || res.Review.Verdict != modelroute.ReviewRefute {
		return nil
	}
	goalPath := strings.TrimSpace(os.Getenv("FAK_GOAL_SPEC"))
	if goalPath == "" {
		return nil
	}
	return appendGoalScratch(goalPath, "NOT_YET review refuted: "+commitReviewSummary(*res.Review))
}

func commitReviewReason(v modelroute.ReviewVerdict) string {
	switch v {
	case modelroute.ReviewPass:
		return "REVIEW_PASS"
	case modelroute.ReviewRefute:
		return "REVIEW_REFUTED"
	case modelroute.ReviewUnavailable:
		return "REVIEW_UNAVAILABLE"
	default:
		return "REVIEW_UNKNOWN"
	}
}

func commitReviewSummary(r modelroute.ReviewResult) string {
	parts := []string{string(r.Verdict)}
	if strings.TrimSpace(r.Model) != "" {
		parts = append(parts, "by "+strings.TrimSpace(r.Model))
	}
	if strings.TrimSpace(r.Reason) != "" {
		parts = append(parts, strings.TrimSpace(r.Reason))
	}
	return strings.Join(parts, ": ")
}

func commitReviewRunID() string {
	iter := strings.TrimSpace(os.Getenv("FAK_GOAL_ITER"))
	if iter == "" {
		return ""
	}
	loopID := strings.TrimSpace(os.Getenv("FAK_GOAL_LOOP"))
	if loopID == "" {
		return "turn-" + iter
	}
	return loopID + "-turn-" + iter
}

// appendGoalScratch appends a refusal/scratch line to the session goal file, opening a
// "# Scratch / last-refusal" section the first time. Shared by the commit gate and the
// loop driver (both record a NOT_YET reason against the same goal file).
func appendGoalScratch(path, line string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	text := string(b)
	if !strings.HasSuffix(text, "\n") {
		text += "\n"
	}
	if !goalHasScratch(text) {
		text += "\n# Scratch / last-refusal\n"
	}
	text += "- " + strings.TrimSpace(line) + "\n"
	return os.WriteFile(path, []byte(text), 0o644)
}

func goalHasScratch(text string) bool {
	for _, line := range strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n") {
		line = strings.ToLower(strings.TrimSpace(line))
		if strings.HasPrefix(line, "#") && strings.HasPrefix(strings.TrimSpace(strings.TrimLeft(line, "#")), "scratch") {
			return true
		}
	}
	return false
}
