package modelroute

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
)

const (
	ReviewTool = "loop_review"

	ReviewPass        ReviewVerdict = "pass"
	ReviewRefute      ReviewVerdict = "refute"
	ReviewUnavailable ReviewVerdict = "unavailable"
)

type ReviewVerdict string

type ReviewRequest struct {
	Model     string `json:"model,omitempty"`
	Objective string `json:"objective,omitempty"`
	Diff      string `json:"diff,omitempty"`
}

type ReviewResult struct {
	Model      string        `json:"model,omitempty"`
	Verdict    ReviewVerdict `json:"verdict"`
	Reason     string        `json:"reason,omitempty"`
	DiffSHA256 string        `json:"diff_sha256,omitempty"`
	ScoutCalls int           `json:"scout_calls,omitempty"`
}

func ReviewDiffWithScout(ctx context.Context, scout Classifier, req ReviewRequest) (ReviewResult, error) {
	if scout == nil {
		return ReviewResult{}, fmt.Errorf("modelroute: ReviewDiffWithScout needs a bound Classifier")
	}
	label, err := scout.Classify(ctx, ReviewSubject(req))
	if err != nil {
		return ReviewResult{}, fmt.Errorf("modelroute: review scout classify: %w", err)
	}
	res, err := ReviewResultFromScoutLabel(req, label)
	if err != nil {
		return ReviewResult{}, err
	}
	res.ScoutCalls = 1
	return res, nil
}

func ReviewSubject(req ReviewRequest) Subject {
	labels := map[string]string{
		"kind":        "pre_commit_diff_review",
		"objective":   req.Objective,
		"diff":        req.Diff,
		"diff_sha256": DiffSHA256(req.Diff),
	}
	if strings.TrimSpace(req.Model) != "" {
		labels["review_model"] = strings.TrimSpace(req.Model)
	}
	return Subject{
		Aspect:       AspectScout,
		Tool:         ReviewTool,
		PromptTokens: estimateReviewTokens(req.Objective, req.Diff),
		Labels:       labels,
	}
}

func ReviewResultFromScoutLabel(req ReviewRequest, label ScoutLabel) (ReviewResult, error) {
	if !label.Valid() {
		return ReviewResult{}, fmt.Errorf("modelroute: review scout returned out-of-vocabulary complexity %q", label.Complexity)
	}
	verdict := ReviewVerdict(strings.ToLower(strings.TrimSpace(firstLabel(label.Labels, "verdict", "review"))))
	switch verdict {
	case ReviewPass, ReviewRefute:
	case "ok", "allow", "approved":
		verdict = ReviewPass
	case "fail", "reject", "rejected", "block", "blocked":
		verdict = ReviewRefute
	default:
		return ReviewResult{}, fmt.Errorf("modelroute: review scout returned verdict %q, want pass or refute", verdict)
	}
	reason := strings.TrimSpace(firstLabel(label.Labels, "reason", "critique", "summary"))
	if reason == "" {
		reason = string(verdict)
	}
	return ReviewResult{
		Model:      strings.TrimSpace(req.Model),
		Verdict:    verdict,
		Reason:     reason,
		DiffSHA256: DiffSHA256(req.Diff),
	}, nil
}

func DiffSHA256(diff string) string {
	sum := sha256.Sum256([]byte(diff))
	return hex.EncodeToString(sum[:])
}

func firstLabel(labels map[string]string, keys ...string) string {
	for _, k := range keys {
		if v := strings.TrimSpace(labels[k]); v != "" {
			return v
		}
	}
	return ""
}

func estimateReviewTokens(parts ...string) int {
	n := 0
	for _, p := range parts {
		n += len(p)
	}
	if n == 0 {
		return 0
	}
	return n/4 + 1
}
