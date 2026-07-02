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
	Model          string         `json:"model,omitempty"`
	Verdict        ReviewVerdict  `json:"verdict"`
	Reason         string         `json:"reason,omitempty"`
	DiffSHA256     string         `json:"diff_sha256,omitempty"`
	ScoutCalls     int            `json:"scout_calls,omitempty"`
	RequiredModels int            `json:"required_models,omitempty"`
	Members        []ReviewMember `json:"members,omitempty"`
}

type ReviewMember struct {
	Model   string        `json:"model,omitempty"`
	Verdict ReviewVerdict `json:"verdict"`
	Reason  string        `json:"reason,omitempty"`
	Error   string        `json:"error,omitempty"`
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

func FoldReviewQuorum(req ReviewRequest, members []ReviewMember, minUsable int) ReviewResult {
	out := ReviewResult{
		Model:          strings.TrimSpace(req.Model),
		DiffSHA256:     DiffSHA256(req.Diff),
		RequiredModels: minUsable,
		Members:        append([]ReviewMember(nil), members...),
	}
	if len(members) == 0 {
		out.Verdict = ReviewUnavailable
		out.Reason = "no review models configured"
		return out
	}
	if minUsable <= 0 {
		minUsable = 1
		out.RequiredModels = minUsable
	}

	var usable, calls int
	var refutes, unavailable []string
	for _, m := range members {
		calls++
		name := strings.TrimSpace(m.Model)
		if name == "" {
			name = "model"
		}
		switch m.Verdict {
		case ReviewPass:
			usable++
		case ReviewRefute:
			usable++
			refutes = append(refutes, memberSummary(name, m.Reason))
		default:
			unavailable = append(unavailable, memberSummary(name, firstNonEmptyReview(m.Error, m.Reason, "unavailable")))
		}
	}
	out.ScoutCalls = calls
	if len(refutes) > 0 {
		out.Verdict = ReviewRefute
		out.Reason = "review quorum refuted: " + strings.Join(refutes, "; ")
		return out
	}
	if usable < minUsable {
		out.Verdict = ReviewRefute
		out.Reason = fmt.Sprintf("review quorum not met: %d/%d usable verdicts, need %d", usable, len(members), minUsable)
		if len(unavailable) > 0 {
			out.Reason += " (" + strings.Join(unavailable, "; ") + ")"
		}
		return out
	}
	out.Verdict = ReviewPass
	out.Reason = fmt.Sprintf("review quorum passed: %d/%d usable verdicts", usable, len(members))
	return out
}

func memberSummary(model, reason string) string {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		return model
	}
	return model + ": " + reason
}

func firstNonEmptyReview(vals ...string) string {
	for _, v := range vals {
		if v = strings.TrimSpace(v); v != "" {
			return v
		}
	}
	return ""
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
