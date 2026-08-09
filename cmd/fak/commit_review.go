package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
)

func commitReviewOptions(model, objective, endpoint, apiKeyEnv string, minModels int) *safecommit.ReviewOptions {
	return reviewOptionsForPrompt(model, objective, endpoint, apiKeyEnv, minModels, commitReviewSystemPrompt, commitReviewPrompt)
}

type reviewPromptBuilder func(objective, diff string) string

func reviewOptionsForPrompt(model, objective, endpoint, apiKeyEnv string, minModels int, systemPrompt string, prompt reviewPromptBuilder) *safecommit.ReviewOptions {
	models := commitReviewModels(model)
	if len(models) == 0 {
		return nil
	}
	apiKey := ""
	if apiKeyEnv = strings.TrimSpace(apiKeyEnv); apiKeyEnv != "" {
		apiKey = strings.TrimSpace(os.Getenv(apiKeyEnv))
	}
	temp := 0.0
	type boundReviewer struct {
		model      string
		classifier modelroute.Classifier
	}
	reviewers := make([]boundReviewer, 0, len(models))
	for _, m := range models {
		m := m
		client := agent.NewHTTPPlanner(endpoint, m, apiKey)
		client.MaxTokens = 256
		client.Temperature = 0
		reviewers = append(reviewers, boundReviewer{
			model: m,
			classifier: modelroute.ClassifierFunc(func(ctx context.Context, s modelroute.Subject) (modelroute.ScoutLabel, error) {
				comp, err := client.Complete(ctx, []agent.Message{
					{Role: agent.RoleSystem, Content: systemPrompt},
					{Role: agent.RoleUser, Content: prompt(s.Labels["objective"], s.Labels["diff"])},
				}, nil, agent.WithMaxTokens(256), agent.WithTemperature(&temp))
				if err != nil {
					return modelroute.ScoutLabel{}, err
				}
				if comp == nil {
					return modelroute.ScoutLabel{}, fmt.Errorf("review model returned nil completion")
				}
				return parseCommitReviewScoutLabel(comp.Message.Content)
			}),
		})
	}
	minModels = defaultReviewMinModels(len(reviewers), minModels)
	return &safecommit.ReviewOptions{
		Model:     strings.Join(models, ","),
		Objective: objective,
		Reviewer: func(ctx context.Context, req modelroute.ReviewRequest) (modelroute.ReviewResult, error) {
			if len(reviewers) == 1 {
				req.Model = reviewers[0].model
				return modelroute.ReviewDiffWithScout(ctx, reviewers[0].classifier, req)
			}
			members := make([]modelroute.ReviewMember, 0, len(reviewers))
			for _, r := range reviewers {
				memberReq := req
				memberReq.Model = r.model
				res, err := modelroute.ReviewDiffWithScout(ctx, r.classifier, memberReq)
				if err != nil {
					members = append(members, modelroute.ReviewMember{
						Model:   r.model,
						Verdict: modelroute.ReviewUnavailable,
						Error:   err.Error(),
					})
					continue
				}
				members = append(members, modelroute.ReviewMember{
					Model:   r.model,
					Verdict: res.Verdict,
					Reason:  res.Reason,
				})
			}
			req.Model = strings.Join(models, ",")
			return modelroute.FoldReviewQuorum(req, members, minModels), nil
		},
	}
}

const commitReviewSystemPrompt = "You are a cheap scout code reviewer. Decide whether the diff should pass or be refuted before commit. Return only JSON: {\"verdict\":\"pass|refute\",\"reason\":\"short reason\"}."

func commitReviewPrompt(objective, diff string) string {
	return "Objective:\n" + strings.TrimSpace(objective) + "\n\nDiff:\n```diff\n" + diff + "\n```\n\nReturn only JSON with verdict pass or refute and a short reason."
}

func commitReviewModels(raw string) []string {
	seen := map[string]bool{}
	var out []string
	for _, part := range strings.FieldsFunc(raw, func(r rune) bool { return r == ',' || r == ';' }) {
		part = strings.TrimSpace(part)
		if part == "" || seen[part] {
			continue
		}
		seen[part] = true
		out = append(out, part)
	}
	return out
}

func defaultReviewMinModels(modelCount, requested int) int {
	if requested > 0 {
		return requested
	}
	if modelCount > 1 {
		if modelCount < 2 {
			return modelCount
		}
		return 2
	}
	return 1
}

func envIntOrDefault(name string, def int) int {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
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
	// Bound the scratch section: without a cap it grows one line per non-terminal turn
	// forever, so every whole-file read+rewrite of GOAL.md in the loop driver becomes
	// O(turns) and a long drive degrades O(N^2) in I/O with unbounded disk (#3453). Only
	// the last scratch line is ever consumed (lastLoopGoalScratchLine / FAK_GOAL_LAST_REFUSAL),
	// so trimming to the most recent entries is loss-free for every reader.
	text = capGoalScratch(text, goalScratchCap)
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		return err
	}
	// The bounded section above is the functional + at-a-glance surface; the FULL refusal
	// history goes to an append-only sidecar the hot loop never re-parses (#3826). Writing
	// it here — the one choke point every scratch call site funnels through (the commit
	// gate's refuted review plus all four loop-driver paths) — is what makes retention
	// unbounded without reintroducing #3453 on either the write or the read side.
	appendGoalScratchLog(path, line, goalScratchLogRotateBytes)
	return nil
}

// goalScratchLogSuffix names the append-only plaintext refusal history that rides beside a
// goal file: GOAL.md -> GOAL.scratch.log. It is deliberately a ".log" so growthgate
// classifies it as ClassLog and its "rotate by size" remedy applies (see the rotation below);
// a name outside that filter would make it an unclassified grower, which is the very defect
// epic #3287 / #3455 track.
const goalScratchLogSuffix = ".scratch.log"

// goalScratchLogRotateBytes caps the ACTIVE sidecar segment. Retention stays unbounded —
// rotation seals the full segment to "<path>.NNN" rather than dropping entries — but the hot
// append target never becomes an unbounded single file. Matched to loopmgr.DefaultRotateBytes
// so the sibling ledger and this sidecar bound their hot files the same way.
const goalScratchLogRotateBytes int64 = 8 << 20

// goalScratchLogStampLayout is RFC3339 pinned to UTC. Fixed-width by construction (20 bytes),
// which is what lets the O(1)-per-turn witness compare byte deltas across turns.
const goalScratchLogStampLayout = "2006-01-02T15:04:05Z"

// goalScratchLogPath maps a goal file to its history sidecar, replacing the extension so
// "GOAL.md" becomes "GOAL.scratch.log". An empty goal path yields an empty sidecar path,
// which callers treat as "no history surface configured".
func goalScratchLogPath(goalPath string) string {
	goalPath = strings.TrimSpace(goalPath)
	if goalPath == "" {
		return ""
	}
	return strings.TrimSuffix(goalPath, filepath.Ext(goalPath)) + goalScratchLogSuffix
}

// appendGoalScratchLog appends one timestamped entry to the goal's history sidecar with a
// single O_APPEND write — no read, no whole-file rewrite — so the per-turn cost is O(1) in
// the number of entries already retained, and the file the drive loop re-parses every turn
// (GOAL.md) is untouched by it.
//
// It is deliberately BEST-EFFORT: this is an observability surface, and the functional write
// (the bounded GOAL.md section that FAK_GOAL_LAST_REFUSAL reads) has already succeeded by the
// time it runs. Failing a drive turn because a supplementary log could not be opened would
// trade a real capability for a nice-to-have, so the error is dropped. The trade-off is that
// a sidecar that cannot be created is silent rather than loud.
func appendGoalScratchLog(goalPath, line string, rotateBytes int64) {
	path := goalScratchLogPath(goalPath)
	if path == "" {
		return
	}
	rotateGoalScratchLog(path, rotateBytes)
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	_, _ = f.WriteString(time.Now().UTC().Format(goalScratchLogStampLayout) + " " + strings.TrimSpace(line) + "\n")
}

// rotateGoalScratchLog seals the active sidecar into the next free "<stem>.NNN.log" segment
// once it reaches rotateBytes, so appends continue into a fresh file. Nothing is dropped or
// rewritten — sealed segments stay on disk and the concatenation of "<stem>.001.log"..
// "<stem>.log", in index order, is the whole history. This is append-only-UNTIL-ROTATED,
// which is what the growthgate ClassLog remedy ("rotate by size") asks of a plaintext grower.
//
// The index is infixed BEFORE the extension rather than appended after it so a sealed segment
// still ends in ".log". That is load-bearing, not cosmetic: growthgate's walk pre-filter
// (isGrowthCandidate) admits a file only by its ".jsonl"/".log"/".err" suffix, so the natural
// "<path>.NNN" naming would drop every sealed segment out of the census entirely — and the
// sealed segments are precisely where the unbounded history accumulates. Keeping the suffix
// keeps them both visible to the census and classified ClassLog (reapable when COLD), which is
// what holds the retention promise of #3826 inside the leak budget epics #3287 / #3455 track.
//
// A non-positive rotateBytes disables rotation (used by callers that bound the file some
// other way). Stat/rename failures leave the active file in place: appending to an oversized
// log is strictly better than losing the turn's entry.
func rotateGoalScratchLog(path string, rotateBytes int64) {
	if rotateBytes <= 0 {
		return
	}
	fi, err := os.Stat(path)
	if err != nil || fi.Size() < rotateBytes {
		return
	}
	ext := filepath.Ext(path)
	stem := strings.TrimSuffix(path, ext)
	// Segment indices are parsed numerically by readers, so the zero-pad width is cosmetic
	// and the scan simply takes the first free slot.
	for i := 1; i < 1000; i++ {
		seg := fmt.Sprintf("%s.%03d%s", stem, i, ext)
		if _, statErr := os.Stat(seg); os.IsNotExist(statErr) {
			_ = os.Rename(path, seg)
			return
		}
	}
}

// goalScratchCap bounds the "# Scratch / last-refusal" section to its most recent entries.
// Sized so a long nightly drive keeps a useful refusal tail without the file growing with
// the turn count; only the final entry is functionally read, the rest is operator context.
const goalScratchCap = 50

// capGoalScratch trims the scratch section of a goal file to its last `cap` entry lines,
// preserving the preamble (goal spec) and the section header verbatim. It is a pure
// function of the file text so it is unit-testable without disk. A non-positive cap, a file
// with no scratch section, or a section already within cap is returned unchanged.
func capGoalScratch(text string, cap int) string {
	if cap <= 0 {
		return text
	}
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	// Header index: last "# … scratch …" heading wins, matching goalHasScratch semantics.
	header := -1
	for i, ln := range lines {
		l := strings.ToLower(strings.TrimSpace(ln))
		if strings.HasPrefix(l, "#") && strings.HasPrefix(strings.TrimSpace(strings.TrimLeft(l, "#")), "scratch") {
			header = i
		}
	}
	if header < 0 {
		return text
	}
	var entries []string
	for _, ln := range lines[header+1:] {
		if strings.TrimSpace(ln) != "" {
			entries = append(entries, ln)
		}
	}
	if len(entries) <= cap {
		return text
	}
	out := append(lines[:header+1:header+1], entries[len(entries)-cap:]...)
	return strings.Join(out, "\n") + "\n"
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
