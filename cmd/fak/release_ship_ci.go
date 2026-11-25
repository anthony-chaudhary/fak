package main

import (
	"encoding/json"
	"strings"
	"time"
)

// An unreadable event cannot establish the absence of failed push runs.
// Accept lowercase event tokens, including future names, but not malformed text.
func releaseShipCIEventsReadable(rows []map[string]any) bool {
	for _, row := range rows {
		event, ok := row["event"].(string)
		if !ok || event == "" {
			return false
		}
		for i, ch := range event {
			if ch >= 'a' && ch <= 'z' {
				continue
			}
			if i > 0 && (ch >= '0' && ch <= '9' || ch == '_') {
				continue
			}
			return false
		}
	}
	return true
}

// releaseShipPushCIRuns keeps diagnostics from supplying release CI evidence,
// even if a response violates the gh query's event filter or omits the field.
func releaseShipPushCIRuns(rows []map[string]any) []map[string]any {
	var push []map[string]any
	for _, row := range rows {
		if event, ok := row["event"].(string); ok && event == "push" {
			push = append(push, row)
		}
	}
	return push
}

func waitReleaseShipCIAppears(result *releaseShipResult, wt string, opts releaseShipOptions, sha string) {
	deadline := time.Now().Add(opts.ciAppearTimeout)
	args := []string{"run", "list", "--workflow", opts.workflow, "--commit", sha, "--event", "push", "--limit", "1", "--json", "databaseId,status,conclusion,url,event"}
	for {
		code, out := releaseShipCmd(result, wt, "gh", args, nil, releaseShipCIPollCommandTimeout)
		if code != 0 {
			result.Warnings = append(result.Warnings, "gh run list failed while waiting for CI to appear: "+tail(out))
			return
		}
		var rows []map[string]any
		if err := json.Unmarshal([]byte(out), &rows); err != nil {
			result.Warnings = append(result.Warnings, "gh run list emitted non-json output while waiting for CI")
			return
		}
		if len(releaseShipPushCIRuns(rows)) > 0 {
			return
		}
		if time.Now().After(deadline) {
			result.Warnings = append(result.Warnings, "no CI run appeared before ci-appear-timeout; release_tag will make the final CI decision")
			return
		}
		time.Sleep(5 * time.Second)
	}
}

func releaseShipSourceCI(result *releaseShipResult, root string, opts releaseShipOptions, sha string) map[string]any {
	if strings.TrimSpace(sha) == "" {
		return map[string]any{"ok": false, "status": "missing_source_sha"}
	}
	args := []string{"run", "list", "--workflow", opts.workflow, "--commit", sha, "--event", "push", "--limit", "1", "--json", "databaseId,status,conclusion,url,headSha,event"}
	code, out := releaseShipCmd(result, root, "gh", args, nil, time.Minute)
	if code != 0 {
		return map[string]any{"ok": false, "status": "unavailable", "tail": tail(out)}
	}
	var rows []map[string]any
	if err := json.Unmarshal([]byte(out), &rows); err != nil {
		return map[string]any{"ok": false, "status": "non_json", "tail": tail(out)}
	}
	if rows == nil {
		return map[string]any{"ok": false, "status": "invalid_response", "source_sha": sha, "reason": "source CI response was not a JSON array"}
	}
	if !releaseShipCIEventsReadable(rows) {
		return map[string]any{"ok": false, "status": "invalid_event", "source_sha": sha, "reason": "CI response contained a missing or malformed event"}
	}
	rows = releaseShipPushCIRuns(rows)
	if len(rows) == 0 {
		missing := map[string]any{"ok": false, "status": "missing", "source_sha": sha, "reason": "no push-triggered CI run found for source"}
		if opts.adoptAncestorCI {
			return releaseShipAdoptAncestorCI(result, root, opts, sha, missing)
		}
		return missing
	}
	row := rows[0]
	conclusion := strings.ToLower(stringFromAny(row["conclusion"]))
	workflowStatus := strings.ToLower(stringFromAny(row["status"]))
	state := releaseShipCIState(workflowStatus, conclusion)
	// A terminal conclusion without completed status is contradictory evidence,
	// not a green result or an inconclusive source that ancestry may replace.
	if conclusion != "" && workflowStatus != "completed" {
		state = "inconsistent"
	}
	if state == "green" {
		return map[string]any{
			"ok":              true,
			"status":          state,
			"workflow_status": workflowStatus,
			"conclusion":      conclusion,
			"source_sha":      sha,
			"run":             row,
		}
	}
	// Ancestor evidence can fill an inconclusive result, never erase an exact failure.
	if opts.adoptAncestorCI && (state == "pending" || state == "missing" || state == "cancelled_or_starved") {
		return releaseShipAdoptAncestorCI(result, root, opts, sha, map[string]any{
			"ok":              false,
			"status":          state,
			"workflow_status": workflowStatus,
			"conclusion":      conclusion,
			"source_sha":      sha,
			"run":             row,
		})
	}
	return map[string]any{
		"ok":              false,
		"status":          state,
		"workflow_status": workflowStatus,
		"conclusion":      conclusion,
		"source_sha":      sha,
		"run":             row,
	}
}

func releaseShipAdoptAncestorCI(result *releaseShipResult, root string, opts releaseShipOptions, sha string, fallback map[string]any) map[string]any {
	// A shallow boundary can hide a failed ancestor while leaving an older
	// green reachable through another merge parent. Require a complete graph
	// before treating merge-base exit 1 as proof of unrelated history.
	code, shallow := releaseShipCmd(result, root, "git", []string{"rev-parse", "--is-shallow-repository"}, nil, releaseShipMergeBaseCommandTimeout)
	if code != 0 || strings.TrimSpace(shallow) != "false" {
		return releaseShipAncestorCIBlocked(fallback, opts.workflow, nil, "repository history is shallow or could not be verified complete")
	}
	workflows := []string{opts.workflow}
	if opts.workflow != "ci-fast.yml" {
		workflows = append(workflows, "ci-fast.yml")
	}
	for _, wf := range workflows {
		args := []string{"run", "list", "--workflow", wf, "--branch", opts.trunk, "--status", "completed", "--event", "push", "--limit", "20", "--json", "databaseId,status,conclusion,url,headSha,updatedAt,event"}
		code, out := releaseShipCmd(result, root, "gh", args, nil, time.Minute)
		// An unreadable workflow can hide failed ancestors; only a successful
		// history query can justify trying the secondary workflow.
		if code != 0 {
			return releaseShipAncestorCIBlocked(fallback, wf, nil, "completed workflow history could not be queried")
		}
		var rows []map[string]any
		if err := json.Unmarshal([]byte(out), &rows); err != nil {
			return releaseShipAncestorCIBlocked(fallback, wf, nil, "completed workflow history was not valid JSON")
		}
		if rows == nil {
			return releaseShipAncestorCIBlocked(fallback, wf, nil, "completed workflow history was not a JSON array")
		}
		if !releaseShipCIEventsReadable(rows) {
			return releaseShipAncestorCIBlocked(fallback, wf, nil, "completed workflow history contained a missing or malformed event")
		}
		rows = releaseShipPushCIRuns(rows)
		var failedAncestors []map[string]any
		for _, row := range rows {
			status := strings.ToLower(stringFromAny(row["status"]))
			conc := strings.ToLower(stringFromAny(row["conclusion"]))
			// Validate every push row before selecting green evidence: a
			// contradictory pending status must not hide a failed conclusion.
			if status != "completed" || conc == "" {
				return releaseShipAncestorCIBlocked(fallback, wf, row, "completed workflow history contained an incomplete or inconsistent push run")
			}
			if releaseShipCIState(status, conc) != "failed" {
				continue
			}
			rsha := strings.TrimSpace(stringFromAny(row["headSha"]))
			if rsha == "" {
				return releaseShipAncestorCIBlocked(fallback, wf, row, "failed run has no source SHA")
			}
			code, _ := releaseShipCmd(result, root, "git", []string{"merge-base", "--is-ancestor", rsha, sha}, nil, releaseShipMergeBaseCommandTimeout)
			if code == 0 {
				failedAncestors = append(failedAncestors, row)
			} else if code != 1 {
				return releaseShipAncestorCIBlocked(fallback, wf, row, "failed run ancestry could not be verified")
			}
		}
		var blocker map[string]any
		if len(failedAncestors) > 0 {
			blocker = failedAncestors[0]
		}
		for _, row := range rows {
			conc := strings.ToLower(stringFromAny(row["conclusion"]))
			rsha := strings.TrimSpace(stringFromAny(row["headSha"]))
			if conc == "success" && rsha != "" {
				mCode, _ := releaseShipCmd(result, root, "git", []string{"merge-base", "--is-ancestor", rsha, sha}, nil, releaseShipMergeBaseCommandTimeout)
				if mCode == 0 {
					// A candidate must include a later fix for every failed run
					// inherited by the source, including failures on merged branches.
					supersedesFailures := true
					for _, failed := range failedAncestors {
						failedSHA := strings.TrimSpace(stringFromAny(failed["headSha"]))
						if failedSHA == rsha {
							blocker = failed
							supersedesFailures = false
							break
						}
						code, _ := releaseShipCmd(result, root, "git", []string{"merge-base", "--is-ancestor", failedSHA, rsha}, nil, releaseShipMergeBaseCommandTimeout)
						if code != 0 {
							blocker = failed
							supersedesFailures = false
							break
						}
					}
					if !supersedesFailures {
						continue
					}
					return map[string]any{
						"ok":                   true,
						"status":               "green",
						"workflow_status":      "completed",
						"conclusion":           "success",
						"source_sha":           sha,
						"adopted_ancestor_sha": rsha,
						"workflow":             wf,
						"run":                  row,
					}
				}
			}
		}
		if len(failedAncestors) > 0 {
			// A secondary workflow cannot erase a failure witnessed by the
			// requested workflow. Preserve the original inconclusive source
			// result and attach the evidence that prevented ancestor adoption.
			return releaseShipAncestorCIBlocked(fallback, wf, blocker, "no green ancestor supersedes the failed source history")
		}
	}
	return fallback
}

func releaseShipAncestorCIBlocked(fallback map[string]any, workflow string, run map[string]any, reason string) map[string]any {
	blocked := make(map[string]any, len(fallback)+1)
	for key, value := range fallback {
		blocked[key] = value
	}
	blocked["ancestor_ci_blocker"] = map[string]any{"workflow": workflow, "run": run, "reason": reason}
	return blocked
}

func releaseShipCIState(status, conclusion string) string {
	if status == "queued" || status == "in_progress" || status == "requested" || status == "waiting" || status == "pending" {
		return "pending"
	}
	switch conclusion {
	case "success":
		return "green"
	case "cancelled", "stale", "skipped":
		return "cancelled_or_starved"
	case "failure", "timed_out", "action_required", "startup_failure":
		return "failed"
	case "":
		return "missing"
	default:
		return "failed"
	}
}
