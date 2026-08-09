// Codex transcript diagnosis: the rollout scanner and the verdict classifier for a
// single Codex session. diagnoseCodexLoop walks the JSONL transcript and folds the
// accumulated tool outcomes, livelock advisories and token counters into a
// codexLoopDiagnosis; the helpers below it sort, summarize and classify what it
// accumulated. Moved here verbatim from the sessions codex loop command unit -- a
// pure relocation with no behaviour change -- so that unit stays under the god-file
// line ceiling the hooks gate enforces.

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

func diagnoseCodexLoop(r io.Reader, path string) (codexLoopDiagnosis, error) {
	d := codexLoopDiagnosis{
		Schema:  codexLoopSchema,
		Path:    path,
		Verdict: "OK",
	}
	calls := map[string]codexPendingToolCall{}
	outcomes := map[codexOutcomeKey]*codexOutcomeAccum{}
	livelocks := map[string]*codexLivelockNotice{}
	var prevOutcome *codexOutcomeKey
	var pendingTokenOutcome *codexOutcomeKey
	var lastResponseItemType string
	var lastResponseCallID string

	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 256*1024), 64*1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var rec struct {
			Timestamp string          `json:"timestamp"`
			Type      string          `json:"type"`
			Payload   json.RawMessage `json:"payload"`
		}
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			continue
		}
		d.LastEventAt = rec.Timestamp
		switch rec.Type {
		case "session_meta":
			applyCodexLoopSessionMeta(&d, rec.Timestamp, rec.Payload)
		case "response_item":
			var item struct {
				Type      string          `json:"type"`
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
				Input     json.RawMessage `json:"input"`
				CallID    string          `json:"call_id"`
				Output    json.RawMessage `json:"output"`
			}
			if json.Unmarshal(rec.Payload, &item) != nil {
				continue
			}
			lastResponseItemType = item.Type
			lastResponseCallID = item.CallID
			switch item.Type {
			case "function_call", "custom_tool_call":
				d.ToolCalls++
				args := item.Arguments
				if len(args) == 0 {
					args = item.Input
				}
				calls[item.CallID] = codexPendingToolCall{
					Tool:       strings.TrimSpace(item.Name),
					ArgsDigest: guardrsi.ArgsDigest(codexLoopRawText(args)),
					Timestamp:  rec.Timestamp,
				}
				pendingTokenOutcome = nil
			case "function_call_output", "custom_tool_call_output":
				d.ToolOutputs++
				call := calls[item.CallID]
				if call.Tool == "" {
					call.Tool = "unknown"
				}
				output := codexLoopRawText(item.Output)
				key := codexOutcomeKey{Tool: call.Tool, OutputDigest: digestCodexLoopText(normalizeCodexLoopText(output))}
				acc := outcomes[key]
				if acc == nil {
					acc = &codexOutcomeAccum{out: codexRepeatedOutcome{
						Tool:           key.Tool,
						OutputDigest:   key.OutputDigest,
						OutputExcerpt:  codexLoopExcerpt(output),
						FirstTimestamp: rec.Timestamp,
					}, argsDigests: map[string]bool{}}
					outcomes[key] = acc
				}
				acc.out.Count++
				acc.out.LastTimestamp = rec.Timestamp
				if call.ArgsDigest != "" {
					acc.argsDigests[call.ArgsDigest] = true
					if acc.out.FirstArgsDigest == "" {
						acc.out.FirstArgsDigest = call.ArgsDigest
					}
				}
				if prevOutcome != nil && *prevOutcome == key {
					acc.currentRun++
				} else {
					acc.currentRun = 1
				}
				if acc.currentRun > acc.out.LongestRun {
					acc.out.LongestRun = acc.currentRun
				}
				copyKey := key
				prevOutcome = &copyKey
				pendingTokenOutcome = &copyKey
				delete(calls, item.CallID)
			}
		case "event_msg":
			var ev struct {
				Type    string          `json:"type"`
				Message string          `json:"message"`
				Info    json.RawMessage `json:"info"`
				Goal    *struct {
					Objective       string `json:"objective"`
					Status          string `json:"status"`
					TokensUsed      int64  `json:"tokensUsed"`
					TimeUsedSeconds int64  `json:"timeUsedSeconds"`
				} `json:"goal"`
				Reason     string `json:"reason"`
				DurationMS int64  `json:"duration_ms"`
			}
			if json.Unmarshal(rec.Payload, &ev) != nil {
				continue
			}
			switch ev.Type {
			case "token_count":
				tok := parseCodexLoopTokenInfo(ev.Info)
				d.LastTokenTotal = tok.total
				d.LastTokenInput = tok.input
				d.LastTokenOutput = tok.output
				if pendingTokenOutcome != nil {
					if acc := outcomes[*pendingTokenOutcome]; acc != nil {
						acc.out.TokenTotal += tok.last
						acc.out.TokenEvents++
					}
					pendingTokenOutcome = nil
				}
			case "agent_message":
				if notice, ok := parseCodexLivelockNotice(rec.Timestamp, ev.Message); ok {
					cur := livelocks[notice.RepeatedCall+"\x00"+notice.Approach]
					if cur == nil {
						livelocks[notice.RepeatedCall+"\x00"+notice.Approach] = &notice
					} else {
						cur.Count++
						cur.LastTimestamp = notice.LastTimestamp
						if notice.MinRepeat < cur.MinRepeat {
							cur.MinRepeat = notice.MinRepeat
						}
						if notice.MaxRepeat > cur.MaxRepeat {
							cur.MaxRepeat = notice.MaxRepeat
						}
					}
				}
			case "thread_goal_updated":
				if ev.Goal != nil {
					d.FinalStatus = ev.Goal.Status
					d.FinalTokensUsed = ev.Goal.TokensUsed
					d.FinalTimeSeconds = ev.Goal.TimeUsedSeconds
				}
			case "turn_aborted":
				d.TurnAborted = true
				d.AbortReason = ev.Reason
				d.AbortDurationMS = ev.DurationMS
			}
		}
	}
	if err := sc.Err(); err != nil {
		return d, err
	}
	if lastResponseItemType == "function_call" || lastResponseItemType == "custom_tool_call" {
		_, pending := calls[lastResponseCallID]
		d.abruptlyEnded = pending && !d.TurnAborted && strings.TrimSpace(d.FinalStatus) == ""
	}

	appendCodexLoopRepeatedOutcomes(&d, outcomes)
	appendCodexLoopLivelockNotices(&d, livelocks)
	classifyCodexLoopDiagnosis(&d)
	return d, nil
}

func applyCodexLoopSessionMeta(d *codexLoopDiagnosis, ts string, payload json.RawMessage) {
	// A subagent rollout starts with its own metadata, then carries the parent
	// session metadata in the inherited context. Only the first record identifies
	// this file; allowing later records to overwrite it makes every child look like
	// the same parent session and corrupts the recent-session report.
	if d.SessionID != "" {
		return
	}
	var meta struct {
		SessionID     string `json:"session_id"`
		ID            string `json:"id"`
		Timestamp     string `json:"timestamp"`
		Originator    string `json:"originator"`
		CLIVersion    string `json:"cli_version"`
		ModelProvider string `json:"model_provider"`
		WorkingDir    string `json:"cwd"`
		Git           struct {
			CommitHash string `json:"commit_hash"`
			Branch     string `json:"branch"`
		} `json:"git"`
	}
	if json.Unmarshal(payload, &meta) == nil {
		d.SessionID = firstNonEmpty(meta.ID, meta.SessionID)
		if meta.ID != "" && meta.SessionID != "" && meta.ID != meta.SessionID {
			d.ParentSessionID = meta.SessionID
		}
		d.StartedAt = firstNonEmpty(meta.Timestamp, ts)
		d.Originator = meta.Originator
		d.CLI = meta.CLIVersion
		d.ModelProvider = meta.ModelProvider
		d.WorkingDir = meta.WorkingDir
		d.GitCommit = meta.Git.CommitHash
		d.GitBranch = meta.Git.Branch
	}
}

func appendCodexLoopRepeatedOutcomes(d *codexLoopDiagnosis, outcomes map[codexOutcomeKey]*codexOutcomeAccum) {
	for _, acc := range outcomes {
		if acc.out.Count < 3 && acc.out.LongestRun < 3 {
			continue
		}
		digests := make([]string, 0, len(acc.argsDigests))
		for digest := range acc.argsDigests {
			digests = append(digests, digest)
		}
		sort.Strings(digests)
		acc.out.ArgsDigestCount = len(digests)
		if acc.out.FirstArgsDigest == "" && len(digests) > 0 {
			acc.out.FirstArgsDigest = digests[0]
		}
		for _, digest := range digests {
			if digest == acc.out.FirstArgsDigest {
				continue
			}
			acc.out.OtherArgsDigests = append(acc.out.OtherArgsDigests, digest)
		}
		if len(acc.out.OtherArgsDigests) > 4 {
			acc.out.OtherArgsDigests = acc.out.OtherArgsDigests[:4]
		}
		d.RepeatedOutcomes = append(d.RepeatedOutcomes, acc.out)
	}
	sort.Slice(d.RepeatedOutcomes, func(i, j int) bool {
		if d.RepeatedOutcomes[i].TokenTotal != d.RepeatedOutcomes[j].TokenTotal {
			return d.RepeatedOutcomes[i].TokenTotal > d.RepeatedOutcomes[j].TokenTotal
		}
		if d.RepeatedOutcomes[i].LongestRun != d.RepeatedOutcomes[j].LongestRun {
			return d.RepeatedOutcomes[i].LongestRun > d.RepeatedOutcomes[j].LongestRun
		}
		if d.RepeatedOutcomes[i].Count != d.RepeatedOutcomes[j].Count {
			return d.RepeatedOutcomes[i].Count > d.RepeatedOutcomes[j].Count
		}
		return d.RepeatedOutcomes[i].Tool < d.RepeatedOutcomes[j].Tool
	})
	if len(d.RepeatedOutcomes) > 5 {
		d.RepeatedOutcomes = d.RepeatedOutcomes[:5]
	}
}

func appendCodexLoopLivelockNotices(d *codexLoopDiagnosis, livelocks map[string]*codexLivelockNotice) {
	for _, n := range livelocks {
		d.LivelockNotices = append(d.LivelockNotices, *n)
	}
	sort.Slice(d.LivelockNotices, func(i, j int) bool {
		if d.LivelockNotices[i].MaxRepeat != d.LivelockNotices[j].MaxRepeat {
			return d.LivelockNotices[i].MaxRepeat > d.LivelockNotices[j].MaxRepeat
		}
		return d.LivelockNotices[i].RepeatedCall < d.LivelockNotices[j].RepeatedCall
	})
}

type codexLoopTokenInfo struct {
	total  int64
	last   int64
	input  int64
	output int64
}

func parseCodexLoopTokenInfo(raw json.RawMessage) codexLoopTokenInfo {
	var info struct {
		Total struct {
			TotalTokens  int64 `json:"total_tokens"`
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
		} `json:"total_token_usage"`
		Last struct {
			TotalTokens  int64 `json:"total_tokens"`
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
		} `json:"last_token_usage"`
	}
	_ = json.Unmarshal(raw, &info)
	return codexLoopTokenInfo{
		total:  info.Total.TotalTokens,
		last:   info.Last.TotalTokens,
		input:  info.Last.InputTokens,
		output: info.Last.OutputTokens,
	}
}

var codexLivelockRE = regexp.MustCompile(`LIVELOCK_DETECTED repeat=([0-9]+) repeated_call=([^ ]+) approach=([^ .]+)`)

func parseCodexLivelockNotice(ts, msg string) (codexLivelockNotice, bool) {
	m := codexLivelockRE.FindStringSubmatch(msg)
	if len(m) != 4 {
		return codexLivelockNotice{}, false
	}
	repeat, _ := strconv.Atoi(m[1])
	return codexLivelockNotice{
		RepeatedCall:   m[2],
		Approach:       m[3],
		Count:          1,
		MinRepeat:      repeat,
		MaxRepeat:      repeat,
		FirstTimestamp: ts,
		LastTimestamp:  ts,
	}, true
}

// codexProgressAckTools are host tools whose output is a constant, content-free
// acknowledgment by design — update_plan always answers "Plan updated" no matter
// which plan it recorded. For these a run of identical outputs across FULLY
// DISTINCT arguments is forward progress (a new plan each turn), not a no-progress
// loop. The same run from a real work tool (exec, shell_command) still signals a
// loop, so this set is deliberately tiny and role-based rather than a blanket
// "benign-looking output" heuristic. (#4278, Epic #4277)
var codexProgressAckTools = map[string]bool{
	"update_plan": true,
}

// codexOutcomeIsForwardProgress reports whether a repeated outcome is distinct
// forward progress rather than a stuck repetition: a constant-ack progress tool
// whose every call carried a distinct argument digest. A progress tool that
// re-submits the SAME arguments (ArgsDigestCount < Count) is still thrashing and
// stays a loop signal, as does any non-progress tool.
func codexOutcomeIsForwardProgress(o codexRepeatedOutcome) bool {
	if !codexProgressAckTools[strings.ToLower(strings.TrimSpace(o.Tool))] {
		return false
	}
	// ArgsDigest never yields an empty digest (empty args canonicalize to "{}"),
	// so ArgsDigestCount == Count means every call carried distinct arguments.
	return o.Count > 0 && o.ArgsDigestCount >= o.Count
}

// codexTopLoopDrivingOutcome returns the highest-ranked repeated outcome that is a
// genuine no-progress loop, skipping forward-progress progress-tool outcomes. The
// bool is false when there is no loop-driving outcome (empty or all forward
// progress).
func codexTopLoopDrivingOutcome(outcomes []codexRepeatedOutcome) (codexRepeatedOutcome, bool) {
	for _, o := range outcomes {
		if codexOutcomeIsForwardProgress(o) {
			continue
		}
		return o, true
	}
	return codexRepeatedOutcome{}, false
}

// applyCodexLoopForwardProgressNote annotates an OK diagnosis whose only repeated
// outcomes were forward progress, so a reader is not puzzled by an OK verdict
// sitting next to a populated repeated-outcome list (and the launch gate can name
// why a high-traffic progress tool was not fused).
func applyCodexLoopForwardProgressNote(d *codexLoopDiagnosis) {
	var progress []string
	for _, o := range d.RepeatedOutcomes {
		if codexOutcomeIsForwardProgress(o) {
			progress = append(progress, fmt.Sprintf("%s:%d", o.Tool, o.Count))
		}
	}
	if len(progress) == 0 {
		return
	}
	d.Reason = "repeated_progress_tool_no_loop"
	d.NextAction = "no hard fuse needed: " + strings.Join(progress, ", ") +
		" repeated a constant acknowledgment across fully distinct arguments (forward planning progress), not a no-progress loop"
}

func classifyCodexLoopDiagnosis(d *codexLoopDiagnosis) {
	top, hasLoop := codexTopLoopDrivingOutcome(d.RepeatedOutcomes)
	if !hasLoop {
		if len(d.LivelockNotices) > 0 {
			d.Verdict = "ACTION"
			d.Reason = "livelock_advisory_seen"
			d.NextAction = "inspect the repeated_call digest and decide whether the admitted-call advisory should become a hard fuse for this tool class"
			d.ObservabilityGaps = append(d.ObservabilityGaps, "Codex session logs carry livelock advisory text, but not a compact repeated-outcome/token-burn summary")
			return
		}
		d.Verdict = "OK"
		applyCodexLoopForwardProgressNote(d)
		applyCodexLoopUnguardedGuidance(d)
		return
	}
	d.Verdict = "LOOP"
	d.Reason = "repeated_tool_output"
	d.NextAction = "stop re-calling the same tool after an invariant failure; continue from the existing state or add a hard fuse for repeated admitted failures"
	if strings.EqualFold(top.Tool, "create_goal") && strings.Contains(strings.ToLower(top.OutputExcerpt), "unfinished goal") {
		d.NextAction = "for create_goal, read/continue the existing goal instead of creating a new one; hard-fuse repeated unfinished-goal failures after the first repeat"
	}
	if applyCodexLoopUnguardedGuidance(d) {
		return
	}
	d.ObservabilityGaps = append(d.ObservabilityGaps,
		"the live gateway emitted an advisory livelock note but still admitted the next identical host-tool call",
		"the Codex session final status records tokens/time but not the top repeated tool outcome that consumed them",
	)
}

func applyCodexLoopUnguardedGuidance(d *codexLoopDiagnosis) bool {
	if !codexLoopDiagnosisUnguarded(*d) {
		return false
	}
	d.NextAction = "launch future Codex sessions through `fak codex` or `fak guard -- codex`; direct model_provider=" + d.ModelProvider + " sessions cannot use the gateway's repeated-result fuse"
	if len(d.RepeatedOutcomes) > 0 {
		d.ObservabilityGaps = append(d.ObservabilityGaps,
			"this Codex session bypassed fak guard (model_provider="+d.ModelProvider+"), so the gateway could not hard-fuse the repeated tool outcome",
			"the Codex session final status records tokens/time but not the top repeated tool outcome that consumed them",
		)
		return true
	}
	d.ObservabilityGaps = append(d.ObservabilityGaps,
		"this Codex session bypassed fak guard (model_provider="+d.ModelProvider+"), so the gateway cannot hard-fuse repeated tool outcomes inside this process",
	)
	return true
}
