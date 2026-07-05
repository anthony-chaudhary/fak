package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

const codexLoopSchema = "fak.sessions.codex_loop.v1"

type codexLoopDiagnosis struct {
	Schema            string                 `json:"schema"`
	Path              string                 `json:"path"`
	SessionID         string                 `json:"session_id,omitempty"`
	Originator        string                 `json:"originator,omitempty"`
	CLI               string                 `json:"cli_version,omitempty"`
	ModelProvider     string                 `json:"model_provider,omitempty"`
	GitCommit         string                 `json:"git_commit,omitempty"`
	GitBranch         string                 `json:"git_branch,omitempty"`
	StartedAt         string                 `json:"started_at,omitempty"`
	LastEventAt       string                 `json:"last_event_at,omitempty"`
	FinalStatus       string                 `json:"final_status,omitempty"`
	FinalTokensUsed   int64                  `json:"final_tokens_used,omitempty"`
	FinalTimeSeconds  int64                  `json:"final_time_seconds,omitempty"`
	TurnAborted       bool                   `json:"turn_aborted,omitempty"`
	AbortReason       string                 `json:"abort_reason,omitempty"`
	AbortDurationMS   int64                  `json:"abort_duration_ms,omitempty"`
	ToolCalls         int                    `json:"tool_calls"`
	ToolOutputs       int                    `json:"tool_outputs"`
	LastTokenTotal    int64                  `json:"last_token_total,omitempty"`
	LastTokenInput    int64                  `json:"last_token_input,omitempty"`
	LastTokenOutput   int64                  `json:"last_token_output,omitempty"`
	RepeatedOutcomes  []codexRepeatedOutcome `json:"repeated_outcomes,omitempty"`
	LivelockNotices   []codexLivelockNotice  `json:"livelock_notices,omitempty"`
	Verdict           string                 `json:"verdict"`
	Reason            string                 `json:"reason,omitempty"`
	NextAction        string                 `json:"next_action,omitempty"`
	ObservabilityGaps []string               `json:"observability_gaps,omitempty"`
}

type codexRepeatedOutcome struct {
	Tool             string   `json:"tool"`
	OutputDigest     string   `json:"output_digest"`
	OutputExcerpt    string   `json:"output_excerpt,omitempty"`
	Count            int      `json:"count"`
	LongestRun       int      `json:"longest_run"`
	FirstTimestamp   string   `json:"first_timestamp,omitempty"`
	LastTimestamp    string   `json:"last_timestamp,omitempty"`
	TokenTotal       int64    `json:"token_total,omitempty"`
	TokenEvents      int      `json:"token_events,omitempty"`
	ArgsDigestCount  int      `json:"args_digest_count,omitempty"`
	FirstArgsDigest  string   `json:"first_args_digest,omitempty"`
	OtherArgsDigests []string `json:"other_args_digests,omitempty"`
}

type codexLivelockNotice struct {
	RepeatedCall   string `json:"repeated_call"`
	Approach       string `json:"approach,omitempty"`
	Count          int    `json:"count"`
	MinRepeat      int    `json:"min_repeat,omitempty"`
	MaxRepeat      int    `json:"max_repeat,omitempty"`
	FirstTimestamp string `json:"first_timestamp,omitempty"`
	LastTimestamp  string `json:"last_timestamp,omitempty"`
}

type codexPendingToolCall struct {
	Tool       string
	ArgsDigest string
	Timestamp  string
}

type codexOutcomeKey struct {
	Tool         string
	OutputDigest string
}

type codexOutcomeAccum struct {
	out            codexRepeatedOutcome
	argsDigests    map[string]bool
	currentRun     int
	latestTokenHit bool
}

func sessionsCodexLoop(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("sessions codex-loop", flag.ContinueOnError)
	fs.SetOutput(stderr)
	sessionID := fs.String("session", "", "Codex session id to find under --codex-home/sessions")
	path := fs.String("path", "", "explicit Codex session JSONL path")
	codexHome := fs.String("codex-home", "", "Codex home directory (default: $CODEX_HOME or ~/.codex)")
	asJSON := fs.Bool("json", false, "emit a machine-readable diagnosis")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak sessions codex-loop [--session ID | --path FILE] [--codex-home DIR] [--json]")
		fs.PrintDefaults()
	}
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if strings.TrimSpace(*path) != "" && strings.TrimSpace(*sessionID) != "" {
		fmt.Fprintln(stderr, "fak sessions codex-loop: use only one of --path or --session")
		return 2
	}
	resolved, err := resolveCodexLoopSessionPath(*codexHome, *sessionID, *path)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop: %v\n", err)
		return 2
	}
	fh, err := os.Open(resolved)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop: open %s: %v\n", resolved, err)
		return 1
	}
	defer fh.Close()
	d, err := diagnoseCodexLoop(fh, resolved)
	if err != nil {
		fmt.Fprintf(stderr, "fak sessions codex-loop: %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, d, "fak sessions codex-loop")
	}
	fmt.Fprint(stdout, renderCodexLoopDiagnosis(d))
	return 0
}

func resolveCodexLoopSessionPath(codexHome, sessionID, path string) (string, error) {
	if p := strings.TrimSpace(path); p != "" {
		return filepath.Clean(expandCodexLoopTilde(p)), nil
	}
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return "", errors.New("need --session ID or --path FILE")
	}
	home := strings.TrimSpace(codexHome)
	if home == "" {
		home = os.Getenv("CODEX_HOME")
	}
	if home == "" {
		userHome, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolve home: %w", err)
		}
		home = filepath.Join(userHome, ".codex")
	}
	root := filepath.Join(expandCodexLoopTilde(home), "sessions")
	type candidate struct {
		path  string
		mtime time.Time
	}
	var matches []candidate
	err := filepath.WalkDir(root, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() || !strings.HasSuffix(d.Name(), ".jsonl") {
			return nil
		}
		if !strings.Contains(d.Name(), sessionID) {
			return nil
		}
		info, ierr := d.Info()
		if ierr != nil {
			return nil
		}
		matches = append(matches, candidate{path: p, mtime: info.ModTime()})
		return nil
	})
	if err != nil {
		return "", fmt.Errorf("walk %s: %w", root, err)
	}
	if len(matches) == 0 {
		return "", fmt.Errorf("session %s not found under %s", sessionID, root)
	}
	sort.Slice(matches, func(i, j int) bool { return matches[i].mtime.After(matches[j].mtime) })
	return matches[0].path, nil
}

func expandCodexLoopTilde(path string) string {
	if path == "~" || strings.HasPrefix(path, "~"+string(os.PathSeparator)) || strings.HasPrefix(path, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			rest := strings.TrimPrefix(path, "~")
			rest = strings.TrimPrefix(rest, "/")
			rest = strings.TrimPrefix(rest, string(os.PathSeparator))
			return filepath.Join(home, filepath.FromSlash(rest))
		}
	}
	return path
}

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
			var meta struct {
				SessionID     string `json:"session_id"`
				ID            string `json:"id"`
				Timestamp     string `json:"timestamp"`
				Originator    string `json:"originator"`
				CLIVersion    string `json:"cli_version"`
				ModelProvider string `json:"model_provider"`
				Git           struct {
					CommitHash string `json:"commit_hash"`
					Branch     string `json:"branch"`
				} `json:"git"`
			}
			if json.Unmarshal(rec.Payload, &meta) == nil {
				d.SessionID = firstNonEmpty(meta.SessionID, meta.ID)
				d.StartedAt = firstNonEmpty(meta.Timestamp, rec.Timestamp)
				d.Originator = meta.Originator
				d.CLI = meta.CLIVersion
				d.ModelProvider = meta.ModelProvider
				d.GitCommit = meta.Git.CommitHash
				d.GitBranch = meta.Git.Branch
			}
		case "response_item":
			var item struct {
				Type      string `json:"type"`
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
				CallID    string `json:"call_id"`
				Output    string `json:"output"`
			}
			if json.Unmarshal(rec.Payload, &item) != nil {
				continue
			}
			switch item.Type {
			case "function_call":
				d.ToolCalls++
				calls[item.CallID] = codexPendingToolCall{
					Tool:       strings.TrimSpace(item.Name),
					ArgsDigest: guardrsi.ArgsDigest(item.Arguments),
					Timestamp:  rec.Timestamp,
				}
				pendingTokenOutcome = nil
			case "function_call_output":
				d.ToolOutputs++
				call := calls[item.CallID]
				if call.Tool == "" {
					call.Tool = "unknown"
				}
				key := codexOutcomeKey{Tool: call.Tool, OutputDigest: digestCodexLoopText(normalizeCodexLoopText(item.Output))}
				acc := outcomes[key]
				if acc == nil {
					acc = &codexOutcomeAccum{out: codexRepeatedOutcome{
						Tool:           key.Tool,
						OutputDigest:   key.OutputDigest,
						OutputExcerpt:  codexLoopExcerpt(item.Output),
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
	for _, n := range livelocks {
		d.LivelockNotices = append(d.LivelockNotices, *n)
	}
	sort.Slice(d.LivelockNotices, func(i, j int) bool {
		if d.LivelockNotices[i].MaxRepeat != d.LivelockNotices[j].MaxRepeat {
			return d.LivelockNotices[i].MaxRepeat > d.LivelockNotices[j].MaxRepeat
		}
		return d.LivelockNotices[i].RepeatedCall < d.LivelockNotices[j].RepeatedCall
	})
	classifyCodexLoopDiagnosis(&d)
	return d, nil
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

func classifyCodexLoopDiagnosis(d *codexLoopDiagnosis) {
	if len(d.RepeatedOutcomes) == 0 {
		if len(d.LivelockNotices) > 0 {
			d.Verdict = "ACTION"
			d.Reason = "livelock_advisory_seen"
			d.NextAction = "inspect the repeated_call digest and decide whether the admitted-call advisory should become a hard fuse for this tool class"
			d.ObservabilityGaps = append(d.ObservabilityGaps, "Codex session logs carry livelock advisory text, but not a compact repeated-outcome/token-burn summary")
			return
		}
		d.Verdict = "OK"
		return
	}
	top := d.RepeatedOutcomes[0]
	d.Verdict = "LOOP"
	d.Reason = "repeated_tool_output"
	d.NextAction = "stop re-calling the same tool after an invariant failure; continue from the existing state or add a hard fuse for repeated admitted failures"
	if strings.EqualFold(top.Tool, "create_goal") && strings.Contains(strings.ToLower(top.OutputExcerpt), "unfinished goal") {
		d.NextAction = "for create_goal, read/continue the existing goal instead of creating a new one; hard-fuse repeated unfinished-goal failures after the first repeat"
	}
	d.ObservabilityGaps = append(d.ObservabilityGaps,
		"the live gateway emitted an advisory livelock note but still admitted the next identical host-tool call",
		"the Codex session final status records tokens/time but not the top repeated tool outcome that consumed them",
	)
}

func renderCodexLoopDiagnosis(d codexLoopDiagnosis) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak sessions codex-loop: %s\n", d.Path)
	if d.SessionID != "" {
		fmt.Fprintf(&b, "  session        : %s", d.SessionID)
		if d.ModelProvider != "" {
			fmt.Fprintf(&b, " provider=%s", d.ModelProvider)
		}
		if d.Originator != "" {
			fmt.Fprintf(&b, " originator=%s", d.Originator)
		}
		b.WriteByte('\n')
	}
	fmt.Fprintf(&b, "  verdict        : %s", d.Verdict)
	if d.Reason != "" {
		fmt.Fprintf(&b, " (%s)", d.Reason)
	}
	b.WriteByte('\n')
	if d.FinalStatus != "" || d.FinalTokensUsed > 0 || d.FinalTimeSeconds > 0 {
		fmt.Fprintf(&b, "  final          : status=%s tokens=%d seconds=%d\n", firstNonEmpty(d.FinalStatus, "unknown"), d.FinalTokensUsed, d.FinalTimeSeconds)
	}
	if d.TurnAborted {
		fmt.Fprintf(&b, "  abort          : reason=%s duration_ms=%d\n", firstNonEmpty(d.AbortReason, "unknown"), d.AbortDurationMS)
	}
	if d.LastTokenTotal > 0 {
		fmt.Fprintf(&b, "  last tokens    : total=%d last_input=%d last_output=%d\n", d.LastTokenTotal, d.LastTokenInput, d.LastTokenOutput)
	}
	fmt.Fprintf(&b, "  tool traffic   : calls=%d outputs=%d\n", d.ToolCalls, d.ToolOutputs)
	if len(d.RepeatedOutcomes) > 0 {
		fmt.Fprintf(&b, "  repeated tool outcomes:\n")
		for _, r := range d.RepeatedOutcomes {
			fmt.Fprintf(&b, "    %s output_digest=%s count=%d longest_run=%d tokens=%d",
				r.Tool, r.OutputDigest, r.Count, r.LongestRun, r.TokenTotal)
			if r.ArgsDigestCount > 0 {
				fmt.Fprintf(&b, " args_digests=%d", r.ArgsDigestCount)
			}
			b.WriteByte('\n')
			if r.OutputExcerpt != "" {
				fmt.Fprintf(&b, "      output: %s\n", r.OutputExcerpt)
			}
		}
	}
	if len(d.LivelockNotices) > 0 {
		fmt.Fprintf(&b, "  livelock notices:\n")
		for _, n := range d.LivelockNotices {
			fmt.Fprintf(&b, "    %s repeat=%d..%d count=%d approach=%s\n",
				n.RepeatedCall, n.MinRepeat, n.MaxRepeat, n.Count, n.Approach)
		}
	}
	if len(d.ObservabilityGaps) > 0 {
		fmt.Fprintf(&b, "  observability gaps:\n")
		for _, gap := range d.ObservabilityGaps {
			fmt.Fprintf(&b, "    - %s\n", gap)
		}
	}
	if d.NextAction != "" {
		fmt.Fprintf(&b, "  next action    : %s\n", d.NextAction)
	}
	return b.String()
}

func normalizeCodexLoopText(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

func digestCodexLoopText(s string) string {
	sum := sha256.Sum256([]byte(s))
	return "sha256:" + hex.EncodeToString(sum[:])
}

var codexLoopSecretishRE = regexp.MustCompile(`(?i)(sk-[A-Za-z0-9_-]{8,}|sk-ant-[A-Za-z0-9_-]{8,}|ghp_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]+|bearer\s+[A-Za-z0-9._-]{12,})`)

func codexLoopExcerpt(s string) string {
	s = normalizeCodexLoopText(s)
	s = codexLoopSecretishRE.ReplaceAllString(s, "[REDACTED]")
	const max = 180
	if len(s) > max {
		return s[:max] + "..."
	}
	return s
}
