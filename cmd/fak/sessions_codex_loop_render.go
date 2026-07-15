package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

func renderCodexLoopDiagnosis(d codexLoopDiagnosis) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak sessions codex-loop: %s\n", d.Path)
	if d.SessionID != "" {
		fmt.Fprintf(&b, "  session        : %s", d.SessionID)
		if d.ParentSessionID != "" {
			fmt.Fprintf(&b, " parent=%s", d.ParentSessionID)
		}
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

func renderCodexLoopRecentReport(r codexLoopRecentReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "fak sessions codex-loop --recent: %s\n", r.CodexHome)
	fmt.Fprintf(&b, "  verdict        : %s", r.Verdict)
	if r.Reason != "" {
		fmt.Fprintf(&b, " (%s)", r.Reason)
	}
	b.WriteByte('\n')
	fmt.Fprintf(&b, "  scope          : scanned=%d limit=%d since_hours=%s\n", r.Scanned, r.Limit, trimFloat(r.SinceHours))
	fmt.Fprintf(&b, "  session counts : LOOP=%d ACTION=%d OK=%d\n", r.LoopCount, r.ActionCount, r.OKCount)
	if len(r.ProviderCounts) > 0 {
		fmt.Fprintf(&b, "  providers      : %s", formatCodexProviderCounts(r.ProviderCounts))
		if r.UnguardedCount > 0 {
			fmt.Fprintf(&b, " unguarded=%d", r.UnguardedCount)
		}
		b.WriteByte('\n')
	}
	if r.LoopCount > 0 {
		fmt.Fprintf(&b, "  loop routes    : guarded=%d direct=%d unknown=%d\n", r.GuardedLoopCount, r.UnguardedLoopCount, r.UnknownLoopCount)
	}
	fmt.Fprintf(&b, "  tool traffic   : calls=%d outputs=%d\n", r.ToolCalls, r.ToolOutputs)
	if r.LastTokenTotalSum > 0 {
		fmt.Fprintf(&b, "  token usage    : cumulative-sum=%d (latest counter per rollout)\n", r.LastTokenTotalSum)
	}
	if r.NextAction != "" {
		fmt.Fprintf(&b, "  next action    : %s\n", r.NextAction)
	}
	if len(r.TopRepeated) > 0 {
		fmt.Fprintf(&b, "  top repeated outcomes:\n")
		for _, out := range r.TopRepeated {
			fmt.Fprintf(&b, "    %s output_digest=%s count=%d longest_run=%d tokens=%d",
				out.Tool, out.OutputDigest, out.Count, out.LongestRun, out.TokenTotal)
			if out.ArgsDigestCount > 0 {
				fmt.Fprintf(&b, " args_digests=%d", out.ArgsDigestCount)
			}
			b.WriteByte('\n')
			if out.OutputExcerpt != "" {
				fmt.Fprintf(&b, "      output: %s\n", out.OutputExcerpt)
			}
		}
	}
	if len(r.Diagnoses) > 0 {
		fmt.Fprintf(&b, "  sessions:\n")
		for _, d := range r.Diagnoses {
			label := firstNonEmpty(d.SessionID, filepath.Base(d.Path))
			fmt.Fprintf(&b, "    %s verdict=%s", label, d.Verdict)
			if d.ParentSessionID != "" {
				fmt.Fprintf(&b, " parent=%s", d.ParentSessionID)
			}
			if d.Reason != "" {
				fmt.Fprintf(&b, " reason=%s", d.Reason)
			}
			if d.LastTokenTotal > 0 {
				fmt.Fprintf(&b, " last_tokens=%d", d.LastTokenTotal)
			}
			if len(d.RepeatedOutcomes) > 0 {
				top := d.RepeatedOutcomes[0]
				fmt.Fprintf(&b, " top=%s:%d", top.Tool, top.Count)
			}
			b.WriteByte('\n')
		}
	}
	return b.String()
}

func codexLoopRawText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text
	}
	var blocks []struct {
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &blocks) == nil {
		parts := make([]string, 0, len(blocks))
		for _, block := range blocks {
			if block.Text != "" {
				parts = append(parts, block.Text)
			}
		}
		if len(parts) > 0 {
			return strings.Join(parts, "\n")
		}
	}
	return string(raw)
}

func formatCodexProviderCounts(counts map[string]int) string {
	keys := make([]string, 0, len(counts))
	for key := range counts {
		if strings.TrimSpace(key) != "" {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", key, counts[key]))
	}
	return strings.Join(parts, " ")
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
