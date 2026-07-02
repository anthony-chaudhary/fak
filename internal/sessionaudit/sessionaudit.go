// Package sessionaudit audits Claude Code session-transcript JSONL files.
package sessionaudit

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

var Pricing = map[string]Rates{
	"opus":   {Input: 15.0, CacheWrite: 18.75, CacheRead: 1.50, Output: 75.0},
	"sonnet": {Input: 3.0, CacheWrite: 3.75, CacheRead: 0.30, Output: 15.0},
	"haiku":  {Input: 0.80, CacheWrite: 1.00, CacheRead: 0.08, Output: 4.0},
	"fable":  {Input: 3.0, CacheWrite: 3.75, CacheRead: 0.30, Output: 15.0},
}

var pricingOrder = []string{"opus", "sonnet", "haiku", "fable"}

var ReadOnlyTools = map[string]bool{
	"Read":                   true,
	"Glob":                   true,
	"Grep":                   true,
	"LS":                     true,
	"NotebookRead":           true,
	"WebFetch":               true,
	"WebSearch":              true,
	"TodoRead":               true,
	"ToolSearch":             true,
	"Monitor":                true,
	"TaskGet":                true,
	"TaskList":               true,
	"TaskOutput":             true,
	"ReadMcpResourceTool":    true,
	"ListMcpResourcesTool":   true,
	"ReadMcpResourceDirTool": true,
}

var ExcludeNamespaceSubstrings = []string{"pytest-of-USER", "AppData-Local-Temp", "workspace", "-ws", "test_"}

const NamespaceIncludePrefix = ""

type Rates struct {
	Input      float64 `json:"input"`
	CacheWrite float64 `json:"cache_write_5m"`
	CacheRead  float64 `json:"cache_read"`
	Output     float64 `json:"output"`
}

type DiscoverOptions struct {
	Roots            []string
	SinceDays        *float64
	NamespacePrefix  string
	IncludeSubagents bool
}

type Transcript struct {
	Root  string  `json:"root"`
	NS    string  `json:"ns"`
	Path  string  `json:"path"`
	Kind  string  `json:"kind"`
	Size  int64   `json:"size"`
	MTime float64 `json:"mtime"`
}

type TokenCounts struct {
	Input       int64 `json:"input"`
	Output      int64 `json:"output"`
	CacheRead   int64 `json:"cache_read"`
	CacheCreate int64 `json:"cache_create"`
	WebSearch   int64 `json:"web_search"`
	WebFetch    int64 `json:"web_fetch"`
	Iterations  int64 `json:"iterations"`
}

type ModelCounts struct {
	Turns       int64 `json:"turns"`
	Input       int64 `json:"input"`
	Output      int64 `json:"output"`
	CacheRead   int64 `json:"cache_read"`
	CacheCreate int64 `json:"cache_create"`
}

type Prompt struct {
	Timestamp string `json:"timestamp"`
	Text      string `json:"text"`
}

type Session struct {
	Path              string                 `json:"path"`
	Session           string                 `json:"session"`
	Kind              string                 `json:"kind,omitempty"`
	Error             string                 `json:"error,omitempty"`
	NRecords          int64                  `json:"n_records"`
	RecordTypes       map[string]int64       `json:"rec_types"`
	Models            map[string]int64       `json:"models"`
	PerModel          map[string]ModelCounts `json:"per_model"`
	AssistantTurns    int64                  `json:"assistant_turns"`
	DupAssistantLines int64                  `json:"dup_assistant_lines"`
	NPrompts          int64                  `json:"n_prompts"`
	Prompts           []Prompt               `json:"prompts,omitempty"`
	NToolUse          int64                  `json:"n_tool_use"`
	NToolResult       int64                  `json:"n_tool_result"`
	Tools             map[string]int64       `json:"tools"`
	ReadOnlyToolCalls int64                  `json:"read_only_tool_calls"`
	ReadOnlyFrac      *float64               `json:"read_only_frac"`
	ToolInputChars    int64                  `json:"tool_input_chars"`
	ToolResultChars   int64                  `json:"tool_result_chars"`
	NThinking         int64                  `json:"n_thinking"`
	NText             int64                  `json:"n_text"`
	Interrupted       int64                  `json:"interrupted"`
	Tokens            TokenCounts            `json:"tokens"`
	TotalInputTokens  int64                  `json:"total_input_tokens"`
	IORatio           *float64               `json:"io_ratio"`
	CacheHitFrac      *float64               `json:"cache_hit_frac"`
	CostUSD           float64                `json:"cost_usd"`
	TSMin             string                 `json:"ts_min,omitempty"`
	TSMax             string                 `json:"ts_max,omitempty"`
	WallSeconds       *float64               `json:"wall_s"`
}

type Aggregate struct {
	NSessions             int64                  `json:"n_sessions"`
	Totals                TokenCounts            `json:"totals"`
	TotalCostUSD          float64                `json:"total_cost_usd"`
	ToolMix               map[string]int64       `json:"tool_mix"`
	PerNamespace          map[string]Namespace   `json:"per_namespace"`
	PerNamespaceCost      map[string]float64     `json:"per_namespace_cost"`
	PerNamespaceTopModel  map[string]string      `json:"per_namespace_top_model"`
	PerNamespaceOpusShare map[string]*float64    `json:"per_namespace_opus_share"`
	PerModel              map[string]ModelCounts `json:"per_model"`
	PerBucket             map[string]ModelCounts `json:"per_bucket"`
	PerTier               map[string]ModelCounts `json:"per_tier"`
	Distributions         Distributions          `json:"dist"`
}

type Namespace struct {
	Sessions  int64 `json:"sessions"`
	Output    int64 `json:"output"`
	CacheRead int64 `json:"cache_read"`
	ToolUse   int64 `json:"tool_use"`
}

type Distributions struct {
	CallsPerSession        StatSet `json:"calls_per_session"`
	OutputTokensPerSession StatSet `json:"output_tokens_per_session"`
	IORatio                StatSet `json:"io_ratio"`
	CacheHitFrac           StatSet `json:"cache_hit_frac"`
	ReadOnlyFrac           StatSet `json:"read_only_frac"`
}

type StatSet struct {
	Median *float64 `json:"median"`
	Mean   *float64 `json:"mean,omitempty"`
	P10    *float64 `json:"p10,omitempty"`
	P90    *float64 `json:"p90,omitempty"`
	Max    *float64 `json:"max,omitempty"`
}

type Summary struct {
	Count   int64       `json:"count"`
	Tokens  TokenCounts `json:"tokens"`
	CostUSD float64     `json:"cost_usd"`
}

type AuditPayload struct {
	Aggregate          Aggregate `json:"aggregate"`
	ExcludedSubagents  *Summary  `json:"excluded_subagents,omitempty"`
	Sessions           []Session `json:"sessions"`
	SubagentSummary    *Summary  `json:"subagent_summary,omitempty"`
	SubagentTranscript int       `json:"subagent_transcripts,omitempty"`
}

func DefaultRoots() []string {
	base := os.Getenv("CLAUDE_CONFIG_DIR")
	if base == "" {
		if home, err := os.UserHomeDir(); err == nil {
			base = filepath.Join(home, ".claude")
		} else {
			base = ".claude"
		}
	}
	return []string{filepath.Join(base, "projects")}
}

func Discover(opts DiscoverOptions) ([]Transcript, error) {
	roots := opts.Roots
	if len(roots) == 0 {
		roots = DefaultRoots()
	}
	var cutoff time.Time
	if opts.SinceDays != nil {
		cutoff = time.Now().Add(-time.Duration(*opts.SinceDays * float64(24*time.Hour)))
	}
	var out []Transcript
	for _, root := range roots {
		entries, err := os.ReadDir(root)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, err
		}
		for _, entry := range entries {
			if !entry.IsDir() {
				continue
			}
			ns := entry.Name()
			if excludedNamespace(ns) || (opts.NamespacePrefix != "" && !strings.HasPrefix(ns, opts.NamespacePrefix)) {
				continue
			}
			nsdir := filepath.Join(root, ns)
			top := map[string]bool{}
			files, err := filepath.Glob(filepath.Join(nsdir, "*.jsonl"))
			if err != nil {
				return nil, err
			}
			for _, p := range files {
				top[p] = true
				if rec, ok := statTranscript(root, ns, p, "session", cutoff); ok {
					out = append(out, rec)
				}
			}
			if !opts.IncludeSubagents {
				continue
			}
			err = filepath.WalkDir(nsdir, func(path string, d os.DirEntry, err error) error {
				if err != nil || d.IsDir() || filepath.Ext(path) != ".jsonl" || top[path] {
					return nil
				}
				if rec, ok := statTranscript(root, ns, path, "subagent", cutoff); ok {
					out = append(out, rec)
				}
				return nil
			})
			if err != nil {
				return nil, err
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].MTime == out[j].MTime {
			return out[i].Path < out[j].Path
		}
		return out[i].MTime > out[j].MTime
	})
	return out, nil
}

func Analyze(path string) Session {
	s := Session{
		Path:        path,
		Session:     strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)),
		RecordTypes: map[string]int64{},
		Models:      map[string]int64{},
		PerModel:    map[string]ModelCounts{},
		Tools:       map[string]int64{},
	}
	f, err := os.Open(path)
	if err != nil {
		s.Error = err.Error()
		return s
	}
	defer f.Close()

	seen := map[string]bool{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 64*1024), 64*1024*1024)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var r transcriptRecord
		if err := json.Unmarshal(line, &r); err != nil {
			continue
		}
		s.RecordTypes[r.Type]++
		s.NRecords++
		if r.Timestamp != "" {
			if s.TSMin == "" || r.Timestamp < s.TSMin {
				s.TSMin = r.Timestamp
			}
			if s.TSMax == "" || r.Timestamp > s.TSMax {
				s.TSMax = r.Timestamp
			}
		}
		switch r.Type {
		case "assistant":
			analyzeAssistant(&s, r, seen)
		case "user":
			analyzeUser(&s, r)
		}
	}
	if err := sc.Err(); err != nil {
		s.Error = err.Error()
		return s
	}
	finalizeSession(&s)
	return s
}

func AggregateSessions(sessions []Session) Aggregate {
	agg := Aggregate{
		ToolMix:               map[string]int64{},
		PerNamespace:          map[string]Namespace{},
		PerNamespaceCost:      map[string]float64{},
		PerNamespaceTopModel:  map[string]string{},
		PerNamespaceOpusShare: map[string]*float64{},
		PerModel:              map[string]ModelCounts{},
		PerBucket:             map[string]ModelCounts{},
		PerTier:               map[string]ModelCounts{},
	}
	nsModels := map[string]map[string]int64{}
	var calls, outs, ios, cacheHits, rofs []float64
	for _, s := range sessions {
		if s.Error != "" {
			continue
		}
		agg.NSessions++
		addTokens(&agg.Totals, s.Tokens)
		agg.TotalCostUSD += s.CostUSD
		addMap(agg.ToolMix, s.Tools)
		ns := namespaceName(s.Path)
		n := agg.PerNamespace[ns]
		n.Sessions++
		n.Output += s.Tokens.Output
		n.CacheRead += s.Tokens.CacheRead
		n.ToolUse += s.NToolUse
		agg.PerNamespace[ns] = n
		agg.PerNamespaceCost[ns] += s.CostUSD
		if nsModels[ns] == nil {
			nsModels[ns] = map[string]int64{}
		}
		for model, c := range s.PerModel {
			nsModels[ns][model] += c.Output
			agg.PerModel[model] = addModelCounts(agg.PerModel[model], c)
		}
		calls = append(calls, float64(s.NToolUse))
		outs = append(outs, float64(s.Tokens.Output))
		if s.IORatio != nil {
			ios = append(ios, *s.IORatio)
		}
		if s.CacheHitFrac != nil {
			cacheHits = append(cacheHits, *s.CacheHitFrac)
		}
		if s.ReadOnlyFrac != nil {
			rofs = append(rofs, *s.ReadOnlyFrac)
		}
	}
	for model, c := range agg.PerModel {
		b := ProviderBucket(model)
		agg.PerBucket[b] = addModelCounts(agg.PerBucket[b], c)
		t := ModelTier(model)
		agg.PerTier[t] = addModelCounts(agg.PerTier[t], c)
	}
	for ns, models := range nsModels {
		var top string
		var topOut, totalOut, opusOut int64
		for model, out := range models {
			totalOut += out
			if out > topOut || (out == topOut && (top == "" || model < top)) {
				top, topOut = model, out
			}
			if ModelTier(model) == "opus" {
				opusOut += out
			}
		}
		if top == "" {
			top = "?"
		}
		agg.PerNamespaceTopModel[ns] = top
		if totalOut == 0 {
			agg.PerNamespaceOpusShare[ns] = nil
		} else {
			v := float64(opusOut) / float64(totalOut)
			agg.PerNamespaceOpusShare[ns] = &v
		}
	}
	agg.Distributions = Distributions{
		CallsPerSession:        stat(calls, true, false, true),
		OutputTokensPerSession: stat(outs, false, false, true),
		IORatio:                stat(ios, false, true, false),
		CacheHitFrac:           stat(cacheHits, false, true, false),
		ReadOnlyFrac:           stat(rofs, false, false, false),
	}
	return agg
}

func SummarizeAnalyses(sessions []Session) Summary {
	var sum Summary
	for _, s := range sessions {
		if s.Error != "" {
			continue
		}
		sum.Count++
		addTokens(&sum.Tokens, s.Tokens)
		sum.CostUSD += s.CostUSD
	}
	return sum
}

func SummarizeTranscripts(records []Transcript) Summary {
	sessions := make([]Session, 0, len(records))
	for _, r := range records {
		sessions = append(sessions, Analyze(r.Path))
	}
	return SummarizeAnalyses(sessions)
}

func ReportMarkdown(sessions []Session, agg Aggregate, nsPrefix string, sinceDays *float64, includeSubagents bool, maxSessions int, discoveredCount int, excludedSubagents *Summary, generated time.Time) string {
	var b strings.Builder
	ok := validSessions(sessions)
	fmt.Fprintln(&b, "# Session-Transcript Audit - active scope")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "**Generated:** %s  \n", generated.Format("2006-01-02T15:04:05"))
	fmt.Fprintf(&b, "**Top-level sessions audited:** %d  .  **Tool:** `fak session-audit` (re-runnable)  \n", agg.NSessions)
	fmt.Fprintf(&b, "**Scope:** %s\n", scopeLine(ok, nsPrefix, sinceDays, includeSubagents, maxSessions))
	if note := maxClipNote(maxSessions, discoveredCount); note != "" {
		fmt.Fprintln(&b, note)
	}
	if note := subagentNote(excludedSubagents); note != "" {
		fmt.Fprintln(&b, note)
	}
	t := agg.Totals
	totalIn := t.Input + t.CacheRead + t.CacheCreate
	fmt.Fprintln(&b, "## Scope totals (EXACT token counts)")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "- **Output tokens (the actual work generated):** %s\n", fmtInt(t.Output))
	fmt.Fprintf(&b, "- **Fresh input tokens (billed, non-cached):** %s\n", fmtInt(t.Input))
	fmt.Fprintf(&b, "- **Cache-read tokens (prompt-cache / KV reuse):** %s\n", fmtInt(t.CacheRead))
	fmt.Fprintf(&b, "- **Cache-creation tokens:** %s\n", fmtInt(t.CacheCreate))
	fmt.Fprintf(&b, "- **Total context ingested:** %s  ->  **machine-wide I:O ratio = %.1f : 1**\n", fmtInt(totalIn), float64(totalIn)/float64(max64(t.Output, 1)))
	fmt.Fprintf(&b, "- **Cache-read share of all ingested context = %.1f%%**\n", float64(t.CacheRead)/float64(max64(totalIn, 1))*100)
	fmt.Fprintf(&b, "- **Web requests - server-tool (`server_tool_use`, billed):** search %s / fetch %s  .  **client tool:** WebSearch %s / WebFetch %s\n",
		fmtInt(t.WebSearch), fmtInt(t.WebFetch), fmtInt(agg.ToolMix["WebSearch"]), fmtInt(agg.ToolMix["WebFetch"]))
	fmt.Fprintf(&b, "- **Multi-iteration count:** %s\n", fmtInt(t.Iterations))
	fmt.Fprintf(&b, "- **Estimated Anthropic-billed cost:** $%s  _(cost uses an ASSUMED price table; token counts above are exact)_\n", fmtFloat(agg.TotalCostUSD, 2))
	if other := otherBuckets(agg.PerBucket); len(other) > 0 {
		parts := make([]string, 0, len(other))
		for _, bucket := range other {
			c := agg.PerBucket[bucket]
			parts = append(parts, fmt.Sprintf("%s (%s output tok, unpriced - add its card)", bucket, fmtInt(c.Output)))
		}
		fmt.Fprintf(&b, "- **Other billing buckets present (NOT in the total above - different invoices):** %s\n", strings.Join(parts, "; "))
	}
	if nb := agg.PerBucket["non-billed (harness)"]; nb.Turns > 0 {
		fmt.Fprintf(&b, "- **Non-billed `<synthetic>` turns (harness-injected, $0):** %s (%s output tok)\n", fmtInt(nb.Turns), fmtInt(nb.Output))
	}
	fmt.Fprintln(&b)
	renderModelMix(&b, agg)
	renderBuckets(&b, agg)
	renderModels(&b, agg)
	renderNamespaces(&b, agg)
	renderOpusHeavySessions(&b, ok)
	renderDistributions(&b, agg.Distributions)
	renderToolMix(&b, agg.ToolMix)
	renderTopSessions(&b, ok)
	return b.String()
}

func DeepMarkdown(s Session) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Trajectory: %s\n", s.Session)
	ioText := "null"
	if s.IORatio != nil {
		ioText = fmt.Sprintf("%.1f", *s.IORatio)
	}
	cacheText := "null"
	if s.CacheHitFrac != nil {
		cacheText = fmt.Sprintf("%.3f", *s.CacheHitFrac)
	}
	fmt.Fprintf(&b, "records=%d turns=%d tool_calls=%d output_tok=%s io=%s cache_hit=%s cost=$%.2f\n",
		s.NRecords, s.AssistantTurns, s.NToolUse, fmtInt(s.Tokens.Output), ioText, cacheText, s.CostUSD)
	fmt.Fprintf(&b, "tools=%v\n", s.Tools)
	fmt.Fprintln(&b, "\n## User asks (the trajectory), in order:")
	for i, p := range s.Prompts {
		txt := strings.Join(strings.Fields(p.Text), " ")
		if len(txt) > 200 {
			txt = txt[:200]
		}
		fmt.Fprintf(&b, "  [%2d] %s  %s\n", i, p.Timestamp, txt)
	}
	return b.String()
}

func ProviderBucket(model string) string {
	if nonBilled(model) {
		return "non-billed (harness)"
	}
	m := strings.ToLower(model)
	for _, b := range []struct {
		name string
		subs []string
	}{
		{"Anthropic (Claude)", []string{"claude", "opus", "sonnet", "haiku", "fable"}},
		{"Google (Gemini)", []string{"gemini", "gemma"}},
		{"OpenAI", []string{"gpt", "o1-", "o3-", "o4-", "davinci"}},
		{"local / self-hosted", []string{"qwen", "llama", "mistral", "mixtral", "phi-", "deepseek"}},
	} {
		for _, sub := range b.subs {
			if strings.Contains(m, sub) {
				return b.name
			}
		}
	}
	return "UNKNOWN (unpriced bucket)"
}

func PriceFor(model string) (Rates, bool) {
	if nonBilled(model) {
		return Rates{}, false
	}
	m := strings.ToLower(model)
	for _, key := range pricingOrder {
		if strings.Contains(m, key) {
			return Pricing[key], true
		}
	}
	return Rates{}, false
}

func CostUSD(model string, input, cacheWrite, cacheRead, output int64) float64 {
	r, ok := PriceFor(model)
	if !ok {
		return 0
	}
	return (float64(input)*r.Input + float64(cacheWrite)*r.CacheWrite + float64(cacheRead)*r.CacheRead + float64(output)*r.Output) / 1e6
}

func ModelCost(model string, c ModelCounts) float64 {
	return CostUSD(model, c.Input, c.CacheCreate, c.CacheRead, c.Output)
}

func ModelTier(model string) string {
	if nonBilled(model) {
		return "<synthetic>"
	}
	m := strings.ToLower(model)
	for _, key := range pricingOrder {
		if strings.Contains(m, key) {
			return key
		}
	}
	return "unpriced"
}

type transcriptRecord struct {
	Type                 string        `json:"type"`
	Timestamp            string        `json:"timestamp"`
	IsMeta               bool          `json:"isMeta"`
	InterruptedMessageID string        `json:"interruptedMessageId"`
	Message              transcriptMsg `json:"message"`
}

type transcriptMsg struct {
	ID         *string         `json:"id"`
	Model      string          `json:"model"`
	Usage      transcriptUsage `json:"usage"`
	Content    json.RawMessage `json:"content"`
	StopReason string          `json:"stop_reason"`
}

type transcriptUsage struct {
	InputTokens              int64             `json:"input_tokens"`
	OutputTokens             int64             `json:"output_tokens"`
	CacheReadInputTokens     int64             `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int64             `json:"cache_creation_input_tokens"`
	ServerToolUse            serverToolUse     `json:"server_tool_use"`
	Iterations               []json.RawMessage `json:"iterations"`
}

type serverToolUse struct {
	WebSearchRequests int64 `json:"web_search_requests"`
	WebFetchRequests  int64 `json:"web_fetch_requests"`
}

type contentBlock struct {
	Type    string          `json:"type"`
	Name    string          `json:"name"`
	Input   json.RawMessage `json:"input"`
	Content json.RawMessage `json:"content"`
	Text    json.RawMessage `json:"text"`
}

func analyzeAssistant(s *Session, r transcriptRecord, seen map[string]bool) {
	msg := r.Message
	if msg.ID != nil {
		if seen[*msg.ID] {
			s.DupAssistantLines++
			return
		}
		seen[*msg.ID] = true
	}
	s.AssistantTurns++
	model := msg.Model
	if model == "" {
		model = "?"
	}
	s.Models[model]++
	u := msg.Usage
	s.Tokens.Input += u.InputTokens
	s.Tokens.Output += u.OutputTokens
	s.Tokens.CacheRead += u.CacheReadInputTokens
	s.Tokens.CacheCreate += u.CacheCreationInputTokens
	s.Tokens.WebSearch += u.ServerToolUse.WebSearchRequests
	s.Tokens.WebFetch += u.ServerToolUse.WebFetchRequests
	s.Tokens.Iterations += int64(len(u.Iterations))
	s.CostUSD += CostUSD(model, u.InputTokens, u.CacheCreationInputTokens, u.CacheReadInputTokens, u.OutputTokens)
	pm := s.PerModel[model]
	pm.Turns++
	pm.Input += u.InputTokens
	pm.Output += u.OutputTokens
	pm.CacheRead += u.CacheReadInputTokens
	pm.CacheCreate += u.CacheCreationInputTokens
	s.PerModel[model] = pm
	var blocks []contentBlock
	if len(msg.Content) > 0 {
		_ = json.Unmarshal(msg.Content, &blocks)
	}
	for _, b := range blocks {
		switch b.Type {
		case "tool_use":
			s.NToolUse++
			name := b.Name
			if name == "" {
				name = "?"
			}
			s.Tools[name]++
			s.ToolInputChars += txtLen(b.Input)
		case "thinking":
			s.NThinking++
		case "text":
			s.NText++
		}
	}
	if r.InterruptedMessageID != "" || msg.StopReason == "interrupted" {
		s.Interrupted++
	}
}

func analyzeUser(s *Session, r transcriptRecord) {
	if len(r.Message.Content) == 0 {
		return
	}
	var blocks []contentBlock
	if err := json.Unmarshal(r.Message.Content, &blocks); err == nil {
		for _, b := range blocks {
			if b.Type == "tool_result" {
				s.NToolResult++
				s.ToolResultChars += txtLen(b.Content)
			}
		}
		return
	}
	var content string
	if err := json.Unmarshal(r.Message.Content, &content); err == nil {
		if looksLikeTypedPrompt(content) && !r.IsMeta {
			txt := strings.TrimSpace(content)
			if len(txt) > 400 {
				txt = txt[:400]
			}
			s.Prompts = append(s.Prompts, Prompt{Timestamp: r.Timestamp, Text: txt})
		}
	}
}

func finalizeSession(s *Session) {
	s.NPrompts = int64(len(s.Prompts))
	totalIn := s.Tokens.Input + s.Tokens.CacheRead + s.Tokens.CacheCreate
	s.TotalInputTokens = totalIn
	if s.Tokens.Output > 0 {
		v := float64(totalIn) / float64(s.Tokens.Output)
		s.IORatio = &v
	}
	if totalIn > 0 {
		v := float64(s.Tokens.CacheRead) / float64(totalIn)
		s.CacheHitFrac = &v
	}
	for name, n := range s.Tools {
		if ReadOnlyTools[name] {
			s.ReadOnlyToolCalls += n
		}
	}
	if s.NToolUse > 0 {
		v := float64(s.ReadOnlyToolCalls) / float64(s.NToolUse)
		s.ReadOnlyFrac = &v
	}
	if s.TSMin != "" && s.TSMax != "" {
		a, ea := parseTimestamp(s.TSMin)
		b, eb := parseTimestamp(s.TSMax)
		if ea == nil && eb == nil {
			v := b.Sub(a).Seconds()
			s.WallSeconds = &v
		}
	}
}

func txtLen(raw json.RawMessage) int64 {
	if len(raw) == 0 {
		return 0
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return int64(len(s))
	}
	var arr []json.RawMessage
	if err := json.Unmarshal(raw, &arr); err == nil {
		var n int64
		for _, item := range arr {
			n += txtLen(item)
		}
		return n
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err == nil {
		if c, ok := obj["content"]; ok {
			return txtLen(c)
		}
		if t, ok := obj["text"]; ok {
			return txtLen(t)
		}
	}
	return 0
}

func looksLikeTypedPrompt(s string) bool {
	st := strings.TrimSpace(s)
	return st != "" && !strings.HasPrefix(st, "<system-reminder>") && !strings.HasPrefix(st, "Caveat:")
}

func excludedNamespace(ns string) bool {
	for _, sub := range ExcludeNamespaceSubstrings {
		if strings.Contains(ns, sub) {
			return true
		}
	}
	return false
}

func statTranscript(root, ns, path, kind string, cutoff time.Time) (Transcript, bool) {
	st, err := os.Stat(path)
	if err != nil || (!cutoff.IsZero() && st.ModTime().Before(cutoff)) {
		return Transcript{}, false
	}
	return Transcript{
		Root:  root,
		NS:    ns,
		Path:  path,
		Kind:  kind,
		Size:  st.Size(),
		MTime: float64(st.ModTime().UnixNano()) / 1e9,
	}, true
}

func nonBilled(model string) bool {
	return model == "" || model == "?" || model == "<synthetic>"
}

func parseTimestamp(s string) (time.Time, error) {
	if t, err := time.Parse(time.RFC3339Nano, s); err == nil {
		return t, nil
	}
	return time.Parse("2006-01-02T15:04:05.000Z07:00", strings.Replace(s, "Z", "+00:00", 1))
}

func addTokens(dst *TokenCounts, src TokenCounts) {
	dst.Input += src.Input
	dst.Output += src.Output
	dst.CacheRead += src.CacheRead
	dst.CacheCreate += src.CacheCreate
	dst.WebSearch += src.WebSearch
	dst.WebFetch += src.WebFetch
	dst.Iterations += src.Iterations
}

func addMap(dst, src map[string]int64) {
	for k, v := range src {
		dst[k] += v
	}
}

func addModelCounts(a, b ModelCounts) ModelCounts {
	a.Turns += b.Turns
	a.Input += b.Input
	a.Output += b.Output
	a.CacheRead += b.CacheRead
	a.CacheCreate += b.CacheCreate
	return a
}

func stat(xs []float64, includeMean bool, roundTenth bool, includeMax bool) StatSet {
	var st StatSet
	if len(xs) == 0 {
		return st
	}
	sort.Float64s(xs)
	med := median(xs)
	if roundTenth {
		med = round(med, 1)
	}
	st.Median = &med
	if includeMean {
		var sum float64
		for _, x := range xs {
			sum += x
		}
		mean := round(sum/float64(len(xs)), 1)
		st.Mean = &mean
	}
	p90 := pct(xs, 90)
	if roundTenth {
		p90 = round(p90, 1)
	}
	st.P90 = &p90
	if !includeMax && len(xs) > 0 {
		p10 := pct(xs, 10)
		if roundTenth {
			p10 = round(p10, 3)
		}
		st.P10 = &p10
	}
	if includeMax {
		max := xs[len(xs)-1]
		st.Max = &max
	}
	return st
}

func median(xs []float64) float64 {
	if len(xs)%2 == 1 {
		return xs[len(xs)/2]
	}
	return (xs[len(xs)/2-1] + xs[len(xs)/2]) / 2
}

func pct(xs []float64, p float64) float64 {
	k := int(math.Round((p / 100) * float64(len(xs)-1)))
	if k < 0 {
		k = 0
	}
	if k >= len(xs) {
		k = len(xs) - 1
	}
	return xs[k]
}

func round(v float64, places int) float64 {
	p := math.Pow10(places)
	return math.Round(v*p) / p
}

func validSessions(sessions []Session) []Session {
	out := make([]Session, 0, len(sessions))
	for _, s := range sessions {
		if s.Error == "" {
			out = append(out, s)
		}
	}
	return out
}

func scopeLine(sessions []Session, nsPrefix string, sinceDays *float64, includeSubagents bool, maxSessions int) string {
	set := map[string]bool{}
	for _, s := range sessions {
		set[namespaceName(s.Path)] = true
	}
	names := make([]string, 0, len(set))
	for ns := range set {
		names = append(names, ns)
	}
	sort.Strings(names)
	nsDesc := "none"
	if len(names) > 8 {
		nsDesc = strings.Join(names[:8], ", ") + fmt.Sprintf(", ... (+%d more)", len(names)-8)
	} else if len(names) > 0 {
		nsDesc = strings.Join(names, ", ")
	}
	nsFilter := nsPrefix
	if nsFilter == "" {
		nsFilter = "all non-excluded namespaces"
	}
	window := "all-time"
	if sinceDays != nil {
		window = "last " + trimFloat(*sinceDays) + " days"
	}
	kinds := "top-level session transcripts"
	if includeSubagents {
		kinds += " (subagents reported separately below)"
	}
	cap := ""
	if maxSessions > 0 {
		cap = fmt.Sprintf("; max transcripts before analysis: %d", maxSessions)
	}
	return fmt.Sprintf("%d namespaces folded (%s); namespace filter: %s; time window: %s; %s%s", len(names), nsDesc, nsFilter, window, kinds, cap)
}

func maxClipNote(maxSessions, discoveredCount int) string {
	if maxSessions <= 0 || discoveredCount <= maxSessions {
		return ""
	}
	return fmt.Sprintf("NOTE: `--max %d` clipped this audit to the newest %d of %d discovered transcripts; use `--ns-prefix <namespace>` or raise `--max` before treating missing namespaces or model usage as absent.",
		maxSessions, maxSessions, discoveredCount)
}

func namespaceName(path string) string {
	return filepath.Base(filepath.Dir(path))
}

func subagentNote(summary *Summary) string {
	if summary == nil || summary.Count == 0 {
		return ""
	}
	return fmt.Sprintf("NOTE: +%d subagent transcripts uncounted; re-run with `--include-subagents` (about +$%s / +%s output tok).",
		summary.Count, fmtFloat(summary.CostUSD, 2), fmtInt(summary.Tokens.Output))
}

func otherBuckets(buckets map[string]ModelCounts) []string {
	var out []string
	for bucket, c := range buckets {
		if bucket != "Anthropic (Claude)" && bucket != "non-billed (harness)" && c.Output > 0 {
			out = append(out, bucket)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if buckets[out[i]].Output == buckets[out[j]].Output {
			return out[i] < out[j]
		}
		return buckets[out[i]].Output > buckets[out[j]].Output
	})
	return out
}

func renderModelMix(b *strings.Builder, agg Aggregate) {
	fmt.Fprintln(b, "## Model-mix KPI (tier shares)")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Tier | Output tok | Output share | Est. cost | Cost share |")
	fmt.Fprintln(b, "|---|---:|---:|---:|---:|")
	totalOutput := int64(0)
	totalCost := 0.0
	for _, c := range agg.PerTier {
		totalOutput += c.Output
	}
	for model, c := range agg.PerModel {
		totalCost += ModelCost(model, c)
	}
	for _, tier := range sortedModelCounts(agg.PerTier) {
		c := agg.PerTier[tier]
		tierCost := 0.0
		for model, mc := range agg.PerModel {
			if ModelTier(model) == tier {
				tierCost += ModelCost(model, mc)
			}
		}
		fmt.Fprintf(b, "| %s | %s | %s | $%s | %s |\n", tier, fmtInt(c.Output), fmtPct(ratio(c.Output, totalOutput)), fmtFloat(tierCost, 2), fmtPct(floatRatio(tierCost, totalCost)))
	}
	fmt.Fprintln(b)
}

func renderBuckets(b *strings.Builder, agg Aggregate) {
	fmt.Fprintln(b, "## Cost by billing bucket (provider) - never sum across these")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Billing bucket | Turns | Output tok | Cache-read tok | Est. cost | Priced? |")
	fmt.Fprintln(b, "|---|---:|---:|---:|---:|:--:|")
	for _, bucket := range sortedModelCounts(agg.PerBucket) {
		c := agg.PerBucket[bucket]
		bcost := 0.0
		for model, mc := range agg.PerModel {
			if ProviderBucket(model) == bucket {
				bcost += ModelCost(model, mc)
			}
		}
		costCell := "- (no card)"
		priced := ""
		if bucket == "Anthropic (Claude)" {
			costCell = "$" + fmtFloat(bcost, 2)
			priced = "yes"
		} else if bucket == "non-billed (harness)" {
			costCell = "$0.00"
		}
		fmt.Fprintf(b, "| %s | %s | %s | %s | %s | %s |\n", bucket, fmtInt(c.Turns), fmtInt(c.Output), fmtInt(c.CacheRead), costCell, priced)
	}
	fmt.Fprintln(b)
}

func renderModels(b *strings.Builder, agg Aggregate) {
	fmt.Fprintln(b, "## Per-model breakdown (token-exact; cost Anthropic-assumed)")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Model | Bucket | Turns | Output tok | Cache-read tok | Est. cost |")
	fmt.Fprintln(b, "|---|---|---:|---:|---:|---:|")
	for _, model := range sortedModelCounts(agg.PerModel) {
		c := agg.PerModel[model]
		costCell := "- (no card)"
		if _, ok := PriceFor(model); ok {
			costCell = "$" + fmtFloat(ModelCost(model, c), 2)
		} else if nonBilled(model) {
			costCell = "$0.00"
		}
		fmt.Fprintf(b, "| %s | %s | %s | %s | %s | %s |\n", model, ProviderBucket(model), fmtInt(c.Turns), fmtInt(c.Output), fmtInt(c.CacheRead), costCell)
	}
	fmt.Fprintln(b)
}

func renderNamespaces(b *strings.Builder, agg Aggregate) {
	fmt.Fprintln(b, "## Per-namespace rollup")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Namespace | Sessions | Output tok | Opus output share | Cache-read tok | Tool calls | Top model (by output) | Est. cost |")
	fmt.Fprintln(b, "|---|---:|---:|---:|---:|---:|---|---:|")
	keys := make([]string, 0, len(agg.PerNamespace))
	for ns := range agg.PerNamespace {
		keys = append(keys, ns)
	}
	sort.Slice(keys, func(i, j int) bool {
		if agg.PerNamespace[keys[i]].Output == agg.PerNamespace[keys[j]].Output {
			return keys[i] < keys[j]
		}
		return agg.PerNamespace[keys[i]].Output > agg.PerNamespace[keys[j]].Output
	})
	for _, ns := range keys {
		v := agg.PerNamespace[ns]
		fmt.Fprintf(b, "| %s | %d | %s | %s | %s | %s | %s | $%s |\n",
			ns, v.Sessions, fmtInt(v.Output), fmtPctPtr(agg.PerNamespaceOpusShare[ns]), fmtInt(v.CacheRead), fmtInt(v.ToolUse), agg.PerNamespaceTopModel[ns], fmtFloat(agg.PerNamespaceCost[ns], 2))
	}
	fmt.Fprintln(b)
}

type opusSessionRow struct {
	Session   Session
	OpusOut   int64
	OpusCost  float64
	OpusShare *float64
	TopModel  string
	TotalCost float64
	TotalOut  int64
}

func renderOpusHeavySessions(b *strings.Builder, sessions []Session) {
	rows := opusHeavySessionRows(sessions)
	if len(rows) == 0 {
		return
	}
	fmt.Fprintln(b, "## Opus-heavy sessions")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Session | NS | Opus output tok | Opus share | Opus est.$ | Total output tok | Total est.$ | Top model |")
	fmt.Fprintln(b, "|---|---|---:|---:|---:|---:|---:|---|")
	if len(rows) > 10 {
		rows = rows[:10]
	}
	for _, row := range rows {
		sid := row.Session.Session
		if len(sid) > 8 {
			sid = sid[:8]
		}
		fmt.Fprintf(b, "| %s | %s | %s | %s | $%s | %s | $%s | %s |\n",
			sid,
			namespaceName(row.Session.Path),
			fmtInt(row.OpusOut),
			fmtPctPtr(row.OpusShare),
			fmtFloat(row.OpusCost, 2),
			fmtInt(row.TotalOut),
			fmtFloat(row.TotalCost, 2),
			row.TopModel)
	}
	fmt.Fprintln(b)
}

func opusHeavySessionRows(sessions []Session) []opusSessionRow {
	rows := make([]opusSessionRow, 0, len(sessions))
	for _, s := range sessions {
		if s.Error != "" {
			continue
		}
		opusOut, opusCost := sessionTierOutputCost(s, "opus")
		if opusOut == 0 {
			continue
		}
		var share *float64
		if s.Tokens.Output > 0 {
			v := float64(opusOut) / float64(s.Tokens.Output)
			share = &v
		}
		rows = append(rows, opusSessionRow{
			Session:   s,
			OpusOut:   opusOut,
			OpusCost:  opusCost,
			OpusShare: share,
			TopModel:  topSessionModel(s),
			TotalCost: s.CostUSD,
			TotalOut:  s.Tokens.Output,
		})
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].OpusOut == rows[j].OpusOut {
			if rows[i].OpusCost == rows[j].OpusCost {
				return rows[i].Session.Path < rows[j].Session.Path
			}
			return rows[i].OpusCost > rows[j].OpusCost
		}
		return rows[i].OpusOut > rows[j].OpusOut
	})
	return rows
}

func sessionTierOutputCost(s Session, tier string) (int64, float64) {
	var output int64
	var cost float64
	for model, counts := range s.PerModel {
		if ModelTier(model) != tier {
			continue
		}
		output += counts.Output
		cost += ModelCost(model, counts)
	}
	return output, cost
}

func topSessionModel(s Session) string {
	top := ""
	var topOut int64
	for model, counts := range s.PerModel {
		if counts.Output > topOut || (counts.Output == topOut && (top == "" || model < top)) {
			top = model
			topOut = counts.Output
		}
	}
	if top == "" {
		return "?"
	}
	return top
}

func renderDistributions(b *strings.Builder, d Distributions) {
	fmt.Fprintln(b, "## Distributions (per session)")
	fmt.Fprintln(b)
	fmt.Fprintf(b, "- **Tool calls/session:** median %s, mean %s, p90 %s, max %s\n",
		fmtStat(d.CallsPerSession.Median), fmtStat(d.CallsPerSession.Mean), fmtStat(d.CallsPerSession.P90), fmtStat(d.CallsPerSession.Max))
	fmt.Fprintf(b, "- **Output tokens/session:** median %s, p90 %s, max %s\n",
		fmtStatInt(d.OutputTokensPerSession.Median), fmtStatInt(d.OutputTokensPerSession.P90), fmtStatInt(d.OutputTokensPerSession.Max))
	fmt.Fprintf(b, "- **I:O ratio/session:** median %s, p90 %s\n", fmtStat(d.IORatio.Median), fmtStat(d.IORatio.P90))
	fmt.Fprintf(b, "- **Cache-hit fraction/session:** median %s, p10 %s, p90 %s\n",
		fmtStat(d.CacheHitFrac.Median), fmtStat(d.CacheHitFrac.P10), fmtStat(d.CacheHitFrac.P90))
	fmt.Fprintf(b, "- **Read-only tool fraction/session:** median %s\n", fmtStat(d.ReadOnlyFrac.Median))
	fmt.Fprintln(b)
}

func renderToolMix(b *strings.Builder, tools map[string]int64) {
	fmt.Fprintln(b, "## Global tool mix")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Tool | Calls | Read-only? |")
	fmt.Fprintln(b, "|---|---:|:--:|")
	keys := sortedCounts(tools)
	if len(keys) > 25 {
		keys = keys[:25]
	}
	for _, name := range keys {
		mark := ""
		if ReadOnlyTools[name] {
			mark = "yes"
		}
		fmt.Fprintf(b, "| %s | %s | %s |\n", name, fmtInt(tools[name]), mark)
	}
	fmt.Fprintln(b)
}

func renderTopSessions(b *strings.Builder, sessions []Session) {
	fmt.Fprintln(b, "## Top 15 sessions by output tokens")
	fmt.Fprintln(b)
	fmt.Fprintln(b, "| Session | NS | Turns | Tool calls | Output tok | I:O | Cache-hit | Est.$ |")
	fmt.Fprintln(b, "|---|---|---:|---:|---:|---:|---:|---:|")
	sort.Slice(sessions, func(i, j int) bool {
		if sessions[i].Tokens.Output == sessions[j].Tokens.Output {
			return sessions[i].Path < sessions[j].Path
		}
		return sessions[i].Tokens.Output > sessions[j].Tokens.Output
	})
	if len(sessions) > 15 {
		sessions = sessions[:15]
	}
	for _, s := range sessions {
		sid := s.Session
		if len(sid) > 8 {
			sid = sid[:8]
		}
		ioCell := "-"
		if s.IORatio != nil {
			ioCell = fmt.Sprintf("%.0f", *s.IORatio)
		}
		chCell := "-"
		if s.CacheHitFrac != nil {
			chCell = fmt.Sprintf("%.0f%%", *s.CacheHitFrac*100)
		}
		fmt.Fprintf(b, "| %s | %s | %d | %d | %s | %s | %s | $%.2f |\n",
			sid, namespaceName(s.Path), s.AssistantTurns, s.NToolUse, fmtInt(s.Tokens.Output), ioCell, chCell, s.CostUSD)
	}
	fmt.Fprintln(b)
}

func sortedModelCounts(m map[string]ModelCounts) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if m[keys[i]].Output == m[keys[j]].Output {
			return keys[i] < keys[j]
		}
		return m[keys[i]].Output > m[keys[j]].Output
	})
	return keys
}

func sortedCounts(m map[string]int64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if m[keys[i]] == m[keys[j]] {
			return keys[i] < keys[j]
		}
		return m[keys[i]] > m[keys[j]]
	})
	return keys
}

func ratio(n, d int64) *float64 {
	if d == 0 {
		return nil
	}
	v := float64(n) / float64(d)
	return &v
}

func floatRatio(n, d float64) *float64 {
	if d == 0 {
		return nil
	}
	v := n / d
	return &v
}

func fmtPct(v *float64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%.1f%%", *v*100)
}

func fmtPctPtr(v *float64) string {
	return fmtPct(v)
}

func fmtInt(n int64) string {
	s := strconv.FormatInt(n, 10)
	var out []byte
	for i, r := range reverse(s) {
		if i > 0 && i%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, byte(r))
	}
	return reverse(string(out))
}

func reverse(s string) string {
	b := []byte(s)
	for i, j := 0, len(b)-1; i < j; i, j = i+1, j-1 {
		b[i], b[j] = b[j], b[i]
	}
	return string(b)
}

func fmtFloat(v float64, places int) string {
	s := fmt.Sprintf("%."+strconv.Itoa(places)+"f", v)
	parts := strings.SplitN(s, ".", 2)
	intPart := parts[0]
	sign := ""
	if strings.HasPrefix(intPart, "-") {
		sign = "-"
		intPart = strings.TrimPrefix(intPart, "-")
	}
	grouped := fmtIntString(intPart)
	if len(parts) == 2 {
		return sign + grouped + "." + parts[1]
	}
	return sign + grouped
}

func fmtIntString(s string) string {
	var out []byte
	for i, r := range reverse(s) {
		if i > 0 && i%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, byte(r))
	}
	return reverse(string(out))
}

func fmtStat(v *float64) string {
	if v == nil {
		return "null"
	}
	if math.Abs(*v-math.Round(*v)) < 1e-9 {
		return fmt.Sprintf("%.0f", *v)
	}
	return strconv.FormatFloat(*v, 'f', -1, 64)
}

func fmtStatInt(v *float64) string {
	if v == nil {
		return "0"
	}
	return fmtInt(int64(math.Round(*v)))
}

func trimFloat(v float64) string {
	return strconv.FormatFloat(v, 'f', -1, 64)
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func WriteJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}
