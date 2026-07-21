// Package sessionaudit audits Claude Code session-transcript JSONL files.
package sessionaudit

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Pricing is the per-MTok USD rate card keyed by a lowercase substring of the
// billed model id (see PriceFor / ModelTier). The Claude tiers use Anthropic's
// published rates; the non-Claude providers use each vendor's published PUBLIC
// per-MTok rates (no per-account/private billing) so the cross-provider
// cost-to-close leaderboard (#4488) measures them instead of folding UNMEASURED
// (#4823). Providers that price cache reads via automatic prefix caching carry
// no separate cache-WRITE premium (a cache miss bills the normal input rate), so
// CacheWrite mirrors Input for those rows. Published rates, retrieved 2026-07-16:
//   - deepseek (deepseek-v4-pro): DeepSeek API docs — cache-miss $0.435, cache-hit $0.003625, output $0.87.
//   - glm (glm-5.2): Z.AI (Zhipu) API pricing — input $1.4, cached input $0.26, output $4.4.
//   - kimi (kimi-k2.6): Moonshot official API (platform.kimi.ai) — cache-miss $0.95, cache-hit $0.16, output $4.00.
var Pricing = map[string]Rates{
	"opus":   {Input: 15.0, CacheWrite: 18.75, CacheRead: 1.50, Output: 75.0},
	"sonnet": {Input: 3.0, CacheWrite: 3.75, CacheRead: 0.30, Output: 15.0},
	"haiku":  {Input: 0.80, CacheWrite: 1.00, CacheRead: 0.08, Output: 4.0},
	"fable":  {Input: 3.0, CacheWrite: 3.75, CacheRead: 0.30, Output: 15.0},
	// Non-Claude providers (#4823) — published public per-MTok rates.
	"deepseek": {Input: 0.435, CacheWrite: 0.435, CacheRead: 0.003625, Output: 0.87},
	"glm":      {Input: 1.4, CacheWrite: 1.4, CacheRead: 0.26, Output: 4.4},
	"kimi":     {Input: 0.95, CacheWrite: 0.95, CacheRead: 0.16, Output: 4.00},
}

// pricingOrder is the substring match order for PriceFor / ModelTier. The
// non-Claude keys carry no overlap with the Claude tiers or each other, so
// append order is stable. "kimi" also covers Moonshot ids that embed "kimi".
var pricingOrder = []string{"opus", "sonnet", "haiku", "fable", "deepseek", "glm", "kimi"}

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
	Behavior          Behavior               `json:"behavior"`
	Confusion         Confusion              `json:"confusion"`
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
	// ModelIdentities maps every billed raw model id in PerModel to its typed
	// canonical identity, so the cost artifact carries the RAW and CANONICAL
	// spellings side by side with resolution provenance (#4635). Non-billed
	// harness rows are omitted.
	ModelIdentities map[string]ModelIdentity `json:"model_identities,omitempty"`
	// UnpricedModels is the explicit UNKNOWN cost hold (#4635): billed raw
	// model ids with NO pricing provenance at all (no canonical fleet id, no
	// published card). Their cost is UNKNOWN — excluded from TotalCostUSD and
	// held, never reported as $0.
	UnpricedModels []string `json:"unpriced_models,omitempty"`
	// UnverifiedClaudeIDs are Claude-family spellings that did NOT resolve to
	// a canonical fleet id but WERE neighbor-priced into TotalCostUSD by the
	// legacy tier-substring heuristic. Their dollars rest on a heuristic, not
	// identity provenance — the #4635 overmatch hazard, surfaced explicitly so
	// a gate can hold on it.
	UnverifiedClaudeIDs []string      `json:"unverified_claude_ids,omitempty"`
	Distributions       Distributions `json:"dist"`
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

type CompactReport struct {
	Schema            string                  `json:"schema"`
	Generated         string                  `json:"generated"`
	Scope             CompactScope            `json:"scope"`
	Totals            CompactTotals           `json:"totals"`
	Tiers             []CompactTier           `json:"tiers"`
	TopLongContext    []CompactLongContext    `json:"top_long_context,omitempty"`
	Recommendations   []CompactRecommendation `json:"recommendations,omitempty"`
	Behavior          *CompactBehavior        `json:"behavior,omitempty"`
	Confusion         *CompactConfusion       `json:"confusion,omitempty"`
	ExcludedSubagents *Summary                `json:"excluded_subagents,omitempty"`
}

type CompactScope struct {
	NamespaceFilter  string   `json:"namespace_filter"`
	Namespaces       []string `json:"namespaces"`
	SinceDays        *float64 `json:"since_days,omitempty"`
	IncludeSubagents bool     `json:"include_subagents"`
	Max              int      `json:"max,omitempty"`
	Discovered       int      `json:"discovered"`
	Audited          int64    `json:"audited"`
	Clipped          bool     `json:"clipped"`
	Scoped           bool     `json:"scoped"`
}

type CompactTotals struct {
	OutputTokens       int64   `json:"output_tokens"`
	FreshInputTokens   int64   `json:"fresh_input_tokens"`
	CacheReadTokens    int64   `json:"cache_read_tokens"`
	CacheCreateTokens  int64   `json:"cache_create_tokens"`
	TotalContextTokens int64   `json:"total_context_tokens"`
	CacheReadShare     float64 `json:"cache_read_share"`
	IORatio            float64 `json:"io_ratio"`
	EstimatedCostUSD   float64 `json:"estimated_cost_usd"`
	// UnpricedModels / UnverifiedClaudeIDs mirror the aggregate's explicit
	// UNKNOWN cost hold (#4635) into the machine-readable totals, so a consumer
	// reading EstimatedCostUSD sees IN THE SAME RECORD that unpriced models are
	// HELD (excluded, cost unknown) and that any unverified Claude spellings
	// were neighbor-priced by heuristic — never that unknown meant free.
	UnpricedModels      []string `json:"unpriced_models,omitempty"`
	UnverifiedClaudeIDs []string `json:"unverified_claude_ids,omitempty"`
}

type CompactTier struct {
	Tier             string  `json:"tier"`
	OutputTokens     int64   `json:"output_tokens"`
	OutputShare      float64 `json:"output_share"`
	EstimatedCostUSD float64 `json:"estimated_cost_usd"`
	CostShare        float64 `json:"cost_share"`
	CacheReadTokens  int64   `json:"cache_read_tokens"`
	Turns            int64   `json:"turns"`
}

type CompactLongContext struct {
	Session            string  `json:"session"`
	Namespace          string  `json:"namespace"`
	TotalContextTokens int64   `json:"total_context_tokens"`
	FreshInputTokens   int64   `json:"fresh_input_tokens"`
	CacheReadTokens    int64   `json:"cache_read_tokens"`
	CacheReadShare     float64 `json:"cache_read_share"`
	OutputTokens       int64   `json:"output_tokens"`
	IORatio            float64 `json:"io_ratio"`
	TopModel           string  `json:"top_model"`
}

type CompactRecommendation struct {
	Kind     string `json:"kind"`
	Severity string `json:"severity"`
	Action   string `json:"action"`
	Reason   string `json:"reason"`
	Evidence string `json:"evidence"`
}

// CompactBehavior folds the per-session stuck/churn Behavior lens (behavior.go)
// across the audited window. The per-session Behavior is computed for every
// transcript but was otherwise stranded: the compact recommendations read only the
// cost/context tiers, so a PROCESS issue that recurs across sessions — the same
// stuck failure-loop showing up session after session, a window dominated by shell
// timeout-kills, or read-discipline churn — was detected and then dropped before it
// could become a gate-able action. This aggregate is the cross-session signature
// JOIN the per-transcript audit never did, so those process issues can reach the
// actions gate alongside cost and long-context pressure.
type CompactBehavior struct {
	Sessions            int64                 `json:"sessions"`
	StuckSessions       int64                 `json:"stuck_sessions"`
	TimeoutKills        int64                 `json:"timeout_kills"`
	SleepPolls          int64                 `json:"sleep_polls"`
	WastedMutationCalls int64                 `json:"wasted_mutation_calls"`
	RecurringFailures   []RecurringFailureRow `json:"recurring_failures,omitempty"`
}

// RecurringFailureRow is one failure CLASS (tool + normalized error signature) that
// tripped the within-session repeat-loop threshold in more than one session. It is
// keyed on the classes the behavior lens already deems significant (FailureMass rows
// are each >= the within-session threshold), so a row here means the SAME stuck loop
// recurred across distinct sessions — a systemic process issue, not a one-off.
type RecurringFailureRow struct {
	Tool           string   `json:"tool"`
	Sig            string   `json:"sig"`
	Sessions       int64    `json:"sessions"`
	Occurrences    int64    `json:"occurrences"`
	Namespaces     []string `json:"namespaces"`
	ExampleSession string   `json:"example_session"`
}

// CompactConfusion folds the per-session prose Confusion lens (confusion.go) across the
// audited window — the semantic counterpart to CompactBehavior. The Behavior fold joins
// stuck TOOL loops; this one joins stuck REASONING: which self-correction / dead-end /
// confusion markers recur across sessions, how many sessions crossed the confused
// threshold, and the most-confused sessions to deep-audit first. It returns nil for a
// window with no markers at all, so a clean window omits the field.
type CompactConfusion struct {
	Sessions         int64 `json:"sessions"`
	ConfusedSessions int64 `json:"confused_sessions"`
	// SilentConfusedSessions counts confused sessions the tool-I/O Behavior lens was
	// BLIND to (behaviorSessionStuck == false): prose thrash while every tool call
	// succeeded. This is the slice that justifies the lens — friction process_issue
	// pressure cannot see — and the primary gate for the confusion_pressure recommendation.
	SilentConfusedSessions int64                `json:"silent_confused_sessions"`
	TotalMarkers           int64                `json:"total_markers"`
	SelfCorrectionTurns    int64                `json:"self_correction_turns"`
	DeadEndTurns           int64                `json:"dead_end_turns"`
	ConfusionTurns         int64                `json:"confusion_turns"`
	RecurringMarkers       []RecurringMarkerRow `json:"recurring_markers,omitempty"`
	TopSessions            []ConfusedSessionRow `json:"top_sessions,omitempty"`
}

// RecurringMarkerRow is one confusion marker CLASS (category + label) that appeared in
// more than one session — a systemic reasoning-friction pattern, not a one-off.
type RecurringMarkerRow struct {
	Category   string   `json:"category"`
	Label      string   `json:"label"`
	Sessions   int64    `json:"sessions"`
	Count      int64    `json:"count"`
	Namespaces []string `json:"namespaces"`
	Example    string   `json:"example"`
}

// ConfusedSessionRow is one session that crossed the confused threshold, ranked so an
// operator deep-audits the worst first.
type ConfusedSessionRow struct {
	Session      string  `json:"session"`
	Namespace    string  `json:"namespace"`
	Markers      int64   `json:"markers"`
	DeadEndTurns int64   `json:"dead_end_turns"`
	Score        float64 `json:"score"`
	// Silent is true when the Behavior lens found no tool-loop signature for this
	// session — the confusion is Behavior-invisible, so this is the row an operator
	// should deep-audit first (ranked ahead of Behavior-corroborated confused rows).
	Silent bool `json:"silent"`
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
			if opts.NamespacePrefix != "" {
				if !strings.HasPrefix(ns, opts.NamespacePrefix) {
					continue
				}
			} else if excludedNamespace(ns) {
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
	lens := newBehaviorLens()
	clens := newConfusionLens()
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
		lens.observeRecord(r)
		switch r.Type {
		case "assistant":
			analyzeAssistant(&s, r, seen, lens, clens)
		case "user":
			analyzeUser(&s, r, lens)
		}
	}
	if err := sc.Err(); err != nil {
		s.Error = err.Error()
		return s
	}
	finalizeSession(&s)
	s.Behavior = lens.summary()
	s.Confusion = clens.summary()
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
		if nonBilled(model) {
			continue
		}
		// Typed identity + explicit UNKNOWN hold (#4635): raw and canonical ids
		// both land in the artifact, and a model with no pricing provenance is
		// HELD by name instead of dissolving into a silent $0 (or, for a
		// Claude-family spelling, a neighboring tier's price).
		if agg.ModelIdentities == nil {
			agg.ModelIdentities = map[string]ModelIdentity{}
		}
		mi := ResolveModelID(model)
		agg.ModelIdentities[model] = mi
		if _, err := StrictModelCostUSD(model, 0, 0, 0, 0); err != nil {
			if _, priced := PriceFor(model); priced {
				// Legacy substring pricing DID charge this id (necessarily a
				// Claude-family spelling — every non-Claude card is admitted by
				// the strict path), so its dollars are heuristic, not identity.
				agg.UnverifiedClaudeIDs = append(agg.UnverifiedClaudeIDs, model)
			} else {
				agg.UnpricedModels = append(agg.UnpricedModels, model)
			}
		}
	}
	sort.Strings(agg.UnpricedModels)
	sort.Strings(agg.UnverifiedClaudeIDs)
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

func BuildCompactReport(sessions []Session, agg Aggregate, nsPrefix string, sinceDays *float64, includeSubagents bool, maxSessions int, discoveredCount int, excludedSubagents *Summary, generated time.Time) CompactReport {
	ok := validSessions(sessions)
	totalContext := agg.Totals.Input + agg.Totals.CacheRead + agg.Totals.CacheCreate
	names := sessionNamespaces(ok)
	nsFilter := nsPrefix
	if nsFilter == "" {
		nsFilter = "all non-excluded namespaces"
	}
	rep := CompactReport{
		Schema:    "fak.session_audit.summary.v1",
		Generated: generated.Format(time.RFC3339),
		Scope: CompactScope{
			NamespaceFilter:  nsFilter,
			Namespaces:       names,
			SinceDays:        sinceDays,
			IncludeSubagents: includeSubagents,
			Max:              maxSessions,
			Discovered:       discoveredCount,
			Audited:          agg.NSessions,
			Clipped:          maxSessions > 0 && discoveredCount > maxSessions,
			Scoped:           nsPrefix != "",
		},
		Totals: CompactTotals{
			OutputTokens:        agg.Totals.Output,
			FreshInputTokens:    agg.Totals.Input,
			CacheReadTokens:     agg.Totals.CacheRead,
			CacheCreateTokens:   agg.Totals.CacheCreate,
			TotalContextTokens:  totalContext,
			CacheReadShare:      floatRatioValue(float64(agg.Totals.CacheRead), float64(totalContext)),
			IORatio:             floatRatioValue(float64(totalContext), float64(agg.Totals.Output)),
			EstimatedCostUSD:    agg.TotalCostUSD,
			UnpricedModels:      agg.UnpricedModels,
			UnverifiedClaudeIDs: agg.UnverifiedClaudeIDs,
		},
		Tiers:             compactTiers(agg),
		TopLongContext:    compactLongContext(ok, 10),
		ExcludedSubagents: excludedSubagents,
	}
	rep.Behavior = aggregateCompactBehavior(ok)
	rep.Confusion = aggregateCompactConfusion(ok)
	rep.Recommendations = compactRecommendations(rep)
	return rep
}

// BuildCompactReportFromDiscovery discovers transcripts, analyzes the selected
// window, and returns the compact machine-readable report used by CLI and HTTP
// action surfaces. Missing default roots are treated as an empty window by
// Discover; other discovery errors are returned to the caller.
func BuildCompactReportFromDiscovery(opts DiscoverOptions, includeSubagents bool, max int, generated time.Time) (CompactReport, error) {
	recs, err := Discover(opts)
	if err != nil {
		return CompactReport{}, err
	}
	totalDiscovered := len(recs)
	if max > 0 && len(recs) > max {
		recs = recs[:max]
	}
	sessions := make([]Session, 0, len(recs))
	for _, rec := range recs {
		s := Analyze(rec.Path)
		s.Kind = rec.Kind
		sessions = append(sessions, s)
	}
	included := make([]Session, 0, len(sessions))
	for _, s := range sessions {
		if !includeSubagents && s.Kind == "subagent" {
			continue
		}
		included = append(included, s)
	}
	agg := AggregateSessions(included)
	var excluded *Summary
	if !includeSubagents {
		allOpts := opts
		allOpts.IncludeSubagents = true
		allRecs, err := Discover(allOpts)
		if err != nil {
			return CompactReport{}, err
		}
		var subRecs []Transcript
		for _, rec := range allRecs {
			if rec.Kind == "subagent" {
				subRecs = append(subRecs, rec)
			}
		}
		if len(subRecs) > 0 {
			sum := SummarizeTranscripts(subRecs)
			excluded = &sum
		}
	}
	return BuildCompactReport(included, agg, opts.NamespacePrefix, opts.SinceDays, includeSubagents, max, totalDiscovered, excluded, generated), nil
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

// CostUSD is the LEGACY lenient cost path: substring tier matching, and 0 for
// a model with no card. It cannot distinguish "free" from "unknown", so report
// surfaces must pair it with the explicit UNKNOWN hold the aggregate carries
// (Aggregate.UnpricedModels / UnverifiedClaudeIDs); callers that need the
// fail-closed contract use StrictModelCostUSD (#4635).
func CostUSD(model string, input, cacheWrite, cacheRead, output int64) float64 {
	r, ok := PriceFor(model)
	if !ok {
		return 0
	}
	return rawCostUSD(r, input, cacheWrite, cacheRead, output)
}

// rawCostUSD prices the four billable token axes under one rate card — the one
// formula both the legacy (CostUSD) and fail-closed (StrictModelCostUSD) paths
// share, so the two can never diverge on arithmetic, only on admission.
func rawCostUSD(r Rates, input, cacheWrite, cacheRead, output int64) float64 {
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
	IsCompactSummary     bool          `json:"isCompactSummary"`
	LastPrompt           string        `json:"lastPrompt"`
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
	Type      string          `json:"type"`
	Name      string          `json:"name"`
	ID        string          `json:"id"`
	ToolUseID string          `json:"tool_use_id"`
	IsError   bool            `json:"is_error"`
	Input     json.RawMessage `json:"input"`
	Content   json.RawMessage `json:"content"`
	Text      json.RawMessage `json:"text"`
}

func analyzeAssistant(s *Session, r transcriptRecord, seen map[string]bool, lens *behaviorLens, clens *confusionLens) {
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
			lens.noteToolUse(b.ID, name, b.Input, canonicalArgs(b.Input))
		case "thinking":
			s.NThinking++
		case "text":
			s.NText++
			clens.noteText(b.Text)
		}
	}
	if r.InterruptedMessageID != "" || msg.StopReason == "interrupted" {
		s.Interrupted++
	}
}

func analyzeUser(s *Session, r transcriptRecord, lens *behaviorLens) {
	if len(r.Message.Content) == 0 {
		return
	}
	var blocks []contentBlock
	if err := json.Unmarshal(r.Message.Content, &blocks); err == nil {
		for _, b := range blocks {
			if b.Type == "tool_result" {
				s.NToolResult++
				s.ToolResultChars += txtLen(b.Content)
				lens.noteToolResult(b.ToolUseID, b.IsError, txtStr(b.Content, 4000))
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

func validSessions(sessions []Session) []Session {
	out := make([]Session, 0, len(sessions))
	for _, s := range sessions {
		if s.Error == "" {
			out = append(out, s)
		}
	}
	return out
}

func sessionNamespaces(sessions []Session) []string {
	set := map[string]bool{}
	for _, s := range sessions {
		set[namespaceName(s.Path)] = true
	}
	names := make([]string, 0, len(set))
	for ns := range set {
		names = append(names, ns)
	}
	sort.Strings(names)
	return names
}

// ProjectNamespace returns the Claude Code projects/<namespace> key for a
// workspace path. Claude derives it by replacing every non-alphanumeric rune in
// the cleaned path with '-': C:\work\fak becomes C--work-fak.
func ProjectNamespace(workspace string) string {
	clean := filepath.Clean(workspace)
	var b strings.Builder
	b.Grow(len(clean))
	for _, r := range clean {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	return b.String()
}

func namespaceName(path string) string {
	return filepath.Base(filepath.Dir(path))
}

// tierOutputCostTotals sums PerTier output tokens and total PerModel
// estimated cost for agg. Shared by compactTiers and renderModelMix.
func tierOutputCostTotals(agg Aggregate) (totalOutput int64, totalCost float64) {
	for _, c := range agg.PerTier {
		totalOutput += c.Output
	}
	for model, c := range agg.PerModel {
		totalCost += ModelCost(model, c)
	}
	return totalOutput, totalCost
}

// modelCostByKey sums ModelCost across agg.PerModel for every model whose
// classify(model) equals key. Shared by the per-tier (ModelTier) and
// per-bucket (ProviderBucket) cost rollups.
func modelCostByKey(agg Aggregate, key string, classify func(string) string) float64 {
	cost := 0.0
	for model, mc := range agg.PerModel {
		if classify(model) == key {
			cost += ModelCost(model, mc)
		}
	}
	return cost
}

func compactTiers(agg Aggregate) []CompactTier {
	totalOutput, totalCost := tierOutputCostTotals(agg)
	out := make([]CompactTier, 0, len(agg.PerTier))
	for _, tier := range sortedModelCounts(agg.PerTier) {
		c := agg.PerTier[tier]
		tierCost := modelCostByKey(agg, tier, ModelTier)
		out = append(out, CompactTier{
			Tier:             tier,
			OutputTokens:     c.Output,
			OutputShare:      floatRatioValue(float64(c.Output), float64(totalOutput)),
			EstimatedCostUSD: tierCost,
			CostShare:        floatRatioValue(tierCost, totalCost),
			CacheReadTokens:  c.CacheRead,
			Turns:            c.Turns,
		})
	}
	return out
}

func compactLongContext(sessions []Session, limit int) []CompactLongContext {
	rows := longContextSessionRows(sessions)
	if limit > 0 && len(rows) > limit {
		rows = rows[:limit]
	}
	out := make([]CompactLongContext, 0, len(rows))
	for _, row := range rows {
		s := row.Session
		cacheReadShare := 0.0
		if row.CacheReadFrac != nil {
			cacheReadShare = *row.CacheReadFrac
		}
		ioRatio := 0.0
		if s.IORatio != nil {
			ioRatio = *s.IORatio
		}
		out = append(out, CompactLongContext{
			Session:            s.Session,
			Namespace:          namespaceName(s.Path),
			TotalContextTokens: row.TotalContext,
			FreshInputTokens:   s.Tokens.Input,
			CacheReadTokens:    s.Tokens.CacheRead,
			CacheReadShare:     cacheReadShare,
			OutputTokens:       s.Tokens.Output,
			IORatio:            ioRatio,
			TopModel:           topSessionModel(s),
		})
	}
	return out
}

const (
	longContextPressureTokens  = int64(20_000_000)
	longContextPressureIORatio = 200.0
)

func compactRecommendations(rep CompactReport) []CompactRecommendation {
	var out []CompactRecommendation
	if rec, ok := compactOpusCostPressure(rep.Tiers); ok {
		out = append(out, rec)
	}
	if rec, ok := compactLongContextPressure(rep.TopLongContext); ok {
		out = append(out, rec)
	}
	if rec, ok := compactProcessIssuePressure(rep.Behavior); ok {
		out = append(out, rec)
	}
	if rec, ok := compactConfusionPressure(rep.Confusion); ok {
		out = append(out, rec)
	}
	if rec, ok := compactUnpricedHold(rep.Totals); ok {
		out = append(out, rec)
	}
	return out
}

// compactUnpricedHold raises the gate-able UNKNOWN-cost hold (#4635) whenever
// the window billed a model with no pricing provenance (cost UNKNOWN, held —
// never $0) or a Claude-family spelling that was neighbor-priced without a
// canonical identity. High severity: either way EstimatedCostUSD is not yet
// evidence-grade for the window, which is precisely what a cost gate must see.
func compactUnpricedHold(t CompactTotals) (CompactRecommendation, bool) {
	if len(t.UnpricedModels) == 0 && len(t.UnverifiedClaudeIDs) == 0 {
		return CompactRecommendation{}, false
	}
	return compactRecommendation(
		"unpriced_model_hold",
		"high",
		"pin each held model id in the canonical identity table (or its published rate card) before trusting EstimatedCostUSD for this window",
		"billed model ids without pricing provenance are HELD as UNKNOWN cost, never reported as $0 or a neighboring model's price",
		fmt.Sprintf("unpriced=%s unverified_claude=%s",
			strings.Join(t.UnpricedModels, ","), strings.Join(t.UnverifiedClaudeIDs, ",")),
	), true
}

func compactOpusCostPressure(tiers []CompactTier) (CompactRecommendation, bool) {
	opus, hasOpus := compactTierByName(tiers, "opus")
	fable, hasFable := compactTierByName(tiers, "fable")
	if !hasOpus || opus.EstimatedCostUSD <= 0 || opus.CostShare < 0.5 {
		return CompactRecommendation{}, false
	}
	if hasFable && fable.OutputShare >= opus.OutputShare {
		return compactRecommendation(
			"opus_cost_pressure",
			"high",
			"keep Fable as the default route and require an explicit Opus justification for cost-heavy long-context work",
			"Fable produced at least as much output as Opus, but Opus carried most estimated cost",
			fmt.Sprintf("opus_cost_share=%.1f%% fable_output_share=%.1f%% opus_output_share=%.1f%%",
				100*opus.CostShare, 100*fable.OutputShare, 100*opus.OutputShare),
		), true
	}
	return compactRecommendation(
		"opus_cost_pressure",
		"medium",
		"audit the top Opus-heavy sessions before launching more Opus turns",
		"Opus carried most estimated cost in the audited window",
		fmt.Sprintf("opus_cost_share=%.1f%% opus_output_share=%.1f%%", 100*opus.CostShare, 100*opus.OutputShare),
	), true
}

func compactLongContextPressure(rows []CompactLongContext) (CompactRecommendation, bool) {
	if len(rows) == 0 {
		return CompactRecommendation{}, false
	}
	top := rows[0]
	if top.TotalContextTokens < longContextPressureTokens && top.IORatio < longContextPressureIORatio {
		return CompactRecommendation{}, false
	}
	severity := "medium"
	if top.TotalContextTokens >= longContextPressureTokens || top.IORatio >= 2*longContextPressureIORatio {
		severity = "high"
	}
	return CompactRecommendation{
		Kind:     "long_context_pressure",
		Severity: severity,
		Action:   "checkpoint or reset the top long-context session before adding more high-cost turns; use ctxvalue/vcache context witnesses to prove shed-token value",
		Reason:   "the largest recent session is dominated by repeated context ingestion",
		Evidence: fmt.Sprintf("session=%s context_tokens=%d io_ratio=%.1f model=%s",
			top.Session, top.TotalContextTokens, top.IORatio, top.TopModel),
	}, true
}

const (
	// processIssueMinSessions: a stuck failure-loop recurring across this many
	// distinct sessions is systemic, not a one-off (1 is already the per-session signal).
	processIssueMinSessions = 2
	// processIssueMinTimeouts: this many shell timeout-kills across the window points
	// at a process/environment issue (a slow command, a wedged tool) rather than a route choice.
	processIssueMinTimeouts = 6
	// processIssueMinChurn: this much read-discipline churn (edit-before-read / stale-read
	// retries) across the window is worth surfacing as a work-hygiene issue.
	processIssueMinChurn = 12
)

// aggregateCompactBehavior folds the per-session Behavior lens across the audited
// window: window totals (timeout-kills, sleep-polls, wasted read-discipline mutations,
// stuck sessions) plus the cross-session recurring-failure join. It returns nil when
// the window carries no behavioral signal at all, so a clean window omits the field.
func aggregateCompactBehavior(sessions []Session) *CompactBehavior {
	cb := &CompactBehavior{}
	type fkey struct{ tool, sig string }
	type facc struct {
		sessions    map[string]bool
		namespaces  map[string]bool
		occurrences int64
		example     string
	}
	classes := map[fkey]*facc{}
	var order []fkey
	for _, s := range sessions {
		if s.Error != "" {
			continue
		}
		cb.Sessions++
		b := s.Behavior
		cb.TimeoutKills += b.TimeoutKills
		cb.SleepPolls += b.SleepPolls
		for _, v := range b.EditChurn {
			cb.WastedMutationCalls += v
		}
		if behaviorSessionStuck(b) {
			cb.StuckSessions++
		}
		ns := namespaceName(s.Path)
		for _, row := range b.FailureMass {
			k := fkey{row.Tool, row.Sig}
			acc := classes[k]
			if acc == nil {
				acc = &facc{sessions: map[string]bool{}, namespaces: map[string]bool{}, example: s.Session}
				classes[k] = acc
				order = append(order, k)
			}
			acc.sessions[s.Session] = true
			if ns != "" {
				acc.namespaces[ns] = true
			}
			acc.occurrences += row.Count
		}
	}
	for _, k := range order {
		acc := classes[k]
		if len(acc.sessions) < processIssueMinSessions {
			continue
		}
		names := make([]string, 0, len(acc.namespaces))
		for n := range acc.namespaces {
			names = append(names, n)
		}
		sort.Strings(names)
		cb.RecurringFailures = append(cb.RecurringFailures, RecurringFailureRow{
			Tool:           k.tool,
			Sig:            k.sig,
			Sessions:       int64(len(acc.sessions)),
			Occurrences:    acc.occurrences,
			Namespaces:     names,
			ExampleSession: acc.example,
		})
	}
	sort.SliceStable(cb.RecurringFailures, func(i, j int) bool {
		if cb.RecurringFailures[i].Sessions != cb.RecurringFailures[j].Sessions {
			return cb.RecurringFailures[i].Sessions > cb.RecurringFailures[j].Sessions
		}
		return cb.RecurringFailures[i].Occurrences > cb.RecurringFailures[j].Occurrences
	})
	if len(cb.RecurringFailures) > 10 {
		cb.RecurringFailures = cb.RecurringFailures[:10]
	}
	if cb.StuckSessions == 0 && cb.TimeoutKills == 0 && cb.SleepPolls == 0 &&
		cb.WastedMutationCalls == 0 && len(cb.RecurringFailures) == 0 {
		return nil
	}
	return cb
}

func behaviorSessionStuck(b Behavior) bool {
	return len(b.RepeatFailures) > 0 || len(b.FailureMass) > 0 || len(b.FileChurn) > 0 || len(b.SuccessLoops) > 0
}

// compactProcessIssuePressure raises a recommendation when the audited window shows a
// recurring process issue: the same stuck failure-loop across >= processIssueMinSessions
// distinct sessions, a shell timeout-kill storm, or heavy read-discipline churn. This is
// the behavioral counterpart to the cost/context pressure recommendations, and flows
// through the same actions gate.
func compactProcessIssuePressure(beh *CompactBehavior) (CompactRecommendation, bool) {
	if beh == nil {
		return CompactRecommendation{}, false
	}
	var top *RecurringFailureRow
	if len(beh.RecurringFailures) > 0 {
		top = &beh.RecurringFailures[0]
	}
	recurring := top != nil && top.Sessions >= processIssueMinSessions
	timeouts := beh.TimeoutKills >= processIssueMinTimeouts
	churn := beh.WastedMutationCalls >= processIssueMinChurn
	if !recurring && !timeouts && !churn {
		return CompactRecommendation{}, false
	}
	severity := "medium"
	if (top != nil && (top.Sessions >= 3 || len(top.Namespaces) >= 2)) || beh.TimeoutKills >= 2*processIssueMinTimeouts {
		severity = "high"
	}
	var reason, evidence string
	switch {
	case recurring:
		reason = "the same stuck failure-loop recurred across multiple sessions in the audited window"
		evidence = fmt.Sprintf("tool=%s sessions=%d occurrences=%d namespaces=%d timeout_kills=%d churn=%d sig=%q",
			top.Tool, top.Sessions, top.Occurrences, len(top.Namespaces), beh.TimeoutKills, beh.WastedMutationCalls, normHead(top.Sig, 80))
	case timeouts:
		reason = "the audited window is dominated by shell timeout-kills, a process/environment issue rather than a model-route choice"
		evidence = fmt.Sprintf("timeout_kills=%d stuck_sessions=%d churn=%d", beh.TimeoutKills, beh.StuckSessions, beh.WastedMutationCalls)
	default:
		reason = "read-discipline churn (edit-before-read / stale-read retries) is high across the audited window"
		evidence = fmt.Sprintf("churn=%d stuck_sessions=%d timeout_kills=%d", beh.WastedMutationCalls, beh.StuckSessions, beh.TimeoutKills)
	}
	return CompactRecommendation{
		Kind:     "process_issue_pressure",
		Severity: severity,
		Action:   "triage the recurring failure/churn signature and fix its root cause (env, tool, or prompt) before launching more sessions; deep-audit an example session to confirm the loop",
		Reason:   reason,
		Evidence: evidence,
	}, true
}

const (
	// confusedSessionMinMarkers: a session carrying this many prose confusion markers
	// (or the dead-end floor below) is a genuine outlier — the base rate is ~0.5% of
	// text turns, so 3 markers in one session is far above noise.
	confusedSessionMinMarkers = 3
	// confusedSessionMinDeadEnds: a session with this many dead-end turns (a repair that
	// visibly failed / the same failure recurring) is confused even with few other markers.
	confusedSessionMinDeadEnds = 2
	// confusionMinSessions: a confusion pattern recurring across this many distinct
	// sessions is systemic, not a one-off (mirrors processIssueMinSessions).
	confusionMinSessions = 2
	// confusionMinDeadEnds: this many dead-end turns across the window points at a
	// misleading signal the agent kept fighting rather than a single unlucky turn.
	confusionMinDeadEnds = 4
)

// aggregateCompactConfusion folds the per-session Confusion lens across the audited
// window: window turn totals, the count of sessions that crossed the confused threshold,
// the cross-session recurring-marker join, and the worst sessions to deep-audit first.
// It returns nil when the window carries no markers at all, so a clean window omits the
// field.
func aggregateCompactConfusion(sessions []Session) *CompactConfusion {
	cc := &CompactConfusion{}
	type mkey struct{ category, label string }
	type macc struct {
		sessions   map[string]bool
		namespaces map[string]bool
		count      int64
		example    string
	}
	classes := map[mkey]*macc{}
	var order []mkey
	var confused []ConfusedSessionRow
	for _, s := range sessions {
		if s.Error != "" {
			continue
		}
		cc.Sessions++
		conf := s.Confusion
		cc.TotalMarkers += conf.TotalMarkers
		cc.SelfCorrectionTurns += conf.SelfCorrectionTurns
		cc.DeadEndTurns += conf.DeadEndTurns
		cc.ConfusionTurns += conf.ConfusionTurns
		ns := namespaceName(s.Path)
		for _, row := range conf.Markers {
			k := mkey{row.Category, row.Label}
			acc := classes[k]
			if acc == nil {
				acc = &macc{sessions: map[string]bool{}, namespaces: map[string]bool{}, example: row.Example}
				classes[k] = acc
				order = append(order, k)
			}
			acc.sessions[s.Session] = true
			if ns != "" {
				acc.namespaces[ns] = true
			}
			acc.count += row.Count
		}
		if confusedSession(conf) {
			cc.ConfusedSessions++
			// Silent = Behavior found no tool-loop signature; this is the slice the
			// Behavior lens is blind to and the one confusion_pressure gates on.
			silent := !behaviorSessionStuck(s.Behavior)
			if silent {
				cc.SilentConfusedSessions++
			}
			confused = append(confused, ConfusedSessionRow{
				Session:      s.Session,
				Namespace:    ns,
				Markers:      conf.TotalMarkers,
				DeadEndTurns: conf.DeadEndTurns,
				Score:        conf.Score,
				Silent:       silent,
			})
		}
	}
	for _, k := range order {
		acc := classes[k]
		if len(acc.sessions) < confusionMinSessions {
			continue
		}
		names := make([]string, 0, len(acc.namespaces))
		for n := range acc.namespaces {
			names = append(names, n)
		}
		sort.Strings(names)
		cc.RecurringMarkers = append(cc.RecurringMarkers, RecurringMarkerRow{
			Category:   k.category,
			Label:      k.label,
			Sessions:   int64(len(acc.sessions)),
			Count:      acc.count,
			Namespaces: names,
			Example:    acc.example,
		})
	}
	sort.SliceStable(cc.RecurringMarkers, func(i, j int) bool {
		if cc.RecurringMarkers[i].Sessions != cc.RecurringMarkers[j].Sessions {
			return cc.RecurringMarkers[i].Sessions > cc.RecurringMarkers[j].Sessions
		}
		return cc.RecurringMarkers[i].Count > cc.RecurringMarkers[j].Count
	})
	if len(cc.RecurringMarkers) > 10 {
		cc.RecurringMarkers = cc.RecurringMarkers[:10]
	}
	sort.SliceStable(confused, func(i, j int) bool {
		// Behavior-invisible (silent) confused sessions rank first — they are the ones a
		// process-issue audit would miss, so they earn the operator's attention.
		if confused[i].Silent != confused[j].Silent {
			return confused[i].Silent
		}
		if confused[i].Markers != confused[j].Markers {
			return confused[i].Markers > confused[j].Markers
		}
		if confused[i].Score != confused[j].Score {
			return confused[i].Score > confused[j].Score
		}
		return confused[i].Session < confused[j].Session
	})
	if len(confused) > 10 {
		confused = confused[:10]
	}
	cc.TopSessions = confused
	if cc.TotalMarkers == 0 {
		return nil
	}
	return cc
}

// confusedSession reports whether one session crossed the confused threshold: enough
// total markers, or enough dead-end turns on their own.
func confusedSession(c Confusion) bool {
	return c.TotalMarkers >= confusedSessionMinMarkers || c.DeadEndTurns >= confusedSessionMinDeadEnds
}

// compactConfusionPressure raises a recommendation when the audited window shows a
// recurring reasoning-friction pattern the Behavior lens is blind to: the same confusion
// marker across multiple sessions, several Behavior-silent confused sessions, or a
// dead-end loop the agent kept fighting. Every trigger is gated on at least one
// Behavior-silent confused session (see SilentConfusedSessions) so it stays strictly
// complementary to compactProcessIssuePressure — it never restates a tool-loop finding
// Behavior already owns. It flows through the same actions gate as the other pressures.
func compactConfusionPressure(cc *CompactConfusion) (CompactRecommendation, bool) {
	if cc == nil {
		return CompactRecommendation{}, false
	}
	var top *RecurringMarkerRow
	if len(cc.RecurringMarkers) > 0 {
		top = &cc.RecurringMarkers[0]
	}
	// The lens only earns a recommendation where the tool-I/O Behavior lens is BLIND:
	// a session that thrashed in prose while every tool call succeeded. Gate on there
	// being at least one such Behavior-silent confused session, so confusion_pressure
	// never merely restates a process_issue_pressure finding Behavior already owns.
	if cc.SilentConfusedSessions == 0 {
		return CompactRecommendation{}, false
	}
	recurring := top != nil && top.Sessions >= confusionMinSessions
	confusedMany := cc.SilentConfusedSessions >= confusionMinSessions
	deadends := cc.DeadEndTurns >= confusionMinDeadEnds
	if !recurring && !confusedMany && !deadends {
		return CompactRecommendation{}, false
	}
	severity := "medium"
	if cc.SilentConfusedSessions >= 3 || cc.DeadEndTurns >= 2*confusionMinDeadEnds || (top != nil && len(top.Namespaces) >= 2) {
		severity = "high"
	}
	var reason, evidence string
	switch {
	case recurring:
		reason = "the same reasoning-confusion marker recurred across multiple sessions — a systemic pattern, not a one-off"
		evidence = fmt.Sprintf("category=%s label=%s sessions=%d count=%d namespaces=%d silent_confused=%d dead_end_turns=%d example=%q",
			top.Category, top.Label, top.Sessions, top.Count, len(top.Namespaces), cc.SilentConfusedSessions, cc.DeadEndTurns, normHead(top.Example, 80))
	case deadends:
		reason = "the audited window is heavy with dead-end turns — repairs that visibly failed or the same failure recurring, a sign the agent kept fighting a misleading signal"
		evidence = fmt.Sprintf("dead_end_turns=%d silent_confused=%d confused_sessions=%d total_markers=%d", cc.DeadEndTurns, cc.SilentConfusedSessions, cc.ConfusedSessions, cc.TotalMarkers)
	default:
		reason = "multiple Behavior-silent sessions crossed the confusion threshold — prose thrash the tool-I/O lens cannot see"
		evidence = fmt.Sprintf("silent_confused=%d confused_sessions=%d total_markers=%d self_correction_turns=%d dead_end_turns=%d", cc.SilentConfusedSessions, cc.ConfusedSessions, cc.TotalMarkers, cc.SelfCorrectionTurns, cc.DeadEndTurns)
	}
	return CompactRecommendation{
		Kind:     "confusion_pressure",
		Severity: severity,
		Action:   "deep-audit the most-confused session to find where the agent lost the thread, then reduce the friction at its source — clearer task framing / earlier reconnaissance for self-correction churn, or fix the misleading signal (flaky test, stale doc, wrong error) behind a dead-end loop",
		Reason:   reason,
		Evidence: evidence,
	}, true
}

func compactRecommendation(kind, severity, action, reason, evidence string) CompactRecommendation {
	return CompactRecommendation{
		Kind:     kind,
		Severity: severity,
		Action:   action,
		Reason:   reason,
		Evidence: evidence,
	}
}

func compactTierByName(tiers []CompactTier, name string) (CompactTier, bool) {
	for _, tier := range tiers {
		if tier.Tier == name {
			return tier, true
		}
	}
	return CompactTier{}, false
}

func WriteJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(v)
}
