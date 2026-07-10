package toolrollup

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
)

// ToolRollupSchema versions the rollup report (and the JSONL [ToolCall] corpus row
// it folds) so a downstream reader can tell which fold discipline produced a table —
// the same schema-stamp convention internal/usagelog and internal/toolshape follow.
const ToolRollupSchema = "fak.toolrollup.v1"

// ToolCall is one tool-call record in the corpus: a single call of a single tool,
// with its token cost, its wall duration, and whether it succeeded. The JSON tags
// are the stable corpus schema. `tool` mirrors internal/trajectory's Turn field
// name; the remaining tags follow the wider repo convention (input_tokens /
// output_tokens, duration_ms, ok) so a row reads alongside the other JSONL ledgers.
//
// OK is fail-closed: a row without an explicit `ok` decodes to false and is counted
// as an errored call, so a corpus can never silently inflate its success rate by
// omitting the field.
type ToolCall struct {
	Tool       string `json:"tool"`          // the tool TYPE name — the rollup key
	TokensIn   int    `json:"input_tokens"`  // input/prompt tokens billed to the call
	TokensOut  int    `json:"output_tokens"` // output/completion tokens billed to the call
	DurationMS int64  `json:"duration_ms"`   // wall-clock duration of the call, milliseconds
	OK         bool   `json:"ok"`            // true on success; false marks an errored call
}

// ToolStat is the aggregate for one distinct tool TYPE across the whole corpus.
// Counts are ints; totals are int64 (a large corpus sums past a 32-bit range);
// means and rates are float64. Share is the tool's fraction of all calls in the
// corpus, in [0,1].
type ToolStat struct {
	Tool           string  `json:"tool"`
	Calls          int     `json:"calls"`
	TotalTokensIn  int64   `json:"total_input_tokens"`
	MeanTokensIn   float64 `json:"mean_input_tokens"`
	TotalTokensOut int64   `json:"total_output_tokens"`
	MeanTokensOut  float64 `json:"mean_output_tokens"`
	TotalDuration  int64   `json:"total_duration_ms"`
	MeanDuration   float64 `json:"mean_duration_ms"`
	Errors         int     `json:"errors"`
	ErrorRate      float64 `json:"error_rate"`
	Share          float64 `json:"share"`
}

// Rollup folds a corpus of tool-call records into one ToolStat per distinct tool
// TYPE. The result is deterministic: sorted by call count descending, then by tool
// name ascending, so the same corpus always renders the same report. An empty or nil
// input returns an empty (non-nil) slice.
func Rollup(records []ToolCall) []ToolStat {
	// Accumulate per tool. A pointer keeps the running sums cheap to update.
	type acc struct {
		calls               int
		tokensIn, tokensOut int64
		duration            int64
		errors              int
	}
	byTool := map[string]*acc{}
	for _, r := range records {
		a := byTool[r.Tool]
		if a == nil {
			a = &acc{}
			byTool[r.Tool] = a
		}
		a.calls++
		a.tokensIn += int64(r.TokensIn)
		a.tokensOut += int64(r.TokensOut)
		a.duration += r.DurationMS
		if !r.OK {
			a.errors++
		}
	}

	total := len(records)
	out := make([]ToolStat, 0, len(byTool))
	for tool, a := range byTool {
		calls := float64(a.calls) // calls >= 1 for any tool present in the map
		st := ToolStat{
			Tool:           tool,
			Calls:          a.calls,
			TotalTokensIn:  a.tokensIn,
			MeanTokensIn:   float64(a.tokensIn) / calls,
			TotalTokensOut: a.tokensOut,
			MeanTokensOut:  float64(a.tokensOut) / calls,
			TotalDuration:  a.duration,
			MeanDuration:   float64(a.duration) / calls,
			Errors:         a.errors,
			ErrorRate:      float64(a.errors) / calls,
		}
		if total > 0 {
			st.Share = float64(a.calls) / float64(total)
		}
		out = append(out, st)
	}

	// Deterministic order: call count desc, then tool name asc. The tool name is
	// unique per group, so this is a total order — the ranking is stable without
	// depending on the map's iteration order.
	sort.Slice(out, func(i, j int) bool {
		if out[i].Calls != out[j].Calls {
			return out[i].Calls > out[j].Calls
		}
		return out[i].Tool < out[j].Tool
	})
	return out
}

// ReadCorpus reads a JSONL corpus — one JSON ToolCall per line — into a slice, the
// same one-record-per-line shape internal/trajectory's ExportTo writes and the
// trajquery `--corpus` reader consumes. Whitespace between records (blank lines) is
// tolerated. A malformed record aborts the read with a line-numbered error rather
// than being silently dropped, so a torn corpus is reported, not misfolded. An empty
// reader is not an error: it yields an empty slice that Rollup folds to no stats.
func ReadCorpus(r io.Reader) ([]ToolCall, error) {
	dec := json.NewDecoder(r)
	out := []ToolCall{}
	for i := 1; dec.More(); i++ {
		var tc ToolCall
		if err := dec.Decode(&tc); err != nil {
			return nil, fmt.Errorf("toolrollup: decode corpus record %d: %w", i, err)
		}
		out = append(out, tc)
	}
	return out, nil
}

// Render writes a stable, human-readable table of the per-tool rollup [Rollup]
// produced — one header row then one row per tool in the fold's deterministic
// most-used-first (tie by name) order, so the same corpus always prints
// byte-identically. Columns: tool, calls, share%, error%, and the mean input/output
// token cost and mean wall duration per call. It answers, at a glance, "what tools
// ran, how often, how expensively, how reliably?" — the toolrollup analogue of the
// per-verb usage table. An empty slice prints just the header (honest-empty, no
// panic). Rates are shown as percentages; token/duration means are rounded to whole
// units for a compact column.
func Render(w io.Writer, stats []ToolStat) {
	fmt.Fprintf(w, "%-20s %8s %8s %8s %10s %10s %10s\n",
		"TOOL", "CALLS", "SHARE%", "ERR%", "MEAN-IN", "MEAN-OUT", "MEAN-MS")
	for _, s := range stats {
		fmt.Fprintf(w, "%-20s %8d %7.1f%% %7.1f%% %10.0f %10.0f %10.0f\n",
			s.Tool, s.Calls, s.Share*100, s.ErrorRate*100,
			s.MeanTokensIn, s.MeanTokensOut, s.MeanDuration)
	}
}
