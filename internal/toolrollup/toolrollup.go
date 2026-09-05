package toolrollup

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
)

// ToolRollupSchema identifies the JSONL corpus and aggregate version.
const ToolRollupSchema = "fak.toolrollup.v1"

// ToolCall records single execution metrics including tokens, duration, and status.
type ToolCall struct {
	Tool       string `json:"tool"`
	TokensIn   int    `json:"input_tokens"`
	TokensOut  int    `json:"output_tokens"`
	DurationMS int64  `json:"duration_ms"`
	OK         bool   `json:"ok"`
}

// ToolStat aggregates execution volume, resource usage, and error metrics for a tool.
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

// Rollup aggregates tool calls into deterministic statistics sorted by frequency.
func Rollup(records []ToolCall) []ToolStat {
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
		calls := float64(a.calls)
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

	sort.Slice(out, func(i, j int) bool {
		if out[i].Calls != out[j].Calls {
			return out[i].Calls > out[j].Calls
		}
		return out[i].Tool < out[j].Tool
	})
	return out
}

// ReadCorpus decodes line-delimited JSON tool call records from an input stream.
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

// Render formats aggregated tool statistics into an aligned text table.
func Render(w io.Writer, stats []ToolStat) {
	fmt.Fprintf(w, "%-20s %8s %8s %8s %10s %10s %10s\n",
		"TOOL", "CALLS", "SHARE%", "ERR%", "MEAN-IN", "MEAN-OUT", "MEAN-MS")
	for _, s := range stats {
		fmt.Fprintf(w, "%-20s %8d %7.1f%% %7.1f%% %10.0f %10.0f %10.0f\n",
			s.Tool, s.Calls, s.Share*100, s.ErrorRate*100,
			s.MeanTokensIn, s.MeanTokensOut, s.MeanDuration)
	}
}
