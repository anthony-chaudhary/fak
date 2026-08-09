package agent

import (
	"fmt"
	"sort"
)

// WidthObservation is one assistant turn folded after the fact. Suppressed turns are
// excluded from batching denominators because the client prohibited parallel calls.
type WidthObservation struct {
	Lane      string `json:"lane"`
	Engine    string `json:"engine"`
	Model     string `json:"model"`
	ToolCalls int    `json:"tool_calls"`
	// ToolItems is the independently processed item count across ToolCalls. Zero is
	// the backward-compatible legacy form and means one item per tool call.
	ToolItems  int  `json:"tool_items,omitempty"`
	Suppressed bool `json:"client_suppressed,omitempty"`
	Success    bool `json:"success"`
}

type WidthSeries struct {
	Lane                 string  `json:"lane"`
	Engine               string  `json:"engine"`
	Model                string  `json:"model"`
	AssistantTurns       int     `json:"assistant_turns"`
	EligibleToolTurns    int     `json:"eligible_tool_turns"`
	SuppressedToolTurns  int     `json:"suppressed_tool_turns"`
	ToolCalls            int     `json:"tool_calls"`
	ToolItems            int     `json:"tool_items"`
	BatchedTurns         int     `json:"batched_turns"`
	SuccessfulTurns      int     `json:"successful_turns"`
	MeanToolCalls        float64 `json:"tool_calls_per_assistant_turn"`
	ItemsPerToolCall     float64 `json:"items_per_tool_call"`
	BatchedTurnRate      float64 `json:"batched_turn_rate"`
	ToolTurnShare        float64 `json:"tool_turn_share"`
	ClientSuppressedRate float64 `json:"client_suppressed_rate"`
	OutcomeRate          float64 `json:"outcome_rate"`
}

type WidthReport struct {
	Schema string        `json:"schema"`
	Series []WidthSeries `json:"series"`
}

func FoldWidth(observations []WidthObservation) WidthReport {
	rows := map[string]*WidthSeries{}
	for _, o := range observations {
		key := o.Lane + "\x00" + o.Engine + "\x00" + o.Model
		r := rows[key]
		if r == nil {
			r = &WidthSeries{Lane: o.Lane, Engine: o.Engine, Model: o.Model}
			rows[key] = r
		}
		r.AssistantTurns++
		if o.Success {
			r.SuccessfulTurns++
		}
		if o.ToolCalls > 0 {
			if o.Suppressed {
				r.SuppressedToolTurns++
			} else {
				r.EligibleToolTurns++
				r.ToolCalls += o.ToolCalls
				items := o.ToolItems
				if items < o.ToolCalls {
					items = o.ToolCalls
				}
				r.ToolItems += items
				if o.ToolCalls >= BatchedTurnMinCalls {
					r.BatchedTurns++
				}
			}
		}
	}
	out := WidthReport{Schema: "fak.tool-width-monitor/1"}
	for _, r := range rows {
		if r.AssistantTurns > 0 {
			r.ToolTurnShare = float64(r.EligibleToolTurns+r.SuppressedToolTurns) / float64(r.AssistantTurns)
			r.OutcomeRate = float64(r.SuccessfulTurns) / float64(r.AssistantTurns)
		}
		if r.EligibleToolTurns > 0 {
			r.MeanToolCalls = float64(r.ToolCalls) / float64(r.EligibleToolTurns)
			r.BatchedTurnRate = float64(r.BatchedTurns) / float64(r.EligibleToolTurns)
		}
		if r.ToolCalls > 0 {
			r.ItemsPerToolCall = float64(r.ToolItems) / float64(r.ToolCalls)
		}
		if total := r.EligibleToolTurns + r.SuppressedToolTurns; total > 0 {
			r.ClientSuppressedRate = float64(r.SuppressedToolTurns) / float64(total)
		}
		out.Series = append(out.Series, *r)
	}
	sort.Slice(out.Series, func(i, j int) bool {
		return fmt.Sprint(out.Series[i].Lane, out.Series[i].Engine, out.Series[i].Model) < fmt.Sprint(out.Series[j].Lane, out.Series[j].Engine, out.Series[j].Model)
	})
	return out
}

type WidthRegression struct {
	Regressed bool    `json:"regressed"`
	Baseline  float64 `json:"baseline"`
	Current   float64 `json:"current"`
	Delta     float64 `json:"delta"`
}

// DetectWidthRegression is a ratchet, not a target: only a downward step from a
// lane's own baseline alarms. Low absolute width alone never does.
func DetectWidthRegression(baseline, current float64, minDrop float64) WidthRegression {
	delta := current - baseline
	return WidthRegression{Regressed: baseline > 0 && delta <= -minDrop, Baseline: baseline, Current: current, Delta: delta}
}
