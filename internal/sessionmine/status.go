package sessionmine

import (
	"errors"
	"fmt"
	"os"
	"sort"
	"time"
)

const IndexHealthSchema = "fak-session-history-status/1"

type SourceHealth struct {
	Provider      string `json:"provider"`
	Files         int    `json:"files"`
	Sessions      int    `json:"sessions"`
	MalformedRows int    `json:"malformed_rows"`
	State         string `json:"state"`
}
type IndexHealth struct {
	Schema              string         `json:"schema"`
	Verdict             string         `json:"verdict"`
	Reason              string         `json:"reason"`
	IndexExists         bool           `json:"index_exists"`
	IndexBytes          int64          `json:"index_bytes,omitempty"`
	IndexSchema         string         `json:"index_schema,omitempty"`
	Watermark           string         `json:"watermark,omitempty"`
	WatermarkAgeSeconds int64          `json:"watermark_age_seconds,omitempty"`
	Providers           []SourceHealth `json:"providers"`
	NextActions         []string       `json:"next_actions,omitempty"`
}

func InspectIndexHealth(path string, now time.Time) IndexHealth {
	out := IndexHealth{Schema: IndexHealthSchema, Verdict: "RED", Reason: "index_missing", Providers: []SourceHealth{{Provider: "claude", State: "missing"}, {Provider: "codex", State: "missing"}}, NextActions: []string{"fak vcache session-history refresh --once --index <index>"}}
	info, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		return out
	}
	if err != nil {
		out.Reason = "index_unreadable"
		return out
	}
	out.IndexExists = true
	out.IndexBytes = info.Size()
	state, err := LoadIndex(path)
	if err != nil {
		out.Reason = "index_invalid"
		out.NextActions = []string{"replace or refresh the incompatible index"}
		return out
	}
	out.IndexSchema = state.Schema
	out.Watermark = state.UpdatedAt
	counts := map[string]*SourceHealth{"claude": {Provider: "claude", State: "missing"}, "codex": {Provider: "codex", State: "missing"}}
	for _, f := range state.Files {
		p := counts[f.Provider]
		if p == nil {
			p = &SourceHealth{Provider: f.Provider}
			counts[f.Provider] = p
		}
		p.Files++
		p.Sessions++
		p.MalformedRows += f.Malformed
		p.State = "covered"
	}
	out.Providers = out.Providers[:0]
	for _, p := range counts {
		out.Providers = append(out.Providers, *p)
	}
	sort.Slice(out.Providers, func(i, j int) bool { return out.Providers[i].Provider < out.Providers[j].Provider })
	missing := 0
	for _, p := range out.Providers {
		if p.Files == 0 {
			missing++
		}
	}
	if state.UpdatedAt == "" || state.UpdatedAt == "all" {
		out.Verdict = "WARN"
		out.Reason = "empty_index"
	} else if ts, e := time.Parse(time.RFC3339, state.UpdatedAt); e == nil {
		out.WatermarkAgeSeconds = max64(0, int64(now.Sub(ts).Seconds()))
		if out.WatermarkAgeSeconds > 48*3600 {
			out.Verdict = "WARN"
			out.Reason = "stale_watermark"
		} else if missing > 0 {
			out.Verdict = "WARN"
			out.Reason = "partial_provider_coverage"
		} else {
			out.Verdict = "GREEN"
			out.Reason = "healthy"
			out.NextActions = nil
		}
	} else {
		out.Verdict = "WARN"
		out.Reason = "watermark_unparseable"
	}
	return out
}
func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
func RenderIndexHealth(s IndexHealth) string {
	return fmt.Sprintf("session history — %s (%s)\nindex: exists=%t bytes=%d schema=%s watermark=%s age=%ds\nproviders: %s\nnext: %s\n", s.Verdict, s.Reason, s.IndexExists, s.IndexBytes, s.IndexSchema, s.Watermark, s.WatermarkAgeSeconds, renderProviders(s.Providers), firstAction(s.NextActions))
}
func renderProviders(ps []SourceHealth) string {
	out := ""
	for i, p := range ps {
		if i > 0 {
			out += ", "
		}
		out += fmt.Sprintf("%s=%s/%d", p.Provider, p.State, p.Files)
	}
	return out
}
func firstAction(a []string) string {
	if len(a) == 0 {
		return "none"
	}
	return a[0]
}
