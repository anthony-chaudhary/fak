package sessionmine

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

const IndexHealthSchema = "fak-session-history-status/1"

type SourceHealth struct {
	Provider      string `json:"provider"`
	State         string `json:"state"`
	Files         int    `json:"files"`
	Sessions      int    `json:"sessions"`
	MalformedRows int    `json:"malformed_rows"`
	AcceptedFiles int    `json:"accepted_files,omitempty"`
	RejectedFiles int    `json:"rejected_files,omitempty"`
}

type RefreshHealth struct {
	State       string               `json:"state"`
	CompletedAt string               `json:"completed_at,omitempty"`
	Outcome     string               `json:"outcome,omitempty"`
	ParsedFiles int                  `json:"parsed_files,omitempty"`
	ReusedFiles int                  `json:"reused_files,omitempty"`
	Outcomes    RefreshOutcomeCounts `json:"outcomes"`
}

type ContentionHealth struct {
	State     string `json:"state"`
	OwnerPID  int    `json:"owner_pid,omitempty"`
	StartedAt string `json:"started_at,omitempty"`
}

type IndexHealth struct {
	Schema              string           `json:"schema"`
	Verdict             string           `json:"verdict"`
	Reason              string           `json:"reason"`
	IndexExists         bool             `json:"index_exists"`
	IndexBytes          int64            `json:"index_bytes,omitempty"`
	IndexSchema         string           `json:"index_schema,omitempty"`
	Watermark           string           `json:"watermark,omitempty"`
	WatermarkAgeSeconds int64            `json:"watermark_age_seconds,omitempty"`
	Providers           []SourceHealth   `json:"providers"`
	LastRefresh         RefreshHealth    `json:"last_refresh"`
	Contention          ContentionHealth `json:"contention"`
	NextActions         []string         `json:"next_actions,omitempty"`
}

type IndexHealthOptions struct {
	IndexPath  string
	CodexRoot  string
	ClaudeRoot string
	Now        time.Time
}

func InspectIndexHealth(path string, now time.Time) IndexHealth {
	return InspectIndexHealthWithOptions(IndexHealthOptions{IndexPath: path, Now: now})
}

func InspectIndexHealthWithOptions(opts IndexHealthOptions) IndexHealth {
	out := IndexHealth{Schema: IndexHealthSchema, Verdict: "RED", Reason: "index_missing", Providers: []SourceHealth{{Provider: "claude", State: "not_checked"}, {Provider: "codex", State: "not_checked"}}, LastRefresh: inspectRefreshReceipt(opts.IndexPath), Contention: inspectContention(opts.IndexPath), NextActions: []string{"fak vcache session-history refresh --once --index <index>"}}
	info, err := os.Stat(opts.IndexPath)
	if errors.Is(err, os.ErrNotExist) {
		return out
	}
	if err != nil {
		out.Reason = "index_unreadable"
		return out
	}
	out.IndexExists, out.IndexBytes = true, info.Size()
	state, err := LoadIndex(opts.IndexPath)
	if err != nil {
		out.Reason = "index_invalid"
		out.NextActions = []string{"replace or refresh the incompatible index"}
		return out
	}
	out.IndexSchema, out.Watermark = state.Schema, state.UpdatedAt
	if opts.CodexRoot != "" || opts.ClaudeRoot != "" {
		out.Providers = []SourceHealth{inspectSource("claude", opts.ClaudeRoot, state), inspectSource("codex", opts.CodexRoot, state)}
	} else {
		out.Providers = indexedProviders(state)
	}
	if out.Contention.State == "live" {
		out.Verdict, out.Reason = "WARN", "refresh_in_progress"
		out.NextActions = []string{"wait for the active refresh, then rerun status"}
		return out
	}
	missing, rejected := 0, 0
	for _, p := range out.Providers {
		if p.State == "missing" || p.State == "empty" {
			missing++
		}
		rejected += p.RejectedFiles
	}
	if state.UpdatedAt == "" || state.UpdatedAt == "all" {
		out.Verdict, out.Reason = "WARN", "empty_index"
	} else if ts, e := time.Parse(time.RFC3339, state.UpdatedAt); e == nil {
		out.WatermarkAgeSeconds = max64(0, int64(opts.Now.Sub(ts).Seconds()))
		if out.WatermarkAgeSeconds > 48*3600 {
			out.Verdict, out.Reason = "WARN", "stale_watermark"
		} else if missing > 0 {
			out.Verdict, out.Reason = "WARN", "partial_provider_coverage"
		} else if rejected > 0 {
			out.Verdict, out.Reason = "WARN", "rejected_source_files"
		} else {
			out.Verdict, out.Reason, out.NextActions = "GREEN", "healthy", nil
		}
	} else {
		out.Verdict, out.Reason = "WARN", "watermark_unparseable"
	}
	return out
}

func indexedProviders(state IndexState) []SourceHealth {
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
	out := make([]SourceHealth, 0, len(counts))
	for _, p := range counts {
		out = append(out, *p)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Provider < out[j].Provider })
	return out
}

func inspectSource(provider, root string, state IndexState) SourceHealth {
	out := SourceHealth{Provider: provider, State: "missing"}
	if root == "" {
		out.State = "not_configured"
		return out
	}
	info, err := os.Stat(root)
	if err != nil || !info.IsDir() {
		return out
	}
	out.State = "empty"
	_ = filepath.WalkDir(root, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil || d.IsDir() || filepath.Ext(path) != ".jsonl" {
			return nil
		}
		out.Files++
		if f, ok := state.Files[sourceFingerprint(provider, path)]; ok {
			out.AcceptedFiles++
			out.Sessions++
			out.MalformedRows += f.Malformed
		} else {
			out.RejectedFiles++
		}
		return nil
	})
	if out.Files > 0 {
		if out.RejectedFiles > 0 {
			out.State = "partial"
		} else {
			out.State = "covered"
		}
	}
	return out
}

func inspectRefreshReceipt(index string) RefreshHealth {
	out := RefreshHealth{State: "missing"}
	b, err := os.ReadFile(refreshReceiptPath(index))
	if err != nil {
		return out
	}
	var r RefreshReceipt
	if json.Unmarshal(b, &r) != nil {
		out.State = "invalid"
		return out
	}
	out.State = "recorded"
	out.CompletedAt = r.CompletedAt
	out.Outcome = r.Outcome
	out.ParsedFiles = r.ParsedFiles
	out.ReusedFiles = r.ReusedFiles
	out.Outcomes = r.Outcomes
	return out
}
func inspectContention(index string) ContentionHealth {
	out := ContentionHealth{State: "idle"}
	b, err := os.ReadFile(refreshLockPath(index))
	if err != nil {
		return out
	}
	var l refreshLock
	if json.Unmarshal(b, &l) != nil {
		out.State = "unknown"
		return out
	}
	out.OwnerPID = l.PID
	out.StartedAt = l.StartedAt
	if processalive.Check(l.PID) {
		out.State = "live"
	} else {
		out.State = "stale"
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
	return fmt.Sprintf("session history — %s (%s)\nindex: exists=%t bytes=%d schema=%s watermark=%s age=%ds\nproviders: %s\nrefresh: %s %s contention=%s pid=%d\nnext: %s\n", s.Verdict, s.Reason, s.IndexExists, s.IndexBytes, s.IndexSchema, s.Watermark, s.WatermarkAgeSeconds, renderProviders(s.Providers), s.LastRefresh.State, s.LastRefresh.Outcome, s.Contention.State, s.Contention.OwnerPID, firstAction(s.NextActions))
}
func renderProviders(ps []SourceHealth) string {
	out := ""
	for i, p := range ps {
		if i > 0 {
			out += ", "
		}
		out += fmt.Sprintf("%s=%s/%d accepted=%d rejected=%d", p.Provider, p.State, p.Files, p.AcceptedFiles, p.RejectedFiles)
	}
	return out
}
func firstAction(a []string) string {
	if len(a) == 0 {
		return "none"
	}
	return a[0]
}
