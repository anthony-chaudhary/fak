// Package stopfailure reads and settles DOS StopFailure breaker markers.
package stopfailure

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const (
	ActionRecentReview       = "RECENT_REVIEW"
	ActionStaleReset         = "STALE_RESET_CANDIDATE"
	ActionHealedNonzero      = "HEALED_NONZERO"
	ActionZeroTotal          = "ZERO_TOTAL"
	DefaultRecentWindowHours = 6
)

type Options struct {
	Root         string
	Now          time.Time
	RecentWindow time.Duration
	SinceWindow  time.Duration
	Limit        int
}

type Marker struct {
	SessionID        string    `json:"session_id"`
	MarkerPath       string    `json:"marker_path"`
	Total            int       `json:"total"`
	Consecutive      int       `json:"consecutive"`
	MTime            time.Time `json:"mtime"`
	AgeSeconds       int64     `json:"age_seconds"`
	SettlementAction string    `json:"settlement_action"`
}

type Plan struct {
	Schema       string              `json:"schema"`
	Root         string              `json:"root"`
	StopDir      string              `json:"stop_dir"`
	RecentHours  int                 `json:"recent_hours"`
	SinceHours   int                 `json:"since_hours,omitempty"`
	Markers      int                 `json:"markers"`
	IgnoredOld   int                 `json:"ignored_old_markers,omitempty"`
	Counts       map[string]int      `json:"counts"`
	Candidates   map[string][]Marker `json:"candidates"`
	Malformed    int                 `json:"malformed_markers,omitempty"`
	AppliedLimit int                 `json:"applied_limit,omitempty"`
}

type ResetResult struct {
	Schema     string   `json:"schema"`
	Applied    bool     `json:"applied"`
	Root       string   `json:"root"`
	Candidates []Marker `json:"candidates"`
	Updated    []Marker `json:"updated"`
	Errors     []string `json:"errors,omitempty"`
}

func BuildPlan(opts Options) (Plan, error) {
	opts = normalizeOptions(opts)
	root, err := filepath.Abs(opts.Root)
	if err != nil {
		return Plan{}, err
	}
	stopDir := filepath.Join(root, ".dos", "stop-failures")
	plan := Plan{
		Schema:      "fak.stopfailure.plan.v1",
		Root:        root,
		StopDir:     relStopDir(),
		RecentHours: int(opts.RecentWindow / time.Hour),
		SinceHours:  int(opts.SinceWindow / time.Hour),
		Counts:      map[string]int{},
		Candidates:  map[string][]Marker{},
	}
	entries, err := os.ReadDir(stopDir)
	if err != nil {
		if os.IsNotExist(err) {
			return plan, nil
		}
		return Plan{}, err
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		marker, err := readMarker(filepath.Join(stopDir, entry.Name()), entry.Name(), opts)
		if err != nil {
			plan.Malformed++
			continue
		}
		if opts.SinceWindow > 0 && time.Duration(marker.AgeSeconds)*time.Second > opts.SinceWindow {
			plan.IgnoredOld++
			continue
		}
		plan.Markers++
		plan.Counts[marker.SettlementAction]++
		plan.Candidates[marker.SettlementAction] = append(plan.Candidates[marker.SettlementAction], marker)
	}
	for action, rows := range plan.Candidates {
		sortMarkers(rows)
		if opts.Limit > 0 && len(rows) > opts.Limit {
			rows = rows[:opts.Limit]
			plan.AppliedLimit = opts.Limit
		}
		plan.Candidates[action] = rows
	}
	return plan, nil
}

func ResetStale(opts Options, apply bool) (ResetResult, error) {
	opts = normalizeOptions(opts)
	allOpts := opts
	allOpts.Limit = 0
	plan, err := BuildPlan(allOpts)
	if err != nil {
		return ResetResult{}, err
	}
	candidates := append([]Marker(nil), plan.Candidates[ActionStaleReset]...)
	if opts.Limit > 0 && len(candidates) > opts.Limit {
		candidates = candidates[:opts.Limit]
	}
	result := ResetResult{
		Schema:     "fak.stopfailure.reset-stale.v1",
		Applied:    apply,
		Root:       plan.Root,
		Candidates: candidates,
	}
	if !apply {
		return result, nil
	}
	for _, marker := range candidates {
		if err := writeResetMarker(plan.Root, marker); err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("%s: %v", marker.MarkerPath, err))
			continue
		}
		marker.Consecutive = 0
		marker.SettlementAction = ActionHealedNonzero
		result.Updated = append(result.Updated, marker)
	}
	return result, nil
}

func normalizeOptions(opts Options) Options {
	if opts.Root == "" {
		opts.Root = "."
	}
	if opts.Now.IsZero() {
		opts.Now = time.Now().UTC()
	}
	if opts.RecentWindow <= 0 {
		opts.RecentWindow = DefaultRecentWindowHours * time.Hour
	}
	return opts
}

func readMarker(path, name string, opts Options) (Marker, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Marker{}, err
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		return Marker{}, err
	}
	info, err := os.Stat(path)
	if err != nil {
		return Marker{}, err
	}
	sessionID := strings.TrimSuffix(name, ".json")
	mtime := info.ModTime().UTC()
	age := int64(opts.Now.Sub(mtime).Seconds())
	if age < 0 {
		age = 0
	}
	total := intValue(doc["total"])
	consecutive := intValue(doc["consecutive"])
	action := ActionStaleReset
	switch {
	case total <= 0:
		action = ActionZeroTotal
	case consecutive <= 0:
		action = ActionHealedNonzero
	case time.Duration(age)*time.Second <= opts.RecentWindow:
		action = ActionRecentReview
	}
	return Marker{
		SessionID:        sessionID,
		MarkerPath:       filepath.ToSlash(filepath.Join(".dos", "stop-failures", name)),
		Total:            total,
		Consecutive:      consecutive,
		MTime:            mtime,
		AgeSeconds:       age,
		SettlementAction: action,
	}, nil
}

func writeResetMarker(root string, marker Marker) error {
	path := filepath.Join(root, filepath.FromSlash(marker.MarkerPath))
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		return err
	}
	doc["consecutive"] = 0
	buf, err := json.Marshal(doc)
	if err != nil {
		return err
	}
	buf = append(buf, '\n')
	return os.WriteFile(path, buf, 0o644)
}

func sortMarkers(rows []Marker) {
	sort.Slice(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		if a.Consecutive != b.Consecutive {
			return a.Consecutive > b.Consecutive
		}
		if a.Total != b.Total {
			return a.Total > b.Total
		}
		if a.AgeSeconds != b.AgeSeconds {
			return a.AgeSeconds < b.AgeSeconds
		}
		return a.SessionID < b.SessionID
	})
}

func intValue(v any) int {
	switch x := v.(type) {
	case int:
		return x
	case int64:
		return int(x)
	case float64:
		return int(x)
	case json.Number:
		n, _ := x.Int64()
		return int(n)
	default:
		return 0
	}
}

func relStopDir() string {
	return filepath.ToSlash(filepath.Join(".dos", "stop-failures"))
}
