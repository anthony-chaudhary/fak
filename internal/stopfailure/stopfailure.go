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
	ActionRecentReview           = "RECENT_REVIEW"
	ActionStaleReset             = "STALE_RESET_CANDIDATE"
	ActionStaleMarkerOnlyArchive = "STALE_MARKER_ONLY_ARCHIVE_CANDIDATE"
	ActionHealedNonzero          = "HEALED_NONZERO"
	ActionZeroTotal              = "ZERO_TOTAL"
	DefaultRecentWindowHours     = 6
	DefaultTranscriptNamespace   = "C--work-fak"
)

type Options struct {
	Root                string
	Now                 time.Time
	RecentWindow        time.Duration
	SinceWindow         time.Duration
	Limit               int
	ClaudeHome          string
	TranscriptNamespace string
}

type Marker struct {
	SessionID         string    `json:"session_id"`
	MarkerPath        string    `json:"marker_path"`
	ArchivePath       string    `json:"archive_path,omitempty"`
	Total             int       `json:"total"`
	Consecutive       int       `json:"consecutive"`
	MTime             time.Time `json:"mtime"`
	AgeSeconds        int64     `json:"age_seconds"`
	Origin            string    `json:"origin"`
	OriginLabels      []string  `json:"origin_labels,omitempty"`
	TranscriptProject string    `json:"transcript_project,omitempty"`
	SettlementAction  string    `json:"settlement_action"`
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

type ArchiveResult struct {
	Schema     string   `json:"schema"`
	Applied    bool     `json:"applied"`
	Root       string   `json:"root"`
	Candidates []Marker `json:"candidates"`
	Archived   []Marker `json:"archived"`
	Errors     []string `json:"errors,omitempty"`
}

type ClearReviewedResult struct {
	Schema     string   `json:"schema"`
	Applied    bool     `json:"applied"`
	Root       string   `json:"root"`
	Requested  []string `json:"requested"`
	Candidates []Marker `json:"candidates"`
	Updated    []Marker `json:"updated"`
	Missing    []string `json:"missing,omitempty"`
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
	origins := newOriginResolver(root, opts)
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		marker, err := readMarker(filepath.Join(stopDir, entry.Name()), entry.Name(), opts, origins)
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

func ArchiveMarkerOnly(opts Options, apply bool) (ArchiveResult, error) {
	opts = normalizeOptions(opts)
	allOpts := opts
	allOpts.Limit = 0
	plan, err := BuildPlan(allOpts)
	if err != nil {
		return ArchiveResult{}, err
	}
	candidates := append([]Marker(nil), plan.Candidates[ActionStaleMarkerOnlyArchive]...)
	if opts.Limit > 0 && len(candidates) > opts.Limit {
		candidates = candidates[:opts.Limit]
	}
	for i := range candidates {
		candidates[i].ArchivePath = archiveMarkerPath(candidates[i])
	}
	result := ArchiveResult{
		Schema:     "fak.stopfailure.archive-marker-only.v1",
		Applied:    apply,
		Root:       plan.Root,
		Candidates: candidates,
	}
	if !apply {
		return result, nil
	}
	archiveDir := filepath.Join(plan.Root, ".dos", "stop-failures", "archive")
	if err := os.MkdirAll(archiveDir, 0o755); err != nil {
		return result, err
	}
	for _, marker := range candidates {
		src := filepath.Join(plan.Root, filepath.FromSlash(marker.MarkerPath))
		dst := filepath.Join(plan.Root, filepath.FromSlash(marker.ArchivePath))
		if err := os.Rename(src, dst); err != nil {
			result.Errors = append(result.Errors, fmt.Sprintf("%s: %v", marker.MarkerPath, err))
			continue
		}
		result.Archived = append(result.Archived, marker)
	}
	return result, nil
}

func ClearReviewed(opts Options, sessionIDs []string, apply bool) (ClearReviewedResult, error) {
	opts = normalizeOptions(opts)
	allOpts := opts
	allOpts.Limit = 0
	plan, err := BuildPlan(allOpts)
	if err != nil {
		return ClearReviewedResult{}, err
	}
	requested := compactSessionIDs(sessionIDs)
	result := ClearReviewedResult{
		Schema:    "fak.stopfailure.clear-reviewed.v1",
		Applied:   apply,
		Root:      plan.Root,
		Requested: requested,
	}
	if len(requested) == 0 {
		result.Errors = append(result.Errors, "at least one --session is required")
		return result, nil
	}
	want := map[string]bool{}
	for _, id := range requested {
		want[id] = true
	}
	for _, marker := range plan.Candidates[ActionRecentReview] {
		if want[marker.SessionID] {
			result.Candidates = append(result.Candidates, marker)
			delete(want, marker.SessionID)
		}
	}
	for id := range want {
		result.Missing = append(result.Missing, id)
	}
	sort.Strings(result.Missing)
	if !apply {
		return result, nil
	}
	for _, marker := range result.Candidates {
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
	if opts.TranscriptNamespace == "" {
		opts.TranscriptNamespace = DefaultTranscriptNamespace
	}
	return opts
}

func readMarker(path, name string, opts Options, origins originResolver) (Marker, error) {
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
	origin := origins.lookup(sessionID)
	action := ActionStaleReset
	switch {
	case total <= 0:
		action = ActionZeroTotal
	case consecutive <= 0:
		action = ActionHealedNonzero
	case time.Duration(age)*time.Second <= opts.RecentWindow:
		action = ActionRecentReview
	case origin.Origin == "marker_only":
		action = ActionStaleMarkerOnlyArchive
	}
	return Marker{
		SessionID:         sessionID,
		MarkerPath:        filepath.ToSlash(filepath.Join(".dos", "stop-failures", name)),
		Total:             total,
		Consecutive:       consecutive,
		MTime:             mtime,
		AgeSeconds:        age,
		Origin:            origin.Origin,
		OriginLabels:      origin.Labels,
		TranscriptProject: origin.TranscriptProject,
		SettlementAction:  action,
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

func archiveMarkerPath(marker Marker) string {
	return filepath.ToSlash(filepath.Join(".dos", "stop-failures", "archive", filepath.Base(marker.MarkerPath)))
}

func compactSessionIDs(values []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, value := range values {
		id := strings.TrimSpace(value)
		if id == "" || seen[id] {
			continue
		}
		seen[id] = true
		out = append(out, id)
	}
	return out
}

type markerOrigin struct {
	Origin            string
	Labels            []string
	TranscriptProject string
}

type originResolver struct {
	root        string
	transcripts map[string]markerOrigin
}

func newOriginResolver(root string, opts Options) originResolver {
	return originResolver{
		root:        root,
		transcripts: transcriptIndex(opts),
	}
}

func (r originResolver) lookup(sessionID string) markerOrigin {
	labels := []string{}
	if pathExists(filepath.Join(r.root, ".dos", "streams", sessionID+".jsonl")) {
		labels = append(labels, "dos_stream")
	}
	transcript := r.transcripts[sessionID]
	if transcript.Origin != "" {
		labels = append(labels, "claude_transcript")
	}
	if len(labels) == 0 {
		return markerOrigin{Origin: "marker_only", Labels: []string{"marker_only"}}
	}
	return markerOrigin{
		Origin:            strings.Join(labels, "+"),
		Labels:            labels,
		TranscriptProject: transcript.TranscriptProject,
	}
}

func transcriptIndex(opts Options) map[string]markerOrigin {
	out := map[string]markerOrigin{}
	for _, root := range claudeProjectRoots(opts) {
		entries, err := os.ReadDir(root)
		if err != nil {
			continue
		}
		for _, entry := range entries {
			if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".jsonl") {
				continue
			}
			sessionID := strings.TrimSuffix(entry.Name(), ".jsonl")
			if _, ok := out[sessionID]; ok {
				continue
			}
			out[sessionID] = markerOrigin{
				Origin:            "claude_transcript",
				TranscriptProject: filepath.Base(root),
			}
		}
	}
	return out
}

func claudeProjectRoots(opts Options) []string {
	namespace := opts.TranscriptNamespace
	if namespace == "" {
		namespace = DefaultTranscriptNamespace
	}
	userHome := strings.TrimSpace(opts.ClaudeHome)
	if userHome == "" {
		if configured := strings.TrimSpace(os.Getenv("CLAUDE_CONFIG_DIR")); configured != "" {
			return []string{filepath.Join(configured, "projects", namespace)}
		}
	}
	if userHome == "" {
		userHome = strings.TrimSpace(os.Getenv("FLEET_USER_HOME"))
	}
	if userHome == "" {
		userHome = strings.TrimSpace(os.Getenv("USERPROFILE"))
	}
	if userHome == "" {
		userHome, _ = os.UserHomeDir()
	}
	if userHome == "" {
		return nil
	}
	matches, err := filepath.Glob(filepath.Join(userHome, ".claude*", "projects", namespace))
	if err != nil {
		return nil
	}
	sort.Strings(matches)
	return matches
}

func pathExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
