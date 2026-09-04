package wipref

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// IsMutatingTool reports whether tool is a known file-mutating tool (write, edit, etc.).
func IsMutatingTool(tool string) bool {
	t := strings.ToLower(strings.TrimSpace(tool))
	t = strings.TrimPrefix(t, "fak_")
	t = strings.TrimPrefix(t, "tools_")
	switch t {
	case "write", "edit", "write_file", "edit_file", "str_replace_editor",
		"create_file", "overwrite_file", "append_file", "replace",
		"replace_file_content", "patch_file", "save_file":
		return true
	}
	if strings.HasPrefix(t, "write_") || strings.HasPrefix(t, "edit_") {
		return true
	}
	if strings.HasSuffix(t, "_write") || strings.HasSuffix(t, "_edit") {
		return true
	}
	return false
}

// ExtractTargetPaths inspects a mutating tool call's arguments and extracts normalized target paths.
func ExtractTargetPaths(tool string, args any) []string {
	if !IsMutatingTool(tool) {
		return nil
	}
	m := toArgsMap(args)
	if len(m) == 0 {
		return nil
	}
	var paths []string
	// Single path fields
	for _, k := range []string{
		"file_path", "filePath", "path", "filepath", "file",
		"target", "filename", "dest", "destination", "target_path", "targetPath",
	} {
		if v, ok := m[k]; ok {
			if s, ok := v.(string); ok && strings.TrimSpace(s) != "" {
				paths = append(paths, NormalizeScopePath(s))
			}
		}
	}
	// Plural path fields
	for _, k := range []string{
		"file_paths", "filePaths", "paths", "files", "targets",
	} {
		if v, ok := m[k]; ok {
			switch slice := v.(type) {
			case []string:
				for _, s := range slice {
					if strings.TrimSpace(s) != "" {
						paths = append(paths, NormalizeScopePath(s))
					}
				}
			case []any:
				for _, item := range slice {
					if s, ok := item.(string); ok && strings.TrimSpace(s) != "" {
						paths = append(paths, NormalizeScopePath(s))
					}
				}
			}
		}
	}
	return dedupeAndSort(paths)
}

// NormalizeScopePath converts a file path into a repo-relative forward-slash path.
func NormalizeScopePath(path string) string {
	path = strings.TrimSpace(strings.ReplaceAll(path, `\`, "/"))
	for strings.HasPrefix(path, "./") {
		path = path[2:]
	}
	path = strings.TrimPrefix(path, "/")
	return strings.TrimSuffix(path, "/")
}

func dedupeAndSort(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	seen := make(map[string]bool, len(in))
	var out []string
	for _, p := range in {
		p = NormalizeScopePath(p)
		if p == "" || seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

func toArgsMap(args any) map[string]any {
	if args == nil {
		return nil
	}
	switch v := args.(type) {
	case map[string]any:
		return v
	case string:
		var m map[string]any
		if err := json.Unmarshal([]byte(v), &m); err == nil {
			return m
		}
		return nil
	case []byte:
		var m map[string]any
		if err := json.Unmarshal(v, &m); err == nil {
			return m
		}
		return nil
	default:
		b, err := json.Marshal(v)
		if err != nil {
			return nil
		}
		var m map[string]any
		if err := json.Unmarshal(b, &m); err == nil {
			return m
		}
		return nil
	}
}

// CheckpointEmitter is called when a checkpoint with updated Scope is emitted.
type CheckpointEmitter func(sessionID string, stamp Stamp) (string, error)

// ScopeTracker manages active session scopes auto-bound by mutating tool calls.
type ScopeTracker struct {
	mu       sync.Mutex
	scopes   map[string][]string
	stamps   map[string]Stamp
	debounce time.Duration
	emitter  CheckpointEmitter
	timers   map[string]*time.Timer
}

// TrackerOption configures a ScopeTracker.
type TrackerOption func(*ScopeTracker)

// WithDebounce sets the debouncing duration for checkpoint emission on tool completion.
// When zero, tool completion emits micro-checkpoints immediately.
func WithDebounce(d time.Duration) TrackerOption {
	return func(t *ScopeTracker) {
		t.debounce = d
	}
}

// WithEmitter configures the callback invoked when a checkpoint with updated Scope is emitted.
func WithEmitter(emitter CheckpointEmitter) TrackerOption {
	return func(t *ScopeTracker) {
		t.emitter = emitter
	}
}

// NewScopeTracker constructs a new ScopeTracker.
func NewScopeTracker(opts ...TrackerOption) *ScopeTracker {
	t := &ScopeTracker{
		scopes: make(map[string][]string),
		stamps: make(map[string]Stamp),
		timers: make(map[string]*time.Timer),
	}
	for _, opt := range opts {
		if opt != nil {
			opt(t)
		}
	}
	return t
}

// RecordToolCall inspects a tool call and appends target paths to the session's active Scope set.
// Returns the newly bound target paths and whether the tool was recognized as mutating.
func (t *ScopeTracker) RecordToolCall(sessionID string, tool string, args any) ([]string, bool) {
	if sessionID == "" || !IsMutatingTool(tool) {
		return nil, false
	}
	targets := ExtractTargetPaths(tool, args)
	if len(targets) == 0 {
		return nil, false
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	cur := t.scopes[sessionID]
	combined := append(cur, targets...)
	t.scopes[sessionID] = dedupeAndSort(combined)
	return targets, true
}

// RecordToolCompletion handles tool execution completion.
// It auto-binds target paths if not already bound, and triggers debounced or micro-checkpoint emission.
func (t *ScopeTracker) RecordToolCompletion(sessionID string, tool string, args any) (Stamp, error) {
	if sessionID == "" {
		return Stamp{}, fmt.Errorf("session ID required")
	}
	t.RecordToolCall(sessionID, tool, args)

	t.mu.Lock()
	if t.debounce > 0 {
		if timer, ok := t.timers[sessionID]; ok && timer != nil {
			timer.Stop()
		}
		t.timers[sessionID] = time.AfterFunc(t.debounce, func() {
			_, _ = t.EmitCheckpoint(sessionID)
		})
		stamp := t.stamps[sessionID]
		stamp.SessionID = sessionID
		stamp.Scope = append([]string(nil), t.scopes[sessionID]...)
		t.mu.Unlock()
		return stamp, nil
	}
	t.mu.Unlock()

	return t.EmitCheckpoint(sessionID)
}

// ActiveScope returns a copy of the session's currently active scope paths.
func (t *ScopeTracker) ActiveScope(sessionID string) []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]string(nil), t.scopes[sessionID]...)
}

// SetScope manually replaces the active scope for a session.
func (t *ScopeTracker) SetScope(sessionID string, scope []string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.scopes[sessionID] = dedupeAndSort(scope)
}

// EmitCheckpoint immediately emits/persists a checkpoint with the updated Scope in wipref.Stamp.
func (t *ScopeTracker) EmitCheckpoint(sessionID string) (Stamp, error) {
	t.mu.Lock()
	if timer, ok := t.timers[sessionID]; ok && timer != nil {
		timer.Stop()
		delete(t.timers, sessionID)
	}
	scope := append([]string(nil), t.scopes[sessionID]...)
	stamp := t.stamps[sessionID]
	stamp.SessionID = sessionID
	stamp.Scope = scope
	stamp.CheckpointedAt = time.Now().Unix()
	stamp.Buildable = true
	t.stamps[sessionID] = stamp
	emitter := t.emitter
	t.mu.Unlock()

	if emitter != nil {
		ref, err := emitter(sessionID, stamp)
		if err != nil {
			return stamp, fmt.Errorf("emit checkpoint for %s: %w", sessionID, err)
		}
		_ = ref
	}
	return stamp, nil
}

// EmitMicroCheckpoint is an alias for EmitCheckpoint to emit a micro-checkpoint on tool completion.
func (t *ScopeTracker) EmitMicroCheckpoint(sessionID string) (Stamp, error) {
	return t.EmitCheckpoint(sessionID)
}

// Flush cancels any pending debounce timer and immediately emits the checkpoint.
func (t *ScopeTracker) Flush(sessionID string) (Stamp, error) {
	return t.EmitCheckpoint(sessionID)
}

// LatestStamp returns the latest recorded Stamp for the session.
func (t *ScopeTracker) LatestStamp(sessionID string) (Stamp, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	s, ok := t.stamps[sessionID]
	return s, ok
}

// Reset clears state for a session.
func (t *ScopeTracker) Reset(sessionID string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if timer, ok := t.timers[sessionID]; ok && timer != nil {
		timer.Stop()
		delete(t.timers, sessionID)
	}
	delete(t.scopes, sessionID)
	delete(t.stamps, sessionID)
}

// DefaultTracker is the process-wide default scope tracker.
var DefaultTracker = NewScopeTracker()

// RecordToolCall admits/records a tool call on DefaultTracker.
func RecordToolCall(sessionID string, tool string, args any) ([]string, bool) {
	return DefaultTracker.RecordToolCall(sessionID, tool, args)
}

// RecordToolCompletion records completion and triggers checkpoint emission on DefaultTracker.
func RecordToolCompletion(sessionID string, tool string, args any) (Stamp, error) {
	return DefaultTracker.RecordToolCompletion(sessionID, tool, args)
}

// ActiveScope returns the active scope for session on DefaultTracker.
func ActiveScope(sessionID string) []string {
	return DefaultTracker.ActiveScope(sessionID)
}

// EmitCheckpoint immediately emits checkpoint for session on DefaultTracker.
func EmitCheckpoint(sessionID string) (Stamp, error) {
	return DefaultTracker.EmitCheckpoint(sessionID)
}
