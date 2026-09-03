package agentopt

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// Family 12: Context engineering & agent design.
//
// Materialized view generator over append-only agent tool logs:
// Folds state mutations (file writes, edits, deletions, test outcomes, working state variables)
// from an append-only event log into a compact current-state view. Presenting only the
// folded materialized view to the agent drastically reduces prompt context consumption
// while retaining full auditability in the background event log.

// MutationType designates the category of state modification represented by an event.
type MutationType string

const (
	// MutationFileWrite represents creating or overwriting a file.
	MutationFileWrite MutationType = "file_write"
	// MutationFileEdit represents modifying an existing file via targeted replacement or update.
	MutationFileEdit MutationType = "file_edit"
	// MutationFileDelete represents removing a file from the active working set.
	MutationFileDelete MutationType = "file_delete"
	// MutationTestOutcome represents the result of a test suite or case execution.
	MutationTestOutcome MutationType = "test_outcome"
	// MutationVariableSet represents setting or updating a working state variable.
	MutationVariableSet MutationType = "variable_set"
	// MutationVariableDelete represents clearing a working state variable.
	MutationVariableDelete MutationType = "variable_delete"
)

// NormalizeMutationType maps user strings and aliases to recognized MutationTypes.
func NormalizeMutationType(raw string) MutationType {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "file_write", "write", "create_file", "file_create":
		return MutationFileWrite
	case "file_edit", "edit", "update_file", "modify_file":
		return MutationFileEdit
	case "file_delete", "delete", "remove_file", "delete_file", "unlink":
		return MutationFileDelete
	case "test_outcome", "test", "test_run", "test_result":
		return MutationTestOutcome
	case "variable_set", "set_variable", "set_var", "var_set", "set":
		return MutationVariableSet
	case "variable_delete", "delete_variable", "del_var", "unset_var", "unset":
		return MutationVariableDelete
	default:
		return MutationType(raw)
	}
}

// TestStatus indicates the outcome of a test execution.
type TestStatus string

const (
	// TestStatusPassed indicates successful test execution.
	TestStatusPassed TestStatus = "passed"
	// TestStatusFailed indicates a test failure.
	TestStatusFailed TestStatus = "failed"
	// TestStatusSkipped indicates a skipped test.
	TestStatusSkipped TestStatus = "skipped"
)

// NormalizeTestStatus maps common strings and aliases to standard TestStatus values.
func NormalizeTestStatus(raw string) TestStatus {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "passed", "pass", "ok", "success", "true":
		return TestStatusPassed
	case "failed", "fail", "failure", "error", "false":
		return TestStatusFailed
	case "skipped", "skip", "ignored":
		return TestStatusSkipped
	default:
		return TestStatus(raw)
	}
}

// FileState captures the current folded state of a file in the workspace.
type FileState struct {
	Path         string    `json:"path"`
	Content      string    `json:"content"`
	LineCount    int       `json:"line_count"`
	ByteSize     int       `json:"byte_size"`
	Version      int       `json:"version"`
	LastModified time.Time `json:"last_modified"`
}

// TestOutcome records the latest result and combined execution count of a test.
type TestOutcome struct {
	Name         string        `json:"name"`
	Status       TestStatus    `json:"status"`
	Output       string        `json:"output,omitempty"`
	Error        string        `json:"error,omitempty"`
	Duration     time.Duration `json:"duration,omitempty"`
	RunCount     int           `json:"run_count"`
	LastExecuted time.Time     `json:"last_executed"`
}

// Event represents an append-only mutation or observation recorded in the agent tool log.
type Event struct {
	ID        string       `json:"id,omitempty"`
	Timestamp time.Time    `json:"timestamp,omitempty"`
	Type      MutationType `json:"type"`

	// File mutation fields
	Path       string `json:"path,omitempty"`
	Content    string `json:"content,omitempty"`
	OldString  string `json:"old_string,omitempty"`
	NewString  string `json:"new_string,omitempty"`
	ReplaceAll bool   `json:"replace_all,omitempty"`

	// Test outcome fields
	TestName string        `json:"test_name,omitempty"`
	Status   TestStatus    `json:"status,omitempty"`
	Output   string        `json:"output,omitempty"`
	Error    string        `json:"error,omitempty"`
	Duration time.Duration `json:"duration,omitempty"`

	// Working state variable fields
	Key   string `json:"key,omitempty"`
	Value any    `json:"value,omitempty"`

	// Scoped metadata
	Metadata map[string]any `json:"metadata,omitempty"`
}

// NewFileWriteEvent constructs a file write event.
func NewFileWriteEvent(path, content string) Event {
	return Event{
		Type:    MutationFileWrite,
		Path:    path,
		Content: content,
	}
}

// NewFileEditEvent constructs a file edit event via string replacement.
func NewFileEditEvent(path, oldString, newString string, replaceAll bool) Event {
	return Event{
		Type:       MutationFileEdit,
		Path:       path,
		OldString:  oldString,
		NewString:  newString,
		ReplaceAll: replaceAll,
	}
}

// NewFileDeleteEvent constructs a file deletion event.
func NewFileDeleteEvent(path string) Event {
	return Event{
		Type: MutationFileDelete,
		Path: path,
	}
}

// NewTestOutcomeEvent constructs a test outcome event.
func NewTestOutcomeEvent(name string, status TestStatus, output string, duration time.Duration) Event {
	return Event{
		Type:     MutationTestOutcome,
		TestName: name,
		Status:   status,
		Output:   output,
		Duration: duration,
	}
}

// NewVariableSetEvent constructs a working state variable set event.
func NewVariableSetEvent(key string, value any) Event {
	return Event{
		Type:  MutationVariableSet,
		Key:   key,
		Value: value,
	}
}

// NewVariableDeleteEvent constructs a working state variable deletion event.
func NewVariableDeleteEvent(key string) Event {
	return Event{
		Type: MutationVariableDelete,
		Key:  key,
	}
}

func (e *Event) resolveFields() {
	if e.Metadata == nil {
		return
	}
	if e.Path == "" {
		if p, ok := e.Metadata["path"].(string); ok {
			e.Path = p
		} else if p, ok := e.Metadata["file_path"].(string); ok {
			e.Path = p
		}
	}
	if e.Content == "" {
		if c, ok := e.Metadata["content"].(string); ok {
			e.Content = c
		}
	}
	if e.OldString == "" {
		if os, ok := e.Metadata["old_string"].(string); ok {
			e.OldString = os
		}
	}
	if e.NewString == "" {
		if ns, ok := e.Metadata["new_string"].(string); ok {
			e.NewString = ns
		}
	}
	if e.TestName == "" {
		if tn, ok := e.Metadata["test_name"].(string); ok {
			e.TestName = tn
		} else if tn, ok := e.Metadata["name"].(string); ok {
			e.TestName = tn
		}
	}
	if e.Status == "" {
		if st, ok := e.Metadata["status"].(string); ok {
			e.Status = NormalizeTestStatus(st)
		}
	}
	if e.Key == "" {
		if k, ok := e.Metadata["key"].(string); ok {
			e.Key = k
		} else if k, ok := e.Metadata["var_key"].(string); ok {
			e.Key = k
		}
	}
	if e.Value == nil {
		if v, ok := e.Metadata["value"]; ok {
			e.Value = v
		} else if v, ok := e.Metadata["var_value"]; ok {
			e.Value = v
		}
	}
}

// FoldedState holds the compacted, current view of all folded state mutations.
type FoldedState struct {
	Files         map[string]FileState   `json:"files"`
	Deletions     map[string]time.Time   `json:"deletions,omitempty"`
	Tests         map[string]TestOutcome `json:"tests"`
	Variables     map[string]any         `json:"variables"`
	EventCount    int                    `json:"event_count"`
	LastEventID   string                 `json:"last_event_id,omitempty"`
	LastEventTime time.Time              `json:"last_event_time,omitempty"`
}

// Clone creates a deep defensive copy of the materialized state.
func (s FoldedState) Clone() FoldedState {
	c := FoldedState{
		Files:         make(map[string]FileState, len(s.Files)),
		Deletions:     make(map[string]time.Time, len(s.Deletions)),
		Tests:         make(map[string]TestOutcome, len(s.Tests)),
		Variables:     make(map[string]any, len(s.Variables)),
		EventCount:    s.EventCount,
		LastEventID:   s.LastEventID,
		LastEventTime: s.LastEventTime,
	}
	for k, v := range s.Files {
		c.Files[k] = v
	}
	for k, v := range s.Deletions {
		c.Deletions[k] = v
	}
	for k, v := range s.Tests {
		c.Tests[k] = v
	}
	for k, v := range s.Variables {
		c.Variables[k] = v
	}
	return c
}

// File retrieves the state of a file if it exists in the active working set.
func (s FoldedState) File(path string) (FileState, bool) {
	f, ok := s.Files[path]
	return f, ok
}

// ActiveFiles returns a sorted slice of active file paths.
func (s FoldedState) ActiveFiles() []string {
	paths := make([]string, 0, len(s.Files))
	for p := range s.Files {
		paths = append(paths, p)
	}
	sort.Strings(paths)
	return paths
}

// Test retrieves the outcome record for a specific test.
func (s FoldedState) Test(name string) (TestOutcome, bool) {
	t, ok := s.Tests[name]
	return t, ok
}

// Variable retrieves the current value of a working state variable.
func (s FoldedState) Variable(key string) (any, bool) {
	v, ok := s.Variables[key]
	return v, ok
}

// TestSummary returns passed, failed, skipped, and total test counts.
func (s FoldedState) TestSummary() (passed, failed, skipped, total int) {
	total = len(s.Tests)
	for _, t := range s.Tests {
		switch t.Status {
		case TestStatusPassed:
			passed++
		case TestStatusFailed:
			failed++
		case TestStatusSkipped:
			skipped++
		}
	}
	return passed, failed, skipped, total
}

// AllTestsPassed reports true if tests have been run and every test passed.
func (s FoldedState) AllTestsPassed() bool {
	if len(s.Tests) == 0 {
		return false
	}
	for _, t := range s.Tests {
		if t.Status != TestStatusPassed {
			return false
		}
	}
	return true
}

// FoldedView represents the compact agent-facing presentation of the folded state.
type FoldedView struct {
	Summary   string            `json:"summary"`
	Files     []string          `json:"files"`
	Tests     map[string]string `json:"tests"`
	Variables map[string]any    `json:"variables"`
	Text      string            `json:"text"`
}

// String returns the formatted text presentation of the folded view.
func (fv FoldedView) String() string {
	return fv.Text
}

// FoldedView produces the compact agent-facing presentation of the materialized state.
func (s FoldedState) FoldedView() FoldedView {
	activeFiles := s.ActiveFiles()
	testNames := make([]string, 0, len(s.Tests))
	for name := range s.Tests {
		testNames = append(testNames, name)
	}
	sort.Strings(testNames)

	varKeys := make([]string, 0, len(s.Variables))
	for k := range s.Variables {
		varKeys = append(varKeys, k)
	}
	sort.Strings(varKeys)

	passed, failed, skipped, totalTests := s.TestSummary()

	summary := fmt.Sprintf("%d files active, %d variables set, %d tests (%d passed, %d failed, %d skipped)",
		len(activeFiles), len(varKeys), totalTests, passed, failed, skipped)

	testStatuses := make(map[string]string, len(s.Tests))
	for _, name := range testNames {
		testStatuses[name] = string(s.Tests[name].Status)
	}

	var sb strings.Builder
	sb.WriteString("=== Current Working State ===\n")

	// Files section
	if len(activeFiles) > 0 {
		sb.WriteString(fmt.Sprintf("Files (%d active):\n", len(activeFiles)))
		for _, p := range activeFiles {
			fs := s.Files[p]
			sb.WriteString(fmt.Sprintf("  - %s (v%d, %d lines, %d bytes)\n", fs.Path, fs.Version, fs.LineCount, fs.ByteSize))
		}
	} else {
		sb.WriteString("Files: none active\n")
	}

	// Deletions note if any
	if len(s.Deletions) > 0 {
		delPaths := make([]string, 0, len(s.Deletions))
		for p := range s.Deletions {
			delPaths = append(delPaths, p)
		}
		sort.Strings(delPaths)
		sb.WriteString(fmt.Sprintf("Deleted Files (%d):\n", len(delPaths)))
		for _, p := range delPaths {
			sb.WriteString(fmt.Sprintf("  - %s\n", p))
		}
	}

	// Variables section
	if len(varKeys) > 0 {
		sb.WriteString(fmt.Sprintf("Variables (%d):\n", len(varKeys)))
		for _, k := range varKeys {
			sb.WriteString(fmt.Sprintf("  - %s: %v\n", k, s.Variables[k]))
		}
	} else {
		sb.WriteString("Variables: none\n")
	}

	// Tests section
	if totalTests > 0 {
		sb.WriteString(fmt.Sprintf("Tests (%d/%d passed):\n", passed, totalTests))
		for _, name := range testNames {
			t := s.Tests[name]
			durStr := ""
			if t.Duration > 0 {
				durStr = fmt.Sprintf(" (%s, %d runs)", t.Duration.Round(time.Millisecond), t.RunCount)
			} else {
				durStr = fmt.Sprintf(" (%d runs)", t.RunCount)
			}
			sb.WriteString(fmt.Sprintf("  - %s: %s%s\n", t.Name, strings.ToUpper(string(t.Status)), durStr))
		}
	} else {
		sb.WriteString("Tests: none executed\n")
	}

	sb.WriteString(fmt.Sprintf("Summary: %s\n", summary))
	sb.WriteString("=============================")

	return FoldedView{
		Summary:   summary,
		Files:     activeFiles,
		Tests:     testStatuses,
		Variables: s.Variables,
		Text:      sb.String(),
	}
}

// ToJSON serializes the materialized state to formatted JSON bytes.
func (s FoldedState) ToJSON() ([]byte, error) {
	return json.MarshalIndent(s, "", "  ")
}

// FoldedStateFromJSON deserializes a materialized state from JSON bytes.
func FoldedStateFromJSON(data []byte) (FoldedState, error) {
	var s FoldedState
	err := json.Unmarshal(data, &s)
	if s.Files == nil {
		s.Files = make(map[string]FileState)
	}
	if s.Deletions == nil {
		s.Deletions = make(map[string]time.Time)
	}
	if s.Tests == nil {
		s.Tests = make(map[string]TestOutcome)
	}
	if s.Variables == nil {
		s.Variables = make(map[string]any)
	}
	return s, err
}

// StateFolder defines the interface for folding append-only tool events into state views.
type StateFolder interface {
	ApplyEvent(event Event)
	FoldEvents(events []Event)
	GetFoldedState() FoldedState
	FoldedView() FoldedView
}

// FoldedStateGenerator folds append-only agent tool events into a compact current state view.
type FoldedStateGenerator struct {
	mu       sync.RWMutex
	eventLog []Event
	state    FoldedState
}

// Ensure FoldedStateGenerator conforms to StateFolder.
var _ StateFolder = (*FoldedStateGenerator)(nil)

// NewFoldedStateGenerator constructs an empty materialized view generator.
func NewFoldedStateGenerator() *FoldedStateGenerator {
	return &FoldedStateGenerator{
		eventLog: make([]Event, 0),
		state: FoldedState{
			Files:     make(map[string]FileState),
			Deletions: make(map[string]time.Time),
			Tests:     make(map[string]TestOutcome),
			Variables: make(map[string]any),
		},
	}
}

// NewFoldedStateGeneratorWithEvents constructs a generator pre-folded with the provided events.
func NewFoldedStateGeneratorWithEvents(events []Event) *FoldedStateGenerator {
	gen := NewFoldedStateGenerator()
	gen.FoldEvents(events)
	return gen
}

// ApplyEvent applies a single event to the generator, folding the state and appending to the audit log.
func (g *FoldedStateGenerator) ApplyEvent(event Event) {
	g.mu.Lock()
	defer g.mu.Unlock()

	event.resolveFields()

	now := event.Timestamp
	if now.IsZero() {
		now = time.Now().UTC()
		event.Timestamp = now
	}

	if event.ID == "" {
		event.ID = fmt.Sprintf("evt-%06d", g.state.EventCount+1)
	}

	mutType := NormalizeMutationType(string(event.Type))

	switch mutType {
	case MutationFileWrite:
		path := event.Path
		prevVersion := 0
		if prev, exists := g.state.Files[path]; exists {
			prevVersion = prev.Version
		}
		g.state.Files[path] = FileState{
			Path:         path,
			Content:      event.Content,
			LineCount:    countLines(event.Content),
			ByteSize:     len(event.Content),
			Version:      prevVersion + 1,
			LastModified: now,
		}
		delete(g.state.Deletions, path)

	case MutationFileEdit:
		path := event.Path
		prevContent := ""
		prevVersion := 0
		if prev, exists := g.state.Files[path]; exists {
			prevContent = prev.Content
			prevVersion = prev.Version
		}

		var updatedContent string
		if event.Content != "" {
			updatedContent = event.Content
		} else if event.OldString != "" {
			if event.ReplaceAll {
				updatedContent = strings.ReplaceAll(prevContent, event.OldString, event.NewString)
			} else {
				updatedContent = strings.Replace(prevContent, event.OldString, event.NewString, 1)
			}
		} else if event.NewString != "" {
			updatedContent = event.NewString
		} else {
			updatedContent = prevContent
		}

		g.state.Files[path] = FileState{
			Path:         path,
			Content:      updatedContent,
			LineCount:    countLines(updatedContent),
			ByteSize:     len(updatedContent),
			Version:      prevVersion + 1,
			LastModified: now,
		}
		delete(g.state.Deletions, path)

	case MutationFileDelete:
		path := event.Path
		delete(g.state.Files, path)
		g.state.Deletions[path] = now

	case MutationTestOutcome:
		name := event.TestName
		status := NormalizeTestStatus(string(event.Status))
		prevRuns := 0
		if prev, exists := g.state.Tests[name]; exists {
			prevRuns = prev.RunCount
		}
		g.state.Tests[name] = TestOutcome{
			Name:         name,
			Status:       status,
			Output:       event.Output,
			Error:        event.Error,
			Duration:     event.Duration,
			RunCount:     prevRuns + 1,
			LastExecuted: now,
		}

	case MutationVariableSet:
		if event.Key != "" {
			g.state.Variables[event.Key] = event.Value
		}

	case MutationVariableDelete:
		if event.Key != "" {
			delete(g.state.Variables, event.Key)
		}
	}

	g.state.EventCount++
	g.state.LastEventID = event.ID
	g.state.LastEventTime = now
	g.eventLog = append(g.eventLog, event)
}

// FoldEvents folds an ordered slice of events into the materialized state.
func (g *FoldedStateGenerator) FoldEvents(events []Event) {
	for _, e := range events {
		g.ApplyEvent(e)
	}
}

// GetFoldedState returns an isolated copy of the current folded state.
func (g *FoldedStateGenerator) GetFoldedState() FoldedState {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.state.Clone()
}

// FoldedView produces the compact agent-facing presentation of the current materialized state.
func (g *FoldedStateGenerator) FoldedView() FoldedView {
	return g.GetFoldedState().FoldedView()
}

// FoldedViewText returns the formatted text string of the folded view for agent prompt injection.
func (g *FoldedStateGenerator) FoldedViewText() string {
	return g.FoldedView().Text
}

// AuditLog returns an isolated copy of the append-only event log.
func (g *FoldedStateGenerator) AuditLog() []Event {
	g.mu.RLock()
	defer g.mu.RUnlock()
	out := make([]Event, len(g.eventLog))
	copy(out, g.eventLog)
	return out
}

// EventCount returns the total number of events recorded in the append-only log.
func (g *FoldedStateGenerator) EventCount() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.eventLog)
}

// Reset clears both the append-only log and the folded state.
func (g *FoldedStateGenerator) Reset() {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.eventLog = make([]Event, 0)
	g.state = FoldedState{
		Files:     make(map[string]FileState),
		Deletions: make(map[string]time.Time),
		Tests:     make(map[string]TestOutcome),
		Variables: make(map[string]any),
	}
}

func countLines(s string) int {
	if s == "" {
		return 0
	}
	lines := strings.Count(s, "\n")
	if !strings.HasSuffix(s, "\n") {
		lines++
	}
	return lines
}
