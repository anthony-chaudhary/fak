package fleetsearch

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
	"github.com/anthony-chaudhary/fak/internal/toolproc"
)

// Config names the three stores and the observation instant. Every store has an
// explicit skip bit so an intentional omission remains visible in coverage.
type Config struct {
	LifecyclePath    string
	RegistrationPath string
	ToolProcessPath  string
	SkipLifecycle    bool
	SkipRegistration bool
	SkipToolProcess  bool
	Now              time.Time
	BootTime         time.Time
	StaleAfter       time.Duration
	Limit            int
}

// Run reads each store independently. A missing, corrupt, unreadable, or skipped
// store becomes coverage evidence rather than an all-or-nothing command error.
func Run(rawQuery string, cfg Config) (Report, error) {
	q, err := ParseQuery(rawQuery, cfg.Limit)
	if err != nil {
		return Report{}, err
	}
	if cfg.Now.IsZero() {
		cfg.Now = time.Now().UTC()
	}

	lifecycle, lifecycleEvidence := loadLifecycle(cfg.LifecyclePath, cfg.SkipLifecycle)
	registrations, registrationEvidence := loadRegistrations(cfg.RegistrationPath, cfg.SkipRegistration)
	tools, toolCoverage := loadToolProcesses(cfg.ToolProcessPath, cfg.SkipToolProcess)
	input := Input{
		Query: q, Lifecycle: lifecycle, Registrations: registrations, ToolProcesses: tools,
		Coverage: []Coverage{lifecycleEvidence, registrationEvidence, toolCoverage},
		Now:      cfg.Now, BootTime: cfg.BootTime, StaleAfter: cfg.StaleAfter,
	}
	report, err := Search(input)
	if err == nil {
		return report, nil
	}

	// ParseEvents validates individual tool rows; Fold additionally validates
	// cross-row state transitions. Treat a bad transition as partial store
	// coverage and retain the other two stores instead of losing the search.
	input.ToolProcesses = nil
	input.Coverage[2].Status = CoverageIncomplete
	input.Coverage[2].Detail = err.Error()
	return Search(input)
}

func loadLifecycle(path string, skip bool) ([]sessionjournal.Event, Coverage) {
	coverage := Coverage{Store: StoreLifecycle, Path: path}
	if skip {
		coverage.Status = CoverageSkipped
		return nil, coverage
	}
	data, err := os.ReadFile(path)
	if err != nil {
		coverage.Status = CoverageUnavailable
		coverage.Detail = err.Error()
		return nil, coverage
	}
	events, health := sessionjournal.ParseEventsReport(string(data))
	coverage.Records = len(events)
	if health.MalformedRows > 0 || health.WrongSchemaRows > 0 || health.MissingIDRows > 0 || health.ScanError != "" || health.ReadError != "" {
		coverage.Status = CoverageIncomplete
		coverage.Detail = lifecycleHealthDetail(health)
	} else {
		coverage.Status = CoverageComplete
	}
	return events, coverage
}

func lifecycleHealthDetail(health sessionjournal.ParseHealth) string {
	var details []string
	if health.MalformedRows > 0 {
		details = append(details, fmt.Sprintf("malformed_rows=%d", health.MalformedRows))
	}
	if health.WrongSchemaRows > 0 {
		details = append(details, fmt.Sprintf("wrong_schema_rows=%d", health.WrongSchemaRows))
	}
	if health.MissingIDRows > 0 {
		details = append(details, fmt.Sprintf("missing_id_rows=%d", health.MissingIDRows))
	}
	if health.ScanError != "" {
		details = append(details, "scan_error="+health.ScanError)
	}
	if health.ReadError != "" {
		details = append(details, "read_error="+health.ReadError)
	}
	return strings.Join(details, "; ")
}

func loadRegistrations(path string, skip bool) ([]sessionregistry.Record, Coverage) {
	coverage := Coverage{Store: StoreRegistration, Path: path}
	if skip {
		coverage.Status = CoverageSkipped
		return nil, coverage
	}
	data, err := os.ReadFile(path)
	if err != nil {
		coverage.Status = CoverageUnavailable
		coverage.Detail = err.Error()
		return nil, coverage
	}
	rows, partial, detail := parseRegistrationStore(data)
	coverage.Records = len(rows)
	if partial {
		coverage.Status = CoverageIncomplete
		coverage.Detail = detail
	} else {
		coverage.Status = CoverageComplete
	}
	return rows, coverage
}

func parseRegistrationStore(data []byte) ([]sessionregistry.Record, bool, string) {
	firstSchema := firstRowSchema(data)
	if firstSchema == sessionjournal.Schema {
		events, health := sessionjournal.ParseEventsReport(string(data))
		rows := registrationsFromLifecycle(events)
		partial := health.MalformedRows > 0 || health.WrongSchemaRows > 0 || health.MissingIDRows > 0 || health.ScanError != "" || health.ReadError != ""
		return rows, partial, lifecycleHealthDetail(health)
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return []sessionregistry.Record{}, false, ""
	}

	latest := map[string]sessionregistry.Record{}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	line := 0
	for scanner.Scan() {
		line++
		raw := bytes.TrimSpace(scanner.Bytes())
		if len(raw) == 0 || bytes.HasPrefix(raw, []byte("#")) {
			continue
		}
		var event sessionregistry.Event
		if err := json.Unmarshal(raw, &event); err != nil {
			return sortedRegistrationRows(latest), true, fmt.Sprintf("line %d: %v", line, err)
		}
		if event.Schema != sessionregistry.Schema {
			return sortedRegistrationRows(latest), true, fmt.Sprintf("line %d: schema %q", line, event.Schema)
		}
		if err := sessionregistry.Validate(event.Record); err != nil {
			return sortedRegistrationRows(latest), true, fmt.Sprintf("line %d: %v", line, err)
		}
		latest[event.Record.RegistrationID] = event.Record
	}
	if err := scanner.Err(); err != nil {
		return sortedRegistrationRows(latest), true, err.Error()
	}
	return sortedRegistrationRows(latest), false, ""
}

func firstRowSchema(data []byte) string {
	for _, line := range bytes.Split(data, []byte{'\n'}) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 || bytes.HasPrefix(line, []byte("#")) {
			continue
		}
		var header struct {
			Schema string `json:"schema"`
		}
		if json.Unmarshal(line, &header) == nil {
			return header.Schema
		}
		return ""
	}
	return ""
}

func registrationsFromLifecycle(events []sessionjournal.Event) []sessionregistry.Record {
	latest := map[string]sessionregistry.Record{}
	for _, session := range sessionjournal.FoldEvents(events) {
		if session.Registration == nil || strings.TrimSpace(session.Registration.RegistrationID) == "" {
			continue
		}
		carry := *session.Registration
		row := sessionregistry.Record{
			Schema: sessionregistry.Schema, RegistrationID: carry.RegistrationID,
			ParentRegistrationID: carry.ParentRegistrationID, ParentAttemptID: carry.ParentAttemptID,
			RootRegistrationID: carry.RootRegistrationID, RootOutcome: carry.RootOutcome, RootIssue: carry.RootIssue,
			TaskID: carry.TaskID, GoalID: carry.GoalID, AttemptID: carry.AttemptID, ResumeOfAttemptID: carry.ResumeOfAttemptID,
			LaunchKind: carry.LaunchKind, Scope: append([]string(nil), carry.Scope...), Lane: carry.Lane, LeaseID: carry.LeaseID,
			Identity: sessionregistry.Identity{
				Runtime: carry.Runtime, SessionID: carry.SessionID, ThreadID: carry.ThreadID, PID: carry.PID,
				ProcessStartedAt: parseRegistrationTime(carry.ProcessStartedAt), HostID: carry.HostID,
			},
			State: sessionregistry.State(carry.State), Reason: carry.Reason, WitnessRef: carry.WitnessRef,
			CreatedAt: parseRegistrationTime(carry.CreatedAt), StartedAt: parseRegistrationTime(carry.StartedAt),
			HeartbeatAt: parseRegistrationTime(carry.HeartbeatAt), TerminalAt: parseRegistrationTime(carry.TerminalAt),
		}
		latest[row.RegistrationID] = row
	}
	return sortedRegistrationRows(latest)
}

func parseRegistrationTime(value string) time.Time {
	t, _ := time.Parse(time.RFC3339Nano, strings.TrimSpace(value))
	return t.UTC()
}

func sortedRegistrationRows(latest map[string]sessionregistry.Record) []sessionregistry.Record {
	rows := make([]sessionregistry.Record, 0, len(latest))
	for _, row := range latest {
		rows = append(rows, row)
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].CreatedAt.Equal(rows[j].CreatedAt) {
			return rows[i].RegistrationID < rows[j].RegistrationID
		}
		return rows[i].CreatedAt.Before(rows[j].CreatedAt)
	})
	return rows
}

func loadToolProcesses(path string, skip bool) ([]toolproc.Event, Coverage) {
	coverage := Coverage{Store: StoreToolProcess, Path: path}
	if skip {
		coverage.Status = CoverageSkipped
		return nil, coverage
	}
	f, err := toolproc.OpenShareDelete(path)
	if err != nil {
		coverage.Status = CoverageUnavailable
		coverage.Detail = err.Error()
		return nil, coverage
	}
	events, parseErr := toolproc.ParseEvents(f)
	closeErr := f.Close()
	if parseErr != nil {
		coverage.Status = CoverageIncomplete
		coverage.Detail = parseErr.Error()
		return nil, coverage
	}
	if closeErr != nil {
		coverage.Status = CoverageIncomplete
		coverage.Detail = closeErr.Error()
		return events, coverage
	}
	coverage.Status = CoverageComplete
	coverage.Records = len(events)
	return events, coverage
}
