package trajctl

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	// Schema is the pinned JSONL row schema for the trajectory-control ledger.
	Schema = "fak-trajctl/1"

	KindObjective = "objective"
	KindScore     = "score"
)

// ObjectiveStatus is the closed lifecycle vocabulary for a live objective.
type ObjectiveStatus string

const (
	StatusActive    ObjectiveStatus = "active"
	StatusPaused    ObjectiveStatus = "paused"
	StatusMet       ObjectiveStatus = "met"
	StatusAbandoned ObjectiveStatus = "abandoned"
)

// WitnessRung names the strength of the evidence behind a ScoreRow.
type WitnessRung string

const (
	// W3 is deterministic evidence: a witnessed commit, green suite, or benchmark
	// harness row. It can gate automated steering.
	W3 WitnessRung = "W3"
	// W2 is structured activity evidence: transcript/session signals that can
	// indicate progress or stall, but may need corroboration.
	W2 WitnessRung = "W2"
	// W1 is a judge or rubric verdict.
	W1 WitnessRung = "W1"
	// W0 is self-report. W0 is recorded for calibration, but must never gate alone.
	W0 WitnessRung = "W0"
)

// Budget is the objective's declared envelope. Zero fields mean unspecified.
type Budget struct {
	Turns  int `json:"turns,omitempty"`
	Tokens int `json:"tokens,omitempty"`
}

// PlanPhase is one declared unit of work an objective can score progress against.
type PlanPhase struct {
	ID    string `json:"id"`
	Title string `json:"title,omitempty"`
}

// Objective is the durable thing being scored.
type Objective struct {
	ID        string          `json:"id"`
	ParentID  string          `json:"parent_id,omitempty"`
	Statement string          `json:"statement"`
	Plan      []PlanPhase     `json:"plan,omitempty"`
	Scorers   []string        `json:"scorers,omitempty"`
	Budget    Budget          `json:"budget,omitempty"`
	Status    ObjectiveStatus `json:"status"`
}

// EvidenceRef is a pointer to the witness behind a score. For W3 commit progress,
// Kind is commonly "commit" and Ref is the SHA.
type EvidenceRef struct {
	Kind   string `json:"kind"`
	Ref    string `json:"ref"`
	Detail string `json:"detail,omitempty"`
}

// ScoreRow is one method's value for one objective at one time. Value is a unit
// interval: 0 means no progress, 1 means fully met for that method.
type ScoreRow struct {
	ObjectiveID string        `json:"objective_id"`
	Value       float64       `json:"value"`
	Method      string        `json:"method"`
	Version     string        `json:"version"`
	Witness     WitnessRung   `json:"witness"`
	Evidence    []EvidenceRef `json:"evidence,omitempty"`
	UnixMillis  int64         `json:"unix_millis,omitempty"`
	SessionID   string        `json:"session_id,omitempty"`
	RunID       string        `json:"run_id,omitempty"`
}

// Row is the append-only ledger envelope. Exactly one payload is set according to
// Kind.
type Row struct {
	Schema    string     `json:"schema"`
	Kind      string     `json:"kind"`
	Objective *Objective `json:"objective,omitempty"`
	Score     *ScoreRow  `json:"score,omitempty"`
}

// State is the folded ledger: latest objective record by id, plus score history in
// append order.
type State struct {
	Objectives map[string]Objective `json:"objectives"`
	Scores     []ScoreRow           `json:"scores"`
}

// ObjectiveRecord builds a ledger row for an Objective.
func ObjectiveRecord(o Objective) Row {
	if o.Status == "" {
		o.Status = StatusActive
	}
	return Row{Schema: Schema, Kind: KindObjective, Objective: &o}
}

// ScoreRecord builds a ledger row for a ScoreRow.
func ScoreRecord(s ScoreRow) Row {
	return Row{Schema: Schema, Kind: KindScore, Score: &s}
}

// Append writes one validated row to path as JSONL, creating parent directories.
func Append(path string, row Row) error {
	if err := Validate(row); err != nil {
		return err
	}
	if dir := filepath.Dir(path); dir != "." && dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	b, err := json.Marshal(row)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(append(b, '\n'))
	return err
}

// ReadLedgerFile reads and parses path. A missing or unreadable ledger is an empty
// first-run state, not an error.
func ReadLedgerFile(path string) []Row {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	return ParseLedger(string(b))
}

// ParseLedger parses JSONL rows, skipping blank, malformed, foreign-schema, and
// invalid rows so a torn append cannot poison the whole fold.
func ParseLedger(content string) []Row {
	var rows []Row
	sc := bufio.NewScanner(strings.NewReader(content))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var row Row
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			continue
		}
		if err := Validate(row); err != nil {
			continue
		}
		rows = append(rows, row)
	}
	return rows
}

// Fold returns the latest objective state and all score rows in append order.
func Fold(rows []Row) State {
	st := State{Objectives: map[string]Objective{}}
	for _, row := range rows {
		switch row.Kind {
		case KindObjective:
			st.Objectives[row.Objective.ID] = *row.Objective
		case KindScore:
			st.Scores = append(st.Scores, *row.Score)
		}
	}
	return st
}

// ScoresFor returns score rows for objectiveID in append order.
func (s State) ScoresFor(objectiveID string) []ScoreRow {
	out := make([]ScoreRow, 0)
	for _, row := range s.Scores {
		if row.ObjectiveID == objectiveID {
			out = append(out, row)
		}
	}
	return out
}

// ObjectiveIDs returns folded objective ids in lexical order for deterministic
// rendering.
func (s State) ObjectiveIDs() []string {
	ids := make([]string, 0, len(s.Objectives))
	for id := range s.Objectives {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// Validate checks that a row is schema-correct and carries a valid payload.
func Validate(row Row) error {
	if row.Schema != Schema {
		return fmt.Errorf("trajctl: schema %q, want %q", row.Schema, Schema)
	}
	switch row.Kind {
	case KindObjective:
		if row.Objective == nil || row.Score != nil {
			return errors.New("trajctl: objective row must carry exactly one objective")
		}
		return validateObjective(*row.Objective)
	case KindScore:
		if row.Score == nil || row.Objective != nil {
			return errors.New("trajctl: score row must carry exactly one score")
		}
		return validateScore(*row.Score)
	default:
		return fmt.Errorf("trajctl: unknown row kind %q", row.Kind)
	}
}

func validateObjective(o Objective) error {
	if o.ID == "" {
		return errors.New("trajctl: objective id is required")
	}
	if o.Statement == "" {
		return errors.New("trajctl: objective statement is required")
	}
	if !validStatus(o.Status) {
		return fmt.Errorf("trajctl: invalid objective status %q", o.Status)
	}
	seen := map[string]bool{}
	for _, phase := range o.Plan {
		if phase.ID == "" {
			return errors.New("trajctl: plan phase id is required")
		}
		if seen[phase.ID] {
			return fmt.Errorf("trajctl: duplicate plan phase %q", phase.ID)
		}
		seen[phase.ID] = true
	}
	return nil
}

func validateScore(s ScoreRow) error {
	if s.ObjectiveID == "" {
		return errors.New("trajctl: score objective id is required")
	}
	if s.Value < 0 || s.Value > 1 {
		return fmt.Errorf("trajctl: score value %v outside [0,1]", s.Value)
	}
	if s.Method == "" {
		return errors.New("trajctl: score method is required")
	}
	if s.Version == "" {
		return errors.New("trajctl: score version is required")
	}
	if !validWitness(s.Witness) {
		return fmt.Errorf("trajctl: invalid witness rung %q", s.Witness)
	}
	for _, ev := range s.Evidence {
		if ev.Kind == "" || ev.Ref == "" {
			return errors.New("trajctl: evidence kind and ref are required")
		}
	}
	return nil
}

func validStatus(s ObjectiveStatus) bool {
	switch s {
	case StatusActive, StatusPaused, StatusMet, StatusAbandoned:
		return true
	default:
		return false
	}
}

func validWitness(w WitnessRung) bool {
	switch w {
	case W3, W2, W1, W0:
		return true
	default:
		return false
	}
}
