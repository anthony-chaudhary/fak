package terminalrelief

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const Schema = "fak.terminal-relief.v1"

type Command struct {
	Argv []string `json:"argv"`
}
type Facts struct {
	PID               int       `json:"pid"`
	Handles           int       `json:"handles"`
	Threads           int       `json:"threads"`
	UnsafeDescendants []string  `json:"unsafe_descendants,omitempty"`
	Dashboards        []Command `json:"dashboards,omitempty"`
}
type State struct {
	Schema       string `json:"schema"`
	PID          int    `json:"pid"`
	Consecutive  int    `json:"consecutive"`
	LastObserved string `json:"last_observed,omitempty"`
	LastApplied  string `json:"last_applied,omitempty"`
}
type Decision struct {
	Schema      string `json:"schema"`
	Verdict     string `json:"verdict"`
	Reason      string `json:"reason"`
	Consecutive int    `json:"consecutive"`
	Apply       bool   `json:"apply"`
	Facts       Facts  `json:"facts"`
	State       State  `json:"state"`
}
type Config struct {
	HandleThreshold, ThreadThreshold, Consecutive int
	Cooldown                                      time.Duration
}

func Decide(f Facts, prior State, cfg Config, now time.Time, apply bool) Decision {
	state := prior
	if state.Schema != Schema || state.PID != f.PID {
		state = State{Schema: Schema, PID: f.PID}
	}
	pressure := f.PID > 0 && (f.Handles >= cfg.HandleThreshold || f.Threads >= cfg.ThreadThreshold)
	if !pressure {
		state.Consecutive = 0
		state.LastObserved = now.UTC().Format(time.RFC3339Nano)
		return Decision{Schema: Schema, Verdict: "BELOW_THRESHOLD", Reason: "terminal pressure is below both thresholds", Facts: f, State: state}
	}
	state.Consecutive++
	state.LastObserved = now.UTC().Format(time.RFC3339Nano)
	d := Decision{Schema: Schema, Verdict: "OBSERVE", Reason: "terminal pressure has not persisted for the configured run", Consecutive: state.Consecutive, Facts: f, State: state}
	if len(f.UnsafeDescendants) > 0 {
		d.Verdict, d.Reason = "ABSTAIN", "terminal owns unrecognized descendants: "+strings.Join(f.UnsafeDescendants, ", ")
		return d
	}
	if len(f.Dashboards) == 0 {
		d.Verdict, d.Reason = "ABSTAIN", "no restorable fak info dashboards were found"
		return d
	}
	if state.Consecutive < cfg.Consecutive {
		return d
	}
	if state.LastApplied != "" {
		if at, err := time.Parse(time.RFC3339Nano, state.LastApplied); err == nil && now.Sub(at) < cfg.Cooldown {
			d.Verdict, d.Reason = "COOLDOWN", "terminal relief is inside the configured cooldown"
			return d
		}
	}
	if !apply {
		d.Verdict, d.Reason = "WOULD_APPLY", "persistent terminal pressure is safe to relieve"
		return d
	}
	d.Verdict, d.Reason, d.Apply = "APPLY", "persistent terminal pressure is safe to relieve", true
	d.State.LastApplied = now.UTC().Format(time.RFC3339Nano)
	d.State.Consecutive = 0
	return d
}
func Load(path string) State {
	b, err := os.ReadFile(path)
	if err != nil {
		return State{}
	}
	var s State
	if json.Unmarshal(b, &s) != nil || s.Schema != Schema {
		return State{}
	}
	return s
}
func Save(path string, s State) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	tmp, err := os.CreateTemp(filepath.Dir(path), ".terminal-relief-*.tmp")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err = tmp.Write(b); err != nil {
		tmp.Close()
		return err
	}
	if err = tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err = tmp.Close(); err != nil {
		return err
	}
	if err = os.Rename(name, path); err != nil {
		return fmt.Errorf("replace terminal relief state: %w", err)
	}
	return nil
}
