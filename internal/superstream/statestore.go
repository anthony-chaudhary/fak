package superstream

// Durable stream state and the served stream registry.
//
// Invariant: one durable summary per stream, O(1) served read, O(W) refresh,
// never the concatenation of queues.
//
//   - State: each stream persists its own StreamState next to its spec
//     (<spec>.state.json, written atomically by `fak superstream step --apply`),
//     so a second invocation resumes instead of restarting at 0 completed.
//     Loading reconciles the persisted state against the edited spec: the spec
//     owns the queue shape and static fields, the state file owns execution
//     fields only. A state file that does not parse, carries an unknown schema,
//     or names a different stream is refused (fail closed) rather than
//     silently replaced with a fresh state.
//   - Registry: .fak/superstream/registry.json holds exactly one small summary
//     per stream. Serving navigation over W streams is one file read (O(1) in
//     queue length, O(W) in summaries); refreshing one stream rewrites that W
//     summaries and never copies any stream's queue into another stream's row.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// StateSchema and RegistrySchema version the two durable documents.
const (
	StateSchema    = "fak.superstream.state.v1"
	RegistrySchema = "fak.superstream.registry.v1"
)

// stateEnvelope is the on-disk shape of one stream's persisted state.
type stateEnvelope struct {
	Schema    string      `json:"schema"`
	SpecPath  string      `json:"spec_path"`
	UpdatedAt time.Time   `json:"updated_at"`
	State     StreamState `json:"state"`
}

// registryEnvelope is the on-disk shape of the served registry snapshot.
type registryEnvelope struct {
	Schema    string            `json:"schema"`
	UpdatedAt time.Time         `json:"updated_at"`
	Streams   []RegistrySummary `json:"streams"`
}

// StateFilePath returns the sibling state file path for a spec path: the
// spec's extension is stripped and ".state.json" appended
// ("a.json" -> "a.state.json", "mystream.super.json" -> "mystream.super.state.json").
func StateFilePath(specPath string) string {
	dir, base := filepath.Split(specPath)
	stem := strings.TrimSuffix(base, filepath.Ext(base))
	return filepath.Join(dir, stem+".state.json")
}

// RegistryFilePath is the cwd-relative location of the served registry.
func RegistryFilePath() string {
	return filepath.Join(".fak", "superstream", "registry.json")
}

// SaveState persists state next to specPath atomically (temp file in the same
// directory, then rename over the target), creating the parent directory if
// needed.
func SaveState(specPath string, state StreamState, now time.Time) error {
	target := StateFilePath(specPath)
	payload, err := json.MarshalIndent(stateEnvelope{
		Schema:    StateSchema,
		SpecPath:  specPath,
		UpdatedAt: now,
		State:     state,
	}, "", "  ")
	if err != nil {
		return fmt.Errorf("encode state for %s: %w", specPath, err)
	}
	if err := writeFileAtomic(target, append(payload, '\n')); err != nil {
		return fmt.Errorf("write state for %s: %w", specPath, err)
	}
	return nil
}

// writeFileAtomic writes data to target via a same-directory temp file and a
// rename, so a reader never observes a half-written document.
func writeFileAtomic(target string, data []byte) error {
	dir := filepath.Dir(target)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, "."+filepath.Base(target)+".tmp-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return err
	}
	if err := os.Rename(tmpName, target); err != nil {
		os.Remove(tmpName)
		return err
	}
	return nil
}

// LoadState returns the persisted state for specPath reconciled against spec.
// found is false (and a fresh state returned) when no state file exists.
//
// Reconciliation rules: the current spec owns the queue shape and every static
// field (Title, Lane, Tree, MaxTurns, MaxTokens, Witness); the persisted state
// owns only the execution fields (Status, TurnsSpent, TokensSpent,
// WitnessResult, CommitSHA, Error) of items that still exist by ID. Persisted
// items missing from the spec are dropped; new spec items start pending.
// Counters are recomputed from the reconciled queue, never trusted from disk.
func LoadState(specPath string, spec StreamSpec) (StreamState, bool, error) {
	norm := spec.NormalizedSpec()
	raw, err := os.ReadFile(StateFilePath(specPath))
	if err != nil {
		if os.IsNotExist(err) {
			return NewStreamState(norm), false, nil
		}
		return StreamState{}, false, fmt.Errorf("read state for %s: %w", specPath, err)
	}

	var env stateEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return StreamState{}, false, fmt.Errorf("parse state %s: %w", StateFilePath(specPath), err)
	}
	if env.Schema != StateSchema {
		return StreamState{}, false, fmt.Errorf("state %s: unsupported schema %q (want %q)", StateFilePath(specPath), env.Schema, StateSchema)
	}
	if env.State.StreamID != spec.ID {
		return StreamState{}, false, fmt.Errorf("state %s: stream id %q does not match spec id %q; refusing misplaced state", StateFilePath(specPath), env.State.StreamID, spec.ID)
	}

	state := reconcileState(norm, env.State)
	return state, true, nil
}

// reconcileState rebuilds the queue from the normalized spec, carrying over the
// execution fields of persisted items that still exist by ID.
func reconcileState(norm StreamSpec, persisted StreamState) StreamState {
	byID := make(map[string]WorkItem, len(persisted.Queue))
	for _, it := range persisted.Queue {
		byID[it.ID] = it
	}

	queue := make([]WorkItem, len(norm.Queue))
	for i, it := range norm.Queue {
		if p, ok := byID[it.ID]; ok {
			it.Status = p.Status
			if it.Status == "" {
				it.Status = ItemPending
			}
			it.TurnsSpent = p.TurnsSpent
			it.TokensSpent = p.TokensSpent
			it.WitnessResult = p.WitnessResult
			it.CommitSHA = p.CommitSHA
			it.Error = p.Error
		}
		queue[i] = it
	}

	idx := persisted.ActiveIndex
	if idx < 0 {
		idx = 0
	}
	if idx > len(queue) {
		idx = len(queue)
	}

	state := StreamState{
		StreamID:          norm.ID,
		Intent:            norm.Intent,
		ActiveIndex:       idx,
		Queue:             queue,
		CurrentLease:      persisted.CurrentLease,
		TotalTurnsSpent:   persisted.TotalTurnsSpent,
		TotalTokensSpent:  persisted.TotalTokensSpent,
		CurrentItemTurns:  persisted.CurrentItemTurns,
		CurrentItemTokens: persisted.CurrentItemTokens,
		Closed:            persisted.Closed,
		CloseReason:       persisted.CloseReason,
	}
	for _, it := range queue {
		switch it.Status {
		case ItemCompleted:
			state.CompletedCount++
		case ItemFailed:
			state.FailedCount++
		case ItemYielded:
			state.YieldedCount++
		case ItemPending, ItemLeaseAcquired, ItemExecuting, ItemWitnessed, ItemCommitted:
			// Not summarized as terminal counters; the queue itself carries these states.
		}
	}
	return state
}

// RegistrySummary is one stream's served row: enough to navigate W streams
// without reading any spec or state file.
type RegistrySummary struct {
	StreamID         string    `json:"stream_id"`
	SpecPath         string    `json:"spec_path"`
	Intent           string    `json:"intent"`
	TotalItems       int       `json:"total_items"`
	Completed        int       `json:"completed"`
	Failed           int       `json:"failed"`
	Yielded          int       `json:"yielded"`
	TotalTurnsSpent  int       `json:"total_turns_spent"`
	TotalTokensSpent int       `json:"total_tokens_spent"`
	Closed           bool      `json:"closed"`
	CloseReason      string    `json:"close_reason,omitempty"`
	ActiveItemID     string    `json:"active_item_id,omitempty"`
	UpdatedAt        time.Time `json:"updated_at"`
}

// SummaryFor builds the served registry row for one stream.
func SummaryFor(specPath string, norm StreamSpec, state StreamState, now time.Time) RegistrySummary {
	activeID := ""
	if active := state.ActiveItem(); active != nil {
		activeID = active.ID
	}
	return RegistrySummary{
		StreamID:         state.StreamID,
		SpecPath:         specPath,
		Intent:           norm.Intent,
		TotalItems:       len(state.Queue),
		Completed:        state.CompletedCount,
		Failed:           state.FailedCount,
		Yielded:          state.YieldedCount,
		TotalTurnsSpent:  state.TotalTurnsSpent,
		TotalTokensSpent: state.TotalTokensSpent,
		Closed:           state.Closed,
		CloseReason:      state.CloseReason,
		ActiveItemID:     activeID,
		UpdatedAt:        now,
	}
}

// LoadRegistry serves the registry snapshot in one file read. A missing
// registry is a valid empty answer (nil, nil), not an error: no streams yet.
func LoadRegistry() ([]RegistrySummary, error) {
	raw, err := os.ReadFile(RegistryFilePath())
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read registry: %w", err)
	}
	var env registryEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, fmt.Errorf("parse registry %s: %w", RegistryFilePath(), err)
	}
	if env.Schema != RegistrySchema {
		return nil, fmt.Errorf("registry %s: unsupported schema %q (want %q)", RegistryFilePath(), env.Schema, RegistrySchema)
	}
	return env.Streams, nil
}

// SaveRegistry rewrites the whole registry atomically.
func SaveRegistry(streams []RegistrySummary, now time.Time) error {
	if streams == nil {
		streams = []RegistrySummary{}
	}
	payload, err := json.MarshalIndent(registryEnvelope{
		Schema:    RegistrySchema,
		UpdatedAt: now,
		Streams:   streams,
	}, "", "  ")
	if err != nil {
		return fmt.Errorf("encode registry: %w", err)
	}
	if err := writeFileAtomic(RegistryFilePath(), append(payload, '\n')); err != nil {
		return fmt.Errorf("write registry: %w", err)
	}
	return nil
}

// UpsertRegistryEntry replaces the row for entry.StreamID (or appends one).
// Refreshing one stream costs O(W) in summaries; no queue is ever copied into
// the registry.
func UpsertRegistryEntry(entry RegistrySummary, now time.Time) error {
	streams, err := LoadRegistry()
	if err != nil {
		return err
	}
	replaced := false
	for i := range streams {
		if streams[i].StreamID == entry.StreamID {
			streams[i] = entry
			replaced = true
			break
		}
	}
	if !replaced {
		streams = append(streams, entry)
	}
	return SaveRegistry(streams, now)
}
