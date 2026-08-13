package sessionjournal

// Durable work-session events use the existing journal's crash-safe framing.
// This is deliberately a typed codec over Journal rather than another state
// store: replay is the only source of the materialized desktop projection.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

const WorkEventSchema = "fak.work-session.event.v1"

type WorkEventKind string

const (
	WorkSessionOpened       WorkEventKind = "SESSION_OPENED"
	WorkTerminalOutput      WorkEventKind = "TERMINAL_OUTPUT"
	WorkEffectIntent        WorkEventKind = "EFFECT_INTENT"
	WorkEffectResolved      WorkEventKind = "EFFECT_RESOLVED"
	WorkMoveTransitionEvent WorkEventKind = "MOVE_TRANSITION"
)

type EffectVerdict string

const (
	EffectKnownNotRun EffectVerdict = "KNOWN_NOT_RUN"
	EffectConfirmed   EffectVerdict = "CONFIRMED"
	EffectUncertain   EffectVerdict = "UNCERTAIN"
)

type WorkEvent struct {
	Schema      string             `json:"schema"`
	SessionID   string             `json:"session_id"`
	Kind        WorkEventKind      `json:"kind"`
	WriterEpoch string             `json:"writer_epoch"`
	Terminal    []byte             `json:"terminal,omitempty"`
	EffectID    string             `json:"effect_id,omitempty"`
	Command     string             `json:"command,omitempty"`
	Check       string             `json:"check,omitempty"`
	Verdict     EffectVerdict      `json:"verdict,omitempty"`
	Residency   *ResidencyIdentity `json:"residency,omitempty"`
	MovePhase   string             `json:"move_phase,omitempty"`
	SourceEpoch string             `json:"source_epoch,omitempty"`
	Destination *PlacementIdentity `json:"destination,omitempty"`
	Checkpoint  string             `json:"checkpoint_hash,omitempty"`
}

type PlacementIdentity struct {
	Provider             string   `json:"provider"`
	AccountRef           string   `json:"account_ref"`
	Model                string   `json:"model"`
	Compute              string   `json:"compute"`
	Capabilities         []string `json:"capabilities,omitempty"`
	CacheLineage         string   `json:"cache_lineage,omitempty"`
	SemanticDegradations []string `json:"semantic_degradations,omitempty"`
}

type WorkMoveTransition struct {
	Phase          string            `json:"phase"`
	SourceEpoch    string            `json:"source_epoch"`
	Destination    PlacementIdentity `json:"destination"`
	CheckpointHash string            `json:"checkpoint_hash,omitempty"`
}

type ResidencyIdentity struct {
	WorkspaceHead   string `json:"workspace_head"`
	WorkspaceDirty  string `json:"workspace_dirty"`
	PolicyHash      string `json:"policy_hash"`
	ToolSchema      string `json:"tool_schema"`
	CredentialEpoch string `json:"credential_epoch"`
	AdapterIdentity string `json:"adapter_identity"`
}

func (r ResidencyIdentity) RecoveryDependency() string {
	if r.WorkspaceHead == "" || r.WorkspaceDirty == "" {
		return "workspace fingerprint unavailable"
	}
	if r.PolicyHash == "" {
		return "policy hash unavailable"
	}
	if r.ToolSchema == "" {
		return "tool schema unavailable"
	}
	if r.CredentialEpoch == "" {
		return "credential epoch unavailable"
	}
	if r.AdapterIdentity == "" {
		return "adapter identity unavailable"
	}
	return ""
}

func (r ResidencyIdentity) Mismatch(current ResidencyIdentity) []string {
	pairs := []struct{ name, old, now string }{
		{"workspace_head", r.WorkspaceHead, current.WorkspaceHead},
		{"workspace_dirty", r.WorkspaceDirty, current.WorkspaceDirty},
		{"policy_hash", r.PolicyHash, current.PolicyHash},
		{"tool_schema", r.ToolSchema, current.ToolSchema},
		{"credential_epoch", r.CredentialEpoch, current.CredentialEpoch},
		{"adapter_identity", r.AdapterIdentity, current.AdapterIdentity},
	}
	var mismatches []string
	for _, pair := range pairs {
		if pair.old != pair.now {
			mismatches = append(mismatches, pair.name)
		}
	}
	return mismatches
}

type WorkEffect struct {
	ID      string        `json:"id"`
	Command string        `json:"command,omitempty"`
	Check   string        `json:"check,omitempty"`
	Verdict EffectVerdict `json:"verdict"`
}

type WorkSessionView struct {
	SessionID          string                `json:"session_id"`
	WriterEpoch        string                `json:"writer_epoch"`
	Transcript         []byte                `json:"transcript"`
	Effects            map[string]WorkEffect `json:"effects"`
	Residency          ResidencyIdentity     `json:"residency"`
	RecoveryDependency string                `json:"recovery_dependency,omitempty"`
	MoveTransitions    []WorkMoveTransition  `json:"move_transitions,omitempty"`
}

type WorkReplay struct {
	Sessions      map[string]WorkSessionView `json:"sessions"`
	RecoveredTail bool                       `json:"recovered_tail"`
	Records       int                        `json:"records"`
}

func AppendWorkEvent(path string, event WorkEvent) error {
	if event.SessionID == "" || event.WriterEpoch == "" {
		return errors.New("sessionjournal: work event requires session_id and writer_epoch")
	}
	if event.Kind != WorkSessionOpened {
		replay, err := ReplayWork(path)
		if err != nil {
			return err
		}
		active := replay.Sessions[event.SessionID].WriterEpoch
		if active == "" || active != event.WriterEpoch {
			return fmt.Errorf("sessionjournal: stale writer epoch %q for session %q (active %q)", event.WriterEpoch, event.SessionID, active)
		}
	}
	event.Schema = WorkEventSchema
	payload, err := json.Marshal(event)
	if err != nil {
		return err
	}
	return appendJSONLine(path, payload)
}

func appendJSONLine(path string, payload []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	lock, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	defer lock.Close()
	if err := flock.TryLock(lock); err != nil {
		return err
	}
	defer flock.Unlock(lock)
	if err := repairWorkTail(path); err != nil {
		return err
	}
	line := append(payload, '\n')
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	if _, err = f.Write(line); err == nil {
		err = f.Sync()
	}
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	return err
}

func repairWorkTail(path string) error {
	content, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) || len(content) == 0 {
		return nil
	}
	if err != nil {
		return err
	}
	if content[len(content)-1] == '\n' {
		return nil
	}
	last := bytes.LastIndexByte(content, '\n')
	return os.Truncate(path, int64(last+1))
}

func ReplayWork(path string) (WorkReplay, error) {
	result := WorkReplay{Sessions: map[string]WorkSessionView{}}
	content, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return result, nil
	}
	if err != nil {
		return result, err
	}
	result.RecoveredTail = len(content) > 0 && content[len(content)-1] != '\n'
	lines := bytes.Split(content, []byte{'\n'})
	for index, line := range lines {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		var event WorkEvent
		if err := json.Unmarshal(line, &event); err != nil {
			if index == len(lines)-1 {
				result.RecoveredTail = true
				break
			}
			return result, fmt.Errorf("sessionjournal: work event %d: %w", index, err)
		}
		if event.Schema != WorkEventSchema {
			continue
		}
		result.Records++
		view := result.Sessions[event.SessionID]
		if view.Effects == nil {
			view.Effects = map[string]WorkEffect{}
		}
		if event.Kind == WorkSessionOpened {
			view.SessionID, view.WriterEpoch = event.SessionID, event.WriterEpoch
			if event.Residency != nil {
				view.Residency = *event.Residency
				view.RecoveryDependency = view.Residency.RecoveryDependency()
			}
		} else if view.WriterEpoch == "" || event.WriterEpoch != view.WriterEpoch {
			return result, fmt.Errorf("sessionjournal: stale writer epoch %q for session %q (active %q)", event.WriterEpoch, event.SessionID, view.WriterEpoch)
		} else {
			switch event.Kind {
			case WorkTerminalOutput:
				view.Transcript = append(view.Transcript, event.Terminal...)
			case WorkEffectIntent:
				if _, exists := view.Effects[event.EffectID]; exists {
					return result, fmt.Errorf("sessionjournal: duplicate effect %q", event.EffectID)
				}
				view.Effects[event.EffectID] = WorkEffect{ID: event.EffectID, Command: event.Command, Check: event.Check, Verdict: EffectUncertain}
			case WorkEffectResolved:
				effect, exists := view.Effects[event.EffectID]
				if !exists {
					return result, fmt.Errorf("sessionjournal: unresolved effect %q", event.EffectID)
				}
				if event.Verdict != EffectKnownNotRun && event.Verdict != EffectConfirmed {
					return result, fmt.Errorf("sessionjournal: invalid conclusive verdict %q", event.Verdict)
				}
				effect.Verdict = event.Verdict
				view.Effects[event.EffectID] = effect
			case WorkMoveTransitionEvent:
				if event.Destination == nil {
					return result, fmt.Errorf("sessionjournal: move transition missing destination")
				}
				view.MoveTransitions = append(view.MoveTransitions, WorkMoveTransition{Phase: event.MovePhase, SourceEpoch: event.SourceEpoch, Destination: *event.Destination, CheckpointHash: event.Checkpoint})
			default:
				return result, fmt.Errorf("sessionjournal: unknown work event kind %q", event.Kind)
			}
		}
		result.Sessions[event.SessionID] = view
	}
	return result, nil
}
