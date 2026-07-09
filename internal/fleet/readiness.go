package fleet

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/linkstate"
)

// LabReadinessSchema is the LEGACY schema id this loader still ACCEPTS on read
// (records written before the three-phase model). New records are written with
// linkstate.Schema (fak.link_state/v1). Kept exported so the migration shim and
// docs can name the old contract.
const LabReadinessSchema = "fak.lab_readiness/v1"

// LabReadiness is the public dispatch gate for lab-backed machine classes: a
// linkstate.State (the general three-phase comms record — WAITING/CLEAR/WORKING)
// specialized with the lab's operator command hints. It is intentionally generic:
// private bridge outputs are folded before they enter this record, so no host,
// channel, token, thread id, transcript, or private path is needed. The embedded
// State carries the scrub + admit-derivation invariants; see internal/linkstate.
type LabReadiness struct {
	linkstate.State
	Commands *LabReadinessCommands `json:"commands,omitempty"`
}

// LabReadinessCommands are the phase-aligned operator hints for driving the gate.
type LabReadinessCommands struct {
	MarkClear   string `json:"mark_clear,omitempty"`
	MarkWaiting string `json:"mark_waiting,omitempty"`
	MarkWorking string `json:"mark_working,omitempty"`
}

func DefaultLabReadinessCommands() LabReadinessCommands {
	return LabReadinessCommands{
		MarkClear:   "fak lab readiness --phase CLEAR --write-default --json",
		MarkWaiting: "fak lab readiness --phase WAITING --write-default --json",
		MarkWorking: "fak lab readiness --phase WORKING --write-default --json",
	}
}

// NewLabReadiness builds a lab readiness record for a machine class in the given
// phase. Empty fields take the phase-appropriate linkstate defaults; admit is
// derived from the phase (CLEAR admits, everything else fails closed).
func NewLabReadiness(machineClass string, phase linkstate.Phase, detail, nextAction, evidence string, checkedAt time.Time) LabReadiness {
	if machineClass == "" {
		machineClass = "gpu-server"
	}
	return LabReadiness{State: linkstate.New(machineClass, phase, detail, nextAction, evidence, checkedAt)}
}

// IndeterminateLabReadiness is the fail-safe "no usable signal" record: WAITING /
// indeterminate, so it never admits dispatch but still names a next step.
func IndeterminateLabReadiness(machineClass, nextAction, evidence string, checkedAt time.Time) LabReadiness {
	if machineClass == "" {
		machineClass = "gpu-server"
	}
	return LabReadiness{State: linkstate.Indeterminate(machineClass, nextAction, evidence, checkedAt)}
}

// labWire is the decode superset: the native fak.link_state/v1 fields PLUS the
// legacy fak.lab_readiness/v1 fields (status, machine_class, admit_lab_dispatch),
// so a record written under EITHER schema decodes without tripping
// DisallowUnknownFields — while any truly foreign/private field is still refused.
type labWire struct {
	Schema string `json:"schema"`
	// native fak.link_state/v1 fields
	Subject       string          `json:"subject"`
	Phase         linkstate.Phase `json:"phase"`
	Detail        string          `json:"detail"`
	AdmitDispatch bool            `json:"admit_dispatch"`
	// legacy fak.lab_readiness/v1 fields (folded onto a phase on read)
	MachineClass     string `json:"machine_class"`
	Status           string `json:"status"`
	AdmitLabDispatch bool   `json:"admit_lab_dispatch"`
	// shared
	CheckedAt  string           `json:"checked_at"`
	NextAction string           `json:"next_action"`
	Evidence   string           `json:"evidence"`
	Commands   *labWireCommands `json:"commands,omitempty"`
}

// labWireCommands accepts BOTH the new phase-aligned command keys and the legacy
// keys, so a legacy record's `commands` object does not trip DisallowUnknownFields.
// Legacy hints are dropped on read (re-stamped with the public defaults at display).
type labWireCommands struct {
	// new phase-aligned keys
	MarkClear   string `json:"mark_clear,omitempty"`
	MarkWaiting string `json:"mark_waiting,omitempty"`
	MarkWorking string `json:"mark_working,omitempty"`
	// legacy keys (accepted, then dropped)
	MarkReady               string `json:"mark_ready,omitempty"`
	MarkWaitPrivateRecovery string `json:"mark_wait_private_recovery,omitempty"`
}

// LoadLabReadiness reads a readiness record in EITHER schema, folds a legacy
// five-state status onto a phase (linkstate.Coarsen), rejects unknown/private
// fields, and re-derives admit_dispatch from the phase — never trusting the file.
// This is the safe-rollover path: the uncommittable private bridge may still emit
// legacy records until its local mirror is updated, so the public reader must
// accept them for one retention cycle. Writes always use the new schema.
func LoadLabReadiness(r io.Reader) (LabReadiness, error) {
	var w labWire
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&w); err != nil {
		return LabReadiness{}, err
	}
	subject := w.Subject
	if subject == "" {
		subject = w.MachineClass
	}
	if subject == "" {
		subject = "gpu-server"
	}
	legacy := w.Schema == LabReadinessSchema || w.Phase == ""
	phase, detail := w.Phase, w.Detail
	if legacy {
		// Legacy (or phase-less) record: fold the old status onto a phase+detail.
		phase, detail = linkstate.Coarsen(w.Status)
	} else if detail == "" {
		detail = linkstate.DefaultDetail(phase)
	}
	// Carry only the new-key command hints, and only from a native record; a legacy
	// record's stale hints are dropped (they are re-stamped with the public defaults).
	var cmds *LabReadinessCommands
	if !legacy && w.Commands != nil {
		cmds = &LabReadinessCommands{
			MarkClear:   w.Commands.MarkClear,
			MarkWaiting: w.Commands.MarkWaiting,
			MarkWorking: w.Commands.MarkWorking,
		}
	}
	rec := LabReadiness{
		State: linkstate.State{
			Schema:        linkstate.Schema,
			Subject:       subject,
			CheckedAt:     w.CheckedAt,
			Phase:         phase,
			Detail:        detail,
			NextAction:    w.NextAction,
			Evidence:      w.Evidence,
			AdmitDispatch: phase == linkstate.Clear, // derived, never trusted from the file
		},
		Commands: cmds,
	}
	if probs := rec.Validate(); len(probs) > 0 {
		return LabReadiness{}, fmt.Errorf("%s", strings.Join(probs, "; "))
	}
	return rec, nil
}

// Validate enforces the embedded linkstate contract plus the lab-specific rule
// that any command hint must exact-match the public default (no private command
// or token may ride along).
func (r LabReadiness) Validate() []string {
	probs := r.State.Validate()
	if r.Commands != nil {
		want := DefaultLabReadinessCommands()
		if r.Commands.MarkClear != "" && r.Commands.MarkClear != want.MarkClear {
			probs = append(probs, "commands.mark_clear must be the default public readiness command")
		}
		if r.Commands.MarkWaiting != "" && r.Commands.MarkWaiting != want.MarkWaiting {
			probs = append(probs, "commands.mark_waiting must be the default public readiness command")
		}
		if r.Commands.MarkWorking != "" && r.Commands.MarkWorking != want.MarkWorking {
			probs = append(probs, "commands.mark_working must be the default public readiness command")
		}
	}
	return probs
}
