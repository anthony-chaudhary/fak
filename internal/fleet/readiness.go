package fleet

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

const LabReadinessSchema = "fak.lab_readiness/v1"

const (
	LabReadyForDevWork    = "READY_FOR_DEV_WORK"
	LabWaitPrivateRecover = "WAIT_PRIVATE_RECOVERY"
	LabGatewayUnreachable = "GATEWAY_UNREACHABLE"
	LabAuthChannelBlocked = "AUTH_OR_CHANNEL_BLOCKED"
	LabIndeterminate      = "INDETERMINATE"
)

// LabReadiness is the public dispatch gate for lab-backed machine classes. It is
// intentionally generic: private bridge outputs are folded before they enter this
// record, so no host, channel, token, thread id, transcript, or private path is needed.
type LabReadiness struct {
	Schema           string                `json:"schema"`
	MachineClass     string                `json:"machine_class"`
	CheckedAt        string                `json:"checked_at"`
	Status           string                `json:"status"`
	NextAction       string                `json:"next_action"`
	Evidence         string                `json:"evidence"`
	AdmitLabDispatch bool                  `json:"admit_lab_dispatch"`
	Commands         *LabReadinessCommands `json:"commands,omitempty"`
}

type LabReadinessCommands struct {
	MarkReady               string `json:"mark_ready,omitempty"`
	MarkWaitPrivateRecovery string `json:"mark_wait_private_recovery,omitempty"`
}

func DefaultLabReadinessCommands() LabReadinessCommands {
	return LabReadinessCommands{
		MarkReady:               "fak lab readiness --status READY_FOR_DEV_WORK --write-default --json",
		MarkWaitPrivateRecovery: "fak lab readiness --status WAIT_PRIVATE_RECOVERY --write-default --json",
	}
}

func NewLabReadiness(machineClass, status, nextAction, evidence string, checkedAt time.Time) LabReadiness {
	if machineClass == "" {
		machineClass = "gpu-server"
	}
	if checkedAt.IsZero() {
		checkedAt = time.Now()
	}
	if status == "" {
		status = LabIndeterminate
	}
	if nextAction == "" {
		nextAction = defaultLabNextAction(status)
	}
	if evidence == "" {
		evidence = defaultLabEvidence(status)
	}
	return LabReadiness{
		Schema:           LabReadinessSchema,
		MachineClass:     machineClass,
		CheckedAt:        checkedAt.UTC().Format(time.RFC3339),
		Status:           status,
		NextAction:       nextAction,
		Evidence:         evidence,
		AdmitLabDispatch: status == LabReadyForDevWork,
	}
}

func IndeterminateLabReadiness(machineClass, nextAction, evidence string, checkedAt time.Time) LabReadiness {
	return NewLabReadiness(machineClass, LabIndeterminate, nextAction, evidence, checkedAt)
}

func LoadLabReadiness(r io.Reader) (LabReadiness, error) {
	var out LabReadiness
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&out); err != nil {
		return LabReadiness{}, err
	}
	if out.Schema == "" {
		out.Schema = LabReadinessSchema
	}
	if out.MachineClass == "" {
		out.MachineClass = "gpu-server"
	}
	out.AdmitLabDispatch = out.Status == LabReadyForDevWork
	if probs := out.Validate(); len(probs) > 0 {
		return LabReadiness{}, fmt.Errorf("%s", strings.Join(probs, "; "))
	}
	return out, nil
}

func (r LabReadiness) Validate() []string {
	var probs []string
	if r.Schema != "" && r.Schema != LabReadinessSchema {
		probs = append(probs, fmt.Sprintf("unsupported schema %q (want %s)", r.Schema, LabReadinessSchema))
	}
	if !knownLabReadinessStatus(r.Status) {
		probs = append(probs, fmt.Sprintf("status %q is not in the closed lab readiness vocabulary", r.Status))
	}
	for field, value := range map[string]string{
		"machine_class": r.MachineClass,
		"next_action":   r.NextAction,
		"evidence":      r.Evidence,
	} {
		if value == "" {
			probs = append(probs, field+" is required")
		} else if !genericTokenish(value) {
			probs = append(probs, field+" must be a generic token-like value")
		}
	}
	if r.CheckedAt != "" {
		if _, err := time.Parse(time.RFC3339, r.CheckedAt); err != nil {
			probs = append(probs, "checked_at must be RFC3339")
		}
	}
	if r.Commands != nil {
		want := DefaultLabReadinessCommands()
		if r.Commands.MarkReady != "" && r.Commands.MarkReady != want.MarkReady {
			probs = append(probs, "commands.mark_ready must be the default public readiness command")
		}
		if r.Commands.MarkWaitPrivateRecovery != "" && r.Commands.MarkWaitPrivateRecovery != want.MarkWaitPrivateRecovery {
			probs = append(probs, "commands.mark_wait_private_recovery must be the default public readiness command")
		}
	}
	return probs
}

func knownLabReadinessStatus(status string) bool {
	switch status {
	case LabReadyForDevWork, LabWaitPrivateRecover, LabGatewayUnreachable, LabAuthChannelBlocked, LabIndeterminate:
		return true
	default:
		return false
	}
}

func genericTokenish(value string) bool {
	if strings.TrimSpace(value) != value || value == "" {
		return false
	}
	for _, ch := range value {
		if (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' {
			continue
		}
		return false
	}
	return true
}

func defaultLabNextAction(status string) string {
	switch status {
	case LabReadyForDevWork:
		return "admit-lab-backed-dispatch"
	case LabWaitPrivateRecover:
		return "confirm-private-control-session"
	case LabGatewayUnreachable:
		return "recover-private-gateway"
	case LabAuthChannelBlocked:
		return "fix-private-auth-or-channel"
	default:
		return "publish-lab-readiness"
	}
}

func defaultLabEvidence(status string) string {
	if status == LabIndeterminate {
		return "no-readiness-record"
	}
	return "scrubbed-private-readback"
}
