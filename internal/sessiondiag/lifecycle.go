package sessiondiag

import (
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

func observedRegistrationProcesses(processes []ProcessEvidence) []sessionregistry.ObservedProcess {
	var out []sessionregistry.ObservedProcess
	for _, p := range processes {
		if p.PID <= 0 || p.StartedAt.IsZero() || !isAgentProcess(p) {
			continue
		}
		out = append(out, sessionregistry.ObservedProcess{PID: p.PID, ProcessStartedAt: p.StartedAt})
	}
	return out
}

func sessionIdentity(s SessionRecord) string {
	if s.Thread != nil && s.Thread.ID != "" {
		return s.Thread.ID
	}
	return strings.TrimPrefix(s.RecordID, "thread:")
}

func buildCleanupActions(sessions []SessionRecord, edges []SpawnEdgeRecord, reconciliations []sessionregistry.Reconciliation) []CleanupAction {
	var out []CleanupAction
	for _, s := range sessions {
		if s.WriterLock != nil && s.Health == HealthFailedButLocked && len(s.ProcessTrees) == 0 {
			out = append(out, CleanupAction{Artifact: "writer_lock", Identity: sessionIdentity(s), Action: "remove", Reason: "authoritative failed turn and no matching process tree"})
		}
		if s.GuardReceipt != nil && len(s.ProcessTrees) == 0 {
			out = append(out, CleanupAction{Artifact: "guard_receipt", Identity: sessionIdentity(s), Action: "retain_receipt", Reason: "launch receipt has no joined process identity; it is not liveness"})
		}
	}
	for _, e := range edges {
		if e.Status != "open" {
			continue
		}
		if e.Parent.State == EndpointActive && e.Child.State == EndpointActive {
			continue
		}
		if e.Parent.State == EndpointUnknown && e.Child.State == EndpointUnknown {
			continue
		}
		out = append(out, CleanupAction{Artifact: "spawn_edge", Identity: e.Parent.ThreadID + "->" + e.Child.ThreadID, Action: "terminalize_unknown", Reason: "open edge lacks a live joined parent/child process pair"})
	}
	for _, r := range reconciliations {
		out = append(out, CleanupAction{Artifact: "registration", Identity: r.RegistrationID, RegistrationID: r.RegistrationID, Action: "append_" + string(r.To), Reason: r.Reason})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Artifact != out[j].Artifact {
			return out[i].Artifact < out[j].Artifact
		}
		return out[i].Identity < out[j].Identity
	})
	return out
}
