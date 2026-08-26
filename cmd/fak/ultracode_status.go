package main

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/orchestration"
)

const ultracodeNodeUnobserved = "unobserved"

// ultracodeChildContextReceipt carries only parent-derived identifiers and a
// digest. Raw child prompts have no representation in the status input.
type ultracodeChildContextReceipt struct {
	NodeID        string `json:"node_id"`
	ParentID      string `json:"parent_id,omitempty"`
	ContextDigest string `json:"context_digest"`
	StateEpoch    string `json:"state_epoch"`
}

type ultracodeNodeLeaseReceipt struct {
	NodeID  string `json:"node_id"`
	Verdict string `json:"verdict"`
	WaitMS  *int64 `json:"wait_ms,omitempty"`
	HoldMS  *int64 `json:"hold_ms,omitempty"`
}

type ultracodeNodeArtifactReceipt struct {
	NodeID string   `json:"node_id"`
	Refs   []string `json:"refs"`
}

type ultracodeNodeTerminalReceipt struct {
	NodeID string `json:"node_id"`
	State  string `json:"state"`
}

type ultracodeNodeGraphReceipt struct {
	Plan          *orchestration.WorkflowPlan            `json:"plan,omitempty"`
	Contexts      []ultracodeChildContextReceipt         `json:"child_receipts,omitempty"`
	Leases        []ultracodeNodeLeaseReceipt            `json:"leases,omitempty"`
	Artifacts     []ultracodeNodeArtifactReceipt         `json:"artifacts,omitempty"`
	Terminals     []ultracodeNodeTerminalReceipt         `json:"terminals,omitempty"`
	Successors    []orchestration.EffectSuccessorReceipt `json:"successors,omitempty"`
	EffectWitness []orchestration.EffectReceipt          `json:"effect_witnesses,omitempty"`
}

type ultracodeNodeStatus struct {
	NodeID         string   `json:"node_id"`
	ParentID       string   `json:"parent_id"`
	Dependencies   []string `json:"dependencies"`
	Role           string   `json:"role"`
	Access         string   `json:"access"`
	ContextDigest  string   `json:"context_digest"`
	StateEpoch     string   `json:"state_epoch"`
	LeaseVerdict   string   `json:"lease_verdict"`
	LeaseWaitMS    *int64   `json:"lease_wait_ms"`
	LeaseHoldMS    *int64   `json:"lease_hold_ms"`
	WorkerState    string   `json:"worker_state"`
	ReservedTokens *int64   `json:"reserved_tokens"`
	UsedTokens     *int64   `json:"used_tokens"`
	ArtifactRefs   []string `json:"artifact_refs"`
	EffectState    string   `json:"effect_state"`
	WitnessState   string   `json:"witness_state"`
	ReconcileState string   `json:"reconcile_state"`
	TerminalState  string   `json:"terminal_state"`
	Attention      string   `json:"attention_reason"`
}

func projectUltracodeNodes(receipt codexOrchestrationLaunchReceipt, workers []orchestrationWorkerStatus) []ultracodeNodeStatus {
	if receipt.Graph == nil || receipt.Graph.Plan == nil {
		return []ultracodeNodeStatus{}
	}
	graph := receipt.Graph
	workerByID := make(map[string]orchestrationWorkerStatus, len(workers))
	for _, worker := range workers {
		workerByID[worker.RoleID] = worker
	}
	contextByID := make(map[string]ultracodeChildContextReceipt, len(graph.Contexts))
	for _, context := range graph.Contexts {
		contextByID[context.NodeID] = context
	}
	leaseByID := make(map[string]ultracodeNodeLeaseReceipt, len(graph.Leases))
	for _, lease := range graph.Leases {
		leaseByID[lease.NodeID] = lease
	}
	artifactsByID := make(map[string][]string, len(graph.Artifacts))
	for _, artifact := range graph.Artifacts {
		artifactsByID[artifact.NodeID] = append(artifactsByID[artifact.NodeID], artifact.Refs...)
	}
	terminalByID := make(map[string]string, len(graph.Terminals))
	for _, terminal := range graph.Terminals {
		terminalByID[terminal.NodeID] = terminal.State
	}
	successorByID := make(map[string]orchestration.EffectSuccessorReceipt, len(graph.Successors))
	for _, successor := range graph.Successors {
		successorByID[successor.NodeID] = successor
	}
	witnessByID := make(map[string]orchestration.EffectReceipt, len(graph.EffectWitness))
	for _, witness := range graph.EffectWitness {
		witnessByID[witness.ChildID] = witness
	}
	budgetByID := make(map[string]orchestration.UltracodeChildBudget, len(receipt.Budget.Children))
	for _, budget := range receipt.Budget.Children {
		budgetByID[budget.ChildID] = budget
	}
	dependencies := make(map[string][]string)
	for _, edge := range graph.Plan.DAG {
		dependencies[edge.To] = append(dependencies[edge.To], edge.From)
	}

	nodes := make([]ultracodeNodeStatus, 0, len(graph.Plan.Roles))
	for _, role := range graph.Plan.Roles {
		if role.ID == "lead" {
			continue
		}
		node := ultracodeNodeStatus{
			NodeID: role.ID, ParentID: ultracodeNodeUnobserved,
			Dependencies: append([]string(nil), dependencies[role.ID]...), Role: role.Purpose,
			Access: string(role.Access.Mode), ContextDigest: ultracodeNodeUnobserved,
			StateEpoch: ultracodeNodeUnobserved, LeaseVerdict: ultracodeNodeUnobserved,
			WorkerState: ultracodeNodeUnobserved, ArtifactRefs: []string{}, EffectState: ultracodeNodeUnobserved,
			WitnessState: ultracodeNodeUnobserved, ReconcileState: ultracodeNodeUnobserved,
			TerminalState: ultracodeNodeUnobserved, Attention: ultracodeNodeUnobserved,
		}
		sort.Strings(node.Dependencies)
		if context, ok := contextByID[role.ID]; ok {
			node.ParentID, node.ContextDigest, node.StateEpoch = observedOr(context.ParentID), observedOr(context.ContextDigest), observedOr(context.StateEpoch)
		}
		if lease, ok := leaseByID[role.ID]; ok {
			node.LeaseVerdict, node.LeaseWaitMS, node.LeaseHoldMS = observedOr(lease.Verdict), lease.WaitMS, lease.HoldMS
		}
		if worker, ok := workerByID[role.ID]; ok {
			node.WorkerState = observedOr(worker.State)
		}
		if terminal, ok := terminalByID[role.ID]; ok {
			node.TerminalState = observedOr(terminal)
		}
		if budget, ok := budgetByID[role.ID]; ok {
			reserved, used := budget.ReservedTokens, budget.ProviderTokens
			node.ReservedTokens = &reserved
			if budget.Covered {
				node.UsedTokens = &used
			}
		}
		node.ArtifactRefs = append(node.ArtifactRefs, artifactsByID[role.ID]...)
		sort.Strings(node.ArtifactRefs)
		if successor, ok := successorByID[role.ID]; ok {
			node.EffectState = "admitted"
			if node.LeaseVerdict == ultracodeNodeUnobserved && successor.LeaseID != "" {
				node.LeaseVerdict = "admitted"
			}
		}
		if witness, ok := witnessByID[role.ID]; ok {
			node.EffectState = string(witness.State)
			node.ReconcileState = string(witness.Reconciliation)
			node.WitnessState = observedOr(witness.Witness.AuthorityID)
		}
		node.Attention = ultracodeNodeAttention(node)
		nodes = append(nodes, node)
	}
	sort.Slice(nodes, func(i, j int) bool { return nodes[i].NodeID < nodes[j].NodeID })
	return nodes
}

func observedOr(value string) string {
	if value == "" {
		return ultracodeNodeUnobserved
	}
	return value
}

func ultracodeNodeAttention(node ultracodeNodeStatus) string {
	switch {
	case node.LeaseVerdict == "contended" || node.LeaseVerdict == "denied":
		return "lease_" + node.LeaseVerdict
	case node.EffectState == string(orchestration.EffectFailed):
		return "effect_failed"
	case node.ReconcileState == string(orchestration.EffectDiverged):
		return "reconcile_diverged"
	case node.TerminalState == "blocked":
		return "terminal_blocked"
	default:
		return ultracodeNodeUnobserved
	}
}

func formatUltracodeNodeBudget(node ultracodeNodeStatus) string {
	if node.ReservedTokens == nil {
		return ultracodeNodeUnobserved
	}
	if node.UsedTokens == nil {
		return fmt.Sprintf("unobserved/%d", *node.ReservedTokens)
	}
	return fmt.Sprintf("%d/%d", *node.UsedTokens, *node.ReservedTokens)
}
