package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/guard"
	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

// registerGuardChildOriginTask seeds the guarded child at the moment its durable
// evidence locations are known. The endpoint remains opt-in: with FAK_TASKMGR off this
// is a zero-work no-op, matching serveTasksSnapshot's existing contract.
func registerGuardChildOriginTask(traceID, agent, policyPath, transcriptPath, budgetEnvelope, stopsLedger string) {
	if !taskManagerEnabled() || strings.TrimSpace(traceID) == "" {
		return
	}
	refs := guardChildOriginEvidence(policyPath, transcriptPath, budgetEnvelope, stopsLedger)
	if len(refs) == 0 {
		return
	}
	taskID := "guard_" + strings.TrimSpace(traceID)
	title := "guarded child " + strings.TrimSpace(agent)
	if strings.TrimSpace(agent) == "" {
		title = "guarded child"
	}
	_, _ = processTaskManager().StartTask(taskmgr.TaskSpec{
		TaskID:       taskID,
		Title:        title,
		Labels:       map[string]string{"origin": "fak_guard", "trace_id": strings.TrimSpace(traceID)},
		EvidenceRefs: refs,
	})
}

func guardChildOriginEvidence(policyPath, transcriptPath, budgetEnvelope, stopsLedger string) []taskmgr.EvidenceRef {
	candidates := []struct {
		label string
		path  string
	}{
		{"policy", policyPath},
		{"transcript", transcriptPath},
		{"budget_envelope", budgetEnvelope},
		{"stop_hook_ledger", stopsLedger},
	}
	refs := make([]taskmgr.EvidenceRef, 0, len(candidates))
	for _, candidate := range candidates {
		// Origin evidence is a durable location contract: guard.EnsureOriginEvidence
		// creates an empty evidence file when the producer has not appended its first
		// row yet, so PathWitness verifies the location rather than marking a valid
		// pre-launch task refused.
		path, ok := guard.EnsureOriginEvidence(candidate.path)
		if !ok {
			continue
		}
		refs = append(refs, taskmgr.EvidenceRef{
			Kind: taskmgr.PathRefKind,
			Ref:  path,
			Note: fmt.Sprintf("guard-origin:%s", candidate.label),
		})
	}
	return refs
}

// guardPolicyOriginEvidencePath names the durable policy evidence location for the
// launch. It writes nothing: the location is proven with the same 0-byte placeholder
// as its siblings above, instead of copying the embedded 34KB capability floor per
// trace (#6093).
func guardPolicyOriginEvidencePath(traceID, explicitPath string) string {
	return guard.PolicyOriginEvidencePath(repoRoot(), traceID, explicitPath)
}
func writeGuardBudgetEnvelopeEvidence(traceID string, contextTokens int, maxDuration string) string {
	if strings.TrimSpace(traceID) == "" {
		return ""
	}
	root := repoRoot()
	if root == "" {
		return ""
	}
	path := filepath.Join(root, ".fak", "guard-origin", strings.TrimSpace(traceID)+"-budget.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return ""
	}
	b, err := json.MarshalIndent(map[string]any{
		"schema":              "fak.guard-budget-origin/1",
		"trace_id":            strings.TrimSpace(traceID),
		"context_tokens_left": contextTokens,
		"max_duration":        strings.TrimSpace(maxDuration),
	}, "", "  ")
	if err != nil {
		return ""
	}
	b = append(b, '\n')
	if err := os.WriteFile(path, b, 0o600); err != nil {
		return ""
	}
	return path
}
