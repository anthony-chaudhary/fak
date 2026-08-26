package harnessweb

import (
	"context"
	"sort"
	"strings"
	"time"
)

// LocalDOSLease is the operator-facing identity of one live DOS lane lease.
type LocalDOSLease struct {
	Lane   string `json:"lane"`
	LoopID string `json:"loop_id,omitempty"`
}

// LocalWorkSource supplies bounded, live local-work identities to the harness UI.
type LocalWorkSource interface {
	LiveIntentKeys(context.Context, string, time.Time) ([]string, error)
	LiveDOSLeases(context.Context, string, time.Time) ([]LocalDOSLease, error)
}

type localWorkSet struct {
	Active int      `json:"active"`
	IDs    []string `json:"ids"`
	Error  string   `json:"error,omitempty"`
}

type localWorkOverview struct {
	IssueIntents localWorkSet `json:"issue_intents"`
	DOSLeases    localWorkSet `json:"dos_leases"`
}

func readLocalWorkOverview(ctx context.Context, source LocalWorkSource, root string, now time.Time) localWorkOverview {
	if source == nil || strings.TrimSpace(root) == "" {
		return localWorkOverview{IssueIntents: localWorkSet{IDs: []string{}}, DOSLeases: localWorkSet{IDs: []string{}}}
	}
	intents, intentErr := source.LiveIntentKeys(ctx, root, now)
	leases, leaseErr := source.LiveDOSLeases(ctx, root, now)
	intentIDs := boundedLocalWorkIDs(intents)
	leaseKeys := make([]string, 0, len(leases))
	for _, lease := range leases {
		key := strings.TrimSpace(lease.Lane)
		if loop := strings.TrimSpace(lease.LoopID); loop != "" {
			key += " (" + loop + ")"
		}
		leaseKeys = append(leaseKeys, key)
	}
	leaseIDs := boundedLocalWorkIDs(leaseKeys)
	result := localWorkOverview{
		IssueIntents: localWorkSet{Active: len(intentIDs), IDs: intentIDs},
		DOSLeases:    localWorkSet{Active: len(leaseIDs), IDs: leaseIDs},
	}
	if intentErr != nil {
		result.IssueIntents.Error = "unavailable"
	}
	if leaseErr != nil {
		result.DOSLeases.Error = "unavailable"
	}
	return result
}

func boundedLocalWorkIDs(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	ids := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if len(value) > maxLocalWorkIDBytes {
			value = value[:maxLocalWorkIDBytes]
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		ids = append(ids, value)
	}
	sort.Strings(ids)
	if len(ids) > maxLocalWorkIDs {
		ids = ids[:maxLocalWorkIDs]
	}
	return ids
}
