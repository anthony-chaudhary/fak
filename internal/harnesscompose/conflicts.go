package harnesscompose

import (
	"errors"
	"fmt"
	"reflect"
	"sort"
)

// ErrArbitrationRequired reports that incompatible verified receipts cannot be
// composed without an explicitly bounded arbitration dispatch.
var ErrArbitrationRequired = errors.New("harnesscompose: receipt arbitration required")

// ReceiptConflict preserves the evidence provenance for incompatible child
// recommendations. Candidates never contain child transcripts.
type ReceiptConflict struct {
	Kind       ReceiptKind
	ID         string
	Candidates []ProfileReceipt
}

// ArbitrationRequest is one bounded unit of conflict resolution.
type ArbitrationRequest struct {
	Conflict ReceiptConflict
	Ordinal  int
	Limit    int
}

// ReceiptArbiter resolves one conflict. The returned receipt must identify the
// same decision and carry accepted verification evidence.
type ReceiptArbiter func(ArbitrationRequest) (ProfileReceipt, error)

// ComposeReceiptsWithArbitration composes compatible receipts directly and
// dispatches at most maxRequests conflict-resolution requests. A zero or
// exhausted bound refuses rather than silently selecting a writer.
func ComposeReceiptsWithArbitration(receipts []ProfileReceipt, maxRequests int, arbitrate ReceiptArbiter) (Result, []ReceiptConflict, error) {
	groups := make(map[string][]ProfileReceipt, len(receipts))
	keys := make([]string, 0, len(receipts))
	for _, receipt := range receipts {
		key := string(receipt.Kind) + "\x00" + receipt.ID
		if _, ok := groups[key]; !ok {
			keys = append(keys, key)
		}
		groups[key] = append(groups[key], receipt)
	}
	sort.Strings(keys)

	resolved := make([]ProfileReceipt, 0, len(keys))
	conflicts := make([]ReceiptConflict, 0)
	for _, key := range keys {
		group := groups[key]
		if len(group) == 1 {
			resolved = append(resolved, group[0])
			continue
		}
		sort.Slice(group, func(i, j int) bool {
			if group[i].EvidenceRef != group[j].EvidenceRef {
				return group[i].EvidenceRef < group[j].EvidenceRef
			}
			return fmt.Sprintf("%#v", group[i].Asset) < fmt.Sprintf("%#v", group[j].Asset)
		})
		compatible := true
		for i := 1; i < len(group); i++ {
			if !reflect.DeepEqual(group[0].Asset, group[i].Asset) {
				compatible = false
				break
			}
		}
		if compatible {
			// Equivalent recommendations need no arbiter; retain deterministic
			// provenance instead of treating arrival order as authoritative.
			resolved = append(resolved, group[0])
			continue
		}
		conflicts = append(conflicts, ReceiptConflict{Kind: group[0].Kind, ID: group[0].ID, Candidates: group})
	}

	if len(conflicts) > maxRequests || (len(conflicts) > 0 && arbitrate == nil) {
		return Result{}, conflicts, fmt.Errorf("%w: %d conflict(s), request bound %d", ErrArbitrationRequired, len(conflicts), maxRequests)
	}
	for i, conflict := range conflicts {
		selected, err := arbitrate(ArbitrationRequest{Conflict: conflict, Ordinal: i + 1, Limit: maxRequests})
		if err != nil {
			return Result{}, conflicts, fmt.Errorf("%w: conflict %s/%s: %v", ErrArbitrationRequired, conflict.Kind, conflict.ID, err)
		}
		if selected.Kind != conflict.Kind || selected.ID != conflict.ID || !selected.Verified || selected.EvidenceRef == "" {
			return Result{}, conflicts, fmt.Errorf("%w: arbiter returned invalid selection for %s/%s", ErrInvalidReceipt, conflict.Kind, conflict.ID)
		}
		resolved = append(resolved, selected)
	}
	result, err := ComposeReceipts(resolved)
	return result, conflicts, err
}
