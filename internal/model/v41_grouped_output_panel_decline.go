package model

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
)

// v41GroupedPanelDeclines counts why the batched prefill grouped-output panel
// fell back to the per-row path (#13864). Every decline is counted; with
// FAK_VULKAN_DISPATCH_PROFILE=1 each decline (at most one per layer per
// prefill) is printed with its running count next to the dispatch profile.
var v41GroupedPanelDeclines = struct {
	sync.Mutex
	counts map[string]int
}{counts: map[string]int{}}

func v41GroupedPanelDecline(layer, rows int, reason string) v41DenseProjectionOutcome {
	v41GroupedPanelDeclines.Lock()
	v41GroupedPanelDeclines.counts[reason]++
	count := v41GroupedPanelDeclines.counts[reason]
	v41GroupedPanelDeclines.Unlock()
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") == "1" {
		fmt.Fprintf(os.Stderr, "v41 grouped_output panel declined reason=%s layer=%d rows=%d count=%d\n", reason, layer, rows, count)
	}
	return v41ProjectionDeclined
}

// v41GroupedPanelDeclineSummary renders "reason=count" pairs in reason order.
func v41GroupedPanelDeclineSummary() string {
	v41GroupedPanelDeclines.Lock()
	defer v41GroupedPanelDeclines.Unlock()
	parts := make([]string, 0, len(v41GroupedPanelDeclines.counts))
	for r, c := range v41GroupedPanelDeclines.counts {
		parts = append(parts, fmt.Sprintf("%s=%d", r, c))
	}
	sort.Strings(parts)
	return strings.Join(parts, ",")
}

func v41GroupedPanelDeclineReset() {
	v41GroupedPanelDeclines.Lock()
	v41GroupedPanelDeclines.counts = map[string]int{}
	v41GroupedPanelDeclines.Unlock()
}
func v41GroupedPanelCallerReason(full, state, owner, sameModel bool) string {
	switch {
	case !full:
		return "caller_not_full_profile"
	case !state:
		return "caller_no_state"
	case !owner:
		return "caller_no_owner"
	case !sameModel:
		return "caller_owner_model_mismatch"
	}
	return "caller_other"
}
