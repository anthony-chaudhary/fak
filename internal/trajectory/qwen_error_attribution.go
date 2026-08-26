package trajectory

// qwenToolErrorAttribution joins parser evidence already used for aggregate
// repeated-failure and mutation-churn totals to an individual tool error.
type qwenToolErrorAttribution struct {
	failureKey     string
	mutationTarget string
}

func applyQwenToolErrorAttribution(events []QwenToolErrorEvent, attributions []qwenToolErrorAttribution, mutationCounts map[string]int) {
	if len(events) != len(attributions) {
		return
	}
	seenFailures := make(map[string]struct{})
	seenMutations := make(map[string]struct{})
	for i := range events {
		attr := attributions[i]
		if attr.failureKey != "" {
			if _, ok := seenFailures[attr.failureKey]; ok {
				events[i].repeatedFailures++
			} else {
				seenFailures[attr.failureKey] = struct{}{}
			}
		}
		if attr.mutationTarget != "" {
			if _, ok := seenMutations[attr.mutationTarget]; !ok {
				seenMutations[attr.mutationTarget] = struct{}{}
				if count := mutationCounts[attr.mutationTarget]; count > 1 {
					events[i].mutationChurn += count - 1
				}
			}
		}
	}
}
