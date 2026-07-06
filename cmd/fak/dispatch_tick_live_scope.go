package main

func liveIssueSet(details map[int]dispatchLiveScope) map[int]bool {
	out := map[int]bool{}
	for issue := range details {
		out[issue] = true
	}
	return out
}

func inFlightDuplicateForPick(opts dispatchTickOptions, numbers []int, hasTarget bool, details map[int]dispatchLiveScope) (dispatchLiveScope, bool) {
	if opts.TargetIssue > 0 {
		live, ok := details[opts.TargetIssue]
		return live, ok
	}
	if hasTarget {
		return dispatchLiveScope{}, false
	}
	for _, issue := range numbers {
		if live, ok := details[issue]; ok {
			return live, true
		}
	}
	return dispatchLiveScope{}, false
}

func dispatchLiveScopeMap(live dispatchLiveScope) map[string]any {
	return map[string]any{
		"issue":    live.Issue,
		"lane":     live.Lane,
		"tree":     append([]string(nil), live.Tree...),
		"log":      live.Log,
		"pid":      live.PID,
		"worker":   live.Worker,
		"lease_id": live.LeaseID,
	}
}
