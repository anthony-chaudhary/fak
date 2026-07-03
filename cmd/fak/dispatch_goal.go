package main

import (
	"fmt"
	"strings"
)

const (
	dispatchGoalProfileThroughput   = "throughput"
	dispatchGoalProfileHighPriority = "high-priority"
)

// normalizeDispatchGoal turns operator-facing goal flags into two pieces of data:
// a durable loop identity (goal) and the picker policy (profile). No goal flag keeps
// the historical loop id; naming a goal makes parallel background loops visible and
// lease-auditable as distinct intent.
func normalizeDispatchGoal(rawGoal, rawProfile string) (goal string, profile string, err error) {
	goal = strings.TrimSpace(rawGoal)
	profile = strings.TrimSpace(rawProfile)
	if profile == "" {
		if p, ok := dispatchGoalProfileAlias(goal); ok {
			profile = p
			if goal != "" {
				goal = p
			}
		}
	}
	if profile == "" {
		profile = dispatchGoalProfileThroughput
	}
	if p, ok := dispatchGoalProfileAlias(profile); ok {
		profile = p
	} else {
		return "", "", fmt.Errorf("dispatch goal profile %q is unknown (want throughput|high-priority)", rawProfile)
	}
	if goal == "" && strings.TrimSpace(rawProfile) != "" {
		goal = profile
	}
	if goal != "" {
		if token := cleanDispatchLeaseToken(goal); token == "" || token == "unknown" {
			return "", "", fmt.Errorf("dispatch goal %q has no usable identity", rawGoal)
		}
	}
	return goal, profile, nil
}

func dispatchGoalProfileAlias(raw string) (string, bool) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "default", "open", "open tasks", "open-tasks", "open_tasks", "backlog", "throughput":
		return dispatchGoalProfileThroughput, true
	case "priority", "prio", "high priority", "high-priority", "high_priority", "highpriority", "p0-p1", "p0", "urgent":
		return dispatchGoalProfileHighPriority, true
	default:
		return "", false
	}
}

func dispatchGoalToken(goal string) string {
	goal = strings.TrimSpace(goal)
	if goal == "" {
		return ""
	}
	token := cleanDispatchLeaseToken(goal)
	if token == "unknown" {
		return ""
	}
	return token
}

func dispatchLeaseHolderForGoal(goal string) string {
	holder := dispatchLeaseHolder()
	if token := dispatchGoalToken(goal); token != "" {
		return holder + " goal=" + token
	}
	return holder
}

func dispatchTickLoopID(backend, goal string) string {
	id := "issue-resolve-dispatch/" + firstString(strings.TrimSpace(backend), "claude")
	if token := dispatchGoalToken(goal); token != "" {
		id += "/" + token
	}
	return id
}
