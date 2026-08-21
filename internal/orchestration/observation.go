package orchestration

import (
	"path"
	"sort"
	"strings"
)

type ObservationValidityReason string

const (
	ObservationInvalid ObservationValidityReason = "INVALID_OBSERVATION"
	ObservationStale   ObservationValidityReason = "STALE_OBSERVATION"

	ObservationStaleGuidance = "rerun the observer against the current state epoch or abstain from reconciliation"
)

// ObservationSnapshot binds a read-only child result to the state and paths it
// observed. StateEpoch is deliberately opaque; the caller drains its change
// feed from the corresponding cursor before asking for a validity decision.
type ObservationSnapshot struct {
	ID         string   `json:"id"`
	StateEpoch string   `json:"state_epoch"`
	ReadSet    []string `json:"read_set"`
}

// ObservationChange is one post-snapshot event projected from the shared change
// feed. ChangedPaths uses repository-relative file or directory paths.
type ObservationChange struct {
	Sequence     uint64   `json:"sequence"`
	StateEpoch   string   `json:"state_epoch,omitempty"`
	ChangedPaths []string `json:"changed_paths"`
}

// ObservationValidityDecision is the reconciliation receipt. A stale decision
// carries only the events and paths that intersected the declared read set.
type ObservationValidityDecision struct {
	Current             bool                      `json:"current"`
	ObservationID       string                    `json:"observation_id"`
	StateEpoch          string                    `json:"state_epoch"`
	ReadSet             []string                  `json:"read_set"`
	Reason              ObservationValidityReason `json:"reason,omitempty"`
	InvalidatingChanges []ObservationChange       `json:"invalidating_changes,omitempty"`
	Guidance            string                    `json:"guidance,omitempty"`
}

// DecideObservationValidity accepts an observation when no post-start change
// intersects its read set. It is pure: feed draining and epoch-to-cursor mapping
// stay with the caller that owns those stores.
func DecideObservationValidity(observation ObservationSnapshot, postStart []ObservationChange) ObservationValidityDecision {
	observation.ID = strings.TrimSpace(observation.ID)
	observation.StateEpoch = strings.TrimSpace(observation.StateEpoch)
	observation.ReadSet = normalizeObservationPaths(observation.ReadSet)
	decision := ObservationValidityDecision{
		Current:       true,
		ObservationID: observation.ID,
		StateEpoch:    observation.StateEpoch,
		ReadSet:       append([]string(nil), observation.ReadSet...),
	}

	if observation.ID == "" || observation.StateEpoch == "" || len(observation.ReadSet) == 0 {
		decision.Current = false
		decision.Reason = ObservationInvalid
		decision.Guidance = "rerun the observer with a bound state epoch and read set or abstain from reconciliation"
		return decision
	}

	for _, change := range postStart {
		relevant := relevantObservationPaths(observation.ReadSet, change.ChangedPaths)
		if len(relevant) == 0 {
			continue
		}
		decision.InvalidatingChanges = append(decision.InvalidatingChanges, ObservationChange{
			Sequence:     change.Sequence,
			StateEpoch:   strings.TrimSpace(change.StateEpoch),
			ChangedPaths: relevant,
		})
	}
	if len(decision.InvalidatingChanges) == 0 {
		return decision
	}

	sort.Slice(decision.InvalidatingChanges, func(i, j int) bool {
		a, b := decision.InvalidatingChanges[i], decision.InvalidatingChanges[j]
		if a.Sequence != b.Sequence {
			return a.Sequence < b.Sequence
		}
		if a.StateEpoch != b.StateEpoch {
			return a.StateEpoch < b.StateEpoch
		}
		return strings.Join(a.ChangedPaths, "\x00") < strings.Join(b.ChangedPaths, "\x00")
	})
	decision.Current = false
	decision.Reason = ObservationStale
	decision.Guidance = ObservationStaleGuidance
	return decision
}

func relevantObservationPaths(readSet, changed []string) []string {
	changed = normalizeObservationPaths(changed)
	relevant := make([]string, 0, len(changed))
	for _, candidate := range changed {
		for _, read := range readSet {
			if observationTreeContains(read, candidate) || observationTreeContains(candidate, read) {
				relevant = append(relevant, candidate)
				break
			}
		}
	}
	return relevant
}

func normalizeObservationPaths(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if value = normalizeObservationPath(value); value != "" {
			seen[value] = struct{}{}
		}
	}
	out := make([]string, 0, len(seen))
	for value := range seen {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func normalizeObservationPath(value string) string {
	value = strings.TrimSpace(strings.ReplaceAll(value, `\`, "/"))
	for strings.HasPrefix(value, "./") {
		value = value[2:]
	}
	value = strings.TrimRight(value, "/")
	for strings.HasSuffix(value, "/**") || strings.HasSuffix(value, "/*") {
		value = strings.TrimSuffix(strings.TrimSuffix(value, "/**"), "/*")
		value = strings.TrimRight(value, "/")
	}
	if value == "" || value == "*" || value == "**" || strings.HasPrefix(value, "/") || strings.ContainsRune(value, 0) {
		return ""
	}
	value = path.Clean(value)
	if value == "." || value == ".." || strings.HasPrefix(value, "../") {
		return ""
	}
	return value
}

func observationTreeContains(tree, candidate string) bool {
	return candidate == tree || strings.HasPrefix(candidate, tree+"/")
}
