package sessionregistry

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"
)

// WIPBindingStatus is the closed vocabulary for joining durable execution
// lineage to an issue. Non-joined values are inventory debt, not permission to
// discard or rewrite a session.
type WIPBindingStatus string

const (
	WIPBindingJoined      WIPBindingStatus = "joined"
	WIPBindingMissing     WIPBindingStatus = "missing"
	WIPBindingAmbiguous   WIPBindingStatus = "ambiguous"
	WIPBindingConflicting WIPBindingStatus = "conflicting"
	WIPBindingStale       WIPBindingStatus = "stale"
)

// WIPIssueIdentity is the stable issue key carried into WIP inventory. Titles
// and task strings are deliberately absent: they are not identity.
type WIPIssueIdentity struct {
	Repository string `json:"repository"`
	Number     int    `json:"number"`
}

func (i WIPIssueIdentity) Key() string { return fmt.Sprintf("%s#%d", i.Repository, i.Number) }

// WIPExecutionBinding describes one root execution and all retry, resume, and
// child attempts below it. Attempt structure never increases logical WIP count.
type WIPExecutionBinding struct {
	RootRegistrationID string            `json:"root_registration_id"`
	Issue              *WIPIssueIdentity `json:"issue,omitempty"`
	RegistrationIDs    []string          `json:"registration_ids"`
	AttemptIDs         []string          `json:"attempt_ids"`
	SessionIDs         []string          `json:"session_ids,omitempty"`
	Status             WIPBindingStatus  `json:"status"`
	Details            []string          `json:"details,omitempty"`
	LatestActivity     time.Time         `json:"latest_activity,omitempty"`
}

// WIPBindingReport is deterministic: bindings are sorted by root registration
// and every identifier/detail slice is sorted and unique.
type WIPBindingReport struct {
	Bindings []WIPExecutionBinding `json:"bindings"`
}

// BuildWIPBindings converts the #6458 registration lineage into issue-bound
// roots. repository supplies the repository for legacy numeric RootIssue
// values. staleBefore is optional; a zero value disables stale classification.
func BuildWIPBindings(rows []Record, repository string, staleBefore time.Time) WIPBindingReport {
	repository = strings.TrimSpace(repository)
	byRoot := make(map[string][]Record)
	for _, row := range rows {
		root := strings.TrimSpace(row.RootRegistrationID)
		if root == "" {
			root = strings.TrimSpace(row.RegistrationID)
		}
		byRoot[root] = append(byRoot[root], row)
	}

	roots := make([]string, 0, len(byRoot))
	for root := range byRoot {
		roots = append(roots, root)
	}
	sort.Strings(roots)

	report := WIPBindingReport{Bindings: make([]WIPExecutionBinding, 0, len(roots))}
	issueRoots := make(map[string][]string)
	for _, root := range roots {
		binding := buildWIPBinding(root, byRoot[root], repository, staleBefore)
		report.Bindings = append(report.Bindings, binding)
		if binding.Issue != nil && binding.Status != WIPBindingConflicting && binding.Status != WIPBindingMissing {
			issueRoots[binding.Issue.Key()] = append(issueRoots[binding.Issue.Key()], root)
		}
	}

	// Two unrelated roots claiming one issue are ambiguous. Linked attempts are
	// already inside one root and therefore never arrive here as extra units.
	for _, rootsForIssue := range issueRoots {
		if len(rootsForIssue) < 2 {
			continue
		}
		sort.Strings(rootsForIssue)
		detail := "issue is claimed by unrelated roots: " + strings.Join(rootsForIssue, ",")
		for i := range report.Bindings {
			if containsString(rootsForIssue, report.Bindings[i].RootRegistrationID) && report.Bindings[i].Status != WIPBindingConflicting {
				report.Bindings[i].Status = WIPBindingAmbiguous
				report.Bindings[i].Details = appendUniqueSorted(report.Bindings[i].Details, detail)
			}
		}
	}
	return report
}

func buildWIPBinding(root string, rows []Record, repository string, staleBefore time.Time) WIPExecutionBinding {
	binding := WIPExecutionBinding{RootRegistrationID: root}
	issues := make(map[string]WIPIssueIdentity)
	invalid := make([]string, 0)
	active := false
	for _, row := range rows {
		binding.RegistrationIDs = append(binding.RegistrationIDs, strings.TrimSpace(row.RegistrationID))
		binding.AttemptIDs = append(binding.AttemptIDs, strings.TrimSpace(row.AttemptID))
		binding.SessionIDs = append(binding.SessionIDs, strings.TrimSpace(row.Identity.SessionID))
		if issue, ok := parseWIPIssue(row.RootIssue, repository); ok {
			issues[issue.Key()] = issue
		} else if strings.TrimSpace(row.RootIssue) != "" {
			invalid = append(invalid, strings.TrimSpace(row.RootIssue))
		}
		activity := row.CreatedAt
		for _, candidate := range []time.Time{row.StartedAt, row.HeartbeatAt, row.TerminalAt} {
			if candidate.After(activity) {
				activity = candidate
			}
		}
		if activity.After(binding.LatestActivity) {
			binding.LatestActivity = activity
		}
		if !isTerminal(row.State) {
			active = true
		}
	}
	binding.RegistrationIDs = compact(binding.RegistrationIDs)
	binding.AttemptIDs = compact(binding.AttemptIDs)
	binding.SessionIDs = compact(binding.SessionIDs)
	sort.Strings(binding.RegistrationIDs)
	sort.Strings(binding.AttemptIDs)
	sort.Strings(binding.SessionIDs)
	invalid = compact(invalid)
	sort.Strings(invalid)

	keys := make([]string, 0, len(issues))
	for key := range issues {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	switch {
	case len(keys) > 1:
		binding.Status = WIPBindingConflicting
		binding.Details = []string{"lineage carries conflicting issue bindings: " + strings.Join(keys, ",")}
	case len(invalid) > 0:
		binding.Status = WIPBindingAmbiguous
		binding.Details = []string{"unparseable issue bindings: " + strings.Join(invalid, ",")}
	case len(keys) == 0:
		binding.Status = WIPBindingMissing
		binding.Details = []string{"lineage has no durable issue binding"}
	default:
		issue := issues[keys[0]]
		binding.Issue = &issue
		if active && !staleBefore.IsZero() && !binding.LatestActivity.IsZero() && binding.LatestActivity.Before(staleBefore) {
			binding.Status = WIPBindingStale
			binding.Details = []string{"active lineage activity predates stale threshold"}
		} else {
			binding.Status = WIPBindingJoined
		}
	}
	return binding
}

func parseWIPIssue(raw, repository string) (WIPIssueIdentity, bool) {
	value := strings.TrimSpace(raw)
	value = strings.TrimPrefix(value, "#")
	if value == "" {
		return WIPIssueIdentity{}, false
	}
	repo := repository
	numberText := value
	if hash := strings.LastIndex(value, "#"); hash >= 0 {
		repo, numberText = strings.TrimSpace(value[:hash]), strings.TrimSpace(value[hash+1:])
	}
	number, err := strconv.Atoi(numberText)
	if err != nil || number <= 0 || strings.TrimSpace(repo) == "" {
		return WIPIssueIdentity{}, false
	}
	return WIPIssueIdentity{Repository: repo, Number: number}, true
}

func appendUniqueSorted(values []string, value string) []string {
	values = append(values, value)
	values = compact(values)
	sort.Strings(values)
	return values
}

func containsString(values []string, target string) bool {
	i := sort.SearchStrings(values, target)
	return i < len(values) && values[i] == target
}
