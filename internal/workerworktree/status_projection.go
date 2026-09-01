package workerworktree

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/markerblock"
)

const (
	StatusProjectionSchema = "fak-worker-worktree-status/1"
	issueStatusBeginMarker = "<!-- fak-worker-worktree-status:begin -->"
	issueStatusEndMarker   = "<!-- fak-worker-worktree-status:end -->"
)

// DisplayState is the bounded operator-facing state of a detached worker worktree.
type DisplayState string

const (
	DisplayActive             DisplayState = "active"
	DisplayUnlandedChanges    DisplayState = "unlanded_changes"
	DisplayLandedWitnessed    DisplayState = "landed_witnessed"
	DisplayCleanupReady       DisplayState = "cleanup_ready"
	DisplayAssociationUnknown DisplayState = "association_unknown"
)

// StatusEvidence contains only the lifecycle facts needed to derive a public
// status. Callers keep paths and raw process metadata on the local boundary.
type StatusEvidence struct {
	IssueNumber      int
	Lane             string
	Session          string
	HeadSHA          string
	BaseSHA          string
	AssociationKnown bool
	OwnerLive        bool
	LeaseLive        bool
	Dirty            bool
	LandedWitnessed  bool
	CleanupReady     bool
}

// StatusProjection is safe to expose through dispatch, harness, and issue
// status surfaces: it contains no filesystem path or raw owner identity.
type StatusProjection struct {
	Schema      string       `json:"schema"`
	State       DisplayState `json:"state"`
	IssueNumber int          `json:"issue_number,omitempty"`
	Lane        string       `json:"lane,omitempty"`
	Session     string       `json:"session,omitempty"`
	Commit      string       `json:"commit,omitempty"`
}

// ProjectStatus maps authoritative local evidence into one closed display state.
// Cleanliness, worker exit, or a released lease never imply completion; only the
// caller's independent landed witness can produce landed_witnessed.
func ProjectStatus(e StatusEvidence) StatusProjection {
	out := StatusProjection{
		Schema:      StatusProjectionSchema,
		IssueNumber: e.IssueNumber,
		Lane:        strings.TrimSpace(e.Lane),
		Session:     strings.TrimSpace(e.Session),
	}
	switch {
	case !e.AssociationKnown:
		out.State = DisplayAssociationUnknown
	case e.OwnerLive || e.LeaseLive:
		out.State = DisplayActive
	case e.Dirty || (differentRevision(e.HeadSHA, e.BaseSHA) && !e.LandedWitnessed):
		out.State = DisplayUnlandedChanges
	case e.LandedWitnessed:
		out.State = DisplayLandedWitnessed
		out.Commit = shortCommit(e.HeadSHA)
	case e.CleanupReady:
		out.State = DisplayCleanupReady
	default:
		out.State = DisplayAssociationUnknown
	}
	return out
}

func differentRevision(head, base string) bool {
	head, base = strings.TrimSpace(head), strings.TrimSpace(base)
	return head != "" && base != "" && head != base
}

func shortCommit(sha string) string {
	sha = strings.TrimSpace(sha)
	if len(sha) > 12 {
		return sha[:12]
	}
	return sha
}

// RenderIssueStatusComment returns an idempotent marker-bounded GitHub comment.
// If an existing comment already contains the block, only that block is replaced.
func RenderIssueStatusComment(existing string, issueNumber int, rows []StatusProjection) (string, error) {
	if issueNumber <= 0 {
		return "", fmt.Errorf("issue number must be positive")
	}
	filtered := make([]StatusProjection, 0, len(rows))
	for _, row := range rows {
		if row.IssueNumber == issueNumber {
			filtered = append(filtered, row)
		}
	}
	sort.SliceStable(filtered, func(i, j int) bool {
		if filtered[i].Lane != filtered[j].Lane {
			return filtered[i].Lane < filtered[j].Lane
		}
		if filtered[i].Session != filtered[j].Session {
			return filtered[i].Session < filtered[j].Session
		}
		return filtered[i].State < filtered[j].State
	})

	var b strings.Builder
	fmt.Fprintln(&b, issueStatusBeginMarker)
	fmt.Fprintf(&b, "### Detached worker worktrees for #%d\n\n", issueNumber)
	if len(filtered) == 0 {
		fmt.Fprintln(&b, "_No associated detached worker worktrees._")
	} else {
		fmt.Fprintln(&b, "| state | lane | session | witnessed commit |")
		fmt.Fprintln(&b, "|---|---|---|---|")
		for _, row := range filtered {
			fmt.Fprintf(&b, "| `%s` | %s | %s | %s |\n", row.State, issueCell(row.Lane), issueCell(row.Session), issueCell(row.Commit))
		}
	}
	fmt.Fprint(&b, issueStatusEndMarker)
	block := b.String()
	if _, ok := markerblock.Extract(existing, issueStatusBeginMarker, issueStatusEndMarker); ok {
		return markerblock.Splice(existing, issueStatusBeginMarker, issueStatusEndMarker, block)
	}
	if strings.TrimSpace(existing) == "" {
		return block, nil
	}
	return strings.TrimRight(existing, "\n") + "\n\n" + block, nil
}

func issueCell(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "—"
	}
	value = strings.ReplaceAll(value, "|", "\\|")
	value = strings.ReplaceAll(value, "\n", " ")
	return "`" + value + "`"
}
