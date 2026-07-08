package auditreason

import "strings"

// ResumeStep is one ordered mutating step of a sequenced tool op — e.g. the
// "commit" then "push" of a `git commit && git push`. Applied records whether
// the step's effect had already landed when the op was killed; ResumeCmd is the
// idempotent command that completes the step when it had not. AppliedMsg and
// PendingMsg are the human phrases folded into the report's tree_state summary.
type ResumeStep struct {
	Name       string
	Applied    bool
	ResumeCmd  string
	AppliedMsg string
	PendingMsg string
}

// ResumeReport is the structured partial-state payload emitted in place of a
// bare termination signal (the exit-143 an agent otherwise has to investigate by
// hand) when a long-running mutating op is killed on timeout. It answers the
// three questions the bare signal forces open: what already applied, what the
// tree looks like now, and the single safe command that resumes the op without
// double-applying the half that already landed.
type ResumeReport struct {
	Op            string   `json:"op"`
	Token         string   `json:"token"`
	Complete      bool     `json:"complete"`
	AppliedSoFar  []string `json:"applied_so_far"`
	Pending       []string `json:"pending"`
	TreeState     string   `json:"tree_state"`
	SafeResumeCmd string   `json:"safe_resume_cmd"`
	Retryable     bool     `json:"retryable"`
}

// ClassifyResume folds an ordered step list into a ResumeReport. token names the
// termination class (typically ToolFailureTimeout). The FIRST not-yet-applied
// step supplies the safe resume command, so resuming completes the sequence from
// exactly where the kill interrupted it.
//
// Because the report is built from an OBSERVED post-kill state (the git tree),
// not from the killed process's transcript, its resume command is safe to rerun:
// a partial op reports Retryable=true even though the raw ToolFailurePartialApply
// vocabulary row is conservatively non-retryable before any such read-back.
func ClassifyResume(op string, token ToolFailure, steps []ResumeStep) ResumeReport {
	r := ResumeReport{
		Op:           op,
		Token:        string(token),
		Complete:     true,
		AppliedSoFar: []string{},
		Pending:      []string{},
	}
	phrases := make([]string, 0, len(steps))
	for _, s := range steps {
		if s.Applied {
			r.AppliedSoFar = append(r.AppliedSoFar, s.Name)
			if strings.TrimSpace(s.AppliedMsg) != "" {
				phrases = append(phrases, s.AppliedMsg)
			}
			continue
		}
		r.Complete = false
		r.Pending = append(r.Pending, s.Name)
		if strings.TrimSpace(s.PendingMsg) != "" {
			phrases = append(phrases, s.PendingMsg)
		}
		if r.SafeResumeCmd == "" {
			r.SafeResumeCmd = strings.TrimSpace(s.ResumeCmd)
		}
	}
	r.TreeState = strings.Join(phrases, "; ")
	r.Retryable = !r.Complete && r.SafeResumeCmd != ""
	return r
}
