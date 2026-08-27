package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ghexec"
	"github.com/anthony-chaudhary/fak/internal/issuecheck"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

const (
	managedThoughtCheckSchema  = "fak.issuecheck.admission.v1"
	managedThoughtCheckReason  = "THOUGHT_CHECK_REQUIRED"
	managedThoughtCheckModeEnv = "FAK_THOUGHT_CHECK_MODE"
	managedIssueEnv            = "FLEET_RESOLVE_ISSUE"
)

var managedThoughtCheckLandMax = 20 * time.Second

type managedThoughtCheckAdmission struct {
	Schema         string  `json:"schema"`
	Mode           string  `json:"mode"`
	Required       bool    `json:"required"`
	OK             bool    `json:"ok"`
	BindingError   bool    `json:"binding_error,omitempty"`
	Issue          int     `json:"issue,omitempty"`
	CommentID      int64   `json:"comment_id,omitempty"`
	IssueDigest    string  `json:"issue_digest,omitempty"`
	CatalogVersion string  `json:"catalog_version,omitempty"`
	Reviewer       string  `json:"reviewer_version,omitempty"`
	ReasonCode     string  `json:"reason_code,omitempty"`
	Reason         string  `json:"reason,omitempty"`
	Matches        []int64 `json:"matching_ids,omitempty"`
}

func (a managedThoughtCheckAdmission) Blocks() bool {
	return a.Required && (a.BindingError || a.Mode != "observe") && !a.OK
}

var managedThoughtCheckRunnerFactory = func(ctx context.Context, dir string) thoughtCheckRunner {
	return func(_ context.Context, args ...string) ([]byte, error) {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("thought-check admission deadline: %w", err)
		}
		cmd := ghexec.Command(ctx, args...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if err != nil {
			if ctx.Err() != nil {
				return nil, fmt.Errorf("thought-check admission deadline: %w", ctx.Err())
			}
			return nil, fmt.Errorf("gh %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
		}
		return out, nil
	}
}

func managedThoughtCheckMode(raw string) (string, error) {
	mode := strings.ToLower(strings.TrimSpace(raw))
	if mode == "" {
		return "enforce", nil
	}
	switch mode {
	case "enforce", "observe", "off":
		return mode, nil
	default:
		return mode, fmt.Errorf("%s must be enforce, observe, or off (got %q)", managedThoughtCheckModeEnv, raw)
	}
}

func managedIssueNumber(raw string) (issue int, present bool, err error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, false, nil
	}
	issue, err = strconv.Atoi(raw)
	if err != nil || issue <= 0 {
		return 0, true, fmt.Errorf("%s must be a positive issue number (got %q)", managedIssueEnv, raw)
	}
	return issue, true, nil
}

func verifyManagedThoughtCheck(ctx context.Context, issueNumber int, modeRaw string, runner thoughtCheckRunner) managedThoughtCheckAdmission {
	admission := managedThoughtCheckAdmission{
		Schema: managedThoughtCheckSchema, Required: true, Issue: issueNumber,
		ReasonCode: managedThoughtCheckReason,
	}
	mode, err := managedThoughtCheckMode(modeRaw)
	admission.Mode = mode
	if err != nil {
		admission.Reason = err.Error()
		return admission
	}
	if mode == "off" {
		admission.OK = true
		admission.ReasonCode = ""
		admission.Reason = "thought-check admission disabled by explicit rollback mode"
		return admission
	}
	if issueNumber <= 0 {
		admission.Reason = "managed issue binding is missing or invalid"
		return admission
	}
	issue, err := fetchThoughtCheckIssue(ctx, runner, "", issueNumber)
	if err != nil {
		admission.Reason = err.Error()
		return admission
	}
	stableOwner, err := fetchThoughtCheckRepoOwner(ctx, runner, "")
	if err != nil {
		admission.Reason = err.Error()
		return admission
	}
	comments, err := fetchThoughtCheckComments(ctx, runner, "", issueNumber, stableOwner)
	if err != nil {
		admission.Reason = err.Error()
		return admission
	}
	verification, err := issuecheck.VerifyComment(issue, comments)
	if err != nil {
		admission.Reason = err.Error()
		return admission
	}
	admission.Matches = append([]int64(nil), verification.MatchingIDs...)
	if !verification.Valid {
		admission.Reason = verification.Reason
		return admission
	}
	admission.OK = true
	admission.ReasonCode = ""
	admission.CommentID = verification.CommentID
	admission.IssueDigest = verification.IssueDigest
	admission.CatalogVersion = verification.CatalogVersion
	admission.Reviewer = verification.ReviewerVersion
	return admission
}

func managedThoughtCheckBinding(root, issueEnv string) (issue int, required bool, err error) {
	envIssue, envPresent, envErr := managedIssueNumber(issueEnv)
	if workerworktree.IsWorkerWorktree(root) {
		intent, loadErr := workerworktree.LoadIntent(root)
		if loadErr != nil {
			return 0, true, fmt.Errorf("load managed worker intent: %w", loadErr)
		}
		if intent.IssueNumber <= 0 {
			return 0, true, fmt.Errorf("managed worker intent has no positive issue binding")
		}
		if envErr != nil {
			return 0, true, envErr
		}
		if envPresent && envIssue != intent.IssueNumber {
			return 0, true, fmt.Errorf("managed worker issue mismatch: intent binds #%d but %s binds #%d", intent.IssueNumber, managedIssueEnv, envIssue)
		}
		return intent.IssueNumber, true, nil
	}
	if envErr != nil {
		return 0, true, envErr
	}
	return envIssue, envPresent, nil
}

func managedWorkerThoughtCheckAdmission(ctx context.Context, root string) managedThoughtCheckAdmission {
	issue, required, bindingErr := managedThoughtCheckBinding(root, os.Getenv(managedIssueEnv))
	if !required {
		return managedThoughtCheckAdmission{Schema: managedThoughtCheckSchema, OK: true}
	}
	mode, modeErr := managedThoughtCheckMode(os.Getenv(managedThoughtCheckModeEnv))
	if bindingErr != nil || modeErr != nil {
		admission := managedThoughtCheckAdmission{
			Schema: managedThoughtCheckSchema, Mode: mode, Required: true, Issue: issue,
			BindingError: true, ReasonCode: managedThoughtCheckReason,
		}
		if bindingErr != nil {
			admission.Reason = "managed worker binding refused: " + bindingErr.Error()
		}
		if modeErr != nil {
			admission.Reason = modeErr.Error()
		}
		return admission
	}
	return verifyManagedThoughtCheck(ctx, issue, os.Getenv(managedThoughtCheckModeEnv), managedThoughtCheckRunnerFactory(ctx, root))
}

var subjectIssueRE = regexp.MustCompile(`#([1-9][0-9]*)`)

func uniqueSubjectIssue(message string) (issue int, found bool, ambiguous bool) {
	subject := strings.SplitN(strings.ReplaceAll(message, "\r\n", "\n"), "\n", 2)[0]
	seen := map[int]bool{}
	for _, match := range subjectIssueRE.FindAllStringSubmatch(subject, -1) {
		n, err := strconv.Atoi(match[1])
		if err == nil && n > 0 {
			seen[n] = true
		}
	}
	if len(seen) == 0 {
		return 0, false, false
	}
	if len(seen) != 1 {
		return 0, false, true
	}
	for n := range seen {
		return n, true, false
	}
	return 0, false, false
}

func isPreparedWorkerMessage(worktreePath, messagePath string) bool {
	if strings.TrimSpace(worktreePath) == "" || strings.TrimSpace(messagePath) == "" {
		return false
	}
	want := filepath.Join(filepath.Dir(filepath.Clean(worktreePath)), ".fak-worker-intents", filepath.Base(filepath.Clean(worktreePath))+".message")
	got, err1 := filepath.Abs(filepath.Clean(messagePath))
	expected, err2 := filepath.Abs(want)
	if err1 != nil || err2 != nil {
		return false
	}
	if runtime.GOOS == "windows" {
		return strings.EqualFold(got, expected)
	}
	return got == expected
}

func worktreeLandThoughtCheckBinding(worktreePath, messagePath, issueEnv string) (int, bool, string) {
	intentIssue, managed, managedErr := managedThoughtCheckBinding(worktreePath, issueEnv)
	if managedErr != nil {
		return 0, true, managedErr.Error()
	}
	if workerworktree.IsWorkerWorktree(worktreePath) {
		if strings.TrimSpace(messagePath) == "" {
			return 0, true, "managed worker land requires its saved commit message"
		}
		if !isPreparedWorkerMessage(worktreePath, messagePath) {
			return 0, true, "managed worker land requires the coordinator-saved commit message path"
		}
		intent, err := workerworktree.LoadIntent(worktreePath)
		if err != nil {
			return 0, true, "load managed worker intent: " + err.Error()
		}
		raw, err := os.ReadFile(messagePath)
		if err != nil {
			return 0, true, "read managed worker commit message: " + err.Error()
		}
		if !bytes.Equal(raw, []byte(intent.Message+"\n")) {
			return 0, true, "managed worker commit message differs from coordinator-saved intent"
		}
		messageIssue, found, ambiguous := uniqueSubjectIssue(string(raw))
		if ambiguous || !found {
			return 0, true, "managed worker commit subject must bind exactly one distinct #N issue"
		}
		if messageIssue != intentIssue {
			return 0, true, fmt.Sprintf("managed worker issue mismatch: intent binds #%d but commit subject binds #%d", intentIssue, messageIssue)
		}
		return intentIssue, true, ""
	}
	envIssue, envPresent, envErr := managedIssueNumber(issueEnv)
	prepared := isPreparedWorkerMessage(worktreePath, messagePath)
	if strings.TrimSpace(messagePath) == "" {
		if envErr != nil {
			return 0, true, envErr.Error()
		}
		return envIssue, managed || envPresent, ""
	}
	raw, readErr := os.ReadFile(messagePath)
	if readErr != nil {
		if prepared || envPresent {
			return 0, true, "read managed worker commit message: " + readErr.Error()
		}
		return 0, false, ""
	}
	msgIssue, found, ambiguous := uniqueSubjectIssue(string(raw))
	if ambiguous {
		return 0, true, "managed worker commit subject must bind exactly one distinct #N issue"
	}
	if envErr != nil {
		return 0, true, envErr.Error()
	}
	if found && envPresent && msgIssue != envIssue {
		return 0, true, fmt.Sprintf("managed worker issue mismatch: subject binds #%d but %s binds #%d", msgIssue, managedIssueEnv, envIssue)
	}
	if found {
		return msgIssue, true, ""
	}
	if envPresent {
		return envIssue, true, ""
	}
	if prepared {
		return 0, true, "prepared worker commit subject must bind exactly one #N issue"
	}
	return 0, false, ""
}

func composeWorktreeThoughtCheckVerify(worktreePath, messagePath string, base workerworktree.VerifyHook) workerworktree.VerifyHook {
	issueEnv := os.Getenv(managedIssueEnv)
	issueNumber, required, bindingErr := worktreeLandThoughtCheckBinding(worktreePath, messagePath, issueEnv)
	if !required {
		return base
	}
	return func(wtPath string) (bool, string) {
		currentIssue, currentRequired, currentBindingErr := worktreeLandThoughtCheckBinding(worktreePath, messagePath, issueEnv)
		if !currentRequired || currentBindingErr != "" || currentIssue != issueNumber {
			if currentBindingErr == "" {
				currentBindingErr = "managed worker issue/message binding changed before verification"
			}
			return false, managedThoughtCheckReason + ": " + currentBindingErr
		}
		mode, modeErr := managedThoughtCheckMode(os.Getenv(managedThoughtCheckModeEnv))
		if bindingErr != "" {
			return false, managedThoughtCheckReason + ": " + bindingErr
		}
		if modeErr != nil {
			return false, managedThoughtCheckReason + ": " + modeErr.Error()
		} else if mode != "off" {
			ctx, cancel := context.WithTimeout(context.Background(), managedThoughtCheckLandMax)
			admission := verifyManagedThoughtCheck(ctx, issueNumber, mode, managedThoughtCheckRunnerFactory(ctx, wtPath))
			cancel()
			if !admission.OK {
				if mode == "observe" {
					_ = writeThoughtCheckJSON(os.Stderr, admission)
				} else {
					return false, managedThoughtCheckReason + ": " + admission.Reason
				}
			}
		}
		finalIssue, finalRequired, finalBindingErr := worktreeLandThoughtCheckBinding(worktreePath, messagePath, issueEnv)
		if !finalRequired || finalBindingErr != "" || finalIssue != issueNumber {
			if finalBindingErr == "" {
				finalBindingErr = "managed worker issue/message binding changed before land"
			}
			return false, managedThoughtCheckReason + ": " + finalBindingErr
		}
		if base != nil {
			return base(wtPath)
		}
		return true, ""
	}
}
