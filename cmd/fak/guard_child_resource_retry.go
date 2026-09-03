package main

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

const (
	guardResourceRestartLimitEnv        = "FLEET_CLAUDE_GUARD_RESOURCE_RESTART_LIMIT"
	guardResourceRestartDefaultLimit    = 3
	guardResourceNoProgressLimitEnv     = "FLEET_CLAUDE_GUARD_RESOURCE_NO_PROGRESS_LIMIT"
	guardResourceNoProgressDefaultLimit = 2
	guardResourceRestartExhaustedReason = "CHILD_RESOURCE_RESTART_EXHAUSTED"
	guardResourceReattachUnavailable    = "CHILD_RESOURCE_REATTACH_UNAVAILABLE"
	guardResourceRestartCauseBudget     = "budget"
	guardResourceRestartCauseNoProgress = "no_progress"
	guardResourceRestartCauseNoReattach = "reattach_unavailable"
)

type guardResourceRetryAction uint8

const (
	guardResourceRetryTerminal guardResourceRetryAction = iota
	guardResourceRetryRelaunch
	guardResourceRetryExhausted
)

type guardResourceRetryVerdict struct {
	Action       guardResourceRetryAction
	Attempt      int
	Limit        int
	Delay        time.Duration
	Cause        string
	NoProgress   int
	ResourceType string
}

type guardResourceRetryState struct {
	restarts        int
	limit           int
	progressHead    string
	noProgress      int
	noProgressLimit int
}

func newGuardResourceRetryState() guardResourceRetryState {
	limit := guardResourceRestartLimit()
	return guardResourceRetryState{
		limit:           limit,
		noProgressLimit: guardResourceNoProgressLimit(limit),
	}
}

func guardResourceRestartLimit() int {
	raw, set := os.LookupEnv(guardResourceRestartLimitEnv)
	if !set || strings.TrimSpace(raw) == "" {
		return guardResourceRestartDefaultLimit
	}
	n, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || n < 0 {
		return guardResourceRestartDefaultLimit
	}
	return n
}

func guardResourceNoProgressLimit(restartLimit int) int {
	raw := strings.TrimSpace(os.Getenv(guardResourceNoProgressLimitEnv))
	limit := guardResourceNoProgressDefaultLimit
	if raw != "" {
		if n, err := strconv.Atoi(raw); err == nil && n >= 0 {
			limit = n
		}
	}
	if restartLimit > 1 && limit >= restartLimit {
		limit = restartLimit - 1
	}
	return limit
}

func guardResourceContainmentReason(reason string) bool {
	switch strings.TrimSpace(reason) {
	case "CHILD_TREE_RSS_LIMIT", "CHILD_TREE_COMMIT_LIMIT", "SYSTEM_COMMIT_HEADROOM":
		return true
	default:
		return false
	}
}

// decide admits only a successful, typed containment event. Collector failures
// stay terminal because a guard that cannot observe the child tree cannot safely
// claim that another generation will remain contained.
func (s *guardResourceRetryState) decide(event guardChildWaitEvent, agentName, currentHead string) guardResourceRetryVerdict {
	verdict := guardResourceRetryVerdict{Action: guardResourceRetryTerminal, Limit: s.limit}
	if event.Kind != guardChildResourceLimit || event.Resource == nil || !event.Resource.Stop {
		return verdict
	}
	verdict.ResourceType = strings.TrimSpace(event.Resource.Reason)
	if !guardResourceContainmentReason(verdict.ResourceType) || s.limit <= 0 {
		return verdict
	}
	if !guardResourceReattachSupported(agentName) {
		verdict.Cause = guardResourceRestartCauseNoReattach
		return verdict
	}
	if s.restarts >= s.limit {
		verdict.Action = guardResourceRetryExhausted
		verdict.Attempt = s.restarts
		verdict.Cause = guardResourceRestartCauseBudget
		return verdict
	}

	nextHead, nextNoProgress := guardNoProgressStep(s.progressHead, currentHead, s.noProgress)
	s.progressHead, s.noProgress = nextHead, nextNoProgress
	if s.noProgressLimit > 0 && s.noProgress >= s.noProgressLimit {
		verdict.Action = guardResourceRetryExhausted
		verdict.Attempt = s.restarts
		verdict.Cause = guardResourceRestartCauseNoProgress
		verdict.NoProgress = s.noProgress
		return verdict
	}

	s.restarts++
	verdict.Action = guardResourceRetryRelaunch
	verdict.Attempt = s.restarts
	verdict.Delay = guardCrashRestartDelay(s.restarts)
	verdict.NoProgress = s.noProgress
	return verdict
}

func guardResourceReattachSupported(agentName string) bool {
	if _, ok := guardContinueFlagForAgent(agentName); ok {
		return true
	}
	return guardIsCodex(agentName)
}

func guardResourceReattachCommand(command []string, agentName, codexStatePath, guardTraceID string) ([]string, error) {
	if flag, ok := guardContinueFlagForAgent(agentName); ok {
		return guardAppendContinueFlag(command, flag), nil
	}
	if guardIsCodex(agentName) {
		return guardCodexResourceResumeCommand(command, codexStatePath, guardTraceID)
	}
	return nil, fmt.Errorf("agent %q has no safe resource reattach transport; recovery: run with a supported harness (claude, codex) or without child resource restarts", strings.TrimSpace(agentName))
}

func guardResourceReattachUnavailableStatus(agentName, traceID string, cause error) string {
	detail := "exact continuation binding unavailable"
	if cause != nil && strings.TrimSpace(cause.Error()) != "" {
		detail = strings.TrimSpace(cause.Error())
	}
	recovery := "rerun the original fak guard invocation with this harness's provider-native resume command"
	if guardIsCodex(agentName) {
		recovery = "run `fak guard -- codex resume` and select the interrupted thread"
	} else if _, ok := guardContinueFlagForAgent(agentName); ok {
		recovery = "run `fak guard -- claude --continue`"
	}
	return fmt.Sprintf("fak guard: %s: child resource containment recovery cannot safely reattach agent %q in place (trace %s): %s; refusing a cold relaunch; recovery: %s",
		guardResourceReattachUnavailable, strings.TrimSpace(agentName), strings.TrimSpace(traceID), detail, recovery)
}

func guardReportResourceRestart(stderr io.Writer, agentName string, verdict guardResourceRetryVerdict, command []string) {
	if stderr == nil {
		return
	}
	fmt.Fprintf(stderr, "fak guard: %s child hit %s; verified tree reap and receipt complete; guard remains up and is reattaching the child after %s (resource restart %d/%d) `%s`\n",
		agentName, verdict.ResourceType, verdict.Delay, verdict.Attempt, verdict.Limit,
		strings.Join(command, " "))
}

func guardResourceRestartGiveUpStatus(verdict guardResourceRetryVerdict, traceID string) string {
	detail := fmt.Sprintf("retry budget %d exhausted", verdict.Limit)
	if verdict.Cause == guardResourceRestartCauseNoProgress {
		detail = fmt.Sprintf("%d consecutive containment retries without HEAD progress", verdict.NoProgress)
	}
	return fmt.Sprintf("fak guard: %s: child resource containment recovery stopped after %s (trace %s); refusing another relaunch; recovery: inspect memory leaks in child process, increase --child-max-memory-mb, or adjust %s",
		guardResourceRestartExhaustedReason, detail, strings.TrimSpace(traceID), guardResourceRestartLimitEnv)
}

func guardRecordResourceRestart(auditJournal *journal.Journal, stderr io.Writer, agentName, traceID string, attempt int) {
	guardEmitRestartHop(auditJournal, stderr, agentName, traceID, guardResourceRestartHop(traceID, agentName, attempt))
}

func guardResourceRestartHop(traceID, agentName string, hop int) journal.RestartHop {
	restart := guardSameTraceRelaunchHop(traceID, agentName, hop)
	if guardResourceReattachSupported(agentName) {
		restart.Handback = guardRestartHandbackContinue
		restart.Status = journal.RestartHopOK
	}
	return restart
}

func guardRecordResourceRestartGiveUp(auditJournal *journal.Journal, agentName, traceID string) {
	if auditJournal != nil {
		auditJournal.AppendCrash(agentName, traceID, guardResourceRestartExhaustedReason, -1)
	}
}

func guardRecordResourceReattachUnavailable(auditJournal *journal.Journal, agentName, traceID string) {
	if auditJournal != nil {
		auditJournal.AppendCrash(agentName, traceID, guardResourceReattachUnavailable, -1)
	}
}
