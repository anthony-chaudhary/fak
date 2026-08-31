package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionrecovery"
)

const (
	sessionRecoveryFailureRSITrigger         = "session_recovery"
	sessionRecoveryFailureRSISubsystem       = "session_recovery"
	sessionRecoveryFailureRSIReceiptIdentity = "fak.session-recovery.v1/result"
)

// sessionMaybeLaunchFailureRSI starts at most one advisory investigation for a
// durable recovered-session failure. It claims the command-scoped slot before
// launch so a broken RSI launcher cannot fan out later attempts.
func sessionMaybeLaunchFailureRSI(stderr io.Writer, session interface{ claim() bool }, result sessionrecovery.Request) bool {
	req, ok := sessionRecoveryFailureRSIAdmission(result)
	if !ok || !session.claim() {
		return false
	}
	if err := guardCrashRSILaunch(req); err != nil {
		if stderr != nil {
			fmt.Fprintf(stderr, "fak session recover: failure RSI launch skipped (%s): %v\n", req.Tag, err)
		}
		return false
	}
	if stderr != nil {
		fmt.Fprintf(stderr, "fak session recover: spawned failure RSI session %s for %s\n", req.Tag, req.Reason)
	}
	return true
}

func sessionRecoveryFailureRSIAdmission(result sessionrecovery.Request) (guardCrashRSIRequest, bool) {
	if strings.TrimSpace(os.Getenv(guardCrashRSIMarkerEnv)) != "" {
		return guardCrashRSIRequest{}, false
	}
	correlationID := strings.TrimSpace(result.ThreadID)
	status := sessionRecoveryFailureRSIStatus(result.Status)
	if correlationID == "" || status == "" {
		return guardCrashRSIRequest{}, false
	}
	agent := guardCrashRSISupportedAgent(result.Harness)
	if agent == "" {
		agent = guardCrashRSISupportedAgent(result.Provider)
	}
	if agent == "" {
		return guardCrashRSIRequest{}, false
	}
	workspace, err := os.Getwd()
	if err != nil || !filepath.IsAbs(workspace) {
		return guardCrashRSIRequest{}, false
	}
	identity := buildIdentityFromRuntime()
	buildCommit := strings.TrimSpace(identity.CommitShort)
	if buildCommit == "" {
		buildCommit = "unstamped"
	}
	buildModule := strings.TrimSpace(identity.ModuleVersion)
	if buildModule == "" {
		buildModule = "cmd/fak"
	}
	signature := guardRSIDigest(strings.Join([]string{
		sessionRecoveryFailureRSITrigger,
		status,
		sessionRecoveryFailureRSISubsystem,
	}, "\x00"))
	source := guardRSIDigest(correlationID)
	req := guardCrashRSIRequest{
		Tag:             "session-recovery-failure-rsi/" + source + "-" + signature,
		Source:          source,
		Agent:           agent,
		Workspace:       workspace,
		Trigger:         sessionRecoveryFailureRSITrigger,
		Reason:          status,
		Subsystem:       sessionRecoveryFailureRSISubsystem,
		Signature:       signature,
		BuildCommit:     buildCommit,
		BuildModule:     buildModule,
		ReceiptIdentity: sessionRecoveryFailureRSIReceiptIdentity,
	}
	req.Prompt = sessionRecoveryFailureRSIPrompt(req)
	return req, true
}

func sessionRecoveryFailureRSIStatus(status string) string {
	status = strings.ToLower(strings.TrimSpace(status))
	switch status {
	case "prompt_failed", "launch_failed", "verification_failed", "launched_unproven":
		return status
	default:
		// Reaped is a durable, typed failure suffix. Keep the exact normalized
		// token so future recovery reapers gain the same bounded side channel.
		if strings.HasSuffix(status, "_reaped") && len(status) > len("_reaped") {
			return status
		}
		return ""
	}
}

func sessionRecoveryFailureRSIPrompt(req guardCrashRSIRequest) string {
	return fmt.Sprintf(`You are the specially tagged recovered-session launch-failure RSI investigation session %s.
Investigate why the ORIGINAL fak session recovery did not launch or prove a productive recovered session. Do not continue the recovered session's task and do not replay its prompt.
Bounded failure context:
- trigger: %s
- typed_status: %s
- subsystem: %s
- harness: %s
- build_commit: %s
- build_module: %s
- source_recovery: %s
- receipt_identity: %s
- normalized_signature: %s
Perform read-only root-cause analysis, identify the smallest durable prevention, and report evidence plus a checkable next step. Do not expose credentials, paths, process IDs, timestamps, error prose, prompts, receipt payloads, or ambient environment values.`, req.Tag, req.Trigger, req.Reason, req.Subsystem, req.Agent, req.BuildCommit, req.BuildModule, req.Source, req.ReceiptIdentity, req.Signature)
}
