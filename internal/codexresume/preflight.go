package codexresume

import (
	"bufio"
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// PreflightVerdict is the closed admission vocabulary for a persisted Codex
// thread. Only RESUMABLE may proceed to process launch.
type PreflightVerdict string

const (
	VerdictResumable           PreflightVerdict = "RESUMABLE"
	VerdictIncompatibleHistory PreflightVerdict = "INCOMPATIBLE_HISTORY"
	VerdictAlreadyActive       PreflightVerdict = "ALREADY_ACTIVE"
	VerdictNotFound            PreflightVerdict = "NOT_FOUND"
	VerdictHistoryUnreadable   PreflightVerdict = "HISTORY_UNREADABLE"
)

// Compatibility is independent of admission. An active thread can also carry
// incompatible history; callers need both facts to choose the safe recovery.
type Compatibility string

const (
	CompatibilityCompatible   Compatibility = "compatible"
	CompatibilityIncompatible Compatibility = "incompatible"
	CompatibilityUnknown      Compatibility = "unknown"
)

// CheckConfig names the candidate and the target wire contract it would be
// resumed against. RequiredFunctionCallItemPrefix is the prefix required on the
// persisted response_item.function_call.id field (not its logical call_id).
type CheckConfig struct {
	ThreadID                       string
	Thread                         ThreadIdentity
	RolloutPath                    string
	CodexHome                      string
	TargetProvider                 string
	TargetWire                     string
	RequiredFunctionCallItemPrefix string
}

// PreflightResult is scrub-safe and complete enough for a bulk-recovery ledger:
// candidate -> compatibility -> admission verdict -> exact recovery action.
type PreflightResult struct {
	ThreadID                         string           `json:"thread_id"`
	Thread                           *ThreadIdentity  `json:"thread,omitempty"`
	Verdict                          PreflightVerdict `json:"verdict"`
	Compatibility                    Compatibility    `json:"compatibility"`
	RolloutPath                      string           `json:"rollout_path,omitempty"`
	SourceProvider                   string           `json:"source_provider,omitempty"`
	SourceWire                       string           `json:"source_wire,omitempty"`
	TargetProvider                   string           `json:"target_provider,omitempty"`
	TargetWire                       string           `json:"target_wire,omitempty"`
	RequiredFunctionCallItemPrefix   string           `json:"required_function_call_item_prefix,omitempty"`
	FunctionCallItems                int              `json:"function_call_items"`
	ObservedFunctionCallItemPrefixes []string         `json:"observed_function_call_item_prefixes,omitempty"`
	IncompatibleFunctionCallItems    int              `json:"incompatible_function_call_items"`
	FirstIncompatibleItemID          string           `json:"first_incompatible_item_id,omitempty"`
	LatestTurnID                     string           `json:"latest_turn_id,omitempty"`
	LatestTurnStatus                 string           `json:"latest_turn_status,omitempty"`
	LatestTurnError                  *TurnError       `json:"latest_turn_error,omitempty"`
	WriterOwnership                  WriterOwnership  `json:"writer_ownership"`
	WriterLockPath                   string           `json:"writer_lock_path,omitempty"`
	WriterLockPresent                bool             `json:"writer_lock_present"`
	FailedWrapperMarked              bool             `json:"failed_wrapper_marked"`
	StaleWriterLockSuspected         bool             `json:"stale_writer_lock_suspected"`
	RecoveryAction                   string           `json:"recovery_action,omitempty"`
	Detail                           string           `json:"detail,omitempty"`
}

type historyFacts struct {
	threadID            string
	sourceProvider      string
	sourceWire          string
	functionCallItems   int
	observedPrefixes    []string
	incompatibleItems   int
	firstIncompatibleID string
	latestTurnID        string
	latestTurnStatus    string
	latestTurnError     *TurnError
	malformedLine       int
}

// Preflight reads the persisted rollout before any process is spawned. It
// refuses a call_-item history against a target that requires fc_ item IDs, and
// it treats a writer lock or an unterminated turn as active even when process
// existence looked healthy to an outer launcher.
func Preflight(cfg CheckConfig) PreflightResult {
	return preflightWithProbe(cfg, nativeOwnershipProbe{})
}

func preflightWithProbe(cfg CheckConfig, probe ownershipProbe) PreflightResult {
	cfg = cfg.withDefaults()
	result := PreflightResult{
		ThreadID:                       cfg.ThreadID,
		Compatibility:                  CompatibilityUnknown,
		TargetProvider:                 cfg.TargetProvider,
		TargetWire:                     cfg.TargetWire,
		RequiredFunctionCallItemPrefix: cfg.RequiredFunctionCallItemPrefix,
	}
	thread, err := resolveCheckThreadIdentity(cfg)
	if err != nil {
		result.Verdict = VerdictNotFound
		result.Detail = err.Error()
		result.RecoveryAction = "supply the exact persisted Codex thread UUID and rerun preflight"
		return result
	}
	result.ThreadID = thread.ID
	result.Thread = &thread
	cfg.ThreadID = thread.ID

	rolloutPath := cfg.RolloutPath
	if rolloutPath == "" {
		var err error
		rolloutPath, err = discoverRollout(cfg.CodexHome, cfg.ThreadID)
		if err != nil {
			result.Verdict = VerdictNotFound
			result.Detail = err.Error()
			result.RecoveryAction = "restore the thread rollout under CODEX_HOME/sessions or choose an existing thread id"
			return result
		}
	}
	absRollout, err := filepath.Abs(rolloutPath)
	if err != nil {
		result.Verdict = VerdictHistoryUnreadable
		result.Detail = err.Error()
		result.RecoveryAction = "supply a readable absolute rollout path and rerun preflight"
		return result
	}
	result.RolloutPath = absRollout
	if strings.HasSuffix(strings.ToLower(absRollout), ".gz") {
		result.Verdict = VerdictHistoryUnreadable
		result.Detail = "the matching rollout is archived and cannot be append-tailed as an authoritative live resume witness"
		result.RecoveryAction = "restore the thread to a live .jsonl rollout before launching; preflight may inspect archives but must not report them RESUMABLE"
		return result
	}
	result.WriterLockPath = filepath.Join(cfg.CodexHome, "thread-writer-locks", cfg.ThreadID+".lock")
	resource, err := NewWriterResourceHandle(thread, result.WriterLockPath)
	if err != nil {
		result.WriterOwnership = invalidWriterOwnership(cfg.ThreadID, result.WriterLockPath, err)
	} else {
		result.WriterOwnership = inspectWriterResourceOwnership(resource, probe)
	}
	result.WriterLockPresent = result.WriterOwnership.LockPresent

	facts, err := inspectHistory(absRollout, cfg.RequiredFunctionCallItemPrefix)
	if err != nil {
		result.Verdict = VerdictHistoryUnreadable
		result.Detail = err.Error()
		result.RecoveryAction = "repair or restore the persisted rollout; do not spawn a resume against unreadable history"
		return result
	}
	if facts.threadID == "" || facts.threadID != cfg.ThreadID {
		result.Verdict = VerdictNotFound
		result.Detail = fmt.Sprintf("rollout session id %q does not match candidate %q", facts.threadID, cfg.ThreadID)
		result.RecoveryAction = "select the rollout whose session_meta.id exactly matches the candidate thread"
		return result
	}
	result.SourceProvider = facts.sourceProvider
	result.SourceWire = facts.sourceWire
	if result.TargetProvider == "" {
		result.TargetProvider = facts.sourceProvider
	}
	result.FunctionCallItems = facts.functionCallItems
	result.ObservedFunctionCallItemPrefixes = facts.observedPrefixes
	result.IncompatibleFunctionCallItems = facts.incompatibleItems
	result.FirstIncompatibleItemID = facts.firstIncompatibleID
	result.LatestTurnID = facts.latestTurnID
	result.LatestTurnStatus = facts.latestTurnStatus
	result.LatestTurnError = facts.latestTurnError
	if facts.incompatibleItems > 0 {
		result.Compatibility = CompatibilityIncompatible
	} else {
		result.Compatibility = CompatibilityCompatible
	}

	activeByHistory := facts.latestTurnStatus == "running"
	result.FailedWrapperMarked = result.WriterLockPresent && facts.latestTurnStatus == "failed"
	result.StaleWriterLockSuspected = result.WriterOwnership.Verdict == WriterOwnershipStaleResidue
	switch {
	case result.WriterOwnership.Verdict == WriterOwnershipLiveOwner:
		result.Verdict = VerdictAlreadyActive
		result.Detail = "a native resource witness proved that a live process owns the Codex writer lock"
		result.RecoveryAction = "wait for the witnessed owner to release the thread, then rerun preflight"
		return result
	case result.WriterOwnership.Verdict == WriterOwnershipUnknown:
		result.Verdict = VerdictAlreadyActive
		result.Detail = "writer ownership could not be inspected conclusively; admission fails closed"
		result.RecoveryAction = "restore permission or platform ownership inspection and rerun preflight; do not delete the lock or terminate a possible owner"
		return result
	case activeByHistory:
		result.Verdict = VerdictAlreadyActive
		result.Detail = "the persisted rollout contains a task_started without a matching terminal task_complete"
		result.RecoveryAction = "wait for the owning process to reach a terminal task_complete, then rerun preflight"
		return result
	}

	if result.Compatibility == CompatibilityIncompatible {
		result.Verdict = VerdictIncompatibleHistory
		result.Detail = fmt.Sprintf(
			"%d persisted function_call item id(s) do not begin with %q; first incompatible id %q",
			result.IncompatibleFunctionCallItems,
			result.RequiredFunctionCallItemPrefix,
			result.FirstIncompatibleItemID,
		)
		result.RecoveryAction = fmt.Sprintf(
			"use a target provider/wire that accepts the persisted item-id prefix, or run an atomic migration that rewrites response_item.function_call.id plus the Codex thread-history projection to %q before retrying; no process was spawned",
			result.RequiredFunctionCallItemPrefix,
		)
		return result
	}

	result.Verdict = VerdictResumable
	result.Detail = "persisted function-call item ids satisfy the target wire and no active writer was observed"
	return result
}

func resolveCheckThreadIdentity(cfg CheckConfig) (ThreadIdentity, error) {
	if cfg.Thread != (ThreadIdentity{}) {
		if err := cfg.Thread.Validate(); err != nil {
			return ThreadIdentity{}, fmt.Errorf("invalid typed thread identity: %w", err)
		}
		if cfg.Thread.Provider != ThreadProviderCodex {
			return ThreadIdentity{}, fmt.Errorf("unsupported thread provider %q", cfg.Thread.Provider)
		}
		if cfg.ThreadID != "" && cfg.ThreadID != cfg.Thread.ID {
			return ThreadIdentity{}, errors.New("thread identity does not match compatibility thread_id")
		}
		return cfg.Thread, nil
	}
	if strings.TrimSpace(cfg.ThreadID) == "" {
		return ThreadIdentity{}, errors.New("thread id is required")
	}
	thread, err := NewCodexThreadIdentity(cfg.ThreadID)
	if err != nil {
		return ThreadIdentity{}, err
	}
	return thread, nil
}

func (cfg CheckConfig) withDefaults() CheckConfig {
	out := cfg
	if out.CodexHome == "" {
		out.CodexHome = defaultCodexHome()
	}
	if out.TargetWire == "" {
		out.TargetWire = "responses"
	}
	if out.RequiredFunctionCallItemPrefix == "" {
		out.RequiredFunctionCallItemPrefix = "fc_"
	}
	return out
}

func defaultCodexHome() string {
	if home := os.Getenv("CODEX_HOME"); home != "" {
		return home
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return ".codex"
	}
	return filepath.Join(home, ".codex")
}

func discoverRollout(codexHome, threadID string) (string, error) {
	root := filepath.Join(codexHome, "sessions")
	var candidates []string
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() {
			return nil
		}
		name := entry.Name()
		if strings.Contains(name, threadID) && (strings.HasSuffix(name, ".jsonl") || strings.HasSuffix(name, ".jsonl.gz")) {
			candidates = append(candidates, path)
		}
		return nil
	})
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("thread %s has no rollout under %s", threadID, root)
		}
		return "", fmt.Errorf("scan Codex rollouts: %w", err)
	}
	sort.Strings(candidates)
	for _, candidate := range candidates {
		facts, inspectErr := inspectHistory(candidate, "")
		if inspectErr == nil && facts.threadID == threadID {
			return candidate, nil
		}
	}
	return "", fmt.Errorf("thread %s has no matching rollout under %s", threadID, root)
}

func inspectHistory(path, requiredPrefix string) (historyFacts, error) {
	reader, closeFn, err := openRollout(path)
	if err != nil {
		return historyFacts{}, fmt.Errorf("open rollout: %w", err)
	}
	defer closeFn()

	var facts historyFacts
	observedPrefixes := make(map[string]struct{})
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	lineNumber := 0
	for scanner.Scan() {
		lineNumber++
		line := scanner.Bytes()
		if len(strings.TrimSpace(string(line))) == 0 {
			continue
		}
		var row struct {
			Type    string `json:"type"`
			Payload struct {
				Type          string          `json:"type"`
				ID            string          `json:"id"`
				TurnID        string          `json:"turn_id"`
				ModelProvider string          `json:"model_provider"`
				Error         json.RawMessage `json:"error"`
				Reason        string          `json:"reason"`
			} `json:"payload"`
		}
		if err := json.Unmarshal(line, &row); err != nil {
			facts.malformedLine = lineNumber
			break
		}
		switch row.Type {
		case "session_meta":
			facts.threadID = row.Payload.ID
			facts.sourceProvider = row.Payload.ModelProvider
		case "response_item":
			facts.sourceWire = "responses"
			if row.Payload.Type == "function_call" {
				facts.functionCallItems++
				if prefix := functionCallItemPrefix(row.Payload.ID); prefix != "" {
					observedPrefixes[prefix] = struct{}{}
				}
				if requiredPrefix != "" && !strings.HasPrefix(row.Payload.ID, requiredPrefix) {
					facts.incompatibleItems++
					if facts.firstIncompatibleID == "" {
						facts.firstIncompatibleID = row.Payload.ID
					}
				}
			}
		case "event_msg":
			switch row.Payload.Type {
			case "task_started":
				facts.latestTurnID = row.Payload.TurnID
				facts.latestTurnStatus = "running"
				facts.latestTurnError = nil
			case "task_complete":
				if row.Payload.TurnID == "" || facts.latestTurnID == "" || row.Payload.TurnID == facts.latestTurnID {
					if turnErr := parseTurnError(row.Payload.Error); turnErr != nil {
						facts.latestTurnStatus = "failed"
						facts.latestTurnError = turnErr
					} else {
						facts.latestTurnStatus = "completed"
						facts.latestTurnError = nil
					}
				}
			case "turn_aborted":
				if row.Payload.Reason == "interrupted" {
					facts.latestTurnStatus = "interrupted"
				}
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return historyFacts{}, fmt.Errorf("scan rollout: %w", err)
	}
	if facts.malformedLine != 0 {
		return historyFacts{}, fmt.Errorf("rollout line %d is not valid JSON", facts.malformedLine)
	}
	for prefix := range observedPrefixes {
		facts.observedPrefixes = append(facts.observedPrefixes, prefix)
	}
	sort.Strings(facts.observedPrefixes)
	return facts, nil
}

func functionCallItemPrefix(id string) string {
	if id == "" {
		return ""
	}
	if index := strings.IndexByte(id, '_'); index >= 0 {
		return id[:index+1]
	}
	return id
}

func openRollout(path string) (io.Reader, func() error, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	if !strings.HasSuffix(strings.ToLower(path), ".gz") {
		return file, file.Close, nil
	}
	gz, err := gzip.NewReader(file)
	if err != nil {
		_ = file.Close()
		return nil, nil, err
	}
	return gz, func() error {
		gzErr := gz.Close()
		fileErr := file.Close()
		if gzErr != nil {
			return gzErr
		}
		return fileErr
	}, nil
}

// Recover runs the deterministic preflight and launches only a RESUMABLE candidate.
// Writer locks remain owned and cleaned up by Codex; recovery never deletes them.
func Recover(ctx context.Context, checkConfig CheckConfig, runConfig Config) (Result, error) {
	preflight := Preflight(checkConfig)
	result := Result{ThreadID: preflight.ThreadID, Preflight: &preflight}
	if preflight.Verdict != VerdictResumable {
		result.Outcome = OutcomeRefused
		return result, nil
	}
	runConfig.RolloutPath = preflight.RolloutPath
	result, err := Run(ctx, runConfig)
	result.ThreadID = preflight.ThreadID
	result.Preflight = &preflight
	if err != nil {
		return result, err
	}
	result.WriterLockCleanup = cleanupOwnedWriterLock(preflight, result)
	return result, nil
}

func cleanupOwnedWriterLock(preflight PreflightResult, result Result) string {
	if preflight.WriterLockPath == "" || preflight.WriterLockPresent {
		return "not_owned"
	}
	switch result.Outcome {
	case OutcomeCompleted, OutcomeCompletedReclaimed, OutcomeTurnFailed, OutcomeTurnFailedReclaimed, OutcomeUpstreamInterrupted:
	default:
		return "not_terminal"
	}
	if _, err := os.Stat(preflight.WriterLockPath); errors.Is(err, os.ErrNotExist) {
		return "not_present"
	} else if err != nil {
		return "inspect_failed"
	}
	return "left_to_owner"
}
