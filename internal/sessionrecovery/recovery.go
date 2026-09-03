package sessionrecovery

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

const ReceiptSchema = "fak-session-recovery-receipt/1"

const (
	ProviderClaude = "claude"
	ProviderCodex  = "codex"

	CategoryProbe           = "probe"
	CategorySubstantive     = "substantive"
	CategoryLive            = "live"
	CategoryIdentityBlocked = "identity_blocked"

	ActionRecover         = "recover"
	ActionExcludeProbe    = "exclude_probe"
	ActionLeaveLive       = "leave_live"
	ActionLoginRequired   = "login_required"
	ActionResolveIdentity = "resolve_identity"
	ActionWaitReset       = "wait_reset"
)

type InventoryReport struct {
	ObservedAt    string    `json:"observed_at,omitempty"`
	WindowSeconds int64     `json:"window_seconds,omitempty"`
	Sessions      []Session `json:"sessions"`
}
type Session struct {
	Thread             *Thread       `json:"thread,omitempty"`
	LatestTurn         *Turn         `json:"latest_turn,omitempty"`
	GuardReceipt       *GuardReceipt `json:"guard_launch_receipt,omitempty"`
	ProcessTrees       []ProcessTree `json:"process_trees,omitempty"`
	Provider           string        `json:"provider,omitempty"`
	Harness            string        `json:"harness,omitempty"`
	HarnessSource      string        `json:"harness_source,omitempty"`
	Category           string        `json:"category,omitempty"`
	Action             string        `json:"action,omitempty"`
	Reason             string        `json:"reason,omitempty"`
	Bucket             string        `json:"bucket,omitempty"`
	Cursor             string        `json:"progress_cursor,omitempty"`
	CursorAt           string        `json:"progress_at,omitempty"`
	HostHandle         string        `json:"host_handle,omitempty"`
	HostHandles        []string      `json:"host_handles,omitempty"`
	IdentityProvenance string        `json:"identity_provenance,omitempty"`
}
type Thread struct {
	ID        string `json:"id"`
	Source    string `json:"source,omitempty"`
	CWD       string `json:"cwd,omitempty"`
	CreatedAt string `json:"created_at,omitempty"`
	UpdatedAt string `json:"updated_at,omitempty"`
}
type Turn struct {
	ID          string `json:"id,omitempty"`
	Status      string `json:"status,omitempty"`
	StartedAt   string `json:"started_at,omitempty"`
	CompletedAt string `json:"completed_at,omitempty"`
}
type GuardReceipt struct {
	RecordedAt string `json:"recorded_at,omitempty"`
}
type ProcessTree struct {
	RootPID  int  `json:"root_pid,omitempty"`
	HasCodex bool `json:"has_codex,omitempty"`
	HasGuard bool `json:"has_fak_guard,omitempty"`
}

type Request struct {
	ThreadID             string   `json:"thread_id"`
	CWD                  string   `json:"cwd,omitempty"`
	Source               string   `json:"source,omitempty"`
	Provider             string   `json:"provider,omitempty"`
	Harness              string   `json:"harness,omitempty"`
	HarnessSource        string   `json:"harness_source,omitempty"`
	Category             string   `json:"category,omitempty"`
	Action               string   `json:"action,omitempty"`
	Argv                 []string `json:"argv"`
	PromptPath           string   `json:"-"`
	Prompt               string   `json:"-"`
	Status               string   `json:"status"`
	Reason               string   `json:"reason,omitempty"`
	ReceiptPath          string   `json:"receipt_path,omitempty"`
	Launched             bool     `json:"launched,omitempty"`
	HostHandles          []string `json:"host_handles,omitempty"`
	IdentityProvenance   string   `json:"identity_provenance,omitempty"`
	QualifyingEvidenceAt string   `json:"qualifying_evidence_at,omitempty"`
}

type Receipt struct {
	Schema               string   `json:"schema"`
	ThreadID             string   `json:"thread_id"`
	Provider             string   `json:"provider,omitempty"`
	Harness              string   `json:"harness,omitempty"`
	HarnessSource        string   `json:"harness_source,omitempty"`
	Category             string   `json:"category,omitempty"`
	CWD                  string   `json:"cwd"`
	Argv                 []string `json:"argv"`
	RecordedAt           string   `json:"recorded_at"`
	UpdatedAt            string   `json:"updated_at,omitempty"`
	State                string   `json:"state"`
	Reason               string   `json:"reason,omitempty"`
	HostHandles          []string `json:"host_handles,omitempty"`
	IdentityProvenance   string   `json:"identity_provenance,omitempty"`
	QualifyingEvidenceAt string   `json:"qualifying_evidence_at,omitempty"`
}

type Options struct {
	ManagerBin  string
	Threads     map[string]bool
	Limit       int
	CWDOverride string
	Prompt      string
	PromptPath  string
	ReceiptDir  string
	CodexBin    string
	Now         time.Time
	Since       time.Duration
}

func Select(report InventoryReport, opts Options) []Request {
	opts = optionsFromReport(report, opts)
	if opts.Limit <= 0 {
		opts.Limit = 1
	}
	if opts.CodexBin == "" {
		opts.CodexBin = "codex"
	}
	if opts.Prompt != "" && opts.ReceiptDir != "" {
		opts.PromptPath = filepath.Join(opts.ReceiptDir, "prompts", promptName(opts.Prompt)+".txt")
	}
	if opts.ManagerBin == "" {
		opts.ManagerBin = "fak"
	}
	rows := append([]Session(nil), report.Sessions...)
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].Thread == nil {
			return false
		}
		if rows[j].Thread == nil {
			return true
		}
		return rows[i].Thread.ID < rows[j].Thread.ID
	})
	// Limit is a logical launch cap, not an allocation request. In particular,
	// --live --all uses an effectively unbounded limit while the cohort remains
	// bounded by discovered rows and explicit IDs.
	out := make([]Request, 0, len(rows)+len(opts.Threads))
	selected := 0
	found := make(map[string]bool, len(opts.Threads))
	for _, row := range rows {
		if row.Thread == nil {
			continue
		}
		id := row.Thread.ID
		explicit := len(opts.Threads) > 0 && opts.Threads[id]
		if len(opts.Threads) > 0 && !explicit {
			continue
		}
		if explicit {
			found[id] = true
		}
		status, reason := recoveryEligibility(row, explicit)
		cwd := strings.TrimSpace(row.Thread.CWD)
		if opts.CWDOverride != "" {
			cwd = opts.CWDOverride
		}
		provider, harnessSource := selectedHarness(row)
		category := row.Category
		if category == "" {
			category = CategorySubstantive
		}
		evidenceAt, evidenceKnown := qualifyingEvidenceAt(row)
		if status == "candidate" && !withinEvidenceWindow(evidenceAt, evidenceKnown, opts) {
			continue
		}
		req := Request{ThreadID: id, CWD: cwd, Source: row.Thread.Source, Provider: provider, Harness: provider, HarnessSource: harnessSource, Category: category, Action: row.Action, Status: status, Reason: reason, HostHandles: append([]string(nil), row.HostHandles...), IdentityProvenance: row.IdentityProvenance, QualifyingEvidenceAt: stampIfKnown(evidenceAt, evidenceKnown)}
		if status != "candidate" {
			// Unified cohort rows remain in the witness even when they are probes,
			// live, waiting for a reset, or identity-blocked. Legacy Codex inventory
			// keeps its historical broad-preview filtering.
			if explicit || row.Provider != "" || row.HostHandle != "" || len(row.HostHandles) > 0 {
				out = append(out, req)
			}
			continue
		}
		if selected >= opts.Limit {
			if row.Provider != "" || row.HostHandle != "" || len(row.HostHandles) > 0 {
				req.Status = "deferred"
				req.Reason = "launch_limit"
				out = append(out, req)
			}
			continue
		}
		if cwd == "" {
			req.Status = "refused"
			req.Reason = "cwd_unknown"
			out = append(out, req)
			continue
		}
		switch provider {
		case ProviderClaude, ProviderCodex:
			req.Argv = recoveryResumeArgv(opts, provider, id, cwd)
		default:
			req.Status = "identity_blocked"
			req.Category = CategoryIdentityBlocked
			req.Action = ActionResolveIdentity
			req.Reason = "harness_identity_unavailable"
			out = append(out, req)
			continue
		}
		req.PromptPath = opts.PromptPath
		req.Prompt = opts.Prompt
		req.ReceiptPath = filepath.Join(opts.ReceiptDir, receiptName(id, cwd, req.Argv)+".json")
		out = append(out, req)
		selected++
	}
	if len(opts.Threads) > 0 && len(out) < opts.Limit {
		ids := make([]string, 0, len(opts.Threads))
		for id := range opts.Threads {
			ids = append(ids, id)
		}
		sort.Strings(ids)
		for _, id := range ids {
			if len(out) >= opts.Limit {
				break
			}
			if !found[id] {
				out = append(out, Request{ThreadID: id, Status: "refused", Reason: "thread_not_found"})
			}
		}
	}
	return out
}

func recoveryResumeArgv(opts Options, provider, threadID, cwd string) []string {
	if opts.PromptPath == "" {
		if provider == ProviderClaude {
			return claudeResumeArgv(opts.ManagerBin, threadID)
		}
		return codexResumeArgv(opts.ManagerBin, opts.CodexBin, threadID, cwd)
	}
	argv := []string{opts.ManagerBin, "session", "recover", "--provider-launch", provider, "--thread", threadID, "--cwd", cwd, "--prompt-file", opts.PromptPath}
	if provider == ProviderCodex {
		argv = append(argv, "--codex", opts.CodexBin)
	}
	return argv
}

func codexResumeArgv(managerBin, codexBin, threadID, cwd string) []string {
	// Recovery waves must not open several interactive Codex TUIs. `exec resume`
	// accepts the exact state_5 UUID and runs independently in its wrapper window.
	// Keep --cd in provider argv as well as the launcher's starting directory:
	// Codex itself must retain the recovered thread's workspace root.
	return []string{managerBin, "guard", "--", codexBin, "exec", "--cd", cwd, "resume", threadID}
}

func claudeResumeArgv(managerBin, sessionID string) []string {
	return []string{managerBin, "guard", "--", "claude", "--resume", sessionID}
}

func recoveryEligibility(row Session, explicit bool) (string, string) {
	switch row.Action {
	case ActionExcludeProbe:
		return "probe", firstNonBlank(row.Reason, "semantic_probe")
	case ActionLeaveLive:
		return "live", firstNonBlank(row.Reason, "live_process_tree")
	case ActionLoginRequired, ActionResolveIdentity:
		return "identity_blocked", firstNonBlank(row.Reason, "login_required")
	case ActionWaitReset:
		return "waiting_reset", firstNonBlank(row.Reason, "reset_not_elapsed")
	case ActionRecover:
		if len(row.ProcessTrees) != 0 {
			return "live", "live_process_tree"
		}
		return "candidate", row.Reason
	}
	if row.Category == CategoryIdentityBlocked {
		return "identity_blocked", firstNonBlank(row.Reason, "exact_identity_unresolved")
	}
	if len(row.ProcessTrees) != 0 {
		return "already_active", "live_process_tree"
	}
	if row.LatestTurn == nil {
		// An explicit exec recovery is operator-directed and process-tree gated.
		// Exec history can omit latest_turn after an abrupt process kill even when
		// the inventory retains the thread ID and authoritative working directory.
		if explicit && row.Thread.Source == "exec" {
			return "candidate", "explicit_dead_exec_no_turn"
		}
		return "refused", "latest_turn_unknown"
	}
	if !strings.EqualFold(row.LatestTurn.Status, "inProgress") {
		return "refused", "latest_turn_" + strings.ToLower(strings.TrimSpace(row.LatestTurn.Status))
	}
	if strings.EqualFold(strings.TrimSpace(row.Thread.Source), "cli") {
		provider, harnessSource := selectedHarness(row)
		// Current Codex state_5 rows use source=cli. Admit that label only when
		// inventory supplied an exact Codex harness identity and UUID; the legacy
		// source fallback must not turn an otherwise unknown cli row into Codex.
		if provider == ProviderCodex && harnessSource != "legacy_source" && isUUID(row.Thread.ID) {
			return "candidate", ""
		}
		return "refused", "source_not_resumable:" + row.Thread.Source
	}
	switch row.Thread.Source {
	case "interactive_tui", "resume_wrapper":
		return "candidate", ""
	case "exec":
		if explicit {
			return "candidate", "explicit_dead_exec"
		}
		return "refused", "exec_requires_explicit_thread"
	default:
		return "refused", "source_not_resumable:" + row.Thread.Source
	}
}

func optionsFromReport(report InventoryReport, opts Options) Options {
	if opts.Now.IsZero() {
		if observedAt, ok := parseEvidenceStamp(report.ObservedAt); ok {
			opts.Now = observedAt
		}
	}
	if opts.Since <= 0 && report.WindowSeconds > 0 {
		opts.Since = time.Duration(report.WindowSeconds) * time.Second
	}
	return opts
}

func qualifyingEvidenceAt(row Session) (time.Time, bool) {
	if row.LatestTurn != nil {
		if at, ok := parseEvidenceStamp(row.LatestTurn.StartedAt); ok {
			return at, true
		}
	}
	if row.Action == ActionRecover || (row.Thread != nil && row.Thread.Source == "exec") {
		if row.Thread != nil {
			if at, ok := parseEvidenceStamp(row.Thread.UpdatedAt); ok {
				return at, true
			}
			if at, ok := parseEvidenceStamp(row.Thread.CreatedAt); ok {
				return at, true
			}
		}
	}
	return time.Time{}, false
}

func journalQualifyingEvidenceAt(row sessionjournal.Session) (time.Time, bool) {
	if !row.LastSeen.IsZero() {
		return row.LastSeen.UTC(), true
	}
	if !row.StartedAt.IsZero() {
		return row.StartedAt.UTC(), true
	}
	return time.Time{}, false
}

func parseEvidenceStamp(value string) (time.Time, bool) {
	value = strings.TrimSpace(value)
	if value == "" {
		return time.Time{}, false
	}
	at, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		return time.Time{}, false
	}
	return at.UTC(), true
}

func withinEvidenceWindow(at time.Time, known bool, opts Options) bool {
	if opts.Since <= 0 {
		return true
	}
	if !known {
		return false
	}
	now := opts.Now.UTC()
	if now.IsZero() {
		now = time.Now().UTC()
	}
	return !at.Before(now.Add(-opts.Since))
}

func stampIfKnown(at time.Time, known bool) string {
	if !known {
		return ""
	}
	return at.UTC().Format(time.RFC3339Nano)
}

func selectedHarness(row Session) (string, string) {
	if harness := providerName(row.Harness); harness != "" {
		return harness, firstNonBlank(row.HarnessSource, "inventory.harness")
	}
	if provider := providerName(row.Provider); provider != "" {
		return provider, "inventory.provider"
	}
	if inferred := normalizeProvider("", row.Thread.Source); inferred != "" {
		return inferred, "legacy_source"
	}
	return "", ""
}

func normalizeProvider(provider, source string) string {
	provider = strings.ToLower(strings.TrimSpace(provider))
	if provider != "" {
		return provider
	}
	if strings.Contains(strings.ToLower(source), "claude") {
		return ProviderClaude
	}
	if strings.Contains(strings.ToLower(source), "codex") {
		return ProviderCodex
	}
	if strings.EqualFold(strings.TrimSpace(source), "session_registration") {
		return ""
	}
	// Legacy inventory rows predate explicit harness identity and are Codex-only.
	return ProviderCodex
}

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

// MergeJournalCrashes adds machine-reboot and dead-process candidates from the
// durable session journal. Journal records are authoritative for cwd: unlike transcript
// project slugs, the recorded path is reversible and survives a machine reboot.
func MergeJournalCrashes(requests []Request, classified []sessionjournal.Classified, opts Options) []Request {
	byID := make(map[string]Request, len(requests))
	for _, req := range requests {
		byID[req.ThreadID] = req
	}
	merged := make([]Request, 0, len(requests)+len(classified))
	seen := make(map[string]bool, len(requests)+len(classified))
	for _, row := range classified {
		if row.Status != sessionjournal.StatusCrashed || seen[row.Session.ID] {
			continue
		}
		evidenceAt, evidenceKnown := journalQualifyingEvidenceAt(row.Session)
		if !withinEvidenceWindow(evidenceAt, evidenceKnown, opts) {
			continue
		}
		if len(opts.Threads) > 0 && !opts.Threads[row.Session.ID] {
			continue
		}
		existing, hasExisting := byID[row.Session.ID]
		cwd := strings.TrimSpace(row.Session.CWD)
		if cwd == "" {
			cwd = firstNonBlank(opts.CWDOverride, existing.CWD)
		}
		req := existing
		if !hasExisting {
			req = Request{ThreadID: row.Session.ID, Category: CategorySubstantive, Action: ActionRecover, Status: "candidate", Reason: row.Reason}
		}
		req.CWD = cwd
		req.Source = "session_journal"
		req.Provider = journalProvider(row.Session, existing)
		req.QualifyingEvidenceAt = stampIfKnown(evidenceAt, evidenceKnown)
		req.Harness = req.Provider
		if req.HarnessSource == "" {
			req.HarnessSource = journalHarnessSource(row.Session, existing)
		}
		prepareJournalRequest(&req, hasExisting, opts)
		merged = append(merged, req)
		seen[row.Session.ID] = true
	}
	for _, req := range requests {
		if seen[req.ThreadID] {
			continue
		}
		merged = append(merged, req)
		seen[req.ThreadID] = true
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = 1
	}
	launches := 0
	for i := range merged {
		if merged[i].Status != "candidate" {
			continue
		}
		if launches >= limit {
			merged[i].Status = "deferred"
			merged[i].Reason = "launch_limit"
			merged[i].Argv = nil
			merged[i].ReceiptPath = ""
			continue
		}
		launches++
	}
	return merged
}

func journalProvider(row sessionjournal.Session, existing Request) string {
	if provider := providerName(existing.Harness); provider != "" {
		return provider
	}
	if provider := providerName(existing.Provider); provider != "" {
		return provider
	}
	if provider := providerName(row.Agent); provider != "" {
		return provider
	}
	for _, arg := range row.Argv {
		if provider := providerName(arg); provider != "" {
			return provider
		}
	}
	return ""
}

func journalHarnessSource(row sessionjournal.Session, existing Request) string {
	if strings.TrimSpace(existing.HarnessSource) != "" {
		return existing.HarnessSource
	}
	if providerName(row.Agent) != "" {
		return "session_journal.agent"
	}
	for _, arg := range row.Argv {
		if providerName(arg) != "" {
			return "session_journal.argv"
		}
	}
	return ""
}

func providerName(value string) string {
	base := strings.ToLower(filepath.Base(strings.ReplaceAll(strings.TrimSpace(value), `\`, "/")))
	base = strings.TrimSuffix(strings.TrimSuffix(strings.TrimSuffix(base, ".exe"), ".cmd"), ".bat")
	switch {
	case base == ProviderClaude || strings.Contains(base, ProviderClaude):
		return ProviderClaude
	case base == ProviderCodex || strings.Contains(base, ProviderCodex):
		return ProviderCodex
	case base == "opencode" || strings.Contains(base, "opencode"):
		return "opencode"
	default:
		return ""
	}
}

func prepareJournalRequest(req *Request, hasStateRequest bool, opts Options) {
	if req.Status != "candidate" {
		return
	}
	if req.CWD == "" {
		req.Status = "skipped"
		req.Reason = "cwd_unknown"
		return
	}
	managerBin := firstNonBlank(opts.ManagerBin, "fak")
	codexBin := firstNonBlank(opts.CodexBin, "codex")
	if opts.Prompt != "" && opts.ReceiptDir != "" {
		opts.PromptPath = filepath.Join(opts.ReceiptDir, "prompts", promptName(opts.Prompt)+".txt")
	}
	opts.ManagerBin = managerBin
	opts.CodexBin = codexBin
	switch req.Provider {
	case ProviderClaude:
		req.Argv = recoveryResumeArgv(opts, req.Provider, req.ThreadID, req.CWD)
	case ProviderCodex:
		if !hasStateRequest || !isUUID(req.ThreadID) {
			req.Category = CategoryIdentityBlocked
			req.Action = ActionResolveIdentity
			req.Status = "identity_blocked"
			req.Reason = "codex_journal_uuid_not_verified_in_state_5"
			req.Argv = nil
			return
		}
		req.Argv = recoveryResumeArgv(opts, req.Provider, req.ThreadID, req.CWD)
	default:
		req.Category = CategoryIdentityBlocked
		req.Action = ActionResolveIdentity
		req.Status = "identity_blocked"
		req.Reason = "exact_resume_provider_blocked:" + req.Provider
		req.Argv = nil
		return
	}
	req.PromptPath = opts.PromptPath
	req.Prompt = opts.Prompt
	req.ReceiptPath = filepath.Join(opts.ReceiptDir, receiptName(req.ThreadID, req.CWD, req.Argv)+".json")
}

func promptName(prompt string) string {
	h := sha256.Sum256([]byte(prompt))
	return hex.EncodeToString(h[:12])
}

func StagePrompt(req Request) error {
	if req.PromptPath == "" || req.Prompt == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(req.PromptPath), 0700); err != nil {
		return err
	}
	f, err := os.OpenFile(req.PromptPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if errors.Is(err, os.ErrExist) {
		b, readErr := os.ReadFile(req.PromptPath)
		if readErr != nil {
			return readErr
		}
		if string(b) != req.Prompt {
			return errors.New("recovery prompt file content mismatch")
		}
		return nil
	}
	if err != nil {
		return err
	}
	_, writeErr := f.WriteString(req.Prompt)
	closeErr := f.Close()
	if writeErr != nil {
		return writeErr
	}
	return closeErr
}

func receiptName(id, cwd string, argv []string) string {
	h := sha256.Sum256([]byte(id + "\x00" + cwd + "\x00" + strings.Join(argv, "\x00")))
	return hex.EncodeToString(h[:12])
}

func WriteReceipt(req Request, now time.Time) (bool, error) {
	if req.Status != "candidate" {
		return false, errors.New("request is not launchable")
	}
	if err := os.MkdirAll(filepath.Dir(req.ReceiptPath), 0700); err != nil {
		return false, err
	}
	f, err := os.OpenFile(req.ReceiptPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if errors.Is(err, os.ErrExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	encErr := json.NewEncoder(f).Encode(Receipt{Schema: ReceiptSchema, ThreadID: req.ThreadID, Provider: req.Provider, Harness: req.Harness, HarnessSource: req.HarnessSource, Category: req.Category, HostHandles: append([]string(nil), req.HostHandles...), IdentityProvenance: req.IdentityProvenance, QualifyingEvidenceAt: req.QualifyingEvidenceAt, CWD: req.CWD, Argv: req.Argv, RecordedAt: now.UTC().Format(time.RFC3339Nano), State: "launch_intent"})
	closeErr := f.Close()
	if encErr != nil {
		return false, encErr
	}
	return true, closeErr
}

func FinalizeReceipt(req Request, state, reason string, now time.Time) error {
	if strings.TrimSpace(state) == "" || state == "launch_intent" {
		return errors.New("receipt final state is required")
	}
	b, err := os.ReadFile(req.ReceiptPath)
	if err != nil {
		return err
	}
	var receipt Receipt
	if err := json.Unmarshal(b, &receipt); err != nil {
		return err
	}
	if receipt.Schema != ReceiptSchema || receipt.ThreadID != req.ThreadID {
		return errors.New("receipt identity mismatch")
	}
	receipt.State = state
	receipt.Reason = reason
	receipt.UpdatedAt = now.UTC().Format(time.RFC3339Nano)
	tmp, err := os.CreateTemp(filepath.Dir(req.ReceiptPath), ".session-recovery-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0600); err != nil {
		tmp.Close()
		return err
	}
	if err := json.NewEncoder(tmp).Encode(receipt); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, req.ReceiptPath)
}

// LaunchHandle is cleanup authority for exactly one process tree created by a
// recovery launch. Identity is immutable evidence; callers must never infer
// cleanup ownership from a later broad inventory snapshot.
type LaunchHandle interface {
	Identity() string
	Reap() error
}

type Launcher interface {
	Launch(Request) (LaunchHandle, error)
}

type VisibleLauncher struct{ TerminalBin string }

type processLaunchHandle struct {
	pid   int
	start string
}

func (h processLaunchHandle) Identity() string {
	if h.start == "" {
		return fmt.Sprintf("pid:%d", h.pid)
	}
	return fmt.Sprintf("pid:%d@%s", h.pid, h.start)
}

func (h processLaunchHandle) Reap() error {
	procs, collectErr := procguard.CollectProcesses()
	if collectErr != "" {
		return fmt.Errorf("verify launch identity before reap: %s", collectErr)
	}
	found := false
	for _, process := range procs {
		if process.PID != h.pid {
			continue
		}
		found = true
		if h.start != "" && process.Start != h.start {
			return fmt.Errorf("launch identity changed for pid %d", h.pid)
		}
		break
	}
	if !found {
		return nil
	}
	if ok, detail := procguard.KillPID(h.pid); !ok {
		if detail == "" {
			detail = "process-tree reaper refused"
		}
		return errors.New(detail)
	}
	return nil
}

type visibleLaunchPlan struct {
	bin   string
	args  []string
	dir   string
	stdin string
}

const terminalAppleScript = `on run argv
	set workingDirectory to item 1 of argv
	set executablePath to item 2 of argv
	set commandText to "cd -- " & quoted form of workingDirectory & " && exec " & quoted form of executablePath
	repeat with argumentIndex from 3 to count of argv
		set commandText to commandText & " " & quoted form of (item argumentIndex of argv)
	end repeat
	tell application "Terminal"
		activate
		do script commandText
	end tell
end run
`

func (l VisibleLauncher) Launch(req Request) (LaunchHandle, error) {
	if len(req.Argv) == 0 {
		return nil, errors.New("visible launch: empty argv")
	}
	command, err := exec.LookPath(req.Argv[0])
	if err != nil {
		return nil, fmt.Errorf("visible launch: resolve %q: %w", req.Argv[0], err)
	}
	plan, err := planVisibleLaunch(runtime.GOOS, l.TerminalBin, req, command)
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(plan.bin, plan.args...)
	cmd.Dir = plan.dir
	if plan.stdin != "" {
		cmd.Stdin = strings.NewReader(plan.stdin)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("visible launch: %w", err)
	}
	pid := cmd.Process.Pid
	start := ""
	if procs, collectErr := procguard.CollectProcesses(); collectErr == "" {
		for _, process := range procs {
			if process.PID == pid {
				start = process.Start
				break
			}
		}
	}
	if err := cmd.Process.Release(); err != nil {
		return nil, err
	}
	return processLaunchHandle{pid: pid, start: start}, nil
}

func planVisibleLaunch(goos, terminalBin string, req Request, command string) (visibleLaunchPlan, error) {
	switch goos {
	case "darwin":
		bin := terminalBin
		if bin == "" {
			bin = "/usr/bin/osascript"
		}
		// A program file of "-" makes every following value run-handler data.
		// The AppleScript source is constant; cwd, command, and arguments never
		// cross into source text and are shell-quoted by AppleScript at runtime.
		args := []string{"-", req.CWD, command}
		args = append(args, req.Argv[1:]...)
		return visibleLaunchPlan{bin: bin, args: args, dir: req.CWD, stdin: terminalAppleScript}, nil
	case "windows":
		bin := terminalBin
		if bin == "" {
			bin = "wt.exe"
		}
		return windowsVisibleLaunchPlan(bin, req, command), nil
	default:
		if terminalBin == "" {
			return visibleLaunchPlan{}, fmt.Errorf("visible launch: unsupported platform %q without an injected terminal", goos)
		}
		// TerminalBin has historically meant a Windows-Terminal-compatible
		// launcher. Preserve that explicit injection contract on other hosts.
		return windowsVisibleLaunchPlan(terminalBin, req, command), nil
	}
}

func windowsVisibleLaunchPlan(bin string, req Request, command string) visibleLaunchPlan {
	args := []string{"-w", "new", "new-tab", "--startingDirectory", req.CWD, "--", command}
	args = append(args, req.Argv[1:]...)
	return visibleLaunchPlan{bin: bin, args: args, dir: req.CWD}
}

func Witness(before, after InventoryReport, threadID string) string {
	var b, a *Session
	for i := range before.Sessions {
		if before.Sessions[i].Thread != nil && before.Sessions[i].Thread.ID == threadID {
			b = &before.Sessions[i]
		}
	}
	for i := range after.Sessions {
		if after.Sessions[i].Thread != nil && after.Sessions[i].Thread.ID == threadID {
			a = &after.Sessions[i]
		}
	}
	if a == nil || len(a.ProcessTrees) == 0 || a.GuardReceipt == nil {
		return "launched_unproven"
	}
	if a.LatestTurn != nil && strings.EqualFold(a.LatestTurn.Status, "completed") {
		return "completed"
	}
	if a.LatestTurn == nil {
		return "active"
	}
	if b == nil || b.LatestTurn == nil || a.LatestTurn.StartedAt > b.LatestTurn.StartedAt {
		return "productive"
	}
	return "active"
}
