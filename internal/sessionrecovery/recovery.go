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
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

const ReceiptSchema = "fak-session-recovery-receipt/1"

type InventoryReport struct {
	Sessions []Session `json:"sessions"`
}
type Session struct {
	Thread       *Thread       `json:"thread,omitempty"`
	LatestTurn   *Turn         `json:"latest_turn,omitempty"`
	GuardReceipt *GuardReceipt `json:"guard_launch_receipt,omitempty"`
	ProcessTrees []ProcessTree `json:"process_trees,omitempty"`
}
type Thread struct {
	ID     string `json:"id"`
	Source string `json:"source,omitempty"`
	CWD    string `json:"cwd,omitempty"`
}
type Turn struct {
	Status    string `json:"status,omitempty"`
	StartedAt string `json:"started_at,omitempty"`
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
	ThreadID    string   `json:"thread_id"`
	CWD         string   `json:"cwd,omitempty"`
	Source      string   `json:"source,omitempty"`
	Argv        []string `json:"argv"`
	Status      string   `json:"status"`
	Reason      string   `json:"reason,omitempty"`
	ReceiptPath string   `json:"receipt_path,omitempty"`
	Launched    bool     `json:"launched,omitempty"`
}

type Receipt struct {
	Schema     string   `json:"schema"`
	ThreadID   string   `json:"thread_id"`
	CWD        string   `json:"cwd"`
	Argv       []string `json:"argv"`
	RecordedAt string   `json:"recorded_at"`
	UpdatedAt  string   `json:"updated_at,omitempty"`
	State      string   `json:"state"`
	Reason     string   `json:"reason,omitempty"`
}

type Options struct {
	ManagerBin  string
	Threads     map[string]bool
	Limit       int
	CWDOverride string
	Prompt      string
	ReceiptDir  string
	CodexBin    string
}

func Select(report InventoryReport, opts Options) []Request {
	if opts.Limit <= 0 {
		opts.Limit = 1
	}
	if opts.CodexBin == "" {
		opts.CodexBin = "codex"
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
	out := make([]Request, 0, opts.Limit)
	found := make(map[string]bool, len(opts.Threads))
	for _, row := range rows {
		if len(out) >= opts.Limit || row.Thread == nil {
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
		req := Request{ThreadID: id, CWD: cwd, Source: row.Thread.Source, Status: status, Reason: reason}
		if status != "candidate" {
			if explicit {
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
		req.Argv = []string{opts.ManagerBin, "guard", "--", opts.CodexBin, "resume", id}
		if opts.Prompt != "" {
			req.Argv = append(req.Argv, opts.Prompt)
		}
		req.ReceiptPath = filepath.Join(opts.ReceiptDir, receiptName(id, cwd, req.Argv)+".json")
		out = append(out, req)
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

func recoveryEligibility(row Session, explicit bool) (string, string) {
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

// MergeJournalCrashes adds machine-reboot and dead-process candidates from the
// durable session journal. Journal records are authoritative for cwd: unlike transcript
// project slugs, the recorded path is reversible and survives a machine reboot.
func MergeJournalCrashes(requests []Request, classified []sessionjournal.Classified, opts Options) []Request {
	merged := make([]Request, 0, opts.Limit)
	seen := make(map[string]bool, len(requests))
	for _, row := range classified {
		if len(merged) >= opts.Limit || row.Status != sessionjournal.StatusCrashed || seen[row.Session.ID] {
			continue
		}
		if len(opts.Threads) > 0 && !opts.Threads[row.Session.ID] {
			continue
		}
		cwd := row.Session.CWD
		if cwd == "" {
			cwd = opts.CWDOverride
		}
		req := Request{ThreadID: row.Session.ID, CWD: cwd, Source: "session_journal", Status: "candidate", Reason: row.Reason}
		if cwd == "" {
			req.Status = "skipped"
			req.Reason = "cwd_unknown"
		} else {
			managerBin := opts.ManagerBin
			if managerBin == "" {
				managerBin = "fak"
			}
			codexBin := opts.CodexBin
			if codexBin == "" {
				codexBin = "codex"
			}
			req.Argv = []string{managerBin, "guard", "--", codexBin, "resume", row.Session.ID}
			if opts.Prompt != "" {
				req.Argv = append(req.Argv, opts.Prompt)
			}
			req.ReceiptPath = filepath.Join(opts.ReceiptDir, receiptName(req.ThreadID, req.CWD, req.Argv)+".json")
		}
		merged = append(merged, req)
		seen[row.Session.ID] = true
	}
	for _, req := range requests {
		if len(merged) >= opts.Limit || seen[req.ThreadID] {
			continue
		}
		merged = append(merged, req)
		seen[req.ThreadID] = true
	}
	return merged
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
	encErr := json.NewEncoder(f).Encode(Receipt{Schema: ReceiptSchema, ThreadID: req.ThreadID, CWD: req.CWD, Argv: req.Argv, RecordedAt: now.UTC().Format(time.RFC3339Nano), State: "launch_intent"})
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

type Launcher interface{ Launch(Request) error }
type VisibleLauncher struct{ TerminalBin string }

func (l VisibleLauncher) Launch(req Request) error {
	if len(req.Argv) == 0 {
		return errors.New("visible launch: empty argv")
	}
	command, err := exec.LookPath(req.Argv[0])
	if err != nil {
		return fmt.Errorf("visible launch: resolve %q: %w", req.Argv[0], err)
	}
	bin := l.TerminalBin
	if bin == "" {
		bin = "wt.exe"
	}
	args := []string{"-w", "new", "new-tab", "--startingDirectory", req.CWD, "--", command}
	args = append(args, req.Argv[1:]...)
	cmd := exec.Command(bin, args...)
	cmd.Dir = req.CWD
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("visible launch: %w", err)
	}
	return cmd.Process.Release()
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
