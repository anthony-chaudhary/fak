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
	Argv        []string `json:"argv"`
	Status      string   `json:"status"`
	Reason      string   `json:"reason,omitempty"`
	ReceiptPath string   `json:"receipt_path,omitempty"`
}

type Receipt struct {
	Schema     string   `json:"schema"`
	ThreadID   string   `json:"thread_id"`
	CWD        string   `json:"cwd"`
	Argv       []string `json:"argv"`
	RecordedAt string   `json:"recorded_at"`
	State      string   `json:"state"`
}

type Options struct {
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
	rows := append([]Session(nil), report.Sessions...)
	sort.Slice(rows, func(i, j int) bool { return rows[i].Thread.ID < rows[j].Thread.ID })
	out := make([]Request, 0, opts.Limit)
	for _, row := range rows {
		if len(out) >= opts.Limit || row.Thread == nil || row.LatestTurn == nil || len(row.ProcessTrees) != 0 {
			continue
		}
		id := row.Thread.ID
		if len(opts.Threads) > 0 && !opts.Threads[id] {
			continue
		}
		if row.Thread.Source != "interactive_tui" && row.Thread.Source != "resume_wrapper" {
			continue
		}
		if !strings.EqualFold(row.LatestTurn.Status, "inProgress") {
			continue
		}
		cwd := strings.TrimSpace(row.Thread.CWD)
		if opts.CWDOverride != "" {
			cwd = opts.CWDOverride
		}
		req := Request{ThreadID: id, CWD: cwd}
		if cwd == "" {
			req.Status = "refused"
			req.Reason = "cwd_unknown"
			out = append(out, req)
			continue
		}
		req.Argv = []string{opts.CodexBin, "resume", id}
		if opts.Prompt != "" {
			req.Argv = append(req.Argv, opts.Prompt)
		}
		req.Status = "candidate"
		req.ReceiptPath = filepath.Join(opts.ReceiptDir, receiptName(id, cwd, req.Argv)+".json")
		out = append(out, req)
	}
	return out
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

type Launcher interface{ Launch(Request) error }
type VisibleLauncher struct{ TerminalBin string }

func (l VisibleLauncher) Launch(req Request) error {
	bin := l.TerminalBin
	if bin == "" {
		bin = "wt.exe"
	}
	args := []string{"-w", "new", "new-tab", "--startingDirectory", req.CWD, "--"}
	args = append(args, req.Argv...)
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
	if a.LatestTurn == nil {
		return "active"
	}
	if b == nil || b.LatestTurn == nil || a.LatestTurn.StartedAt > b.LatestTurn.StartedAt {
		return "productive"
	}
	return "active"
}
