package adjudicator

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// writeReceiptLimit bounds post-execution ownership evidence per adjudicator run.
// The ledger shares the existing ResetRun lifecycle with synthesized-tool state.
const writeReceiptLimit = 256

type writeReceiptKey struct {
	trace string
	path  string
}

type writeReceipt struct {
	operation uint64
	order     uint64
}

var writeReceiptOrder atomic.Uint64

// ObserveResult records local write effects only after the kernel has observed a
// successful, committed engine completion. It is intentionally a structural
// optional interface: kernels can notify adjudicators without extending the ABI.
func (a *Adjudicator) ObserveResult(_ context.Context, c *abi.ToolCall, r *abi.Result) {
	if a == nil || c == nil || r == nil || c.TraceID == "" || r.Status != abi.StatusOK || r.Outcome != abi.OutcomeCommitted {
		return
	}
	for _, target := range completedWriteTargets(c) {
		path, ok := canonicalLocalReceiptPath(a.receiptRoot, target)
		if !ok {
			continue
		}
		key := writeReceiptKey{trace: c.TraceID, path: path}
		a.authored.Store(key, writeReceipt{operation: c.SeqNo, order: writeReceiptOrder.Add(1)})
		a.trimWriteReceipts()
	}
}

// AuthoredPath reports whether path has a live successful-write receipt for the
// same trace. Empty trace IDs and paths fail closed. operation identifies the
// committed tool call and is provided for audit/read-back.
func (a *Adjudicator) hasPriorWriteReceipt(traceID, path string, before uint64) bool {
	if before == 0 {
		return false
	}
	op, ok := a.AuthoredPath(traceID, path)
	return ok && op < before
}

func (a *Adjudicator) AuthoredPath(traceID, path string) (operation uint64, ok bool) {
	if a == nil || traceID == "" || path == "" {
		return 0, false
	}
	canonical, eligible := canonicalLocalReceiptPath(a.receiptRoot, path)
	if !eligible {
		return 0, false
	}
	v, found := a.authored.Load(writeReceiptKey{trace: traceID, path: canonical})
	if !found {
		return 0, false
	}
	receipt, typed := v.(writeReceipt)
	return receipt.operation, typed
}

func (a *Adjudicator) trimWriteReceipts() {
	for {
		count := 0
		var oldestKey writeReceiptKey
		var oldest writeReceipt
		a.authored.Range(func(k, v any) bool {
			key, keyOK := k.(writeReceiptKey)
			receipt, valueOK := v.(writeReceipt)
			if !keyOK || !valueOK {
				return true
			}
			count++
			if oldest.order == 0 || receipt.order < oldest.order {
				oldestKey, oldest = key, receipt
			}
			return true
		})
		if count <= writeReceiptLimit || oldest.order == 0 {
			return
		}
		a.authored.Delete(oldestKey)
	}
}

func completedWriteTargets(c *abi.ToolCall) []string {
	args := decodeArgs(context.Background(), c)
	if len(args) == 0 {
		return nil
	}
	tool := strings.ToLower(c.Tool)
	if tool == "bash" || tool == "shell" || tool == "shell_command" || tool == "powershell" {
		command, _ := argString(args, "command")
		if command == "" || destructiveWriteCommand(command) {
			return nil
		}
		return commandWriteTargets(command)
	}
	if !writeShapedLower(tool) {
		return nil
	}
	if target := targetPath(args); target != "" {
		return []string{target}
	}
	return nil
}

func destructiveWriteCommand(command string) bool {
	for _, segment := range shellSegments(command) {
		words := shellWords(segment)
		start := commandWordStart(words)
		if start >= len(words) {
			continue
		}
		switch strings.ToLower(words[start].text) {
		case "rm", "del", "erase", "rmdir", "shred", "truncate":
			return true
		}
	}
	return false
}

func canonicalLocalReceiptPath(root, path string) (string, bool) {
	path = cleanShellOperand(path)
	if root == "" || path == "" || strings.ContainsAny(path, "*?[]{}") {
		return "", false
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", false
	}
	absolute = filepath.Clean(absolute)
	// Resolve an existing parent so aliases through directory symlinks cannot
	// manufacture a second ownership identity for a not-yet-existing target.
	if parent, err := filepath.EvalSymlinks(filepath.Dir(absolute)); err == nil {
		absolute = filepath.Join(parent, filepath.Base(absolute))
	}
	root = canonicalReceiptRoot(root)
	rel, err := filepath.Rel(root, absolute)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || filepath.IsAbs(rel) {
		return "", false
	}
	if filepath.Separator == '\\' {
		absolute = strings.ToLower(absolute)
	}
	return absolute, true
}

func canonicalReceiptRoot(root string) string {
	root = filepath.Clean(root)
	if resolved, err := filepath.EvalSymlinks(root); err == nil {
		return resolved
	}
	return root
}

// receiptWorkspaceRoot discovers the local workspace once, when the
// adjudicator is constructed. Receipt reads/writes remain subprocess-free.
func receiptWorkspaceRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		if info, err := os.Stat(filepath.Join(dir, ".git")); err == nil && info != nil {
			return filepath.Clean(dir)
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}
