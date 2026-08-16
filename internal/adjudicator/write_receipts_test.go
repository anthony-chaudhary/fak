package adjudicator

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func receiptCall(trace, tool, args string, seq uint64) *abi.ToolCall {
	return &abi.ToolCall{TraceID: trace, Tool: tool, SeqNo: seq, Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)}}
}

func TestCanonicalLocalReceiptPathResolvesRootSymlink(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Windows temp roots are not symlink aliases")
	}
	realRoot := t.TempDir()
	aliasRoot := filepath.Join(t.TempDir(), "workspace")
	if err := os.Symlink(realRoot, aliasRoot); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	target := filepath.Join(aliasRoot, "scratch.txt")
	if err := os.WriteFile(target, []byte("scratch"), 0644); err != nil {
		t.Fatal(err)
	}
	got, ok := canonicalLocalReceiptPath(aliasRoot, target)
	wantRoot, err := filepath.EvalSymlinks(realRoot)
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(wantRoot, "scratch.txt")
	if !ok || got != want {
		t.Fatalf("canonicalLocalReceiptPath = %q,%v, want %q,true", got, ok, want)
	}
}

func TestWriteReceiptPostExecutionBoundary(t *testing.T) {
	rootBytes, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Clean(filepath.Join(rootBytes, "..", ".."))
	target := filepath.Join(root, "zz_receipt_boundary.tmp")
	alias := filepath.Join(root, "internal", "..", "zz_receipt_boundary.tmp")
	a := New(Policy{})
	call := receiptCall("trace-a", "write_file", fmt.Sprintf(`{"path":%q}`, target), 41)

	if _, ok := a.AuthoredPath("trace-a", target); ok {
		t.Fatal("intent alone created receipt")
	}
	a.ObserveResult(context.Background(), call, &abi.Result{Call: call, Status: abi.StatusError, Outcome: abi.OutcomeCommitted})
	if _, ok := a.AuthoredPath("trace-a", target); ok {
		t.Fatal("failed execution created receipt")
	}
	a.ObserveResult(context.Background(), call, &abi.Result{Call: call, Status: abi.StatusOK, Outcome: abi.OutcomeCommitted})
	if op, ok := a.AuthoredPath("trace-a", alias); !ok || op != 41 {
		t.Fatalf("same-trace alias read-back = %d,%v", op, ok)
	}
	if _, ok := a.AuthoredPath("trace-b", target); ok {
		t.Fatal("cross-trace receipt claim admitted")
	}
	a.ResetRun()
	if _, ok := a.AuthoredPath("trace-a", target); ok {
		t.Fatal("reset retained receipt")
	}
}

func TestWriteReceiptFailsClosedAndEvicts(t *testing.T) {
	rootBytes, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Clean(filepath.Join(rootBytes, "..", ".."))
	a := New(Policy{})
	first := filepath.Join(root, "zz_receipt_000.tmp")
	for i := 0; i <= writeReceiptLimit; i++ {
		target := filepath.Join(root, fmt.Sprintf("zz_receipt_%03d.tmp", i))
		call := receiptCall("trace-a", "write_file", fmt.Sprintf(`{"path":%q}`, target), uint64(i+1))
		a.ObserveResult(context.Background(), call, &abi.Result{Call: call, Status: abi.StatusOK, Outcome: abi.OutcomeCommitted})
	}
	if _, ok := a.AuthoredPath("trace-a", first); ok {
		t.Fatal("oldest receipt survived bounded eviction")
	}
	if _, ok := a.AuthoredPath("", filepath.Join(root, "zz_receipt_256.tmp")); ok {
		t.Fatal("empty trace admitted")
	}
	if _, ok := a.AuthoredPath("trace-a", filepath.Join(root, "..", "outside.tmp")); ok {
		t.Fatal("external target admitted")
	}
}
