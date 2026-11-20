package model

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=5ms lane=default
func TestQwen35BatchDiagnosticIncludesSplit(t *testing.T) {
	// Check the actual dispatcher, without sleep-based timing thresholds or a
	// replacement split implementation. This is an ordering guard, not a timer
	// accuracy, backend execution, or end-to-end latency witness.
	file, err := parser.ParseFile(token.NewFileSet(), "qwen35_batch_decode.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var body *ast.BlockStmt
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "stepBatchQwen35HybridQ4KIndependent" {
			body = fn.Body
		}
	}
	if body == nil {
		t.Fatal("missing independent batch dispatcher")
	}
	var split, closeSplit, diagnostic token.Pos
	var rows string
	var calls int
	ast.Inspect(body, func(node ast.Node) bool {
		if call, ok := node.(*ast.CallExpr); ok {
			if name, ok := call.Fun.(*ast.Ident); ok && name.Name == "splitScaledLogits" {
				calls++
			}
		}
		return true
	})
	for _, stmt := range body.List {
		switch stmt := stmt.(type) {
		case *ast.AssignStmt:
			if len(stmt.Lhs) != 1 || len(stmt.Rhs) != 1 {
				continue
			}
			name, ok := stmt.Lhs[0].(*ast.Ident)
			if !ok {
				continue
			}
			if name.Name == "diagSplit" {
				closeSplit = stmt.Pos()
			}
			if call, ok := stmt.Rhs[0].(*ast.CallExpr); ok {
				if fn, ok := call.Fun.(*ast.Ident); ok && fn.Name == "splitScaledLogits" {
					split, rows = stmt.Pos(), name.Name
				}
			}
		case *ast.IfStmt:
			if name, ok := stmt.Cond.(*ast.Ident); ok && name.Name == "qwen35BatchDiagEnabled" {
				diagnostic = stmt.Pos()
			}
		}
	}
	if calls != 1 || split == token.NoPos || closeSplit <= split || diagnostic <= closeSplit {
		t.Fatalf("split must execute once before its timing and diagnostic: calls=%d split=%d close=%d diagnostic=%d", calls, split, closeSplit, diagnostic)
	}
	last, ok := body.List[len(body.List)-1].(*ast.ReturnStmt)
	if !ok || last.Pos() <= diagnostic || len(last.Results) != 2 {
		t.Fatal("dispatcher must return the split rows after the diagnostic")
	}
	gotRows, rowsOK := last.Results[0].(*ast.Ident)
	accepted, acceptedOK := last.Results[1].(*ast.Ident)
	if !rowsOK || gotRows.Name != rows || !acceptedOK || accepted.Name != "true" {
		t.Fatal("dispatcher must return the already measured rows as accepted")
	}
}

// diagBoundaryBackend wraps the portable batch backend and makes the backend
// call dominate the microsecond readback, so the dispatcher's call/read
// boundary is observable in the stage-timing diagnostic.
type diagBoundaryBackend struct {
	*portableQwen35BatchBackend
	busy time.Duration
}

func (b *diagBoundaryBackend) Qwen35IndependentDecodeBatch(req compute.Qwen35IndependentDecodeBatchRequest) (compute.Qwen35IndependentDecodeBatchResult, error) {
	deadline := time.Now().Add(b.busy)
	for time.Now().Before(deadline) {
	}
	return b.portableQwen35BatchBackend.Qwen35IndependentDecodeBatch(req)
}

func newDiagBoundaryBatch(t *testing.T, lanes int) (*BatchSession, *diagBoundaryBackend, []int) {
	t.Helper()
	m := NewSynthetic(qwen35HybridTestCfg())
	inner := &portableQwen35BatchBackend{sequencePrefillBackend: newSequencePrefillBackend(m), batchPath: compute.Qwen35IndependentDecodeBatchPath}
	be := &diagBoundaryBackend{portableQwen35BatchBackend: inner, busy: 2 * time.Millisecond}
	bs := &BatchSession{M: m, Seqs: make([]*Session, lanes)}
	ids := make([]int, lanes)
	for i := range bs.Seqs {
		var err error
		bs.Seqs[i], err = m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(bs.Seqs[i].Close)
		ids[i] = 3 + 4*i
	}
	return bs, be, ids
}

// fak-test:runtime fast est=10ms lane=default
// Connected host regression for the dispatcher's stage-timing boundaries:
// backend work must land in call_us and not read_us, read/free and the row
// split must retain distinct intervals, and total must cover the measured
// split. The sibling AST guard pins the split ordering; this one pins the
// call/read boundary the AST cannot see.
func TestQwen35IndependentDecodeBatchDiagnosticBoundaries(t *testing.T) {
	oldDiag := qwen35BatchDiagEnabled
	qwen35BatchDiagEnabled = true
	t.Cleanup(func() { qwen35BatchDiagEnabled = oldDiag })

	oldStderr := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stderr = w
	t.Cleanup(func() {
		_ = w.Close()
		os.Stderr = oldStderr
	})

	bs, be, ids := newDiagBoundaryBatch(t, 2)
	got, accepted := stepBatchQwen35HybridQ4KIndependent(bs, ids)
	if !accepted || len(got) != 2 || be.batchCalls != 1 {
		t.Fatalf("accepted=%t rows=%d calls=%d", accepted, len(got), be.batchCalls)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	line := ""
	for _, l := range strings.Split(string(out), "\n") {
		if strings.HasPrefix(l, "fak_batch_model_diag ") {
			line = l
		}
	}
	if line == "" {
		t.Fatalf("no diagnostic line in output: %q", string(out))
	}
	var idsN, prep, lineage, call, backend, hal, read, split, total int
	if _, err := fmt.Sscanf(line, "fak_batch_model_diag ids=%d prep_us=%d lineage_us=%d call_us=%d backend_us=%d hal_us=%d read_us=%d split_us=%d total_us=%d",
		&idsN, &prep, &lineage, &call, &backend, &hal, &read, &split, &total); err != nil {
		t.Fatalf("parse %q: %v", line, err)
	}
	if idsN != 2 {
		t.Fatalf("ids=%d want=2", idsN)
	}
	if backend <= 0 || call < backend || hal < 0 || hal > call {
		t.Fatalf("call_us=%d backend_us=%d hal_us=%d: backend work must land inside call_us", call, backend, hal)
	}
	if read >= backend {
		t.Fatalf("read_us=%d backend_us=%d: read interval absorbed the backend call", read, backend)
	}
	if total < split {
		t.Fatalf("total_us=%d split_us=%d: total must cover the measured split", total, split)
	}
}
