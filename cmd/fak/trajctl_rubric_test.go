package main

// trajctl_rubric_test.go — issue #2544: the CLI half of the rubric gate.
// Declaring with --rubric-base-url makes ONE generation call and caches the
// rubric on the objective's ledger row; a failed generation fails the declare
// with nothing appended.

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

func TestTrajctlDeclareRubricGate(t *testing.T) {
	var calls atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"choices":[{"message":{"tool_calls":[{"function":{"name":"emit_rubric","arguments":"{\"criteria\":[{\"id\":\"c1\",\"text\":\"stay in the docs tree\"},{\"id\":\"c2\",\"text\":\"finish within budget\"}]}"}}]}}],"usage":{"total_tokens":90}}`))
	}))
	defer srv.Close()

	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	var out, errb bytes.Buffer
	if code := runTrajctl(&out, &errb, []string{
		"declare", "--id", "obj-rubric", "--statement", "Migrate the docs.",
		"--plan", "move pages", "--rubric-base-url", srv.URL, "--rubric-model", "judge-1",
		"--ledger", ledger,
	}); code != 0 {
		t.Fatalf("declare with rubric gate = %d (stderr=%q)", code, errb.String())
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("rubric generation calls = %d, want exactly 1 (declare-time, cached)", got)
	}
	st := trajctl.Fold(trajctl.ReadLedgerFile(ledger))
	obj, ok := st.Objectives["obj-rubric"]
	if !ok || obj.Rubric == nil {
		t.Fatalf("declared objective row lost the cached rubric: %+v", obj)
	}
	if len(obj.Rubric.Criteria) != 2 || obj.Rubric.Criteria[0].ID != "c1" || obj.Rubric.Source != "judge-1" {
		t.Errorf("cached rubric = %+v", obj.Rubric)
	}

	// Without the gate flag: no call, no rubric — the #2535 declare unchanged.
	if code := runTrajctl(&out, &errb, []string{
		"declare", "--id", "obj-bare", "--statement", "Migrate the docs.", "--ledger", ledger,
	}); code != 0 {
		t.Fatalf("bare declare = %d (stderr=%q)", code, errb.String())
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("bare declare must not call the gateway, calls = %d", got)
	}
	st = trajctl.Fold(trajctl.ReadLedgerFile(ledger))
	if st.Objectives["obj-bare"].Rubric != nil {
		t.Errorf("bare declare must cache no rubric")
	}
}

func TestTrajctlDeclareRubricFailClosed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()

	ledger := filepath.Join(t.TempDir(), "trajctl.jsonl")
	var out, errb bytes.Buffer
	if code := runTrajctl(&out, &errb, []string{
		"declare", "--id", "obj-fail", "--statement", "Migrate the docs.",
		"--rubric-base-url", srv.URL, "--ledger", ledger,
	}); code != 1 {
		t.Fatalf("failed rubric generation must fail the declare, got %d (stderr=%q)", code, errb.String())
	}
	if _, err := os.Stat(ledger); !os.IsNotExist(err) {
		t.Errorf("failed declare must append nothing, ledger exists (err=%v)", err)
	}
}
