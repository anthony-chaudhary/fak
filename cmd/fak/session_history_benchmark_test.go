package main

import (
	"bytes"
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/sessionmine"
	"testing"
)

func TestSessionHistoryBenchmarkCLI(t *testing.T) {
	var out, errOut bytes.Buffer
	rc := runSessionHistory(&out, &errOut, []string{"benchmark", "--sizes", "10", "--repetitions", "1"})
	if rc != 0 {
		t.Fatalf("rc=%d err=%s", rc, errOut.String())
	}
	var rep sessionmine.RefreshBenchmarkReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatal(err)
	}
	if rep.Schema != sessionmine.RefreshBenchmarkSchema || len(rep.Scales) != 1 {
		t.Fatalf("report=%+v", rep)
	}
}
