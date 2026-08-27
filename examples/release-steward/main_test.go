package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestSixMonthLifecycleWitness(t *testing.T) {
	var b bytes.Buffer
	if err := run(&b); err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(b.String()), "\n")
	if len(lines) != 6 {
		t.Fatalf("output=%q", b.String())
	}
	var e Export
	if err := json.Unmarshal([]byte(lines[5]), &e); err != nil {
		t.Fatal(err)
	}
	if e.MacroID != "macro/release-steward" || e.Sessions != 3 || e.Restarts != 2 || e.InboxDelivered != 2 || e.Delegations != 1 || e.MicroOperations != 1 || e.State != "retired" {
		t.Fatalf("lifecycle=%+v", e)
	}
	if e.RawChildHistoryRetained || len(e.DurableMemory) != 2 {
		t.Fatalf("promotion boundary=%+v", e)
	}
	for _, r := range e.Receipts {
		if r.Model == "" || r.Fleet == "" || r.Outcome == "" {
			t.Fatalf("incomplete receipt=%+v", r)
		}
	}
}
