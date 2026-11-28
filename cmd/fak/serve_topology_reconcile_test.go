package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/inferencetopology"
)

// fak-test:runtime fast est=1s
func TestServeTopologyShadowReconcile(t *testing.T) {
	s, err := gateway.New(gateway.Config{})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	m := gateway.NewFleetMembership(gateway.MembershipConfig{
		UnhealthyAfter: 1,
		Probe:          func(context.Context, gateway.WorkerSpec) bool { return true },
	})
	if err := m.Add(gateway.WorkerSpec{ID: "r0"}); err != nil {
		t.Fatalf("m.Add: %v", err)
	}
	s.SetFleetMembership(m)

	if err := configureServeTopologyShadow(s, nil); err != nil {
		t.Fatalf("nil path: got %v, want nil", err)
	}
	empty := ""
	if err := configureServeTopologyShadow(s, &empty); err != nil {
		t.Fatalf("empty path: got %v, want nil", err)
	}

	missing := filepath.Join(t.TempDir(), "missing.json")
	if err := configureServeTopologyShadow(s, &missing); err == nil {
		t.Fatal("missing path: got nil, want error")
	}

	dir := t.TempDir()
	if err := configureServeTopologyShadow(s, &dir); err == nil {
		t.Fatal("directory path: got nil, want error")
	}

	searchFile, err := os.Open(filepath.Join("..", "..", "internal", "inferencetopology", "testdata", "runtime_search.json"))
	if err != nil {
		t.Fatalf("open runtime_search.json: %v", err)
	}
	defer searchFile.Close()
	search, err := inferencetopology.DecodeRuntimeSearch(searchFile)
	if err != nil {
		t.Fatalf("DecodeRuntimeSearch: %v", err)
	}

	in := gateway.TopologyShadowInput{
		Schema:          gateway.TopologyShadowSchema,
		Search:          search,
		Incumbent:       1,
		HysteresisTicks: 1,
		Objective: gateway.PoolObjective{
			MinReplicas: 1,
			MaxReplicas: 3,
			ScaleStep:   1,
			QueueHigh:   4,
			QueueLow:    1,
			KVUtilHigh:  0.9,
			KVUtilLow:   0.5,
		},
	}
	payload, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	shadowPath := filepath.Join(t.TempDir(), "topology_shadow.json")
	if err := os.WriteFile(shadowPath, payload, 0o600); err != nil {
		t.Fatalf("os.WriteFile: %v", err)
	}

	if err := configureServeTopologyShadow(s, &shadowPath); err != nil {
		t.Fatalf("valid shadow file: got %v, want nil", err)
	}
	if _, ok := s.ReconcileTopologyShadow(t.Context(), time.Now()); !ok {
		t.Fatal("expected shadow reconciler to be installed on server")
	}
}
