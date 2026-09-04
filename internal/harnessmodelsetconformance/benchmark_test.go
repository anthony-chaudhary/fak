package harnessmodelsetconformance_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessmodelset"
	"github.com/anthony-chaudhary/fak/internal/modelinventory"
	"github.com/anthony-chaudhary/fak/internal/modelsetlock"
	"github.com/anthony-chaudhary/fak/internal/modelsetreceipt"
	"github.com/anthony-chaudhary/fak/internal/modelsetresolve"
)

// Invariant: benchmark operations must execute without external network I/O or stateful disk mutations.
//
// Contract: benchmark measurements execute deterministic in-memory evaluations against canonical fixtures.

// BenchmarkModelSetResolution measures the performance of resolving candidate models to roles in a model set.
func BenchmarkModelSetResolution(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", "harness.model-set.json"))
	if err != nil {
		b.Fatalf("read intent fixture: %v", err)
	}
	intent, err := harnessmodelset.ParseJSON(raw)
	if err != nil {
		b.Fatalf("parse intent: %v", err)
	}
	inventory, diagnostics := modelinventory.Normalize(successObservations(), conformanceTime)
	if len(diagnostics) != 0 {
		b.Fatalf("normalize inventory: %s", diagnostics)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		resolution, err := modelsetresolve.Resolve(intent, inventory, conformanceTime)
		if err != nil {
			b.Fatalf("modelsetresolve.Resolve: %v", err)
		}
		if len(resolution.Roles) != 2 {
			b.Fatalf("unexpected role count: %d", len(resolution.Roles))
		}
	}
}

// BenchmarkModelSetLockGeneration measures canonical lock generation over resolved model sets.
func BenchmarkModelSetLockGeneration(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", "harness.model-set.json"))
	if err != nil {
		b.Fatalf("read intent fixture: %v", err)
	}
	intent, err := harnessmodelset.ParseJSON(raw)
	if err != nil {
		b.Fatalf("parse intent: %v", err)
	}
	inventory, diagnostics := modelinventory.Normalize(successObservations(), conformanceTime)
	if len(diagnostics) != 0 {
		b.Fatalf("normalize inventory: %s", diagnostics)
	}
	resolution, err := modelsetresolve.Resolve(intent, inventory, conformanceTime)
	if err != nil {
		b.Fatalf("modelsetresolve.Resolve: %v", err)
	}
	inputs := lockInputs(intent, inventory)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		lock, err := modelsetlock.New(inputs, resolution)
		if err != nil {
			b.Fatalf("modelsetlock.New: %v", err)
		}
		if lock.ContentDigest == "" {
			b.Fatal("unexpected empty lock content digest")
		}
	}
}

// BenchmarkModelSetReceiptEvaluation measures startup compatibility evaluation for model-set receipts.
func BenchmarkModelSetReceiptEvaluation(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", "harness.model-set.json"))
	if err != nil {
		b.Fatalf("read intent fixture: %v", err)
	}
	intent, err := harnessmodelset.ParseJSON(raw)
	if err != nil {
		b.Fatalf("parse intent: %v", err)
	}
	inventory, diagnostics := modelinventory.Normalize(successObservations(), conformanceTime)
	if len(diagnostics) != 0 {
		b.Fatalf("normalize inventory: %s", diagnostics)
	}
	resolution, err := modelsetresolve.Resolve(intent, inventory, conformanceTime)
	if err != nil {
		b.Fatalf("modelsetresolve.Resolve: %v", err)
	}
	expectation, err := modelsetreceipt.Bind(intent, resolution, inventory)
	if err != nil {
		b.Fatalf("modelsetreceipt.Bind: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		receipt, err := modelsetreceipt.Evaluate(expectation, intent, resolution, inventory, conformanceTime)
		if err != nil {
			b.Fatalf("modelsetreceipt.Evaluate: %v", err)
		}
		if receipt.Status != modelsetreceipt.StatusCompatible {
			b.Fatalf("unexpected receipt status: %s", receipt.Status)
		}
	}
}
