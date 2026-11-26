package model

import (
	"os"
	"runtime"
	"strings"
	"testing"
)

// TestMain caps the process-wide matmul pool for the test binary. The pool spins idle
// workers for parSpinBudget before parking, so a full-width pool burns every core on
// tiny fixtures that never need more than a few workers.
func TestMain(m *testing.M) {
	if strings.TrimSpace(os.Getenv("FAK_WORKERS")) == "" && strings.TrimSpace(os.Getenv("FAK_BUDGET")) == "" {
		if err := SetWorkers(min(8, runtime.NumCPU())); err != nil {
			panic(err)
		}
	}
	os.Exit(m.Run())
}
