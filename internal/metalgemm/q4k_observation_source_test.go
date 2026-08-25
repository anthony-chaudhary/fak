package metalgemm

import (
	"os"
	"strings"
	"testing"
)

// This source witness keeps the Darwin-only cgo path compile-safe on hosts that cannot type-check
// Objective-C bridge calls: the repeated-batch helper must receive the owning observation, and its
// only caller must forward the same observation rather than referencing an out-of-scope variable.
func TestQ4KRepeatedBatchForwardsExecutionObservation(t *testing.T) {
	source, err := os.ReadFile("q4k.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(source)
	for _, want := range []string{
		"func (w *Q4KWeight) gemvBatchRepeatedWithEvents(Xcat []float32, n int, Ycat []float32, observation *ExecutionObservation)",
		"w.gemvBatchRepeatedWithEvents(Xcat, n, Ycat, observation)",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("q4k repeated-batch observation forwarding missing %q", want)
		}
	}
}
