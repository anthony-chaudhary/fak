package nativefirst

import "testing"

func TestScanLine(t *testing.T) {
	tests := []struct {
		name string
		line string
		want bool
	}{
		{"default", "Qwen3.8 native performance defaults to llama.cpp.", true},
		{"fallback", "Native falls back to llama-server.", true},
		{"auto backend", "The native backend auto-selects llamacpp.", true},
		{"benchmark", "Benchmark fak-native against llama.cpp.", false},
		{"parity", "Use llama.cpp explicitly for parity diagnosis.", false},
		{"borrowing", "Study and borrow a llama.cpp kernel.", false},
		{"interop", "The interop adapter delegates explicitly to llama-server.", false},
		{"unrelated", "The default policy is fail closed.", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ScanLine(tt.line)
			if (got != nil) != tt.want {
				t.Fatalf("ScanLine(%q) = %+v, want finding=%v", tt.line, got, tt.want)
			}
			if got != nil && (got.Phrase == "" || got.Reason != Guidance) {
				t.Fatalf("incomplete finding: %+v", got)
			}
		})
	}
}
