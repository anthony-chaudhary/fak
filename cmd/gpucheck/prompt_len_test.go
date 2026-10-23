package main

import (
	"bytes"
	"encoding/json"
	"io"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestCyclePromptRepeatsBaseToLength(t *testing.T) {
	base := []int{4, 5, 6}
	if got := cyclePrompt(base, 0); !reflect.DeepEqual(got, base) {
		t.Fatalf("cyclePrompt(base, 0) = %v, want base", got)
	}
	if got, want := cyclePrompt(base, 7), []int{4, 5, 6, 4, 5, 6, 4}; !reflect.DeepEqual(got, want) {
		t.Fatalf("cyclePrompt(base, 7) = %v, want %v", got, want)
	}
	if got, want := cyclePrompt(base, 2), []int{4, 5}; !reflect.DeepEqual(got, want) {
		t.Fatalf("cyclePrompt(base, 2) = %v, want %v", got, want)
	}
}

func TestRunRefusesNegativePromptLen(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := run([]string{"-prompt-len", "-1", "-hf", "x"}, &stdout, &stderr); code != 2 {
		t.Fatalf("run(-prompt-len -1) = %d, want 2 (stderr %q)", code, stderr.String())
	}
}

// fak-test:runtime fast est=1s lane=default
func TestRunReceiptRecordsPromptLen(t *testing.T) {
	dir := writeLeanQ8Checkpoint(t)
	be := &leanQ8UploadBackend{Backend: compute.Default()}
	compute.Register(be)
	for _, tc := range []struct {
		args []string
		want int
	}{{nil, 12}, {[]string{"-prompt-len", "20"}, 20}} {
		var stdout bytes.Buffer
		args := append([]string{"-lean", "-json", "-n", "2", "-backend", be.Name(), "-hf", dir}, tc.args...)
		code, panicked := runRecovering(args, &stdout, io.Discard)
		if panicked != nil {
			t.Fatalf("run %v panicked: %v", tc.args, panicked)
		}
		var got parityReceipt
		if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
			t.Fatalf("decode receipt: %v", err)
		}
		if code != 0 || got.PromptLen != tc.want {
			t.Fatalf("run %v: exit=%d prompt_len=%d, want 0 and %d", tc.args, code, got.PromptLen, tc.want)
		}
	}
}
