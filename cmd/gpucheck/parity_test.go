package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestEvaluateParityRejectsFalseGreens(t *testing.T) {
	tests := []struct {
		name     string
		in       parityInput
		want     parityReasonCode
		compared int
	}{
		{
			name: "truncated common prefix",
			in:   parityInput{Requested: 3, VocabSize: 100, Reference: []int{1, 2, 3}, Device: []int{1, 2}, RequestedBackend: "cuda", Observed: backendIdentity{Name: "cuda", DeviceMemory: true}},
			want: reasonAsymmetricStreams,
		},
		{
			name: "empty streams",
			in:   parityInput{Requested: 3, VocabSize: 100, RequestedBackend: "cuda", Observed: backendIdentity{Name: "cuda", DeviceMemory: true}},
			want: reasonEmptyStream,
		},
		{
			name: "excess tokens",
			in:   parityInput{Requested: 2, VocabSize: 100, Reference: []int{1, 2, 3}, Device: []int{1, 2, 3}, RequestedBackend: "cuda", Observed: backendIdentity{Name: "cuda", DeviceMemory: true}},
			want: reasonExcessTokens,
		},
		{
			name: "out of vocabulary",
			in:   parityInput{Requested: 2, VocabSize: 10, Reference: []int{1, 10}, Device: []int{1, 10}, RequestedBackend: "cuda", Observed: backendIdentity{Name: "cuda", DeviceMemory: true}},
			want: reasonTokenOutOfRange,
		},
		{
			name:     "token mismatch",
			in:       parityInput{Requested: 2, VocabSize: 100, Reference: []int{1, 2}, Device: []int{1, 3}, RequestedBackend: "cuda", Observed: backendIdentity{Name: "cuda", DeviceMemory: true}},
			want:     reasonTokenMismatch,
			compared: 2,
		},
		{
			name: "backend fallback",
			in:   parityInput{Requested: 2, VocabSize: 100, Reference: []int{1, 2}, Device: []int{1, 2}, RequestedBackend: "cuda", Observed: backendIdentity{Name: "cpu-ref"}, RequireNative: true},
			want: reasonBackendMismatch,
		},
		{
			name: "host backend",
			in:   parityInput{Requested: 2, VocabSize: 100, Reference: []int{1, 2}, Device: []int{1, 2}, RequestedBackend: "cpu-ref", Observed: backendIdentity{Name: "cpu-ref"}, RequireNative: true},
			want: reasonBackendNotDevice,
		},
		{
			name:     "short without EOS",
			in:       parityInput{Requested: 3, VocabSize: 100, Reference: []int{1, 2}, Device: []int{1, 2}, RequestedBackend: "cuda", Observed: backendIdentity{Name: "cuda", DeviceMemory: true}},
			want:     reasonShortWithoutEOS,
			compared: 2,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := evaluateParity(tc.in)
			if got.Pass || got.ReasonCode != tc.want {
				t.Fatalf("pass=%t reason_code=%q, want pass=false reason_code=%q", got.Pass, got.ReasonCode, tc.want)
			}
			if got.ComparedTokens != tc.compared {
				t.Fatalf("compared_tokens=%d, want %d", got.ComparedTokens, tc.compared)
			}
			if tc.want == reasonBackendMismatch && !got.FallbackActive {
				t.Fatal("backend mismatch did not record fallback_active")
			}
		})
	}
}

func TestEvaluateParityAllowsRequestedLengthOrSharedEOS(t *testing.T) {
	tests := []parityInput{
		{Requested: 3, VocabSize: 100, Reference: []int{1, 2, 3}, Device: []int{1, 2, 3}, RequestedBackend: "cuda", Observed: backendIdentity{Name: "cuda", Tier: "sm_120", DeviceMemory: true}, RequireNative: true},
		{Requested: 3, VocabSize: 100, Reference: []int{1, 99}, Device: []int{1, 99}, RequestedBackend: "cuda", Observed: backendIdentity{Name: "cuda", Tier: "sm_120", DeviceMemory: true}, RequireNative: true, IsEOS: func(id int) bool { return id == 99 }},
	}
	for _, in := range tests {
		got := evaluateParity(in)
		if !got.Pass || got.ComparedTokens != len(in.Reference) || got.Engine != gpucheckNativeEngine || got.FallbackActive || got.FallbackScope != "backend_session_selection" {
			t.Fatalf("pass=%t compared=%d engine=%q fallback=%t scope=%q", got.Pass, got.ComparedTokens, got.Engine, got.FallbackActive, got.FallbackScope)
		}
	}
}

func TestRunRejectsNonPositiveCountBeforeModelLoad(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := run([]string{"-hf", t.TempDir(), "-n", "0"}, &stdout, &stderr); code != 2 {
		t.Fatalf("exit = %d, stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if stdout.Len() != 0 {
		t.Fatalf("non-JSON refusal wrote stdout: %q", stdout.String())
	}
}

func TestRunHelpSucceeds(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := run([]string{"-h"}, &stdout, &stderr); code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if stderr.Len() == 0 {
		t.Fatal("help stream is empty")
	}
}

func TestRunJSONFailureReceiptOnlyOnStdout(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := run([]string{"-json", "-hf", t.TempDir(), "-n", "-1", "-backend", "cuda", "-require-native"}, &stdout, &stderr); code != 2 {
		t.Fatalf("exit = %d, stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("JSON mode wrote stderr: %q", stderr.String())
	}
	var got parityReceipt
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode receipt: %v; raw=%q", err, stdout.String())
	}
	if got.Pass || got.Engine != gpucheckNativeEngine || got.RequestedBackend != "cuda" || got.ReasonCode != reasonInvalidTokenCount {
		t.Fatalf("pass=%t engine=%q backend=%q reason_code=%q", got.Pass, got.Engine, got.RequestedBackend, got.ReasonCode)
	}
}

func TestRunRejectsZeroVocabularyBeforePromptConstruction(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"vocab_size":0}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := run([]string{"-json", "-hf", dir}, &stdout, &stderr); code != 1 {
		t.Fatalf("exit=%d stderr=%q", code, stderr.String())
	}
	var got parityReceipt
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode receipt: %v", err)
	}
	if got.ReasonCode != reasonInvalidVocabulary || got.ComparedTokens != 0 {
		t.Fatalf("reason_code=%q compared_tokens=%d", got.ReasonCode, got.ComparedTokens)
	}
}

type failingWriter struct{}

func (failingWriter) Write([]byte) (int, error) { return 0, errors.New("write refused") }

func TestRunJSONWriterFailureCannotPass(t *testing.T) {
	var stderr bytes.Buffer
	if code := run([]string{"-json", "-n", "0"}, failingWriter{}, &stderr); code != 1 {
		t.Fatalf("exit=%d, want 1", code)
	}
	if stderr.Len() != 0 {
		t.Fatalf("JSON mode wrote stderr: %q", stderr.String())
	}
}
