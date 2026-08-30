package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/deploymanifest"
)

func TestServeNativeQwenQ4KPrefillChunkDefaultAndBounds(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse(nil); err != nil {
		t.Fatal(err)
	}
	if got := *sf.nativeQwenQ4KPrefillChunk; got != defaultNativeQwenQ4KPrefillChunk {
		t.Fatalf("default = %d, want %d", got, defaultNativeQwenQ4KPrefillChunk)
	}

	report := effectiveServeConfigWithQwen38Runtime(sf, deploymanifest.Manifest{}, false, explicitFlagNames(fs))
	got := report.Values["native_qwen_q4k_prefill_chunk_tokens"]
	if got.Value != defaultNativeQwenQ4KPrefillChunk || got.Source != "built-in" {
		t.Fatalf("default effective-config receipt = %#v, want value %d from built-in", got, defaultNativeQwenQ4KPrefillChunk)
	}

	for _, tokens := range []int{minNativeQwenQ4KPrefillChunk, defaultNativeQwenQ4KPrefillChunk, maxNativeQwenQ4KPrefillChunk} {
		t.Run(strconv.Itoa(tokens), func(t *testing.T) {
			t.Setenv("FAK_INKERNEL_QWEN_Q4K_PREFILL_CHUNK_TOKENS", "ambient")
			if err := validateNativeQwenQ4KPrefillChunk(tokens); err != nil {
				t.Fatalf("apply %d: %v", tokens, err)
			}
			if got := os.Getenv("FAK_INKERNEL_QWEN_Q4K_PREFILL_CHUNK_TOKENS"); got != "ambient" {
				t.Fatalf("legacy environment mutated to %q", got)
			}
		})
	}
}

func TestServeNativeQwenQ4KPrefillChunkRefusesOutOfRangeBeforePropagation(t *testing.T) {
	for _, tokens := range []int{minNativeQwenQ4KPrefillChunk - 1, -1, 0, maxNativeQwenQ4KPrefillChunk + 1} {
		t.Run(strconv.Itoa(tokens), func(t *testing.T) {
			const ambient = "do-not-overwrite"
			t.Setenv("FAK_INKERNEL_QWEN_Q4K_PREFILL_CHUNK_TOKENS", ambient)
			err := validateNativeQwenQ4KPrefillChunk(tokens)
			if err == nil {
				t.Fatalf("apply %d succeeded, want refusal", tokens)
			}
			if got := os.Getenv("FAK_INKERNEL_QWEN_Q4K_PREFILL_CHUNK_TOKENS"); got != ambient {
				t.Fatalf("refusal propagated %q, want ambient %q preserved", got, ambient)
			}
			want := "want 128..8192"
			if !strings.Contains(err.Error(), nativeQwenQ4KPrefillChunkFlag) || !strings.Contains(err.Error(), want) {
				t.Fatalf("error = %q, want flag and %q", err, want)
			}
		})
	}
}

func TestServeNativeQwenQ4KPrefillChunkRefusesAtStartupBeforeLoad(t *testing.T) {
	if value := os.Getenv("FAK_TEST_NATIVE_QWEN_Q4K_PREFILL_CHUNK_TOKENS"); value != "" {
		cmdServe([]string{"--" + nativeQwenQ4KPrefillChunkFlag, value, "--gguf", "must-not-load.gguf", "--print-effective-config"})
		return
	}

	for _, value := range []string{"127", "8193"} {
		t.Run(value, func(t *testing.T) {
			cmd := exec.Command(os.Args[0], "-test.run=^TestServeNativeQwenQ4KPrefillChunkRefusesAtStartupBeforeLoad$")
			cmd.Env = append(os.Environ(), "FAK_TEST_NATIVE_QWEN_Q4K_PREFILL_CHUNK_TOKENS="+value)
			output, err := cmd.CombinedOutput()
			var exitErr *exec.ExitError
			if !errors.As(err, &exitErr) || exitErr.ExitCode() != 2 {
				t.Fatalf("startup with %s exit = %v; output:\n%s", value, err, output)
			}
			want := fmt.Sprintf("--%s=%s: want 128..8192", nativeQwenQ4KPrefillChunkFlag, value)
			if !strings.Contains(string(output), want) {
				t.Fatalf("startup with %s omitted pre-load refusal %q:\n%s", value, want, output)
			}
			if strings.Contains(string(output), "must-not-load.gguf") {
				t.Fatalf("startup with %s reached model-path handling after refusal:\n%s", value, output)
			}
		})
	}
}

func TestServeNativeQwenQ4KPrefillChunkReceiptEvidence(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--" + nativeQwenQ4KPrefillChunkFlag, "2048"}); err != nil {
		t.Fatal(err)
	}
	if err := validateNativeQwenQ4KPrefillChunk(*sf.nativeQwenQ4KPrefillChunk); err != nil {
		t.Fatal(err)
	}
	if got := serveNativePlannerConfig(sf).QwenQ4KPrefillChunkTokens; got != 2048 {
		t.Fatalf("typed planner contract = %d, want 2048", got)
	}

	report := effectiveServeConfigWithQwen38Runtime(sf, deploymanifest.Manifest{}, false, explicitFlagNames(fs))
	got, ok := report.Values["native_qwen_q4k_prefill_chunk_tokens"]
	if !ok {
		t.Fatal("effective-config receipt omits native_qwen_q4k_prefill_chunk_tokens")
	}
	if got.Value != 2048 || got.Source != "flag" {
		t.Fatalf("effective-config receipt = %#v, want value 2048 from flag", got)
	}

	if runtime := *sf.qwen38Runtime; runtime != qwen38RuntimeNative {
		t.Fatalf("chunk flag changed runtime to %q, want fak-native %q", runtime, qwen38RuntimeNative)
	}
}
