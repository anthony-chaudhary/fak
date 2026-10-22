package modelengine

import (
	"bytes"
	"errors"
	"strings"
	"testing"
)

func setNativeKVEnv(t *testing.T, maxBlocks, blockTokens string) {
	t.Helper()
	t.Setenv("FAK_MODEL_DIR", "")
	t.Setenv("FAK_NATIVE_KV_MAX_BLOCKS", maxBlocks)
	t.Setenv("FAK_NATIVE_KV_BLOCK_TOKENS", blockTokens)
	t.Setenv("FAK_NATIVE_KV_PREEMPT_MODE", "")
	t.Setenv("FAK_NATIVE_KV_VICTIM_RULE", "")
}

func countingKVProbe(calls *int) func() (uint64, bool) {
	return func() (uint64, bool) {
		*calls++
		return 10 << 20, true
	}
}

func nativeKVEnvErrorNames(err error) map[string]string {
	got := map[string]string{}
	var walk func(error)
	walk = func(err error) {
		if err == nil {
			return
		}
		if e, ok := err.(*NativeKVEnvError); ok {
			got[e.Name] = e.Value
			return
		}
		if j, ok := err.(interface{ Unwrap() []error }); ok {
			for _, inner := range j.Unwrap() {
				walk(inner)
			}
		}
	}
	walk(err)
	return got
}

// fak-test:runtime fast est=1s lane=default
func TestNativeKVCapacityEnvRejectsMalformed(t *testing.T) {
	const bytesPerToken = 1024
	cases := []struct {
		name            string
		maxBlocks       string
		blockTokens     string
		wantNames       map[string]string
		wantMaxBlocks   int
		wantBlockTokens int
		wantProbeCalls  int
	}{
		{name: "max-non-integer", maxBlocks: "abc", wantNames: map[string]string{"FAK_NATIVE_KV_MAX_BLOCKS": "abc"}},
		{name: "max-negative", maxBlocks: "-3", wantNames: map[string]string{"FAK_NATIVE_KV_MAX_BLOCKS": "-3"}},
		{name: "max-fractional", maxBlocks: "1.5", wantNames: map[string]string{"FAK_NATIVE_KV_MAX_BLOCKS": "1.5"}},
		{name: "max-unit-suffix", maxBlocks: "4k", wantNames: map[string]string{"FAK_NATIVE_KV_MAX_BLOCKS": "4k"}},
		{name: "max-padded", maxBlocks: " 7", wantNames: map[string]string{"FAK_NATIVE_KV_MAX_BLOCKS": " 7"}},
		{name: "block-non-integer", maxBlocks: "7", blockTokens: "abc", wantNames: map[string]string{"FAK_NATIVE_KV_BLOCK_TOKENS": "abc"}, wantMaxBlocks: 7},
		{name: "block-negative", maxBlocks: "7", blockTokens: "-1", wantNames: map[string]string{"FAK_NATIVE_KV_BLOCK_TOKENS": "-1"}, wantMaxBlocks: 7},
		{name: "block-bad-max-unset", blockTokens: "abc", wantNames: map[string]string{"FAK_NATIVE_KV_BLOCK_TOKENS": "abc"}, wantMaxBlocks: 576, wantProbeCalls: 1},
		{
			name: "both", maxBlocks: "abc", blockTokens: "-1",
			wantNames: map[string]string{"FAK_NATIVE_KV_MAX_BLOCKS": "abc", "FAK_NATIVE_KV_BLOCK_TOKENS": "-1"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			setNativeKVEnv(t, tc.maxBlocks, tc.blockTokens)
			calls := 0
			p, err := nativePreemptionPolicyFromEnvChecked(countingKVProbe(&calls), bytesPerToken)
			if !errors.Is(err, ErrNativeKVEnvInvalid) {
				t.Fatalf("errors.Is(err, ErrNativeKVEnvInvalid) = false; err = %v", err)
			}
			got := nativeKVEnvErrorNames(err)
			if len(got) != len(tc.wantNames) {
				t.Fatalf("offending vars = %v, want %v", got, tc.wantNames)
			}
			for name, raw := range tc.wantNames {
				if got[name] != raw {
					t.Fatalf("offending vars = %v, want %v", got, tc.wantNames)
				}
			}
			if p.MaxBlocks != tc.wantMaxBlocks || p.BlockTokens != tc.wantBlockTokens {
				t.Fatalf("policy = MaxBlocks %d BlockTokens %d, want %d %d", p.MaxBlocks, p.BlockTokens, tc.wantMaxBlocks, tc.wantBlockTokens)
			}
			if p.Mode != NativePreemptSwap || p.VictimRule != NativePreemptVictimMostRecent {
				t.Fatalf("policy mode/victim = %v/%v, want swap/most-recent", p.Mode, p.VictimRule)
			}
			if calls != tc.wantProbeCalls {
				t.Fatalf("probe calls = %d, want %d", calls, tc.wantProbeCalls)
			}
			calls = 0
			if wrapped := nativePreemptionPolicyFromEnvWithProbe(countingKVProbe(&calls), bytesPerToken); wrapped != p {
				t.Fatalf("unchecked policy = %+v, want %+v", wrapped, p)
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestNativeKVCapacityEnvAcceptsDocumentedValues(t *testing.T) {
	const bytesPerToken = 1024
	cases := []struct {
		name            string
		maxBlocks       string
		blockTokens     string
		wantMaxBlocks   int
		wantBlockTokens int
		wantProbeCalls  int
	}{
		{name: "unset-probed", wantMaxBlocks: 576, wantProbeCalls: 1},
		{name: "max-zero", maxBlocks: "0"},
		{name: "max-positive", maxBlocks: "7", wantMaxBlocks: 7},
		{name: "block-zero", maxBlocks: "7", blockTokens: "0", wantMaxBlocks: 7},
		{name: "block-positive", maxBlocks: "7", blockTokens: "16", wantMaxBlocks: 7, wantBlockTokens: 16},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			setNativeKVEnv(t, tc.maxBlocks, tc.blockTokens)
			calls := 0
			p, err := nativePreemptionPolicyFromEnvChecked(countingKVProbe(&calls), bytesPerToken)
			if err != nil {
				t.Fatalf("err = %v, want nil", err)
			}
			if p.MaxBlocks != tc.wantMaxBlocks || p.BlockTokens != tc.wantBlockTokens {
				t.Fatalf("policy = MaxBlocks %d BlockTokens %d, want %d %d", p.MaxBlocks, p.BlockTokens, tc.wantMaxBlocks, tc.wantBlockTokens)
			}
			if calls != tc.wantProbeCalls {
				t.Fatalf("probe calls = %d, want %d", calls, tc.wantProbeCalls)
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestEngineKVPreemptionEnvErrSurfacesMalformedCapacity(t *testing.T) {
	const maxName, blockName = "FAK_NATIVE_KV_MAX_BLOCKS", "FAK_NATIVE_KV_BLOCK_TOKENS"
	cases := []struct {
		name          string
		maxBlocks     string
		blockTokens   string
		wantMaxBlocks int
		wantBad       []string
	}{
		{name: "max-invalid", maxBlocks: "abc", wantBad: []string{maxName}},
		{name: "block-invalid-max-armed", maxBlocks: "7", blockTokens: "abc", wantMaxBlocks: 7, wantBad: []string{blockName}},
		{name: "max-invalid-block-valid", maxBlocks: "abc", blockTokens: "32", wantBad: []string{maxName}},
		{name: "both-invalid", maxBlocks: "abc", blockTokens: "-1", wantBad: []string{maxName, blockName}},
		{name: "max-zero", maxBlocks: "0"},
		{name: "max-positive", maxBlocks: "7", wantMaxBlocks: 7},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			setNativeKVEnv(t, tc.maxBlocks, tc.blockTokens)
			var notice bytes.Buffer
			prev := nativeEnvNotice
			nativeEnvNotice = &notice
			t.Cleanup(func() { nativeEnvNotice = prev })

			e := New()
			if got := e.KVPreemptionStats().MaxBlocks; got != tc.wantMaxBlocks {
				t.Fatalf("MaxBlocks = %d, want %d", got, tc.wantMaxBlocks)
			}
			_ = e.KVPreemptionStats()
			err := e.KVPreemptionEnvErr()
			if len(tc.wantBad) == 0 {
				if err != nil || notice.Len() != 0 {
					t.Fatalf("KVPreemptionEnvErr = %v, notice = %q; want nil and empty", err, notice.String())
				}
				return
			}
			if !errors.Is(err, ErrNativeKVEnvInvalid) {
				t.Fatalf("errors.Is(KVPreemptionEnvErr, ErrNativeKVEnvInvalid) = false; err = %v", err)
			}
			if got := nativeKVEnvErrorNames(err); len(got) != len(tc.wantBad) {
				t.Fatalf("offending vars = %v, want %v", got, tc.wantBad)
			}
			if want := "[fak] " + NativeKVEnvNotice(err) + "\n"; notice.String() != want {
				t.Fatalf("notice = %q, want exactly one line %q", notice.String(), want)
			}
			bad := map[string]bool{}
			for _, name := range tc.wantBad {
				bad[name] = true
			}
			line := notice.String()
			for _, name := range []string{maxName, blockName} {
				if got := strings.Contains(line, name+"="); got != bad[name] {
					t.Fatalf("notice names %s = %v, want %v: %q", name, got, bad[name], line)
				}
				if got := strings.Contains(line, nativeKVEnvFallback(name)); got != bad[name] {
					t.Fatalf("notice states %s fallback = %v, want %v: %q", name, got, bad[name], line)
				}
			}
		})
	}
}
