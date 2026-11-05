package config

import (
	"strconv"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms lane=default
// Estimate only: scalar configuration checks stop before host resource reads.
func TestValidateGossipSuspectDuration(t *testing.T) {
	t.Parallel()
	type testCase struct {
		name      string
		suspectMs int
		enabled   bool
		invalid   bool
	}
	cases := []testCase{
		{"enabled zero", 0, true, true},
		{"enabled negative", -1, true, true},
		{"one millisecond", 1, true, false},
		{"default duration", DefaultConfig().GossipSuspectMs, true, false},
		{"disabled zero", 0, false, false},
		{"disabled negative", -1, false, false},
	}
	maxDuration := int64(1<<63 - 1)
	limit := maxDuration / (2 * int64(time.Millisecond))
	if strconv.IntSize == 64 {
		cases = append(cases,
			testCase{"last safe doubled duration", int(limit), true, false},
			testCase{"doubling overflow", int(limit + 1), true, true},
			testCase{"conversion overflow", int(maxDuration/int64(time.Millisecond) + 1), true, true},
			testCase{"maximum int", int(maxDuration), true, true},
			testCase{"disabled oversized", int(maxDuration), false, false},
		)
	} else {
		cases = append(cases, testCase{"maximum 32-bit int", int(^uint(0) >> 1), true, false})
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := DefaultConfig()
			cfg.ClusterEnabled = tc.enabled
			cfg.GossipSuspectMs = tc.suspectMs
			// Successful duration admission must reach this existing next error
			// without reading host memory or allocating server resources.
			cfg.MaxMemoryGB = 0
			err := cfg.Validate()
			if err == nil {
				t.Fatal("validation passed the deliberately invalid memory canary")
			}
			if tc.invalid {
				if !strings.HasPrefix(err.Error(), "gossip_suspect_ms must be in [1, ") {
					t.Fatalf("suspect duration error = %v", err)
				}
			} else if err.Error() != "max_memory_gb must be > 0" {
				t.Fatalf("valid or unused duration did not reach memory canary: %v", err)
			}
			if tc.enabled && !tc.invalid {
				duration := time.Duration(tc.suspectMs) * time.Millisecond
				if duration <= 0 || duration*2 <= 0 || duration*2/2 != duration {
					t.Fatalf("admitted duration does not safely double: %v", duration)
				}
			}
			if cfg.GossipSuspectMs != tc.suspectMs {
				t.Fatalf("validation changed duration from %d to %d", tc.suspectMs, cfg.GossipSuspectMs)
			}
		})
	}
}
