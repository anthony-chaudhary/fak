package config

import (
	"strconv"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms lane=default
func TestValidateGossipPingInterval(t *testing.T) {
	t.Parallel()
	type testCase struct {
		name    string
		pingMs  int
		enabled bool
		invalid bool
	}
	cases := []testCase{
		{"enabled zero", 0, true, true},
		{"enabled negative", -1, true, true},
		{"one millisecond", 1, true, false},
		{"default interval", DefaultConfig().GossipPingMs, true, false},
		{"disabled zero", 0, false, false},
		{"disabled negative", -1, false, false},
	}
	maxDuration := int64(1<<63 - 1)
	limit := maxDuration / int64(time.Millisecond)
	if strconv.IntSize == 64 {
		cases = append(cases,
			testCase{"last representable", int(limit), true, false},
			testCase{"negative overflow", int(limit + 1), true, true},
			testCase{"positive wraparound", int(2*limit + 2), true, true},
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
			cfg.GossipPingMs = tc.pingMs
			// This existing next validation error is a canary: successful interval
			// admission must reach it without reading host memory or resource limits.
			cfg.MaxMemoryGB = 0
			err := cfg.Validate()
			if err == nil {
				t.Fatal("validation passed the deliberately invalid memory canary")
			}
			if tc.invalid {
				if !strings.HasPrefix(err.Error(), "gossip_ping_ms must be in [1, ") {
					t.Fatalf("interval error = %v", err)
				}
			} else if err.Error() != "max_memory_gb must be > 0" {
				t.Fatalf("valid or unused interval did not reach memory canary: %v", err)
			}
			if cfg.GossipPingMs != tc.pingMs {
				t.Fatalf("validation changed interval from %d to %d", tc.pingMs, cfg.GossipPingMs)
			}
		})
	}
}
