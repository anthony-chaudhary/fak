package gateway

import (
	"math"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

// metrics_kvprefix_rejected_adversarial_test.go -- adversarial + edge-case sweep for the
// rejected cache-tier gateway metric (#10144, parent #912). The spine (72a2a465b) exposes
// fak_gateway_kv_prefix_tier_accesses_rejected_total from cacheobs.Default. This file drives
// the HOSTILE input classes the spine did not: empty, oversized, negative, mixed-valid and
// every-dimension-malformed tier accesses, and pins the gateway render seam against the
// failure modes an out-of-vocabulary observation can induce -- a label explosion keyed on the
// malformed dimension, a double-count, a leak into accepted denominators, a negative/aliased
// value, a nondeterministic body, or a panic. Every case goes through the real Server
// renderMetrics() seam (srv.renderMetrics), not the Observer in isolation.
//
// The cacheobs accounting layer already pins per-dimension rejection at the observer
// (internal/cacheobs/tiers_edge_test.go, failure_paths_test.go); this sweep pins the GATEWAY
// projection of it, which is the surface #10144 owns.

func TestRejectedTierGatewayMetricAdversarialSweep(t *testing.T) {
	const metric = "fak_gateway_kv_prefix_tier_accesses_rejected_total"

	// Install an isolated observer for the duration of the test so hostile inputs cannot
	// perturb sibling tests, and so the deltas below are exact rather than process-shared.
	orig := cacheobs.Default
	cacheobs.Default = cacheobs.New()
	t.Cleanup(func() { cacheobs.Default = orig })

	srv := newTestServer(t)

	hostile := []struct {
		name   string
		access cacheobs.TierAccess
	}{
		{
			name: "oversized tier only",
			access: cacheobs.TierAccess{
				Tier: cacheobs.CacheTier(1 << 20), Op: cacheobs.OpRead,
				Outcome: cacheobs.OutcomeHit, Backend: cacheobs.BackendMemory,
			},
		},
		{
			name: "oversized operation only",
			access: cacheobs.TierAccess{
				Tier: cacheobs.TierLocalPrefix, Op: cacheobs.TierOp(1 << 20),
				Outcome: cacheobs.OutcomeHit, Backend: cacheobs.BackendMemory,
			},
		},
		{
			name: "oversized outcome only",
			access: cacheobs.TierAccess{
				Tier: cacheobs.TierLocalPrefix, Op: cacheobs.OpRead,
				Outcome: cacheobs.TierOutcome(1 << 20), Backend: cacheobs.BackendMemory,
			},
		},
		{
			name: "oversized backend only",
			access: cacheobs.TierAccess{
				Tier: cacheobs.TierLocalPrefix, Op: cacheobs.OpRead,
				Outcome: cacheobs.OutcomeHit, Backend: cacheobs.BackendClass(1 << 20),
			},
		},
		{
			name: "negative tier only",
			access: cacheobs.TierAccess{
				Tier: cacheobs.CacheTier(-1), Op: cacheobs.OpRead,
				Outcome: cacheobs.OutcomeHit, Backend: cacheobs.BackendMemory,
			},
		},
		{
			name: "negative operation only",
			access: cacheobs.TierAccess{
				Tier: cacheobs.TierLocalPrefix, Op: cacheobs.TierOp(-1),
				Outcome: cacheobs.OutcomeHit, Backend: cacheobs.BackendMemory,
			},
		},
		{
			name: "negative outcome only",
			access: cacheobs.TierAccess{
				Tier: cacheobs.TierLocalPrefix, Op: cacheobs.OpRead,
				Outcome: cacheobs.TierOutcome(-1), Backend: cacheobs.BackendMemory,
			},
		},
		{
			name: "negative backend only",
			access: cacheobs.TierAccess{
				Tier: cacheobs.TierLocalPrefix, Op: cacheobs.OpRead,
				Outcome: cacheobs.OutcomeHit, Backend: cacheobs.BackendClass(-1),
			},
		},
		{
			name: "every dimension uint8-max malformed with extreme payload",
			access: cacheobs.TierAccess{
				Tier: cacheobs.CacheTier(255), Op: cacheobs.TierOp(255),
				Outcome: cacheobs.TierOutcome(255), Backend: cacheobs.BackendClass(255),
				Bytes: math.MaxInt64, BytesKnown: true,
				Latency: time.Duration(math.MaxInt64), LatencyKnown: true,
			},
		},
		{
			name: "hostile operation with otherwise valid shape",
			access: cacheobs.TierAccess{
				Tier: cacheobs.TierSharedStore, Op: cacheobs.TierOp(7),
				Outcome: cacheobs.OutcomeError, Backend: cacheobs.BackendRemote,
			},
		},
	}

	// The empty access is the closed-vocabulary ZERO value: a well-formed request, not a
	// rejection. It must move accepted denominators and must NOT advance the rejected counter.
	var zeroTier cacheobs.TierAccess
	if err := cacheobs.Default.ObserveTierStrict(zeroTier); err != nil {
		t.Fatalf("zero-value TierAccess is inside the closed vocabulary, got refusal: %v", err)
	}

	before := cacheobs.Default.Snapshot()
	for i, tc := range hostile {
		t.Run(tc.name, func(t *testing.T) {
			pre := cacheobs.Default.Snapshot()
			preText := srv.renderMetrics()

			cacheobs.Default.ObserveTier(tc.access)

			post := cacheobs.Default.Snapshot()
			postText := srv.renderMetrics()

			if post.RejectedTierAccesses != pre.RejectedTierAccesses+1 {
				t.Fatalf("hostile access did not advance rejected by exactly one: before=%d after=%d",
					pre.RejectedTierAccesses, post.RejectedTierAccesses)
			}
			// A hostile observation is dropped WHOLE: accepted denominators must not move.
			if post.Turns != pre.Turns || post.PromptTokens != pre.PromptTokens ||
				post.ReusedTokens != pre.ReusedTokens {
				t.Fatalf("hostile tier access leaked into accepted denominators: before=%+v after=%+v", pre, post)
			}

			// The rejected row must render exactly once, unlabeled, and round-trip the
			// observer's uint64 exactly. A label keyed on the malformed dimension, a
			// duplicate row, or an int64-aliased value fails here.
			rowCount := 0
			for _, ln := range strings.Split(postText, "\n") {
				if strings.HasPrefix(ln, metric+" ") || strings.HasPrefix(ln, metric+"{") {
					rowCount++
				}
			}
			if rowCount != 1 {
				t.Fatalf("rejected metric must render exactly one row, got %d:\n%s", rowCount, postText)
			}
			if strings.Contains(postText, metric+"{") || strings.Contains(postText, metric+"_") {
				t.Fatalf("rejected tier counter must be unlabeled and have no derived series:\n%s", postText)
			}
			if got := metricUint64(t, postText, metric); got != post.RejectedTierAccesses {
				t.Fatalf("scraped rejected = %d, observer = %d", got, post.RejectedTierAccesses)
			}
			if strings.Contains(metricLine(postText, metric), "-") {
				t.Fatalf("rejected counter rendered a negative value: %q", metricLine(postText, metric))
			}

			// HELP/TYPE header stability under every hostile input.
			for _, want := range []string{"# HELP " + metric + " ", "# TYPE " + metric + " counter"} {
				if got := strings.Count(postText, want); got != 1 {
					t.Fatalf("header %q rendered %d times, want exactly 1:\n%s", want, got, postText)
				}
			}

			// Determinism: the same observer state must render byte-identically.
			if again := srv.renderMetrics(); again != postText {
				t.Fatalf("case %d (%s) rendered nondeterministic bytes", i, tc.name)
			}
			_ = preText
		})
	}

	// The isolated observer saw exactly the zero-value request plus len(hostile) rejections.
	final := cacheobs.Default.Snapshot()
	if final.RejectedTierAccesses != uint64(len(hostile)) {
		t.Fatalf("total rejected = %d, want %d (one per hostile case)", final.RejectedTierAccesses, len(hostile))
	}
	if final.Turns != before.Turns {
		t.Fatalf("rejected sweep changed turns: before=%d after=%d", before.Turns, final.Turns)
	}
}

// TestRejectedTierStrictRefusalNamesEveryBadDimension pins the strict path's adversarial
// contract through the gateway-facing observer: a fully malformed access must be refused
// with EVERY bad dimension named (not first-error short-circuit) and counted exactly once,
// while a nil observer is refused rather than silently counted.
func TestRejectedTierStrictRefusalNamesEveryBadDimension(t *testing.T) {
	o := cacheobs.New()
	bad := cacheobs.TierAccess{
		Tier: cacheobs.CacheTier(255), Op: cacheobs.TierOp(255),
		Outcome: cacheobs.TierOutcome(255), Backend: cacheobs.BackendClass(255),
	}
	err := o.ObserveTierStrict(bad)
	if err == nil {
		t.Fatalf("fully malformed access must be refused, got nil")
	}
	msg := err.Error()
	for _, want := range []string{"cache tier", "tier operation", "tier outcome", "backend class"} {
		if !strings.Contains(msg, want) {
			t.Fatalf("refusal %q does not name %q dimension", msg, want)
		}
	}
	if got := o.Snapshot().RejectedTierAccesses; got != 1 {
		t.Fatalf("strict refusal counted %d, want exactly 1", got)
	}

	var nilObserver *cacheobs.Observer
	if err := nilObserver.ObserveTierStrict(bad); err == nil {
		t.Fatalf("nil observer must be refused, not silently counted")
	}
}
