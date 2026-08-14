package syspromptmmu

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/promptmmu"
)

// steering_test.go — the #3308 witness: the terseness steering segment is byte-stable
// per level, level 0 is a refused no-op, distinct levels are distinct bytes, and splicing
// a steering segment after the breakpoint leaves the resident prefix digest UNCHANGED
// (the cache-safety invariant).

// TestSteeringSegmentByteStable asserts the same level always produces identical bytes
// (and an identical Witness), wrapped in the steering sentinel, in the Rung-3 overlay
// segment shape.
func TestSteeringSegmentByteStable(t *testing.T) {
	for level := 1; level <= 4; level++ {
		a, okA := steeringSegment(level)
		b, okB := steeringSegment(level)
		if !okA || !okB {
			t.Fatalf("level %d: ok=(%v,%v), want (true,true)", level, okA, okB)
		}
		if !bytes.Equal(a.Content, b.Content) {
			t.Errorf("level %d: two calls produced different bytes", level)
		}
		if a.Witness != b.Witness || a.Witness != WitnessFor(a.Content) {
			t.Errorf("level %d: witness not stable/content-derived: %q vs %q", level, a.Witness, b.Witness)
		}
		if a.Kind != cachemeta.SegMessage {
			t.Errorf("level %d: kind %q, want %q (overlay tail content)", level, a.Kind, cachemeta.SegMessage)
		}
		text := string(a.Content)
		if !strings.HasPrefix(text, steeringSentinelOpen) || !strings.HasSuffix(text, steeringSentinelClose) {
			t.Errorf("level %d: segment is not sentinel-wrapped: %q", level, text)
		}
	}
}

// TestSteeringLevelZeroNoOp asserts level 0 (and any out-of-range level) is refused with
// the zero segment — steering is opt-in, and an unknown level must never fabricate text.
func TestSteeringLevelZeroNoOp(t *testing.T) {
	for _, level := range []int{0, -1, 5, 100} {
		seg, ok := steeringSegment(level)
		if ok {
			t.Errorf("level %d: ok=true, want false (no-op)", level)
		}
		if len(seg.Content) != 0 || seg.Witness != "" || seg.Tokens != 0 {
			t.Errorf("level %d: refused segment is not the zero value: %+v", level, seg)
		}
	}
}

// TestSteeringLevelsDistinct asserts every valid level produces distinct bytes (and a
// distinct witness), so a level change is always a real block replacement.
func TestSteeringLevelsDistinct(t *testing.T) {
	seen := map[string]int{}
	for level := 1; level <= 4; level++ {
		seg, ok := steeringSegment(level)
		if !ok {
			t.Fatalf("level %d: ok=false, want true", level)
		}
		if prev, dup := seen[seg.Witness]; dup {
			t.Errorf("levels %d and %d produced identical bytes", prev, level)
		}
		seen[seg.Witness] = level
	}
}

// TestSteeringSpliceCacheSafe is the cache-safety proof: splicing a steering segment in
// after the breakpoint (alongside the queried overlay) leaves the resident spine+policy
// prefix byte-identical AND the Rung-6 re-derived prefix digest unchanged — the cached
// prefix still hits while only the small tail block changes.
func TestSteeringSpliceCacheSafe(t *testing.T) {
	plan := BaseContextPlan()
	body := bodyWith(t, BuildSystemValue(plan, []cachemeta.PromptSegment{overlaySeg("a card")}), nil)

	before := AuditBaseContext(body)
	if before.Status != AuditOK {
		t.Fatalf("pre-splice audit: status %q, want %q", before.Status, AuditOK)
	}
	_, prefixEnd, _, ok := promptmmu.ArraySplicePoints(body, "system")
	if !ok {
		t.Fatal("could not anchor the cached prefix on the built body")
	}
	prefix := append([]byte(nil), body[:prefixEnd]...)

	steer, ok := steeringSegment(3)
	if !ok {
		t.Fatal("level 3 must produce a segment")
	}
	res := SpliceSystemOverlay(body, plan, []cachemeta.PromptSegment{overlaySeg("a card"), steer}, decodeOK)
	if !res.Changed {
		t.Fatalf("expected a splice, got identity (%s)", res.SkipReason)
	}
	if len(res.Body) < len(prefix) || !bytes.Equal(res.Body[:len(prefix)], prefix) {
		t.Fatal("a steering segment broke the resident-prefix byte invariant")
	}

	after := AuditBaseContext(res.Body)
	if after.Status != AuditOK {
		t.Fatalf("post-splice audit: status %q, want %q", after.Status, AuditOK)
	}
	if after.GotDigest != before.GotDigest || after.GotDigest != before.ExpectDigest {
		t.Fatalf("resident prefix digest moved across the steering splice: before=%q after=%q",
			before.GotDigest, after.GotDigest)
	}

	// A level change replaces the block in place through the same swap — the prefix
	// digest still never moves.
	steer2, ok := steeringSegment(1)
	if !ok {
		t.Fatal("level 1 must produce a segment")
	}
	res2 := SpliceSystemOverlay(res.Body, plan, []cachemeta.PromptSegment{overlaySeg("a card"), steer2}, decodeOK)
	if !res2.Changed {
		t.Fatalf("expected a level-change splice, got identity (%s)", res2.SkipReason)
	}
	if !bytes.Equal(res2.Body[:len(prefix)], prefix) {
		t.Fatal("a steering level change broke the resident-prefix byte invariant")
	}
	if a := AuditBaseContext(res2.Body); a.Status != AuditOK || a.GotDigest != before.ExpectDigest {
		t.Fatalf("level-change audit: status=%q digest=%q, want ok/unchanged", a.Status, a.GotDigest)
	}
}

func TestSteeringTextUsesPositiveSignalFirstFrame(t *testing.T) {
	positive := []string{"lead with the answer", "state the result first", "deliver the essential result", "return the requested artifact"}
	for i, want := range positive {
		level := i + 1
		text, ok := steeringText(level)
		if !ok {
			t.Fatalf("steeringText(%d) missing", level)
		}
		lower := strings.ToLower(text)
		if !strings.Contains(lower, want) {
			t.Errorf("steeringText(%d) = %q, want positive direction %q", level, text, want)
		}
		for _, prohibition := range []string{"no preamble", "no recap", "do not", "avoid "} {
			if strings.Contains(lower, prohibition) {
				t.Errorf("steeringText(%d) uses prohibition %q: %q", level, prohibition, text)
			}
		}
	}
}

func TestSignalFirstSkillMatchesNativeProfile(t *testing.T) {
	path := filepath.Join("..", "..", ".claude", "skills", "signal-first", "SKILL.md")
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read native skill: %v", err)
	}
	text := strings.ToLower(string(body))
	for _, want := range []string{
		"name: signal-first",
		"caveman-style output",
		"put the result, decision, or next action in the first line",
		"when compression competes with correctness, keep correctness",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("native skill missing %q", want)
		}
	}
}
