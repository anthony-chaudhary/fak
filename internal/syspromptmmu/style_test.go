package syspromptmmu

import (
	"strings"
	"testing"
)

// style_test.go — the #5051 witness: a NAMED style is the user-facing selection surface
// over the #3308 terseness producer. The contract under test is the one the ticket makes
// load-bearing: the name set is CLOSED, an unrecognized name is fail-safe (resolves to
// `full` / steering OFF and never fabricates a block), and the read-out exposes the exact
// bytes a style would append so an operator can SEE and PROVE the seam without a model.

// TestStyleLevelClosedSet pins the name→level map the ticket specifies. These pairings are
// the API: changing one silently re-steers every session that selected that name.
func TestStyleLevelClosedSet(t *testing.T) {
	want := map[string]int{
		StyleFull:    SteeringOff,
		StyleConcise: 1,
		StyleBrief:   2,
		StyleTerse:   3,
		StyleMinimal: 4,
	}
	for name, wantLevel := range want {
		level, known := StyleLevel(name)
		if !known {
			t.Errorf("style %q: known=false, want true (it is in the closed set)", name)
			continue
		}
		if level != wantLevel {
			t.Errorf("style %q: level %d, want %d", name, level, wantLevel)
		}
	}
}

// TestStyleUnknownIsFailSafe is the refusal witness. An unknown name must resolve to the
// OFF level and report known=false — never fabricate a steering block, mirroring
// steeringText's closed-set contract. This is the property that keeps a typo in an env var
// from silently steering a session.
func TestStyleUnknownIsFailSafe(t *testing.T) {
	for _, name := range []string{"", "  ", "verbose", "full-ish", "5", "minimal!", "<script>"} {
		level, known := StyleLevel(name)
		if known {
			t.Errorf("style %q: known=true, want false (not in the closed set)", name)
		}
		if level != SteeringOff {
			t.Errorf("style %q: level %d, want SteeringOff(%d)", name, level, SteeringOff)
		}
		got := DescribeStyle(name)
		if got.Applied || got.Segment != "" || got.Witness != "" {
			t.Errorf("style %q: unknown name fabricated a block: %+v", name, got)
		}
		if got.Style != StyleFull {
			t.Errorf("style %q: resolved style %q, want %q (fail-safe default)", name, got.Style, StyleFull)
		}
	}
}

// TestStyleNamesAreOrderedAndComplete asserts StyleNames is exactly the closed set, ordered
// by increasing terseness — the order an operator sees in a read-out or a help string.
func TestStyleNamesAreOrderedAndComplete(t *testing.T) {
	names := StyleNames()
	if len(names) != 11 {
		t.Fatalf("StyleNames() returned %d names, want 11: %v", len(names), names)
	}
	lastLevel := -1
	seen := map[string]bool{}
	for _, name := range names {
		level, known := StyleLevel(name)
		if !known {
			t.Errorf("StyleNames() returned %q which StyleLevel does not know", name)
			continue
		}
		if level < lastLevel {
			t.Errorf("StyleNames() not ordered by nondecreasing terseness at %q (level %d after %d)", name, level, lastLevel)
		}
		lastLevel = level
		seen[name] = true
	}
	for _, name := range []string{StyleFull, StyleConcise, StyleBrief, StyleTerse, StyleMinimal, "native:low", "native:medium", "native:high", "caveman:native:low", "caveman:native:medium", "caveman:native:high"} {
		if !seen[name] {
			t.Fatalf("StyleNames() missing %q: %v", name, names)
		}
	}
}

// TestStyleSegmentMatchesProducer is the composition witness: a named style must append
// EXACTLY the bytes its level's steering segment produces. If these ever diverge, the
// selection surface has become a second source of truth for the steering text.
func TestStyleSegmentMatchesProducer(t *testing.T) {
	for _, name := range StyleNames() {
		level, known := StyleLevel(name)
		if !known {
			t.Fatalf("style %q unknown", name)
		}
		got := DescribeStyle(name)
		if got.Style != name || got.Level != level || !got.Known {
			t.Errorf("style %q: readout mismatch: %+v", name, got)
		}
		seg, ok := SteeringSegment(level)
		if !ok {
			// level 0 (full) — steering is off, so the readout must carry no block.
			if got.Applied || got.Segment != "" || got.Witness != "" {
				t.Errorf("style %q (level %d): steering is off but readout carries a block: %+v", name, level, got)
			}
			continue
		}
		if !got.Applied {
			t.Errorf("style %q: applied=false, want true (level %d produces a segment)", name, level)
		}
		if got.Segment != string(seg.Content) {
			t.Errorf("style %q: readout segment is not the producer's bytes:\n got %q\nwant %q", name, got.Segment, string(seg.Content))
		}
		if got.Witness != seg.Witness {
			t.Errorf("style %q: witness %q, want %q", name, got.Witness, seg.Witness)
		}
		if !strings.HasPrefix(got.Segment, steeringSentinelOpen) {
			t.Errorf("style %q: readout segment is not sentinel-wrapped: %q", name, got.Segment)
		}
	}
}

// TestStyleNameNormalization asserts selection is forgiving about surrounding space and
// letter case — an env var typed by a human — while the SET itself stays closed. A
// normalized name resolves to the same level as its canonical form and nothing else.
func TestStyleNameNormalization(t *testing.T) {
	for _, in := range []string{"TERSE", "  terse", "terse  ", "Terse", " TeRsE "} {
		level, known := StyleLevel(in)
		if !known || level != 3 {
			t.Errorf("StyleLevel(%q) = (%d,%v), want (3,true)", in, level, known)
		}
		if got := DescribeStyle(in); got.Style != StyleTerse {
			t.Errorf("DescribeStyle(%q).Style = %q, want %q (canonicalized)", in, got.Style, StyleTerse)
		}
	}
}

// TestStyleFromEnv drives the actual selection surface: the env knob a user sets. Unset and
// unrecognized both mean OFF, and the read-out names the resolved style either way so the
// operator can tell "I chose full" from "my value was not understood".
func TestStyleFromEnv(t *testing.T) {
	cases := []struct {
		env         string
		wantStyle   string
		wantLevel   int
		wantKnown   bool
		wantApplied bool
	}{
		{"", StyleFull, SteeringOff, false, false},
		{"full", StyleFull, SteeringOff, true, false},
		{"concise", StyleConcise, 1, true, true},
		{"brief", StyleBrief, 2, true, true},
		{"terse", StyleTerse, 3, true, true},
		{"minimal", StyleMinimal, 4, true, true},
		{"MINIMAL", StyleMinimal, 4, true, true},
		{"nonsense", StyleFull, SteeringOff, false, false},
	}
	for _, tc := range cases {
		got := StyleFromEnv(func(k string) string {
			if k != StyleEnvVar {
				t.Errorf("read env %q, want %q", k, StyleEnvVar)
			}
			return tc.env
		})
		if got.Style != tc.wantStyle || got.Level != tc.wantLevel || got.Known != tc.wantKnown || got.Applied != tc.wantApplied {
			t.Errorf("StyleFromEnv(%q) = %+v, want style=%q level=%d known=%v applied=%v",
				tc.env, got, tc.wantStyle, tc.wantLevel, tc.wantKnown, tc.wantApplied)
		}
	}
}

// TestStyleFromEnvNilGetenvIsOff guards the zero-config path: a nil lookup must not panic
// and must report steering OFF.
func TestStyleFromEnvNilGetenvIsOff(t *testing.T) {
	got := StyleFromEnv(nil)
	if got.Style != StyleFull || got.Applied || got.Level != SteeringOff {
		t.Errorf("StyleFromEnv(nil) = %+v, want the fail-safe full/off readout", got)
	}
}

// TestStyleReadoutIsByteStable asserts two read-outs of the same style are identical — the
// same byte-stability property #3308 proves for the producer, carried through the selection
// surface so a style cannot become a per-turn cache-buster.
func TestStyleReadoutIsByteStable(t *testing.T) {
	for _, name := range StyleNames() {
		a, b := DescribeStyle(name), DescribeStyle(name)
		if a != b {
			t.Errorf("style %q: two read-outs differ: %+v vs %+v", name, a, b)
		}
	}
}

func TestComposableStyleFamiliesAndIntensities(t *testing.T) {
	tests := []struct {
		name      string
		level     int
		family    string
		intensity string
	}{
		{"native:low", 1, StyleFamilyNative, "low"},
		{"native:medium", 2, StyleFamilyNative, "medium"},
		{"native:high", 3, StyleFamilyNative, "high"},
		{"caveman:native:low", 1, StyleFamilyCaveman, "low"},
		{"caveman:native:medium", 2, StyleFamilyCaveman, "medium"},
		{"caveman:native:high", 3, StyleFamilyCaveman, "high"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DescribeStyle(tt.name)
			if !got.Known || !got.Applied || got.Level != tt.level || got.Family != tt.family || got.Intensity != tt.intensity {
				t.Fatalf("DescribeStyle(%q) = %+v", tt.name, got)
			}
			seg, ok := StyleSegment(tt.name)
			if !ok {
				t.Fatalf("StyleSegment(%q) refused", tt.name)
			}
			want, _ := SteeringSegment(tt.level)
			if string(seg.Content) != string(want.Content) || seg.Witness != want.Witness {
				t.Fatalf("StyleSegment(%q) drifted from governed level %d", tt.name, tt.level)
			}
		})
	}
}

func TestOriginalStyleIsNotSilentlyAliased(t *testing.T) {
	for _, name := range []string{"original", "caveman:original", "native:original", "caveman:auto", "ponytail:medium"} {
		got := DescribeStyle(name)
		if got.Known || got.Applied || got.Level != SteeringOff {
			t.Fatalf("DescribeStyle(%q) = %+v; foreign/or unsupported profiles must fail safe", name, got)
		}
	}
}
