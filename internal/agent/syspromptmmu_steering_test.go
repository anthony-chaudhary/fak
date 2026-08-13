package agent

// syspromptmmu_steering_test.go — the #5047 acceptance witness: terseness steering (#3308's
// producer) is WIRED to the live request path. The producer was request-path-dead — it could
// mint a segment but nothing ever appended one — so these tests drive the real system-block
// spine (BuildOwnedSystemBlock, the owned loop's one authored system block) and prove the
// three contract points the issue names:
//
//   - OFF by default: an unconfigured request path emits the byte-identical value it emitted
//     before this wiring (the forward path does not move);
//   - cache-safe when ON: the resident prefix bytes, the re-derived prefix digest, AND the
//     cache_control breakpoint index are all unchanged vs. OFF — only tail bytes are added;
//   - idempotent: the same level re-applied is byte-identical.
//
// Honest fence (inherited from #3308): this measures appended bytes and prefix identity. It
// asserts NO token savings — that needs a holdout comparison this rung does not run.

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/promptmmu"
	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

// steerItems is the authored capability overlay every steering case is built on, so the
// steering block is proven to land ALONGSIDE queried capability segments, not instead of them.
func steerItems() [][]byte {
	return [][]byte{
		[]byte("capability card: search_flights(origin, destination, date) — queried on demand"),
	}
}

// breakpointIndex reports the index of the system block carrying the single cache_control
// breakpoint — the cached-prefix boundary the steering segment must never move.
func breakpointIndex(t *testing.T, b SystemBlock) int {
	t.Helper()
	idx, _, _, ok := promptmmu.ArraySplicePoints(b.RequestBody(), "system")
	if !ok {
		t.Fatal("could not anchor the cache_control breakpoint on the built body")
	}
	return idx
}

// TestOwnedSystemBlockSteeringOffByDefault pins the opt-in contract: with FAK_STEERING_LEVEL
// unset, the request path applies no steering and the value is byte-identical to the
// level-explicit OFF build — the forward path is exactly what it was before #5047.
func TestOwnedSystemBlockSteeringOffByDefault(t *testing.T) {
	t.Setenv(syspromptmmu.SteeringEnvVar, "")

	def := BuildOwnedSystemBlock(steerItems(), spmmuPassWitness)
	off := buildOwnedSystemBlockAt(steerItems(), spmmuPassWitness, syspromptmmu.SteeringOff)

	if def.Steering != syspromptmmu.SteeringOff {
		t.Fatalf("default steering = %d, want SteeringOff (steering must never self-enable)", def.Steering)
	}
	if !bytes.Equal(def.Value, off.Value) {
		t.Fatal("unset knob did not produce the byte-identical no-steering value")
	}
	if strings.Contains(string(def.Value), "fak:steering") {
		t.Fatal("a steering block reached the default (off) forward path")
	}
	if !def.CacheStable() {
		t.Fatalf("default block not cache-stable: audit %q", def.Audit.Status)
	}
}

// TestOwnedSystemBlockSteeringEnvKnob proves the env knob is what turns steering on at the
// request path, and that an out-of-range knob is reported as the no-op it really is.
func TestOwnedSystemBlockSteeringEnvKnob(t *testing.T) {
	t.Setenv(syspromptmmu.SteeringEnvVar, "3")
	on := BuildOwnedSystemBlock(steerItems(), spmmuPassWitness)
	if on.Steering != 3 {
		t.Fatalf("FAK_STEERING_LEVEL=3 applied level %d, want 3", on.Steering)
	}
	if !strings.Contains(string(on.Value), "fak:steering") {
		t.Fatal("level 3 configured but no steering block reached the request path")
	}

	t.Setenv(syspromptmmu.SteeringEnvVar, "9")
	bad := BuildOwnedSystemBlock(steerItems(), spmmuPassWitness)
	if bad.Steering != syspromptmmu.SteeringOff {
		t.Fatalf("out-of-range knob applied level %d, want SteeringOff", bad.Steering)
	}
	if strings.Contains(string(bad.Value), "fak:steering") {
		t.Fatal("an out-of-range knob fabricated a steering block")
	}
}

// TestOwnedSystemBlockSteeringCacheSafe is THE acceptance gate: steering on leaves the
// resident prefix bytes, the re-derived prefix digest, and the breakpoint index all
// unchanged vs. off. Only after-breakpoint tail bytes are added — so the cached prefix
// still hits while the model's verbosity is steered.
func TestOwnedSystemBlockSteeringCacheSafe(t *testing.T) {
	off := buildOwnedSystemBlockAt(steerItems(), spmmuPassWitness, syspromptmmu.SteeringOff)
	if !off.CacheStable() {
		t.Fatalf("baseline (off) not cache-stable: audit %q", off.Audit.Status)
	}
	offBreak := breakpointIndex(t, off)

	// The resident head is the byte-exact prefix every turn must reuse (array still open).
	head := OwnedResidentHead()
	resident := head[:len(head)-1] // drop the trailing ']'

	for level := 1; level <= 4; level++ {
		on := buildOwnedSystemBlockAt(steerItems(), spmmuPassWitness, level)

		if on.Steering != level {
			t.Errorf("level %d: reported %d", level, on.Steering)
		}
		if !on.CacheStable() {
			t.Errorf("level %d: not cache-stable: audit %q", level, on.Audit.Status)
			continue
		}
		// Invariant 1+2: the resident prefix is byte-identical and its digest never moves.
		if !bytes.HasPrefix(on.Value, resident) {
			t.Errorf("level %d: steering re-serialized the resident prefix", level)
		}
		if on.Audit.GotDigest != off.Audit.GotDigest {
			t.Errorf("level %d: resident prefix digest moved: %q vs off %q",
				level, on.Audit.GotDigest, off.Audit.GotDigest)
		}
		if on.Audit.GotDigest != on.Audit.ExpectDigest {
			t.Errorf("level %d: realized spine != planned spine", level)
		}
		// The breakpoint does not move: steering lands strictly AFTER it.
		if got := breakpointIndex(t, on); got != offBreak {
			t.Errorf("level %d: cache_control breakpoint moved %d -> %d", level, offBreak, got)
		}
		// The steering block rides alongside the queried capability card, not instead of it.
		if on.Overlays != off.Overlays {
			t.Errorf("level %d: steering displaced %d authored overlay item(s)", level, off.Overlays-on.Overlays)
		}
		// It only ADDS tail bytes — the whole off-value prefix survives (the array's
		// off-tail is a prefix of the on-tail, since steering appends last).
		if len(on.Value) <= len(off.Value) {
			t.Errorf("level %d: steering added %d bytes, want > 0", level, len(on.Value)-len(off.Value))
		}
		if !bytes.HasPrefix(on.Value, off.Value[:len(off.Value)-1]) {
			t.Errorf("level %d: steering perturbed the pre-existing blocks", level)
		}
	}
}

// TestOwnedSystemBlockSteeringIdempotent asserts re-applying the same level on the request
// path is byte-identical — a re-issued turn at a steady level can never bust the cache.
func TestOwnedSystemBlockSteeringIdempotent(t *testing.T) {
	a := buildOwnedSystemBlockAt(steerItems(), spmmuPassWitness, 2)
	b := buildOwnedSystemBlockAt(steerItems(), spmmuPassWitness, 2)
	if !bytes.Equal(a.Value, b.Value) {
		t.Fatal("the same steering level produced different bytes across two builds")
	}

	// A level CHANGE moves only tail bytes — the resident prefix digest still holds.
	c := buildOwnedSystemBlockAt(steerItems(), spmmuPassWitness, 4)
	if bytes.Equal(a.Value, c.Value) {
		t.Fatal("levels 2 and 4 produced identical bytes")
	}
	if c.Audit.GotDigest != a.Audit.GotDigest {
		t.Fatalf("a level change moved the resident prefix digest: %q -> %q",
			a.Audit.GotDigest, c.Audit.GotDigest)
	}
}

func TestNamedCavemanStyleFlowsThroughOwnedSystemBlock(t *testing.T) {
	t.Setenv(syspromptmmu.StyleEnvVar, "caveman:native:medium")
	t.Setenv(syspromptmmu.SteeringEnvVar, "4") // named style takes precedence

	b := BuildOwnedSystemBlock(steerItems(), spmmuPassWitness)
	if b.Steering != 2 || b.Style != "caveman:native:medium" || b.StyleFamily != syspromptmmu.StyleFamilyCaveman+":"+syspromptmmu.StyleFamilyNative || b.StyleIntensity != "medium" {
		t.Fatalf("named style readout = steering=%d style=%q family=%q intensity=%q", b.Steering, b.Style, b.StyleFamily, b.StyleIntensity)
	}
	seg, _ := syspromptmmu.StyleSegment("caveman:native:medium")
	if !bytes.Contains(b.Value, seg.Content) {
		t.Fatal("owned system block does not carry selected named style bytes")
	}
}

func TestUnknownNamedStyleFailsSafeInsteadOfFallingThrough(t *testing.T) {
	t.Setenv(syspromptmmu.StyleEnvVar, "caveman:original")
	t.Setenv(syspromptmmu.SteeringEnvVar, "4")

	b := BuildOwnedSystemBlock(steerItems(), spmmuPassWitness)
	if b.Steering != syspromptmmu.SteeringOff {
		t.Fatalf("unknown explicit named style fell through to numeric steering: %+v", b)
	}
	if b.Style != syspromptmmu.StyleFull || b.StyleFamily != syspromptmmu.StyleFamilyNative || b.StyleIntensity != "off" {
		t.Fatalf("unknown style readout = style=%q family=%q intensity=%q", b.Style, b.StyleFamily, b.StyleIntensity)
	}
}
