package ggufload

import (
	"strings"
	"testing"
)

// gguf_load_mode_test.go pins the LoadMode enum + ResolveLoadMode resolution contract
// (gguf_mmap.go): EVERY request × support combination — including mmapSupported=false
// as on native Windows — must resolve to the expected load path, and every degraded
// explicit request must carry a non-empty, pinned human-readable reason instead of
// failing. Also pins that the FAK_GGUF_MMAP opt-in gate maps to an mmap request with
// its historical semantics intact. Design reference (ADAPT): ggml-org/llama.cpp
// src/llama-model-loader.cpp:559@83078fec0db82d6b5a00d9599062c38c39145755 (MIT).

func TestResolveLoadModeDegradesWithReason(t *testing.T) {
	// Every degrade reason must name the fallback target so an operator reading the
	// log knows exactly which reader path took over.
	const fallbackPhrase = "falling back to plain file reads"

	cases := []struct {
		name           string
		requested      LoadMode
		mmapSupported  bool
		mlockSupported bool
		wantResolved   LoadMode
		wantReason     string
	}{
		// --- AUTO: the policy pick. mmap when supported, else none; never a reason.
		{"auto/mmap+mlock supported", LoadModeAuto, true, true, LoadModeMmap, ""},
		{"auto/mmap only", LoadModeAuto, true, false, LoadModeMmap, ""},
		{"auto/none supported (Windows)", LoadModeAuto, false, true, LoadModeNone, ""},
		{"auto/nothing supported", LoadModeAuto, false, false, LoadModeNone, ""},

		// --- NONE: honored as-is, never degraded.
		{"none/mmap+mlock supported", LoadModeNone, true, true, LoadModeNone, ""},
		{"none/mmap only", LoadModeNone, true, false, LoadModeNone, ""},
		{"none/none supported (Windows)", LoadModeNone, false, true, LoadModeNone, ""},
		{"none/nothing supported", LoadModeNone, false, false, LoadModeNone, ""},

		// --- MMAP: honored when supported; degrades to none with a reason otherwise.
		{"mmap/honored (both supported)", LoadModeMmap, true, true, LoadModeMmap, ""},
		{"mmap/honored (mlock irrelevant)", LoadModeMmap, true, false, LoadModeMmap, ""},
		{"mmap/degraded (Windows)", LoadModeMmap, false, true, LoadModeNone, "mmap not supported on this platform; falling back to plain file reads"},
		{"mmap/degraded (nothing supported)", LoadModeMmap, false, false, LoadModeNone, "mmap not supported on this platform; falling back to plain file reads"},

		// --- MLOCK: honored when supported; degrades to the plain-file-load path
		// (not the mapped one) with a reason otherwise.
		{"mlock/honored (both supported)", LoadModeMlock, true, true, LoadModeMlock, ""},
		{"mlock/honored (mmap irrelevant)", LoadModeMlock, false, true, LoadModeMlock, ""},
		{"mlock/degraded (mmap available too)", LoadModeMlock, true, false, LoadModeNone, "mlock not supported on this platform; falling back to plain file reads"},
		{"mlock/degraded (nothing supported)", LoadModeMlock, false, false, LoadModeNone, "mlock not supported on this platform; falling back to plain file reads"},

		// --- DIRECT-IO: never honored in this build (no direct-I/O syscalls); always
		// degrades to none with a reason, regardless of either capability flag.
		{"direct-io/degraded (both supported)", LoadModeDirectIO, true, true, LoadModeNone, "direct I/O not implemented in this build; falling back to plain file reads"},
		{"direct-io/degraded (mmap only)", LoadModeDirectIO, true, false, LoadModeNone, "direct I/O not implemented in this build; falling back to plain file reads"},
		{"direct-io/degraded (Windows)", LoadModeDirectIO, false, true, LoadModeNone, "direct I/O not implemented in this build; falling back to plain file reads"},
		{"direct-io/degraded (nothing supported)", LoadModeDirectIO, false, false, LoadModeNone, "direct I/O not implemented in this build; falling back to plain file reads"},

		// --- UNKNOWN values: closed contract — degrade to none, naming the value.
		{"unknown 99 (both supported)", LoadMode(99), true, true, LoadModeNone, "unknown load mode LoadMode(99); falling back to plain file reads"},
		{"unknown -7 (nothing supported)", LoadMode(-7), false, false, LoadModeNone, "unknown load mode LoadMode(-7); falling back to plain file reads"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, reason := ResolveLoadMode(tc.requested, tc.mmapSupported, tc.mlockSupported)
			if got != tc.wantResolved {
				t.Fatalf("ResolveLoadMode(%s, mmap=%v, mlock=%v) resolved %s, want %s",
					tc.requested, tc.mmapSupported, tc.mlockSupported, got, tc.wantResolved)
			}
			if reason != tc.wantReason {
				t.Fatalf("reason = %q, want %q", reason, tc.wantReason)
			}
			// Degraded explicit requests must carry a human-readable reason naming the
			// fallback target; honored picks and AUTO's policy pick must carry none.
			degraded := got != tc.requested && tc.requested != LoadModeAuto
			if degraded && !strings.Contains(reason, fallbackPhrase) {
				t.Fatalf("degraded request %s has reason %q without %q", tc.requested, reason, fallbackPhrase)
			}
			if !degraded && reason != "" {
				t.Fatalf("%s: non-degraded case has reason %q, want empty", tc.name, reason)
			}
		})
	}
}

// TestGGUFMmapGateMapsToMmapRequest pins that FAK_GGUF_MMAP keeps mapping to an mmap
// REQUEST with its historical semantics exactly: only the documented truthy spellings
// enable it, and every other value (including unset) preserves the historical
// os.Open + ReadAt default (a none request). It also pins the end-to-end shape of the
// gate→resolver bridge: where the platform supports mmap the gate-on request is
// honored; where it does not (native Windows) it degrades with a reason instead of
// failing — matching today's openShardReader fall-through.
func TestGGUFMmapGateMapsToMmapRequest(t *testing.T) {
	// Gate off (unset / falsy): the historical default — a none request.
	for _, off := range []string{"", "0", "off", "false", "banana"} {
		t.Setenv("FAK_GGUF_MMAP", off)
		resetGGUFMmapGateForTest()
		if got := ggufMmapRequest(); got != LoadModeNone {
			t.Fatalf("FAK_GGUF_MMAP=%q request = %s, want none", off, got)
		}
	}

	// Gate on: every documented truthy spelling maps to an mmap request.
	for _, on := range []string{"1", "on", "true", "TRUE", " On "} {
		t.Setenv("FAK_GGUF_MMAP", on)
		resetGGUFMmapGateForTest()
		if got := ggufMmapRequest(); got != LoadModeMmap {
			t.Fatalf("FAK_GGUF_MMAP=%q request = %s, want mmap", on, got)
		}
	}

	// End-to-end shape of the gate→resolver bridge.
	t.Setenv("FAK_GGUF_MMAP", "1")
	resetGGUFMmapGateForTest()
	req := ggufMmapRequest()
	if req != LoadModeMmap {
		t.Fatalf("gate-on request = %s, want mmap", req)
	}
	if got, reason := ResolveLoadMode(req, true, false); got != LoadModeMmap || reason != "" {
		t.Fatalf("gate-on request on an mmap-capable platform resolved (%s, %q), want (mmap, \"\")", got, reason)
	}
	if got, reason := ResolveLoadMode(req, false, false); got != LoadModeNone || reason == "" {
		t.Fatalf("gate-on request on a non-mmap platform (Windows) resolved (%s, %q), want (none, non-empty reason)", got, reason)
	}

	// Restore the default arm for any later test in this process.
	t.Setenv("FAK_GGUF_MMAP", "")
	resetGGUFMmapGateForTest()
}
