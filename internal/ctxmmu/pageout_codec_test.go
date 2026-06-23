package ctxmmu

import "testing"

// TestPageOutCodecDefaultsToBlob proves the MMU pages cold/quarantined bytes
// through the in-memory "blob" codec unless FAK_PAGEOUT_BACKEND selects another —
// preserving the v0.1 default exactly.
func TestPageOutCodecDefaultsToBlob(t *testing.T) {
	if got := pageOutBackendID(); got != "blob" {
		t.Fatalf("default page-out codec id = %q, want \"blob\"", got)
	}
	if got := New().codecID(); got != "blob" {
		t.Fatalf("constructed MMU codec id = %q, want \"blob\"", got)
	}
}

// TestPageOutCodecEnvOverride proves FAK_PAGEOUT_BACKEND redirects the MMU to a
// durable codec (e.g. "blobfs") so quarantined bytes can survive a restart.
func TestPageOutCodecEnvOverride(t *testing.T) {
	t.Setenv("FAK_PAGEOUT_BACKEND", "blobfs")
	if got := pageOutBackendID(); got != "blobfs" {
		t.Fatalf("env override id = %q, want \"blobfs\"", got)
	}
	if got := New().codecID(); got != "blobfs" {
		t.Fatalf("MMU honored env codec id = %q, want \"blobfs\"", got)
	}
}

// TestZeroValueMMUFallsBackToBlob proves a struct-literal MMU that bypassed the
// constructors still resolves a usable codec id (fail-safe, never the empty id).
func TestZeroValueMMUFallsBackToBlob(t *testing.T) {
	var m MMU
	if got := m.codecID(); got != "blob" {
		t.Fatalf("zero-value MMU codec id = %q, want \"blob\"", got)
	}
}
