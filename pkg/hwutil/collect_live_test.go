package hwutil

// Live smoke: one real Collect() against this host's OS surfaces, asserting
// the wire contract holds end to end (schema, identity, null-on-error,
// closed error codes) without pinning any host-specific numbers.

import (
	"context"
	"os/exec"
	"runtime"
	"testing"
	"time"
)

// fak-test:runtime medium est=30s lane=default
func TestCollectLiveHostSmoke(t *testing.T) {
	if runtime.GOOS == "windows" {
		if _, err := exec.LookPath("powershell.exe"); err != nil {
			t.Skip("powershell.exe unavailable")
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	snap, err := Collect(ctx, Options{Interval: 200 * time.Millisecond, Samples: 1})
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if snap.Schema != Schema || snap.Host == "" || snap.Os == "" || snap.Arch == "" {
		t.Fatalf("identity missing: %+v", snap)
	}
	if snap.At.IsZero() {
		t.Fatal("at is zero")
	}
	if err := snap.Validate(); err != nil {
		t.Fatalf("Validate on live snapshot: %v", err)
	}
	// Every family must be either populated (>=1 numeric counter) or honestly
	// nil plus a named error — never silently zero-filled.
	populated := 0
	if snap.CPU != nil && snap.CPU.UtilPct != nil {
		populated++
	}
	if snap.DRAM != nil && snap.DRAM.TotalBytes != nil {
		populated++
	}
	if len(snap.Disks) > 0 {
		populated++
	}
	if len(snap.Net) > 0 {
		populated++
	}
	if len(snap.GPUs) > 0 {
		populated++
	}
	if snap.Error != "" && populated == 0 {
		t.Fatalf("total failure must carry a closed code, got %q", snap.Error)
	}
	t.Logf("live snapshot: counters=%d families_populated=%d unread=%d error=%q",
		snap.NumericCounterCount(), populated, snap.UnreadCount(), snap.Error)
}
