package lease

import (
	"testing"
	"time"
)

func TestLeaseGrantAndProtect(t *testing.T) {
	lt := NewTable(30000)

	keyHash := uint64(12345)
	lt.Grant(keyHash, 5000) // 5 second lease

	if !lt.IsProtected(keyHash) {
		t.Error("key should be protected after grant")
	}

	// Unknown key
	if lt.IsProtected(99999) {
		t.Error("unknown key should not be protected")
	}
}

func TestLeaseExpiry(t *testing.T) {
	lt := NewTable(30000)

	keyHash := uint64(12345)
	lt.Grant(keyHash, 50) // 50ms lease

	if !lt.IsProtected(keyHash) {
		t.Error("key should be protected immediately after grant")
	}

	time.Sleep(100 * time.Millisecond)

	if lt.IsProtected(keyHash) {
		t.Error("key should not be protected after lease expiry")
	}
}

func TestPinUnpin(t *testing.T) {
	lt := NewTable(30000)

	keyHash := uint64(12345)
	lt.Pin(keyHash)

	if !lt.IsProtected(keyHash) {
		t.Error("pinned key should be protected")
	}

	lt.Unpin(keyHash)
	if lt.IsProtected(keyHash) {
		t.Error("unpinned key should not be protected")
	}
}

func TestLeaseMaxDuration(t *testing.T) {
	lt := NewTable(100) // 100ms max

	keyHash := uint64(12345)
	lt.Grant(keyHash, 99999) // request way more than max

	// Should expire based on max (100ms), not requested (99999ms)
	time.Sleep(200 * time.Millisecond)
	if lt.IsProtected(keyHash) {
		t.Error("lease should have expired based on max duration")
	}
}

func TestCleanup(t *testing.T) {
	lt := NewTable(30000)

	for i := uint64(0); i < 100; i++ {
		lt.Grant(i, 50) // 50ms
	}

	if lt.LeaseCount() != 100 {
		t.Errorf("lease count: got %d, want 100", lt.LeaseCount())
	}

	time.Sleep(100 * time.Millisecond)
	lt.Cleanup()

	if lt.LeaseCount() != 0 {
		t.Errorf("lease count after cleanup: got %d, want 0", lt.LeaseCount())
	}
}
