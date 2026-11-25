package ctxmmu

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/flock"
)

func quarantineTestDigest(i int) string { return fmt.Sprintf("%064x", i) }

// A peer holding the WAL lock for longer than one attempt must not brick the
// authority: once the lock is released, writes and reads recover.
func TestQuarantineLedgerTransientLockFailureRecovers(t *testing.T) {
	p := filepath.Join(t.TempDir(), "quarantine.jsonl")
	l := NewQuarantineLedger(p)
	if err := l.RecordQuarantine(quarantineTestDigest(1)); err != nil {
		t.Fatal(err)
	}
	holder, err := os.OpenFile(p+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer holder.Close()
	if err := flock.TryLock(holder); err != nil {
		t.Fatalf("hold ledger lock: %v", err)
	}
	if err := l.RecordQuarantine(quarantineTestDigest(2)); err == nil {
		t.Fatal("expected a lock failure while a peer holds the lock")
	}
	if err := flock.Unlock(holder); err != nil {
		t.Fatal(err)
	}
	if err := l.RecordQuarantine(quarantineTestDigest(3)); err != nil {
		t.Errorf("write refused after the lock was released: %v", err)
	}
	if l.IsQuarantined(quarantineTestDigest(99)) {
		t.Errorf("unrelated digest reads quarantined after one transient lock failure")
	}
	if !l.IsQuarantined(quarantineTestDigest(1)) || !l.IsQuarantined(quarantineTestDigest(3)) {
		t.Errorf("committed denials lost across the lock failure")
	}
}

// Fresh denials must never push out a live quarantined digest: a full authority
// refuses the new denial instead of evicting the oldest live entry.
func TestQuarantineLedgerFloodCannotEvictLiveDenial(t *testing.T) {
	p := filepath.Join(t.TempDir(), "quarantine.jsonl")
	l := NewQuarantineLedger(p)
	base := time.Now()
	tick := 0
	l.nowFn = func() time.Time { tick++; return base.Add(time.Duration(tick) * time.Second) }
	l.maxEntries, l.softCap = 8, 6
	live := quarantineTestDigest(1000)
	if err := l.RecordQuarantine(live); err != nil {
		t.Fatal(err)
	}
	refused := 0
	for i := 0; i < 40; i++ {
		if err := l.RecordQuarantine(quarantineTestDigest(i)); errors.Is(err, ErrQuarantineLedgerCapacity) {
			refused++
		}
	}
	if refused == 0 {
		t.Fatal("flood never reached the capacity refusal; test does not exercise the cap")
	}
	if !l.IsQuarantined(live) {
		t.Errorf("live quarantined digest evicted by a flood of fresh digests (fail-open)")
	}
}

// A crash can tear only the final appended record. The committed denials before
// it must survive, the torn bytes must not brick reads, and the next append must
// land on a clean line.
func TestQuarantineLedgerTornTailRecovers(t *testing.T) {
	p := filepath.Join(t.TempDir(), "quarantine.jsonl")
	good, err := marshalQuarantineRecordAt("quarantine", quarantineTestDigest(1), time.Now().UnixNano())
	if err != nil {
		t.Fatal(err)
	}
	torn := []byte(`{"op":"quarantine","digest":"` + quarantineTestDigest(2)[:20])
	if err := os.WriteFile(p, append(good, torn...), 0o600); err != nil {
		t.Fatal(err)
	}
	l := NewQuarantineLedger(p)
	if !l.IsQuarantined(quarantineTestDigest(1)) {
		t.Errorf("committed denial before the torn tail was lost")
	}
	if l.IsQuarantined(quarantineTestDigest(99)) {
		t.Errorf("torn final line bricks the authority: unrelated digest reads quarantined")
	}
	if err := l.RecordQuarantine(quarantineTestDigest(3)); err != nil {
		t.Fatalf("record refused after torn final line: %v", err)
	}
	reloaded := NewQuarantineLedger(p)
	if reloaded.loadErr != nil {
		t.Fatalf("WAL left unparseable after torn-tail repair: %v", reloaded.loadErr)
	}
	if !reloaded.IsQuarantined(quarantineTestDigest(3)) || !reloaded.IsQuarantined(quarantineTestDigest(1)) {
		t.Errorf("reload after torn-tail repair lost a denial")
	}
}

// Mid-file damage is not a crash artifact and must keep failing closed.
func TestQuarantineLedgerMidFileCorruptionFailsClosed(t *testing.T) {
	p := filepath.Join(t.TempDir(), "quarantine.jsonl")
	good, err := marshalQuarantineRecordAt("quarantine", quarantineTestDigest(1), time.Now().UnixNano())
	if err != nil {
		t.Fatal(err)
	}
	data := append(append([]byte("{corrupt\n"), good...), good...)
	if err := os.WriteFile(p, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if !NewQuarantineLedger(p).IsQuarantined(quarantineTestDigest(99)) {
		t.Fatal("mid-file corruption must fail closed")
	}
}
