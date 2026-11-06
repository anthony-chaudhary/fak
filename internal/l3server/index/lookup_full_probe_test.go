package index

import (
	"bytes"
	"context"
	"os"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// fak-test:runtime integration est=6s lane=optin
// Conservative deadline estimate: isolate a regressed infinite probe in an
// owned child, so failure cannot strand a spinning goroutine in the test suite.
func TestSwissTableLookupFullProbe(t *testing.T) {
	if os.Getenv("FAK_SWISS_TABLE_PROBE_CHILD") == "1" {
		for _, capacity := range []uint64{16, 32, 64} {
			tbl, expected := fullProbeTable(t, capacity)
			for hash, value := range expected {
				entry, _, found := tbl.Lookup(hash, 1)
				if !found || entry.KeyHash != hash || entry.ValueOffset != value {
					t.Fatalf("capacity %d lost existing %x: %+v found=%t", capacity, hash, entry, found)
				}
			}
			missing := (capacity*capacity+capacity+7)<<7 | 1
			for _, key := range []struct {
				hash   uint64
				length uint16
			}{
				{missing, 1},
				{capacity<<7 | 1, 1}, // deleted key
				{(capacity*capacity+capacity-1)<<7 | 1, 2}, // existing hash, wrong length
			} {
				if _, _, found := tbl.Lookup(key.hash, key.length); found {
					t.Fatalf("capacity %d falsely found %x length %d", capacity, key.hash, key.length)
				}
			}
			if _, grew := tbl.Insert(missing, Entry{KeyLen: 1, ValueOffset: 999}); !grew {
				t.Fatal("full-control insertion did not reach existing growth path")
			}
			expected[missing] = 999
			if tbl.Capacity() != capacity*2 || tbl.Count() != uint64(len(expected)) || tbl.Tombstones() != 0 {
				t.Fatal("growth accounting changed")
			}
			for hash, value := range expected {
				entry, _, found := tbl.Lookup(hash, 1)
				if !found || entry.ValueOffset != value {
					t.Fatalf("growth lost %x: %+v found=%t", hash, entry, found)
				}
			}
		}
		return
	}
	t.Parallel()
	// This setup uses only terminating baseline operations. Verify the reachable
	// no-empty state before putting any potentially endless lookup in the child.
	for _, capacity := range []uint64{16, 32, 64} {
		fullProbeTable(t, capacity)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	cmd := windowgate.CommandContext(ctx, executable, "-test.run=^TestSwissTableLookupFullProbe$", "-test.count=1")
	cmd.Env = append(os.Environ(), "FAK_SWISS_TABLE_PROBE_CHILD=1")
	cmd.WaitDelay = time.Second
	var output bytes.Buffer
	cmd.Stdout, cmd.Stderr = &output, &output
	child, err := windowgate.StartOwnedProcess(t, cmd)
	if err != nil {
		t.Fatalf("start lookup probe: %v", err)
	}
	err = child.Wait()
	if ctx.Err() != nil {
		t.Fatalf("lookup probe exceeded its deadline: %v\n%s", ctx.Err(), output.String())
	}
	if err != nil {
		t.Fatalf("lookup probe failed: %v\n%s", err, output.String())
	}
}

func fullProbeTable(t *testing.T, capacity uint64) (*Table, map[uint64]uint64) {
	t.Helper()
	tbl := NewTable(capacity)
	expected := make(map[uint64]uint64)
	live := capacity * MaxLoadNumerator / MaxLoadDenominator
	for i := uint64(0); i < live; i++ {
		hash := ((i+1)*capacity)<<7 | 1 // collide at home zero
		tbl.Insert(hash, Entry{KeyLen: 1, ValueOffset: i + 1})
		expected[hash] = i + 1
	}
	for i := uint64(0); i < capacity-live; i++ {
		hash := ((i+1)*capacity)<<7 | 1
		if !tbl.Delete(hash, 1) {
			t.Fatal("fixture deletion failed")
		}
		delete(expected, hash)
	}
	for i := live; i < capacity; i++ {
		hash := (capacity*capacity+i)<<7 | 1 // fill empties before tombstones
		tbl.Insert(hash, Entry{KeyLen: 1, ValueOffset: i + 1})
		expected[hash] = i + 1
	}
	if tbl.Capacity() != capacity || tbl.Count() != live || tbl.Tombstones() != capacity-live {
		t.Fatal("fixture changed capacity or entry accounting")
	}
	for _, control := range tbl.ctrl[:capacity] {
		if control == CtrlEmpty {
			t.Fatal("fixture still has an empty control byte")
		}
	}
	return tbl, expected
}
