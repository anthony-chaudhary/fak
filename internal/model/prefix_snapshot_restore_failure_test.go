package model

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type prefixRestoreFailureBackend struct {
	*recordingQwen35Backend
	fail    bool
	uploads []compute.Tensor
	frees   map[compute.Buffer]int
	kv      *prefixRestoreFailureKV
}

type prefixRestoreFailureKV struct {
	compute.KVStore
	fail  bool
	frees int
}

func (kv *prefixRestoreFailureKV) SnapshotToHost() (compute.KVHostSnapshot, error) {
	return compute.SnapshotKVToHost(kv.KVStore)
}

func (kv *prefixRestoreFailureKV) Free() {
	kv.frees++
	kv.KVStore.Free()
	if kv.fail {
		panic("secondary KV cleanup failure")
	}
}

func (b *prefixRestoreFailureBackend) RestoreKVFromHost(state compute.KVHostSnapshot) (compute.KVStore, error) {
	kv, err := compute.RestoreKVFromHost(b.recordingQwen35Backend.Backend, state)
	if err != nil {
		return nil, err
	}
	b.kv = &prefixRestoreFailureKV{KVStore: kv, fail: b.fail}
	return b.kv, nil
}

func (b *prefixRestoreFailureBackend) UploadClass(tensor compute.Tensor, as compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if b.fail && len(b.uploads) == 2 {
		panic("primary recurrent upload failure")
	}
	out := b.recordingQwen35Backend.UploadClass(tensor, as, class, site)
	b.uploads = append(b.uploads, out)
	return out
}

func (b *prefixRestoreFailureBackend) Free(tensor compute.Tensor) {
	b.frees[tensor.Buf()]++
	b.recordingQwen35Backend.Free(tensor)
	if b.fail && len(b.uploads) != 0 && tensor.Buf() == b.uploads[0].Buf() {
		panic("secondary tensor cleanup failure")
	}
}

// fak-test:runtime fast est=100ms
func TestHostPrefixRestoreUploadFailureRollsBackEveryNewOwner(t *testing.T) {
	m := NewSynthetic(qwen35HybridTestCfg())
	base := newRecordingQwen35Backend(m)
	s := m.NewBackendSession(base)
	defer s.Close()
	s.Prefill([]int{3, 7, 11})
	snapshot, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	host, err := snapshot.CloneToHost()
	if err != nil {
		t.Fatal(err)
	}
	defer host.Close()
	before, err := host.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	be := &prefixRestoreFailureBackend{
		recordingQwen35Backend: base,
		fail:                   true,
		frees:                  make(map[compute.Buffer]int),
	}
	host.backend, host.qwen35.backend = be, be
	out, err := host.Restore()
	if out != nil || err == nil || !strings.Contains(err.Error(), "primary recurrent upload failure") {
		t.Fatalf("failed restore published out=%v err=%v", out, err)
	}
	if be.kv == nil || be.kv.frees != 1 || len(be.uploads) != 2 {
		t.Fatalf("rollback KV=%+v uploads=%d", be.kv, len(be.uploads))
	}
	for _, tensor := range be.uploads {
		if got := be.frees[tensor.Buf()]; got != 1 {
			t.Fatalf("new tensor released %d times, want once", got)
		}
	}
	after, err := host.MarshalBinary()
	if err != nil || !bytes.Equal(before, after) {
		t.Fatalf("failed restore changed reusable host prefix: %v", err)
	}
	// The same image must still restore completely on a later attempt.
	be.fail = false
	retry, err := host.Restore()
	if err != nil || retry == nil {
		t.Fatalf("retry failed: %v", err)
	}
	defer retry.Close()
	roundTrip, err := retry.CloneToHost()
	if err != nil {
		t.Fatal(err)
	}
	defer roundTrip.Close()
	wire, err := roundTrip.MarshalBinary()
	if err != nil || !bytes.Equal(before, wire) {
		t.Fatalf("retry did not restore complete prefix: %v", err)
	}
}
