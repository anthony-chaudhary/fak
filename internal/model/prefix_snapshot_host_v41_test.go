package model

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=10ms lane=default
// Direct API coverage: current automatic device-pressure admission separately
// refuses unqualified V4.1 backends. No physical GPU execution is implied here.
func TestHostPrefixSnapshotRefusesV41BeforeTransfer(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		mark func(*PrefixSnapshot)
	}{
		{"model without markers", func(p *PrefixSnapshot) { p.Cache.cfg.ModelType = "deepseek_v41" }},
		{"continuation", func(p *PrefixSnapshot) { p.v41 = &v41ForwardSnapshot{hadState: true, history: []int{1}} }},
		{"absent continuation marker", func(p *PrefixSnapshot) { p.v41 = &v41ForwardSnapshot{} }},
		{"count without flag", func(p *PrefixSnapshot) { p.v41Tokens = 1 }},
		{"count flag without count", func(p *PrefixSnapshot) { p.hasV41Tokens = true }},
		{"identity without flag", func(p *PrefixSnapshot) { p.v41DeviceIdentity = p.Backend }},
		{"identity flag without identity", func(p *PrefixSnapshot) { p.hasV41DeviceIdentity = true }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			snap, be, kv := hostV41GuardSnapshot(t)
			defer snap.Close()
			tc.mark(snap)
			cache, continuation, identity := snap.Cache, snap.v41, snap.v41DeviceIdentity
			count, hasCount, hasIdentity := snap.v41Tokens, snap.hasV41Tokens, snap.hasV41DeviceIdentity
			want := snap.v41.clone()
			host, err := snap.CloneToHost()
			if !errors.Is(err, ErrHostPrefixSnapshotState) || host != nil {
				t.Fatalf("stage returned host=%v err=%v, want explicit refusal", host, err)
			}
			if kv.reads != 0 || be.restores != 0 {
				t.Fatal("refused state reached a transfer or allocation seam")
			}
			if snap.Cache != cache || snap.halKV != kv || snap.v41 != continuation ||
				snap.v41DeviceIdentity != identity || snap.v41Tokens != count ||
				snap.hasV41Tokens != hasCount || snap.hasV41DeviceIdentity != hasIdentity ||
				!reflect.DeepEqual(snap.v41, want) {
				t.Fatal("refusal consumed or changed the complete live snapshot")
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestHostPrefixSnapshotRestoreRefusesV41BeforeAllocation(t *testing.T) {
	t.Parallel()
	snap, be, kv := hostV41GuardSnapshot(t)
	defer snap.Close()
	host, err := snap.CloneToHost()
	if err != nil || host == nil || kv.reads != 1 {
		t.Fatalf("ordinary model staging failed: %v", err)
	}
	defer host.Close()
	// A decoded or directly constructed image may declare V4.1 even though its
	// type cannot hold the required continuation. Restore must check independently.
	host.cache.cfg.ModelType = "deepseek_v41"
	cache, backend := host.cache, host.backend
	wantKV := host.kv.Clone()
	live, err := host.Restore()
	if !errors.Is(err, ErrHostPrefixSnapshotState) || live != nil {
		t.Fatalf("restore returned live=%v err=%v, want explicit refusal", live, err)
	}
	if be.restores != 0 || host.cache != cache || host.backend != backend || !reflect.DeepEqual(host.kv, wantKV) {
		t.Fatal("refusal allocated a KV owner or consumed the host image")
	}
	// Keep the existing ordinary-model route and payload ownership intact.
	host.cache.cfg.ModelType = "llama"
	live, err = host.Restore()
	if err != nil || live == nil || be.restores != 1 {
		t.Fatalf("ordinary model restore failed: %v", err)
	}
	defer live.Close()
	got, err := compute.SnapshotKVToHost(live.halKV)
	if err != nil || !reflect.DeepEqual(got, wantKV) {
		t.Fatalf("ordinary model payload changed: %v", err)
	}
}

type hostV41GuardBackend struct {
	compute.Backend
	restores int
}

func (b *hostV41GuardBackend) RestoreKVFromHost(state compute.KVHostSnapshot) (compute.KVStore, error) {
	b.restores++
	return compute.RestoreKVFromHost(b.Backend, state)
}

type hostV41GuardKV struct {
	compute.KVStore
	reads int
}

func (k *hostV41GuardKV) SnapshotToHost() (compute.KVHostSnapshot, error) {
	k.reads++
	return compute.SnapshotKVToHost(k.KVStore)
}

func hostV41GuardSnapshot(t *testing.T) (*PrefixSnapshot, *hostV41GuardBackend, *hostV41GuardKV) {
	t.Helper()
	be := &hostV41GuardBackend{Backend: compute.Pick("cpu-ref")}
	state := compute.KVHostSnapshot{
		Config: compute.KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 1},
		Pos:    []int{0}, K: [][]float32{{1}}, KRaw: [][]float32{{2}}, V: [][]float32{{3}},
	}
	base, err := compute.RestoreKVFromHost(be.Backend, state)
	if err != nil {
		t.Fatal(err)
	}
	kv := &hostV41GuardKV{KVStore: base}
	cache := NewKVCache(Config{ModelType: "llama", NumLayers: 1, NumKVHeads: 1, HeadDim: 1})
	return &PrefixSnapshot{Cache: cache, Backend: be, halKV: kv, Tokens: 1}, be, kv
}
