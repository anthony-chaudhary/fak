package agent

import (
	"context"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// Keep the production CPU-backed forward and record every generic KV owner,
// including snapshots. No backend name or prefix qualification is substituted.
type v41PrefixPositionBackend struct {
	*v41CompleteBackend
	live map[*v41PrefixPositionKV]bool
}

type v41PrefixPositionKV struct {
	compute.KVStore
	owner *v41PrefixPositionBackend
}

func (b *v41PrefixPositionBackend) track(kv compute.KVStore) compute.KVStore {
	k := &v41PrefixPositionKV{KVStore: kv, owner: b}
	b.live[k] = true
	return k
}

func (b *v41PrefixPositionBackend) NewKV(cfg compute.KVConfig) compute.KVStore {
	return b.track(b.Backend.NewKV(cfg))
}

func (k *v41PrefixPositionKV) Clone() compute.KVStore {
	return k.owner.track(k.KVStore.Clone())
}

func (k *v41PrefixPositionKV) Free() {
	if !k.owner.live[k] {
		panic("V4.1 prefix position witness: duplicate KV retirement")
	}
	delete(k.owner.live, k)
	k.KVStore.Free()
}

// The real generated-continuation consumer requires snapshot.Tokens to match
// the committed prompt plus forwarded tokens. V4.1's generic HAL KV stays
// empty; its complete continuation lives in the snapshot's V4.1 owner instead.
// This is a software ownership/admission regression, not Halo qualification.
// Refs #12640.
// fak-test:runtime medium est=2s lane=default
func TestV41GeneratedPrefixPosition(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "0")
	ref := compute.Default()
	if ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("position witness requires the qualified CPU reference backend")
	}
	b := &v41PrefixPositionBackend{
		v41CompleteBackend: &v41CompleteBackend{Backend: ref, available: true},
		live:               map[*v41PrefixPositionKV]bool{},
	}
	m := v41CompleteModel(t)
	defer m.CloseWeights()
	if !inKernelPlannerPrefixReuseSupported(m, b) {
		t.Fatal("existing qualified CPU route was refused")
	}
	s, err := m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	p := &InKernelPlanner{m: m, backend: b, tree: radixkv.New(0)}
	prompt, forwarded := []int{1, 3}, []int{5}
	tokens := append(append([]int(nil), prompt...), forwarded...)
	defer p.tree.EvictPrefix(tokens)
	s.Prefill(prompt)
	logits := s.Step(forwarded[0])
	ctx := context.Background()

	// A wrong expected length still rejects the snapshot and frees its KV
	// clone. Correcting the authority must not remove the consumer's check.
	p.admitGeneratedContinuation(ctx, s, prompt, []int{5, 6}, logits)
	if len(b.live) != 1 {
		t.Fatalf("rejected capture retained %d KV owners, want source only", len(b.live))
	}
	p.admitGeneratedContinuation(ctx, s, prompt, forwarded, logits)
	node, snap, matched, tier, err := p.tree.LookupSnapshotTiered(tokens)
	p.tree.Done(node)
	if snap != nil {
		defer snap.Close()
	}
	if err != nil || snap == nil || matched != len(tokens) || tier != radixkv.SnapshotTierDeviceL1 {
		t.Fatalf("generated continuation missing: matched=%d want=%d tier=%s snapshot=%t err=%v", matched, len(tokens), tier, snap != nil, err)
	}
	if snap.Tokens != len(tokens) || len(b.live) != 3 {
		t.Fatalf("snapshot tokens=%d live owners=%d, want %d tokens and source/cache/fork", snap.Tokens, len(b.live), len(tokens))
	}
	target, err := m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	defer target.Close()
	want := append([]float32(nil), s.Step(7)...)
	s.Close()
	if err := snap.Restore(target); err != nil {
		t.Fatal(err)
	}
	snap.Close()
	got := target.Step(7)
	if len(got) == 0 || len(got) != len(want) {
		t.Fatal("restored continuation changed logit shape")
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("restored logit %d differs from unsnapshotted continuation", i)
		}
	}
	for kv := range b.live {
		if kv.Len() != 0 {
			t.Fatal("position bridge populated the unrelated generic HAL KV")
		}
	}
	target.Close()
	if removed := p.tree.EvictPrefix(tokens); removed != len(tokens) || len(b.live) != 0 {
		t.Fatalf("retirement removed=%d live KV=%d, want %d and zero", removed, len(b.live), len(tokens))
	}
	if (m.Cfg.InKernelBackendPrefixReuseSupportedFor(&v41GateBackend{Backend: ref, name: "rocm"})) {
		t.Fatal("position metadata qualified an unwitnessed ROCm backend")
	}
}
