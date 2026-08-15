package radixkv

import (
	"context"
	"encoding/binary"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/l3kv"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type remoteObjectServer struct {
	mu      sync.Mutex
	objects map[string][]byte
	puts    int
	gets    int
}

func (s *remoteObjectServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	key := r.URL.Path[1:]
	s.mu.Lock()
	defer s.mu.Unlock()
	switch r.Method {
	case http.MethodHead:
		if _, ok := s.objects[key]; ok {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusNotFound)
	case http.MethodPut:
		payload, _ := io.ReadAll(r.Body)
		s.objects[key] = append([]byte(nil), payload...)
		s.puts++
		time.Sleep(time.Millisecond)
		w.WriteHeader(http.StatusCreated)
	case http.MethodGet:
		payload, ok := s.objects[key]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		s.gets++
		time.Sleep(time.Millisecond)
		_, _ = w.Write(payload)
	default:
		w.WriteHeader(http.StatusMethodNotAllowed)
	}
}

func TestRemoteL3HTTPRestoresAfterAllLocalStateIsRemoved(t *testing.T) {
	objects := &remoteObjectServer{objects: map[string][]byte{}}
	server := httptest.NewServer(objects)
	defer server.Close()
	store, err := l3kv.NewRemoteStore(t.TempDir(), server.URL, "")
	if err != nil {
		t.Fatal(err)
	}

	cfg := remoteL3TestConfig()
	m := model.NewSynthetic(cfg)
	be := &deviceCapsBackend{Backend: compute.Default()}
	tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 1<<30, EvictionLRU)
	if err := tree.ConfigureRemoteSnapshotStore(store, "synthetic-l3-test", be, m.Cfg); err != nil {
		t.Fatal(err)
	}
	ids := []int{3, 7, 11, 13}
	digest := insertRemoteL3Snapshot(t, tree, m, be, ids)

	if got := tree.StageSnapshotToHost(digest); got.Outcome != SnapshotTransferOK {
		t.Fatalf("stage host: %+v", got)
	}
	staged := tree.StageSnapshotToRemote(context.Background(), digest)
	if staged.Outcome != SnapshotTransferOK || staged.BytesMoved <= 0 {
		t.Fatalf("stage remote: %+v", staged)
	}
	objects.mu.Lock()
	putCount, remoteObjects := objects.puts, len(objects.objects)
	objects.mu.Unlock()
	if putCount != 1 || remoteObjects != 1 {
		t.Fatalf("off-process writes puts=%d objects=%d, want one HTTP payload", putCount, remoteObjects)
	}
	if tree.EvictHotSnapshot(digest) != len(ids) {
		t.Fatal("device L1 owner was not removed")
	}
	if tree.EvictHostSnapshot(digest) != len(ids) {
		t.Fatal("host L2 owner was not removed")
	}
	if stats := tree.Stats(); stats.DeviceSnapshotBytes != 0 || stats.HostSnapshotBytes != 0 {
		t.Fatalf("local state remained before remote restore: %+v", stats)
	}

	n, recovered, matched, tier, err := tree.LookupSnapshotTieredContext(context.Background(), ids)
	if err != nil {
		t.Fatal(err)
	}
	defer tree.Done(n)
	if recovered == nil || matched != len(ids) || tier != SnapshotTierRemoteL3 {
		t.Fatalf("remote lookup snapshot=%v matched=%d tier=%q", recovered, matched, tier)
	}
	restored, err := m.NewBackendSessionChecked(be)
	if err != nil {
		recovered.Close()
		t.Fatal(err)
	}
	defer restored.Close()
	if err := recovered.Restore(restored); err != nil {
		recovered.Close()
		t.Fatal(err)
	}
	recovered.Close()
	fresh, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer fresh.Close()
	fresh.Prefill(ids)
	const next = 17
	if diff := maxAbsDiff(restored.Step(next), fresh.Step(next)); diff != 0 {
		t.Fatalf("remote continuation drift max_abs_diff=%g", diff)
	}

	stats := tree.Stats()
	if stats.L3Hits != 1 || stats.L3HitTokens != len(ids) || stats.L3StageBytes <= 0 || stats.L3RestoreBytes <= 0 ||
		stats.L3StageNanos <= 0 || stats.L3RestoreNanos <= 0 || stats.L3ReferencedBytes <= 0 {
		t.Fatalf("remote tier telemetry = %+v", stats)
	}
	objects.mu.Lock()
	if objects.gets != 1 {
		objects.mu.Unlock()
		t.Fatalf("off-process GETs=%d, want 1", objects.gets)
	}
	for key, payload := range objects.objects {
		payload[len(payload)-1] ^= 0x80
		objects.objects[key] = payload
	}
	objects.mu.Unlock()

	_, bad, _, badTier, err := tree.LookupSnapshotTieredContext(context.Background(), ids)
	if err == nil || bad != nil || badTier != SnapshotTierRemoteL3 {
		t.Fatalf("tampered off-process bytes snapshot=%v tier=%q err=%v", bad, badTier, err)
	}
	if got := tree.Stats(); got.L3Faults != 1 || got.L3RestoreFaults != 1 {
		t.Fatalf("tampered HTTP payload telemetry = %+v", got)
	}
}

type memorySnapshotStore struct {
	data   map[string][]byte
	getErr error
}

func (s *memorySnapshotStore) Put(_ context.Context, key string, payload []byte) error {
	if s.data == nil {
		s.data = map[string][]byte{}
	}
	s.data[key] = append([]byte(nil), payload...)
	return nil
}

func (s *memorySnapshotStore) Get(_ context.Context, key string) ([]byte, bool, error) {
	if s.getErr != nil {
		return nil, false, s.getErr
	}
	payload, ok := s.data[key]
	return append([]byte(nil), payload...), ok, nil
}

func TestRemoteL3FailsClosedOnEnvelopeControls(t *testing.T) {
	unavailable := errors.New("remote unavailable")
	tests := []struct {
		name string
		want error
		edit func(*Tree, *memorySnapshotStore, string, int)
	}{
		{
			name: "corruption",
			want: ErrRemoteSnapshotIntegrity,
			edit: func(_ *Tree, store *memorySnapshotStore, digest string, _ int) {
				store.data[digest][len(store.data[digest])-1] ^= 0x40
			},
		},
		{
			name: "version",
			want: ErrRemoteSnapshotVersion,
			edit: func(_ *Tree, store *memorySnapshotStore, digest string, _ int) {
				binary.BigEndian.PutUint16(store.data[digest][len(remoteSnapshotMagic):], remoteSnapshotVersion+1)
			},
		},
		{
			name: "digest",
			want: ErrRemoteSnapshotDigest,
			edit: func(tree *Tree, store *memorySnapshotStore, digest string, tokens int) {
				payload, err := decodeRemoteSnapshotEnvelope(store.data[digest], digest, tree.snapshotScope(""), tokens)
				if err != nil {
					panic(err)
				}
				store.data[digest] = encodeRemoteSnapshotEnvelope("different-digest", tree.snapshotScope(""), tokens, payload)
			},
		},
		{
			name: "scope",
			want: ErrRemoteSnapshotScope,
			edit: func(tree *Tree, store *memorySnapshotStore, digest string, tokens int) {
				payload, err := decodeRemoteSnapshotEnvelope(store.data[digest], digest, tree.snapshotScope(""), tokens)
				if err != nil {
					panic(err)
				}
				store.data[digest] = encodeRemoteSnapshotEnvelope(digest, "another-model\x00another-tenant", tokens, payload)
			},
		},
		{
			name: "unavailable",
			want: unavailable,
			edit: func(_ *Tree, store *memorySnapshotStore, _ string, _ int) { store.getErr = unavailable },
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := remoteL3TestConfig()
			m := model.NewSynthetic(cfg)
			be := &deviceCapsBackend{Backend: compute.Default()}
			store := &memorySnapshotStore{}
			tree := NewWithTierBudgetsAndEvictionPolicy(0, 0, 0, EvictionLRU)
			if err := tree.ConfigureRemoteSnapshotStore(store, "synthetic-l3-test", be, m.Cfg); err != nil {
				t.Fatal(err)
			}
			ids := []int{3, 7, 11}
			digest := insertRemoteL3Snapshot(t, tree, m, be, ids)
			if got := tree.StageSnapshotToRemote(context.Background(), digest); got.Outcome != SnapshotTransferOK {
				t.Fatalf("stage remote: %+v", got)
			}
			if tree.EvictHotSnapshot(digest) != len(ids) {
				t.Fatal("hot owner was not removed")
			}
			tt.edit(tree, store, digest, len(ids))

			n, snap, matched, tier, err := tree.LookupSnapshotTieredContext(context.Background(), ids)
			tree.Done(n)
			if snap != nil {
				snap.Close()
				t.Fatal("faulted remote bytes installed a live snapshot")
			}
			if !errors.Is(err, tt.want) || matched != len(ids) || tier != SnapshotTierRemoteL3 {
				t.Fatalf("lookup matched=%d tier=%q err=%v, want %v", matched, tier, err, tt.want)
			}
			stats := tree.Stats()
			if stats.DeviceSnapshotBytes != 0 || stats.HostSnapshotBytes != 0 || stats.L3Hits != 0 ||
				stats.L3Faults != 1 || stats.L3RestoreFaults != 1 {
				t.Fatalf("fail-closed state = %+v", stats)
			}
		})
	}
}

func insertRemoteL3Snapshot(t *testing.T, tree *Tree, m *model.Model, be compute.Backend, ids []int) string {
	t.Helper()
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	s.Prefill(ids)
	snap, err := s.PrefixSnapshot()
	s.Close()
	if err != nil {
		t.Fatal(err)
	}
	root, _ := tree.Lookup(nil)
	leaf, err := tree.InsertSnapshot(root, ids, snap, nil)
	if err != nil {
		snap.Close()
		t.Fatal(err)
	}
	tree.Done(leaf)
	_, candidates := tree.PressuredSnapshotCandidates()
	if len(candidates) != 1 {
		t.Fatalf("pressure candidates=%+v, want one", candidates)
	}
	return candidates[0].Digest
}

func remoteL3TestConfig() model.Config {
	return model.Config{
		HiddenSize:       32,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 64,
		VocabSize:        64,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       63,
	}
}
