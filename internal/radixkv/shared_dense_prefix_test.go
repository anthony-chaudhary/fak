package radixkv

import (
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func denseSharingModel() *model.Model {
	return model.NewSynthetic(model.Config{HiddenSize: 32, NumLayers: 4, NumHeads: 4, NumKVHeads: 2, HeadDim: 8, IntermediateSize: 64, VocabSize: 512, RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: -1})
}

func denseSharingFill(t *testing.T, tree *Tree, ids []int, kv *model.KVCache, logits []float32) *node {
	t.Helper()
	b, matched := tree.Lookup(ids)
	n := tree.InsertCloneWithLogits(b, ids[matched:], kv, logits)
	if n == nil || !n.HasKV() {
		t.Fatal("owned fill did not retain payload")
	}
	tree.Done(n)
	return n
}

// Estimate is conservative scoping, not measured timing: includes nine extra oracle prefills.
// fak-test:runtime medium est=15s lane=default
func TestSharedDensePrefixRetainedBytesAndReuse(t *testing.T) {
	m := denseSharingModel()
	legacy, shared := New(0), New(0)
	if !shared.EnableSharedDensePrefixes() {
		t.Fatal("empty store refused selector")
	}
	prefix := make([]int, 128)
	for i := range prefix {
		prefix[i] = i + 1
	}
	var last []int
	for branch := 0; branch < 8; branch++ {
		ids := append([]int(nil), prefix...)
		for i := 0; i < 16; i++ {
			ids = append(ids, 200+branch*16+i)
		}
		s := m.NewSession()
		logits := append([]float32(nil), s.Prefill(ids)...)
		denseSharingFill(t, legacy, ids, s.Cache, logits)
		n := denseSharingFill(t, shared, ids, s.Cache, logits)
		if n.densePrefix == nil {
			t.Fatal("eligible clone-owning route remained legacy")
		}
		s.Close()
		last = ids
	}
	// Verify every retained leaf and the structural shared prefix before comparing
	// tree-owned bytes. These temporary oracle sessions are not counted as tree residency.
	for branch := 0; branch < 8; branch++ {
		ids := append([]int(nil), prefix...)
		for i := 0; i < 16; i++ {
			ids = append(ids, 200+branch*16+i)
		}
		oracle := m.NewSession()
		wantLogits := append([]float32(nil), oracle.Prefill(ids)...)
		for _, tree := range []*Tree{legacy, shared} {
			n, matched := tree.Lookup(ids)
			if matched != len(ids) || !n.HasKV() {
				t.Fatalf("branch %d lost exact payload coverage: matched=%d", branch, matched)
			}
			assertSharedDenseCoverageCache(t, oracle.Cache, n.CloneKV())
			assertSharedDenseCoverageFloats(t, wantLogits, n.Logits())
			tree.Done(n)
		}
		oracle.Close()
	}
	prefixOracle := m.NewSession()
	prefixOracle.Prefill(prefix)
	for _, tree := range []*Tree{legacy, shared} {
		n, matched := tree.Lookup(prefix)
		if matched != len(prefix) || !n.HasKV() {
			t.Fatalf("shared intermediate lost payload coverage: matched=%d", matched)
		}
		assertSharedDenseCoverageCache(t, prefixOracle.Cache, n.CloneKV())
		// A split intermediate has no cached final-token logits: leaf logits
		// must not be falsely attributed to the shortened prefix.
		if n.Logits() != nil {
			t.Fatal("split intermediate inherited invalid descendant logits")
		}
		tree.Done(n)
	}
	prefixOracle.Close()
	oldBytes, newBytes := legacy.Stats().CPUCacheBytes, shared.Stats().CPUCacheBytes
	if oldBytes <= 0 || newBytes <= 0 || newBytes > oldBytes/2 {
		t.Fatalf("fixed retained-numeric-byte target missed: legacy=%d shared=%d; target>=50%% reduction", oldBytes, newBytes)
	}
	if legacy.Stats().PrefixTokens != shared.Stats().PrefixTokens {
		t.Fatal("logical prefix coverage changed")
	}
	// This is a private session continuation, not a zero-copy Session claim.
	a, ma := legacy.Lookup(last)
	b, mb := shared.Lookup(last)
	if ma != mb {
		t.Fatal("prefix matches changed")
	}
	sa, sb := m.NewSession(), m.NewSession()
	sa.Cache = a.CloneKV()
	sb.Cache = b.CloneKV()
	legacy.Done(a)
	shared.Done(b)
	la, lb := sa.Step(9), sb.Step(9)
	for i := range la {
		if math.Float32bits(la[i]) != math.Float32bits(lb[i]) {
			t.Fatal("private suffix logits differ")
		}
	}
	sa.Close()
	sb.Close()
	t.Logf("fixed geometry layers=4 tokens=128+16 branches=8 page=16 stride=16 planes=3: legacy_retained_bytes=%d shared_retained_bytes=%d; not RSS/latency", oldBytes, newBytes)
	for _, tree := range []*Tree{legacy, shared} {
		for _, n := range tree.root.children {
			tree.EvictNode(n)
		}
		if tree.Stats().CPUCacheBytes != 0 {
			t.Fatal("final eviction retained CPU owners")
		}
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestSharedDensePrefixBudgetScopesAndLegacy(t *testing.T) {
	m := newSyntheticTiny()
	s := m.NewSession()
	defer s.Close()
	ids := []int{1, 2, 3, 4, 5}
	logits := append([]float32(nil), s.Prefill(ids)...)
	plan, ok := model.PlanDensePrefixCapture(s.Cache, nil)
	if !ok {
		t.Fatal("fixture is not eligible dense FP32")
	}
	tree := New(0)
	tree.EnableSharedDensePrefixes()
	tree.SetCPUCacheByteBudget(plan.AdditionalBytes() + cpuLogitsBytes(len(logits)))
	root, _ := tree.Lookup(ids)
	held := tree.InsertCloneWithLogits(root, ids, s.Cache, logits)
	before := tree.Stats().CPUCacheBytes
	other, _ := tree.Lookup([]int{9, 8, 7, 6, 5})
	refused := tree.InsertCloneWithLogits(other, []int{9, 8, 7, 6, 5}, s.Cache, logits)
	if refused != other || !held.HasKV() || tree.Stats().CPUCacheBytes != before {
		t.Fatal("pinned capacity refusal mutated retained owner")
	}
	tree.Done(refused)
	tree.Done(held)
	n := denseSharingFill(t, tree, []int{9, 8, 7, 6, 5}, s.Cache, logits)
	if held.HasKV() || n.densePrefix == nil {
		t.Fatal("unleased victim was not reclaimed")
	}
	// Commit/abort replacement must release the old immutable owner exactly once.
	tree.CommitFlight(n, s.Cache, logits)
	if n.densePrefix != nil || n.KV() != s.Cache {
		t.Fatal("legacy flight replacement changed live pointer")
	}
	tree.AbortFlight(n, ErrFlightAbandoned)
	if tree.Stats().CPUCacheBytes != 0 {
		t.Fatal("aborted replacement remained reachable")
	}

	privateSource := s.Cache.Clone()
	legacy := New(0)
	legacy.EnableSharedDensePrefixes()
	b, _ := legacy.Lookup(ids)
	live := legacy.InsertWithLogits(b, ids, s.Cache, logits)
	if live.KV() != s.Cache {
		t.Fatal("live-pointer insertion contract changed")
	}
	s.Step(6)
	if live.KV().Len() != s.Cache.Len() {
		t.Fatal("live-pointer growth stopped")
	}
	legacy.Done(live)

	scopedTree := New(0)
	scopedTree.EnableSharedDensePrefixes()
	scoped := WrapScoped(scopedTree)
	alice, bob := CacheIdentity{Tenant: "a", Agent: "one"}, CacheIdentity{Tenant: "b", Agent: "two"}
	if err := scoped.AdmitPrivate(alice, ids, privateSource, logits); err != nil {
		t.Fatal(err)
	}
	if got, _, matched, _, err := scoped.Lookup(bob, ids); err != nil || got != nil || matched != 0 {
		t.Fatal("private namespace leaked")
	}
	boundary, _ := scopedTree.LookupNS("private/tenant/a", ids)
	if boundary.densePrefix == nil {
		t.Fatal("scoped clone admission did not select immutable store")
	}
	scopedTree.Done(boundary)
	got, _, matched, _, err := scoped.Lookup(alice, ids)
	if err != nil || matched != len(ids) || got == nil {
		t.Fatal("private owner lost its cache")
	}
	got.K[0][0] = 12345
	again, _, _, _, _ := scoped.Lookup(alice, ids)
	if again.K[0][0] == 12345 {
		t.Fatal("scoped materialization aliases cached bytes")
	}
	if err := scoped.Promote(ScopeTenant, alice, ids); err != nil {
		t.Fatal(err)
	}
	if got, _, matched, _, err := scoped.Lookup(bob, ids); err != nil || got == nil || matched != len(ids) {
		t.Fatal("explicit promotion failed")
	}
	scoped.RevokeFleet(ids)
	if got, _, matched, _, _ := scoped.Lookup(bob, ids); got != nil || matched != 0 {
		t.Fatal("revoked fleet payload remains visible")
	}
	if !reflect.DeepEqual(ids, []int{1, 2, 3, 4, 5}) {
		t.Fatal("caller token slice mutated")
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestSharedDensePrefixStaleBoundaryPreservesLeases(t *testing.T) {
	for _, descendant := range []bool{false, true} {
		name := "direct"
		if descendant {
			name = "descendant"
		}
		t.Run(name, func(t *testing.T) {
			m := newSyntheticTiny()
			s := m.NewSession()
			defer s.Close()
			ids := []int{1, 2, 3, 4}
			logits := append([]float32(nil), s.Prefill(ids)...)
			tree := New(0)
			tree.EnableSharedDensePrefixes()
			first, _ := tree.Lookup(ids)
			stale, _ := tree.Lookup(ids)
			held := tree.InsertCloneWithLogits(first, ids, s.Cache, logits)
			if held == nil || held.densePrefix == nil {
				t.Fatal("missing immutable owner")
			}
			if descendant {
				// No extra prefix metadata fits, so the replaced edge root is
				// metadata-only and the leased dense owner is its descendant.
				tree.SetCPUCacheByteBudget(tree.Stats().CPUCacheBytes)
				mid, matched := tree.Lookup(ids[:2])
				if matched != 2 || mid == held || mid.densePrefix != nil {
					t.Fatal("expected metadata-only intermediate")
				}
				tree.Done(mid)
			}
			edge := tree.root.children[ids[0]]
			before := tree.Stats().CPUCacheBytes
			rootRefs, heldRefs := stale.refs, held.refs
			want := held.CloneKV()
			refused := tree.InsertWithLogits(stale, ids, s.Cache, logits)
			if refused != stale || tree.root.children[ids[0]] != edge || stale.refs != rootRefs || held.refs != heldRefs {
				t.Fatal("stale insertion detached an owner or transferred a lease")
			}
			got := held.CloneKV()
			if !held.HasKV() || got == nil || !reflect.DeepEqual(got, want) || tree.Stats().CPUCacheBytes != before {
				t.Fatal("refused insertion changed leased payload or retained-byte accounting")
			}
			tree.Done(refused)
			tree.Done(held)
			tree.EvictNode(edge)
			if held.HasKV() || tree.Stats().CPUCacheBytes != 0 {
				t.Fatal("final release retained an immutable owner")
			}
		})
	}
}

// assertSharedDenseCoverageCache compares semantic metadata through clones (which
// reset transient mutation generations) and checks every numeric bit explicitly.
func assertSharedDenseCoverageCache(t *testing.T, want, got *model.KVCache) {
	t.Helper()
	if want == nil || got == nil || !reflect.DeepEqual(want.Clone(), got.Clone()) {
		t.Fatal("retained configuration/positions/lineage/payload differ from fresh oracle")
	}
	for plane, rows := range [][][]float32{want.K, want.Kraw, want.V} {
		other := [][][]float32{got.K, got.Kraw, got.V}[plane]
		if len(rows) != len(other) {
			t.Fatal("retained layer coverage differs")
		}
		for l := range rows {
			assertSharedDenseCoverageFloats(t, rows[l], other[l])
		}
	}
}

func assertSharedDenseCoverageFloats(t *testing.T, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("numeric payload length=%d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			t.Fatalf("numeric payload bit mismatch at %d", i)
		}
	}
}

// This is a direct teardown unit, not evidence of an ordinary in-flight abort route.
// fak-test:runtime fast est=500ms lane=default
func TestSharedDensePrefixDirectImmutableAbort(t *testing.T) {
	m := newSyntheticTiny()
	s := m.NewSession()
	defer s.Close()
	ids := []int{1, 2, 3, 4, 5}
	logits := append([]float32(nil), s.Prefill(ids)...)
	tree := New(0)
	if !tree.EnableSharedDensePrefixes() {
		t.Fatal("empty store refused selector")
	}
	n := denseSharingFill(t, tree, ids, s.Cache, logits)
	owned := n.densePrefix
	if owned == nil || tree.Stats().CPUCacheBytes <= 0 {
		t.Fatal("direct abort fixture has no immutable owner")
	}
	tree.AbortFlight(n, ErrFlightAbandoned)
	if n.densePrefix != nil || owned.Materialize() != nil || tree.Stats().CPUCacheBytes != 0 {
		t.Fatal("direct immutable abort retained backing or reachable accounting")
	}
	tree.AbortFlight(n, ErrFlightAbandoned)
	if owned.Materialize() != nil || tree.Stats().CPUCacheBytes != 0 {
		t.Fatal("repeated direct abort changed released ownership")
	}
	boundary, matched := tree.Lookup(ids)
	defer tree.Done(boundary)
	if matched != 0 || boundary.HasKV() {
		t.Fatal("aborted immutable owner remains reusable")
	}
}
