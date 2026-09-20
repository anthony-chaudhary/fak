package radixkv

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestContinuationSoftRetention is the #13385 witness: a scoped, owner-scoped,
// finite, soft-retention, generation-bound continuation handle for a privately
// admitted KV prefix. Every sub-test fights the SPEC adversarially rather than
// the implementation's internals — it asserts what the ticket promises, and
// probes the edges (foreign owner, absent admission, eviction, split,
// idempotent activation, release semantics).
func TestContinuationSoftRetention(t *testing.T) {
	cfg := model.Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 1}
	ownerA := CacheIdentity{Tenant: "tenant-a", Agent: "worker-1"}
	ownerB := CacheIdentity{Tenant: "tenant-b", Agent: "worker-2"}

	newKV := func() *model.KVCache {
		kv := model.NewKVCache(cfg)
		kv.K[0] = []float32{1, 2, 3, 4}
		kv.V[0] = []float32{5, 6, 7, 8}
		return kv
	}
	// nodeFor resolves the exact-token node under the owner's tenant namespace and
	// releases the Lookup lease so the node is not pinned by the test itself.
	nodeFor := func(t *testing.T, s *ScopedTree, owner CacheIdentity, tokens []int) *node {
		t.Helper()
		ns, err := scopeNamespace(ScopeTenant, owner)
		if err != nil {
			t.Fatalf("scopeNamespace: %v", err)
		}
		n, matched := s.tree.LookupNS(ns, tokens)
		if n == nil || matched != len(tokens) {
			if n != nil {
				s.tree.Done(n)
			}
			t.Fatalf("nodeFor: matched=%d want=%d", matched, len(tokens))
		}
		s.tree.Done(n)
		return n
	}

	t.Run("PositiveAdmissionYieldsDormantHandle", func(t *testing.T) {
		s := NewScoped(0)
		tokens := []int{1, 2, 3, 4}
		if err := s.AdmitPrivate(ownerA, tokens, newKV(), []float32{.1, .9}); err != nil {
			t.Fatalf("AdmitPrivate: %v", err)
		}
		h, err := s.BeginPrivateContinuation(ownerA, tokens)
		if err != nil {
			t.Fatalf("BeginPrivateContinuation: %v", err)
		}
		if h == nil {
			t.Fatal("nil handle on a privately admitted full prefix")
		}
		if !h.Dormant() || h.Active() {
			t.Fatalf("fresh handle dormant=%v active=%v want dormant", h.Dormant(), h.Active())
		}
		if h.Owner() != ownerA {
			t.Fatalf("Owner()=%+v want %+v", h.Owner(), ownerA)
		}
		if h.Generation() == 0 {
			t.Fatal("Generation()==0; a handle must be generation-bound")
		}
	})

	t.Run("UnadmittedPrefixCannotYieldHandle", func(t *testing.T) {
		s := NewScoped(0)
		tokens := []int{10, 20, 30}
		// Only a strict prefix is admitted — Begin requires the EXACT full path.
		if err := s.AdmitPrivate(ownerA, tokens[:2], newKV(), nil); err != nil {
			t.Fatalf("AdmitPrivate(prefix): %v", err)
		}
		h, err := s.BeginPrivateContinuation(ownerA, tokens)
		if !errors.Is(err, ErrContinuationNotAdmitted) {
			t.Fatalf("strict-prefix begin err=%v want ErrContinuationNotAdmitted", err)
		}
		if h != nil {
			t.Fatal("handle returned for an unadmitted (only prefixed) path")
		}
		// A wholly absent prefix is likewise refused.
		h, err = s.BeginPrivateContinuation(ownerA, []int{99, 98})
		if !errors.Is(err, ErrContinuationNotAdmitted) || h != nil {
			t.Fatalf("absent begin h=%v err=%v want nil/ErrContinuationNotAdmitted", h, err)
		}
	})

	t.Run("ForeignOwnerCannotBegin", func(t *testing.T) {
		s := NewScoped(0)
		tokens := []int{5, 6, 7}
		if err := s.AdmitPrivate(ownerA, tokens, newKV(), nil); err != nil {
			t.Fatalf("AdmitPrivate: %v", err)
		}
		h, err := s.BeginPrivateContinuation(ownerB, tokens)
		if !errors.Is(err, ErrContinuationNotAdmitted) {
			t.Fatalf("foreign-owner begin err=%v want ErrContinuationNotAdmitted", err)
		}
		if h != nil {
			t.Fatal("foreign owner obtained a handle for another owner's prefix")
		}
		// The mismatch path the API exposes for Activate is the tree-identity
		// guard: a handle minted by a different tree must be refused, never
		// silently applied.
		other := NewScoped(0)
		if err := other.AdmitPrivate(ownerA, tokens, newKV(), nil); err != nil {
			t.Fatalf("other AdmitPrivate: %v", err)
		}
		ah, err := other.BeginPrivateContinuation(ownerA, tokens)
		if err != nil {
			t.Fatalf("other BeginPrivateContinuation: %v", err)
		}
		if err := s.ActivateContinuation(ah); !errors.Is(err, ErrContinuationUnsupported) {
			t.Fatalf("cross-tree activate err=%v want ErrContinuationUnsupported", err)
		}
		if err := s.ReleaseContinuation(ah); !errors.Is(err, ErrContinuationUnsupported) {
			t.Fatalf("cross-tree release err=%v want ErrContinuationUnsupported", err)
		}
	})

	t.Run("ForeignOwnerNilKVIsNotAWitness", func(t *testing.T) {
		s := NewScoped(0)
		tokens := []int{41, 42, 43}
		// A pure-accounting admission (nil kv, nil snapshot) returns a nil error
		// but carries NO reusable resident payload. The structural node existing
		// must not be mistaken for an admission witness.
		if err := s.AdmitPrivate(ownerB, tokens, nil, nil); err != nil {
			t.Fatalf("AdmitPrivate(nil kv) must not error for pure accounting: %v", err)
		}
		h, err := s.BeginPrivateContinuation(ownerB, tokens)
		if !errors.Is(err, ErrContinuationUnsupported) {
			t.Fatalf("nil-kv begin err=%v want ErrContinuationUnsupported", err)
		}
		if h != nil {
			t.Fatal("handle minted from a node with no complete reusable payload")
		}
	})

	t.Run("SoftRetentionIsFiniteAndUnpinned", func(t *testing.T) {
		s := NewScoped(0)
		tokens := []int{1, 2, 3}
		if err := s.AdmitPrivate(ownerA, tokens, newKV(), nil); err != nil {
			t.Fatalf("AdmitPrivate: %v", err)
		}
		h, err := s.BeginPrivateContinuation(ownerA, tokens)
		if err != nil {
			t.Fatalf("BeginPrivateContinuation: %v", err)
		}
		if err := s.ActivateContinuation(h); err != nil {
			t.Fatalf("ActivateContinuation: %v", err)
		}
		if !h.Active() {
			t.Fatal("handle not Active() after Activate")
		}
		n := nodeFor(t, s, ownerA, tokens)
		if s.tree.IsNodePinned(n) {
			t.Fatal("activated continuation node is pinned; a soft continuation must never pin")
		}
		if tier := s.tree.NodeTier(n); tier != Tier2IdleSubagent {
			t.Fatalf("NodeTier=%v want Tier2IdleSubagent (priority %d < 60)", tier, ContinuationSoftPriority)
		}
		ret, ok := s.tree.NodeRetention(n)
		if !ok {
			t.Fatal("activated node carries no retention request")
		}
		if ret.Priority != ContinuationSoftPriority {
			t.Fatalf("retention priority=%d want %d", ret.Priority, ContinuationSoftPriority)
		}
		if ret.TTL <= 0 || ret.TTL == RetainForever {
			t.Fatalf("retention TTL=%d must be finite and > 0 (RetainForever=%d)", ret.TTL, RetainForever)
		}
	})

	t.Run("ReclaimableUnderPressure", func(t *testing.T) {
		s := NewScoped(8)
		tokens := []int{1, 2, 3}
		if err := s.AdmitPrivate(ownerA, tokens, newKV(), nil); err != nil {
			t.Fatalf("AdmitPrivate: %v", err)
		}
		h, err := s.BeginPrivateContinuation(ownerA, tokens)
		if err != nil {
			t.Fatalf("BeginPrivateContinuation: %v", err)
		}
		if err := s.ActivateContinuation(h); err != nil {
			t.Fatalf("ActivateContinuation: %v", err)
		}
		n := nodeFor(t, s, ownerA, tokens)
		if s.tree.IsNodePinned(n) {
			t.Fatal("activated node is pinned and would be immune under pressure")
		}
		// Drive real budget pressure with unrelated admissions; the soft node is
		// Tier2 and TTL-bounded, so it is a legitimate victim.
		for i := 0; i < 12; i++ {
			other := []int{100 + i, 200 + i, 300 + i}
			if err := s.AdmitPrivate(ownerA, other, newKV(), nil); err != nil {
				t.Fatalf("pressure AdmitPrivate %d: %v", i, err)
			}
		}
		_, _, matched, _, err := s.Lookup(ownerA, tokens)
		if err != nil {
			t.Fatalf("Lookup: %v", err)
		}
		stats := s.tree.Stats()
		if stats.Tokens > 8 {
			t.Fatalf("token count %d exceeds budget 8 after pressure", stats.Tokens)
		}
		// The activated prefix is either evicted (matched==0) or still resident;
		// either is acceptable, but it must NOT have been protected as a pin.
		if h.Active() && matched == len(tokens) {
			// still resident and within budget: soft, not pinned — already asserted.
			_ = h
		}
		if err := s.ReleaseContinuation(h); err != nil && !errors.Is(err, ErrContinuationNodeGone) && !errors.Is(err, ErrContinuationUnsupported) {
			t.Fatalf("release after pressure err=%v want nil or typed gone/unsupported", err)
		}
	})

	t.Run("RepeatedActivationNeverExtendsExpiry", func(t *testing.T) {
		s := NewScoped(0)
		tokens := []int{7, 8, 9}
		if err := s.AdmitPrivate(ownerA, tokens, newKV(), nil); err != nil {
			t.Fatalf("AdmitPrivate: %v", err)
		}
		h, err := s.BeginPrivateContinuation(ownerA, tokens)
		if err != nil {
			t.Fatalf("BeginPrivateContinuation: %v", err)
		}
		if err := s.ActivateContinuation(h); err != nil {
			t.Fatalf("first activate: %v", err)
		}
		n := nodeFor(t, s, ownerA, tokens)
		first, ok := s.tree.NodeRetention(n)
		if !ok {
			t.Fatal("no retention after first activation")
		}
		// Advance the logical clock with unrelated lookups/inserts.
		for i := 0; i < 5; i++ {
			_ = s.AdmitPrivate(ownerA, []int{500 + i, 600 + i}, newKV(), nil)
			if _, _, _, _, err := s.Lookup(ownerA, []int{500 + i}); err != nil {
				t.Fatalf("clock advance lookup: %v", err)
			}
		}
		if err := s.ActivateContinuation(h); err != nil {
			t.Fatalf("second activate: %v", err)
		}
		second, ok := s.tree.NodeRetention(n)
		if !ok {
			t.Fatal("no retention after second activation")
		}
		if second.Admitted != first.Admitted || second.TTL != first.TTL {
			t.Fatalf("re-activation refreshed policy: first(adm=%d ttl=%d) second(adm=%d ttl=%d)",
				first.Admitted, first.TTL, second.Admitted, second.TTL)
		}
	})

	t.Run("ReleaseAffectsOnlyItsOwnGeneration", func(t *testing.T) {
		s := NewScoped(0)
		aTok := []int{1, 1, 1}
		bTok := []int{2, 2, 2}
		if err := s.AdmitPrivate(ownerA, aTok, newKV(), nil); err != nil {
			t.Fatalf("AdmitPrivate(a): %v", err)
		}
		if err := s.AdmitPrivate(ownerA, bTok, newKV(), nil); err != nil {
			t.Fatalf("AdmitPrivate(b): %v", err)
		}
		ha, err := s.BeginPrivateContinuation(ownerA, aTok)
		if err != nil {
			t.Fatalf("Begin(a): %v", err)
		}
		hb, err := s.BeginPrivateContinuation(ownerA, bTok)
		if err != nil {
			t.Fatalf("Begin(b): %v", err)
		}
		if err := s.ActivateContinuation(ha); err != nil {
			t.Fatalf("Activate(a): %v", err)
		}
		if err := s.ActivateContinuation(hb); err != nil {
			t.Fatalf("Activate(b): %v", err)
		}
		na := nodeFor(t, s, ownerA, aTok)
		before, ok := s.tree.NodeRetention(na)
		if !ok {
			t.Fatal("no retention on a")
		}
		if err := s.ReleaseContinuation(hb); err != nil {
			t.Fatalf("Release(b): %v", err)
		}
		if ha.Active() != true {
			t.Fatal("releasing b deactivated a's handle")
		}
		after, ok := s.tree.NodeRetention(na)
		if !ok {
			t.Fatal("a lost its retention when b was released")
		}
		if after.Priority != before.Priority || after.Admitted != before.Admitted || after.TTL != before.TTL {
			t.Fatalf("releasing b mutated a's retention: before=%+v after=%+v", before, after)
		}
		if hb.Active() {
			t.Fatal("released handle still reports Active()")
		}
	})

	t.Run("ReleaseDoesNotEraseLaterPolicyDecision", func(t *testing.T) {
		s := NewScoped(0)
		tokens := []int{3, 3, 3}
		if err := s.AdmitPrivate(ownerA, tokens, newKV(), nil); err != nil {
			t.Fatalf("AdmitPrivate: %v", err)
		}
		h, err := s.BeginPrivateContinuation(ownerA, tokens)
		if err != nil {
			t.Fatalf("BeginPrivateContinuation: %v", err)
		}
		if err := s.ActivateContinuation(h); err != nil {
			t.Fatalf("Activate: %v", err)
		}
		n := nodeFor(t, s, ownerA, tokens)
		// A LATER, unrelated policy decision (priority 80 -> Tier1) replaces the
		// continuation's soft policy. Release must not clobber it.
		later := RetentionRequest{Priority: 80, TTL: 5, Admitted: int64(s.tree.clock)}
		if err := s.tree.SetNodeRetention(n, later); err != nil {
			t.Fatalf("SetNodeRetention(later): %v", err)
		}
		if err := s.ReleaseContinuation(h); err != nil {
			t.Fatalf("Release: %v", err)
		}
		got, ok := s.tree.NodeRetention(n)
		if !ok {
			t.Fatal("release erased the later policy decision entirely")
		}
		if got.Priority != 80 {
			t.Fatalf("release clobbered a later decision: priority=%d want 80", got.Priority)
		}
		if got.TTL != 5 {
			t.Fatalf("release clobbered later TTL: %d want 5", got.TTL)
		}
	})

	t.Run("EvictedNodeIsTypedStaleNoRecreation", func(t *testing.T) {
		s := NewScoped(0)
		tokens := []int{4, 4, 4}
		if err := s.AdmitPrivate(ownerA, tokens, newKV(), nil); err != nil {
			t.Fatalf("AdmitPrivate: %v", err)
		}
		h, err := s.BeginPrivateContinuation(ownerA, tokens)
		if err != nil {
			t.Fatalf("BeginPrivateContinuation: %v", err)
		}
		if err := s.ActivateContinuation(h); err != nil {
			t.Fatalf("Activate: %v", err)
		}
		n := nodeFor(t, s, ownerA, tokens)
		if freed := s.tree.EvictNode(n); freed == 0 {
			t.Fatal("EvictNode freed no tokens for an activated continuation node")
		}
		err = s.ReleaseContinuation(h)
		if !errors.Is(err, ErrContinuationNodeGone) && !errors.Is(err, ErrContinuationGeneration) {
			t.Fatalf("release after evict err=%v want typed NodeGone/Generation", err)
		}
		err = s.ActivateContinuation(h)
		if !errors.Is(err, ErrContinuationNodeGone) && !errors.Is(err, ErrContinuationGeneration) {
			t.Fatalf("re-activate after evict err=%v want typed NodeGone/Generation", err)
		}
		// No node may be recreated by a stale handle: the exact path must not
		// resolve to a full match.
		ns, _ := scopeNamespace(ScopeTenant, ownerA)
		rn, matched := s.tree.LookupNS(ns, tokens)
		if rn != nil {
			s.tree.Done(rn)
		}
		if matched == len(tokens) {
			t.Fatalf("stale handle recreated a full node match: matched=%d", matched)
		}
	})

	t.Run("SplitOrReplacedGenerationIsRefused", func(t *testing.T) {
		s := NewScoped(0)
		p := []int{6, 6}
		if err := s.AdmitPrivate(ownerA, p, newKV(), nil); err != nil {
			t.Fatalf("AdmitPrivate(p): %v", err)
		}
		h, err := s.BeginPrivateContinuation(ownerA, p)
		if err != nil {
			t.Fatalf("Begin(p): %v", err)
		}
		if err := s.ActivateContinuation(h); err != nil {
			t.Fatalf("Activate(p): %v", err)
		}
		// Admit a LONGER prefix beginning with P, forcing a split at P.
		longer := []int{6, 6, 7, 7}
		if err := s.AdmitPrivate(ownerA, longer, newKV(), nil); err != nil {
			t.Fatalf("AdmitPrivate(longer): %v", err)
		}
		// The original handle must fail typed rather than apply policy to the
		// wrong (replaced) node.
		actErr := s.ActivateContinuation(h)
		relErr := s.ReleaseContinuation(h)
		actTyped := errors.Is(actErr, ErrContinuationGeneration) || errors.Is(actErr, ErrContinuationNodeGone)
		relTyped := errors.Is(relErr, ErrContinuationGeneration) || errors.Is(relErr, ErrContinuationNodeGone)
		if !actTyped && !relTyped {
			t.Fatalf("split handle neither activate nor release returned a typed stale error: act=%v rel=%v", actErr, relErr)
		}
	})

	t.Run("SnapshotPayloadArm", func(t *testing.T) {
		s := NewScoped(0)
		tokens := []int{9, 9, 9}
		snap := &model.PrefixSnapshot{Cache: model.NewKVCache(cfg)}
		if err := s.AdmitPrivateSnapshot(ownerA, tokens, snap, []float32{3}); err != nil {
			t.Fatalf("AdmitPrivateSnapshot: %v", err)
		}
		h, err := s.BeginPrivateContinuation(ownerA, tokens)
		if err != nil {
			t.Fatalf("BeginPrivateContinuation(snapshot): %v", err)
		}
		if h == nil {
			t.Fatal("nil handle for a resident snapshot prefix")
		}
		if err := s.ActivateContinuation(h); err != nil {
			t.Fatalf("Activate(snapshot): %v", err)
		}
		n := nodeFor(t, s, ownerA, tokens)
		if tier := s.tree.NodeTier(n); tier != Tier2IdleSubagent {
			t.Fatalf("snapshot arm tier=%v want Tier2IdleSubagent", tier)
		}
		ret, ok := s.tree.NodeRetention(n)
		if !ok || ret.Priority != ContinuationSoftPriority {
			t.Fatalf("snapshot arm retention=%+v ok=%v want priority %d", ret, ok, ContinuationSoftPriority)
		}
	})

	t.Run("NoPinNoRetainForever", func(t *testing.T) {
		s := NewScoped(0)
		tokens := []int{8, 8, 8}
		if err := s.AdmitPrivate(ownerA, tokens, newKV(), nil); err != nil {
			t.Fatalf("AdmitPrivate: %v", err)
		}
		h, err := s.BeginPrivateContinuation(ownerA, tokens)
		if err != nil {
			t.Fatalf("BeginPrivateContinuation: %v", err)
		}
		if err := s.ActivateContinuation(h); err != nil {
			t.Fatalf("Activate: %v", err)
		}
		n := nodeFor(t, s, ownerA, tokens)
		ret, ok := s.tree.NodeRetention(n)
		if !ok {
			t.Fatal("no retention on activated node")
		}
		if ret.Priority >= 60 {
			t.Fatalf("retention priority=%d must be < 60 (Tier0/Tier1 would be a pin)", ret.Priority)
		}
		if ret.TTL == RetainForever {
			t.Fatal("retention TTL is RetainForever; the ticket forbids an unbounded window")
		}
		if ret.TTL == 0 {
			t.Fatal("retention TTL==0 (== RetainForever); must be finite and positive")
		}
		if s.tree.IsNodePinned(n) {
			t.Fatal("activated continuation node reports pinned")
		}
	})
}

// TestContinuationHostL2PayloadAdmission is the #13385 Gap A witness: a
// continuation is minted whenever a COMPLETE reusable payload is resident in ANY
// physical tier, not only hot L1. Here the hot copy is demoted to host DRAM L2
// (snapshot==nil, hostSnapshot!=nil) and Begin must still succeed, Activate must
// apply the same SOFT finite policy, and Status must report Active. The tree's own
// liveness predicate is snapshot!=nil || hostSnapshot!=nil || remoteSnapshot!=nil.
func TestContinuationHostL2PayloadAdmission(t *testing.T) {
	cfg := model.Config{
		HiddenSize:       16,
		NumLayers:        1,
		NumHeads:         2,
		NumKVHeads:       1,
		HeadDim:          8,
		IntermediateSize: 32,
		VocabSize:        32,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       31,
	}
	ownerA := CacheIdentity{Tenant: "tenant-a", Agent: "worker-1"}

	m := model.NewSynthetic(cfg)
	be := &deviceCapsBackend{Backend: compute.Default()}
	ids := []int{3, 7, 11, 13}
	sess, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	sess.Prefill(ids)
	snap, err := sess.PrefixSnapshot()
	sess.Close()
	if err != nil {
		t.Fatal(err)
	}

	// Host-L2 with a bounded hot snapshot budget; the seam the existing
	// host_l2_test.go uses is StageSnapshotToHost + EvictHotSnapshot.
	tree := NewWithTierBudgetsAndEvictionPolicy(0, 1<<30, 1<<30, EvictionLRU)
	s := WrapScoped(tree)
	if !tree.HostL2Enabled() {
		t.Fatal("tree does not report host L2 enabled")
	}
	if err := s.AdmitPrivateSnapshot(ownerA, ids, snap, []float32{3}); err != nil {
		t.Fatalf("AdmitPrivateSnapshot: %v", err)
	}

	// Demote the hot copy to host DRAM L2: stage first (leaves hot untouched),
	// then release the hot owner. A complete local copy (hostSnapshot) survives,
	// so the record incarnation must survive too.
	_, candidates := tree.PressuredSnapshotCandidates()
	if len(candidates) != 1 {
		t.Fatalf("candidates=%+v want exactly one hot owner", candidates)
	}
	if staged := tree.StageSnapshotToHost(candidates[0].Digest); staged.Outcome != SnapshotTransferOK {
		t.Fatalf("StageSnapshotToHost=%+v want OK", staged)
	}
	if evicted := tree.EvictHotSnapshot(candidates[0].Digest); evicted != len(ids) {
		t.Fatalf("EvictHotSnapshot positions=%d want %d", evicted, len(ids))
	}

	// Confirm the physical state the gap is about: hot copy gone, host copy present.
	ns, err := scopeNamespace(ScopeTenant, ownerA)
	if err != nil {
		t.Fatalf("scopeNamespace: %v", err)
	}
	n, matched := tree.LookupNS(ns, ids)
	if n == nil || matched != len(ids) {
		if n != nil {
			tree.Done(n)
		}
		t.Fatalf("post-demotion lookup node=%v matched=%d want full match", n, matched)
	}
	if n.snapshot != nil {
		tree.Done(n)
		t.Fatal("hot snapshot still resident; demotion did not happen")
	}
	if n.hostSnapshot == nil {
		tree.Done(n)
		t.Fatal("host L2 snapshot absent; demotion did not install the L2 copy")
	}
	if !n.hasRecord || n.recordGen == 0 {
		tree.Done(n)
		t.Fatal("demotion across tiers killed the record incarnation")
	}
	tree.Done(n)

	h, err := s.BeginPrivateContinuation(ownerA, ids)
	if err != nil {
		t.Fatalf("BeginPrivateContinuation(host-L2 payload): %v", err)
	}
	if h == nil {
		t.Fatal("nil handle for a host-L2-resident complete payload")
	}
	// Not-yet-activated status must agree with the handle's own dormancy.
	if st, err := s.ContinuationStatus(h); err != nil || st != ContinuationDormant {
		t.Fatalf("pre-activate status=%v err=%v want dormant/nil", st, err)
	}
	if err := s.ActivateContinuation(h); err != nil {
		t.Fatalf("ActivateContinuation(host-L2 payload): %v", err)
	}
	if !h.Active() {
		t.Fatal("handle not Active after Activate on a host-L2 payload")
	}
	st, err := s.ContinuationStatus(h)
	if err != nil {
		t.Fatalf("ContinuationStatus after activate: %v", err)
	}
	if st != ContinuationActive {
		t.Fatalf("status=%v want ContinuationActive", st)
	}

	n, matched = tree.LookupNS(ns, ids)
	if n == nil || matched != len(ids) {
		if n != nil {
			tree.Done(n)
		}
		t.Fatalf("post-activate lookup node=%v matched=%d", n, matched)
	}
	defer tree.Done(n)
	if tier := tree.NodeTier(n); tier != Tier2IdleSubagent {
		t.Fatalf("host-L2 arm tier=%v want Tier2IdleSubagent (priority %d < 60)", tier, ContinuationSoftPriority)
	}
	ret, ok := tree.NodeRetention(n)
	if !ok {
		t.Fatal("host-L2 activation left no retention request")
	}
	if ret.Priority != ContinuationSoftPriority {
		t.Fatalf("host-L2 retention priority=%d want %d", ret.Priority, ContinuationSoftPriority)
	}
	if ret.TTL <= 0 || ret.TTL == RetainForever {
		t.Fatalf("host-L2 retention TTL=%d must be finite and > 0 (RetainForever=%d)", ret.TTL, RetainForever)
	}
	if tree.IsNodePinned(n) {
		t.Fatal("host-L2 continuation node reports pinned")
	}
}

// TestContinuationStatusLifecycle is the #13385 Gap B witness: ContinuationStatus
// classifies a handle across its whole lifecycle without mutating it. Each phase
// asserts both the typed ContinuationState and the paired error.
func TestContinuationStatusLifecycle(t *testing.T) {
	cfg := model.Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 1}
	ownerA := CacheIdentity{Tenant: "tenant-a", Agent: "worker-1"}

	newKV := func() *model.KVCache {
		kv := model.NewKVCache(cfg)
		kv.K[0] = []float32{1, 2, 3, 4}
		kv.V[0] = []float32{5, 6, 7, 8}
		return kv
	}

	t.Run("DormantThenActive", func(t *testing.T) {
		s := NewScoped(0)
		tokens := []int{1, 2, 3, 4}
		if err := s.AdmitPrivate(ownerA, tokens, newKV(), nil); err != nil {
			t.Fatalf("AdmitPrivate: %v", err)
		}
		h, err := s.BeginPrivateContinuation(ownerA, tokens)
		if err != nil {
			t.Fatalf("BeginPrivateContinuation: %v", err)
		}
		if st, err := s.ContinuationStatus(h); err != nil || st != ContinuationDormant {
			t.Fatalf("after Begin status=%v err=%v want dormant/nil", st, err)
		}
		if err := s.ActivateContinuation(h); err != nil {
			t.Fatalf("ActivateContinuation: %v", err)
		}
		if st, err := s.ContinuationStatus(h); err != nil || st != ContinuationActive {
			t.Fatalf("after Activate status=%v err=%v want active/nil", st, err)
		}
	})

	t.Run("ReleaseDeactivatesSelf", func(t *testing.T) {
		s := NewScoped(0)
		tokens := []int{5, 6, 7}
		if err := s.AdmitPrivate(ownerA, tokens, newKV(), nil); err != nil {
			t.Fatalf("AdmitPrivate: %v", err)
		}
		h, err := s.BeginPrivateContinuation(ownerA, tokens)
		if err != nil {
			t.Fatalf("BeginPrivateContinuation: %v", err)
		}
		if err := s.ActivateContinuation(h); err != nil {
			t.Fatalf("ActivateContinuation: %v", err)
		}
		if err := s.ReleaseContinuation(h); err != nil {
			t.Fatalf("ReleaseContinuation: %v", err)
		}
		// The node is still resident with a reusable payload, so the spec puts the
		// handle back to the created-but-not-active state: a typed non-active status
		// (dormant) with no error. It must NOT continue to report active.
		st, err := s.ContinuationStatus(h)
		if st == ContinuationActive {
			t.Fatal("status still ContinuationActive after Release")
		}
		if err != nil {
			t.Fatalf("status after Release err=%v want nil for a live dormant handle", err)
		}
		if st != ContinuationDormant {
			t.Fatalf("status after Release=%v want ContinuationDormant", st)
		}
		if h.Active() {
			t.Fatal("released handle still reports Active()")
		}
	})

	t.Run("EvictedIsTypedNodeGone", func(t *testing.T) {
		s := NewScoped(0)
		tokens := []int{8, 8, 8}
		if err := s.AdmitPrivate(ownerA, tokens, newKV(), nil); err != nil {
			t.Fatalf("AdmitPrivate: %v", err)
		}
		h, err := s.BeginPrivateContinuation(ownerA, tokens)
		if err != nil {
			t.Fatalf("BeginPrivateContinuation: %v", err)
		}
		if err := s.ActivateContinuation(h); err != nil {
			t.Fatalf("ActivateContinuation: %v", err)
		}
		ns, err := scopeNamespace(ScopeTenant, ownerA)
		if err != nil {
			t.Fatalf("scopeNamespace: %v", err)
		}
		n, matched := s.tree.LookupNS(ns, tokens)
		if n == nil || matched != len(tokens) {
			if n != nil {
				s.tree.Done(n)
			}
			t.Fatalf("pre-evict lookup node=%v matched=%d", n, matched)
		}
		s.tree.Done(n)
		if freed := s.tree.EvictNode(n); freed == 0 {
			t.Fatal("EvictNode freed no tokens")
		}
		st, err := s.ContinuationStatus(h)
		if st != ContinuationEvicted {
			t.Fatalf("after evict status=%v want ContinuationEvicted", st)
		}
		if !errors.Is(err, ErrContinuationNodeGone) {
			t.Fatalf("after evict status err=%v want ErrContinuationNodeGone", err)
		}
	})

	t.Run("ReplacedRecordIsTypedStale", func(t *testing.T) {
		s := NewScoped(0)
		tokens := []int{9, 9, 9}
		snap := &model.PrefixSnapshot{Cache: model.NewKVCache(cfg)}
		if err := s.AdmitPrivateSnapshot(ownerA, tokens, snap, []float32{3}); err != nil {
			t.Fatalf("AdmitPrivateSnapshot: %v", err)
		}
		h, err := s.BeginPrivateContinuation(ownerA, tokens)
		if err != nil {
			t.Fatalf("BeginPrivateContinuation: %v", err)
		}
		if err := s.ActivateContinuation(h); err != nil {
			t.Fatalf("ActivateContinuation: %v", err)
		}
		if st, err := s.ContinuationStatus(h); err != nil || st != ContinuationActive {
			t.Fatalf("pre-replace status=%v err=%v want active/nil", st, err)
		}
		// Re-admit a snapshot on the SAME path: InsertSnapshot mints a fresh record
		// generation on the node, so the old handle no longer names the record it
		// admitted and must be classified stale, typed.
		replacement := &model.PrefixSnapshot{Cache: model.NewKVCache(cfg)}
		if err := s.AdmitPrivateSnapshot(ownerA, tokens, replacement, []float32{4}); err != nil {
			t.Fatalf("AdmitPrivateSnapshot(replacement): %v", err)
		}
		st, err := s.ContinuationStatus(h)
		if st != ContinuationStale {
			t.Fatalf("after record replacement status=%v want ContinuationStale", st)
		}
		if !errors.Is(err, ErrContinuationGeneration) {
			t.Fatalf("after record replacement status err=%v want ErrContinuationGeneration", err)
		}
	})
}
