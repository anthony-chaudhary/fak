package agent

// inkernel_warm_test.go — CW-04 (fak#13344): the native KV cache WARM API witness.
//
// The leaf's named witness is TestInKernelWarmPrefixReadback. It proves the three
// acceptance criteria the issue names, all software-observable on the synthetic model:
//
//  1. Warm success requires an EXACT compatible snapshot restore under the spec's own
//     tenant/agent scope; a structural MatchLen hit or a nil-KV entry cannot succeed.
//  2. A following real suffix request reuses the stable boundary with cold-reference
//     parity; a disabled/unsupported cache path reports explicit status, never a hit.
//  3. The receipt is defensive and bounded: it carries counts/digests/closed tokens only,
//     never prompt text or instruction contents.
//
// This is a correctness/residency witness, NOT a throughput measurement: PrimeDuration is
// a host wall-clock observation and no hardware cache-hit claim is made.

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// warmFixtureInputs is the stable input set the warm witness derives a descriptor from.
// It is deliberately SHORT: the witness materializes a real prefix, so the byte-level
// tokenizer multiplies text length by ~1 token/byte and the synthetic context is small.
func warmFixtureInputs() WarmPrefixInputs {
	return WarmPrefixInputs{
		Instructions: []byte("Follow the repo conventions."),
		SystemBlocks: [][]byte{[]byte("Default-deny capability floor.")},
		Tools:        warmPrefixFixtureTools(),
		KVLayout:     "fp32",
		AdapterID:    "native-inkernel",
	}
}

// warmCfg is the warm witness's synthetic model config. It must be forward-runnable (the
// warm materializes real KV through the production prefill), so every dimension is set —
// and VocabSize covers the byte-level probe tokenizer's ~259 ids so the real EncodePrompt
// boundary can be embedded and prefilled.
func warmCfg() model.Config {
	return model.Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 320, RMSNormEps: 1e-5, RopeTheta: 10000, EOSTokenID: 319,
	}
}

// warmFixturePlanner builds a reuse-enabled planner over the synthetic model, with a real
// tokenizer so the descriptor/warm path exercises the production EncodePrompt seam. It is
// the same fixture family as warmPrefixFixturePlanner but runs the served forward.
func warmFixturePlanner(t *testing.T) *InKernelPlanner {
	t.Helper()
	m := model.NewSynthetic(warmCfg())
	p := &InKernelPlanner{m: m, modelID: "synthetic", tok: loadProbeTok(t), tree: radixkv.New(0)}
	// A scoped tree so the warm readback runs the authenticated private path the served
	// path uses (AdmitPrivate/Lookup), not only the shared tree.
	p.scopedTree = radixkv.WrapScopedWithLocker(p.tree, &p.mu)
	return p
}

// decodeScoped runs one demand turn under an authenticated cache scope, collecting the
// generated token ids. Unlike decode it binds the scope so a scoped warm is visible to the
// demand lookup — the same cache the warm readback consulted.
func decodeScoped(p *InKernelPlanner, ctx context.Context, ids []int, maxNew int) (gen []int) {
	_, _, _, _, _, _, _ = p.generateReusedContext(ctx, ids, maxNew, 0, 0, 0, map[int]bool{}, func(id int) bool {
		gen = append(gen, id)
		return false
	})
	return gen
}

// TestInKernelWarmPrefixReadback is the leaf's named witness (fak#13344). A warm must
// PROVE a restore it observed: the descriptor's stable prefix is materialized through the
// CW-03 populate purpose, then read back under the spec's own scope.
func TestInKernelWarmPrefixReadback(t *testing.T) {
	ctx := context.Background()

	t.Run("warm materializes the stable prefix and reads it back under the spec scope", func(t *testing.T) {
		p := warmFixturePlanner(t)
		in := warmFixtureInputs()

		spec, err := p.DeriveWarmPrefix("tenant-a", "agent-1", in)
		if err != nil {
			t.Fatalf("DeriveWarmPrefix: %v", err)
		}
		if !spec.Bounded() {
			t.Fatalf("descriptor not bounded: %+v", spec)
		}
		p.SetWarmPrefixInputs(in)

		scoped := WithPrefixCacheIdentity(ctx, spec.Scope.Tenant, spec.Scope.Agent)
		receipt, err := p.WarmPrefix(scoped, spec)
		if err != nil {
			t.Fatalf("WarmPrefix: %v", err)
		}
		if !receipt.Usable() || !receipt.Ready {
			t.Fatalf("warm did not become ready: %+v", receipt)
		}
		if receipt.Status != WarmStatusReady {
			t.Fatalf("status = %q, want %q", receipt.Status, WarmStatusReady)
		}
		if receipt.RestoredTokens < spec.StableTokens || receipt.RestoredTokens == 0 {
			t.Fatalf("restored %d tokens, want >= stable boundary %d", receipt.RestoredTokens, spec.StableTokens)
		}
		if receipt.RequestedTokens != spec.StableTokens {
			t.Fatalf("requested %d, want %d", receipt.RequestedTokens, spec.StableTokens)
		}
		if !receipt.ZeroGenerated {
			t.Fatalf("warm reported generated tokens: %+v", receipt)
		}
		if receipt.Identity != spec.Identity {
			t.Fatalf("receipt identity %q != spec identity %q", receipt.Identity, spec.Identity)
		}
		if receipt.Scope.Tenant != "tenant-a" || receipt.Scope.Agent != "agent-1" {
			t.Fatalf("receipt scope not carried: %+v", receipt.Scope)
		}
		// The first-demand latch flips only AFTER a successful readback.
		if !p.kvPrefixEverAdmitted.Load() {
			t.Fatalf("first-demand latch not flipped after a ready warm")
		}
	})

	t.Run("a following suffix request reuses the stable boundary with cold parity", func(t *testing.T) {
		p := warmFixturePlanner(t)
		in := warmFixtureInputs()
		spec, err := p.DeriveWarmPrefix("tenant-a", "agent-1", in)
		if err != nil {
			t.Fatalf("DeriveWarmPrefix: %v", err)
		}
		p.SetWarmPrefixInputs(in)
		scoped := WithPrefixCacheIdentity(ctx, spec.Scope.Tenant, spec.Scope.Agent)
		if receipt, err := p.WarmPrefix(scoped, spec); err != nil || !receipt.Ready {
			t.Fatalf("WarmPrefix: err=%v receipt=%+v", err, receipt)
		}

		// A real suffix request continues the warmed prefix. The stable boundary must be
		// matched on the demand path (matched >= stable boundary), and the output must
		// equal a cold run of the same continuation.
		stableMsgs := []Message{{Role: RoleSystem, Content: string(in.Instructions)}}
		for _, block := range in.SystemBlocks {
			stableMsgs = append(stableMsgs, Message{Role: RoleSystem, Content: string(block)})
		}
		enc, err := p.EncodePrompt(ctx, stableMsgs, in.Tools)
		if err != nil {
			t.Fatalf("EncodePrompt: %v", err)
		}
		suffix := append(append([]int(nil), enc.TokenIDs...), synthIDs(warmCfg().VocabSize, 5, 1441)...)

		// The demand turn runs under the SAME authenticated scope the warm restored under,
		// because the scoped readback and the scoped demand lookup must consult the same
		// tenant/agent cache. generateReused binds no scope, so use the ctx-accepting form.
		warmOut := decodeScoped(p, scoped, suffix, 4)
		warmMatched, err := p.scopedTree.MatchLen(spec.Scope, suffix)
		if err != nil {
			t.Fatalf("scoped MatchLen: %v", err)
		}

		cold := warmFixturePlanner(t)
		coldMatched, err := cold.scopedTree.MatchLen(spec.Scope, suffix)
		if err != nil {
			t.Fatalf("cold scoped MatchLen: %v", err)
		}
		coldOut := decodeScoped(cold, scoped, suffix, 4)

		if warmMatched < spec.StableTokens {
			t.Fatalf("warmed demand matched %d, want >= stable boundary %d", warmMatched, spec.StableTokens)
		}
		if coldMatched != 0 {
			t.Fatalf("cold reference matched %d, want 0", coldMatched)
		}
		if !eqInts(warmOut, coldOut) {
			t.Fatalf("warmed continuation changed output: warm=%v cold=%v", warmOut, coldOut)
		}
	})

	t.Run("structural MatchLen alone is insufficient — a nil payload is not a warm", func(t *testing.T) {
		// The readback predicate is what encodes "a MatchLen / nil-KV entry is not a warm".
		// Exercise it directly on a tree whose ONLY entry is a nil-KV structural admission,
		// so the assertion isolates the predicate from the populate that precedes it in a
		// real warm (which would otherwise admit a real payload and mask the miss).
		p := warmFixturePlanner(t)
		scope := radixkv.CacheIdentity{Tenant: "tenant-a", Agent: "agent-1"}
		tokens := synthIDs(warmCfg().VocabSize, 12, 4321)

		// Structural-only: a node exists at the full token boundary with no snapshot/KV.
		if err := p.scopedTree.AdmitPrivate(scope, tokens, nil, nil); err != nil {
			t.Fatalf("AdmitPrivate(nil): %v", err)
		}
		if matched, err := p.scopedTree.MatchLen(scope, tokens); err != nil || matched != len(tokens) {
			t.Fatalf("precondition: structural MatchLen = %d/%v, want %d", matched, err, len(tokens))
		}
		if got, _, ok := p.warmRestoreReadback(scope, tokens); ok {
			t.Fatalf("readback accepted a nil-KV structural entry (matched=%d) as a warm", got)
		}
	})

	t.Run("unsupported planner refuses closed with an explicit status", func(t *testing.T) {
		// A planner with no tree cannot warm at all: it must refuse, never report ready.
		p := &InKernelPlanner{m: model.NewSynthetic(warmCfg()), modelID: "synthetic", tok: loadProbeTok(t)}
		in := warmFixtureInputs()
		// Build a spec against a warm-capable planner so the descriptor itself is valid;
		// then warm it on the unsupported planner.
		good := warmFixturePlanner(t)
		spec, err := good.DeriveWarmPrefix("tenant-a", "agent-1", in)
		if err != nil {
			t.Fatalf("DeriveWarmPrefix: %v", err)
		}
		p.SetWarmPrefixInputs(in)
		receipt, err := p.WarmPrefix(WithPrefixCacheIdentity(ctx, "tenant-a", "agent-1"), spec)
		if !errors.Is(err, ErrWarmPrefixUnsupported) {
			t.Fatalf("unsupported warm err = %v, want ErrWarmPrefixUnsupported", err)
		}
		if receipt.Ready || receipt.Usable() {
			t.Fatalf("unsupported warm reported ready: %+v", receipt)
		}
		if receipt.Status != WarmStatusUnsupported || receipt.Reason != "no_tree" {
			t.Fatalf("status/reason = %q/%q, want unsupported/no_tree", receipt.Status, receipt.Reason)
		}
	})

	t.Run("an unbounded descriptor and a planner without inputs refuse closed", func(t *testing.T) {
		p := warmFixturePlanner(t)

		// No SetWarmPrefixInputs: the boundary cannot be re-encoded, so the warm refuses
		// rather than guessing it from prompt text.
		in := warmFixtureInputs()
		spec, err := p.DeriveWarmPrefix("tenant-a", "agent-1", in)
		if err != nil {
			t.Fatalf("DeriveWarmPrefix: %v", err)
		}
		receipt, err := p.WarmPrefix(WithPrefixCacheIdentity(ctx, "tenant-a", "agent-1"), spec)
		if !errors.Is(err, ErrWarmPrefixUnsupported) {
			t.Fatalf("no-inputs warm err = %v, want ErrWarmPrefixUnsupported", err)
		}
		if receipt.Ready {
			t.Fatalf("no-inputs warm reported ready: %+v", receipt)
		}

		// An unbounded (zero-value) descriptor is refused before any cache work.
		p.SetWarmPrefixInputs(in)
		zero := WarmPrefixSpec{}
		receipt, err = p.WarmPrefix(WithPrefixCacheIdentity(ctx, "tenant-a", "agent-1"), zero)
		if !errors.Is(err, ErrWarmPrefixUnsupported) {
			t.Fatalf("unbounded warm err = %v, want ErrWarmPrefixUnsupported", err)
		}
		if receipt.Reason != "unbounded_descriptor" {
			t.Fatalf("unbounded warm reason = %q, want unbounded_descriptor", receipt.Reason)
		}
	})

	t.Run("the receipt carries no prompt text or instruction contents", func(t *testing.T) {
		p := warmFixturePlanner(t)
		in := warmFixtureInputs()
		spec, err := p.DeriveWarmPrefix("tenant-a", "agent-1", in)
		if err != nil {
			t.Fatalf("DeriveWarmPrefix: %v", err)
		}
		p.SetWarmPrefixInputs(in)
		scoped := WithPrefixCacheIdentity(ctx, spec.Scope.Tenant, spec.Scope.Agent)
		receipt, err := p.WarmPrefix(scoped, spec)
		if err != nil {
			t.Fatalf("WarmPrefix: %v", err)
		}
		blob, err := json.Marshal(receipt)
		if err != nil {
			t.Fatalf("marshal receipt: %v", err)
		}
		rendered := strings.ToLower(string(blob))
		for _, needle := range []string{
			"work precisely", "follow the repository conventions",
			"you operate inside fak", "default-deny",
		} {
			if strings.Contains(rendered, needle) {
				t.Fatalf("receipt leaked prompt/instruction content %q: %s", needle, blob)
			}
		}
	})

	t.Run("ordinary generation is unchanged after the warm API exists", func(t *testing.T) {
		prompt := synthIDs(warmCfg().VocabSize, 20, 909)
		p := warmFixturePlanner(t)
		got, matched := decode(p, prompt, 5)
		ref := warmFixturePlanner(t)
		want, wantMatched := decode(ref, prompt, 5)
		if !eqInts(got, want) || matched != wantMatched {
			t.Fatalf("ordinary decode changed: got=%v/%d want=%v/%d", got, matched, want, wantMatched)
		}
	})
}
