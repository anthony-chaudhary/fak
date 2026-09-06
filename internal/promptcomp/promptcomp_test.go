package promptcomp

import (
	"errors"
	"math/rand"
	"strings"
	"testing"
)

// TestRegistrationAndDigest verifies SHA-256 digest computation and validation.
func TestRegistrationAndDigest(t *testing.T) {
	// 1. Verify NewPromptPart computes SHA-256 automatically
	content := "Identity: Universal Fak Agent Root Spine."
	expectedDigest := ComputeDigest(content)
	part := NewPromptPart("core.identity", content, KindSpine, 0)
	if part.Digest != expectedDigest {
		t.Fatalf("expected digest %s, got %s", expectedDigest, part.Digest)
	}

	// 2. Verify Register with valid digest succeeds
	reg := NewRegistry()
	if err := reg.Register(part); err != nil {
		t.Fatalf("expected register to succeed, got %v", err)
	}

	// 3. Verify Get retrieves the registered part
	got, ok := reg.Get("core.identity")
	if !ok {
		t.Fatalf("expected part 'core.identity' to be found")
	}
	if got.ID != part.ID || got.Digest != part.Digest {
		t.Fatalf("retrieved part mismatch: %+v vs %+v", got, part)
	}

	// 4. Verify duplicate ID returns ErrDuplicateID
	if err := reg.Register(part); !errors.Is(err, ErrDuplicateID) {
		t.Fatalf("expected ErrDuplicateID on duplicate registration, got %v", err)
	}

	// 5. Verify empty ID returns ErrMissingID
	badIDPart := PromptPart{
		ID:      "",
		Content: "Valid Content",
		Digest:  ComputeDigest("Valid Content"),
		Kind:    KindSpine,
	}
	if err := reg.Register(badIDPart); !errors.Is(err, ErrMissingID) {
		t.Fatalf("expected ErrMissingID on empty ID, got %v", err)
	}

	// 6. Verify empty content returns ErrEmptyContent
	badContentPart := PromptPart{
		ID:      "test.empty.content",
		Content: "   ",
		Digest:  ComputeDigest("   "),
		Kind:    KindSpine,
	}
	if err := reg.Register(badContentPart); !errors.Is(err, ErrEmptyContent) {
		t.Fatalf("expected ErrEmptyContent on whitespace content, got %v", err)
	}

	// 7. Verify mismatched digest returns ErrDigestMismatch
	badDigestPart := PromptPart{
		ID:      "test.bad.digest",
		Content: "Valid Content",
		Digest:  "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		Kind:    KindSpine,
	}
	if err := reg.Register(badDigestPart); !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("expected ErrDigestMismatch on invalid digest, got %v", err)
	}

	// 8. Verify empty digest returns ErrDigestMismatch
	emptyDigestPart := PromptPart{
		ID:      "test.no.digest",
		Content: "Valid Content",
		Digest:  "",
		Kind:    KindSpine,
	}
	if err := reg.Register(emptyDigestPart); !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("expected ErrDigestMismatch on missing digest, got %v", err)
	}

	// 9. Verify PromptPart.Validate
	pVal := PromptPart{ID: "val.test", Content: "Some content"}
	if err := pVal.Validate(); err != nil {
		t.Fatalf("expected Validate to succeed, got %v", err)
	}
	if pVal.Digest != ComputeDigest("Some content") {
		t.Fatalf("expected Validate to populate digest, got %s", pVal.Digest)
	}
}

// TestCyclicDependencyDetection verifies Kahn's/DFS detects cycles and returns ErrCyclicDependency.
func TestCyclicDependencyDetection(t *testing.T) {
	// Case 1: Direct 2-node cycle (A -> B -> A)
	pA := NewPromptPart("A", "Content A", KindContract, 1, WithDependsOn("B"))
	pB := NewPromptPart("B", "Content B", KindContract, 2, WithDependsOn("A"))
	asm2 := NewAssemblerWithParts([]PromptPart{pA, pB})
	_, err := asm2.Resolve(Env{})
	if !errors.Is(err, ErrCyclicDependency) {
		t.Fatalf("expected ErrCyclicDependency for 2-cycle, got %v", err)
	}
	_, err = asm2.Assemble(Env{})
	if !errors.Is(err, ErrCyclicDependency) {
		t.Fatalf("expected ErrCyclicDependency from Assemble for 2-cycle, got %v", err)
	}

	// Case 2: Self-cycle (A -> A)
	pSelf := NewPromptPart("self", "Self Content", KindContract, 1, WithDependsOn("self"))
	asmSelf := NewAssemblerWithParts([]PromptPart{pSelf})
	_, err = asmSelf.Resolve(Env{})
	if !errors.Is(err, ErrCyclicDependency) {
		t.Fatalf("expected ErrCyclicDependency for self-cycle, got %v", err)
	}

	// Case 3: 3-node cycle (A -> B -> C -> A)
	p1 := NewPromptPart("n1", "Content 1", KindContract, 1, WithDependsOn("n2"))
	p2 := NewPromptPart("n2", "Content 2", KindContract, 2, WithDependsOn("n3"))
	p3 := NewPromptPart("n3", "Content 3", KindContract, 3, WithDependsOn("n1"))
	asm3 := NewAssemblerWithParts([]PromptPart{p1, p2, p3})
	_, err = asm3.Resolve(Env{})
	if !errors.Is(err, ErrCyclicDependency) {
		t.Fatalf("expected ErrCyclicDependency for 3-cycle, got %v", err)
	}

	// Case 4: Disjoint graph containing a cycle
	valid1 := NewPromptPart("v1", "Valid 1", KindSpine, 0)
	valid2 := NewPromptPart("v2", "Valid 2", KindPolicy, 1, WithDependsOn("v1"))
	c1 := NewPromptPart("c1", "Cycle 1", KindTools, 1, WithDependsOn("c2"))
	c2 := NewPromptPart("c2", "Cycle 2", KindTools, 2, WithDependsOn("c1"))
	asmDisjoint := NewAssemblerWithParts([]PromptPart{valid1, valid2, c1, c2})
	_, err = asmDisjoint.Resolve(Env{})
	if !errors.Is(err, ErrCyclicDependency) {
		t.Fatalf("expected ErrCyclicDependency for disjoint cycle, got %v", err)
	}

	// Case 5: Valid DAG without cycles (A -> B, C -> B, D -> A, D -> C)
	b := NewPromptPart("B", "Content B", KindSpine, 0)
	a := NewPromptPart("A", "Content A", KindPolicy, 1, WithDependsOn("B"))
	c := NewPromptPart("C", "Content C", KindContract, 2, WithDependsOn("B"))
	d := NewPromptPart("D", "Content D", KindTools, 3, WithDependsOn("A", "C"))
	asmDAG := NewAssemblerWithParts([]PromptPart{d, c, b, a})
	resolved, err := asmDAG.Resolve(Env{})
	if err != nil {
		t.Fatalf("expected DAG to resolve without error, got %v", err)
	}
	if len(resolved) != 4 {
		t.Fatalf("expected 4 resolved parts, got %d", len(resolved))
	}
}

// TestConflictDetection verifies conflicting parts produce ErrConflict.
func TestConflictDetection(t *testing.T) {
	// Case 1: Part A declares conflict with B
	pA := NewPromptPart("worker.fast", "Fast Worker", KindContract, 1, WithConflictsWith("worker.thorough"))
	pB := NewPromptPart("worker.thorough", "Thorough Worker", KindContract, 2)
	asm := NewAssemblerWithParts([]PromptPart{pA, pB})
	_, err := asm.Resolve(Env{})
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("expected ErrConflict when A conflicts with B, got %v", err)
	}
	_, err = asm.Assemble(Env{})
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("expected ErrConflict from Assemble, got %v", err)
	}

	// Case 2: Part B declares conflict with A
	pA2 := NewPromptPart("worker.a", "Worker A", KindContract, 1)
	pB2 := NewPromptPart("worker.b", "Worker B", KindContract, 2, WithConflictsWith("worker.a"))
	asm2 := NewAssemblerWithParts([]PromptPart{pA2, pB2})
	_, err = asm2.Resolve(Env{})
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("expected ErrConflict when B conflicts with A, got %v", err)
	}

	// Case 3: Mutual conflict
	pA3 := NewPromptPart("part.x", "X", KindContract, 1, WithConflictsWith("part.y"))
	pB3 := NewPromptPart("part.y", "Y", KindContract, 2, WithConflictsWith("part.x"))
	asm3 := NewAssemblerWithParts([]PromptPart{pA3, pB3})
	_, err = asm3.Resolve(Env{})
	if !errors.Is(err, ErrConflict) {
		t.Fatalf("expected ErrConflict on mutual conflict, got %v", err)
	}

	// Case 4: Conflict where conflicting part is inactive (predicate false) -> no conflict
	pA4 := NewPromptPart("active.part", "Active", KindContract, 1, WithConflictsWith("inactive.part"))
	pB4 := NewPromptPart("inactive.part", "Inactive", KindContract, 2, WithPredicate(func(e Env) bool {
		return false
	}))
	asm4 := NewAssemblerWithParts([]PromptPart{pA4, pB4})
	resolved, err := asm4.Resolve(Env{})
	if err != nil {
		t.Fatalf("expected no conflict when conflicting part is inactive, got %v", err)
	}
	if len(resolved) != 1 || resolved[0].ID != "active.part" {
		t.Fatalf("expected only active.part in resolved set, got %+v", resolved)
	}
}

// TestMissingDependency verifies missing dependencies produce ErrMissingDependency.
func TestMissingDependency(t *testing.T) {
	// Case 1: Part depends on a non-existent part
	p := NewPromptPart("contract.worker", "Worker Contract", KindContract, 1, WithDependsOn("non.existent.part"))
	asm := NewAssemblerWithParts([]PromptPart{p})
	_, err := asm.Resolve(Env{})
	if !errors.Is(err, ErrMissingDependency) {
		t.Fatalf("expected ErrMissingDependency for non-existent dependency, got %v", err)
	}
	_, err = asm.Assemble(Env{})
	if !errors.Is(err, ErrMissingDependency) {
		t.Fatalf("expected ErrMissingDependency from Assemble, got %v", err)
	}

	// Case 2: Part depends on a part whose predicate evaluates to false
	pParent := NewPromptPart("parent.part", "Parent", KindSpine, 0, WithPredicate(func(e Env) bool {
		return e.AgentTier == "coordinator"
	}))
	pChild := NewPromptPart("child.part", "Child", KindContract, 1, WithDependsOn("parent.part"))
	asmPredicate := NewAssemblerWithParts([]PromptPart{pParent, pChild})

	// When AgentTier is "leaf", parent is inactive -> missing dependency
	_, err = asmPredicate.Resolve(Env{AgentTier: "leaf"})
	if !errors.Is(err, ErrMissingDependency) {
		t.Fatalf("expected ErrMissingDependency when dependency predicate is false, got %v", err)
	}

	// When AgentTier is "coordinator", parent is active -> resolves successfully
	resolved, err := asmPredicate.Resolve(Env{AgentTier: "coordinator"})
	if err != nil {
		t.Fatalf("expected successful resolution when dependency predicate is true, got %v", err)
	}
	if len(resolved) != 2 {
		t.Fatalf("expected 2 resolved parts, got %d", len(resolved))
	}
}

// TestDeterministicAssemblyOrdering verifies Kind order (Spine -> Policy -> Contract -> Tools -> Overlay) and secondary Rank ordering.
func TestDeterministicAssemblyOrdering(t *testing.T) {
	parts := []PromptPart{
		NewPromptPart("overlay.debug", "OVERLAY_DEBUG", KindOverlay, 10),
		NewPromptPart("tools.git", "TOOLS_GIT", KindTools, 5),
		NewPromptPart("tools.read", "TOOLS_READ", KindTools, 1),
		NewPromptPart("contract.worker", "CONTRACT_WORKER", KindContract, 5),
		NewPromptPart("contract.base", "CONTRACT_BASE", KindContract, 1),
		NewPromptPart("contract.alpha", "CONTRACT_ALPHA", KindContract, 1), // same Kind & Rank as base, but ID 'alpha' < 'base'
		NewPromptPart("policy.sandbox", "POLICY_SANDBOX", KindPolicy, 2),
		NewPromptPart("policy.floor", "POLICY_FLOOR", KindPolicy, 1),
		NewPromptPart("spine.core", "SPINE_CORE", KindSpine, 0),
	}

	// Register in deliberately reversed/scrambled order into Registry
	reg := NewRegistry()
	for i := len(parts) - 1; i >= 0; i-- {
		if err := reg.Register(parts[i]); err != nil {
			t.Fatalf("failed to register part: %v", err)
		}
	}

	asm := NewAssembler(reg)
	resolved, err := asm.Resolve(Env{})
	if err != nil {
		t.Fatalf("resolve failed: %v", err)
	}

	expectedIDs := []string{
		"spine.core",      // KindSpine (0), Rank 0
		"policy.floor",    // KindPolicy (1), Rank 1
		"policy.sandbox",  // KindPolicy (1), Rank 2
		"contract.alpha",  // KindContract (2), Rank 1, ID alpha
		"contract.base",   // KindContract (2), Rank 1, ID base
		"contract.worker", // KindContract (2), Rank 5
		"tools.read",      // KindTools (3), Rank 1
		"tools.git",       // KindTools (3), Rank 5
		"overlay.debug",   // KindOverlay (4), Rank 10
	}

	if len(resolved) != len(expectedIDs) {
		t.Fatalf("expected %d resolved parts, got %d", len(expectedIDs), len(resolved))
	}

	for i, expectedID := range expectedIDs {
		if resolved[i].ID != expectedID {
			t.Errorf("part[%d] = %s; want %s (Kind=%v, Rank=%d)",
				i, resolved[i].ID, expectedID, resolved[i].Kind, resolved[i].Rank)
		}
	}

	// Verify Assemble concatenates with newline separators
	assembled, err := asm.Assemble(Env{})
	if err != nil {
		t.Fatalf("assemble failed: %v", err)
	}

	var expectedContents []string
	for _, id := range expectedIDs {
		part, _ := reg.Get(id)
		expectedContents = append(expectedContents, part.Content)
	}
	expectedText := strings.Join(expectedContents, "\n")

	if assembled != expectedText {
		t.Fatalf("assembled text mismatch:\ngot:\n%s\nwant:\n%s", assembled, expectedText)
	}
}

// TestPredicateFiltering verifies parts are included/excluded based on Env.
func TestPredicateFiltering(t *testing.T) {
	parts := []PromptPart{
		NewPromptPart("spine.root", "Spine Root", KindSpine, 0), // Unconditional
		NewPromptPart("contract.coord", "Coordinator Contract", KindContract, 1, WithPredicate(func(e Env) bool {
			return e.AgentTier == "coordinator"
		})),
		NewPromptPart("contract.leaf", "Leaf Contract", KindContract, 1, WithPredicate(func(e Env) bool {
			return e.AgentTier == "leaf"
		})),
		NewPromptPart("policy.model_qwen", "Qwen Specific Policy", KindPolicy, 1, WithPredicate(func(e Env) bool {
			return e.Model == "qwen3.8" || e.ModelFamily == "qwen3.8"
		})),
		NewPromptPart("overlay.high_context", "High Context Card", KindOverlay, 1, WithPredicate(func(e Env) bool {
			return e.ContextBudget >= 32000
		})),
		NewPromptPart("tools.sandbox", "Sandbox Tools", KindTools, 1, WithPredicate(func(e Env) bool {
			if val, ok := e.Metadata["sandbox"].(bool); ok {
				return val
			}
			return false
		})),
	}

	asm := NewAssemblerWithParts(parts)

	// Scenario 1: Coordinator on Qwen with small context, no sandbox
	envCoord := Env{
		Model:         "qwen3.8",
		AgentTier:     "coordinator",
		ContextBudget: 8000,
	}
	resolvedCoord, err := asm.Resolve(envCoord)
	if err != nil {
		t.Fatalf("resolve failed for coord: %v", err)
	}
	coordIDs := make(map[string]bool)
	for _, p := range resolvedCoord {
		coordIDs[p.ID] = true
	}
	if !coordIDs["spine.root"] || !coordIDs["contract.coord"] || !coordIDs["policy.model_qwen"] {
		t.Errorf("missing expected parts for coordinator: %+v", coordIDs)
	}
	if coordIDs["contract.leaf"] || coordIDs["overlay.high_context"] || coordIDs["tools.sandbox"] {
		t.Errorf("unexpected parts included for coordinator: %+v", coordIDs)
	}

	// Scenario 2: Leaf on Claude with large context and sandbox
	envLeaf := Env{
		Model:         "claude-3-7-sonnet",
		AgentTier:     "leaf",
		ContextBudget: 64000,
		Metadata: map[string]any{
			"sandbox": true,
		},
	}
	resolvedLeaf, err := asm.Resolve(envLeaf)
	if err != nil {
		t.Fatalf("resolve failed for leaf: %v", err)
	}
	leafIDs := make(map[string]bool)
	for _, p := range resolvedLeaf {
		leafIDs[p.ID] = true
	}
	if !leafIDs["spine.root"] || !leafIDs["contract.leaf"] || !leafIDs["overlay.high_context"] || !leafIDs["tools.sandbox"] {
		t.Errorf("missing expected parts for leaf: %+v", leafIDs)
	}
	if leafIDs["contract.coord"] || leafIDs["policy.model_qwen"] {
		t.Errorf("unexpected parts included for leaf: %+v", leafIDs)
	}
}

// TestByteDeterminism verifies 1,000 iterations producing bit-exact identical strings.
func TestByteDeterminism(t *testing.T) {
	canonicalParts := []PromptPart{
		NewPromptPart("p.spine", "SPINE: Root immutable sink", KindSpine, 0),
		NewPromptPart("p.policy.1", "POLICY 1: Default deny floor", KindPolicy, 1, WithDependsOn("p.spine")),
		NewPromptPart("p.policy.2", "POLICY 2: Execution limits", KindPolicy, 2, WithDependsOn("p.policy.1")),
		NewPromptPart("p.contract.1", "CONTRACT 1: Worker instructions", KindContract, 1, WithDependsOn("p.policy.2")),
		NewPromptPart("p.contract.2", "CONTRACT 2: Delivery receipts", KindContract, 2, WithDependsOn("p.contract.1")),
		NewPromptPart("p.tools.1", "TOOLS 1: Read/Edit/Grep", KindTools, 1, WithDependsOn("p.contract.1")),
		NewPromptPart("p.tools.2", "TOOLS 2: Bash execution", KindTools, 2, WithDependsOn("p.tools.1")),
		NewPromptPart("p.overlay.1", "OVERLAY 1: Memory notes", KindOverlay, 1, WithDependsOn("p.contract.2")),
		NewPromptPart("p.overlay.2", "OVERLAY 2: Error recovery", KindOverlay, 2, WithDependsOn("p.overlay.1")),
	}

	env := Env{
		Model:         "qwen3.8",
		AgentTier:     "leaf",
		ContextBudget: 16000,
	}

	// Establish reference baseline
	baseAsm := NewAssemblerWithParts(canonicalParts)
	baselineOutput, err := baseAsm.Assemble(env)
	if err != nil {
		t.Fatalf("baseline assemble failed: %v", err)
	}
	baselineDigest := ComputeDigest(baselineOutput)

	// Seed deterministic PRNG for shuffle testing
	rng := rand.New(rand.NewSource(42))

	for i := 0; i < 1000; i++ {
		// Scramble candidate slice order to ensure assembler is immune to input permutation
		shuffled := make([]PromptPart, len(canonicalParts))
		copy(shuffled, canonicalParts)
		rng.Shuffle(len(shuffled), func(a, b int) {
			shuffled[a], shuffled[b] = shuffled[b], shuffled[a]
		})

		asm := NewAssemblerWithParts(shuffled)
		out, err := asm.Assemble(env)
		if err != nil {
			t.Fatalf("iteration %d assemble failed: %v", i, err)
		}
		if out != baselineOutput {
			t.Fatalf("iteration %d produced non-identical string output", i)
		}
		digest := ComputeDigest(out)
		if digest != baselineDigest {
			t.Fatalf("iteration %d produced digest %s != baseline %s", i, digest, baselineDigest)
		}
	}
}

// TestPrefixCacheInvarianceAcrossTiers tests Zone 1 prefix cache invariance.
func TestPrefixCacheInvarianceAcrossTiers(t *testing.T) {
	spine := NewPromptPart("spine.core", "UNIVERSAL_ROOT_SPINE_ATTENTION_SINK", KindSpine, 0)
	policy := NewPromptPart("policy.floor", "DEFAULT_DENY_CAPABILITY_FLOOR", KindPolicy, 1, WithDependsOn("spine.core"))

	coordinatorContract := NewPromptPart(
		"contract.coordinator",
		"ORCHESTRATE_WORKER_WAVE_AND_FOLD_RECEIPTS",
		KindContract,
		1,
		WithDependsOn("policy.floor"),
		WithPredicate(func(e Env) bool { return e.AgentTier == "coordinator" }),
	)
	coordinatorTools := NewPromptPart(
		"tools.all",
		"CATALOG: [Read, Edit, Bash, Glob, Grep, Launch, Reconcile]",
		KindTools,
		1,
		WithDependsOn("contract.coordinator"),
		WithPredicate(func(e Env) bool { return e.AgentTier == "coordinator" }),
	)

	leafContract := NewPromptPart(
		"contract.leaf",
		"EXECUTE_BOUNDED_S1_TASK_AND_RUN_ONE_WITNESS",
		KindContract,
		1,
		WithDependsOn("policy.floor"),
		WithPredicate(func(e Env) bool { return e.AgentTier == "leaf" }),
	)
	leafTools := NewPromptPart(
		"tools.fileops",
		"CATALOG: [Read, Edit, Bash]",
		KindTools,
		1,
		WithDependsOn("contract.leaf"),
		WithPredicate(func(e Env) bool { return e.AgentTier == "leaf" }),
	)

	pool := []PromptPart{spine, policy, coordinatorContract, coordinatorTools, leafContract, leafTools}

	coordPrompt, err := CompileParts(pool, Env{AgentTier: "coordinator"})
	if err != nil {
		t.Fatalf("coordinator compile failed: %v", err)
	}

	leafPrompt, err := CompileParts(pool, Env{AgentTier: "leaf"})
	if err != nil {
		t.Fatalf("leaf compile failed: %v", err)
	}

	// Assert Zone 1 invariance
	if coordPrompt.PrefixBytes != leafPrompt.PrefixBytes {
		t.Fatalf("prefix byte length mismatch: coord=%d, leaf=%d", coordPrompt.PrefixBytes, leafPrompt.PrefixBytes)
	}
	if coordPrompt.Zone1Content != leafPrompt.Zone1Content {
		t.Fatalf("Zone 1 content mismatch:\nCoord: %q\nLeaf: %q", coordPrompt.Zone1Content, leafPrompt.Zone1Content)
	}
	if coordPrompt.Zone1Digest != leafPrompt.Zone1Digest {
		t.Fatalf("Zone 1 digest mismatch: coord=%s, leaf=%s", coordPrompt.Zone1Digest, leafPrompt.Zone1Digest)
	}

	// Assert Zone 2 divergence
	if coordPrompt.Zone2Content == leafPrompt.Zone2Content {
		t.Fatal("Zone 2 content should differ between coordinator and leaf")
	}
	if coordPrompt.TotalDigest == leafPrompt.TotalDigest {
		t.Fatal("Total digest should differ between coordinator and leaf")
	}
}

// TestCompileZoneInversion tests that a Zone 1 prefix fragment depending on a Zone 2 fragment returns ErrZoneInversion.
func TestCompileZoneInversion(t *testing.T) {
	parts := []PromptPart{
		NewPromptPart("spine.core", "Root Spine", KindSpine, 0, WithDependsOn("contract.worker")),
		NewPromptPart("contract.worker", "Worker Contract", KindContract, 1),
	}
	_, err := CompileParts(parts, Env{})
	if !errors.Is(err, ErrZoneInversion) {
		t.Fatalf("expected ErrZoneInversion, got %v", err)
	}
}

// TestRegistryOperations tests general registry operations.
func TestRegistryOperations(t *testing.T) {
	reg := NewRegistry()
	p1 := NewPromptPart("spine.core", "Root spine", KindSpine, 0)
	p2 := NewPromptPart("policy.floor", "Default deny", KindPolicy, 1)

	if err := reg.Register(p1); err != nil {
		t.Fatalf("register p1 failed: %v", err)
	}
	if err := reg.Register(p2); err != nil {
		t.Fatalf("register p2 failed: %v", err)
	}

	if reg.Len() != 2 {
		t.Fatalf("expected 2 parts, got %d", reg.Len())
	}

	list := reg.List()
	if len(list) != 2 || list[0].ID != "spine.core" || list[1].ID != "policy.floor" {
		t.Fatalf("unexpected list: %+v", list)
	}
}
