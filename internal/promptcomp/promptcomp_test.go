package promptcomp

import (
	"errors"
	"strings"
	"testing"
)

func TestPromptPartValidation(t *testing.T) {
	// Missing ID
	p := PromptPart{Content: "Hello world"}
	if err := p.Validate(); !errors.Is(err, ErrMissingID) {
		t.Fatalf("expected ErrMissingID, got %v", err)
	}

	// Empty Content
	p = PromptPart{ID: "test.id", Content: "   "}
	if err := p.Validate(); !errors.Is(err, ErrEmptyContent) {
		t.Fatalf("expected ErrEmptyContent, got %v", err)
	}

	// Automatic Digest Computation
	p = PromptPart{ID: "test.id", Content: "Hello world"}
	if err := p.Validate(); err != nil {
		t.Fatalf("validation failed: %v", err)
	}
	expectedDigest := ComputeDigest("Hello world")
	if p.Digest != expectedDigest {
		t.Fatalf("expected digest %s, got %s", expectedDigest, p.Digest)
	}

	// Digest Mismatch
	p = PromptPart{ID: "test.id", Content: "Hello world", Digest: "deadbeef"}
	if err := p.Validate(); !errors.Is(err, ErrDigestMismatch) {
		t.Fatalf("expected ErrDigestMismatch, got %v", err)
	}
}

func TestRegistryOperations(t *testing.T) {
	reg := NewRegistry()

	part1 := PromptPart{
		ID:      "spine.core",
		Content: "You are an agent operating in the fak kernel.",
		Kind:    KindSpine,
		Rank:    0,
	}
	if err := reg.Register(part1); err != nil {
		t.Fatalf("failed to register part1: %v", err)
	}

	// Duplicate ID
	if err := reg.Register(part1); !errors.Is(err, ErrDuplicateID) {
		t.Fatalf("expected ErrDuplicateID, got %v", err)
	}

	// Get
	got, ok := reg.Get("spine.core")
	if !ok || got.ID != "spine.core" {
		t.Fatalf("expected to get spine.core, got ok=%v, part=%+v", ok, got)
	}

	// Non-existent Get
	_, ok = reg.Get("non.existent")
	if ok {
		t.Fatal("expected ok=false for non-existent part")
	}

	// List
	part2 := PromptPart{
		ID:      "policy.floor",
		Content: "Default deny capability floor.",
		Kind:    KindSafety,
		Rank:    1,
	}
	if err := reg.Register(part2); err != nil {
		t.Fatalf("failed to register part2: %v", err)
	}

	list := reg.List()
	if len(list) != 2 {
		t.Fatalf("expected 2 parts in list, got %d", len(list))
	}
	if list[0].ID != "spine.core" || list[1].ID != "policy.floor" {
		t.Fatalf("unexpected list order: %s, %s", list[0].ID, list[1].ID)
	}
}

func TestCompileTopologicalOrderAndDependencies(t *testing.T) {
	reg := NewRegistry()

	// Register fragments out of topological order
	_ = reg.Register(PromptPart{
		ID:        "tools.fileops",
		Content:   "Tools: Read, Edit, Bash.",
		Kind:      KindTools,
		Rank:      10,
		DependsOn: []string{"contract.worker"},
	})
	_ = reg.Register(PromptPart{
		ID:        "contract.worker",
		Content:   "Role: S1 Leaf Worker.",
		Kind:      KindContract,
		Rank:      5,
		DependsOn: []string{"policy.floor"},
	})
	_ = reg.Register(PromptPart{
		ID:        "policy.floor",
		Content:   "Policy: Default-deny capability floor.",
		Kind:      KindSafety,
		Rank:      1,
		DependsOn: []string{"spine.core"},
	})
	_ = reg.Register(PromptPart{
		ID:      "spine.core",
		Content: "Identity: Root fak agent spine.",
		Kind:    KindSpine,
		Rank:    0,
	})

	compiled, err := reg.Compile(Env{})
	if err != nil {
		t.Fatalf("compilation failed: %v", err)
	}

	expectedOrder := []string{"spine.core", "policy.floor", "contract.worker", "tools.fileops"}
	if len(compiled.Parts) != len(expectedOrder) {
		t.Fatalf("expected %d parts, got %d", len(expectedOrder), len(compiled.Parts))
	}
	for i, expectedID := range expectedOrder {
		if compiled.Parts[i].ID != expectedID {
			t.Errorf("part[%d] = %s; want %s", i, compiled.Parts[i].ID, expectedID)
		}
	}

	// Verify Zone 1 vs Zone 2
	if !strings.Contains(compiled.Zone1Content, "Identity: Root fak agent spine.") {
		t.Errorf("Zone 1 missing spine: %s", compiled.Zone1Content)
	}
	if !strings.Contains(compiled.Zone1Content, "Policy: Default-deny capability floor.") {
		t.Errorf("Zone 1 missing policy: %s", compiled.Zone1Content)
	}
	if strings.Contains(compiled.Zone1Content, "Role: S1 Leaf Worker.") {
		t.Errorf("Zone 1 incorrectly contains contract: %s", compiled.Zone1Content)
	}
	if !strings.Contains(compiled.Zone2Content, "Role: S1 Leaf Worker.") {
		t.Errorf("Zone 2 missing contract: %s", compiled.Zone2Content)
	}
	if compiled.PrefixBytes != len(compiled.Zone1Content) {
		t.Errorf("prefixBytes %d != len(Zone1Content) %d", compiled.PrefixBytes, len(compiled.Zone1Content))
	}
}

func TestCompileMissingDependency(t *testing.T) {
	parts := []PromptPart{
		{
			ID:        "contract.worker",
			Content:   "Role: Worker.",
			Kind:      KindContract,
			DependsOn: []string{"missing.dep"},
		},
	}
	_, err := CompileParts(parts, Env{})
	if !errors.Is(err, ErrMissingDependency) {
		t.Fatalf("expected ErrMissingDependency, got %v", err)
	}
}

func TestCompileConflictingFragments(t *testing.T) {
	parts := []PromptPart{
		{
			ID:            "contract.coordinator",
			Content:       "Role: Coordinator.",
			Kind:          KindContract,
			ConflictsWith: []string{"contract.worker"},
		},
		{
			ID:      "contract.worker",
			Content: "Role: Worker.",
			Kind:    KindContract,
		},
	}
	_, err := CompileParts(parts, Env{})
	if !errors.Is(err, ErrConflictingFragments) {
		t.Fatalf("expected ErrConflictingFragments, got %v", err)
	}
}

func TestCompileCyclicDependency(t *testing.T) {
	parts := []PromptPart{
		{
			ID:        "part.a",
			Content:   "Content A",
			Kind:      KindContract,
			DependsOn: []string{"part.b"},
		},
		{
			ID:        "part.b",
			Content:   "Content B",
			Kind:      KindContract,
			DependsOn: []string{"part.c"},
		},
		{
			ID:        "part.c",
			Content:   "Content C",
			Kind:      KindContract,
			DependsOn: []string{"part.a"},
		},
	}
	_, err := CompileParts(parts, Env{})
	if !errors.Is(err, ErrCyclicDependency) {
		t.Fatalf("expected ErrCyclicDependency, got %v", err)
	}
}

func TestCompileZoneInversion(t *testing.T) {
	// Zone 1 part depending on Zone 2 part
	parts := []PromptPart{
		{
			ID:        "spine.core",
			Content:   "Root Spine",
			Kind:      KindSpine,
			DependsOn: []string{"contract.worker"},
		},
		{
			ID:      "contract.worker",
			Content: "Worker Contract",
			Kind:    KindContract,
		},
	}
	_, err := CompileParts(parts, Env{})
	if !errors.Is(err, ErrZoneInversion) {
		t.Fatalf("expected ErrZoneInversion, got %v", err)
	}
}

func TestCompilePredicateFiltering(t *testing.T) {
	parts := []PromptPart{
		{
			ID:      "spine.core",
			Content: "Root Spine",
			Kind:    KindSpine,
		},
		{
			ID:      "contract.concise",
			Content: "Concise Contract for Small Local Model",
			Kind:    KindContract,
			Predicate: func(e Env) bool {
				return e.IsSmallLocal
			},
		},
		{
			ID:      "contract.full",
			Content: "Full Reasoning Scaffolding Contract",
			Kind:    KindContract,
			Predicate: func(e Env) bool {
				return !e.IsSmallLocal
			},
		},
	}

	// Case 1: Small local model
	envSmall := Env{IsSmallLocal: true}
	cSmall, err := CompileParts(parts, envSmall)
	if err != nil {
		t.Fatalf("compilation failed for small model: %v", err)
	}
	if !strings.Contains(cSmall.Raw, "Concise Contract") {
		t.Errorf("expected concise contract in small model prompt: %s", cSmall.Raw)
	}
	if strings.Contains(cSmall.Raw, "Full Reasoning") {
		t.Errorf("did not expect full reasoning in small model prompt: %s", cSmall.Raw)
	}

	// Case 2: Frontier / Cloud model
	envLarge := Env{IsSmallLocal: false}
	cLarge, err := CompileParts(parts, envLarge)
	if err != nil {
		t.Fatalf("compilation failed for large model: %v", err)
	}
	if !strings.Contains(cLarge.Raw, "Full Reasoning") {
		t.Errorf("expected full reasoning contract in large model prompt: %s", cLarge.Raw)
	}
	if strings.Contains(cLarge.Raw, "Concise Contract") {
		t.Errorf("did not expect concise contract in large model prompt: %s", cLarge.Raw)
	}
}

func TestPrefixCacheInvarianceAcrossTiers(t *testing.T) {
	// Shared Zone 1 fragments
	spine := PromptPart{ID: "spine.core", Content: "UNIVERSAL_ROOT_SPINE_ATTENTION_SINK", Kind: KindSpine, Rank: 0}
	policy := PromptPart{ID: "policy.floor", Content: "DEFAULT_DENY_CAPABILITY_FLOOR", Kind: KindSafety, Rank: 1, DependsOn: []string{"spine.core"}}

	// Coordinator Zone 2
	coordinatorContract := PromptPart{
		ID:        "contract.coordinator",
		Content:   "ORCHESTRATE_WORKER_WAVE_AND_FOLD_RECEIPTS",
		Kind:      KindContract,
		DependsOn: []string{"policy.floor"},
		Predicate: func(e Env) bool { return e.AgentTier == "coordinator" },
	}
	coordinatorTools := PromptPart{
		ID:        "tools.all",
		Content:   "CATALOG: [Read, Edit, Bash, Glob, Grep, Launch, Reconcile]",
		Kind:      KindTools,
		DependsOn: []string{"contract.coordinator"},
		Predicate: func(e Env) bool { return e.AgentTier == "coordinator" },
	}

	// Leaf Worker Zone 2
	leafContract := PromptPart{
		ID:        "contract.leaf",
		Content:   "EXECUTE_BOUNDED_S1_TASK_AND_RUN_ONE_WITNESS",
		Kind:      KindContract,
		DependsOn: []string{"policy.floor"},
		Predicate: func(e Env) bool { return e.AgentTier == "leaf" },
	}
	leafTools := PromptPart{
		ID:        "tools.fileops",
		Content:   "CATALOG: [Read, Edit, Bash]",
		Kind:      KindTools,
		DependsOn: []string{"contract.leaf"},
		Predicate: func(e Env) bool { return e.AgentTier == "leaf" },
	}

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

func Test1000IterationsBitExactDeterminism(t *testing.T) {
	parts := []PromptPart{
		{ID: "p.1", Content: "Content 1", Kind: KindSpine, Rank: 1},
		{ID: "p.2", Content: "Content 2", Kind: KindSafety, Rank: 2, DependsOn: []string{"p.1"}},
		{ID: "p.3", Content: "Content 3", Kind: KindContract, Rank: 3, DependsOn: []string{"p.2"}},
		{ID: "p.4", Content: "Content 4", Kind: KindTools, Rank: 4, DependsOn: []string{"p.3"}},
		{ID: "p.5", Content: "Content 5", Kind: KindOverlay, Rank: 5, DependsOn: []string{"p.3"}},
	}

	base, err := CompileParts(parts, Env{})
	if err != nil {
		t.Fatalf("base compile failed: %v", err)
	}

	for i := 0; i < 1000; i++ {
		iter, err := CompileParts(parts, Env{})
		if err != nil {
			t.Fatalf("iteration %d compile failed: %v", i, err)
		}
		if iter.Raw != base.Raw {
			t.Fatalf("iteration %d raw mismatch", i)
		}
		if iter.TotalDigest != base.TotalDigest {
			t.Fatalf("iteration %d digest mismatch", i)
		}
		if iter.PrefixBytes != base.PrefixBytes {
			t.Fatalf("iteration %d prefix bytes mismatch", i)
		}
	}
}
