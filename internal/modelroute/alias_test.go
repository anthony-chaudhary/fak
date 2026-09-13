package modelroute

import (
	"strconv"
	"strings"
	"sync"
	"testing"
)

// TestAliasResolveMultiHop proves a chain A->B->C resolves to the terminal target C,
// and reports isAlias=true.
func TestAliasResolveMultiHop(t *testing.T) {
	s, err := NewAliasStore(AliasRegistry{Aliases: []Alias{
		{Name: "a", Target: "b"},
		{Name: "b", Target: "c"},
	}})
	if err != nil {
		t.Fatalf("NewAliasStore: %v", err)
	}
	got, isAlias, err := s.Resolve("a")
	if err != nil {
		t.Fatalf("Resolve(a): %v", err)
	}
	if !isAlias || got != "c" {
		t.Fatalf("Resolve(a) = (%q, %v), want (c, true)", got, isAlias)
	}
	// A non-alias passes through untouched with isAlias=false.
	got, isAlias, err = s.Resolve("z")
	if err != nil || isAlias || got != "z" {
		t.Fatalf("Resolve(z) = (%q, %v, %v), want (z, false, nil)", got, isAlias, err)
	}
}

// TestAliasRejectsCycle asserts a direct A->B->A cycle is refused at construction.
func TestAliasRejectsCycle(t *testing.T) {
	_, err := NewAliasStore(AliasRegistry{Aliases: []Alias{
		{Name: "a", Target: "b"},
		{Name: "b", Target: "a"},
	}})
	if err == nil {
		t.Fatal("expected cycle rejection, got nil")
	}
	if !strings.Contains(err.Error(), "cycle") {
		t.Fatalf("error = %v, want a cycle message", err)
	}
}

// TestAliasRejectsSelfCycle asserts A->A is refused.
func TestAliasRejectsSelfCycle(t *testing.T) {
	_, err := NewAliasStore(AliasRegistry{Aliases: []Alias{
		{Name: "a", Target: "a"},
	}})
	if err == nil {
		t.Fatal("expected self-cycle rejection, got nil")
	}
}

// TestAliasDepthBoundary pins the ONE shared edge budget: a chain of exactly
// MaxAliasDepth edges VALIDATES and RESOLVES to its terminal, while a chain of
// MaxAliasDepth+1 edges is REJECTED by Validate. This is the off-by-one the old
// Validate/Resolve split allowed (Validate blessed 10 edges, Resolve walked only 9).
func TestAliasDepthBoundary(t *testing.T) {
	// buildChain returns n0->n1->...->n<edges>, where n<edges> is a non-alias
	// terminal (absent from the table).
	buildChain := func(edges int) AliasRegistry {
		reg := AliasRegistry{Version: AliasVersion}
		for i := 0; i < edges; i++ {
			reg.Aliases = append(reg.Aliases, Alias{Name: aliasName(i), Target: aliasName(i + 1)})
		}
		return reg
	}

	// Exactly MaxAliasDepth edges: valid, and Resolve walks the whole chain.
	atLimit := buildChain(MaxAliasDepth)
	if err := atLimit.Validate(); err != nil {
		t.Fatalf("chain of exactly MaxAliasDepth (%d) edges must validate, got %v", MaxAliasDepth, err)
	}
	s, err := NewAliasStore(atLimit)
	if err != nil {
		t.Fatalf("NewAliasStore(at limit): %v", err)
	}
	target, isAlias, err := s.Resolve(aliasName(0))
	if err != nil {
		t.Fatalf("Resolve at-limit chain: %v", err)
	}
	if !isAlias || target != aliasName(MaxAliasDepth) {
		t.Fatalf("Resolve at-limit = (%q, %v), want (%q, true)", target, isAlias, aliasName(MaxAliasDepth))
	}

	// One edge over the budget: refused by Validate.
	overLimit := buildChain(MaxAliasDepth + 1)
	if err := overLimit.Validate(); err == nil {
		t.Fatalf("chain of MaxAliasDepth+1 (%d) edges must be rejected", MaxAliasDepth+1)
	}
}

// TestAliasResolveDefendsDepth asserts Resolve itself errors past MaxAliasDepth even
// when the table was mutated past its validation checks (defense in depth).
func TestAliasResolveDefendsDepth(t *testing.T) {
	// A store whose map was force-populated with an over-deep raw chain, bypassing
	// Validate/Set — mirroring a memory corruption or a future mutation path.
	s := &AliasStore{aliases: map[string]string{}}
	for i := 0; i < MaxAliasDepth+2; i++ {
		s.aliases[aliasName(i)] = aliasName(i + 1)
	}
	if _, _, err := s.Resolve(aliasName(0)); err == nil {
		t.Fatalf("expected depth-cap error, got nil")
	}
}

// TestAliasSetRedirects proves a hot Set reassigns an alias and the very next
// Resolve returns the NEW target.
func TestAliasSetRedirects(t *testing.T) {
	s, err := NewAliasStore(AliasRegistry{Aliases: []Alias{{Name: "prod", Target: "small"}}})
	if err != nil {
		t.Fatalf("NewAliasStore: %v", err)
	}
	if got, _, _ := s.Resolve("prod"); got != "small" {
		t.Fatalf("before Set: Resolve(prod) = %q, want small", got)
	}
	if err := s.Set("prod", "large"); err != nil {
		t.Fatalf("Set: %v", err)
	}
	got, isAlias, err := s.Resolve("prod")
	if err != nil {
		t.Fatalf("Resolve after Set: %v", err)
	}
	if !isAlias || got != "large" {
		t.Fatalf("after Set: Resolve(prod) = (%q, %v), want (large, true)", got, isAlias)
	}
	if !s.Remove("prod") {
		t.Fatal("Remove(prod) = false, want true")
	}
	if _, isAlias, _ := s.Resolve("prod"); isAlias {
		t.Fatal("after Remove: prod still resolves as an alias")
	}
	// A Set that would close a cycle against the LIVE table is refused.
	if err := s.Set("x", "y"); err != nil {
		t.Fatal(err)
	}
	if err := s.Set("y", "x"); err == nil {
		t.Fatal("expected live cycle rejection from Set, got nil")
	}
}

// TestAliasConcurrentResolveDuringSet is the race witness: many readers Resolve while
// a writer reassigns, under -race this must be clean.
func TestAliasConcurrentResolveDuringSet(t *testing.T) {
	s, err := NewAliasStore(AliasRegistry{Aliases: []Alias{{Name: "a", Target: "t0"}}})
	if err != nil {
		t.Fatalf("NewAliasStore: %v", err)
	}
	const readers = 16
	const iterations = 500
	var wg sync.WaitGroup
	stop := make(chan struct{})
	for i := 0; i < readers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
					if _, _, err := s.Resolve("a"); err != nil {
						t.Errorf("Resolve: %v", err)
						return
					}
					_ = s.List()
				}
			}
		}()
	}
	for i := 0; i < iterations; i++ {
		if err := s.Set("a", aliasName(i)); err != nil {
			t.Fatalf("Set: %v", err)
		}
	}
	close(stop)
	wg.Wait()
}

// TestAliasJSONRoundTrip covers JSON round-trip and DisallowUnknownFields rejection.
func TestAliasJSONRoundTrip(t *testing.T) {
	reg := DefaultAliases()
	parsed, err := ParseAliases(reg.JSON())
	if err != nil {
		t.Fatalf("ParseAliases(JSON): %v", err)
	}
	if len(parsed.Aliases) != len(reg.Aliases) {
		t.Fatalf("round-trip aliases = %d, want %d", len(parsed.Aliases), len(reg.Aliases))
	}
	if _, err := ParseAliases([]byte(`{"aliases":[{"name":"a","target":"b","bogus":1}]}`)); err == nil {
		t.Fatal("expected DisallowUnknownFields rejection, got nil")
	}
}

func aliasName(i int) string {
	return "n" + strconv.Itoa(i)
}