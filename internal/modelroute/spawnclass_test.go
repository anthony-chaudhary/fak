package modelroute

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func spawnClassRoster(scs ...SpawnClass) Roster {
	return Roster{
		Accounts:     []Account{{ID: "laptop", Kind: KindLocal, BaseURL: "http://127.0.0.1:11434/v1"}},
		SpawnClasses: scs,
	}
}

// The declaration is the whole point: an operator says what their agent types do, and
// that is the only thing allowed to give a spawn a class.
func TestSpawnClassForResolvesAnOperatorsDeclaration(t *testing.T) {
	r := spawnClassRoster(
		SpawnClass{Type: "explore", Class: ClassRoutine},
		SpawnClass{Type: "code-reviewer", Class: ClassNormalImpl},
		SpawnClass{Type: "release-cutter", Class: ClassSecurityRelease},
	)
	for _, tc := range []struct {
		name string
		typ  string
		want WorkClass
	}{
		{"a declared type resolves", "explore", ClassRoutine},
		{"so does another", "code-reviewer", ClassNormalImpl},
		{"including the strictest floor", "release-cutter", ClassSecurityRelease},
		// A roster is hand-written, so the token is normalized the way every other
		// declared token in this package is.
		{"case does not matter", "EXPLORE", ClassRoutine},
		{"nor does surrounding space", "  explore\t", ClassRoutine},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := r.SpawnClassFor(tc.typ)
			if !ok || got != tc.want {
				t.Errorf("SpawnClassFor(%q) = %q/%v, want %q/true", tc.typ, got, ok, tc.want)
			}
		})
	}
}

// FAIL-CLOSED. Every way a declaration can be absent must read as absent, because
// PlaceSpawn refuses an empty class and that refusal is the floor. The cost of an
// omission is that the spawn keeps the placement it has today; the cost of a default
// would be a sub-agent on a cheap rung nobody put it on.
func TestAnUndeclaredSpawnTypeResolvesToNothing(t *testing.T) {
	r := spawnClassRoster(SpawnClass{Type: "explore", Class: ClassRoutine})
	for _, tc := range []struct {
		name string
		typ  string
	}{
		{"a type nobody declared", "code-reviewer"},
		{"an empty type", ""},
		{"whitespace only", "   "},
		// Never a prefix and never a glob: "explore" must not answer for a type whose
		// name merely starts the same way.
		{"a prefix of a declared type", "expl"},
		{"a type a declared one is a prefix of", "explore-and-delete"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got, ok := r.SpawnClassFor(tc.typ); ok {
				t.Errorf("SpawnClassFor(%q) = %q/true, want no declaration", tc.typ, got)
			}
		})
	}
	// A roster with no spawn_classes block at all is the common case and must behave
	// identically to one whose types simply do not match.
	empty := spawnClassRoster()
	if _, ok := empty.SpawnClassFor("explore"); ok {
		t.Error("a roster declaring no spawn classes must resolve none")
	}
}

// A Roster built in code can carry a class token Validate never saw. An unrecognized
// class must read as "nothing was declared" rather than flowing onward as a WorkClass
// that PolicyFor does not know — the same rule ClassOf applies to a mistyped label.
func TestAnUnrecognizedClassIsNotADeclaration(t *testing.T) {
	r := spawnClassRoster(SpawnClass{Type: "explore", Class: WorkClass("routien")})
	if got, ok := r.SpawnClassFor("explore"); ok {
		t.Errorf("a typo'd class resolved to %q/true; a typo is not a declaration", got)
	}
	// And the empty class likewise — an entry that declares nothing declares nothing.
	blank := spawnClassRoster(SpawnClass{Type: "explore"})
	if _, ok := blank.SpawnClassFor("explore"); ok {
		t.Error("an entry with no class must not read as a declaration")
	}
}

// The load-bearing validation property: a bad entry is LOUD. A silently-dropped
// declaration looks identical to a correct one from the operator's side while behaving
// like an undeclared one, which is the failure that leaves nobody pointing at the typo.
func TestValidateRefusesAMalformedSpawnClass(t *testing.T) {
	for _, tc := range []struct {
		name string
		scs  []SpawnClass
		want string
	}{
		{"an empty type", []SpawnClass{{Class: ClassRoutine}}, "empty type"},
		{"an empty class", []SpawnClass{{Type: "explore"}}, "empty class"},
		{"an unknown class", []SpawnClass{{Type: "explore", Class: "cheap"}}, "unknown work class"},
		{"a duplicate type", []SpawnClass{
			{Type: "explore", Class: ClassRoutine},
			{Type: "Explore ", Class: ClassNormalImpl},
		}, "duplicate spawn class"},
		{"a type carrying a route delimiter", []SpawnClass{{Type: "a/b", Class: ClassRoutine}}, "route delimiter"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := spawnClassRoster(tc.scs...).Validate()
			if err == nil {
				t.Fatalf("Validate accepted %+v", tc.scs)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("Validate error = %v, want it to mention %q", err, tc.want)
			}
		})
	}
	// The unknown-class message must name the actual options, so a typo is answered
	// with the vocabulary rather than a bare rejection.
	err := spawnClassRoster(SpawnClass{Type: "explore", Class: "cheap"}).Validate()
	if err == nil || !strings.Contains(err.Error(), string(ClassRoutine)) {
		t.Errorf("the unknown-class error must render the vocabulary: %v", err)
	}
}

// A well-formed declaration passes, and a roster that declares none is unchanged — the
// block is additive, so no existing roster starts failing to load.
func TestValidateAcceptsADeclaredSpawnClassAndAnAbsentBlock(t *testing.T) {
	if err := spawnClassRoster(SpawnClass{Type: "explore", Class: ClassRoutine}).Validate(); err != nil {
		t.Errorf("a well-formed spawn class was refused: %v", err)
	}
	if err := spawnClassRoster().Validate(); err != nil {
		t.Errorf("a roster declaring no spawn classes was refused: %v", err)
	}
	// Absence stays absent on the wire: a pre-track-E roster round-trips byte-identically.
	b, err := json.Marshal(spawnClassRoster())
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "spawn_classes") {
		t.Errorf("an undeclared block must not serialize: %s", b)
	}
}

// The block has to be writable in the file a user is actually handed, which is a different
// claim from "the struct round-trips" — the shipped example is parsed with
// DisallowUnknownFields, so a key spelled wrong here fails loudly rather than being
// ignored. It also keeps the example honest about the whole vocabulary: an operator
// copying it can see what each of the four classes looks like as a declaration.
func TestTheShippedExampleRosterDeclaresItsSpawnClasses(t *testing.T) {
	path := filepath.Join("..", "..", "examples", "model-accounts.example.json")
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	r, err := ParseRoster(b) // ParseRoster validates, so a malformed entry fails here
	if err != nil {
		t.Fatalf("the shipped example roster must parse and validate: %v", err)
	}
	seen := map[WorkClass]bool{}
	for _, sc := range r.SpawnClasses {
		got, ok := r.SpawnClassFor(sc.Type)
		if !ok {
			t.Fatalf("the example declares type %q but it does not resolve", sc.Type)
		}
		if got != sc.Class {
			t.Fatalf("type %q resolved to %q, want the declared %q", sc.Type, got, sc.Class)
		}
		seen[got] = true
	}
	for _, want := range []WorkClass{ClassRoutine, ClassNormalImpl, ClassUltraHard, ClassSecurityRelease} {
		if !seen[want] {
			t.Errorf("the example roster should show class %q at least once", want)
		}
	}
	// And a type the example does not name stays undeclared, so copying the file does
	// not silently classify a fleet's own agent types.
	if _, ok := r.SpawnClassFor("some-fleets-own-agent-type"); ok {
		t.Error("an agent type the example never names must not resolve")
	}
}

// The declaration composes with the decision it exists to feed. This is the seam track E
// was blocked on: a declared type yields a class, and that class is what PlaceSpawn
// requires — while an undeclared one still hits the refusal, unchanged.
func TestTheDeclarationIsWhatUnblocksPlaceSpawn(t *testing.T) {
	r := spawnClassRoster(SpawnClass{Type: "explore", Class: ClassRoutine})
	class, ok := r.SpawnClassFor("explore")
	if !ok {
		t.Fatal("the declared type must resolve")
	}
	if _, err := r.PlaceSpawn(Placement{}, class, nil); err != nil &&
		strings.Contains(err.Error(), "requires a work class") {
		t.Errorf("a declared class must satisfy PlaceSpawn's class requirement: %v", err)
	}
	// The undeclared path still refuses, and that refusal is the floor this whole
	// declaration exists to reach honestly rather than to bypass.
	undeclared, _ := r.SpawnClassFor("code-reviewer")
	_, err := r.PlaceSpawn(Placement{}, undeclared, nil)
	if err == nil || !strings.Contains(err.Error(), "requires a work class") {
		t.Errorf("an undeclared spawn must still be refused a placement, got err=%v", err)
	}
}
