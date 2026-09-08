package hooks

import (
	"os"
	"path/filepath"
	"testing"
)

// TestVerbTierGate verifies the VERB_UNTIERED hygiene gate behavior:
// 1. Live tree in real repo has complete verb tier coverage.
// 2. Untiered verbs in cmd/fak/main.go or cmd/fak-dev/main.go are detected and reported.
// 3. Files missing or malformed fail open (ErrCouldNotRun).
// 4. Push scoping only blocks when cmd/fak/main.go, cmd/fak-dev/main.go, or internal/devindex/tiers.go is in the push delta.
// 5. Gate registration in HygieneGates() with PushScoped: true.
func TestVerbTierGate(t *testing.T) {
	t.Run("LiveTreeClean", func(t *testing.T) {
		root := repoRoot(t)
		tree, err := ReadTrackedTree(root)
		if err != nil {
			t.Skipf("ReadTrackedTree: %v", err)
		}
		findings, gerr := gateVerbTierTree(tree)
		if gerr != nil {
			t.Fatalf("gate error on live tree: %v", gerr)
		}
		if len(findings) != 0 {
			t.Fatalf("expected 0 VERB_UNTIERED findings on live tree, got %d: %+v", len(findings), findings)
		}
	})

	t.Run("Registration", func(t *testing.T) {
		check := HygieneGateByName("VERB_UNTIERED")
		if check == nil {
			t.Fatal("VERB_UNTIERED is not registered in HygieneGates()")
		}
		found := false
		for _, g := range HygieneGates() {
			if g.Name == "VERB_UNTIERED" {
				found = true
				if g.DefaultOff {
					t.Error("VERB_UNTIERED should not be DefaultOff")
				}
				if !g.PushScoped {
					t.Error("VERB_UNTIERED must have PushScoped: true")
				}
			}
		}
		if !found {
			t.Fatal("VERB_UNTIERED gate not found in HygieneGates()")
		}
	})

	t.Run("FailsOpenWhenSourceMissing", func(t *testing.T) {
		emptyTree := &TrackedTree{
			Root:      t.TempDir(),
			Paths:     []string{},
			fileCache: map[string]fileEntry{},
		}
		if _, err := gateVerbTierTree(emptyTree); err != ErrCouldNotRun {
			t.Fatalf("expected ErrCouldNotRun on empty tree, got %v", err)
		}
	})

	buildSynthTree := func(t *testing.T, mainContent, devContent, tiersContent string) *TrackedTree {
		t.Helper()
		root := t.TempDir()
		write := func(rel, content string) {
			p := filepath.Join(root, filepath.FromSlash(rel))
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
		}

		var paths []string
		if tiersContent != "" {
			write(verbTiersFile, tiersContent)
			paths = append(paths, verbTiersFile)
		}
		if mainContent != "" {
			write(mainCmdFile, mainContent)
			paths = append(paths, mainCmdFile)
		}
		if devContent != "" {
			write(devCmdFile, devContent)
			paths = append(paths, devCmdFile)
		}

		return &TrackedTree{
			Root:      root,
			Paths:     paths,
			fileCache: map[string]fileEntry{},
		}
	}

	synthTiers := `package devindex

type VerbTier string
const (
	TierFrontdoor VerbTier = "frontdoor"
	TierDev       VerbTier = "dev"
	TierHidden    VerbTier = "hidden"
)

var verbTiers = map[string]VerbTier{
	"run": TierFrontdoor,
	"test-verb": TierDev,
	"secret-gate": TierHidden,
}
`

	synthMainClean := `package main

import "os"

func main() {
	switch os.Args[1] {
	case "run":
		println("running")
	case "test-verb":
		println("testing")
	default:
		println("default")
	}
}
`

	synthMainWithUntiered := `package main

import "os"

func main() {
	switch os.Args[1] {
	case "run":
		println("running")
	case "test-verb":
		println("testing")
	case "unclassified-verb":
		println("unclassified")
	default:
		println("default")
	}
}
`

	synthDevWithUntiered := `package main

func run(argv []string) int {
	switch argv[0] {
	case "unclassified-dev-verb":
		return 0
	default:
		return 1
	}
}
`

	t.Run("CleanSyntheticTree", func(t *testing.T) {
		tree := buildSynthTree(t, synthMainClean, "", synthTiers)
		findings, err := gateVerbTierTree(tree)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(findings) != 0 {
			t.Fatalf("expected 0 findings, got %d: %+v", len(findings), findings)
		}
	})

	t.Run("UntieredMainVerbDetected", func(t *testing.T) {
		tree := buildSynthTree(t, synthMainWithUntiered, "", synthTiers)
		findings, err := gateVerbTierTree(tree)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(findings) != 1 {
			t.Fatalf("expected 1 finding, got %d: %+v", len(findings), findings)
		}
		if findings[0].Gate != "VERB_UNTIERED" {
			t.Errorf("expected Gate=VERB_UNTIERED, got %s", findings[0].Gate)
		}
		if findings[0].File != mainCmdFile {
			t.Errorf("expected File=%s, got %s", mainCmdFile, findings[0].File)
		}
		if findings[0].Advisory {
			t.Error("finding should be hard by default (not advisory)")
		}
	})

	t.Run("UntieredDevVerbDetected", func(t *testing.T) {
		tree := buildSynthTree(t, synthMainClean, synthDevWithUntiered, synthTiers)
		findings, err := gateVerbTierTree(tree)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(findings) != 1 {
			t.Fatalf("expected 1 finding, got %d: %+v", len(findings), findings)
		}
		if findings[0].Gate != "VERB_UNTIERED" {
			t.Errorf("expected Gate=VERB_UNTIERED, got %s", findings[0].Gate)
		}
		if findings[0].File != devCmdFile {
			t.Errorf("expected File=%s, got %s", devCmdFile, findings[0].File)
		}
	})

	t.Run("CaseLineAliasesInheritTier", func(t *testing.T) {
		mainWithAlias := `package main

import "os"

func main() {
	switch os.Args[1] {
	case "test-verb", "-t", "--test":
		println("testing")
	default:
		println("default")
	}
}
`
		tree := buildSynthTree(t, mainWithAlias, "", synthTiers)
		findings, err := gateVerbTierTree(tree)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(findings) != 0 {
			t.Fatalf("expected 0 findings with alias on same case line, got %d: %+v", len(findings), findings)
		}
	})

	t.Run("PushScoping", func(t *testing.T) {
		findings := []Finding{
			{Gate: "VERB_UNTIERED", File: mainCmdFile, Detail: "dispatched verb untiered"},
		}

		// When scoped is false (fallback), stays hard/blocking.
		fallback := ScopeVerbTierFindings(findings, nil, false)
		if fallback[0].Advisory {
			t.Errorf("unscoped fallback should remain blocking: %+v", fallback[0])
		}

		// When scoped is true and cmd/fak/main.go is touched, stays blocking.
		onMain := ScopeVerbTierFindings(findings, []string{mainCmdFile}, true)
		if onMain[0].Advisory {
			t.Errorf("push touching main.go should remain blocking: %+v", onMain[0])
		}

		// When scoped is true and cmd/fak-dev/main.go is touched, stays blocking.
		onDev := ScopeVerbTierFindings(findings, []string{devCmdFile}, true)
		if onDev[0].Advisory {
			t.Errorf("push touching fak-dev/main.go should remain blocking: %+v", onDev[0])
		}

		// When scoped is true and internal/devindex/tiers.go is touched, stays blocking.
		onTiers := ScopeVerbTierFindings(findings, []string{verbTiersFile}, true)
		if onTiers[0].Advisory {
			t.Errorf("push touching tiers.go should remain blocking: %+v", onTiers[0])
		}

		// When scoped is true and unrelated path is touched, demotes to advisory.
		unrelated := ScopeVerbTierFindings(findings, []string{"docs/readme.md"}, true)
		if !unrelated[0].Advisory {
			t.Errorf("push not touching CLI dispatch or tier tables should demote to advisory: %+v", unrelated[0])
		}

		// Verify ScopeTierDeclaredFindings integration
		viaTierDeclared := ScopeTierDeclaredFindings(findings, []string{"docs/readme.md"}, true)
		if !viaTierDeclared[0].Advisory {
			t.Errorf("ScopeTierDeclaredFindings should also scope VERB_UNTIERED: %+v", viaTierDeclared[0])
		}
	})
}
