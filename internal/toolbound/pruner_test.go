package toolbound

import (
	"reflect"
	"testing"
)

func TestPruneMultipleIgnoredDirs(t *testing.T) {
	dp := NewDirectoryPruner("testdata", "vendor", "build")

	paths := []string{
		"testdata/sample.json",
		"vendor/github.com/pkg/file.go",
		"build/bin/output",
		"internal/toolbound/pruner.go",
		"cmd/fak/main.go",
		"README.md",
	}

	expected := []string{
		"internal/toolbound/pruner.go",
		"cmd/fak/main.go",
		"README.md",
	}

	got := dp.Prune(paths)
	if !reflect.DeepEqual(got, expected) {
		t.Fatalf("dp.Prune got %v, want %v", got, expected)
	}

	// Verify standalone helper PruneSearchPaths
	gotHelper := PruneSearchPaths(paths, []string{"testdata", "vendor", "build"})
	if !reflect.DeepEqual(gotHelper, expected) {
		t.Fatalf("PruneSearchPaths got %v, want %v", gotHelper, expected)
	}

	// Check IsIgnored directly
	for _, p := range []string{"testdata/sample.json", "vendor/github.com/pkg/file.go", "build/bin/output"} {
		if !dp.IsIgnored(p) {
			t.Errorf("expected IsIgnored(%q) == true, got false", p)
		}
	}
	for _, p := range expected {
		if dp.IsIgnored(p) {
			t.Errorf("expected IsIgnored(%q) == false, got true", p)
		}
	}
}

func TestPathNormalization(t *testing.T) {
	dp := NewDirectoryPruner("testdata", "vendor", "build")

	// Windows backslashes in paths
	winPaths := []string{
		`vendor\github.com\pkg\file.go`,
		`testdata\fixtures\sample.json`,
		`build\bin\output.exe`,
		`internal\toolbound\pruner.go`,
		`cmd\fak\main.go`,
	}

	expected := []string{
		`internal\toolbound\pruner.go`,
		`cmd\fak\main.go`,
	}

	got := dp.Prune(winPaths)
	if !reflect.DeepEqual(got, expected) {
		t.Fatalf("Prune with Windows paths got %v, want %v", got, expected)
	}

	// Windows backslashes in ignoredDirs configuration
	winDP := NewDirectoryPruner(`internal\testdata`, `pkg\vendor`)
	nestedPaths := []string{
		"internal/testdata/file.txt",
		`internal\testdata\file.txt`,
		"pkg/vendor/module.go",
		`pkg\vendor\module.go`,
		"internal/toolbound/pruner.go",
		"pkg/other/module.go",
	}

	nestedExpected := []string{
		"internal/toolbound/pruner.go",
		"pkg/other/module.go",
	}

	nestedGot := winDP.Prune(nestedPaths)
	if !reflect.DeepEqual(nestedGot, nestedExpected) {
		t.Fatalf("Prune with Windows-configured dirs got %v, want %v", nestedGot, nestedExpected)
	}

	// Path with ./ and leading slash
	if !dp.IsIgnored("./vendor/pkg/foo.go") {
		t.Errorf("expected ./vendor/pkg/foo.go to be ignored")
	}
	if !dp.IsIgnored(".\\vendor\\pkg\\foo.go") {
		t.Errorf("expected .\\vendor\\pkg\\foo.go to be ignored")
	}
	if !dp.IsIgnored("/vendor/pkg/foo.go") {
		t.Errorf("expected /vendor/pkg/foo.go to be ignored")
	}
}

func TestEmptyIgnoreList(t *testing.T) {
	dp := NewDirectoryPruner()

	paths := []string{
		"main.go",
		"vendor/pkg/foo.go",
		"testdata/sample.json",
	}

	got := dp.Prune(paths)
	if !reflect.DeepEqual(got, paths) {
		t.Fatalf("expected all paths preserved with empty ignoredDirs, got %v", got)
	}

	// Standalone helper with nil or empty
	if gotNil := PruneSearchPaths(paths, nil); !reflect.DeepEqual(gotNil, paths) {
		t.Fatalf("PruneSearchPaths with nil ignoredDirs got %v, want %v", gotNil, paths)
	}
	if gotEmpty := PruneSearchPaths(paths, []string{}); !reflect.DeepEqual(gotEmpty, paths) {
		t.Fatalf("PruneSearchPaths with empty ignoredDirs got %v, want %v", gotEmpty, paths)
	}

	// IsIgnored with empty ignore list
	for _, p := range paths {
		if dp.IsIgnored(p) {
			t.Errorf("expected IsIgnored(%q) == false with empty pruner", p)
		}
	}
}

func TestCleanEdgeCases(t *testing.T) {
	dp := NewDirectoryPruner("testdata", "vendor", "build", "internal/testdata")

	t.Run("root paths", func(t *testing.T) {
		rootPaths := []string{
			"main.go",
			"README.md",
			"./go.mod",
			"/main.go",
			"testdata.go",
			"vendor.json",
			"build.sh",
		}
		for _, p := range rootPaths {
			if dp.IsIgnored(p) {
				t.Errorf("root path %q should not be ignored", p)
			}
		}

		// Root path directories like "." or "/"
		if dp.IsIgnored(".") {
			t.Errorf("\".\" should not be ignored")
		}
		if dp.IsIgnored("/") {
			t.Errorf("\"/\" should not be ignored")
		}
		if dp.IsIgnored("") {
			t.Errorf("\"\" should not be ignored")
		}
	})

	t.Run("subdirectories and sibling prefixes", func(t *testing.T) {
		// Sibling directories with matching prefix but not matching /
		siblings := []string{
			"testdata_old/sample.json",
			"vendor2/file.go",
			"build-artifacts/bin",
			"internal/testdata_fixtures/sample.json",
		}
		for _, p := range siblings {
			if dp.IsIgnored(p) {
				t.Errorf("sibling directory path %q should not be ignored", p)
			}
		}

		// Nested subdirectory ignore
		if !dp.IsIgnored("internal/testdata/file.json") {
			t.Errorf("internal/testdata/file.json should be ignored")
		}
		if !dp.IsIgnored("internal/testdata/sub/nested.json") {
			t.Errorf("internal/testdata/sub/nested.json should be ignored")
		}
		if dp.IsIgnored("internal/other/file.json") {
			t.Errorf("internal/other/file.json should not be ignored")
		}
	})

	t.Run("exact match", func(t *testing.T) {
		exactMatches := []string{
			"testdata",
			"testdata/",
			"vendor",
			"vendor/",
			"build",
			"build/",
			"internal/testdata",
			"internal/testdata/",
			`testdata\`,
			`vendor\`,
		}
		for _, p := range exactMatches {
			if !dp.IsIgnored(p) {
				t.Errorf("exact directory match %q should be ignored", p)
			}
		}
	})

	t.Run("nil and empty paths", func(t *testing.T) {
		if got := dp.Prune(nil); got != nil {
			t.Errorf("Prune(nil) expected nil, got %v", got)
		}
		if got := dp.Prune([]string{}); got == nil || len(got) != 0 {
			t.Errorf("Prune([]string{}) expected empty slice, got %v", got)
		}
	})
}

func TestAddIgnored(t *testing.T) {
	dp := NewDirectoryPruner()
	if dp.IsIgnored("vendor/foo.go") {
		t.Fatalf("expected vendor/foo.go not ignored before AddIgnored")
	}

	dp.AddIgnored("vendor")
	if !dp.IsIgnored("vendor/foo.go") {
		t.Fatalf("expected vendor/foo.go ignored after AddIgnored(vendor)")
	}

	// Duplicate addition should be idempotent
	dp.AddIgnored("vendor", "vendor/", `vendor\`)
	ignored := dp.Ignored()
	if len(ignored) != 1 {
		t.Fatalf("expected 1 ignored dir after duplicate AddIgnored, got %d: %v", len(ignored), ignored)
	}

	// Add second directory and whitespace/empty values
	dp.AddIgnored("", "  ", ".", "build")
	if !dp.IsIgnored("build/bin") {
		t.Fatalf("expected build/bin ignored after AddIgnored(build)")
	}
	ignored = dp.Ignored()
	if len(ignored) != 2 {
		t.Fatalf("expected 2 ignored dirs, got %d: %v", len(ignored), ignored)
	}
}
