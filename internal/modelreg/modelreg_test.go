package modelreg

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// withCacheRoot points FAK_MODELS_DIR at a temp dir for the duration of a test, so
// Load reads a controlled registry.json and cache tree instead of the real user cache.
func withCacheRoot(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("FAK_MODELS_DIR", dir)
	return dir
}

func TestResolveHFURIPassesThrough(t *testing.T) {
	withCacheRoot(t)
	const uri = "hf://owner/repo/model.gguf"
	got, expanded := Resolve(uri)
	if got != uri || expanded {
		t.Fatalf("Resolve(%q) = (%q, %v); want (%q, false)", uri, got, expanded, uri)
	}
}

func TestResolveLocalPathPassesThrough(t *testing.T) {
	withCacheRoot(t)
	f := filepath.Join(t.TempDir(), "local.gguf")
	if err := os.WriteFile(f, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	got, expanded := Resolve(f)
	if got != f || expanded {
		t.Fatalf("Resolve(localpath) = (%q, %v); want (%q, false)", got, expanded, f)
	}
}

func TestResolveEmbeddedAlias(t *testing.T) {
	withCacheRoot(t)
	got, expanded := Resolve("smollm2")
	if !expanded {
		t.Fatalf("Resolve(smollm2) did not expand; got %q", got)
	}
	if want := Catalog["smollm2"]; got != want {
		t.Fatalf("Resolve(smollm2) = %q; want embedded target %q", got, want)
	}
}

func TestResolveUnknownReturnedUnchanged(t *testing.T) {
	withCacheRoot(t)
	got, expanded := Resolve("definitely-not-a-known-name")
	if expanded || got != "definitely-not-a-known-name" {
		t.Fatalf("Resolve(unknown) = (%q, %v); want unchanged, false", got, expanded)
	}
}

func TestUserOverlayWinsOverEmbedded(t *testing.T) {
	dir := withCacheRoot(t)
	// Override the embedded "smollm2" with a user target.
	const userTarget = "hf://me/my-smollm/custom.gguf"
	writeRegistry(t, dir, map[string]string{"smollm2": userTarget, "mine": "hf://me/mine/m.gguf"})

	r, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got, _ := r.Resolve("smollm2"); got != userTarget {
		t.Errorf("user overlay did not win: Resolve(smollm2) = %q; want %q", got, userTarget)
	}
	if got, expanded := r.Resolve("mine"); !expanded || got != "hf://me/mine/m.gguf" {
		t.Errorf("user-only alias not resolved: (%q, %v)", got, expanded)
	}
	// Source annotation must reflect the override.
	for _, e := range r.Entries() {
		if e.Name == "smollm2" && e.Source != "user" {
			t.Errorf("overridden smollm2 Source = %q; want user", e.Source)
		}
	}
}

func TestMalformedRegistryIsAnErrorButResolveFallsBack(t *testing.T) {
	dir := withCacheRoot(t)
	if err := os.WriteFile(filepath.Join(dir, registryFileName), []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(); err == nil {
		t.Fatal("Load with malformed registry.json: want error, got nil")
	}
	// Package-level Resolve must still resolve embedded names despite the bad file.
	got, expanded := Resolve("smollm2")
	if !expanded || got != Catalog["smollm2"] {
		t.Fatalf("Resolve fallback after malformed file = (%q, %v); want embedded target", got, expanded)
	}
}

func TestEntriesReportsLocalCacheStatus(t *testing.T) {
	dir := withCacheRoot(t)
	// Place a file at the cache path the embedded smollm2 hf:// ref maps to.
	r, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	var smol ListEntry
	for _, e := range r.Entries() {
		if e.Name == "smollm2" {
			smol = e
		}
	}
	if smol.Cached() {
		t.Fatal("smollm2 should not be cached in a fresh temp cache")
	}
	_ = dir // cache root is wired via FAK_MODELS_DIR; this asserts the not-cached default path
}

// TestCodingAliasesResolveAndAreFlagged covers the #1058 curation: each seeded
// Qwen2.5-Coder alias resolves to a concrete single-file hf:// GGUF, is flagged coding
// via IsCoding, and surfaces Coding=true in its Entries() row; a non-coder chat alias
// stays unflagged so the column actually discriminates.
func TestCodingAliasesResolveAndAreFlagged(t *testing.T) {
	withCacheRoot(t)
	coders := []string{"qwen2.5-coder:1.5b", "qwen2.5-coder:3b", "qwen2.5-coder:7b"}
	for _, name := range coders {
		got, expanded := Resolve(name)
		if !expanded {
			t.Errorf("Resolve(%q) did not expand; got %q", name, got)
		}
		if want := Catalog[name]; got != want || want == "" {
			t.Errorf("Resolve(%q) = %q; want embedded target %q", name, got, want)
		}
		if !IsCoding(name) {
			t.Errorf("IsCoding(%q) = false; want true", name)
		}
	}
	// A capable but non-coder chat alias must NOT be flagged coding.
	if IsCoding("qwen2.5:7b") {
		t.Error("IsCoding(qwen2.5:7b) = true; a non-coder chat model must not be flagged")
	}
	if IsCoding("definitely-not-a-known-name") {
		t.Error("IsCoding(unknown) = true; want false")
	}

	// The Entries() row carries the same flag for `fak ls`.
	r, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, e := range r.Entries() {
		if IsCoding(e.Name) && !e.Coding {
			t.Errorf("Entries(): %q is coding but row Coding=false", e.Name)
		}
		if !IsCoding(e.Name) && e.Coding {
			t.Errorf("Entries(): %q is not coding but row Coding=true", e.Name)
		}
		seen[e.Name] = true
	}
	for _, name := range coders {
		if !seen[name] {
			t.Errorf("Entries() omitted coding alias %q", name)
		}
	}
}

// TestDefaultLocalCodingAliasIsACuratedCoder pins the no-name default: it must be one of
// the curated coding aliases and resolve to a concrete embedded target, so the
// one-command `fak guard --local`/`--gguf` path (epic #1056) has a known-good model with
// no user knowledge of aliases.
func TestDefaultLocalCodingAliasIsACuratedCoder(t *testing.T) {
	withCacheRoot(t)
	if !IsCoding(DefaultLocalCodingAlias) {
		t.Fatalf("DefaultLocalCodingAlias %q is not flagged coding", DefaultLocalCodingAlias)
	}
	got, expanded := Resolve(DefaultLocalCodingAlias)
	if !expanded || got == "" || got == DefaultLocalCodingAlias {
		t.Fatalf("Resolve(default %q) = (%q, %v); want a concrete expanded target",
			DefaultLocalCodingAlias, got, expanded)
	}
}

// TestUserOverlayKeepsEmbeddedCodingFlag asserts that overriding a coding alias's TARGET
// via registry.json does not strip its coding nature — IsCoding checks the embedded set,
// so the flag tracks the model's identity, not the (now user-pointed) weights.
func TestUserOverlayKeepsEmbeddedCodingFlag(t *testing.T) {
	dir := withCacheRoot(t)
	writeRegistry(t, dir, map[string]string{
		"qwen2.5-coder:3b": "hf://me/my-coder/custom.gguf",
	})
	r, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range r.Entries() {
		if e.Name == "qwen2.5-coder:3b" {
			if e.Source != "user" {
				t.Errorf("overridden coder Source = %q; want user", e.Source)
			}
			if !e.Coding {
				t.Error("user override stripped the embedded coding flag; want kept")
			}
		}
	}
}

func writeRegistry(t *testing.T, dir string, m map[string]string) {
	t.Helper()
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, registryFileName), data, 0o644); err != nil {
		t.Fatal(err)
	}
}
