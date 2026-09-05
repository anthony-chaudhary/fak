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

func TestResolveGLM53FlashAliases(t *testing.T) {
	withCacheRoot(t)
	cases := []struct {
		alias string
		want  string
	}{
		{"glm-5.3-flash", "hf://zai-org/GLM-5.3-Flash@04c4e9e95c5da8862dced7e5056455116f83a7e0"},
		{"glm-5.3-flash:bf16", "hf://zai-org/GLM-5.3-Flash-BF16@f12e0fe1f6b2ea274c11a569582edfd99d993c5e"},
		{"glm53", "hf://zai-org/GLM-5.3-Flash@04c4e9e95c5da8862dced7e5056455116f83a7e0"},
	}

	for _, tc := range cases {
		t.Run(tc.alias, func(t *testing.T) {
			got, expanded := Resolve(tc.alias)
			if !expanded || got != tc.want {
				t.Fatalf("Resolve(%q) = (%q, %v); want (%q, true)", tc.alias, got, expanded, tc.want)
			}
		})
	}
}

func TestResolveQwen25HalfBForGuard(t *testing.T) {
	withCacheRoot(t)
	const want = "hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf"
	got, expanded := Resolve("qwen2.5:0.5b")
	if !expanded || got != want {
		t.Fatalf("Resolve(qwen2.5:0.5b) = (%q, %v); want (%q, true)", got, expanded, want)
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

// TestNarratorAliasFailsClosed pins the Qwen3.6 narrator contract after its
// downloadable source was independently witnessed returning HTTP 404.
func TestNarratorAliasFailsClosed(t *testing.T) {
	withCacheRoot(t)
	const alias = "qwen3.6-27b"
	got, expanded := Resolve(alias)
	if expanded || got != alias {
		t.Fatalf("Resolve(%q) = %q, %v; want unresolved alias", alias, got, expanded)
	}
	if IsCoding(alias) {
		t.Errorf("IsCoding(%q) = true; the narrator reasoning model must not be flagged coding", alias)
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

// TestResolveNormalizesDashesToColons covers #1115: a user naturally types
// "qwen2.5-1.5b" but the registry uses "qwen2.5:1.5b". The normalization step
// treats both forms as equivalent for lookup, accepting the more intuitive dash
// separator without breaking the canonical colon form.
func TestResolveNormalizesDashesToColons(t *testing.T) {
	withCacheRoot(t)

	cases := []struct {
		name string
		want string
	}{
		{"qwen2.5-0.5b", "qwen2.5:0.5b"},
		{"qwen2.5-1.5b", "qwen2.5:1.5b"},
		{"qwen2.5-7b", "qwen2.5:7b"},
		{"qwen2.5-coder-1.5b", "qwen2.5-coder:1.5b"},
		{"qwen2.5-coder-3b", "qwen2.5-coder:3b"},
		{"qwen2.5-coder-7b", "qwen2.5-coder:7b"},
		{"smollm2-135m", "smollm2:135m"},
	}

	for _, tc := range cases {
		got, expanded := Resolve(tc.name)
		if !expanded {
			t.Errorf("Resolve(%q) did not expand; got %q", tc.name, got)
			continue
		}
		if want := Catalog[tc.want]; got != want {
			t.Errorf("Resolve(%q) = %q; want embedded target for %q (which is %q)",
				tc.name, got, tc.want, want)
		}
	}
	// The canonical colon form must still work.
	got, expanded := Resolve("qwen2.5:1.5b")
	if !expanded || got != Catalog["qwen2.5:1.5b"] {
		t.Errorf("Resolve(qwen2.5:1.5b) = (%q, %v); want expanded to embedded target",
			got, expanded)
	}
}

func TestQwen36AliasFailsClosedWithExactStagingContract(t *testing.T) {
	if got, ok := Resolve("qwen3.6-27b"); ok || got != "qwen3.6-27b" {
		t.Fatalf("Resolve(qwen3.6-27b) = %q, %v; want unresolved alias", got, ok)
	}

	got, ok := Blocked(" QWEN3.6-27B ")
	if !ok {
		t.Fatal("qwen3.6-27b missing fail-closed contract")
	}
	if got.Filename != "Qwen3.6-27B-Q4_K_M.gguf" || got.Bytes != 16547398784 || got.SHA256 != "33625d8dc3a5dd8d88c324d47db58561b11f7072816287078bfe58b4c55782f9" {
		t.Fatalf("Blocked(qwen3.6-27b) = %+v; want exact pinned artifact", got)
	}
	if got.Reason == "" {
		t.Fatal("blocked source must explain why the registry refuses it")
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

func TestEmbeddedQwen38Aliases(t *testing.T) {
	want := map[string]string{
		"qwen38":              "hf://unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/Qwen3.8-27B-Q4_K_M.gguf",
		"qwen38:27b":          "hf://unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/Qwen3.8-27B-Q4_K_M.gguf",
		"qwen38:27b-fp8":      "hf://Qwen/Qwen3.8-27B-FP8",
		"qwen38:27b-q2k":      "hf://unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q2_K_XL.gguf",
		"qwen38:27b-ud-q2kxl": "hf://unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q2_K_XL.gguf",
	}
	for alias, target := range want {
		t.Run(alias, func(t *testing.T) {
			if got := Catalog[alias]; got != target {
				t.Fatalf("Catalog[%q] = %q, want %q", alias, got, target)
			}
			if !IsCoding(alias) {
				t.Fatalf("Qwen3.8 alias %q must be marked coding-capable", alias)
			}
		})
	}
}

func TestQwen38IsFirstClassDefault(t *testing.T) {
	withCacheRoot(t)
	if DefaultAlias != "qwen38:27b" {
		t.Fatalf("DefaultAlias = %q, want qwen38:27b", DefaultAlias)
	}
	want := "hf://unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe/Qwen3.8-27B-Q4_K_M.gguf"
	if got := DefaultRef(); got != want {
		t.Fatalf("DefaultRef() = %q, want %q", got, want)
	}
	got, expanded := Resolve("default")
	if !expanded || got != want {
		t.Fatalf("Resolve(default) = %q, %v; want %q, true", got, expanded, want)
	}
	if !IsCoding(DefaultAlias) {
		t.Fatal("default model must retain coding/tool capability")
	}
}
