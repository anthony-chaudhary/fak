// Package modelreg is the friendly-name → model-ref registry that lets a user
// say `fak run qwen2.5:7b` or `fak serve --gguf smollm2` instead of typing a full
// hf:// URI or hunting for a local .gguf path. It is the small piece that makes
// fak's run-by-name surface feel like Ollama/llama.cpp, where you pull and run a
// model by a short name.
//
// A registry is the merge of two layers:
//
//   - an embedded default catalog (Catalog) compiled into the binary, so a handful
//     of names resolve on a clean install with no config; the seeds are hf:// refs
//     the repo already exercises, so they are known-good.
//   - a user-writable registry.json under the model cache root
//     (<cache>/fak-models/registry.json), which OVERRIDES and EXTENDS the embedded
//     catalog. A user alias with the same name as an embedded one wins.
//
// Resolve is the single entry point every model-ref-accepting surface (run, serve
// --gguf, model load) routes through first: a bare local path or an hf:// URI passes
// through untouched, and a known alias expands to its target ref.
package modelreg

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/hfhub"
)

// Catalog is the embedded default alias → target map. Targets are hf:// URIs the
// repo already loads elsewhere (so a name here is known-resolvable), or could be a
// bare path for a vendored model. Keep this SMALL and curated — it is the day-one
// "names that just work" set, not an exhaustive index. Users extend it via
// registry.json. The keys follow the Ollama-style `family:size` spelling where it
// reads naturally, plus a couple of bare names.
// Every target below was verified fetchable (a public repo, an exact single-file
// GGUF that resolves with no HF token) before being seeded, so a clean install can
// `fak pull <name>` without auth. Re-verify with a HEAD on the resolve URL if you add
// one; HF returns 401 (not 404) for a nonexistent repo, so a typo'd repo path silently
// looks like an auth wall.
var Catalog = map[string]string{
	// SmolLM2-135M-Instruct — the smallest "it actually chats" GGUF, ideal for
	// `fak run smollm2 "hi"` on a laptop with no GPU. bartowski is the canonical
	// public GGUF re-publisher (the base HuggingFaceTB repo ships no GGUF).
	"smollm2":      "hf://bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf",
	"smollm2:135m": "hf://bartowski/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf",
	// Qwen2.5 instruct GGUFs: a 1.5B Q8 for a quick CPU run and a 7B Q4_K_M
	// (single file, ~4.7 GB) for a more capable laptop/GPU run.
	"qwen2.5:1.5b": "hf://mradermacher/Qwen2.5-1.5B-GGUF/Qwen2.5-1.5B.Q8_0.gguf",
	"qwen2.5:7b":   "hf://bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
}

// Entry is one resolved alias: the name a user types and the target it expands to.
type Entry struct {
	Name   string // the friendly name, e.g. "qwen2.5:7b"
	Target string // the target model ref: an hf:// URI or a local path
	Source string // "embedded" or "user" — where the mapping came from
}

// registryFileName is the user-writable overlay under the cache root.
const registryFileName = "registry.json"

// Registry is the merged view of the embedded catalog and the user overlay.
type Registry struct {
	// entries is keyed by alias name; a user entry shadows an embedded one.
	entries map[string]Entry
	// cacheDir is the model cache root (<cache>/fak-models), the parent of both the
	// hub/ download tree and registry.json. Empty until Load resolves it.
	cacheDir string
}

// Load builds the registry by overlaying the user registry.json (if present) on top
// of the embedded Catalog. A missing or unreadable user file is not an error — the
// embedded catalog alone is a valid registry. A malformed user file IS an error, so
// a typo is surfaced rather than silently dropping the user's aliases.
func Load() (*Registry, error) {
	r := &Registry{entries: make(map[string]Entry, len(Catalog)), cacheDir: cacheRoot()}
	for name, target := range Catalog {
		r.entries[name] = Entry{Name: name, Target: target, Source: "embedded"}
	}
	path := filepath.Join(r.cacheDir, registryFileName)
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return r, nil
		}
		return nil, fmt.Errorf("modelreg: read %s: %w", path, err)
	}
	var user map[string]string
	if err := json.Unmarshal(data, &user); err != nil {
		return nil, fmt.Errorf("modelreg: parse %s: %w", path, err)
	}
	for name, target := range user {
		r.entries[name] = Entry{Name: name, Target: strings.TrimSpace(target), Source: "user"}
	}
	return r, nil
}

// cacheRoot returns the model cache root (<cache>/fak-models), honoring the same
// FAK_MODELS_DIR override hfhub uses so the registry sits beside the hub/ tree.
func cacheRoot() string {
	if dir := os.Getenv("FAK_MODELS_DIR"); dir != "" {
		return dir
	}
	if dir, err := os.UserCacheDir(); err == nil {
		return filepath.Join(dir, "fak-models")
	}
	return filepath.Join(".", ".fak-models")
}

// Resolve maps a user-supplied model ref to a concrete target ref. The rule, in
// order:
//
//  1. a bare hf:// URI passes through untouched (already concrete);
//  2. a string that names an existing file on disk passes through untouched (a
//     local .gguf path the user typed);
//  3. a known alias expands to its target;
//  4. otherwise the input is returned unchanged so the caller's own loader can try
//     it (and produce its own not-found error) — Resolve never invents a ref.
//
// The bool reports whether an alias expansion happened, so a caller can log "qwen2.5:7b
// → hf://…". A nil receiver resolves against the embedded catalog only.
func (r *Registry) Resolve(ref string) (string, bool) {
	ref = strings.TrimSpace(ref)
	if hfhub.IsURI(ref) {
		return ref, false
	}
	if _, err := os.Stat(ref); err == nil {
		return ref, false
	}
	if e, ok := r.lookup(ref); ok {
		return e.Target, true
	}
	return ref, false
}

func (r *Registry) lookup(name string) (Entry, bool) {
	if r == nil {
		t, ok := Catalog[name]
		return Entry{Name: name, Target: t, Source: "embedded"}, ok
	}
	e, ok := r.entries[name]
	return e, ok
}

// Resolve is the package-level convenience over a freshly loaded registry, for the
// common one-shot call site (serve --gguf, model load). It falls back to the embedded
// catalog if the user overlay cannot be loaded, so a malformed registry.json never
// blocks resolving an embedded name.
func Resolve(ref string) (string, bool) {
	r, err := Load()
	if err != nil {
		var embedded *Registry // nil → embedded-only path in (*Registry).lookup
		return embedded.Resolve(ref)
	}
	return r.Resolve(ref)
}

// Entries returns the merged alias set, sorted by name, for `fak ls`. Each entry is
// annotated with whether its target is locally cached (LocalPath non-empty) and the
// cached file size in bytes.
func (r *Registry) Entries() []ListEntry {
	out := make([]ListEntry, 0, len(r.entries))
	for _, e := range r.entries {
		le := ListEntry{Entry: e}
		if hfhub.IsURI(e.Target) {
			if ref, err := hfhub.ParseURI(e.Target); err == nil && ref.File != "" {
				c := hfhub.NewClient()
				p := c.CachePath(ref)
				if fi, err := os.Stat(p); err == nil && !fi.IsDir() {
					le.LocalPath = p
					le.SizeBytes = fi.Size()
				}
			}
		} else if fi, err := os.Stat(e.Target); err == nil && !fi.IsDir() {
			// A bare-path alias: cached iff the file exists.
			le.LocalPath = e.Target
			le.SizeBytes = fi.Size()
		}
		out = append(out, le)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// ListEntry is one row of `fak ls`: an alias plus its local-cache status.
type ListEntry struct {
	Entry
	LocalPath string // non-empty if the target is downloaded/present locally
	SizeBytes int64  // size of the local file, when LocalPath is set
}

// Cached reports whether this entry's target is present on disk.
func (le ListEntry) Cached() bool { return le.LocalPath != "" }
