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
	// Qwen2.5-Coder instruct GGUFs — the CODING-tuned, TOOL-CALL-capable tier the
	// `fak guard --local`/`--gguf` coding loop wants (#1058, epic #1056). They emit the
	// Qwen2.5 == Hermes `<tool_call>` dialect the in-kernel planner already lifts into
	// structured tool calls, so a small one of these can actually DRIVE a coding harness,
	// not just chat. Three rungs trade footprint against tool-call reliability:
	//   - 1.5B Q4_K_M (~0.99 GB) — the FLOOR that still tool-calls, for a tiny box.
	//   - 3B  Q4_K_M (~1.93 GB) — the curated DEFAULT (see DefaultLocalCodingAlias):
	//     smallest size that reliably emits well-formed multi-call turns on CPU.
	//   - 7B  Q4_K_M (~4.68 GB) — the most capable laptop/GPU rung.
	// bartowski is the canonical public GGUF re-publisher already used above. Each repo
	// returned HTTP 200 unauthenticated and each filename below was confirmed present in
	// its -GGUF repo's file tree before seeding (HF returns 401 not 404 for a missing
	// repo, so a 200 + a tree hit is the real fetchability witness; re-verify on edit).
	"qwen2.5-coder:1.5b": "hf://bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
	"qwen2.5-coder:3b":   "hf://bartowski/Qwen2.5-Coder-3B-Instruct-GGUF/Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf",
	"qwen2.5-coder:7b":   "hf://bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
	// Ornith 1.0 — DeepReinforce's MIT-licensed Qwen3.5-family agentic-coding models
	// (released 2026-06-25, HF org deepreinforce-ai; the collection is exactly 7 public
	// repos — 9B/35B/397B + GGUF/FP8 siblings, NO 31B). Bare "ornith" and "ornith:9b-gguf"
	// seed the laptop-runnable 9B Q4_K_M single-file GGUF — the "just works" default, like
	// smollm2 above. The ":9b/:35b/:397b" aliases point at the full-precision safetensors
	// model repos and the "-fp8" aliases at the FP8 siblings, listed for `fak ls`
	// discoverability. NOTE: the in-kernel FP8 compressed-tensors path is not yet parsed
	// (epic #1026 child #F), so the "-fp8" targets resolve but do not yet load in-kernel.
	// Every repo below returned HTTP 200 and each seeded GGUF filename was confirmed present
	// in its -GGUF repo before seeding (re-verify with a HEAD on the resolve URL if you edit).
	"ornith":          "hf://deepreinforce-ai/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf",
	"ornith:9b-gguf":  "hf://deepreinforce-ai/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf",
	"ornith:9b":       "hf://deepreinforce-ai/Ornith-1.0-9B",
	"ornith:35b":      "hf://deepreinforce-ai/Ornith-1.0-35B",
	"ornith:35b-gguf": "hf://deepreinforce-ai/Ornith-1.0-35B-GGUF/ornith-1.0-35b-Q4_K_M.gguf",
	"ornith:35b-fp8":  "hf://deepreinforce-ai/Ornith-1.0-35B-FP8",
	"ornith:397b":     "hf://deepreinforce-ai/Ornith-1.0-397B",
	"ornith:397b-fp8": "hf://deepreinforce-ai/Ornith-1.0-397B-FP8",
}

// codingAliases is the set of Catalog aliases that are CODING-tuned and emit the
// `<tool_call>` dialect the in-kernel planner lifts — i.e. the ones that can actually
// drive a coding harness agentically, not just chat (#1058). `fak ls` flags these so a
// user picking a model for `fak guard --local`/`--gguf -- claude` knows which names are
// the coding loop's targets. It is intentionally a small explicit set, not a name-pattern
// heuristic: "qwen2.5:7b" is a fine chat model but is NOT seeded here because the coding
// loop wants the Coder-tuned weights. Keep it in sync with the Qwen2.5-Coder seeds above.
var codingAliases = map[string]bool{
	"qwen2.5-coder:1.5b": true,
	"qwen2.5-coder:3b":   true,
	"qwen2.5-coder:7b":   true,
}

// DefaultLocalCodingAlias is the model `fak guard --local`/`--gguf` picks when the user
// names no model — the smallest curated coding alias that still RELIABLY tool-calls, so
// the one-command local-coding path (epic #1056) works with zero knowledge of aliases or
// quant names. It resolves to Qwen2.5-Coder-3B-Instruct Q4_K_M, a single ~1.93 GB GGUF:
// the 1.5B floor tool-calls but drops/garbles multi-call turns more often, and the 7B is
// a heavier download than a "just works" default should pull. Callers must resolve this
// through a Registry so a user registry.json override of the same alias still wins.
const DefaultLocalCodingAlias = "qwen2.5-coder:3b"

// IsCoding reports whether an alias is one of the curated coding / tool-call-capable
// models. It checks the embedded set; a user-overlaid alias of the same name keeps the
// embedded coding flag (the overlay changes the target, not the model's nature), while a
// user-only alias is not flagged because fak cannot vouch for an arbitrary user target.
func IsCoding(alias string) bool { return codingAliases[strings.TrimSpace(alias)] }

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
//  3. a known alias expands to its target (dashes are normalized to colons to
//     accept both "qwen2.5-1.5b" and "qwen2.5:1.5b");
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
	// Try normalizing dashes to colons for family:size aliases (#1115).
	// The pattern is: <family>-<size> → <family>:<size>, where the dash to
	// replace is the second-last or last dash, depending on whether the family
	// contains a version component (e.g., "qwen2.5-coder-1.5b" has three dashes,
	// and the size separator is the second one).
	if e, ok := r.tryDashedAliases(ref); ok {
		return e.Target, true
	}
	return ref, false
}

// tryDashedAliases attempts to normalize dashed model names to the canonical
// colon-separated form. It tries replacing the last dash first (for simple
// "family-size" forms like "smollm2-135m" or "qwen2.5-coder-1.5b" → "qwen2.5-coder:1.5b"),
// then falls back to replacing the second-last dash (for "family-version-size"
// forms like "qwen2.5-1.5b" → "qwen2.5:1.5b").
func (r *Registry) tryDashedAliases(ref string) (Entry, bool) {
	parts := strings.Split(ref, "-")
	if len(parts) < 2 {
		return Entry{}, false
	}

	// Try replacing the last dash first (e.g., "qwen2.5-coder-1.5b" → "qwen2.5-coder:1.5b",
	// "smollm2-135m" → "smollm2:135m")
	lastIdx := len(parts) - 1
	normalized := strings.Join(parts[:lastIdx], "-") + ":" + parts[lastIdx]
	if e, ok := r.lookup(normalized); ok {
		return e, true
	}

	// Try replacing the second-last dash (e.g., "qwen2.5-1.5b" → "qwen2.5:1.5b")
	if len(parts) >= 3 {
		secondLastIdx := len(parts) - 2
		normalized = strings.Join(parts[:secondLastIdx], "-") + ":" + parts[secondLastIdx] + "-" + parts[lastIdx]
		if e, ok := r.lookup(normalized); ok {
			return e, true
		}
	}

	return Entry{}, false
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
		le := ListEntry{Entry: e, Coding: codingAliases[e.Name]}
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
	Coding    bool   // true for a curated coding / tool-call-capable alias (#1058)
}

// Cached reports whether this entry's target is present on disk.
func (le ListEntry) Cached() bool { return le.LocalPath != "" }
