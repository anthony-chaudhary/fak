package witness

// cache.go — the content-addressed WITNESS VERDICT CACHE (#2152): memoize a
// resolved verdict keyed on the immutable git objects it was computed FROM, so
// re-verification of an unchanged ref is a file read instead of a re-run of the
// evidence rung.
//
// WHY THIS IS SOUND (the load-bearing invariant): a witnessed verdict for a FIXED
// object is immutable. Which files a commit touched (notests), whether commit A is
// an ancestor of commit B (ancestor), whether any message in the history reachable
// from a fixed HEAD matches a pattern (grep) — none of these can change once the
// SHAs are pinned. So the cache key is derived from `git rev-parse` of the claim's
// anchors, never from the symbolic ref: a moved ref (new HEAD, rewritten branch)
// produces a DIFFERENT key and misses — invalidation is content-addressing, not
// bookkeeping (the #2152 contract: "invalidated only when the underlying ref
// changes").
//
// WHAT IS CACHEABLE (closed list — everything else always re-resolves):
//
//	ancestor:<ref>  keyed on (sha(ref), sha(HEAD))      ancestry of fixed commits is immutable
//	notests:<ref>   keyed on (sha(ref))                 a commit's file list is immutable
//	symptom:<ref>   keyed on (sha(ref), exec-mode)      same, plus which rung was armed
//	grep:<pattern>  keyed on (pattern, sha(HEAD))       history reachable from a fixed tip
//
// path: / clean: / committed: read the WORKING TREE or the index — mutable state
// with no content address — and commit:<ref> IS a bare resolution (caching it saves
// nothing). exec: runs arbitrary commands against the current tree; rsl: is a
// flagged spike. All of those stay uncached, always.
//
// ABSTAIN IS NEVER CACHED: an abstain is infrastructure uncertainty (git missing,
// bad ref), not evidence; freezing it would pin a transient failure as a durable
// answer.
//
// FLEET-SHARED BY PLACEMENT: entries live under <git-common-dir>/fak/witness-cache/.
// Every agent session on the shared clone resolves the same common dir, so one
// agent's verified verdict is every peer's cache hit — and the workspace identity
// the key needs is carried by WHERE the file lives, not by a field. The dir is
// inside .git, so it can never be committed or leak into the tree. Writes are
// best-effort (temp file + rename, errors swallowed): the cache accelerates, it
// never gates — a cache failure degrades to exactly the uncached behavior.
//
// ESCAPE HATCH: FAK_WITNESS_CACHE=off (or 0/false/no) disables both read and write.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// CacheFlagEnv disables the verdict cache when set to off/0/false/no. Default on:
// every key pins immutable objects, so a stale hit is structurally impossible —
// the switch exists for forensics (measuring rung latency) and belt-and-braces.
const CacheFlagEnv = "FAK_WITNESS_CACHE"

const (
	cacheKeySchema   = "fak-witness-cache-key/v1"
	cacheEntrySchema = "fak-witness-verdict-cache/v1"
	cacheVerbVerify  = "dos_verify"
)

func cacheEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(CacheFlagEnv))) {
	case "off", "0", "false", "no":
		return false
	default:
		return true
	}
}

type cacheKeyMaterial struct {
	Schema       string   `json:"schema"`
	Verb         string   `json:"verb"`
	Parts        []string `json:"parts,omitempty"`
	ResolvedRefs []string `json:"resolved_refs,omitempty"`
}

// WitnessCacheKey returns the deterministic key material for a witnessed verdict.
// Callers pass already-resolved refs (commit/tree SHAs) rather than symbolic refs;
// this is the stale-ref invalidation rule. If a branch, tag, or HEAD moves, the
// caller resolves a different SHA and therefore gets a different key.
func WitnessCacheKey(verb string, resolvedRefs []string, parts ...string) string {
	m := cacheKeyMaterial{
		Schema:       cacheKeySchema,
		Verb:         strings.TrimSpace(verb),
		Parts:        cleanCacheParts(parts),
		ResolvedRefs: cleanCacheParts(resolvedRefs),
	}
	b, err := json.Marshal(m)
	if err != nil {
		return ""
	}
	return string(b)
}

// ResolveRefCacheKey resolves refs through git and builds a WitnessCacheKey from
// their immutable SHAs. This is the generic API for both dos_verify and
// dos_commit_audit-style callers: use the verb to split verdict namespaces, refs
// for invalidation, and parts for non-ref dimensions such as claim kind or grep
// pattern.
func ResolveRefCacheKey(ctx context.Context, run Runner, dir, verb string, refs []string, parts ...string) (string, bool) {
	if strings.TrimSpace(verb) == "" {
		return "", false
	}
	if run == nil {
		run = gitRunner
	}
	cleanRefs := cleanCacheParts(refs)
	if len(cleanRefs) == 0 {
		return WitnessCacheKey(verb, nil, parts...), true
	}
	args := append([]string{"rev-parse", "--verify", "--quiet"}, cleanRefs...)
	// `rev-parse --verify` accepts exactly one rev; multi-anchor keys use plain
	// rev-parse and require one resolved SHA per requested ref.
	if len(cleanRefs) > 1 {
		args = append([]string{"rev-parse"}, cleanRefs...)
	}
	out, code, err := run(ctx, dir, args...)
	if err != nil || code != 0 {
		return "", false
	}
	shas := strings.Fields(out)
	if len(shas) != len(cleanRefs) {
		return "", false
	}
	return WitnessCacheKey(verb, shas, parts...), true
}

func cleanCacheParts(parts []string) []string {
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// CachedVerdict is one memoized verdict on disk. The key is re-checked on read —
// a mismatched entry is ignored, never trusted. Verdict is deliberately generic
// so the same cache can hold dos_verify outcomes and future commit-audit labels.
type CachedVerdict struct {
	Schema  string `json:"schema"`
	Key     string `json:"key"`
	Verb    string `json:"verb"`
	Subject string `json:"subject,omitempty"`
	Verdict string `json:"verdict"`
}

// VerdictCache is the fleet-shared on-disk verdict cache rooted at
// <git-common-dir>/fak/witness-cache.
type VerdictCache struct {
	dir string
}

// WitnessCacheDir maps a git common dir to the shared witness-cache directory.
func WitnessCacheDir(gitCommonDir string) string {
	return filepath.Join(gitCommonDir, "fak", "witness-cache")
}

// NewVerdictCache constructs a deterministic file-backed verdict cache.
func NewVerdictCache(dir string) *VerdictCache {
	return &VerdictCache{dir: dir}
}

// Get returns the memoized verdict for key, if a valid entry exists. Unreadable,
// unparseable, stale-schema, or key-mismatched entries are misses.
func (c *VerdictCache) Get(key string) (CachedVerdict, bool) {
	if c == nil || strings.TrimSpace(c.dir) == "" || strings.TrimSpace(key) == "" {
		return CachedVerdict{}, false
	}
	b, err := os.ReadFile(c.path(key))
	if err != nil {
		return CachedVerdict{}, false
	}
	var e CachedVerdict
	if json.Unmarshal(b, &e) != nil || e.Schema != cacheEntrySchema || e.Key != key || strings.TrimSpace(e.Verdict) == "" {
		return CachedVerdict{}, false
	}
	return e, true
}

// Put memoizes a verdict, best-effort. The cache accelerates; it never gates.
// Entries are written via temp-file + rename so concurrent readers see a whole
// JSON object or no object. Rewriting the same entry is deterministic: no wall
// clock or process-local data is serialized.
func (c *VerdictCache) Put(e CachedVerdict) bool {
	if c == nil || strings.TrimSpace(c.dir) == "" || strings.TrimSpace(e.Key) == "" || strings.TrimSpace(e.Verdict) == "" {
		return false
	}
	e.Schema = cacheEntrySchema
	if err := os.MkdirAll(c.dir, 0o755); err != nil {
		return false
	}
	b, err := json.Marshal(e)
	if err != nil {
		return false
	}
	tmp, err := os.CreateTemp(c.dir, ".entry-*.tmp")
	if err != nil {
		return false
	}
	name := tmp.Name()
	_, werr := tmp.Write(b)
	cerr := tmp.Close()
	if werr != nil || cerr != nil {
		os.Remove(name)
		return false
	}
	p := c.path(e.Key)
	if err := os.Rename(name, p); err == nil {
		return true
	}
	// Windows refuses Rename over an existing destination. Heal corrupt stale
	// entries by replacing them; if another peer already wrote a valid entry, the
	// remove may briefly turn a hit into a miss, which only degrades to recompute.
	_ = os.Remove(p)
	if err := os.Rename(name, p); err != nil {
		os.Remove(name)
		return false
	}
	return true
}

func (c *VerdictCache) path(key string) string {
	sum := sha256.Sum256([]byte(key))
	return filepath.Join(c.dir, hex.EncodeToString(sum[:])+".json")
}

// cacheKey derives the content-addressed key for a (kind, arg) claim, or ok=false
// when the claim kind is not cacheable (or the cache is disabled, or an anchor does
// not resolve). One `git rev-parse` call resolves every anchor SHA the key needs —
// the whole price of a lookup — so a hit costs one O(1) plumbing call instead of
// the rung it memoizes (a history walk, a commit diff, or the red-then-green
// symptom execution).
func (r *Resolver) cacheKey(ctx context.Context, kind, arg string) (string, bool) {
	if !cacheEnabled() {
		return "", false
	}
	var anchors []string
	var parts []string
	switch kind {
	case "ancestor":
		anchors = []string{arg + "^{commit}", "HEAD"}
		parts = []string{kind}
	case "notests":
		anchors = []string{arg + "^{commit}"}
		parts = []string{kind}
	case "symptom":
		anchors = []string{arg + "^{commit}"}
		if SymptomExecEnabled() {
			parts = []string{kind, "mode=exec"}
		} else {
			parts = []string{kind, "mode=struct"}
		}
	case "grep":
		anchors = []string{"HEAD"}
		parts = []string{kind, arg}
	default:
		return "", false
	}
	return ResolveRefCacheKey(ctx, r.run, r.dir, cacheVerbVerify, anchors, parts...)
}

// verdictCache maps a Resolver to <git-common-dir>/fak/witness-cache/.
// The common dir is resolved once per Resolver through the SAME Runner seam as the
// evidence rungs (so tests drive it); an unresolvable common dir disables the cache
// for this Resolver ("" memoized), degrading to uncached behavior.
func (r *Resolver) verdictCache(ctx context.Context) (*VerdictCache, bool) {
	r.cacheOnce.Do(func() {
		out, code, err := r.run(ctx, r.dir, "rev-parse", "--git-common-dir")
		if err != nil || code != 0 {
			return
		}
		dir := strings.TrimSpace(out)
		if dir == "" {
			return
		}
		if !filepath.IsAbs(dir) && r.dir != "" {
			dir = filepath.Join(r.dir, dir)
		}
		if st, err := os.Stat(dir); err != nil || !st.IsDir() {
			return
		}
		r.cacheDir = WitnessCacheDir(dir)
	})
	if r.cacheDir == "" {
		return nil, false
	}
	return NewVerdictCache(r.cacheDir), true
}

// cacheGet returns the memoized verdict for key, if a valid entry exists. An
// unreadable, unparseable, or key-mismatched entry is a miss — the cache never
// invents evidence.
func (r *Resolver) cacheGet(ctx context.Context, key string) (abi.WitnessOutcome, bool) {
	c, ok := r.verdictCache(ctx)
	if !ok {
		return abi.WitnessAbstain, false
	}
	e, hit := c.Get(key)
	if !hit || e.Verb != cacheVerbVerify {
		return abi.WitnessAbstain, false
	}
	switch e.Verdict {
	case "confirmed":
		return abi.WitnessConfirmed, true
	case "refuted":
		return abi.WitnessRefuted, true
	}
	return abi.WitnessAbstain, false
}

// cachePut memoizes a confirmed/refuted verdict, best-effort: a write failure is
// swallowed (the cache accelerates, it never gates). Abstain is never written. The
// entry lands via temp-file + rename so a concurrent peer reads a whole entry or
// none — the same clone is shared by many agent sessions.
func (r *Resolver) cachePut(ctx context.Context, key, claim string, out abi.WitnessOutcome) {
	var verdict string
	switch out {
	case abi.WitnessConfirmed:
		verdict = "confirmed"
	case abi.WitnessRefuted:
		verdict = "refuted"
	default:
		return
	}
	c, ok := r.verdictCache(ctx)
	if !ok {
		return
	}
	_ = c.Put(CachedVerdict{Key: key, Verb: cacheVerbVerify, Subject: claim, Verdict: verdict})
}
