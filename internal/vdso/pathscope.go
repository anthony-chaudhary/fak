package vdso

import (
	"encoding/json"
	"path/filepath"
	"strings"
)

// pathscope.go — the PER-PATH write-generation invalidator (#795).
//
// The hierarchical eraser in scope.go keys invalidation on a (namespace, entity) tag
// derived from the TOOL NAME (book_flight -> "flights"), which is exactly right for the
// travel-domain demo tools but produces NOTHING for the coding-agent tools that matter in
// the fak guard / fak serve topology: Read, Edit, Write, Glob, Grep. Their names don't
// match any nsKeyword, so namespaceOf returns "" and every file Read binds only the root
// tag — meaning ANY write anywhere invalidates it (a full flush, never a precise erase),
// the cross-agent net-loss scope.go exists to avoid.
//
// The fix is the missing rung the epic (#794) named: bind a Read's cache entry to the
// generation of the FILESYSTEM PATH it read, and bump exactly that path's generation when
// the kernel adjudicates a Write/Edit to it. A file path is just an entity under a
// reserved "files" namespace, so it rides the existing epoch vector, bumpAndPublish, and
// keyLocked UNCHANGED — v.nodes["files:<path>"] IS the per-path write-generation counter.
// This is the closed-loop edge over open-loop provider caching: the kernel SAW the Write,
// so it knows precisely which Read to strand, by path, not by a coarse world flush.
//
// Soundness rests on one invariant: a read and a write to the SAME file must canonicalize
// to the SAME path string, or the write would not strand the read it changed.
// fileCanonPath is the single canonicalizer both sides call. It is intentionally
// conservative — when a tool's path argument is absent or unparseable (a Bash command can
// touch arbitrary paths and carries no single file_path), fileEntityOf returns "" and the
// call falls back to the namespace/root tagging, which only ever OVER-invalidates.

// filesNamespace is the reserved depth-1 tag for filesystem-path-scoped entries. It is
// outside the nsKeywords tool-name taxonomy on purpose: file tools (Read/Edit/Write) are
// identified by carrying a path ARGUMENT, not by a name keyword, so this namespace is
// assigned by fileEntityOf, never by classifyNamespace.
const filesNamespace = "files"

// filePathArgKeys are the argument keys, in priority order, under which a file-shaped tool
// names its target path. file_path is the Claude Code harness convention (Read/Edit/Write);
// path is the generic fallback. Both read-shaped (Read) and write-shaped (Edit/Write) tools
// use the same keys, so a Read and an Edit of one file extract the identical raw path.
var filePathArgKeys = []string{"file_path", "path", "filename", "filepath"}

// fileEntityOf returns the canonical filesystem path a file-shaped tool call targets, or ""
// when the call names no single path (e.g. Bash, or a glob/grep over a directory). The
// returned string is the "entity" under the files namespace; it is canonicalized so a read
// and a write to the same file collide on it regardless of separator style or "." segments.
// Read and write share this one extractor, so the per-path generation a write bumps is the
// exact one a prior read bound.
func fileEntityOf(args []byte) string {
	if len(args) == 0 {
		return ""
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(args, &m) != nil {
		return ""
	}
	for _, k := range filePathArgKeys {
		raw, ok := m[k]
		if !ok {
			continue
		}
		var s string
		if json.Unmarshal(raw, &s) != nil || s == "" {
			continue
		}
		if cp := fileCanonPath(s); cp != "" {
			return cp
		}
	}
	return ""
}

// fileCanonPath normalizes a path string so the SAME file always yields the SAME tag,
// regardless of how the read and the write happened to spell it. It cleans "." / ".."
// segments and normalizes separators to forward slashes (so a Windows agent's
// "C:\work\x" and a tool that echoes "C:/work/x" collide). It does NOT lowercase — paths
// are case-sensitive on the agent's filesystem on Linux, and lowercasing would falsely
// alias two distinct files, which is the one error class soundness cannot tolerate (a
// write to a.txt invalidating a read of A.txt is over-invalidation = safe, but the
// reverse — a read of a.txt served stale because the write hit A.txt — is NOT, so we
// never merge case-distinct paths). It does NOT resolve symlinks or make the path
// absolute: a relative read and a relative write in the same session share a cwd, and
// resolving here would require filesystem access on a pure-data path; an absolute and a
// relative spelling of one file simply land in different tags (conservative — they don't
// cross-invalidate, but neither is served stale because each only matches its own spelling).
func fileCanonPath(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	// filepath.Clean uses the OS separator; normalize to '/' afterwards so the tag is
	// stable across the separator a tool happened to emit.
	cleaned := filepath.ToSlash(filepath.Clean(s))
	if cleaned == "." || cleaned == "" {
		return ""
	}
	return cleaned
}

// filePathTag returns the leaf tag for a canonical path, e.g. "files:C:/work/x.go".
func filePathTag(canon string) string { return filesNamespace + ":" + canon }

// fileShapedButUnnamed reports that args CARRY a file-path key (so the call is a file
// tool) but its path won't canonicalize to a usable entity. Such a read can't bind a
// path leaf, so a per-path write would not strand it — the cacheability gate
// (resourceMisnamed) must refuse it to preserve "a hit equals a fresh call". A call with
// no file-path key at all is not file-shaped and returns false (it is some other tool,
// soundly handled by the namespace/root tagging).
func fileShapedButUnnamed(args []byte) bool {
	if len(args) == 0 {
		return false
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(args, &m) != nil {
		return false
	}
	hasPathKey := false
	for _, k := range filePathArgKeys {
		if _, ok := m[k]; ok {
			hasPathKey = true
			break
		}
	}
	if !hasPathKey {
		return false
	}
	return fileEntityOf(args) == ""
}

// fileReadChain returns the root->leaf tag chain for a file-shaped READ, or nil when the
// call names no single path (so the caller falls through to the namespace tagger). The
// chain mirrors the (root, namespace, entity) shape every other read binds, so keyLocked
// and the coherence bus treat it identically. Bound at Resource granularity only: at
// coarser granularities a read already reaches its target depth via the namespace/root
// chain, and a per-path leaf would be stranded by a namespace-level write it should ignore.
func (v *VDSO) fileReadChain(args []byte) []string {
	ent := v.fileLeafEntity(args)
	if ent == "" {
		return nil
	}
	return []string{rootTag, filesNamespace, filePathTag(ent)}
}

// fileLeafEntity returns the single filesystem path a file-shaped call names — the leaf
// both fileReadChain and fileWriteTags bind to "files:<path>" — or "" when per-path scope
// does not apply (coarser than Resource granularity, or the call names no single path, so
// the caller falls back to the namespace/root chain). Shared so the read and write sides
// derive the leaf identically and can never drift in what counts as a file-shaped call.
func (v *VDSO) fileLeafEntity(args []byte) string {
	if v.GranularityOf() != Resource {
		return ""
	}
	return fileEntityOf(args)
}

// fileWriteTags returns the single finest tag a file-shaped WRITE must bump — the path's
// own leaf — or nil when the write names no single path (so the caller falls back to the
// namespace/root flush, which over-invalidates soundly). A write to path P bumps
// "files:P", which strands exactly the reads whose chain contains "files:P" (that file's
// reads) and leaves every other file's cached reads warm.
func (v *VDSO) fileWriteTags(args []byte) []string {
	ent := v.fileLeafEntity(args)
	if ent == "" {
		return nil
	}
	return []string{filePathTag(ent)}
}
