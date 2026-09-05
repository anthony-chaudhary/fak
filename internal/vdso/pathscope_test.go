package vdso

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// pathscope_test.go — witnesses for the per-path write-generation invalidator (#795):
// a file Read is served from cache, an Edit/Write to the SAME path strands exactly that
// file's reads, an Edit to a DIFFERENT path leaves them warm (precision, not a full
// flush), path spellings of one file collide, and a file read whose path can't be named
// is refused rather than served stale.

// TestPathScope_WriteInvalidatesSameFileSparesOther is the headline closed-loop witness:
// the kernel saw the Edit, so it strands only that file's cached Read — the unfair edge
// over a coarse world flush that would erase every other file's warmed reads.
func TestPathScope_WriteInvalidatesSameFileSparesOther(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)

	readA := roCall("Read", `{"file_path":"/work/a.go"}`)
	readB := roCall("Read", `{"file_path":"/work/b.go"}`)
	fillAndExpectHit(t, v, readA, `package a`)
	fillAndExpectHit(t, v, readB, `package b`)

	// An Edit to a.go bumps only files:/work/a.go.
	v.Emit(completeEvent(wrCall("Edit", `{"file_path":"/work/a.go","old_string":"x","new_string":"y"}`), `{"ok":true}`))

	if hits(t, v, readA) {
		t.Errorf("Read(a.go) still hits after Edit(a.go) — the write that changed the file did not strand its read")
	}
	if !hits(t, v, readB) {
		t.Errorf("Read(b.go) MISSED after Edit(a.go) — an edit to one file must not erase another file's cached read")
	}
}

// TestPathScope_WriteShapedNames covers the other write-shaped file tools (Write, not just
// Edit) and confirms the tool-name write-shape heuristic routes them to the invalidator.
func TestPathScope_WriteShapedNames(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)

	read := roCall("Read", `{"file_path":"/etc/config.yaml"}`)
	fillAndExpectHit(t, v, read, `key: val`)

	v.Emit(completeEvent(wrCall("Write", `{"file_path":"/etc/config.yaml","content":"key: new"}`), `{"ok":true}`))
	if hits(t, v, read) {
		t.Errorf("Read(config.yaml) still hits after Write(config.yaml)")
	}
}

// TestPathScope_CanonicalCollision proves a read and a write to the SAME file collide on
// the path tag even when spelled differently ("./a.go" vs "a.go" vs "x/../a.go"), so the
// write strands the read. A mismatch here would be the soundness-fatal stale serve.
func TestPathScope_CanonicalCollision(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)

	// Read spells the path one way...
	read := roCall("Read", `{"file_path":"./pkg/a.go"}`)
	fillAndExpectHit(t, v, read, `package a`)

	// ...the Edit spells the same file another way. They must canonicalize to one tag.
	v.Emit(completeEvent(wrCall("Edit", `{"file_path":"pkg/sub/../a.go","old_string":"x","new_string":"y"}`), `{"ok":true}`))
	if hits(t, v, read) {
		t.Errorf("Read('./pkg/a.go') still hits after Edit('pkg/sub/../a.go') — same file, different spelling, not invalidated")
	}
}

// TestPathScope_UnnamedFileReadNotCached is the soundness gate: a file-shaped read that
// carries a file_path key but whose value won't canonicalize to a usable path must NOT be
// tier-2 cached — otherwise a per-path write (which does not bump the root) could never
// strand it. It must always miss (reach the engine), which is sound.
func TestPathScope_UnnamedFileReadNotCached(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)

	// A file_path that cleans to "." names no file — refused.
	bad := roCall("Read", `{"file_path":"."}`)
	v.Emit(completeEvent(bad, `whatever`))
	if hits(t, v, bad) {
		t.Errorf("a file read with an un-nameable path was tier-2 cached — a per-path write could serve it stale")
	}
}

// TestPathScope_NonFileReadStillCached confirms the gate does not over-fire: a read with
// NO file_path arg (a genuine non-file tool) is unaffected and stays cacheable under the
// existing namespace/root chain.
func TestPathScope_NonFileReadStillCached(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)

	// search_direct_flight has no file_path key — it's the demo namespace tool, must still
	// cache exactly as before this change.
	flight := roCall("search_direct_flight", sfoJFK)
	fillAndExpectHit(t, v, flight, `{"flights":["AA1"]}`)
}

// TestPathScope_BashDoesNotPathInvalidate confirms a Bash call (no single file_path, can
// touch arbitrary paths) is NOT given a path tag — it falls through to the namespace/root
// flush. A bash write therefore invalidates path-bound reads via the root, conservatively.
func TestPathScope_BashDoesNotPathInvalidate(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)

	read := roCall("Read", `{"file_path":"/work/a.go"}`)
	fillAndExpectHit(t, v, read, `package a`)

	// A Bash run carries a command string, not a file_path → write tags fall back to root.
	// "run" is write-shaped, so this is a destructive completion that flushes the root.
	v.Emit(completeEvent(wrCall("run_bash", `{"command":"echo hi > /work/a.go"}`), `{"ok":true}`))
	if hits(t, v, read) {
		t.Errorf("Read(a.go) survived an untaggable Bash write — a write that cannot name its path must flush conservatively")
	}
}

// TestFileCanonPath is a direct unit witness on the canonicalizer both sides share.
func TestFileCanonPath(t *testing.T) {
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}

	for _, caseInsensitive := range []bool{false, true} {
		orig := isCaseInsensitiveOS
		isCaseInsensitiveOS = caseInsensitive
		name := "case-sensitive"
		if caseInsensitive {
			name = "case-insensitive"
		}

		t.Run(name, func(t *testing.T) {
			defer func() { isCaseInsensitiveOS = orig }()

			wantRelA := filepath.ToSlash(filepath.Join(cwd, "a.go"))
			wantRelC := filepath.ToSlash(filepath.Join(cwd, "a", "c.go"))
			wantWin := "C:/work/x.go"
			wantAbsUpper := "/work/A.go"
			if caseInsensitive {
				wantRelA = strings.ToLower(wantRelA)
				wantRelC = strings.ToLower(wantRelC)
				wantWin = "c:/work/x.go"
				wantAbsUpper = "/work/a.go"
			}

			cases := []struct{ in, want string }{
				{"/work/a.go", "/work/a.go"},
				{"/work/A.go", wantAbsUpper},
				{"./a.go", wantRelA},
				{"a/b/../c.go", wantRelC},
				{"  /work/a.go  ", "/work/a.go"},
				{".", ""},
				{"", ""},
				{"C:\\work\\x.go", wantWin},
			}
			for _, tc := range cases {
				got := fileCanonPath(tc.in)
				if got != tc.want {
					t.Errorf("fileCanonPath(%q) = %q, want %q", tc.in, got, tc.want)
				}
			}
		})
	}
}

// TestPathScope_CaseInsensitiveInvalidation tests that on case-insensitive filesystems
// (Windows, macOS), Read("README.md") followed by Write("readme.md") invalidates the
// cached README.md.
func TestPathScope_CaseInsensitiveInvalidation(t *testing.T) {
	orig := isCaseInsensitiveOS
	isCaseInsensitiveOS = true
	defer func() { isCaseInsensitiveOS = orig }()

	v := New(64)
	v.SetGranularity(Resource)

	read := roCall("Read", `{"file_path":"README.md"}`)
	fillAndExpectHit(t, v, read, `# fak`)

	// Mutate using different casing
	v.Emit(completeEvent(wrCall("Write", `{"file_path":"readme.md","content":"# updated"}`), `{"ok":true}`))
	if hits(t, v, read) {
		t.Errorf("Read('README.md') still hits after Write('readme.md') on case-insensitive OS; write did not invalidate case-variant read")
	}
}

// TestPathScope_RelativeAndAbsoluteCollision tests that relative and absolute paths
// for the same file map to the same tag and invalidate each other in both directions.
func TestPathScope_RelativeAndAbsoluteCollision(t *testing.T) {
	v := New(64)
	v.SetGranularity(Resource)

	cwd, err := os.Getwd()
	if err != nil {
		t.Fatalf("Getwd: %v", err)
	}
	absPath := filepath.ToSlash(filepath.Join(cwd, "README.md"))

	// Direction 1: Read via relative path, Write via absolute path
	readRel := roCall("Read", `{"file_path":"README.md"}`)
	fillAndExpectHit(t, v, readRel, `# fak`)

	writeAbs := wrCall("Write", fmt.Sprintf(`{"file_path":%q,"content":"# new"}`, absPath))
	v.Emit(completeEvent(writeAbs, `{"ok":true}`))

	if hits(t, v, readRel) {
		t.Errorf("Read('README.md') still hits after Write(%q); relative and absolute paths did not collide", absPath)
	}

	// Direction 2: Read via absolute path, Write via relative path
	readAbs := roCall("Read", fmt.Sprintf(`{"file_path":%q}`, absPath))
	fillAndExpectHit(t, v, readAbs, `# new`)

	writeRel := wrCall("Write", `{"file_path":"./README.md","content":"# newer"}`)
	v.Emit(completeEvent(writeRel, `{"ok":true}`))

	if hits(t, v, readAbs) {
		t.Errorf("Read(%q) still hits after Write('./README.md'); absolute read not invalidated by relative write", absPath)
	}
}

// TestPathScope_CaseSensitivePreservesDistinctTags tests that on case-sensitive filesystems
// (Linux default, isCaseInsensitiveOS = false), distinct casing paths do NOT cross-invalidate.
func TestPathScope_CaseSensitivePreservesDistinctTags(t *testing.T) {
	orig := isCaseInsensitiveOS
	isCaseInsensitiveOS = false
	defer func() { isCaseInsensitiveOS = orig }()

	v := New(64)
	v.SetGranularity(Resource)

	readUpper := roCall("Read", `{"file_path":"/work/README.md"}`)
	fillAndExpectHit(t, v, readUpper, `# uppercase`)

	// Mutate different casing on case-sensitive OS
	v.Emit(completeEvent(wrCall("Write", `{"file_path":"/work/readme.md","content":"# lowercase"}`), `{"ok":true}`))

	// On case-sensitive OS, /work/README.md and /work/readme.md are distinct files;
	// readUpper should still hit.
	if !hits(t, v, readUpper) {
		t.Errorf("Read('/work/README.md') was invalidated by Write('/work/readme.md') on case-sensitive OS; distinct casing should not cross-invalidate")
	}
}
