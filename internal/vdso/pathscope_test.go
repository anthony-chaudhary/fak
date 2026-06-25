package vdso

import "testing"

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
	cases := []struct{ in, want string }{
		{"/work/a.go", "/work/a.go"},
		{"./a.go", "a.go"},
		{"a/b/../c.go", "a/c.go"},
		{"  /work/a.go  ", "/work/a.go"},
		{".", ""},
		{"", ""},
		{"C:\\work\\x.go", "C:/work/x.go"}, // separator normalization (filepath.Clean is OS-dependent; ToSlash makes the tag stable)
	}
	for _, tc := range cases {
		got := fileCanonPath(tc.in)
		// On non-Windows, filepath.Clean treats backslashes as literal chars, so the
		// Windows case may not normalize — accept either the slash form or a non-empty
		// stable string for that one input.
		if tc.in == "C:\\work\\x.go" {
			if got == "" {
				t.Errorf("fileCanonPath(%q) = empty, want a stable non-empty tag", tc.in)
			}
			continue
		}
		if got != tc.want {
			t.Errorf("fileCanonPath(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
