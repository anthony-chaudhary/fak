package codetools

import (
	"context"
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

// TestApplyPatchSingleHunk: single file, single hunk modification.
func TestApplyPatchSingleHunk(t *testing.T) {
	ts, root := newTestToolset(t)
	p := filepath.Join(root, "single.txt")
	mustWrite(t, p, "line1\nline2\nline3\n")

	diff := `--- a/single.txt
+++ b/single.txt
@@ -1,3 +1,3 @@
 line1
-line2
+line2_modified
 line3
`
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: diff}))
	if bad {
		t.Fatalf("ApplyPatch single hunk failed: %s", out)
	}

	content, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read single.txt: %v", err)
	}
	want := "line1\nline2_modified\nline3\n"
	if string(content) != want {
		t.Fatalf("content = %q, want %q", string(content), want)
	}

	var res PatchResult
	if err := json.Unmarshal(out, &res); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	if len(res.FilesModified) != 1 || res.FilesModified[0] != "single.txt" {
		t.Fatalf("files_modified = %v, want [single.txt]", res.FilesModified)
	}
	if res.HunksApplied != 1 {
		t.Fatalf("hunks_applied = %d, want 1", res.HunksApplied)
	}
}

// TestApplyPatchMultiHunk: single file with multiple hunks.
func TestApplyPatchMultiHunk(t *testing.T) {
	ts, root := newTestToolset(t)
	p := filepath.Join(root, "multi.txt")
	mustWrite(t, p, "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\n")

	diff := `--- a/multi.txt
+++ b/multi.txt
@@ -2,3 +2,3 @@
 l2
-l3
+l3_new
 l4
@@ -7,3 +7,3 @@
 l7
-l8
+l8_new
 l9
`
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: diff}))
	if bad {
		t.Fatalf("ApplyPatch multi hunk failed: %s", out)
	}

	content, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read multi.txt: %v", err)
	}
	want := "l1\nl2\nl3_new\nl4\nl5\nl6\nl7\nl8_new\nl9\nl10\n"
	if string(content) != want {
		t.Fatalf("content = %q, want %q", string(content), want)
	}

	var res PatchResult
	if err := json.Unmarshal(out, &res); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	if len(res.FilesModified) != 1 || res.FilesModified[0] != "multi.txt" {
		t.Fatalf("files_modified = %v, want [multi.txt]", res.FilesModified)
	}
	if res.HunksApplied != 2 {
		t.Fatalf("hunks_applied = %d, want 2", res.HunksApplied)
	}
}

// TestApplyPatchCreateFile: creating a new file from patch.
func TestApplyPatchCreateFile(t *testing.T) {
	ts, root := newTestToolset(t)
	diff := `--- /dev/null
+++ b/new_created.txt
@@ -0,0 +1,3 @@
+lineA
+lineB
+lineC
`
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: diff}))
	if bad {
		t.Fatalf("ApplyPatch create file failed: %s", out)
	}

	p := filepath.Join(root, "new_created.txt")
	content, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read new_created.txt: %v", err)
	}
	want := "lineA\nlineB\nlineC\n"
	if string(content) != want {
		t.Fatalf("content = %q, want %q", string(content), want)
	}

	var res PatchResult
	if err := json.Unmarshal(out, &res); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	if len(res.FilesCreated) != 1 || res.FilesCreated[0] != "new_created.txt" {
		t.Fatalf("files_created = %v, want [new_created.txt]", res.FilesCreated)
	}
	if res.HunksApplied != 1 {
		t.Fatalf("hunks_applied = %d, want 1", res.HunksApplied)
	}
}

// TestApplyPatchDeleteFile: deleting a file from patch.
func TestApplyPatchDeleteFile(t *testing.T) {
	ts, root := newTestToolset(t)
	p := filepath.Join(root, "to_delete.txt")
	mustWrite(t, p, "first\nsecond\n")

	diff := `--- a/to_delete.txt
+++ /dev/null
@@ -1,2 +0,0 @@
-first
-second
`
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: diff}))
	if bad {
		t.Fatalf("ApplyPatch delete file failed: %s", out)
	}

	if _, err := os.Stat(p); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("file still exists: %v", err)
	}

	var res PatchResult
	if err := json.Unmarshal(out, &res); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	if len(res.FilesDeleted) != 1 || res.FilesDeleted[0] != "to_delete.txt" {
		t.Fatalf("files_deleted = %v, want [to_delete.txt]", res.FilesDeleted)
	}
}

// TestApplyPatchFuzzTolerance: patch with minor line offset drift succeeds within fuzz margin.
func TestApplyPatchFuzzTolerance(t *testing.T) {
	ts, root := newTestToolset(t)
	p := filepath.Join(root, "fuzz_target.txt")
	// Target lines have shifted down by 2 lines compared to diff's expected OldStart
	initialContent := "preamble1\npreamble2\nstart_marker\nreplace_me\nend_marker\n"
	mustWrite(t, p, initialContent)

	// Diff expects start_marker at line 1 (drift = +2 lines)
	diff := `--- a/fuzz_target.txt
+++ b/fuzz_target.txt
@@ -1,3 +1,3 @@
 start_marker
-replace_me
+replaced_val
 end_marker
`
	// With default FuzzMargin (2) or explicit FuzzMargin: 2, this must succeed
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{
		Patch:      diff,
		FuzzMargin: 2,
	}))
	if bad {
		t.Fatalf("ApplyPatch with fuzz margin 2 failed: %s", out)
	}

	content, err := os.ReadFile(p)
	if err != nil {
		t.Fatalf("read file: %v", err)
	}
	want := "preamble1\npreamble2\nstart_marker\nreplaced_val\nend_marker\n"
	if string(content) != want {
		t.Fatalf("content = %q, want %q", string(content), want)
	}

	// Also verify trailing whitespace tolerance
	pWS := filepath.Join(root, "whitespace.txt")
	mustWrite(t, pWS, "func test() {  \n    return 42 \n}\n")
	diffWS := `--- a/whitespace.txt
+++ b/whitespace.txt
@@ -1,3 +1,3 @@
 func test() {
-    return 42
+    return 100
 }
`
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: diffWS}))
	if bad {
		t.Fatalf("ApplyPatch with trailing whitespace tolerance failed: %s", out)
	}
	contentWS, _ := os.ReadFile(pWS)
	wantWS := "func test() {  \n    return 100\n}\n"
	if string(contentWS) != wantWS {
		t.Fatalf("whitespace content = %q, want %q", string(contentWS), wantWS)
	}
}

// TestApplyPatchCASConflict: invalid expected_version fails with CAS error.
func TestApplyPatchCASConflict(t *testing.T) {
	ts, root := newTestToolset(t)
	p := filepath.Join(root, "cas_test.txt")
	mustWrite(t, p, "alpha\nbeta\n")

	diff := `--- a/cas_test.txt
+++ b/cas_test.txt
@@ -1,2 +1,2 @@
 alpha
-beta
+gamma
`
	// Wrong version hash must fail with FS_STALE_VERSION
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{
		Patch:           diff,
		ExpectedVersion: "0000000000000000000000000000000000000000000000000000000000000000",
	}))
	if !bad || errCode(t, out) != CodeStaleVersion {
		t.Fatalf("expected FS_STALE_VERSION, got bad=%v code=%s out=%s", bad, errCode(t, out), out)
	}

	// Verify file is unchanged
	content, _ := os.ReadFile(p)
	if string(content) != "alpha\nbeta\n" {
		t.Fatalf("file changed on CAS failure: %q", string(content))
	}
}

// TestApplyPatchConfinement: path traversal ../../etc/passwd is refused.
func TestApplyPatchConfinement(t *testing.T) {
	ts, _ := newTestToolset(t)

	// Attempt path traversal out of workspace
	diffTraversal := `--- a/../../etc/passwd
+++ b/../../etc/passwd
@@ -1,1 +1,1 @@
-root
+hacked
`
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: diffTraversal}))
	if !bad || errCode(t, out) != CodePathEscape {
		t.Fatalf("expected PATH_ESCAPE for traversal, got bad=%v code=%s out=%s", bad, errCode(t, out), out)
	}
}

// TestApplyPatchMultiFileRollback: failure during multi-file patch rolls back already modified files.
func TestApplyPatchMultiFileRollback(t *testing.T) {
	ts, root := newTestToolset(t)
	p1 := filepath.Join(root, "file1.txt")
	p2 := filepath.Join(root, "file2.txt")
	mustWrite(t, p1, "orig1\n")
	mustWrite(t, p2, "orig2\n")

	// Multi-file patch where file1 hunk matches, but file2 hunk fails to match
	diff := `--- a/file1.txt
+++ b/file1.txt
@@ -1,1 +1,1 @@
-orig1
+mod1
--- a/file2.txt
+++ b/file2.txt
@@ -1,1 +1,1 @@
-nonexistent_line
+mod2
`
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, map[string]any{
		"patch":       diff,
		"fuzz_margin": 0,
	}))
	if !bad || errCode(t, out) != CodeEditConflict {
		t.Fatalf("expected EDIT_CONFLICT, got bad=%v code=%s out=%s", bad, errCode(t, out), out)
	}

	// Verify file1.txt was NOT modified (rolled back)
	c1, _ := os.ReadFile(p1)
	if string(c1) != "orig1\n" {
		t.Fatalf("file1.txt changed despite patch failure: %q", string(c1))
	}
	c2, _ := os.ReadFile(p2)
	if string(c2) != "orig2\n" {
		t.Fatalf("file2.txt changed despite patch failure: %q", string(c2))
	}
}

func TestPatch_ApplyUnifiedDiff(t *testing.T) {
	ts, root := newTestToolset(t)

	// Single hunk test
	p1 := filepath.Join(root, "foo.txt")
	mustWrite(t, p1, "alpha\nbeta\ngamma\ndelta\n")

	diffSingle := `--- a/foo.txt
+++ b/foo.txt
@@ -1,4 +1,4 @@
 alpha
-beta
+beta_modified
 gamma
 delta
`
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: diffSingle}))
	if bad {
		t.Fatalf("apply single hunk: %s", out)
	}

	content, err := os.ReadFile(p1)
	if err != nil {
		t.Fatalf("read foo.txt: %v", err)
	}
	wantSingle := "alpha\nbeta_modified\ngamma\ndelta\n"
	if string(content) != wantSingle {
		t.Fatalf("foo.txt content = %q, want %q", string(content), wantSingle)
	}

	res := decodeResult(t, out)
	if res["path"] != "foo.txt" {
		t.Fatalf("res[path] = %v, want foo.txt", res["path"])
	}
	if res["action"] != "modified" {
		t.Fatalf("res[action] = %v, want modified", res["action"])
	}
	if v, ok := res["version"].(string); !ok || v == "" {
		t.Fatalf("res[version] missing or empty: %v", res["version"])
	}

	// Multi-hunk test
	p2 := filepath.Join(root, "multi.txt")
	mustWrite(t, p2, "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\n")

	diffMulti := `--- a/multi.txt
+++ b/multi.txt
@@ -2,3 +2,3 @@
 l2
-l3
+l3_new
 l4
@@ -7,3 +7,3 @@
 l7
-l8
+l8_new
 l9
`
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: diffMulti}))
	if bad {
		t.Fatalf("apply multi hunk: %s", out)
	}

	content, err = os.ReadFile(p2)
	if err != nil {
		t.Fatalf("read multi.txt: %v", err)
	}
	wantMulti := "l1\nl2\nl3_new\nl4\nl5\nl6\nl7\nl8_new\nl9\nl10\n"
	if string(content) != wantMulti {
		t.Fatalf("multi.txt content = %q, want %q", string(content), wantMulti)
	}

	// CRLF line ending preservation test
	p3 := filepath.Join(root, "crlf.txt")
	mustWrite(t, p3, "line1\r\nline2\r\nline3\r\n")

	diffCRLF := `--- a/crlf.txt
+++ b/crlf.txt
@@ -1,3 +1,3 @@
 line1
-line2
+line2_crlf
 line3
`
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: diffCRLF}))
	if bad {
		t.Fatalf("apply crlf: %s", out)
	}
	content, err = os.ReadFile(p3)
	if err != nil {
		t.Fatalf("read crlf.txt: %v", err)
	}
	wantCRLF := "line1\r\nline2_crlf\r\nline3\r\n"
	if string(content) != wantCRLF {
		t.Fatalf("crlf.txt content = %q, want %q", string(content), wantCRLF)
	}
}

func TestPatch_FuzzTolerance(t *testing.T) {
	ts, root := newTestToolset(t)

	// File has two extra lines at the top compared to when the diff was generated (offset drift = +2)
	p := filepath.Join(root, "drift.txt")
	initialContent := "extra1\nextra2\ntarget_start\nold_val\ntarget_end\n"
	mustWrite(t, p, initialContent)

	// The diff was generated at line 1
	diff := `--- a/drift.txt
+++ b/drift.txt
@@ -1,3 +1,3 @@
 target_start
-old_val
+new_val
 target_end
`

	// Fuzz 0: must fail because target_start is at line 3, not line 1
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, map[string]any{
		"patch":       diff,
		"fuzz_margin": 0,
	}))
	if !bad || errCode(t, out) != CodeEditConflict {
		t.Fatalf("Fuzz:0 expected EDIT_CONFLICT, got bad=%v out=%s", bad, out)
	}
	// Verify file was NOT modified
	b, _ := os.ReadFile(p)
	if string(b) != initialContent {
		t.Fatalf("file changed on failed fuzz 0: %q", string(b))
	}

	// Fuzz 1: drift is 2, so fuzz 1 must still fail
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, map[string]any{
		"patch":       diff,
		"fuzz_margin": 1,
	}))
	if !bad || errCode(t, out) != CodeEditConflict {
		t.Fatalf("Fuzz:1 expected EDIT_CONFLICT, got bad=%v out=%s", bad, out)
	}
	b, _ = os.ReadFile(p)
	if string(b) != initialContent {
		t.Fatalf("file changed on failed fuzz 1: %q", string(b))
	}

	// Fuzz 2: drift is 2, so fuzz 2 matches!
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{
		Patch:      diff,
		FuzzMargin: 2,
	}))
	if bad {
		t.Fatalf("Fuzz:2 failed: %s", out)
	}

	b, _ = os.ReadFile(p)
	want := "extra1\nextra2\ntarget_start\nnew_val\ntarget_end\n"
	if string(b) != want {
		t.Fatalf("file after fuzz 2 = %q, want %q", string(b), want)
	}

	// Negative drift test: content was removed, so hunk at line 4 is actually at line 2 (drift = -2)
	pNeg := filepath.Join(root, "drift_neg.txt")
	mustWrite(t, pNeg, "start\ntarget\nold_neg\nend\n")
	diffNeg := `--- a/drift_neg.txt
+++ b/drift_neg.txt
@@ -4,3 +4,3 @@
 target
-old_neg
+new_neg
 end
`
	// Fuzz 1 fails
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, map[string]any{
		"patch":       diffNeg,
		"fuzz_margin": 1,
	}))
	if !bad || errCode(t, out) != CodeEditConflict {
		t.Fatalf("negative drift Fuzz:1 expected EDIT_CONFLICT, got bad=%v out=%s", bad, out)
	}

	// Fuzz 2 succeeds
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{
		Patch:      diffNeg,
		FuzzMargin: 2,
	}))
	if bad {
		t.Fatalf("negative drift Fuzz:2 failed: %s", out)
	}
	b, _ = os.ReadFile(pNeg)
	wantNeg := "start\ntarget\nnew_neg\nend\n"
	if string(b) != wantNeg {
		t.Fatalf("file after negative drift fuzz 2 = %q, want %q", string(b), wantNeg)
	}
}

func TestPatch_PathTraversalDefense(t *testing.T) {
	ts, _ := newTestToolset(t)

	// 1. Relative path escape: ..
	diffEscapeRel := `--- a/../escaped.txt
+++ b/../escaped.txt
@@ -1,1 +1,1 @@
-old
+new
`
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: diffEscapeRel}))
	if !bad || errCode(t, out) != CodePathEscape {
		t.Fatalf("relative escape: got bad=%v code=%s out=%s", bad, errCode(t, out), out)
	}

	// 2. Absolute path escape outside workspace
	diffEscapeAbs := `--- /etc/passwd
+++ /etc/passwd
@@ -1,1 +1,1 @@
-root
+hacked
`
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: diffEscapeAbs}))
	if !bad || errCode(t, out) != CodePathEscape {
		t.Fatalf("absolute escape: got bad=%v code=%s out=%s", bad, errCode(t, out), out)
	}

	// 3. Protected control subtree: .git
	diffGit := `--- a/.git/config
+++ b/.git/config
@@ -1,1 +1,1 @@
-old
+new
`
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: diffGit}))
	if !bad || errCode(t, out) != CodeProtectedPath {
		t.Fatalf(".git target: got bad=%v code=%s out=%s", bad, errCode(t, out), out)
	}

	// 4. Protected control subtree: .dos
	diffDos := `--- a/.dos/journal
+++ b/.dos/journal
@@ -1,1 +1,1 @@
-old
+new
`
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: diffDos}))
	if !bad || errCode(t, out) != CodeProtectedPath {
		t.Fatalf(".dos target: got bad=%v code=%s out=%s", bad, errCode(t, out), out)
	}
}

func TestPatch_CASConflict(t *testing.T) {
	ts, root := newTestToolset(t)
	p := filepath.Join(root, "cas.txt")
	mustWrite(t, p, "line1\nline2\n")

	v1 := observedVersion(t, ts, "cas.txt")
	if v1 == "" {
		t.Fatal("initial observed version is empty")
	}

	// Peer modifies the file
	mustWrite(t, p, "line1_peer\nline2\n")
	v2 := observedVersion(t, ts, "cas.txt")
	if v2 == v1 {
		t.Fatal("version did not change after peer edit")
	}

	// Apply patch with stale expected_version v1
	diff := `--- a/cas.txt
+++ b/cas.txt
@@ -1,2 +1,2 @@
-line1
+line1_mine
 line2
`
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{
		Patch:           diff,
		ExpectedVersion: v1,
	}))
	if !bad || errCode(t, out) != CodeStaleVersion {
		t.Fatalf("stale expected_version: want FS_STALE_VERSION, got bad=%v code=%s out=%s", bad, errCode(t, out), out)
	}

	// Content must still be the peer's version
	b, _ := os.ReadFile(p)
	if string(b) != "line1_peer\nline2\n" {
		t.Fatalf("file clobbered despite stale version: %q", string(b))
	}

	// Apply patch with current expected_version v2
	diffFresh := `--- a/cas.txt
+++ b/cas.txt
@@ -1,2 +1,2 @@
-line1_peer
+line1_mine
 line2
`
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{
		Patch:           diffFresh,
		ExpectedVersion: v2,
	}))
	if bad {
		t.Fatalf("fresh expected_version failed: %s", out)
	}

	b, _ = os.ReadFile(p)
	if string(b) != "line1_mine\nline2\n" {
		t.Fatalf("file content = %q, want line1_mine\nline2\n", string(b))
	}
}

func TestPatch_FileCreationDeletion(t *testing.T) {
	ts, root := newTestToolset(t)

	// 1. File creation via /dev/null
	createDiff := `--- /dev/null
+++ b/created.txt
@@ -0,0 +1,3 @@
+first line
+second line
+third line
`
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: createDiff}))
	if bad {
		t.Fatalf("create patch failed: %s", out)
	}

	createdPath := filepath.Join(root, "created.txt")
	b, err := os.ReadFile(createdPath)
	if err != nil {
		t.Fatalf("read created.txt: %v", err)
	}
	wantCreated := "first line\nsecond line\nthird line\n"
	if string(b) != wantCreated {
		t.Fatalf("created.txt content = %q, want %q", string(b), wantCreated)
	}

	// 2. Creating an already existing file must refuse with ALREADY_EXISTS
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: createDiff}))
	if !bad || errCode(t, out) != CodeExists {
		t.Fatalf("duplicate create: want ALREADY_EXISTS, got bad=%v code=%s out=%s", bad, errCode(t, out), out)
	}

	// 3. File deletion via /dev/null
	deleteDiff := `--- a/created.txt
+++ /dev/null
@@ -1,3 +0,0 @@
-first line
-second line
-third line
`
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: deleteDiff}))
	if bad {
		t.Fatalf("delete patch failed: %s", out)
	}

	// File must no longer exist
	_, err = os.Stat(createdPath)
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("deleted file still exists: err = %v", err)
	}

	// 4. Deleting a non-existent file must refuse with NOT_FOUND
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, PatchArgs{Patch: deleteDiff}))
	if !bad || errCode(t, out) != CodeNotFound {
		t.Fatalf("delete missing: want NOT_FOUND, got bad=%v code=%s out=%s", bad, errCode(t, out), out)
	}
}
