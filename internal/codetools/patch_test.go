package codetools

import (
	"context"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"testing"
)

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
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{Patch: diffSingle}))
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
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{Patch: diffMulti}))
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
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{Patch: diffCRLF}))
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
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{
		Patch: diff,
		Fuzz:  0,
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
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{
		Patch: diff,
		Fuzz:  1,
	}))
	if !bad || errCode(t, out) != CodeEditConflict {
		t.Fatalf("Fuzz:1 expected EDIT_CONFLICT, got bad=%v out=%s", bad, out)
	}
	b, _ = os.ReadFile(p)
	if string(b) != initialContent {
		t.Fatalf("file changed on failed fuzz 1: %q", string(b))
	}

	// Fuzz 2: drift is 2, so fuzz 2 matches!
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{
		Patch: diff,
		Fuzz:  2,
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
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{
		Patch: diffNeg,
		Fuzz:  1,
	}))
	if !bad || errCode(t, out) != CodeEditConflict {
		t.Fatalf("negative drift Fuzz:1 expected EDIT_CONFLICT, got bad=%v out=%s", bad, out)
	}

	// Fuzz 2 succeeds
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{
		Patch: diffNeg,
		Fuzz:  2,
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
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{Patch: diffEscapeRel}))
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
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{Patch: diffEscapeAbs}))
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
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{Patch: diffGit}))
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
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{Patch: diffDos}))
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
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{
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
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{
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
	out, bad := ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{Patch: createDiff}))
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
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{Patch: createDiff}))
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
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{Patch: deleteDiff}))
	if bad {
		t.Fatalf("delete patch failed: %s", out)
	}

	// File must no longer exist
	_, err = os.Stat(createdPath)
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("deleted file still exists: err = %v", err)
	}

	// 4. Deleting a non-existent file must refuse with NOT_FOUND
	out, bad = ts.ApplyPatch(context.Background(), argsOf(t, ApplyPatchArgs{Patch: deleteDiff}))
	if !bad || errCode(t, out) != CodeNotFound {
		t.Fatalf("delete missing: want NOT_FOUND, got bad=%v code=%s out=%s", bad, errCode(t, out), out)
	}
}
