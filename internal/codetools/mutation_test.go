package codetools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestWriteCreateOverwriteAndModes(t *testing.T) {
	ts, root := newTestToolset(t)
	out, bad := ts.write(context.Background(), argsOf(t, WriteArgs{FilePath: "dir/a.txt", Content: "one", Mode: "create"}))
	if bad {
		t.Fatalf("create: %s", out)
	}
	if b, _ := os.ReadFile(filepath.Join(root, "dir", "a.txt")); string(b) != "one" {
		t.Fatalf("created = %q", b)
	}
	if out, bad = ts.write(context.Background(), argsOf(t, WriteArgs{FilePath: "dir/a.txt", Content: "two", Mode: "create"})); !bad || errCode(t, out) != CodeExists {
		t.Fatalf("second create = %s", out)
	}
	if out, bad = ts.write(context.Background(), argsOf(t, WriteArgs{FilePath: "missing.txt", Content: "x", Mode: "overwrite", ExpectedVersion: "fv1:missing"})); !bad || errCode(t, out) != CodeStaleVersion {
		t.Fatalf("missing overwrite = %s", out)
	}
	version := observedVersion(t, ts, "dir/a.txt")
	if out, bad = ts.write(context.Background(), argsOf(t, WriteArgs{FilePath: "dir/a.txt", Content: "two", Mode: "overwrite", ExpectedVersion: version})); bad {
		t.Fatalf("overwrite: %s", out)
	}
	if next, _ := decodeResult(t, out)["version"].(string); next == "" || next == version {
		t.Fatalf("overwrite version = %q, want non-empty changed token", next)
	}
}

func TestWriteUpsertCreatesButRequiresAnObservationToReplace(t *testing.T) {
	ts, root := newTestToolset(t)
	out, bad := ts.write(context.Background(), argsOf(t, WriteArgs{FilePath: "a.txt", Content: "one", Mode: "upsert"}))
	if bad {
		t.Fatalf("upsert create: %s", out)
	}
	if out, bad = ts.write(context.Background(), argsOf(t, WriteArgs{FilePath: "a.txt", Content: "blind", Mode: "upsert"})); !bad || errCode(t, out) != CodeStaleVersion {
		t.Fatalf("blind upsert replace = %s", out)
	}
	if got, _ := os.ReadFile(filepath.Join(root, "a.txt")); string(got) != "one" {
		t.Fatalf("blind upsert changed bytes to %q", got)
	}
	version := observedVersion(t, ts, "a.txt")
	if out, bad = ts.write(context.Background(), argsOf(t, WriteArgs{FilePath: "a.txt", Content: "two", Mode: "upsert", ExpectedVersion: version})); bad {
		t.Fatalf("observed upsert replace: %s", out)
	}
}

func TestEditExactConflictAndReplaceAll(t *testing.T) {
	ts, root := newTestToolset(t)
	p := filepath.Join(root, "a.txt")
	mustWrite(t, p, "old old")
	version := observedVersion(t, ts, "a.txt")
	out, bad := ts.edit(context.Background(), argsOf(t, EditArgs{FilePath: "a.txt", OldString: "old", NewString: "new", ExpectedVersion: version}))
	if !bad || errCode(t, out) != CodeEditConflict {
		t.Fatalf("ambiguous = %s", out)
	}
	out, bad = ts.edit(context.Background(), argsOf(t, EditArgs{FilePath: "a.txt", OldString: "old", NewString: "new", ReplaceAll: true, ExpectedVersion: version}))
	if bad {
		t.Fatalf("replace all: %s", out)
	}
	if b, _ := os.ReadFile(p); string(b) != "new new" {
		t.Fatalf("edited = %q", b)
	}
	version = observedVersion(t, ts, "a.txt")
	out, bad = ts.edit(context.Background(), argsOf(t, EditArgs{FilePath: "a.txt", OldString: "absent", NewString: "x", ExpectedVersion: version}))
	if !bad || errCode(t, out) != CodeEditConflict {
		t.Fatalf("mismatch = %s", out)
	}
}

func TestEditRefusesAStaleObservedVersionWithoutChangingPeerBytes(t *testing.T) {
	ts, root := newTestToolset(t)
	p := filepath.Join(root, "a.txt")
	mustWrite(t, p, "anchor old")

	readOut, bad := ts.read(context.Background(), argsOf(t, ReadArgs{FilePath: "a.txt"}))
	if bad {
		t.Fatalf("Read: %s", readOut)
	}
	version, _ := decodeResult(t, readOut)["version"].(string)
	if version == "" {
		t.Fatal("Read returned an empty observed version")
	}

	mustWrite(t, p, "peer old")
	out, bad := ts.edit(context.Background(), argsOf(t, map[string]any{
		"file_path":        "a.txt",
		"old_string":       "old",
		"new_string":       "mine",
		"expected_version": version,
	}))
	if !bad || errCode(t, out) != CodeStaleVersion {
		t.Fatalf("stale Edit = %s, want FS_STALE_VERSION", out)
	}
	if got, err := os.ReadFile(p); err != nil || string(got) != "peer old" {
		t.Fatalf("final bytes = %q, %v; want peer bytes unchanged", got, err)
	}
}

func TestOverwriteRefusesAStaleObservedVersionWithoutChangingPeerBytes(t *testing.T) {
	ts, root := newTestToolset(t)
	p := filepath.Join(root, "a.txt")
	mustWrite(t, p, "mine")
	version := observedVersion(t, ts, "a.txt")
	mustWrite(t, p, "peer")

	out, bad := ts.write(context.Background(), argsOf(t, WriteArgs{
		FilePath: "a.txt", Content: "overwrite", Mode: "overwrite", ExpectedVersion: version,
	}))
	if !bad || errCode(t, out) != CodeStaleVersion {
		t.Fatalf("stale overwrite = %s, want FS_STALE_VERSION", out)
	}
	if got, err := os.ReadFile(p); err != nil || string(got) != "peer" {
		t.Fatalf("final bytes = %q, %v; want peer bytes unchanged", got, err)
	}
}

func TestMutationTraversalSymlinkCancellationAndBound(t *testing.T) {
	ts, root := newTestToolset(t)
	for _, tc := range []struct {
		name string
		run  func() ([]byte, bool)
		code string
	}{
		{"write traversal", func() ([]byte, bool) {
			return ts.write(context.Background(), argsOf(t, WriteArgs{FilePath: "../escape", Content: "x", Mode: "create"}))
		}, CodePathEscape},
		{"edit traversal", func() ([]byte, bool) {
			return ts.edit(context.Background(), argsOf(t, EditArgs{FilePath: "../escape", OldString: "x", NewString: "y", ExpectedVersion: "fv1:test"}))
		}, CodePathEscape},
		{"canceled", func() ([]byte, bool) {
			return ts.write(deadCtx(), argsOf(t, WriteArgs{FilePath: "x", Content: "x", Mode: "create"}))
		}, CodeCanceled},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out, bad := tc.run()
			if !bad || errCode(t, out) != tc.code {
				t.Fatalf("got %s", out)
			}
		})
	}
	bounded, _ := New(Config{Root: root, Limits: Limits{MaxWriteBytes: 3}})
	out, bad := bounded.write(context.Background(), argsOf(t, WriteArgs{FilePath: "big", Content: "four", Mode: "create"}))
	if !bad || errCode(t, out) != CodeTooLarge {
		t.Fatalf("bound = %s", out)
	}
	outside := t.TempDir()
	mustWrite(t, filepath.Join(outside, "secret"), "secret")
	link := filepath.Join(root, "link")
	if err := os.Symlink(filepath.Join(outside, "secret"), link); err != nil {
		if runtime.GOOS == "windows" {
			t.Skip("symlink privilege")
		}
		t.Fatal(err)
	}
	out, bad = ts.write(context.Background(), argsOf(t, WriteArgs{FilePath: "link", Content: "no", Mode: "overwrite", ExpectedVersion: "fv1:test"}))
	if !bad || !strings.Contains(string(out), CodeSymlinkEscape) {
		t.Fatalf("symlink write = %s", out)
	}
}

func TestConcurrentEditsFromOneObservationHaveOneWinner(t *testing.T) {
	ts, root := newTestToolset(t)
	p := filepath.Join(root, "a.txt")
	mustWrite(t, p, "old")
	version := observedVersion(t, ts, "a.txt")
	bodies := [][]byte{
		argsOf(t, EditArgs{FilePath: "a.txt", OldString: "old", NewString: "one", ExpectedVersion: version}),
		argsOf(t, EditArgs{FilePath: "a.txt", OldString: "old", NewString: "two", ExpectedVersion: version}),
	}
	start := make(chan struct{})
	results := make(chan struct {
		out []byte
		bad bool
	}, len(bodies))
	var wg sync.WaitGroup
	for _, body := range bodies {
		wg.Add(1)
		go func(body []byte) {
			defer wg.Done()
			<-start
			out, bad := ts.edit(context.Background(), body)
			results <- struct {
				out []byte
				bad bool
			}{out: out, bad: bad}
		}(body)
	}
	close(start)
	wg.Wait()
	close(results)
	winners, stale := 0, 0
	for result := range results {
		if !result.bad {
			winners++
		} else if errCode(t, result.out) == CodeStaleVersion {
			stale++
		} else {
			t.Fatalf("concurrent Edit = %s", result.out)
		}
	}
	if winners != 1 || stale != 1 {
		t.Fatalf("concurrent results: winners=%d stale=%d", winners, stale)
	}
	if got, _ := os.ReadFile(p); string(got) != "one" && string(got) != "two" {
		t.Fatalf("final bytes = %q", got)
	}
}

func TestMutationRefusesReplacedIdentityAndFinalSymlink(t *testing.T) {
	ts, root := newTestToolset(t)
	p := filepath.Join(root, "a.txt")
	mustWrite(t, p, "same old")
	version := observedVersion(t, ts, "a.txt")
	replacement := filepath.Join(root, "replacement.txt")
	mustWrite(t, replacement, "same old")
	if err := os.Remove(p); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(replacement, p); err != nil {
		t.Fatal(err)
	}
	out, bad := ts.edit(context.Background(), argsOf(t, EditArgs{FilePath: "a.txt", OldString: "old", NewString: "mine", ExpectedVersion: version}))
	if !bad || errCode(t, out) != CodeStaleVersion {
		t.Fatalf("identity replacement Edit = %s", out)
	}

	target := filepath.Join(root, "target.txt")
	mustWrite(t, target, "target old")
	version = observedVersion(t, ts, "a.txt")
	if err := os.Remove(p); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, p); err != nil {
		if runtime.GOOS == "windows" {
			t.Skip("symlink privilege")
		}
		t.Fatal(err)
	}
	out, bad = ts.edit(context.Background(), argsOf(t, EditArgs{FilePath: "a.txt", OldString: "old", NewString: "mine", ExpectedVersion: version}))
	if !bad || errCode(t, out) != CodeSymlinkEscape {
		t.Fatalf("final symlink Edit = %s", out)
	}
	if got, _ := os.ReadFile(target); string(got) != "target old" {
		t.Fatalf("symlink target changed to %q", got)
	}
}

func TestMutationRefusesAnAncestorSymlinkSwappedAfterRead(t *testing.T) {
	ts, root := newTestToolset(t)
	left := filepath.Join(root, "left")
	right := filepath.Join(root, "right")
	mustWrite(t, filepath.Join(left, "a.txt"), "left old")
	mustWrite(t, filepath.Join(right, "a.txt"), "right old")
	alias := filepath.Join(root, "alias")
	if err := os.Symlink(left, alias); err != nil {
		if runtime.GOOS == "windows" {
			t.Skip("symlink privilege")
		}
		t.Fatal(err)
	}
	version := observedVersion(t, ts, "alias/a.txt")
	if err := os.Remove(alias); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(right, alias); err != nil {
		t.Fatal(err)
	}

	out, bad := ts.edit(context.Background(), argsOf(t, EditArgs{
		FilePath: "alias/a.txt", OldString: "old", NewString: "mine", ExpectedVersion: version,
	}))
	if !bad || errCode(t, out) != CodeStaleVersion {
		t.Fatalf("ancestor-swap Edit = %s, want FS_STALE_VERSION", out)
	}
	if got, _ := os.ReadFile(filepath.Join(right, "a.txt")); string(got) != "right old" {
		t.Fatalf("swapped ancestor redirected mutation: %q", got)
	}
}

func TestOwnedLoopCatalogAndRegisteredEnginesCarryVersions(t *testing.T) {
	ts, root := newTestToolset(t)
	p := filepath.Join(root, "a.txt")
	mustWrite(t, p, "anchor old")
	ts.RegisterEngines()

	var editSchema map[string]any
	for _, def := range Catalog() {
		if def.Name == ToolEdit {
			if err := json.Unmarshal(def.Parameters, &editSchema); err != nil {
				t.Fatal(err)
			}
		}
	}
	properties, _ := editSchema["properties"].(map[string]any)
	if _, ok := properties["expected_version"]; !ok {
		t.Fatal("model-facing Edit schema omits expected_version")
	}

	execute := func(tool string, args any) ([]byte, bool) {
		t.Helper()
		body := argsOf(t, args)
		call := &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: body}, Meta: CallMeta(tool, "owned-loop")}
		if verdict := ts.Adjudicate(context.Background(), call); verdict.Kind != abi.VerdictAllow {
			t.Fatalf("%s verdict = %+v", tool, verdict)
		}
		engine := abi.Engine(call.Engine)
		if engine == nil {
			t.Fatalf("%s engine %q was not registered", tool, call.Engine)
		}
		result, err := engine.Complete(context.Background(), call)
		if err != nil {
			t.Fatalf("%s dispatch: %v", tool, err)
		}
		return bytesOf(context.Background(), result.Payload), result.Status == abi.StatusError
	}

	readOut, bad := execute(ToolRead, ReadArgs{FilePath: "a.txt"})
	if bad {
		t.Fatalf("registered Read: %s", readOut)
	}
	version, _ := decodeResult(t, readOut)["version"].(string)
	mustWrite(t, p, "peer old")
	staleOut, bad := execute(ToolEdit, EditArgs{FilePath: "a.txt", OldString: "old", NewString: "mine", ExpectedVersion: version})
	if !bad || errCode(t, staleOut) != CodeStaleVersion {
		t.Fatalf("registered stale Edit = %s", staleOut)
	}
	freshOut, _ := execute(ToolRead, ReadArgs{FilePath: "a.txt"})
	freshVersion, _ := decodeResult(t, freshOut)["version"].(string)
	editOut, bad := execute(ToolEdit, EditArgs{FilePath: "a.txt", OldString: "old", NewString: "mine", ExpectedVersion: freshVersion})
	if bad {
		t.Fatalf("registered fresh Edit = %s", editOut)
	}
	if next, _ := decodeResult(t, editOut)["version"].(string); next == "" || next == freshVersion {
		t.Fatalf("registered Edit version = %q", next)
	}
	if got, _ := os.ReadFile(p); string(got) != "peer mine" {
		t.Fatalf("final bytes = %q", got)
	}
}

func TestMutationRungDefaultDenyAndCacheContract(t *testing.T) {
	ts, _ := newTestToolset(t)
	mk := func(tool string, meta map[string]string) *abi.ToolCall {
		return &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: argsOf(t, WriteArgs{FilePath: "a", Content: "x", Mode: "create"})}, Meta: meta}
	}
	v := ts.Adjudicate(context.Background(), mk(ToolWrite, CallMeta(ToolWrite, "p")))
	if v.Kind != abi.VerdictAllow || v.By != RungName {
		t.Fatalf("allow verdict=%+v", v)
	}
	denied, _ := New(Config{Root: t.TempDir(), Policy: Policy{Allow: map[string]bool{ToolRead: true}}})
	v = denied.Adjudicate(context.Background(), mk(ToolWrite, CallMeta(ToolWrite, "p")))
	if v.Kind != abi.VerdictDeny || v.Meta["code"] != CodeDefaultDeny {
		t.Fatalf("default deny=%+v", v)
	}
	v = ts.Adjudicate(context.Background(), mk(ToolWrite, map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}))
	if v.Kind != abi.VerdictDeny || v.Meta["code"] != CodeCacheScope {
		t.Fatalf("cache scope=%+v", v)
	}
	escape := mk(ToolWrite, map[string]string{"readOnlyHint": "true", "idempotentHint": "true"})
	escape.Args.Inline = argsOf(t, WriteArgs{FilePath: "../escape", Content: "x", Mode: "create"})
	v = ts.Adjudicate(context.Background(), escape)
	if v.Meta["code"] != CodePathEscape {
		t.Fatalf("canonicalization did not precede cache matching: %+v", v)
	}
}

func TestMutationsRefuseProtectedControlSubtrees(t *testing.T) {
	ts, root := newTestToolset(t)
	for _, rel := range []string{".git/config", ".dos/leases.json"} {
		path := filepath.Join(root, filepath.FromSlash(rel))
		mustWrite(t, path, "protected")
		out, bad := ts.write(context.Background(), argsOf(t, WriteArgs{FilePath: rel, Content: "changed", Mode: "overwrite", ExpectedVersion: "irrelevant"}))
		if !bad || errCode(t, out) != CodeProtectedPath {
			t.Fatalf("write %s = %s", rel, out)
		}
		out, bad = ts.edit(context.Background(), argsOf(t, EditArgs{FilePath: rel, OldString: "protected", NewString: "changed", ExpectedVersion: "irrelevant"}))
		if !bad || errCode(t, out) != CodeProtectedPath {
			t.Fatalf("edit %s = %s", rel, out)
		}
		if got, err := os.ReadFile(path); err != nil || string(got) != "protected" {
			t.Fatalf("protected bytes %s = %q, %v", rel, got, err)
		}
	}
}

func TestEditConflictRecoveryRegisteredEngine(t *testing.T) {
	const original = "header\nfirst\nvalue=old\nsecond\nvalue=old\nfooter\n"
	const unique = "first\nvalue=old"
	for _, tc := range []struct {
		name, old, retry, detail string
		unresolved               bool
	}{
		{"missing", "value = old", unique, "matched 0 occurrences", false},
		{"ambiguous", "value=old", unique, "matched 2 occurrences", false},
		{"missing-unresolved", "value = old", "still absent", "matched 0 occurrences", true},
		{"ambiguous-unresolved", "value=old", "value=old", "matched 2 occurrences", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ts, root := newTestToolset(t)
			p := filepath.Join(root, "a.txt")
			mustWrite(t, p, original)
			ts.RegisterEngines()
			reads, edits := 0, 0
			execute := func(tool string, args any) ([]byte, bool) {
				t.Helper()
				if tool == ToolRead {
					reads++
					a := args.(ReadArgs)
					if a.FilePath != "a.txt" || a.Offset != 2 || a.Limit != 2 {
						t.Fatalf("unbounded or redirected Read: %+v", a)
					}
				} else if tool == ToolEdit {
					edits++
				}
				call := &abi.ToolCall{Tool: tool, Args: abi.Ref{Kind: abi.RefInline, Inline: argsOf(t, args)}, Meta: CallMeta(tool, "conflict-recovery")}
				if v := ts.Adjudicate(context.Background(), call); v.Kind != abi.VerdictAllow {
					t.Fatalf("%s verdict = %+v", tool, v)
				}
				engine := abi.Engine(call.Engine)
				if engine == nil {
					t.Fatalf("unregistered engine %q", call.Engine)
				}
				result, err := engine.Complete(context.Background(), call)
				if err != nil {
					t.Fatal(err)
				}
				return bytesOf(context.Background(), result.Payload), result.Status == abi.StatusError
			}
			assertBytes := func(want string) {
				t.Helper()
				got, err := os.ReadFile(p)
				if err != nil || string(got) != want {
					t.Fatalf("file bytes = %q, error %v; want %q", got, err, want)
				}
			}
			read := func() string {
				t.Helper()
				out, bad := execute(ToolRead, ReadArgs{FilePath: "a.txt", Offset: 2, Limit: 2})
				if bad {
					t.Fatalf("Read = %s", out)
				}
				r := decodeResult(t, out)
				if r["content"] != unique || r["bytes"] != float64(len(unique)) || r["truncated"] != true {
					t.Fatalf("Read bounds = %s", out)
				}
				version, _ := r["version"].(string)
				if version == "" {
					t.Fatal("Read omitted version")
				}
				return version
			}
			version := read()
			out, bad := execute(ToolEdit, EditArgs{FilePath: "a.txt", OldString: tc.old, NewString: "unintended", ExpectedVersion: version})
			if !bad || errCode(t, out) != CodeEditConflict {
				t.Fatalf("conflict = %s", out)
			}
			assertBytes(original)
			for _, fragment := range []string{tc.detail, "Read", "same authorized file_path", "offset and limit", "expected_version", "unique", "explicit", "stop", "not changed"} {
				if !strings.Contains(string(out), fragment) {
					t.Errorf("diagnostic lacks %q: %s", fragment, out)
				}
			}
			if strings.Contains(string(out), tc.old) || strings.Contains(string(out), "unintended") {
				t.Errorf("diagnostic echoed edit content: %s", out)
			}
			fresh := read()
			if fresh != version {
				t.Fatal("conflict changed version")
			}
			assertBytes(original)
			out, bad = execute(ToolEdit, EditArgs{FilePath: "a.txt", OldString: tc.retry, NewString: "first\nvalue=new", ExpectedVersion: fresh})
			if tc.unresolved {
				if !bad || errCode(t, out) != CodeEditConflict {
					t.Fatalf("unresolved retry = %s", out)
				}
				assertBytes(original)
			} else {
				if bad {
					t.Fatalf("explicit unique retry = %s", out)
				}
				assertBytes("header\nfirst\nvalue=new\nsecond\nvalue=old\nfooter\n")
			}
			if reads != 2 || edits != 2 {
				t.Fatalf("calls: Read=%d Edit=%d; want initial read/edit + one reread/retry", reads, edits)
			}
		})
	}
}
