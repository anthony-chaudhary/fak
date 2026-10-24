package safecommit

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipref"
)

type peerHistoryEntry struct {
	mode string
	body string
}

type peerHistoryFixture struct {
	t   *testing.T
	dir string
	git func(...string) string
}

func newPeerHistoryFixture(t *testing.T) *peerHistoryFixture {
	t.Helper()
	dir := t.TempDir()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	t.Cleanup(cancel)
	f := &peerHistoryFixture{t: t, dir: dir}
	f.git = func(args ...string) string {
		t.Helper()
		out, code, err := realRunner(ctx, dir, args...)
		if err != nil || code != 0 {
			t.Fatalf("fixture git %v: code=%d err=%v output=%s", args, code, err, out)
		}
		return strings.TrimSpace(out)
	}
	f.git("init", "-q")
	f.git("config", "user.name", "Peer History Fixture")
	f.git("config", "user.email", "peer-history@example.invalid")
	return f
}

// commit constructs an exact tree through the fixture index. It never checks out
// a historical tree, which keeps every branch shape independent and explicit.
func (f *peerHistoryFixture) commit(parents []string, message string, entries map[string]peerHistoryEntry) string {
	f.t.Helper()
	f.git("read-tree", "--empty")
	paths := make([]string, 0, len(entries))
	for path := range entries {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	for _, path := range paths {
		entry := entries[path]
		full := filepath.Join(f.dir, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(full), 0o700); err != nil {
			f.t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(entry.body), 0o600); err != nil {
			f.t.Fatal(err)
		}
		blob := f.git("hash-object", "-w", "--", path)
		f.git("update-index", "--add", "--cacheinfo", entry.mode+","+blob+","+path)
	}
	tree := f.git("write-tree")
	args := []string{"commit-tree", tree}
	for _, parent := range parents {
		args = append(args, "-p", parent)
	}
	args = append(args, "-m", message)
	return f.git(args...)
}

func peerHistoryEntries(aMode, aBody, bBody string) map[string]peerHistoryEntry {
	return map[string]peerHistoryEntry{
		"owned/a.txt": {mode: aMode, body: aBody},
		"owned/b.txt": {mode: "100644", body: bBody},
	}
}

func peerHistoryStamp(t *testing.T, peer, start string, scope []string) string {
	t.Helper()
	stamp, err := wipref.EncodeStamp(wipref.Stamp{
		SessionID: peer,
		StartSHA:  start,
		Scope:     scope,
		Buildable: false,
	})
	if err != nil {
		t.Fatal(err)
	}
	return stamp
}

func (f *peerHistoryFixture) refs() string {
	f.t.Helper()
	return f.git("for-each-ref", "--sort=refname", "--format=%(refname) %(objectname)")
}

func (f *peerHistoryFixture) resolve(t *testing.T, run Runner, paths []string) (map[string]gitPeerOwnerInfo, error) {
	t.Helper()
	before := f.refs()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	owners, malformed, err := resolveGitPeerOwnerDetails(ctx, run, f.dir, paths, "self")
	if len(malformed) != 0 {
		t.Fatalf("unexpected malformed refs: %v", malformed)
	}
	if after := f.refs(); after != before {
		t.Fatalf("peer ownership lookup mutated refs:\nbefore=%s\nafter=%s", before, after)
	}
	return owners, err
}

func peerHistoryHasFindObject(args []string) bool {
	for _, arg := range args {
		if strings.HasPrefix(arg, "--find-object=") {
			return true
		}
	}
	return false
}

// fak-test:runtime medium est=30s lane=default
func TestPeerWIPCompletePayloadLandedHistory(t *testing.T) {
	const peer = "peer-history"
	paths := []string{"owned/a.txt", "owned/b.txt"}
	baseEntries := peerHistoryEntries("100644", "base-a\n", "base-b\n")
	payloadEntries := peerHistoryEntries("100644", "payload-a\n", "payload-b\n")
	headEntries := peerHistoryEntries("100644", "head-a\n", "head-b\n")

	t.Run("complete-payload-landed-at-one-later-ancestor", func(t *testing.T) {
		t.Parallel()
		f := newPeerHistoryFixture(t)
		base := f.commit(nil, "base", baseEntries)
		checkpoint := f.commit([]string{base}, peerHistoryStamp(t, peer, base, paths), payloadEntries)
		f.git("update-ref", "refs/fak/wip/"+peer, checkpoint)
		landed := f.commit([]string{base}, "land complete checkpoint payload", payloadEntries)
		head := f.commit([]string{landed}, "advance after landing", headEntries)
		f.git("update-ref", "HEAD", head)

		owners, err := f.resolve(t, realRunner, paths)
		if err != nil {
			t.Fatal(err)
		}
		if len(owners) != 0 {
			t.Fatalf("fully landed immutable checkpoint retained ownership: %+v", owners)
		}
	})

	tests := []struct {
		name  string
		build func(*testing.T, *peerHistoryFixture) (checkpoint, head string)
		scope []string
		msg   func(*testing.T, string) string
	}{
		{
			name: "delta-has-file-outside-stamped-scope",
			build: func(t *testing.T, f *peerHistoryFixture) (string, string) {
				baseWithExtra := peerHistoryEntries("100644", "base-a\n", "base-b\n")
				baseWithExtra["outside.txt"] = peerHistoryEntry{mode: "100644", body: "base-outside\n"}
				payloadWithExtra := peerHistoryEntries("100644", "payload-a\n", "payload-b\n")
				payloadWithExtra["outside.txt"] = peerHistoryEntry{mode: "100644", body: "payload-outside\n"}
				base := f.commit(nil, "base", baseWithExtra)
				checkpoint := f.commit([]string{base}, peerHistoryStamp(t, peer, base, paths), payloadWithExtra)
				landedScopedOnly := f.commit([]string{base}, "land only stamped entries", map[string]peerHistoryEntry{
					"owned/a.txt": {mode: "100644", body: "payload-a\n"},
					"owned/b.txt": {mode: "100644", body: "payload-b\n"},
					"outside.txt": {mode: "100644", body: "base-outside\n"},
				})
				headWithExtra := peerHistoryEntries("100644", "head-a\n", "head-b\n")
				headWithExtra["outside.txt"] = peerHistoryEntry{mode: "100644", body: "head-outside\n"}
				return checkpoint, f.commit([]string{landedScopedOnly}, "advance", headWithExtra)
			},
		},
		{
			name: "stamped-scope-has-unchanged-member",
			build: func(t *testing.T, f *peerHistoryFixture) (string, string) {
				base := f.commit(nil, "base", baseEntries)
				checkpointEntries := peerHistoryEntries("100644", "payload-a\n", "base-b\n")
				checkpoint := f.commit([]string{base}, peerHistoryStamp(t, peer, base, paths), checkpointEntries)
				landed := f.commit([]string{base}, "land changed member", checkpointEntries)
				return checkpoint, f.commit([]string{landed}, "advance", headEntries)
			},
		},
		{
			name: "partial-match",
			build: func(t *testing.T, f *peerHistoryFixture) (string, string) {
				base := f.commit(nil, "base", baseEntries)
				checkpoint := f.commit([]string{base}, peerHistoryStamp(t, peer, base, paths), payloadEntries)
				partial := f.commit([]string{base}, "only a lands", peerHistoryEntries("100644", "payload-a\n", "base-b\n"))
				return checkpoint, f.commit([]string{partial}, "advance", headEntries)
			},
		},
		{
			name: "different-commit-per-entry",
			build: func(t *testing.T, f *peerHistoryFixture) (string, string) {
				base := f.commit(nil, "base", baseEntries)
				checkpoint := f.commit([]string{base}, peerHistoryStamp(t, peer, base, paths), payloadEntries)
				a := f.commit([]string{base}, "a only", peerHistoryEntries("100644", "payload-a\n", "base-b\n"))
				b := f.commit([]string{a}, "b only after a changed", peerHistoryEntries("100644", "head-a\n", "payload-b\n"))
				return checkpoint, f.commit([]string{b}, "advance", headEntries)
			},
		},
		{
			name: "pre-base-match",
			build: func(t *testing.T, f *peerHistoryFixture) (string, string) {
				preBase := f.commit(nil, "old payload", payloadEntries)
				base := f.commit([]string{preBase}, "checkpoint base", baseEntries)
				checkpoint := f.commit([]string{base}, peerHistoryStamp(t, peer, base, paths), payloadEntries)
				return checkpoint, f.commit([]string{base}, "advance without landing", headEntries)
			},
		},
		{
			name: "nonancestor-sibling",
			build: func(t *testing.T, f *peerHistoryFixture) (string, string) {
				base := f.commit(nil, "base", baseEntries)
				checkpoint := f.commit([]string{base}, peerHistoryStamp(t, peer, base, paths), payloadEntries)
				_ = f.commit([]string{base}, "payload on sibling", payloadEntries)
				return checkpoint, f.commit([]string{base}, "main advances", headEntries)
			},
		},
		{
			name: "executable-mode-mismatch",
			build: func(t *testing.T, f *peerHistoryFixture) (string, string) {
				base := f.commit(nil, "base", baseEntries)
				checkpoint := f.commit([]string{base}, peerHistoryStamp(t, peer, base, paths), payloadEntries)
				wrongMode := f.commit([]string{base}, "content lands executable", peerHistoryEntries("100755", "payload-a\n", "payload-b\n"))
				return checkpoint, f.commit([]string{wrongMode}, "advance", peerHistoryEntries("100755", "head-a\n", "head-b\n"))
			},
		},
		{
			name: "malformed-stamp",
			build: func(t *testing.T, f *peerHistoryFixture) (string, string) {
				base := f.commit(nil, "base", baseEntries)
				checkpoint := f.commit([]string{base}, "fak-wip: {malformed", payloadEntries)
				landed := f.commit([]string{base}, "land payload", payloadEntries)
				return checkpoint, f.commit([]string{landed}, "advance", headEntries)
			},
		},
		{
			name: "stamp-session-does-not-match-ref",
			build: func(t *testing.T, f *peerHistoryFixture) (string, string) {
				base := f.commit(nil, "base", baseEntries)
				checkpoint := f.commit([]string{base}, peerHistoryStamp(t, "different-peer", base, paths), payloadEntries)
				landed := f.commit([]string{base}, "land payload", payloadEntries)
				return checkpoint, f.commit([]string{landed}, "advance", headEntries)
			},
		},
		{
			name: "metadata-only-stamp-with-payload",
			build: func(t *testing.T, f *peerHistoryFixture) (string, string) {
				base := f.commit(nil, "base", baseEntries)
				stamp, err := wipref.EncodeStamp(wipref.Stamp{
					SessionID: peer, StartSHA: base, Scope: paths, MetadataOnly: true,
				})
				if err != nil {
					t.Fatal(err)
				}
				checkpoint := f.commit([]string{base}, stamp, payloadEntries)
				landed := f.commit([]string{base}, "land payload", payloadEntries)
				return checkpoint, f.commit([]string{landed}, "advance", headEntries)
			},
		},
		{
			name: "ambiguous-stamp",
			build: func(t *testing.T, f *peerHistoryFixture) (string, string) {
				base := f.commit(nil, "base", baseEntries)
				first := peerHistoryStamp(t, peer, base, paths)
				second := peerHistoryStamp(t, "other-peer", base, paths)
				checkpoint := f.commit([]string{base}, first+"\n"+second, payloadEntries)
				landed := f.commit([]string{base}, "land payload", payloadEntries)
				return checkpoint, f.commit([]string{landed}, "advance", headEntries)
			},
		},
		{
			name: "stamp-parent-mismatch",
			build: func(t *testing.T, f *peerHistoryFixture) (string, string) {
				base := f.commit(nil, "base", baseEntries)
				other := f.commit([]string{base}, "other", baseEntries)
				checkpoint := f.commit([]string{base}, peerHistoryStamp(t, peer, other, paths), payloadEntries)
				landed := f.commit([]string{base}, "land payload", payloadEntries)
				return checkpoint, f.commit([]string{landed}, "advance", headEntries)
			},
		},
		{
			name: "multiple-parents",
			build: func(t *testing.T, f *peerHistoryFixture) (string, string) {
				base := f.commit(nil, "base", baseEntries)
				other := f.commit([]string{base}, "other", baseEntries)
				checkpoint := f.commit([]string{base, other}, peerHistoryStamp(t, peer, base, paths), payloadEntries)
				landed := f.commit([]string{base}, "land payload", payloadEntries)
				return checkpoint, f.commit([]string{landed}, "advance", headEntries)
			},
		},
		{
			name: "unsupported-symlink-entry",
			build: func(t *testing.T, f *peerHistoryFixture) (string, string) {
				base := f.commit(nil, "base", baseEntries)
				symlinkPayload := peerHistoryEntries("120000", "target.txt", "payload-b\n")
				checkpoint := f.commit([]string{base}, peerHistoryStamp(t, peer, base, paths), symlinkPayload)
				landed := f.commit([]string{base}, "land symlink payload", symlinkPayload)
				return checkpoint, f.commit([]string{landed}, "advance", headEntries)
			},
		},
		{
			name: "unsupported-deletion-entry",
			build: func(t *testing.T, f *peerHistoryFixture) (string, string) {
				base := f.commit(nil, "base", baseEntries)
				deletedPayload := map[string]peerHistoryEntry{
					"owned/a.txt": {mode: "100644", body: "payload-a\n"},
				}
				checkpoint := f.commit([]string{base}, peerHistoryStamp(t, peer, base, paths), deletedPayload)
				landed := f.commit([]string{base}, "land deletion payload", deletedPayload)
				return checkpoint, f.commit([]string{landed}, "advance", headEntries)
			},
		},
		{
			name:  "unsupported-directory-scope",
			scope: []string{"owned"},
			build: func(t *testing.T, f *peerHistoryFixture) (string, string) {
				base := f.commit(nil, "base", baseEntries)
				checkpoint := f.commit([]string{base}, peerHistoryStamp(t, peer, base, []string{"owned"}), payloadEntries)
				landed := f.commit([]string{base}, "land payload", payloadEntries)
				return checkpoint, f.commit([]string{landed}, "advance", headEntries)
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			f := newPeerHistoryFixture(t)
			checkpoint, head := tc.build(t, f)
			f.git("update-ref", "refs/fak/wip/"+peer, checkpoint)
			f.git("update-ref", "HEAD", head)
			owners, err := f.resolve(t, realRunner, paths)
			if err != nil {
				t.Fatal(err)
			}
			if got := []string{owners[paths[0]].Peer, owners[paths[1]].Peer}; !reflect.DeepEqual(got, []string{peer, peer}) {
				t.Fatalf("unsupported or incomplete proof cleared ownership: got %v owners=%+v", got, owners)
			}
		})
	}

	t.Run("history-lookup-error-retains-claim", func(t *testing.T) {
		t.Parallel()
		f := newPeerHistoryFixture(t)
		base := f.commit(nil, "base", baseEntries)
		checkpoint := f.commit([]string{base}, peerHistoryStamp(t, peer, base, paths), payloadEntries)
		f.git("update-ref", "refs/fak/wip/"+peer, checkpoint)
		landed := f.commit([]string{base}, "land payload", payloadEntries)
		f.git("update-ref", "HEAD", f.commit([]string{landed}, "advance", headEntries))
		findObjectCalls := 0
		run := func(ctx context.Context, dir string, args ...string) (string, int, error) {
			if peerHistoryHasFindObject(args) {
				findObjectCalls++
				return "fixture lookup failure", 128, nil
			}
			return realRunner(ctx, dir, args...)
		}
		owners, err := f.resolve(t, run, paths)
		if err != nil {
			t.Fatalf("history lookup error must fail closed to retained ownership, not fail attribution: %v", err)
		}
		if findObjectCalls == 0 {
			t.Fatal("fixture did not exercise historical find-object lookup")
		}
		if owners[paths[0]].Peer != peer || owners[paths[1]].Peer != peer {
			t.Fatalf("lookup error cleared ownership: %+v", owners)
		}
	})

	t.Run("candidate-ancestry-error-retains-claim", func(t *testing.T) {
		t.Parallel()
		f := newPeerHistoryFixture(t)
		base := f.commit(nil, "base", baseEntries)
		checkpoint := f.commit([]string{base}, peerHistoryStamp(t, peer, base, paths), payloadEntries)
		f.git("update-ref", "refs/fak/wip/"+peer, checkpoint)
		landed := f.commit([]string{base}, "land payload", payloadEntries)
		f.git("update-ref", "HEAD", f.commit([]string{landed}, "advance", headEntries))
		ancestryCalls := 0
		run := func(ctx context.Context, dir string, args ...string) (string, int, error) {
			if len(args) > 0 && args[0] == "merge-base" {
				ancestryCalls++
				if ancestryCalls == 1 {
					return "fixture ancestry failure", 2, nil
				}
			}
			return realRunner(ctx, dir, args...)
		}
		owners, err := f.resolve(t, run, paths)
		if err != nil {
			t.Fatalf("ancestry lookup error must fail closed to retained ownership, not fail attribution: %v", err)
		}
		if ancestryCalls == 0 {
			t.Fatal("fixture did not exercise candidate ancestry lookup")
		}
		if owners[paths[0]].Peer != peer || owners[paths[1]].Peer != peer {
			t.Fatalf("ancestry lookup error cleared ownership: %+v", owners)
		}
	})
}

// fak-test:runtime medium est=5s lane=default
func TestPeerWIPHistoricalCandidateBudgetFailsClosed(t *testing.T) {
	const (
		peer = "peer-budget"
		path = "owned/a.txt"
	)
	f := newPeerHistoryFixture(t)
	baseEntries := map[string]peerHistoryEntry{path: {mode: "100644", body: "base\n"}}
	payloadEntries := map[string]peerHistoryEntry{path: {mode: "100644", body: "payload\n"}}
	base := f.commit(nil, "base", baseEntries)
	checkpoint := f.commit([]string{base}, peerHistoryStamp(t, peer, base, []string{path}), payloadEntries)
	f.git("update-ref", "refs/fak/wip/"+peer, checkpoint)

	// Fast-import makes 129 distinct qualifying payload commits without paying
	// hundreds of fixture process launches. A bounded proof must retain the claim
	// rather than select from an incomplete history prefix.
	var stream strings.Builder
	const matches = 129
	for i := 0; i < matches*2; i++ {
		mark := i + 1
		body := "payload\n"
		if i%2 == 1 {
			body = "base\n"
		}
		fmt.Fprintf(&stream, "commit refs/heads/peer-history-budget\nmark :%d\ncommitter Fixture <fixture@example.invalid> %d +0000\ndata 1\nx\n", mark, 1700000000+i)
		if i == 0 {
			fmt.Fprintf(&stream, "from %s\n", base)
		} else {
			fmt.Fprintf(&stream, "from :%d\n", i)
		}
		fmt.Fprintf(&stream, "M 100644 inline %s\ndata %d\n%s\n", path, len(body), body)
	}
	stream.WriteString("done\n")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	cmd := newGitCmd(ctx, f.dir, "fast-import", "--quiet", "--done")
	cmd.Stdin = strings.NewReader(stream.String())
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("fixture fast-import: %v: %s", err, out)
	}
	f.git("update-ref", "HEAD", f.git("rev-parse", "refs/heads/peer-history-budget"))
	findObjectCalls := 0
	run := func(ctx context.Context, dir string, args ...string) (string, int, error) {
		if peerHistoryHasFindObject(args) {
			findObjectCalls++
		}
		return realRunner(ctx, dir, args...)
	}
	owners, err := f.resolve(t, run, []string{path})
	if err != nil {
		t.Fatal(err)
	}
	if findObjectCalls == 0 {
		t.Fatal("fixture did not exercise bounded historical candidate search")
	}
	if owners[path].Peer != peer {
		t.Fatalf("candidate budget exhaustion cleared ownership: %+v", owners)
	}
}
