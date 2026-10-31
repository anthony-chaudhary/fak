package safecommit

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

// fak-test:runtime medium est=5s lane=default
func TestPeerWIPNativeParkAttribution(t *testing.T) {
	f := newPeerHistoryFixture(t)
	base := f.commit(nil, "base", peerHistoryEntries("100644", "base-a\n", "base-b\n"))
	f.git("update-ref", "HEAD", base)
	owned := filepath.Join(f.dir, "owned", "a.txt")
	peer := filepath.Join(f.dir, "owned", "b.txt")
	if err := os.WriteFile(owned, []byte("owned local edit\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(peer, []byte("unrelated peer edit\n"), 0600); err != nil {
		t.Fatal(err)
	}
	index := f.git("write-tree")
	rec, err := safesync.Park(context.Background(), safesync.ParkOptions{
		Repo: f.dir, Session: "self", Paths: []string{"owned/a.txt"}, TargetRef: base, Apply: true,
	})
	if err != nil || !rec.OK {
		t.Fatalf("native Park: err=%v status=%s", err, rec.Status)
	}
	oid := f.git("rev-parse", rec.CheckpointRef)
	// Preserve the legacy format as well: the native message needs no fak-wip stamp.
	legacy := "refs/fak/wip/self/park"
	f.git("update-ref", legacy, oid)
	refs := f.refs()
	check := func(wantPeer string) {
		t.Helper()
		result, err := ValidatePathAttribution(context.Background(), realRunner, f.dir,
			[]string{"owned/a.txt"}, PathAttributionOptions{SessionID: "self"})
		if err != nil {
			t.Fatal(err)
		}
		if wantPeer == "" {
			if !result.OK || len(result.PeerSessions) != 0 {
				t.Fatalf("native self Park was attributed to a peer: reason=%s peers=%v", result.Reason, result.PeerSessions)
			}
		} else if result.OK || result.Reason != ReasonPeerWIPCollision || !reflect.DeepEqual(result.PeerSessions, []string{wantPeer}) {
			t.Fatalf("peer claim lost: ok=%v reason=%s peers=%v", result.OK, result.Reason, result.PeerSessions)
		}
	}
	check("")
	if f.refs() != refs || f.git("rev-parse", "HEAD") != base || f.git("write-tree") != index {
		t.Fatal("Park attribution changed refs, HEAD or index")
	}
	for path, want := range map[string]string{owned: "owned local edit\n", peer: "unrelated peer edit\n"} {
		got, err := os.ReadFile(path)
		if err != nil || string(got) != want {
			t.Fatalf("dirty bytes changed: path=%s err=%v", path, err)
		}
	}
	// Exclusion is per ref, never per object, including a peer's identical Park.
	f.git("update-ref", "refs/fak/wip/peer/park", oid)
	check("peer/park")
	if f.git("rev-parse", legacy) != oid || f.git("rev-parse", rec.CheckpointRef) != oid {
		t.Fatal("checkpoint identity changed")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestPeerWIPNativeParkIdentity(t *testing.T) {
	oid := strings.Repeat("a", 40)
	message := "fak: park self\n\nPaths: owned/a.txt, owned/b.txt\n"
	for _, peer := range []string{"self/park", "self/park-" + oid} {
		if got := nativeSelfParkScope(peer, oid, message, "self"); !reflect.DeepEqual(got, []string{"owned/a.txt", "owned/b.txt"}) {
			t.Fatalf("native identity rejected: %s", peer)
		}
	}
	for _, peer := range []string{"peer/park", "self-other/park", "self/parked", "self/park/extra", "self/park-" + strings.Repeat("b", 40), "self/park-" + oid[:12]} {
		if nativeSelfParkScope(peer, oid, message, "self") != nil {
			t.Errorf("unowned namespace accepted: %s", peer)
		}
	}
	for _, body := range []string{
		"fak: park peer\n\nPaths: owned/a.txt\n",
		message + "extra\n", message + "fak-wip: {}\n",
		"fak: park self\n\nPaths: \n",
		"fak: park self\n\nPaths: owned/a.txt, owned/a.txt\n",
		"fak: park self\n\nPaths: owned/b.txt, owned/a.txt\n",
		"fak: park self\n\nPaths: owned/./a.txt\n",
		"fak: park self\n\nPaths: ../a.txt\n",
		"fak: park self\n\nPaths: owned/a,b.txt\n",
		"fak: park self\n\nPaths: owned/a.txt\nPaths: owned/b.txt\n",
		"fak: park self\n\nPaths: " + strings.Repeat("x", 64*1024) + "\n",
	} {
		if nativeSelfParkScope("self/park", oid, body, "self") != nil {
			t.Errorf("ambiguous message accepted: %q", body[:min(len(body), 100)])
		}
	}
	for _, self := range []string{"", "self/other", " self", "-self"} {
		if nativeSelfParkScope(self+"/park", oid, "fak: park "+self+"\n\nPaths: owned/a.txt\n", self) != nil {
			t.Errorf("invalid session accepted: %q", self)
		}
	}
}

// fak-test:runtime medium est=8s lane=default
func TestPeerWIPNativeParkCompleteProof(t *testing.T) {
	f := newPeerHistoryFixture(t)
	baseEntries := peerHistoryEntries("100644", "base-a\n", "base-b\n")
	base := f.commit(nil, "base", baseEntries)
	other := f.commit(nil, "unrelated root", baseEntries)
	f.git("update-ref", "HEAD", base)
	payload := peerHistoryEntries("100644", "edit-a\n", "edit-b\n")
	checkpoint := f.commit([]string{base}, "fak: park self\n\nPaths: owned/a.txt, owned/b.txt", payload)
	scope := []string{"owned/a.txt", "owned/b.txt"}
	prove := func(oid, head string, paths []string, runner Runner) bool {
		return peerWIPNativeParkProven(context.Background(), runner, f.dir,
			peerWIPCheckpoint{oid: oid, parkScope: paths}, head)
	}
	if !prove(checkpoint, base, scope, realRunner) {
		t.Fatal("valid native Park proof rejected")
	}
	unchanged := f.commit([]string{base}, "one changed selected file", peerHistoryEntries("100644", "edit-a\n", "base-b\n"))
	if !prove(unchanged, base, scope, realRunner) {
		t.Fatal("unchanged selected regular file rejected")
	}
	addedEntries := peerHistoryEntries("100644", "base-a\n", "base-b\n")
	addedEntries["owned/[x].txt"] = peerHistoryEntry{mode: "100755", body: "literal added\n"}
	added := f.commit([]string{base}, "addition", addedEntries)
	if !prove(added, base, []string{"owned/[x].txt"}, realRunner) {
		t.Fatal("literal regular-file addition rejected")
	}
	symlink := f.commit([]string{base}, "symlink", peerHistoryEntries("120000", "target", "edit-b\n"))
	symlinkBase := f.commit(nil, "symlink base", peerHistoryEntries("120000", "target", "base-b\n"))
	regularAfterLink := f.commit([]string{symlinkBase}, "type change", payload)
	deleted := f.commit([]string{base}, "deleted", map[string]peerHistoryEntry{"owned/b.txt": payload["owned/b.txt"]})
	root := f.commit(nil, "root checkpoint", payload)
	merged := f.commit([]string{base, other}, "merge checkpoint", payload)
	nonancestor := f.commit([]string{other}, "unrelated checkpoint", payload)
	for _, tc := range []struct {
		name, oid, head string
		paths           []string
	}{
		{"outside-scope", checkpoint, base, scope[:1]},
		{"directory-scope", checkpoint, base, []string{"owned"}},
		{"symlink", symlink, base, scope},
		{"symlink-preimage", regularAfterLink, symlinkBase, scope},
		{"deletion", deleted, base, scope},
		{"root", root, base, scope},
		{"merge", merged, base, scope},
		{"nonancestor", nonancestor, base, scope},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if prove(tc.oid, tc.head, tc.paths, realRunner) {
				t.Fatal("unsupported checkpoint cleared ownership")
			}
		})
	}
	// Complete records can be missing despite valid NUL framing. Verify both
	// selected-tree correspondence and the independent outside-scope exit code.
	for _, fault := range []string{"parent-record", "selected-delta", "outside-delta", "trailing-byte", "lookup-error", "lookup-timeout"} {
		t.Run(fault, func(t *testing.T) {
			hit := false
			runner := func(ctx context.Context, dir string, args ...string) (string, int, error) {
				out, code, err := realRunner(ctx, dir, args...)
				if args[0] == "ls-tree" && args[3] == base && fault == "parent-record" {
					hit = true
					_, out, _ = strings.Cut(out, "\x00")
				}
				if args[0] == "diff-tree" {
					switch fault {
					case "selected-delta", "outside-delta":
						hit = true
						parts := strings.Split(out, "\x00")
						out = strings.Join(parts[:len(parts)-3], "\x00") + "\x00"
					case "trailing-byte":
						hit = true
						out = strings.TrimSuffix(out, "\x00")
					case "lookup-error":
						hit = true
						return "", 128, errors.New("fixture lookup failed")
					case "lookup-timeout":
						hit = true
						return "", -1, context.DeadlineExceeded
					}
				}
				return out, code, err
			}
			paths := scope
			if fault == "outside-delta" {
				paths = scope[:1]
			}
			if prove(checkpoint, base, paths, runner) || !hit {
				t.Fatalf("incomplete proof accepted or fault unexercised: hit=%v", hit)
			}
		})
	}
}
