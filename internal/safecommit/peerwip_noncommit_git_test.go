package safecommit

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipref"
)

// fak-test:runtime medium est=3s lane=default
func TestPeerWIPNonCommitRefsPreserveRealGitAttribution(t *testing.T) {
	// ~25 fixture git spawns must not drain the budget each attribution check gets.
	fixtureCtx, cancelFixture := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancelFixture()
	dir := t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		out, code, err := realRunner(fixtureCtx, dir, args...)
		if err != nil || code != 0 {
			t.Fatalf("fixture git %v: code=%d err=%v output=%s", args, code, err, out)
		}
		return strings.TrimSpace(out)
	}
	write := func(path, body string) {
		t.Helper()
		full := filepath.Join(dir, filepath.FromSlash(path))
		if err := os.MkdirAll(filepath.Dir(full), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	git("init")
	git("config", "user.name", "Fixture")
	git("config", "user.email", "fixture@example.invalid")
	write("corpus/owned.txt", "base\n")
	write("corpus/free.txt", "base\n")
	git("add", "--", "corpus/owned.txt", "corpus/free.txt")
	baseTree := git("write-tree")
	base := git("commit-tree", baseTree, "-m", "base")
	git("update-ref", "HEAD", base)

	// These are native Git refs to each noncommit object kind. An annotated
	// tag even carries a scope stamp, but must not acquire checkpoint ownership.
	blob := git("rev-parse", base+":corpus/free.txt")
	tagMessage, err := wipref.EncodeStamp(wipref.Stamp{SessionID: "c-tag", Scope: []string{"corpus"}})
	if err != nil {
		t.Fatal(err)
	}
	git("-c", "tag.gpgSign=false", "tag", "-a", "-m", tagMessage, "fixture-noncommit", base)
	tag := git("rev-parse", "refs/tags/fixture-noncommit")
	git("update-ref", "refs/fak/wip/a-blob", blob)
	git("update-ref", "refs/fak/wip/b-tree", baseTree)
	git("update-ref", "refs/fak/wip/c-tag", tag)
	wantMalformed := []string{
		"refs/fak/wip/a-blob (blob)",
		"refs/fak/wip/b-tree (tree)",
		"refs/fak/wip/c-tag (tag)",
	}

	// Keep the real checkpoint unreferenced until the all-noncommit check.
	// Its delta owns only owned.txt; free.txt stays outside that ownership.
	write("corpus/owned.txt", "peer checkpoint\n")
	git("add", "--", "corpus/owned.txt")
	peerTree := git("write-tree")
	peer := git("commit-tree", peerTree, "-p", base, "-m", "peer checkpoint")
	git("read-tree", base)
	write("corpus/owned.txt", "working edit\n")
	write("corpus/free.txt", "unrelated edit\n")

	check := func(stage string, paths, wantCollisions []string) {
		t.Helper()
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		res, err := ValidatePathAttribution(ctx, realRunner, dir, paths, PathAttributionOptions{SessionID: "self"})
		if err != nil {
			t.Fatalf("%s attribution: %v", stage, err)
		}
		if !reflect.DeepEqual(res.MalformedPeerRefs, wantMalformed) {
			t.Fatalf("%s malformed refs = %v, want %v", stage, res.MalformedPeerRefs, wantMalformed)
		}
		if len(res.ExpandedPaths) == 0 {
			t.Fatalf("%s fixture did not exercise a dirty path: %+v", stage, res)
		}
		if len(wantCollisions) == 0 {
			if !res.OK || res.Reason != "" || len(res.CollidingPaths) != 0 || len(res.PeerSessions) != 0 || !reflect.DeepEqual(res.EffectivePaths, paths) {
				t.Fatalf("%s unrelated work must be admitted: %+v", stage, res)
			}
			return
		}
		if res.OK || res.Reason != ReasonPeerWIPCollision || !reflect.DeepEqual(res.CollidingPaths, wantCollisions) || !reflect.DeepEqual(res.PeerSessions, []string{"d-peer"}) {
			t.Fatalf("%s valid checkpoint must still own its path: %+v", stage, res)
		}
	}
	check("all-noncommit", []string{"corpus"}, nil)
	git("update-ref", "refs/fak/wip/d-peer", peer)
	check("mixed-unrelated", []string{"corpus/free.txt"}, nil)
	check("mixed-collision", []string{"corpus"}, []string{"corpus/owned.txt"})
}
