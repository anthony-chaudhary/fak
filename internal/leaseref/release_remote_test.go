package leaseref

import (
	"context"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Real Git, including an injected remote replacement at the actual push boundary.
func TestReleaseFencedRemote(t *testing.T) {
	for _, name := range []string{"converges", "remote_replacement", "local_replacement", "unavailable", "already_absent"} {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			root := t.TempDir()
			remote, local := filepath.Join(root, "remote.git"), filepath.Join(root, "local")
			git := func(dir string, args ...string) string {
				t.Helper()
				cmd := exec.Command("git", args...)
				cmd.Dir = dir
				out, err := cmd.CombinedOutput()
				if err != nil {
					t.Fatalf("git %v: %v %s", args, err, out)
				}
				return strings.TrimSpace(string(out))
			}
			git(root, "init", "--bare", remote)
			git(root, "clone", remote, local)
			store := NewInDir(local)
			now := time.Now()
			rec, av, err := store.AcquireFenced(ctx, Record{ID: "lane", Holder: "owner", TTLSeconds: 3600, TreeGlobs: []string{"a/**"}}, now)
			if err != nil || !av.OK {
				t.Fatalf("acquire: %+v %v", av, err)
			}
			if _, err := store.Sync(ctx, "origin", true, false); err != nil {
				t.Fatal(err)
			}
			old := git(local, "rev-parse", refPrefix+"lane")
			_, av, err = store.AcquireFenced(ctx, Record{ID: "peer", Holder: "other", TTLSeconds: 3600, TreeGlobs: []string{"b/**"}}, now)
			if err != nil || !av.OK {
				t.Fatalf("peer acquire: %+v %v", av, err)
			}
			if _, err := store.Sync(ctx, "origin", true, false); err != nil {
				t.Fatal(err)
			}
			peerOID := git(remote, "rev-parse", refPrefix+"peer")
			if name == "unavailable" {
				git(local, "remote", "set-url", "origin", filepath.Join(root, "missing.git"))
			}
			if name == "already_absent" {
				git(remote, "update-ref", "-d", refPrefix+"lane", old)
				git(local, "update-ref", "-d", refPrefix+"lane", old)
			}
			raced := false
			runner := store.run
			store.run = func(c context.Context, dir string, args ...string) (string, int, error) {
				if name == "remote_replacement" && !raced && args[0] == "push" {
					raced = true
					git(remote, "update-ref", refPrefix+"lane", peerOID, old)
				}
				if name == "local_replacement" && !raced && args[0] == "update-ref" && len(args) > 1 && args[1] == "-d" {
					raced = true
					git(local, "update-ref", refPrefix+"lane", peerOID, old)
				}
				return runner(c, dir, args...)
			}
			verdict, err := store.ReleaseFencedRemote(ctx, "origin", "lane", "owner", rec.Generation, now)
			if name == "remote_replacement" {
				if !raced || err != nil || verdict.OK || verdict.Reason != ReasonLeaseContended {
					t.Fatalf("race not refused: raced=%v verdict=%+v err=%v", raced, verdict, err)
				}
				if got := git(remote, "rev-parse", refPrefix+"lane"); got != peerOID {
					t.Fatal("remote replacement lost")
				}
				if got := git(local, "rev-parse", refPrefix+"lane"); got != old {
					t.Fatal("failed remote release mutated local lease")
				}
			} else if name == "local_replacement" {
				if !raced || err != nil || verdict.OK || verdict.Reason != ReasonLeaseContended {
					t.Fatalf("local race: %+v %v", verdict, err)
				}
				if got := git(local, "rev-parse", refPrefix+"lane"); got != peerOID {
					t.Fatal("local replacement lost")
				}
				if got := git(remote, "for-each-ref", "--format=%(refname)", refPrefix+"lane"); got != "" {
					t.Fatal("remote release did not precede local CAS")
				}
			} else if name == "unavailable" {
				if err == nil || verdict.OK {
					t.Fatalf("unavailable remote: %+v %v", verdict, err)
				}
				if got := git(local, "rev-parse", refPrefix+"lane"); got != old {
					t.Fatal("unavailable remote lost local lease")
				}
			} else {
				if err != nil || !verdict.OK {
					t.Fatalf("release: %+v %v", verdict, err)
				}
				if _, err := store.Sync(ctx, "origin", false, true); err != nil {
					t.Fatal(err)
				}
				if _, ok, err := store.Get(ctx, "lane"); err != nil || ok {
					t.Fatalf("released lease resurrected: exists=%v err=%v", ok, err)
				}
				if got := git(remote, "for-each-ref", "--format=%(refname)", refPrefix+"lane"); got != "" {
					t.Fatalf("remote lease remains: %s", got)
				}
			}
			if got := git(remote, "rev-parse", refPrefix+"peer"); got != peerOID {
				t.Fatal("unrelated peer changed")
			}
		})
	}
}
