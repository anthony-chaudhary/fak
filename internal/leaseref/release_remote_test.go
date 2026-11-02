package leaseref

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime integration est=3s lane=default
func TestReleaseFencedRemoteContract(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	for _, mode := range []string{"success", "remote-replacement", "local-replacement", "unavailable", "uncertain-push", "both-absent", "local-absent", "history-replacement", "local-alias", "history-alias", "different-pushurl", "multiple-pushurl", "advertised-alias"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			dir, remote := t.TempDir(), filepath.Join(t.TempDir(), "remote.git")
			run := func(cwd string, args ...string) string {
				t.Helper()
				cmd := exec.CommandContext(ctx, "git", args...)
				cmd.Dir = cwd
				out, err := cmd.CombinedOutput()
				if err != nil {
					t.Fatalf("fixture git %v: %v: %s", args, err, out)
				}
				return strings.TrimSpace(string(out))
			}
			run(dir, "init", "-q")
			run(dir, "init", "-q", "--bare", remote)
			run(dir, "remote", "add", "origin", remote)
			s := NewInDir(dir)
			now := time.Unix(1800000000, 0)
			rec, admission, err := s.AcquireFenced(ctx, Record{ID: "release-contract", Holder: "owner", TreeGlobs: []string{"owned/**"}, TTLSeconds: 3600}, now)
			if err != nil || !admission.OK {
				t.Fatalf("seed admission: %+v %v", admission, err)
			}
			ref := rec.Ref()
			original := run(dir, "rev-parse", ref)
			peer := Record{ID: "unrelated-peer", Holder: "peer", AcquiredAt: now.Unix(), TTLSeconds: 3600}
			if _, err := s.Acquire(ctx, peer); err != nil {
				t.Fatal(err)
			}
			peerOID := run(dir, "rev-parse", peer.Ref())
			run(dir, "push", "-q", "origin", ref, peer.Ref())
			historyFloor := int64(9)
			if mode == "history-replacement" || mode == "history-alias" {
				historyFloor = 0
			}
			if err := s.writeHistoryRecord(ctx, rec.ID, historyFloor, now); err != nil {
				t.Fatal(err)
			}
			aliasTarget := "refs/fak/locks/alias-peer"
			run(dir, "update-ref", aliasTarget, original)
			historyTarget := historyRefPrefix + "alias-peer"
			historyOID := run(dir, "rev-parse", historyRefPrefix+rec.ID)
			run(dir, "update-ref", historyTarget, historyOID)
			if mode == "different-pushurl" {
				other := filepath.Join(t.TempDir(), "other.git")
				run(dir, "init", "-q", "--bare", other)
				run(dir, "remote", "set-url", "--push", "origin", other)
			}
			if mode == "multiple-pushurl" {
				run(dir, "remote", "set-url", "--add", "--push", "origin", remote)
				run(dir, "remote", "set-url", "--add", "--push", "origin", remote)
			}
			if mode == "both-absent" || mode == "local-absent" {
				run(dir, "update-ref", "-d", ref, original)
			}
			if mode == "both-absent" {
				run(remote, "update-ref", "-d", ref, original)
			}
			newer := rec
			newer.Holder, newer.Generation = "replacement", rec.Generation+1
			replacement, err := s.writeBlob(ctx, []byte(fmt.Sprintf(
				"{\"id\":%q,\"holder\":%q,\"generation\":%d,\"acquired_unix\":%d,\"ttl_seconds\":3600}",
				newer.ID, newer.Holder, newer.Generation, now.Unix())))
			if err != nil {
				t.Fatal(err)
			}
			// Make the replacement object available remotely without changing a ref.
			run(dir, "push", "-q", remote, replacement+":refs/fak/locks/fixture-object")
			run(remote, "update-ref", "-d", "refs/fak/locks/fixture-object", replacement)
			var calls [][]string
			fired := false
			wrapped := NewWithRunner(func(c context.Context, cwd string, args ...string) (string, int, error) {
				calls = append(calls, append([]string(nil), args...))
				if args[0] == "fetch" {
					t.Fatal("release performed fetch")
				}
				if mode == "unavailable" && args[0] == "ls-remote" {
					return "private transport payload must not escape", 128, nil
				}
				if (mode == "history-replacement" || mode == "history-alias") && !fired && len(args) >= 4 && args[0] == "update-ref" && args[1] == "--no-deref" && args[2] == historyRefPrefix+rec.ID {
					fired = true
					if mode == "history-alias" {
						run(dir, "symbolic-ref", historyRefPrefix+rec.ID, historyTarget)
					} else if err := s.writeHistoryRecord(ctx, rec.ID, 77, now); err != nil {
						t.Fatal(err)
					}
				}
				if mode == "advertised-alias" && args[0] == "ls-remote" {
					return "ref: " + peer.Ref() + "\t" + ref + "\n" + original + "\t" + ref + "\n", 0, nil
				}
				if args[0] == "push" {
					wantFence := "--force-with-lease=" + ref + ":" + original
					if len(args) != 5 || args[1] != "--porcelain" || args[2] != wantFence || args[3] != remote || args[4] != ":"+ref {
						t.Fatalf("unfenced/broad remote write: %v", args)
					}
					if mode == "remote-replacement" && !fired {
						fired = true
						run(remote, "update-ref", ref, replacement, original)
					}
					out, code, err := gitRunner(c, cwd, args...)
					if mode == "local-replacement" && !fired && code == 0 && err == nil {
						fired = true
						run(dir, "update-ref", ref, replacement, original)
					}
					if mode == "local-alias" && !fired && code == 0 && err == nil {
						fired = true
						run(dir, "symbolic-ref", ref, aliasTarget)
					}
					if mode == "uncertain-push" && code == 0 && err == nil {
						return "untrusted credential-bearing transport output", 128, nil
					}
					return out, code, err
				}
				return gitRunner(c, cwd, args...)
			}, dir)
			v, err := wrapped.ReleaseFencedRemote(ctx, rec.ID, rec.Holder, rec.Generation, now, "origin")
			oid := func(cwd string) string {
				o, code, e := gitRunner(ctx, cwd, "rev-parse", "--verify", "--quiet", ref)
				if e != nil || (code != 0 && code != 1) {
					t.Fatalf("ref observation: %d %v", code, e)
				}
				return strings.TrimSpace(o)
			}
			localOID, remoteOID := oid(dir), oid(remote)
			wantLocal, wantRemote := original, original
			switch mode {
			case "success", "both-absent":
				wantLocal, wantRemote = "", ""
				if err != nil || !v.OK {
					t.Fatalf("release = %+v, %v", v, err)
				}
				if _, err := s.Sync(ctx, "origin", false, true); err != nil {
					t.Fatal(err)
				}
				if got := oid(dir); got != "" {
					t.Fatalf("later fetch resurrected %s", got)
				}
			case "remote-replacement":
				wantRemote = replacement
				if err == nil || v.OK || !fired {
					t.Fatalf("remote race = %+v, %v; fired=%v", v, err, fired)
				}
			case "local-replacement":
				wantLocal, wantRemote = replacement, ""
				if err != nil || v.OK || v.Reason != ReasonLeaseContended || !fired || !strings.Contains(v.Detail, "partial release") {
					t.Fatalf("local race = %+v, %v", v, err)
				}
			case "uncertain-push":
				wantRemote = ""
				if err == nil || v.OK || !strings.Contains(err.Error(), "uncertain") {
					t.Fatalf("uncertain transport = %+v, %v", v, err)
				}
			case "unavailable", "history-replacement", "different-pushurl", "multiple-pushurl", "advertised-alias":
				if err == nil || v.OK {
					t.Fatalf("refusal = %+v, %v", v, err)
				}
			case "local-alias", "history-alias":
				if !fired {
					t.Fatal("same-OID symbolic race was not injected")
				}
				// --no-deref protects peer targets. Some supported Git versions
				// can replace the alias itself when its referent matches oldOID.
				// Require an honest outcome whichever CAS result Git supplies.
				if v.OK {
					if err != nil {
						t.Fatal(err)
					}
					wantLocal, wantRemote = "", ""
				} else if mode == "local-alias" {
					wantRemote = ""
				}
			case "local-absent":
				wantLocal = ""
				if err != nil || v.OK || v.Reason != ReasonStaleLease {
					t.Fatalf("missing local evidence = %+v, %v", v, err)
				}
			}
			if localOID != wantLocal || remoteOID != wantRemote {
				t.Fatalf("local/remote=%s/%s, want %s/%s", localOID, remoteOID, wantLocal, wantRemote)
			}
			if err != nil && (strings.Contains(err.Error(), "credential-bearing") || strings.Contains(err.Error(), "private transport")) {
				t.Fatalf("transport payload leaked: %v", err)
			}
			if run(dir, "rev-parse", peer.Ref()) != peerOID || run(remote, "rev-parse", peer.Ref()) != peerOID {
				t.Fatal("unrelated peer changed")
			}
			if run(dir, "rev-parse", aliasTarget) != original || run(dir, "rev-parse", historyTarget) != historyOID {
				t.Fatal("same-OID symbolic replacement changed a peer target")
			}
			history, present, err := s.ReadHistory(ctx, rec.ID)
			if err != nil || !present || history.Generation < historyFloor {
				t.Fatalf("history regressed: %+v %v %v", history, present, err)
			}
			if mode == "history-replacement" && (!fired || history.Generation != 77) {
				t.Fatalf("history replacement lost: %+v; fired=%v", history, fired)
			}
			pushes := 0
			for _, args := range calls {
				if args[0] == "push" {
					pushes++
				}
			}
			if pushes > 1 {
				t.Fatalf("automatic retry: %d pushes", pushes)
			}
		})
	}
}

// fak-test:runtime integration est=1s lane=default
func TestReleaseFencedRemoteAdmissionBeforeTransport(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	dir := t.TempDir()
	if out, err := exec.Command("git", "-C", dir, "init", "-q").CombinedOutput(); err != nil {
		t.Fatalf("init: %v %s", err, out)
	}
	ctx := context.Background()
	now := time.Unix(1800000000, 0)
	s := NewInDir(dir)
	rec, v, err := s.AcquireFenced(ctx, Record{ID: "fenced", Holder: "owner", TTLSeconds: 10}, now)
	if err != nil || !v.OK {
		t.Fatalf("seed: %+v %v", v, err)
	}
	if out, err := exec.Command("git", "-C", dir, "symbolic-ref", refPrefix+"symbolic", rec.Ref()).CombinedOutput(); err != nil {
		t.Fatalf("symbolic fixture: %v %s", err, out)
	}
	for _, tc := range []struct {
		name, id, holder string
		generation       int64
		at               time.Time
	}{
		{"wrong-holder", rec.ID, "other", rec.Generation, now},
		{"expired-stale-generation", rec.ID, rec.Holder, rec.Generation + 1, now.Add(time.Hour)},
		{"reserved", "session-guard", rec.Holder, rec.Generation, now},
		{"symbolic", "symbolic", rec.Holder, rec.Generation, now},
		{"invalid", "../main", rec.Holder, rec.Generation, now},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wrapped := NewWithRunner(func(c context.Context, cwd string, args ...string) (string, int, error) {
				if args[0] == "ls-remote" || args[0] == "push" || args[0] == "update-ref" {
					t.Fatalf("refused admission reached side effect/transport: %v", args)
				}
				return gitRunner(c, cwd, args...)
			}, dir)
			got, err := wrapped.ReleaseFencedRemote(ctx, tc.id, tc.holder, tc.generation, tc.at, "origin")
			if got.OK || (err == nil && got.Reason != ReasonStaleLease) {
				t.Fatalf("admission = %+v, %v", got, err)
			}
		})
	}
}
