package main

// gitd_test.go is the #5622 witness at the COMMAND seam. The package tests in
// internal/gitbroker already prove the cache, the fail-open fallback, and the
// deadline with two Clients inside ONE process. The claim that can only be made
// here is the one the issue actually rests on: SEPARATE OS PROCESSES sharing a
// warm broker they discovered with no env hand-wiring.

import (
	"context"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gitbroker"
)

// gitdTestRepo builds a throwaway git repo holding one blob and returns its root
// and the blob's OID. The repo lives under a SHORT temp path: the rendezvous
// socket is derived from this root, and an AF_UNIX path is length-bounded.
func gitdTestRepo(t *testing.T) (root, oid, body string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir, err := os.MkdirTemp("", "gd")
	if err != nil {
		t.Fatalf("temp repo: %v", err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })
	run := func(args ...string) string {
		cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	run("init", "-q")
	body = "brokered across processes\n"
	if err := os.WriteFile(filepath.Join(dir, "f.txt"), []byte(body), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	return dir, run("hash-object", "-w", "f.txt"), body
}

// gitdRunClientProcess re-execs THIS test binary as a genuinely separate OS
// process that performs one brokered read and reports its provenance. Nothing but
// the repo path crosses the boundary -- the child discovers the socket itself,
// which is the "no env hand-wiring" half of the acceptance gate.
func gitdRunClientProcess(t *testing.T, repo, oid string) (provenance, payload string) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=TestGitdHelperClientProcess", "-test.v")
	cmd.Env = append(os.Environ(),
		"FAK_GITD_HELPER=1",
		"FAK_GITD_HELPER_REPO="+repo,
		"FAK_GITD_HELPER_OID="+oid,
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("client process failed: %v\n%s", err, out)
	}
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if rest, ok := strings.CutPrefix(line, "HELPER_OK "); ok {
			f := strings.Fields(rest)
			if len(f) != 2 {
				t.Fatalf("malformed helper line %q", line)
			}
			raw, decErr := hex.DecodeString(f[1])
			if decErr != nil {
				t.Fatalf("helper payload not hex: %v", decErr)
			}
			return f[0], string(raw)
		}
		if strings.HasPrefix(line, "HELPER_ERR ") {
			t.Fatalf("client process errored: %s", line)
		}
	}
	t.Fatalf("client process produced no result line:\n%s", out)
	return "", ""
}

// TestGitdHelperClientProcess is the child half of the cross-process tests. It is
// inert unless the parent sets FAK_GITD_HELPER, so a normal `go test` run just
// skips it.
func TestGitdHelperClientProcess(t *testing.T) {
	if os.Getenv("FAK_GITD_HELPER") != "1" {
		t.Skip("child of the cross-process gitd tests; not run directly")
	}
	c := &gitbroker.Client{RepoRoot: os.Getenv("FAK_GITD_HELPER_REPO")}
	res, err := c.Object(context.Background(), os.Getenv("FAK_GITD_HELPER_OID"))
	if err != nil {
		fmt.Printf("HELPER_ERR %v\n", err)
		return
	}
	fmt.Printf("HELPER_OK %s %s\n", res.Provenance, hex.EncodeToString(res.Data))
}

// gitdWaitReady blocks until a broker for repo answers, so a test never races the
// listener.
func gitdWaitReady(t *testing.T, repo string) {
	t.Helper()
	c := &gitbroker.Client{RepoRoot: repo}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if _, ok := c.Stats(context.Background()); ok {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("broker never became reachable")
}

// TestGitdCachesAcrossSeparateClientProcesses is THE acceptance witness: two
// distinct OS processes, and the second one is a cache hit. That is the claim an
// in-process pool (#5621) cannot make and the entire reason this rung exists.
func TestGitdCachesAcrossSeparateClientProcesses(t *testing.T) {
	repo, oid, body := gitdTestRepo(t)
	srv, err := gitbroker.Serve(gitbroker.Config{RepoRoot: repo})
	if err != nil {
		t.Fatalf("Serve: %v", err)
	}
	defer srv.Close()
	gitdWaitReady(t, repo)

	prov1, data1 := gitdRunClientProcess(t, repo, oid)
	if prov1 != string(gitbroker.Broker) {
		t.Fatalf("first process provenance = %q, want %q (a cold cache must reach git)", prov1, gitbroker.Broker)
	}
	prov2, data2 := gitdRunClientProcess(t, repo, oid)
	if prov2 != string(gitbroker.Cache) {
		t.Fatalf("second process provenance = %q, want %q -- a SEPARATE process must reuse the warm entry", prov2, gitbroker.Cache)
	}
	if data1 != body || data2 != body {
		t.Fatalf("payloads differ from the file:\n first  %q\n second %q\n want   %q", data1, data2, body)
	}
	if st := srv.Stats(); st.Hits != 1 || st.Misses != 1 {
		t.Fatalf("stats = %+v, want exactly 1 miss then 1 hit across the two processes", st)
	}
}

// TestGitdDeadBrokerProducesByteIdenticalOutput is the fail-open witness across
// the process boundary: with the broker gone, a separate client process still
// produces the SAME bytes, tagged fallback-spawn. Only latency and provenance may
// differ -- never the answer.
func TestGitdDeadBrokerProducesByteIdenticalOutput(t *testing.T) {
	repo, oid, body := gitdTestRepo(t)
	srv, err := gitbroker.Serve(gitbroker.Config{RepoRoot: repo})
	if err != nil {
		t.Fatalf("Serve: %v", err)
	}
	gitdWaitReady(t, repo)
	provLive, dataLive := gitdRunClientProcess(t, repo, oid)
	if provLive != string(gitbroker.Broker) {
		t.Fatalf("live provenance = %q, want %q", provLive, gitbroker.Broker)
	}

	srv.Close() // the broker dies mid-run

	provDead, dataDead := gitdRunClientProcess(t, repo, oid)
	if provDead != string(gitbroker.FallbackSpawn) {
		t.Fatalf("post-mortem provenance = %q, want %q -- 'the broker is down' must not be spelled like a served answer",
			provDead, gitbroker.FallbackSpawn)
	}
	if dataDead != dataLive || dataDead != body {
		t.Fatalf("a dead broker changed the ANSWER:\n live %q\n dead %q\n want %q", dataLive, dataDead, body)
	}
}

// TestGitdStatusSpellsDownDifferentlyFromIdle is the stallscan rule on this
// command's own surface. A broker that is DOWN and a broker that is UP but has
// served nothing are both "zero activity"; reporting them the same way is exactly
// the confusion internal/stallscan exists to prevent.
func TestGitdStatusSpellsDownDifferentlyFromIdle(t *testing.T) {
	repo, _, _ := gitdTestRepo(t)

	var down, up strings.Builder
	if code := runGitd(&down, &down, []string{"--repo", repo, "--status"}); code == 0 {
		t.Fatalf("--status exited 0 with no broker running; output:\n%s", down.String())
	}
	if !strings.Contains(down.String(), "status=down") {
		t.Fatalf("a dead broker did not say so:\n%s", down.String())
	}

	srv, err := gitbroker.Serve(gitbroker.Config{RepoRoot: repo})
	if err != nil {
		t.Fatalf("Serve: %v", err)
	}
	defer srv.Close()
	gitdWaitReady(t, repo)

	if code := runGitd(&up, &up, []string{"--repo", repo, "--status"}); code != 0 {
		t.Fatalf("--status exited %d against a live broker; output:\n%s", code, up.String())
	}
	if !strings.Contains(up.String(), "status=up") || !strings.Contains(up.String(), "served=0") {
		t.Fatalf("an idle-but-live broker did not report up/served=0:\n%s", up.String())
	}
	if strings.Contains(up.String(), "status=down") {
		t.Fatalf("a live broker reported down:\n%s", up.String())
	}
}
