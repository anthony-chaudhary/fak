package workerworktree

import (
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime medium est=40s lane=default
func TestTopologyCandidatePoolAdoptTrimsExcess(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}

	t.Run("trims_idle_excess_on_adopt", func(t *testing.T) {
		root, parent := newSiblingTopologyRepo(t)
		baseTime := time.Now().Add(-10 * time.Minute)
		var created []string
		for i := 0; i < TopologyCandidatePoolCap+2; i++ {
			dir := filepath.Join(parent, strings.TrimSuffix(topologyCandidatePoolPattern(), "*")+"adopttrim"+strconv.Itoa(i))
			mustGit(t, root, "worktree", "add", "--detach", dir, "HEAD")
			ts := baseTime.Add(time.Duration(i) * time.Second)
			if err := os.Chtimes(dir, ts, ts); err != nil {
				t.Fatal(err)
			}
			created = append(created, dir)
		}
		t.Cleanup(func() {
			for _, dir := range created {
				removeCandidateWithRegistration(root, dir, defaultGit)
			}
		})

		if got := livePoolCount(root, parent, defaultGit); got != TopologyCandidatePoolCap+2 {
			t.Fatalf("pre-adopt live pool count = %d, want %d", got, TopologyCandidatePoolCap+2)
		}

		s := newTopologyCandidateSession(root, defaultGit)
		defer s.close()
		if ok, detail := s.create("HEAD"); !ok {
			t.Fatalf("create: %s", detail)
		}
		if !s.reused || !s.pooled {
			t.Fatalf("expected warm pooled adoption, got reused=%v pooled=%v dir=%q", s.reused, s.pooled, s.dir)
		}
		if _, err := os.Stat(s.dir); err != nil {
			t.Fatalf("adopted checkout %q missing after trim: %v", s.dir, err)
		}
		// Check BEFORE s.close() so the assertion witnesses adoptPooled itself,
		// not the bounded drain in close().
		if got := livePoolCount(root, parent, defaultGit); got != TopologyCandidatePoolCap {
			t.Fatalf("live pool count immediately after warm adopt = %d, want %d", got, TopologyCandidatePoolCap)
		}
	})

	t.Run("preserves_active_peer_lock_while_trimming_idle", func(t *testing.T) {
		root, parent := newSiblingTopologyRepo(t)
		baseTime := time.Now().Add(-10 * time.Minute)
		var created []string
		for i := 0; i < TopologyCandidatePoolCap+2; i++ {
			dir := filepath.Join(parent, strings.TrimSuffix(topologyCandidatePoolPattern(), "*")+"adoptpeer"+strconv.Itoa(i))
			mustGit(t, root, "worktree", "add", "--detach", dir, "HEAD")
			ts := baseTime.Add(time.Duration(i) * time.Second)
			if err := os.Chtimes(dir, ts, ts); err != nil {
				t.Fatal(err)
			}
			created = append(created, dir)
		}
		// Hold the oldest checkout as an active peer session.
		peerDir := created[0]
		peerLock, ok := lockCandidatePoolDir(peerDir)
		if !ok {
			t.Fatalf("could not lock peer checkout %q", peerDir)
		}
		t.Cleanup(func() {
			releaseCandidatePool(peerLock)
			for _, dir := range created {
				removeCandidateWithRegistration(root, dir, defaultGit)
			}
		})

		s := newTopologyCandidateSession(root, defaultGit)
		defer s.close()
		if ok, detail := s.create("HEAD"); !ok {
			t.Fatalf("create: %s", detail)
		}
		if !s.reused || !s.pooled {
			t.Fatalf("expected warm pooled adoption, got reused=%v pooled=%v dir=%q", s.reused, s.pooled, s.dir)
		}
		if samePath(s.dir, peerDir) {
			t.Fatalf("adopted peer's actively locked checkout %q", peerDir)
		}
		if _, err := os.Stat(peerDir); err != nil {
			t.Fatalf("active peer checkout %q was removed by adopt trim: %v", peerDir, err)
		}
		if _, err := os.Stat(s.dir); err != nil {
			t.Fatalf("adopted checkout %q missing after trim: %v", s.dir, err)
		}
		if got := livePoolCount(root, parent, defaultGit); got != TopologyCandidatePoolCap {
			t.Fatalf("live pool count immediately after warm adopt with peer lock = %d, want %d", got, TopologyCandidatePoolCap)
		}
	})
}
