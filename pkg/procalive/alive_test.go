package procalive

import (
	"errors"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

const helperEnv = "PROCALIVE_HELPER_BLOCK"

// fak-test:runtime fast est=10ms
func TestProcAliveHelper(t *testing.T) {
	if os.Getenv(helperEnv) == "" {
		t.Skip("helper process only")
	}
	_, _ = io.Copy(io.Discard, os.Stdin)
}

func startHelper(t *testing.T) *exec.Cmd {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestProcAliveHelper$")
	cmd.Env = append(os.Environ(), helperEnv+"=1")
	// The helper blocks on this pipe until it is killed.
	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatalf("helper stdin: %v", err)
	}
	t.Cleanup(func() { _ = stdin.Close() })
	if err := cmd.Start(); err != nil {
		t.Fatalf("start helper: %v", err)
	}
	return cmd
}

// fak-test:runtime fast est=1s
func TestAliveTracksAChildFromRunningToReaped(t *testing.T) {
	cmd := startHelper(t)
	pid := cmd.Process.Pid
	if !Alive(pid) {
		_ = cmd.Process.Kill()
		t.Fatalf("Alive(%d) = false for a running child", pid)
	}
	_ = cmd.Process.Kill()
	_ = cmd.Wait()
	if Alive(pid) {
		t.Fatalf("Alive(%d) = true for a killed and reaped child", pid)
	}
}

// fak-test:runtime fast est=10ms
func TestAliveSelfAndInvalidPids(t *testing.T) {
	if !Alive(os.Getpid()) {
		t.Fatal("Alive(self) = false")
	}
	for _, pid := range []int{0, -1} {
		if Alive(pid) {
			t.Fatalf("Alive(%d) = true", pid)
		}
	}
}

var tasklistPidProbe = regexp.MustCompile(`"tasklist".*PID eq`)

// tasklistProbeExempt names probes outside every lock-reclaim path. Empty in
// the public tree; private-only probe files are exempted on the private side.
var tasklistProbeExempt = map[string]bool{}

// A lock reclaimer that shells out to tasklist reads a live holder as dead once
// host load pushes the probe past its timeout, and the lock stops excluding.
//
// fak-test:runtime medium est=15s
func TestNoTasklistPidProbes(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	var hits []string
	for _, top := range []string{"platform", "cmd"} {
		if _, statErr := os.Stat(filepath.Join(root, top)); errors.Is(statErr, fs.ErrNotExist) {
			continue // tree absent in this repo layout; nothing to scan
		}
		found, scanErr := scanTasklistProbes(root, top)
		if scanErr != nil {
			t.Fatal(scanErr)
		}
		hits = append(hits, found...)
	}
	if len(hits) > 0 {
		t.Fatalf("tasklist PID probes found; use procalive.Alive: %v", hits)
	}
}

func scanTasklistProbes(root, top string) ([]string, error) {
	var hits []string
	walkErr := filepath.WalkDir(filepath.Join(root, top), func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if name := d.Name(); name == "testdata" || strings.HasPrefix(name, ".") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		rel, relErr := filepath.Rel(root, path)
		if relErr != nil {
			return relErr
		}
		rel = filepath.ToSlash(rel)
		if tasklistProbeExempt[rel] {
			return nil
		}
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		for i, line := range strings.Split(string(data), "\n") {
			if tasklistPidProbe.MatchString(line) {
				hits = append(hits, rel+":"+strconv.Itoa(i+1))
			}
		}
		return nil
	})
	return hits, walkErr
}
