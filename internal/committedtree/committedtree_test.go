package committedtree

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
)

const (
	representativeFileCount = 17_742
	testGitBatchWindow      = 256
)

func TestExtractPreservesRawBlobsModesAndIgnoresLinks(t *testing.T) {
	repo := newRepository(t)
	raw := []byte{'r', 'a', 'w', '\r', '\n', 0, 0xff, '\n'}
	tree := importFocusedTree(t, repo, raw)

	dir, err := Extract(repo, tree)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.RemoveAll(dir) })

	for _, path := range []string{"regular.bin", filepath.Join("nested", "executable.bin")} {
		got, err := os.ReadFile(filepath.Join(dir, path))
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		if !bytes.Equal(got, raw) {
			t.Fatalf("%s = %v, want raw object bytes %v", path, got, raw)
		}
	}
	for _, path := range []string{"link", "module"} {
		if _, err := os.Lstat(filepath.Join(dir, path)); !errors.Is(err, fs.ErrNotExist) {
			t.Fatalf("ignored %s: Lstat error = %v, want not-exist", path, err)
		}
	}

	if runtime.GOOS != "windows" {
		assertPerm(t, filepath.Join(dir, "regular.bin"), 0o644)
		assertPerm(t, filepath.Join(dir, "nested", "executable.bin"), 0o755)
	}
}

func TestExtractUsesOneTreeListingAndOneCatFileBatch(t *testing.T) {
	repo := newRepository(t)
	tree := importFocusedTree(t, repo, []byte("blob\n"))
	var calls [][]string
	command := func(ctx context.Context, repo string, args ...string) *exec.Cmd {
		calls = append(calls, append([]string(nil), args...))
		return gitCommand(ctx, repo, args...)
	}
	if err := extractWithGit(context.Background(), repo, tree, t.TempDir(), command); err != nil {
		t.Fatal(err)
	}
	if len(calls) != 2 {
		t.Fatalf("git calls = %v, want exactly one ls-tree and one cat-file", calls)
	}
	if got := strings.Join(calls[0], " "); !strings.HasPrefix(got, "ls-tree ") {
		t.Fatalf("first git call = %q, want ls-tree", got)
	}
	if got := strings.Join(calls[1], " "); got != "cat-file --batch" {
		t.Fatalf("second git call = %q, want cat-file --batch", got)
	}
}

func TestExtractPipelinesBoundedBlobRequests(t *testing.T) {
	command := func(ctx context.Context, _ string, args ...string) *exec.Cmd {
		if len(args) > 0 && args[0] == "ls-tree" {
			return processTreeHelperCommandWithCount("cat-block", testGitBatchWindow)(ctx, "", "ls-tree")
		}
		return processTreeHelperCommand("require-pipeline", "")(ctx, "", "cat-file")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	dir := t.TempDir()
	if err := extractWithGit(ctx, "unused", "unused", dir, command); err != nil {
		t.Fatal(err)
	}
	if got, err := os.ReadFile(filepath.Join(dir, "file-255")); err != nil || string(got) != "x" {
		t.Fatalf("last pipelined blob = %q, %v; want x", got, err)
	}
}

func TestMaterializeBlobRejectsTraversal(t *testing.T) {
	root := t.TempDir()
	entry := treeEntry{mode: 0o644, oid: strings.Repeat("a", 40), path: "../escape"}
	err := writeBlobFile(bufio.NewReader(strings.NewReader("")), root, entry)
	if err == nil || !strings.Contains(err.Error(), "escapes extraction root") {
		t.Fatalf("materialize traversal error = %v", err)
	}
	if _, err := os.Lstat(filepath.Join(filepath.Dir(root), "escape")); !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("escape path exists or could not be checked: %v", err)
	}
}

func TestExtractHonorsContextDeadline(t *testing.T) {
	command := func(ctx context.Context, _ string, _ ...string) *exec.Cmd {
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=TestCommittedTreeSlowHelperProcess")
		cmd.Env = append(os.Environ(), "GO_WANT_COMMITTEDTREE_SLOW_HELPER=1")
		return cmd
	}
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	err := extractWithGit(ctx, t.TempDir(), "HEAD", t.TempDir(), command)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("deadline error = %v, want context deadline exceeded", err)
	}
	if elapsed := time.Since(start); elapsed >= 5*time.Second {
		t.Fatalf("deadline cleanup took %s, want <5s", elapsed)
	}
}

func TestExtractDeadlineKillsWindowsDescendantHoldingStdout(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows Job Object descendant witness")
	}
	pidFile := filepath.Join(t.TempDir(), "descendant.pid")
	ctx := &triggeredDeadline{done: make(chan struct{})}
	result := make(chan error, 1)
	repo := t.TempDir()
	dir := t.TempDir()
	go func() {
		result <- extractWithGit(ctx, repo, "HEAD", dir, processTreeHelperCommand("cat-block", pidFile))
	}()

	pid, err := waitForHelperPID(pidFile, result, 10*time.Second)
	if err != nil {
		close(ctx.done)
		select {
		case <-result:
		case <-time.After(5 * time.Second):
		}
		t.Fatal(err)
	}
	defer killWindowsProcessTree(pid)
	if !windowsProcessAlive(pid) {
		close(ctx.done)
		t.Fatalf("descendant %d was not alive before the deadline", pid)
	}

	expiredAt := time.Now()
	close(ctx.done)
	select {
	case err := <-result:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("deadline error = %v, want context deadline exceeded", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("extraction remained blocked after its deadline")
	}
	if elapsed := time.Since(expiredAt); elapsed >= 5*time.Second {
		t.Fatalf("deadline teardown took %s, want <5s", elapsed)
	}
	if !waitForWindowsProcessGone(pid, 10*time.Second) {
		t.Fatalf("stdout-holding descendant %d survived extraction deadline", pid)
	}
}

func TestExtractEarlyErrorKillsWindowsLSTreeDescendant(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows Job Object descendant witness")
	}
	pidFile := filepath.Join(t.TempDir(), "descendant.pid")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	start := time.Now()
	err := extractWithGit(ctx, t.TempDir(), "HEAD", t.TempDir(), processTreeHelperCommand("ls-traversal", pidFile))
	if err == nil || !strings.Contains(err.Error(), "escapes extraction root") {
		t.Fatalf("early traversal error = %v", err)
	}
	if elapsed := time.Since(start); elapsed >= 5*time.Second {
		t.Fatalf("early-error cleanup took %s, want <5s", elapsed)
	}
	pid, readErr := readHelperPID(pidFile)
	if readErr != nil {
		t.Fatalf("read ls-tree descendant PID: %v", readErr)
	}
	defer killWindowsProcessTree(pid)
	if !waitForWindowsProcessGone(pid, 10*time.Second) {
		t.Fatalf("ls-tree descendant %d survived early extraction error", pid)
	}
}

func TestCommittedTreeSlowHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_COMMITTEDTREE_SLOW_HELPER") != "1" {
		return
	}
	time.Sleep(30 * time.Second)
	os.Exit(0)
}

func TestCommittedTreeProcessTreeHelper(t *testing.T) {
	mode := os.Getenv("GO_WANT_COMMITTEDTREE_PROCESS_HELPER")
	if mode == "" {
		cmd := exec.Command(os.Args[0], "-test.run=^TestCommittedTreeProcessTreeHelper$")
		cmd.Env = append(os.Environ(), "GO_WANT_COMMITTEDTREE_PROCESS_HELPER=ls-cat-block")
		output, err := cmd.Output()
		if err != nil {
			t.Fatalf("ls-tree helper failed: %v", err)
		}
		want := []byte("100644 blob " + strings.Repeat("a", 40) + "\tfile.bin\x00")
		if !bytes.Equal(output, want) {
			t.Fatalf("ls-tree helper output = %q, want %q", output, want)
		}
		return
	}
	switch mode {
	case "ls-cat-block":
		if count, _ := strconv.Atoi(os.Getenv("GO_WANT_COMMITTEDTREE_LS_COUNT")); count > 0 {
			for i := 0; i < count; i++ {
				_, _ = fmt.Fprintf(os.Stdout, "100644 blob %040x\tfile-%03d%c", i+1, i, byte(0))
			}
		} else {
			_, _ = fmt.Fprintf(os.Stdout, "100644 blob %s\tfile.bin%c", strings.Repeat("a", 40), byte(0))
		}
	case "cat-cat-block":
		_, _ = bufio.NewReader(os.Stdin).ReadString('\n')
		spawnStdoutHoldingDescendant()
	case "cat-require-pipeline":
		scanner := bufio.NewScanner(os.Stdin)
		oids := make([]string, 0, testGitBatchWindow)
		for scanner.Scan() {
			oids = append(oids, scanner.Text())
			if len(oids) == testGitBatchWindow {
				break
			}
		}
		if len(oids) != testGitBatchWindow {
			os.Exit(3)
		}
		for _, oid := range oids {
			_, _ = fmt.Fprintf(os.Stdout, "%s blob 1\nx\n", oid)
		}
	case "ls-ls-traversal":
		spawnStdoutHoldingDescendant()
		_, _ = fmt.Fprintf(os.Stdout, "100644 blob %s\t../escape%c", strings.Repeat("a", 40), byte(0))
		time.Sleep(30 * time.Second)
	case "cat-ls-traversal":
		_, _ = bufio.NewReader(os.Stdin).ReadString('\n')
	case "descendant":
		time.Sleep(30 * time.Second)
	default:
		os.Exit(2)
	}
	os.Exit(0)
}

func TestExtractCleansTemporaryDirectoryAfterFailure(t *testing.T) {
	repo := newRepository(t)
	parent := t.TempDir()
	dir, err := extractTemp(repo, "definitely-not-an-object", parent)
	if err == nil {
		t.Fatal("Extract succeeded for an invalid object")
	}
	if dir != "" {
		t.Fatalf("failed Extract returned directory %q", dir)
	}
	matches, globErr := filepath.Glob(filepath.Join(parent, "fak-committed-tree-*"))
	if globErr != nil {
		t.Fatal(globErr)
	}
	if len(matches) != 0 {
		t.Fatalf("failed Extract left temporary directories: %v", matches)
	}
}

func TestExtractRepresentativeScaleWithinBudgetAndIgnoresAutoCRLF(t *testing.T) {
	repo := newRepository(t)
	want := [][]byte{
		[]byte("line one\r\nline two\r\n"),
		[]byte("line one\nline two\n"),
		{0, 1, 2, '\r', '\n', 0xff},
		[]byte("small blob without a trailing newline"),
	}
	tree := importScaleTree(t, repo, representativeFileCount, want)

	for _, autoCRLF := range []string{"true", "false"} {
		t.Run("core.autocrlf="+autoCRLF, func(t *testing.T) {
			runGit(t, repo, nil, "config", "core.autocrlf", autoCRLF)
			start := time.Now()
			dir, err := Extract(repo, tree)
			elapsed := time.Since(start)
			if err != nil {
				t.Fatalf("Extract: %v", err)
			}
			defer func() { _ = os.RemoveAll(dir) }()
			if elapsed >= 60*time.Second {
				t.Fatalf("materialized %d files in %s, want <60s", representativeFileCount, elapsed)
			}

			count := 0
			err = filepath.WalkDir(dir, func(_ string, entry fs.DirEntry, err error) error {
				if err != nil {
					return err
				}
				if !entry.IsDir() {
					count++
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
			if count != representativeFileCount {
				t.Fatalf("materialized files = %d, want %d", count, representativeFileCount)
			}
			for _, i := range []int{0, 1, 2, 3, representativeFileCount / 2, representativeFileCount - 1} {
				got, err := os.ReadFile(filepath.Join(dir, scalePath(i)))
				if err != nil {
					t.Fatalf("read scale file %d: %v", i, err)
				}
				if !bytes.Equal(got, want[i%len(want)]) {
					t.Fatalf("scale file %d under core.autocrlf=%s = %v, want raw %v", i, autoCRLF, got, want[i%len(want)])
				}
			}
			t.Logf("materialized_files=%d core.autocrlf=%s elapsed=%s budget=60s", count, autoCRLF, elapsed)
		})
	}
}

func assertPerm(t *testing.T, path string, want os.FileMode) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != want {
		t.Fatalf("%s mode = %#o, want %#o", path, got, want)
	}
}

func newRepository(t *testing.T) string {
	t.Helper()
	repo := filepath.Join(t.TempDir(), "repo")
	if err := os.Mkdir(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	runGit(t, repo, nil, "init", "--quiet")
	return repo
}

func importFocusedTree(t *testing.T, repo string, raw []byte) string {
	t.Helper()
	var stream bytes.Buffer
	writeFastImportBlob(&stream, 1, raw)
	writeFastImportBlob(&stream, 2, []byte("regular.bin"))
	stream.WriteString("commit refs/heads/child\nmark :3\ncommitter Fixture <fixture@example.invalid> 0 +0000\ndata 0\n\n")
	stream.WriteString("commit refs/heads/main\ncommitter Fixture <fixture@example.invalid> 0 +0000\ndata 0\n")
	stream.WriteString("M 100644 :1 regular.bin\n")
	stream.WriteString("M 100755 :1 nested/executable.bin\n")
	stream.WriteString("M 120000 :2 link\n")
	stream.WriteString("M 160000 :3 module\n\n")
	stream.WriteString("done\n")
	runGit(t, repo, &stream, "fast-import", "--quiet")
	return strings.TrimSpace(runGit(t, repo, nil, "rev-parse", "refs/heads/main"))
}

func importScaleTree(t *testing.T, repo string, count int, bodies [][]byte) string {
	t.Helper()
	var stream bytes.Buffer
	for i, body := range bodies {
		writeFastImportBlob(&stream, i+1, body)
	}
	stream.WriteString("commit refs/heads/main\ncommitter Fixture <fixture@example.invalid> 0 +0000\ndata 0\n")
	for i := 0; i < count; i++ {
		fmt.Fprintf(&stream, "M 100644 :%d %s\n", i%len(bodies)+1, scalePath(i))
	}
	stream.WriteString("\ndone\n")
	runGit(t, repo, &stream, "fast-import", "--quiet")
	return strings.TrimSpace(runGit(t, repo, nil, "rev-parse", "refs/heads/main"))
}

func writeFastImportBlob(stream *bytes.Buffer, mark int, body []byte) {
	fmt.Fprintf(stream, "blob\nmark :%d\ndata %d\n", mark, len(body))
	stream.Write(body)
	stream.WriteByte('\n')
}

func scalePath(i int) string {
	return fmt.Sprintf("files/d%03d/f%05d.bin", i/128, i)
}

func runGit(t *testing.T, repo string, stdin *bytes.Buffer, args ...string) string {
	t.Helper()
	cmd := gitCommand(context.Background(), repo, args...)
	if stdin != nil {
		cmd.Stdin = stdin
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
	}
	return string(out)
}

type triggeredDeadline struct {
	done chan struct{}
}

func (*triggeredDeadline) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c *triggeredDeadline) Done() <-chan struct{}     { return c.done }
func (c *triggeredDeadline) Value(any) any             { return nil }
func (c *triggeredDeadline) Err() error {
	select {
	case <-c.done:
		return context.DeadlineExceeded
	default:
		return nil
	}
}

func processTreeHelperCommandWithCount(scenario string, count int) commandFactory {
	return func(ctx context.Context, _ string, args ...string) *exec.Cmd {
		cmd := processTreeHelperCommand(scenario, "")(ctx, "", args...)
		cmd.Env = append(cmd.Env, "GO_WANT_COMMITTEDTREE_LS_COUNT="+strconv.Itoa(count))
		return cmd
	}
}

func processTreeHelperCommand(scenario, pidFile string) commandFactory {
	return func(ctx context.Context, _ string, args ...string) *exec.Cmd {
		role := "unknown"
		if len(args) > 0 && args[0] == "ls-tree" {
			role = "ls"
		} else if len(args) > 0 && args[0] == "cat-file" {
			role = "cat"
		}
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCommittedTreeProcessTreeHelper$")
		cmd.Env = append(os.Environ(),
			"GO_WANT_COMMITTEDTREE_PROCESS_HELPER="+role+"-"+scenario,
			"GO_WANT_COMMITTEDTREE_PID_FILE="+pidFile,
		)
		return cmd
	}
}

func spawnStdoutHoldingDescendant() {
	cmd := exec.Command(os.Args[0], "-test.run=^TestCommittedTreeProcessTreeHelper$")
	cmd.Env = append(os.Environ(), "GO_WANT_COMMITTEDTREE_PROCESS_HELPER=descendant")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		os.Exit(3)
	}
	if err := os.WriteFile(os.Getenv("GO_WANT_COMMITTEDTREE_PID_FILE"), []byte(strconv.Itoa(cmd.Process.Pid)), 0o600); err != nil {
		_ = cmd.Process.Kill()
		os.Exit(4)
	}
}

func waitForHelperPID(path string, result <-chan error, timeout time.Duration) (int, error) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if pid, err := readHelperPID(path); err == nil {
			return pid, nil
		}
		select {
		case err := <-result:
			return 0, fmt.Errorf("extraction returned before descendant started: %w", err)
		case <-time.After(25 * time.Millisecond):
		}
	}
	return 0, fmt.Errorf("timed out waiting for descendant PID")
}

func readHelperPID(path string) (int, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(raw)))
	if err != nil || pid <= 0 {
		return 0, fmt.Errorf("invalid helper PID %q", raw)
	}
	return pid, nil
}

func windowsProcessAlive(pid int) bool {
	out, err := exec.Command("tasklist.exe", "/FI", fmt.Sprintf("PID eq %d", pid), "/FO", "CSV", "/NH").Output()
	if err != nil {
		return false
	}
	return strings.Contains(string(out), fmt.Sprintf("\",\"%d\",\"", pid))
}

func waitForWindowsProcessGone(pid int, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if !windowsProcessAlive(pid) {
			return true
		}
		time.Sleep(50 * time.Millisecond)
	}
	return !windowsProcessAlive(pid)
}

func killWindowsProcessTree(pid int) {
	if runtime.GOOS == "windows" && pid > 0 && windowsProcessAlive(pid) {
		_ = exec.Command("taskkill.exe", "/PID", strconv.Itoa(pid), "/T", "/F").Run()
	}
}
