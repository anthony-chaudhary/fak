package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipref"
)

func TestWipOwnerRealGitTwoSessionClaims(t *testing.T) {
	repo := initWipOwnerRepo(t)
	writeOwnerPath(t, repo, "live.txt")
	writeOwnerPath(t, repo, "ambiguous.txt")
	writeOwnerPath(t, repo, "expired.txt")
	writeOwnerPath(t, repo, "unclaimed.txt")

	now := time.Now().Unix()
	checkpointOwnerSession(t, repo, "session-a", now, "live.txt", "ambiguous.txt")
	checkpointOwnerSession(t, repo, "session-b", now, "ambiguous.txt")
	checkpointOwnerSession(t, repo, "session-a", now-int64((25*time.Hour)/time.Second), "expired.txt")

	got, err := wipOwner(context.Background(), repo, nil, 24*time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	states := map[string]wipref.OwnState{}
	for _, row := range got.Paths {
		states[row.Path] = row.State
	}
	want := map[string]wipref.OwnState{
		"live.txt":      wipref.OwnClaimedLive,
		"ambiguous.txt": wipref.OwnAmbiguous,
		"expired.txt":   wipref.OwnClaimedExpired,
		"unclaimed.txt": wipref.OwnUnclaimed,
	}
	for path, state := range want {
		if states[path] != state {
			t.Errorf("%s: got %s, want %s", path, states[path], state)
		}
	}
}

func TestRunWipOwnerUnclaimedExitAndJSON(t *testing.T) {
	repo := initWipOwnerRepo(t)
	writeOwnerPath(t, repo, "claimed.txt")
	writeOwnerPath(t, repo, "at-risk.txt")
	checkpointOwnerSession(t, repo, "session-a", time.Now().Unix(), "claimed.txt")

	var out, errOut bytes.Buffer
	code := runWipOwner(&out, &errOut, []string{"-C", repo, "--json", "--unclaimed"})
	if code != 3 {
		t.Fatalf("code=%d, stderr=%s", code, errOut.String())
	}
	var got struct {
		Paths []wipref.Ownership `json:"paths"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("json: %v\n%s", err, out.String())
	}
	if len(got.Paths) != 1 || got.Paths[0].Path != "at-risk.txt" || got.Paths[0].State != wipref.OwnUnclaimed {
		t.Fatalf("unexpected filtered result: %+v", got.Paths)
	}

	out.Reset()
	errOut.Reset()
	code = runWipOwner(&out, &errOut, []string{"-C", repo, "--unclaimed", "claimed.txt"})
	if code != 0 {
		t.Fatalf("claimed-only code=%d, stderr=%s", code, errOut.String())
	}
}

func initWipOwnerRepo(t *testing.T) string {
	t.Helper()
	repo := t.TempDir()
	gitOwner(t, repo, "init", "-q")
	gitOwner(t, repo, "config", "user.email", "test@example.com")
	gitOwner(t, repo, "config", "user.name", "Test")
	if err := os.WriteFile(filepath.Join(repo, "base.txt"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gitOwner(t, repo, "add", "base.txt")
	gitOwner(t, repo, "commit", "-qm", "base")
	return repo
}

func writeOwnerPath(t *testing.T, repo, path string) {
	t.Helper()
	full := filepath.Join(repo, filepath.FromSlash(path))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte(path+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func checkpointOwnerSession(t *testing.T, repo, session string, at int64, paths ...string) {
	t.Helper()
	index := filepath.Join(t.TempDir(), "index")
	env := append(os.Environ(), "GIT_INDEX_FILE="+index)
	gitOwnerEnv(t, repo, env, "read-tree", "HEAD")
	args := append([]string{"add", "--"}, paths...)
	gitOwnerEnv(t, repo, env, args...)
	tree := strings.TrimSpace(gitOwnerEnv(t, repo, env, "write-tree"))
	stamp, err := wipref.EncodeStamp(wipref.Stamp{
		SessionID: session, CheckpointedAt: at,
	})
	if err != nil {
		t.Fatal(err)
	}
	commit := strings.TrimSpace(gitOwnerEnvInput(t, repo, env, stamp, "commit-tree", tree, "-p", "HEAD"))
	gitOwner(t, repo, "update-ref", fmt.Sprintf("%s%s/test-%d", wipref.RefNamespace, session, at), commit)
}

func gitOwner(t *testing.T, repo string, args ...string) string {
	t.Helper()
	return gitOwnerEnv(t, repo, nil, args...)
}
func gitOwnerEnv(t *testing.T, repo string, env []string, args ...string) string {
	t.Helper()
	return gitOwnerEnvInput(t, repo, env, "", args...)
}
func gitOwnerEnvInput(t *testing.T, repo string, env []string, input string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
	if env != nil {
		cmd.Env = env
	}
	if input != "" {
		cmd.Stdin = strings.NewReader(input)
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
	}
	return string(out)
}

func Example_runWipOwner_realGit() {
	fmt.Println("covered by TestWipOwnerRealGitTwoSessionClaims")
	// Output: covered by TestWipOwnerRealGitTwoSessionClaims
}
