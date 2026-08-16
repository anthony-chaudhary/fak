package adjudicator

import (
	"context"
	"crypto/sha1"
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func selfDisposalRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	cmd := exec.Command("git", "init", "-q", root)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}
	return root
}

func selfDisposalWrite(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("scratch"), 0644); err != nil {
		t.Fatal(err)
	}
}

func writeSelfDisposalText(t *testing.T, path, contents string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0644); err != nil {
		t.Fatal(err)
	}
}

func selfDisposalReceipt(a *Adjudicator, trace, path string, seq uint64) {
	call := receiptCall(trace, "write_file", fmt.Sprintf(`{"path":%q}`, path), seq)
	a.ObserveResult(context.Background(), call, &abi.Result{Call: call, Status: abi.StatusOK, Outcome: abi.OutcomeCommitted})
}

func selfDisposalVerdict(a *Adjudicator, trace, command string, seq uint64) abi.Verdict {
	call := receiptCall(trace, "Bash", fmt.Sprintf(`{"command":%q}`, command), seq)
	return a.Adjudicate(context.Background(), call)
}

func TestSelfAuthoredUntrackedRemovalAllowsUnrelatedGlobalInclude(t *testing.T) {
	home := t.TempDir()
	included := filepath.Join(home, "included.gitconfig")
	writeSelfDisposalText(t, included, "[user]\n\tname = CI Runner\n")
	writeSelfDisposalText(t, filepath.Join(home, ".gitconfig"), fmt.Sprintf("[include]\n\tpath = %q\n", filepath.ToSlash(included)))
	t.Setenv("HOME", home)
	if runtime.GOOS == "windows" {
		t.Setenv("USERPROFILE", home)
	}

	root := selfDisposalRepo(t)
	target := filepath.Join(root, "scratch.txt")
	selfDisposalWrite(t, target)
	a := New(Policy{Allow: map[string]bool{"Bash": true}})
	a.receiptRoot = root
	selfDisposalReceipt(a, "trace-include", target, 1)
	call := receiptCall("trace-include", "Bash", `{"command":"rm scratch.txt"}`, 2)
	if !a.selfAuthoredUntrackedRemoval(call, decodeArgs(context.Background(), call)) {
		t.Fatal("unrelated global include suppressed eligible self-disposal")
	}
}

func TestSelfAuthoredUntrackedRemovalAllowsUnrelatedConditionalInclude(t *testing.T) {
	home := t.TempDir()
	included := filepath.Join(home, "runner.gitconfig")
	writeSelfDisposalText(t, included, "[safe]\n\tdirectory = *\n")
	writeSelfDisposalText(t, filepath.Join(home, ".gitconfig"), fmt.Sprintf("[includeIf %q]\n\tpath = %q\n", "gitdir:~/work/", filepath.ToSlash(included)))
	t.Setenv("HOME", home)
	if runtime.GOOS == "windows" {
		t.Setenv("USERPROFILE", home)
	}

	root := selfDisposalRepo(t)
	target := filepath.Join(root, "scratch.txt")
	selfDisposalWrite(t, target)
	a := New(Policy{Allow: map[string]bool{"Bash": true}})
	a.receiptRoot = root
	selfDisposalReceipt(a, "trace-conditional", target, 1)
	call := receiptCall("trace-conditional", "Bash", `{"command":"rm scratch.txt"}`, 2)
	if !a.selfAuthoredUntrackedRemoval(call, decodeArgs(context.Background(), call)) {
		t.Fatal("unrelated conditional include suppressed eligible self-disposal")
	}
}

func TestSelfAuthoredUntrackedRemovalRejectsIncludedExternalExcludes(t *testing.T) {
	home := t.TempDir()
	included := filepath.Join(home, "included.gitconfig")
	writeSelfDisposalText(t, included, "[core]\n\texcludesFile = ~/.global-ignore\n")
	writeSelfDisposalText(t, filepath.Join(home, ".gitconfig"), fmt.Sprintf("[include]\n\tpath = %q\n", filepath.ToSlash(included)))
	t.Setenv("HOME", home)
	if runtime.GOOS == "windows" {
		t.Setenv("USERPROFILE", home)
	}

	root := selfDisposalRepo(t)
	target := filepath.Join(root, "scratch.txt")
	selfDisposalWrite(t, target)
	a := New(Policy{Allow: map[string]bool{"Bash": true}})
	a.receiptRoot = root
	selfDisposalReceipt(a, "trace-excludes", target, 1)
	call := receiptCall("trace-excludes", "Bash", `{"command":"rm scratch.txt"}`, 2)
	if a.selfAuthoredUntrackedRemoval(call, decodeArgs(context.Background(), call)) {
		t.Fatal("included external excludes did not fail closed")
	}
}

func TestSelfAuthoredUntrackedRemovalAllowsOnlyPriorSameTraceReceipt(t *testing.T) {
	root := selfDisposalRepo(t)
	target := filepath.Join(root, "scratch.txt")
	selfDisposalWrite(t, target)
	a := New(Policy{Allow: map[string]bool{"Bash": true}})
	a.receiptRoot = root

	if v := selfDisposalVerdict(a, "trace-a", "rm scratch.txt", 2); v.Kind != abi.VerdictRequireWitness {
		t.Fatalf("intent without receipt = %v, want RequireWitness", v.Kind)
	}
	selfDisposalReceipt(a, "trace-a", target, 1)
	if op, ok := a.AuthoredPath("trace-a", target); !ok || op != 1 {
		t.Fatalf("receipt = %d,%v", op, ok)
	}
	direct := receiptCall("trace-a", "Bash", `{"command":"rm scratch.txt"}`, 2)
	if !a.selfAuthoredUntrackedRemoval(direct, decodeArgs(context.Background(), direct)) {
		parsed, parseOK := plainSingleRMTarget("rm scratch.txt")
		canonical, canonOK := canonicalLocalReceiptPath(root, filepath.Join(root, parsed))
		info, statErr := os.Lstat(canonical)
		prior := a.hasPriorWriteReceipt("trace-a", canonical, 2)
		tracked, indexErr := gitIndexTracks(root, canonical)
		t.Fatalf("direct failed: parsed=%q/%v canonical=%q/%v stat=%v/%v prior=%v tracked=%v/%v root=%q", parsed, parseOK, canonical, canonOK, info, statErr, prior, tracked, indexErr, root)
	}
	if v := selfDisposalVerdict(a, "trace-b", "rm scratch.txt", 2); v.Kind != abi.VerdictRequireWitness {
		t.Fatalf("cross trace = %v, want RequireWitness", v.Kind)
	}
	v := selfDisposalVerdict(a, "trace-a", "rm scratch.txt", 2)
	if v.Kind != abi.VerdictAllow || v.By != "monitor/reversibility" || v.Meta["witness"] != "trace-authored-git-untracked" {
		t.Fatalf("same trace prior receipt = %+v, want witnessed Allow", v)
	}
	if v := selfDisposalVerdict(a, "trace-a", "rm scratch.txt", 1); v.Kind != abi.VerdictRequireWitness {
		t.Fatalf("same/later operation = %v, want RequireWitness", v.Kind)
	}
	if v := selfDisposalVerdict(a, "trace-a", "rm scratch.txt", 0); v.Kind != abi.VerdictRequireWitness {
		t.Fatalf("missing sequence = %v, want RequireWitness", v.Kind)
	}
	a.ResetRun()
	if v := selfDisposalVerdict(a, "trace-a", "rm scratch.txt", 2); v.Kind != abi.VerdictRequireWitness {
		t.Fatalf("after reset = %v, want RequireWitness", v.Kind)
	}
}

func selfDisposalGit(t *testing.T, root string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v: %s", args, err, out)
	}
}

func TestSelfAuthoredRemovalKeepsGitIndexedFilesWitnessed(t *testing.T) {
	for _, tc := range []struct {
		name   string
		commit bool
	}{
		{name: "staged-new"},
		{name: "committed", commit: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := selfDisposalRepo(t)
			target := filepath.Join(root, "owned.txt")
			selfDisposalWrite(t, target)
			selfDisposalGit(t, root, "add", "owned.txt")
			if tc.commit {
				selfDisposalGit(t, root, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-q", "-m", "fixture")
			}
			tracked, err := gitIndexTracks(root, target)
			if err != nil || !tracked {
				t.Fatalf("gitIndexTracks = %v,%v, want true,nil", tracked, err)
			}
			a := New(Policy{Allow: map[string]bool{"Bash": true}})
			a.receiptRoot = root
			selfDisposalReceipt(a, "trace-a", target, 1)
			if v := selfDisposalVerdict(a, "trace-a", "rm owned.txt", 2); v.Kind != abi.VerdictRequireWitness {
				t.Fatalf("indexed removal = %v, want RequireWitness", v.Kind)
			}
		})
	}
}

func TestGitIndexTracksUntrackedAndRejectsCorruption(t *testing.T) {
	root := selfDisposalRepo(t)
	trackedPath := filepath.Join(root, "tracked.txt")
	selfDisposalWrite(t, trackedPath)
	selfDisposalGit(t, root, "add", "tracked.txt")
	target := filepath.Join(root, "loose.txt")
	selfDisposalWrite(t, target)
	tracked, err := gitIndexTracks(root, target)
	if err != nil || tracked {
		t.Fatalf("untracked = %v,%v, want false,nil", tracked, err)
	}
	index := filepath.Join(root, ".git", "index")
	data, err := os.ReadFile(index)
	if err != nil {
		t.Fatal(err)
	}
	data[len(data)-1] ^= 0xff
	if err := os.WriteFile(index, data, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := gitIndexTracks(root, target); err == nil {
		t.Fatal("corrupt index did not fail closed")
	}
}

func selfDisposalCall(trace, tool, command, workdir string, seq uint64) *abi.ToolCall {
	args := fmt.Sprintf(`{"command":%q}`, command)
	if workdir != "" {
		args = fmt.Sprintf(`{"command":%q,"workdir":%q}`, command, workdir)
	}
	return receiptCall(trace, tool, args, seq)
}

func TestSelfAuthoredUntrackedRemovalAliasesAndWorkdir(t *testing.T) {
	root := selfDisposalRepo(t)
	target := filepath.Join(root, "sub", "scratch.txt")
	selfDisposalWrite(t, target)
	a := New(Policy{Allow: map[string]bool{"Bash": true}})
	a.receiptRoot = root
	selfDisposalReceipt(a, "trace-a", target, 1)
	cases := []struct{ name, command, workdir string }{
		{"relative", "rm sub/scratch.txt", ""},
		{"end-of-options", "rm -- sub/scratch.txt", ""},
		{"absolute", `rm "` + filepath.ToSlash(target) + `"`, ""},
		{"workdir", "rm scratch.txt", filepath.Join(root, "sub")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			call := selfDisposalCall("trace-a", "Bash", tc.command, tc.workdir, 2)
			if !a.selfAuthoredUntrackedRemoval(call, decodeArgs(context.Background(), call)) {
				t.Fatal("helper rejected eligible alias")
			}
			v := a.Adjudicate(context.Background(), call)
			if v.Kind != abi.VerdictAllow || v.Meta["witness"] != "trace-authored-git-untracked" {
				t.Fatalf("verdict = %+v", v)
			}
		})
	}
}

func TestSelfAuthoredUntrackedRemovalProtectsIgnoredReceipt(t *testing.T) {
	root := selfDisposalRepo(t)
	if err := os.WriteFile(filepath.Join(root, ".gitignore"), []byte("*.log\n"), 0644); err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(root, "scratch.log")
	selfDisposalWrite(t, target)
	a := New(Policy{Allow: map[string]bool{"Bash": true}})
	a.receiptRoot = root
	selfDisposalReceipt(a, "trace-a", target, 1)
	if v := selfDisposalVerdict(a, "trace-a", "rm scratch.log", 2); v.Kind != abi.VerdictRequireWitness {
		t.Fatalf("ignored receipt-authored file = %+v, want RequireWitness", v)
	}
}

func TestGitIndexTracksRejectsUnsupportedVersion(t *testing.T) {
	root := selfDisposalRepo(t)
	target := filepath.Join(root, "loose.txt")
	selfDisposalWrite(t, target)
	tracked := filepath.Join(root, "tracked.txt")
	selfDisposalWrite(t, tracked)
	selfDisposalGit(t, root, "add", "tracked.txt")
	index := filepath.Join(root, ".git", "index")
	data, err := os.ReadFile(index)
	if err != nil {
		t.Fatal(err)
	}
	binary.BigEndian.PutUint32(data[4:8], 4)
	sum := sha1.Sum(data[:len(data)-sha1.Size])
	copy(data[len(data)-sha1.Size:], sum[:])
	if err := os.WriteFile(index, data, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := gitIndexTracks(root, target); err == nil {
		t.Fatal("unsupported index version did not fail closed")
	}
}

func TestSelfAuthoredUntrackedRemovalRejectsNonPlainCommands(t *testing.T) {
	root := selfDisposalRepo(t)
	target := filepath.Join(root, "scratch.txt")
	selfDisposalWrite(t, target)
	a := New(Policy{Allow: map[string]bool{"Bash": true, "shell": true, "shell_command": true, "Exec": true}})
	a.receiptRoot = root
	selfDisposalReceipt(a, "trace-a", target, 1)

	cases := []struct {
		name    string
		tool    string
		command string
	}{
		{"force", "Bash", "rm -f scratch.txt"},
		{"recursive", "Bash", "rm -r scratch.txt"},
		{"multiple", "Bash", "rm scratch.txt other.txt"},
		{"glob", "Bash", "rm *.txt"},
		{"variable", "Bash", "rm $x"},
		{"substitution", "Bash", "rm $(echo scratch.txt)"},
		{"backticks", "Bash", "rm `echo scratch.txt`"},
		{"chain", "Bash", "rm scratch.txt && echo x"},
		{"env-wrapper", "Bash", "env rm scratch.txt"},
		{"assignment", "Bash", "X=1 rm scratch.txt"},
		{"absolute-program", "Bash", "/bin/rm scratch.txt"},
		{"other-absolute-program", "Bash", "/tmp/rm scratch.txt"},
		{"unsupported-tool", "Exec", "rm scratch.txt"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			call := selfDisposalCall("trace-a", tc.tool, tc.command, "", 2)
			if a.selfAuthoredUntrackedRemoval(call, decodeArgs(context.Background(), call)) {
				t.Fatal("non-plain command admitted by helper")
			}
			if v := a.Adjudicate(context.Background(), call); v.Kind != abi.VerdictRequireWitness {
				t.Fatalf("verdict = %+v, want RequireWitness", v)
			}
		})
	}
}

func TestSelfAuthoredUntrackedRemovalRejectsFailedWriteAndNonRegularTargets(t *testing.T) {
	root := selfDisposalRepo(t)
	a := New(Policy{Allow: map[string]bool{"Bash": true}})
	a.receiptRoot = root

	failed := filepath.Join(root, "failed.txt")
	selfDisposalWrite(t, failed)
	failedCall := receiptCall("trace-a", "write_file", fmt.Sprintf(`{"path":%q}`, failed), 1)
	a.ObserveResult(context.Background(), failedCall, &abi.Result{Call: failedCall, Status: abi.StatusError, Outcome: abi.OutcomeCommitted})
	if v := selfDisposalVerdict(a, "trace-a", "rm failed.txt", 2); v.Kind != abi.VerdictRequireWitness {
		t.Fatalf("failed write = %+v, want RequireWitness", v)
	}

	dir := filepath.Join(root, "scratch-dir")
	if err := os.Mkdir(dir, 0755); err != nil {
		t.Fatal(err)
	}
	selfDisposalReceipt(a, "trace-a", dir, 1)
	if v := selfDisposalVerdict(a, "trace-a", "rm scratch-dir", 2); v.Kind != abi.VerdictRequireWitness {
		t.Fatalf("directory = %+v, want RequireWitness", v)
	}

	missing := filepath.Join(root, "missing.txt")
	selfDisposalReceipt(a, "trace-a", missing, 1)
	if v := selfDisposalVerdict(a, "trace-a", "rm missing.txt", 2); v.Kind != abi.VerdictRequireWitness {
		t.Fatalf("missing = %+v, want RequireWitness", v)
	}

	realTarget := filepath.Join(root, "real.txt")
	selfDisposalWrite(t, realTarget)
	link := filepath.Join(root, "link.txt")
	if err := os.Symlink(realTarget, link); err != nil {
		t.Logf("symlink unavailable: %v", err)
		return
	}
	selfDisposalReceipt(a, "trace-a", link, 1)
	if v := selfDisposalVerdict(a, "trace-a", "rm link.txt", 2); v.Kind != abi.VerdictRequireWitness {
		t.Fatalf("symlink = %+v, want RequireWitness", v)
	}
}
