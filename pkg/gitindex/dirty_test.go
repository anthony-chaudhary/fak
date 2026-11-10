package gitindex

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"
)

// isolateGit points git at an empty global config and no system config so the
// temp repos behave identically on every host.
func isolateGit(t testing.TB) {
	t.Helper()
	cfg := filepath.Join(t.TempDir(), "gitconfig")
	if err := os.WriteFile(cfg, []byte("[user]\n\tname = t\n\temail = t@example.invalid\n[init]\n\tdefaultBranch = main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("GIT_CONFIG_GLOBAL", cfg)
	t.Setenv("GIT_CONFIG_NOSYSTEM", "1")
}

func git(t testing.TB, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("git %v: %v: %s", args, err, stderr.String())
	}
	return string(out)
}

func write(t testing.TB, dir, rel, content string) {
	t.Helper()
	p := filepath.Join(dir, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// newRepo makes a committed temp repo. Files are back-dated so the commit's
// index is not racily clean unless a test wants it to be.
func newRepo(t testing.TB, files map[string]string, extraConfig ...string) string {
	t.Helper()
	isolateGit(t)
	dir := t.TempDir()
	git(t, dir, "init", "-q")
	git(t, dir, "config", "core.autocrlf", "false")
	for i := 0; i+1 < len(extraConfig); i += 2 {
		git(t, dir, "config", extraConfig[i], extraConfig[i+1])
	}
	old := time.Now().Add(-time.Hour)
	for rel, c := range files {
		write(t, dir, rel, c)
		if err := os.Chtimes(filepath.Join(dir, filepath.FromSlash(rel)), old, old); err != nil {
			t.Fatal(err)
		}
	}
	git(t, dir, "add", "-A")
	git(t, dir, "commit", "-q", "-m", "init")
	return dir
}

// oracle is git's content-based answer: refresh stat info, then diff-files.
// It rewrites the index, so call it AFTER the code under test.
func oracle(t testing.TB, dir string) []string {
	t.Helper()
	exec.Command("git", "-C", dir, "update-index", "-q", "--refresh").Run() // exit 1 when dirty
	out := git(t, dir, "diff-files", "--name-only", "-z")
	var got []string
	for _, p := range strings.Split(out, "\x00") {
		if p != "" {
			got = append(got, p)
		}
	}
	sort.Strings(got)
	return dedupe(got)
}

func dedupe(s []string) []string {
	var out []string
	for i, p := range s {
		if i == 0 || p != s[i-1] {
			out = append(out, p)
		}
	}
	return out
}

func norm(s []string) []string {
	if len(s) == 0 {
		return nil
	}
	return s
}

// parity runs every mode and asserts each equals git's oracle.
func parity(t *testing.T, dir string) Result {
	t.Helper()
	auto, err := ModifiedTrackedWith(dir, Options{})
	if err != nil {
		t.Fatalf("auto: %v", err)
	}
	viaGit, err := ModifiedTrackedWith(dir, Options{Mode: FallbackGit})
	if err != nil {
		t.Fatalf("git mode: %v", err)
	}
	want := oracle(t, dir)
	if !reflect.DeepEqual(norm(auto.Modified), norm(want)) {
		t.Errorf("auto = %q, git diff-files = %q (maybe=%q)", auto.Modified, want, auto.Maybe)
	}
	if !reflect.DeepEqual(norm(viaGit.Modified), norm(want)) {
		t.Errorf("FallbackGit = %q, git diff-files = %q", viaGit.Modified, want)
	}
	return auto
}

var baseFiles = map[string]string{
	"a.txt":                 "alpha\n",
	"b.txt":                 "bravo\n",
	"dir/c.txt":             "charlie\n",
	"dir/sub/d.txt":         "delta\n",
	"dir/sub/deeper/e.txt":  "echo\n",
	"dir/sub/deeper/e2.txt": "echo2\n",
	"z/last.go":             "package z\n",
}

func TestParityCleanModifyDelete(t *testing.T) {
	dir := newRepo(t, baseFiles)
	r := parity(t, dir)
	if len(r.Modified) != 0 || r.UsedGit || r.Hashed != 0 {
		t.Fatalf("clean repo: %+v (want no hashing, no git)", r)
	}
	write(t, dir, "a.txt", "alpha changed\n")
	write(t, dir, "dir/sub/d.txt", "DELTA\n") // same size, new mtime
	if err := os.Remove(filepath.Join(dir, "dir", "c.txt")); err != nil {
		t.Fatal(err)
	}
	r = parity(t, dir)
	if want := []string{"a.txt", "dir/c.txt", "dir/sub/d.txt"}; !reflect.DeepEqual(r.Modified, want) {
		t.Fatalf("got %q want %q", r.Modified, want)
	}
	if r.UsedGit {
		t.Fatalf("plain edits should not need git: maybe=%q", r.Maybe)
	}
}

func TestParityTouchedButUnchanged(t *testing.T) {
	dir := newRepo(t, baseFiles)
	now := time.Now().Add(-time.Minute)
	os.Chtimes(filepath.Join(dir, "b.txt"), now, now)
	write(t, dir, "dir/c.txt", "charlie\n") // rewritten, identical bytes
	r := parity(t, dir)
	if len(r.Modified) != 0 || r.Hashed < 2 {
		t.Fatalf("touched-only must hash and stay clean: %+v", r)
	}
}

func TestParityRacySameSecondEdit(t *testing.T) {
	isolateGit(t)
	dir := t.TempDir()
	git(t, dir, "init", "-q")
	git(t, dir, "config", "core.autocrlf", "false")
	write(t, dir, "racy.txt", "aaaa\n")
	write(t, dir, "other.txt", "x\n")
	// Pin the file's mtime at/after the index write so the entry is racy.
	future := time.Now().Add(30 * time.Second).Truncate(time.Second)
	p := filepath.Join(dir, "racy.txt")
	os.Chtimes(p, future, future)
	git(t, dir, "add", "-A")
	// Same size AND same mtime: stat alone cannot see this edit.
	write(t, dir, "racy.txt", "bbbb\n")
	os.Chtimes(p, future, future)
	idx, err := ReadIndexFile(filepath.Join(dir, ".git", "index"))
	if err != nil {
		t.Fatal(err)
	}
	st, _ := os.Stat(p)
	for i := range idx.Entries {
		e := &idx.Entries[i]
		if e.Path == "racy.txt" && (!statMatches(e, st) || !racy(idx, e)) {
			t.Fatalf("want stat-equal racy entry: %+v index mtime %v", e, idx.Mtime)
		}
	}
	r := parity(t, dir)
	if !reflect.DeepEqual(r.Modified, []string{"racy.txt"}) {
		t.Fatalf("got %q", r.Modified)
	}
}

func TestParityIndexV4AndV3(t *testing.T) {
	files := map[string]string{}
	for k, v := range baseFiles {
		files[k] = v
	}
	for i := 0; i < 40; i++ {
		files[fmt.Sprintf("pkg/longish/prefix/name_%02d.txt", i)] = fmt.Sprint(i, "\n")
	}
	dir := newRepo(t, files)
	for _, ver := range []string{"4", "3", "2", "4"} {
		git(t, dir, "update-index", "--index-version", ver)
		idx, err := ReadIndexFile(filepath.Join(dir, ".git", "index"))
		if err != nil {
			t.Fatalf("v%s: %v", ver, err)
		}
		// git writes v2 when "3" is requested but no entry needs extended flags.
		if fmt.Sprint(idx.Version) != ver && !(ver == "3" && idx.Version == 2) {
			t.Fatalf("version %d want %s", idx.Version, ver)
		}
		var paths []string
		for _, e := range idx.Entries {
			paths = append(paths, e.Path)
		}
		want := strings.Split(strings.TrimRight(git(t, dir, "ls-files", "-z"), "\x00"), "\x00")
		if !reflect.DeepEqual(paths, want) {
			t.Fatalf("v%s paths mismatch:\n got %q\nwant %q", ver, paths, want)
		}
	}
	write(t, dir, "pkg/longish/prefix/name_07.txt", "seven!\n")
	os.Remove(filepath.Join(dir, "pkg", "longish", "prefix", "name_33.txt"))
	r := parity(t, dir)
	if len(r.Modified) != 2 {
		t.Fatalf("v4 got %q", r.Modified)
	}
}

func TestParityFlags(t *testing.T) {
	dir := newRepo(t, baseFiles)
	git(t, dir, "update-index", "--skip-worktree", "a.txt")
	git(t, dir, "update-index", "--assume-unchanged", "b.txt")
	write(t, dir, "a.txt", "skip-worktree edit\n")
	write(t, dir, "b.txt", "assume-unchanged edit\n")
	write(t, dir, "new.txt", "intent\n")
	write(t, dir, "untracked.txt", "never reported\n")
	git(t, dir, "add", "-N", "new.txt")
	idx, err := ReadIndexFile(filepath.Join(dir, ".git", "index"))
	if err != nil {
		t.Fatal(err)
	}
	flags := map[string]string{}
	for _, e := range idx.Entries {
		flags[e.Path] = fmt.Sprint(e.SkipWorktree, e.AssumeValid, e.IntentToAdd)
	}
	if flags["a.txt"] != "true false false" || flags["b.txt"] != "false true false" || flags["new.txt"] != "false false true" {
		t.Fatalf("flags parsed wrong: %v", flags)
	}
	parity(t, dir)
}

func TestParityCRLFAutocrlf(t *testing.T) {
	dir := newRepo(t, baseFiles, "core.autocrlf", "true")
	// CRLF rewrite of an LF blob changes the size: git status/diff-files
	// call it modified without reading content, and so do we.
	write(t, dir, "a.txt", "alpha\r\n")
	write(t, dir, "b.txt", "brav\r\n") // same size as "bravo\n", real edit
	r := parity(t, dir)
	if !reflect.DeepEqual(r.Modified, []string{"a.txt", "b.txt"}) {
		t.Fatalf("autocrlf: %+v", r)
	}
}

func TestParityCRLFSmudgedIndexUsesGit(t *testing.T) {
	dir := newRepo(t, baseFiles, "core.autocrlf", "true")
	write(t, dir, "a.txt", "alpha\r\n")   // clean once converted
	write(t, dir, "b.txt", "bravo two\n") // dirty
	// read-tree zeroes cached stat data (size 0), forcing content compares
	// where eol conversion decides the answer.
	git(t, dir, "read-tree", "HEAD")
	r := parity(t, dir)
	if !r.UsedGit || len(r.Maybe) == 0 {
		t.Fatalf("CRLF-normalised match must defer to git: %+v", r)
	}
	none, err := ModifiedTrackedWith(dir, Options{Mode: FallbackNone})
	if err != nil {
		t.Fatal(err)
	}
	if none.UsedGit || len(none.Modified) < len(r.Modified) {
		t.Fatalf("FallbackNone must be a git-free superset: %+v", none)
	}
}

func TestParityCRLFNoAutocrlf(t *testing.T) {
	dir := newRepo(t, baseFiles) // autocrlf=false: a CRLF rewrite is dirty
	write(t, dir, "a.txt", "alpha\r\n")
	r := parity(t, dir)
	if !reflect.DeepEqual(r.Modified, []string{"a.txt"}) {
		t.Fatalf("%+v", r)
	}
}

func TestParityAttributesFilter(t *testing.T) {
	files := map[string]string{".gitattributes": "*.dat filter=nope\n", "x.dat": "one\n", "y.txt": "y\n"}
	dir := newRepo(t, files)
	write(t, dir, "x.dat", "two\n")
	write(t, dir, "y.txt", "yy\n")
	r := parity(t, dir)
	if !r.UsedGit {
		t.Fatalf("filter attribute must route mismatches to git: %+v", r)
	}
}

func TestParityGitFile(t *testing.T) {
	isolateGit(t)
	base := t.TempDir()
	dir := filepath.Join(base, "wt")
	git(t, base, "init", "-q", "--separate-git-dir", filepath.Join(base, "store.git"), dir)
	if st, err := os.Stat(filepath.Join(dir, ".git")); err != nil || st.IsDir() {
		t.Fatalf(".git should be a file: %v", err)
	}
	git(t, dir, "config", "core.autocrlf", "false")
	write(t, dir, "f.txt", "f\n")
	write(t, dir, "g.txt", "g\n")
	git(t, dir, "add", "-A")
	git(t, dir, "commit", "-q", "-m", "x")
	write(t, dir, "f.txt", "ff\n")
	r := parity(t, dir)
	if !reflect.DeepEqual(r.Modified, []string{"f.txt"}) {
		t.Fatalf("%+v", r)
	}
}

func TestParseIndexRejectsCorruption(t *testing.T) {
	dir := newRepo(t, baseFiles)
	data, err := os.ReadFile(filepath.Join(dir, ".git", "index"))
	if err != nil {
		t.Fatal(err)
	}
	bad := append([]byte(nil), data...)
	bad[20] ^= 0xff
	if _, err := ParseIndex(bad); err == nil || !strings.Contains(err.Error(), "checksum") {
		t.Fatalf("want checksum error, got %v", err)
	}
	if _, err := ParseIndex(data[:10]); err == nil {
		t.Fatal("truncated index accepted")
	}
}

func TestDecodeVarint(t *testing.T) {
	// git varint.c: 0x7f=127, 0x80 0x00=128, 0x80 0x7f=255, 0x81 0x00=256.
	for _, c := range []struct {
		in   []byte
		want uint64
	}{{[]byte{0}, 0}, {[]byte{0x7f}, 127}, {[]byte{0x80, 0}, 128}, {[]byte{0x80, 0x7f}, 255}, {[]byte{0x81, 0}, 256}} {
		got, n := decodeVarint(c.in)
		if got != c.want || n != len(c.in) {
			t.Errorf("%x: got %d/%d want %d/%d", c.in, got, n, c.want, len(c.in))
		}
	}
}

func TestScanAttributes(t *testing.T) {
	cr, content := scanAttributes([]byte("# c\n*.bin -text\n*.log -text whitespace=cr-at-eol\nx merge=union\n"))
	if cr || content {
		t.Fatalf("benign attrs flagged: %v %v", cr, content)
	}
	if cr, content = scanAttributes([]byte("*.sh text eol=lf\n")); !cr || content {
		t.Fatalf("eol attrs: %v %v", cr, content)
	}
	if _, content = scanAttributes([]byte("*.psd filter=lfs\n")); !content {
		t.Fatal("filter not flagged")
	}
}

// repoTop is this repository's working tree (read-only use).
func repoTop(t testing.TB) string {
	out, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		t.Skip("not inside a git checkout")
	}
	return strings.TrimSpace(string(out))
}

// TestParityThisRepo compares against git on the real checkout without
// writing its index (GIT_OPTIONAL_LOCKS=0, porcelain diff refreshes in-core).
func TestParityThisRepo(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	root := repoTop(t)
	got, err := ModifiedTrackedWith(root, Options{})
	if err != nil {
		t.Fatal(err)
	}
	want, err := gitModified("git", root, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(norm(got.Modified), norm(dedupe(want))) {
		t.Fatalf("this repo: gitindex=%q git=%q", got.Modified, want)
	}
	t.Logf("entries=%d hashed=%d maybe=%d usedGit=%v modified=%d", got.Entries, got.Hashed, len(got.Maybe), got.UsedGit, len(got.Modified))
}

func BenchmarkGitIndexModifiedTracked(b *testing.B) {
	root := repoTop(b)
	for i := 0; i < b.N; i++ {
		if _, err := ModifiedTracked(root); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGitStatusPorcelain(b *testing.B) {
	root := repoTop(b)
	for i := 0; i < b.N; i++ {
		cmd := exec.Command("git", "-C", root, "status", "--porcelain")
		cmd.Env = append(os.Environ(), "GIT_OPTIONAL_LOCKS=0")
		if err := cmd.Run(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGitStatusPorcelainNoUntracked(b *testing.B) {
	root := repoTop(b)
	for i := 0; i < b.N; i++ {
		cmd := exec.Command("git", "-C", root, "status", "--porcelain", "--untracked-files=no")
		cmd.Env = append(os.Environ(), "GIT_OPTIONAL_LOCKS=0")
		if err := cmd.Run(); err != nil {
			b.Fatal(err)
		}
	}
}
