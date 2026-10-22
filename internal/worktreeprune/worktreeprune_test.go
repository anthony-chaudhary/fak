package worktreeprune

import (
	"errors"
	"io/fs"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

const (
	windowsLane = "C:/work/fak/.worktrees/ticket-a/.git"
	wslLane     = "/mnt/c/work/fak/.worktrees/ticket-b/.git"
	posixLane   = "/home/dev/fak/.worktrees/ticket-c/.git"
)

// foreignLane is a gitdir the running OS cannot resolve: the shape that stranded
// every unlocked lane when a broad prune ran from the other side of WSL.
func foreignLane() string {
	if runtime.GOOS == "windows" {
		return wslLane
	}
	return windowsLane
}

func writeReg(t *testing.T, common, name, gitdir string, locked bool) string {
	t.Helper()
	admin := filepath.Join(common, "worktrees", name)
	if err := os.MkdirAll(admin, 0o755); err != nil {
		t.Fatal(err)
	}
	if gitdir != "" {
		if err := os.WriteFile(filepath.Join(admin, "gitdir"), []byte(gitdir+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if locked {
		if err := os.WriteFile(filepath.Join(admin, "locked"), []byte("held\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return admin
}

// addOwnWorktree mirrors `git worktree add`: an admin entry whose gitdir names
// the worktree's .git file, and a .git file pointing back at the admin entry.
func addOwnWorktree(t *testing.T, common, name string) (wt, admin string) {
	t.Helper()
	wt = filepath.Join(t.TempDir(), "wt")
	if err := os.MkdirAll(wt, 0o755); err != nil {
		t.Fatal(err)
	}
	admin = writeReg(t, common, name, filepath.ToSlash(filepath.Join(wt, ".git")), false)
	if err := os.WriteFile(filepath.Join(wt, ".git"), []byte("gitdir: "+filepath.ToSlash(admin)+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return wt, admin
}

func mustExist(t *testing.T, p string) {
	t.Helper()
	if _, err := os.Stat(p); err != nil {
		t.Fatalf("%s must survive: %v", p, err)
	}
}

func mustBeGone(t *testing.T, p string) {
	t.Helper()
	if _, err := os.Stat(p); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("%s must be removed, stat err=%v", p, err)
	}
}

// gitForWindowsUNCLane is the gitdir Git for Windows 2.55 records for a lane
// added at \\localhost\C$\...: forward slashes, whatever the input form.
const gitForWindowsUNCLane = "//localhost/C$/work/fak/.worktrees/ticket-unc/.git"

func TestIsForeignOSPathFor(t *testing.T) {
	type row struct {
		path string
		want bool
	}
	posixHost := []row{
		{windowsLane, true},
		{`C:\work\fak\.worktrees\ticket-a\.git`, true},
		{`\\server\share\fak`, true},
		{`\\?\C:\work\fak\.git`, true},
		{gitForWindowsUNCLane, true},
		{"//wsl.localhost/Ubuntu-24.04/home/user/fak/.git", true},
		{"//server/share/fak", true},
		{"//?/C:/work/fak/.worktrees/ticket-a/.git", true},
		{"//?/UNC/server/share/fak/.git", true},
		{"//./C:/work/fak/.worktrees/ticket-a/.git", true},
		{wslLane, false},
		{"/mnt/d/lanes/ticket-d/.git", false},
		{posixLane, false},
		{"/root/fak/.worktrees/ticket-e/.git", false},
		{"/tmp/lane/.git", false},
		{"", false},
	}
	cases := map[string][]row{
		"linux":  posixHost,
		"darwin": posixHost,
		"windows": {
			{windowsLane, false},
			{`c:\work\fak`, false},
			{`\\server\share\fak`, false},
			{wslLane, true},
			{"/mnt/d/lanes/ticket-d/.git", true},
			{posixLane, true},
			{"/home/user/fak/.worktrees/ticket-c/.git", true},
			{"/root/fak/.worktrees/ticket-e/.git", true},
			{"/tmp/lane/.git", true},
			{"/", true},
			{`\tmp\lane`, true},
			{gitForWindowsUNCLane, true},
			{"//wsl.localhost/Ubuntu-24.04/home/user/fak/.git", true},
			{"//?/C:/work/fak/.worktrees/ticket-a/.git", true},
			{"//./C:/work/fak/.worktrees/ticket-a/.git", true},
			{"", false},
		},
	}
	for goos, rows := range cases {
		for _, tc := range rows {
			if got := isForeignOSPathFor(goos, tc.path); got != tc.want {
				t.Errorf("isForeignOSPathFor(%q, %q) = %v, want %v", goos, tc.path, got, tc.want)
			}
		}
	}
}

func TestForeignRegistrationsFoldsFakeSet(t *testing.T) {
	regs := []Registration{
		{Name: "ticket-a", Gitdir: windowsLane},
		{Name: "ticket-b", Gitdir: wslLane},
		{Name: "ticket-c", Gitdir: posixLane},
		{Name: "no-gitdir"},
	}
	names := func(rs []Registration) []string {
		var out []string
		for _, r := range rs {
			out = append(out, r.Name)
		}
		return out
	}
	if got := names(foreignFor("linux", regs)); len(got) != 1 || got[0] != "ticket-a" {
		t.Fatalf("linux foreign = %v, want [ticket-a]", got)
	}
	if got := names(foreignFor("windows", regs)); len(got) != 2 || got[0] != "ticket-b" || got[1] != "ticket-c" {
		t.Fatalf("windows foreign = %v, want [ticket-b ticket-c]", got)
	}
}

func TestGuardBroadPruneRefusesForeignRegistration(t *testing.T) {
	cases := []struct {
		name    string
		goos    string
		gitdirs []string
		want    error
	}{
		{"linux native only", "linux", []string{posixLane, wslLane}, nil},
		{"linux sees windows lane", "linux", []string{posixLane, windowsLane}, ErrForeignRegistration},
		{"linux sees forward-slash unc lane", "linux", []string{posixLane, "//wsl.localhost/Ubuntu-24.04/home/user/fak/.worktrees/ticket-a/.git"}, ErrForeignRegistration},
		{"linux sees git for windows unc lane", "linux", []string{posixLane, gitForWindowsUNCLane}, ErrForeignRegistration},
		{"linux sees device-path lane", "linux", []string{posixLane, "//?/C:/work/fak/.worktrees/ticket-a/.git"}, ErrForeignRegistration},
		{"darwin native only", "darwin", []string{posixLane, "/tmp/lane/.git"}, nil},
		{"darwin sees forward-slash unc lane", "darwin", []string{posixLane, "//server/share/fak/.git"}, ErrForeignRegistration},
		{"windows native only", "windows", []string{windowsLane}, nil},
		{"windows sees wsl mount lane", "windows", []string{windowsLane, wslLane}, ErrForeignRegistration},
		{"windows sees posix lane", "windows", []string{windowsLane, posixLane}, ErrForeignRegistration},
		{"windows sees root-home lane", "windows", []string{windowsLane, "/root/fak/.worktrees/ticket-e/.git"}, ErrForeignRegistration},
		{"windows sees unc lane", "windows", []string{windowsLane, gitForWindowsUNCLane}, ErrForeignRegistration},
		{"locked foreign still refuses", "linux", []string{"locked:" + windowsLane}, ErrForeignRegistration},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			common := t.TempDir()
			for i, g := range tc.gitdirs {
				locked := len(g) > 7 && g[:7] == "locked:"
				if locked {
					g = g[7:]
				}
				writeReg(t, common, string(rune('a'+i)), g, locked)
			}
			err := guardFor(tc.goos, common)
			if tc.want == nil && err != nil {
				t.Fatalf("guard refused a native-only set: %v", err)
			}
			if tc.want != nil && !errors.Is(err, tc.want) {
				t.Fatalf("guard err = %v, want errors.Is %v", err, tc.want)
			}
		})
	}
}

func TestGuardBroadPruneAllowsEmptyAndFailsClosedOnUnreadable(t *testing.T) {
	if err := guardFor("linux", t.TempDir()); err != nil {
		t.Fatalf("no worktrees dir must allow: %v", err)
	}
	common := t.TempDir()
	if err := os.WriteFile(filepath.Join(common, "worktrees"), []byte("not a dir"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := guardFor(runtime.GOOS, common); !errors.Is(err, ErrRegistrationsUnreadable) {
		t.Fatalf("unreadable set err = %v, want ErrRegistrationsUnreadable", err)
	}
}

// TestTargetedCleanupRemovesOnlyOwnBesideForeignLanes is the incident contract:
// with foreign-OS lanes registered, the broad prune is refused, yet the caller's
// own registration is still removed and every peer entry survives untouched.
func TestTargetedCleanupRemovesOnlyOwnBesideForeignLanes(t *testing.T) {
	common := t.TempDir()
	foreignAdmin := writeReg(t, common, "ticket-foreign", foreignLane(), false)
	nativePeer := filepath.Join(t.TempDir(), "peer")
	if err := os.MkdirAll(nativePeer, 0o755); err != nil {
		t.Fatal(err)
	}
	nativeAdmin := writeReg(t, common, "ticket-native", filepath.ToSlash(filepath.Join(nativePeer, ".git")), false)
	wt, ownAdmin := addOwnWorktree(t, common, "wt")

	if err := GuardBroadPrune(common); !errors.Is(err, ErrForeignRegistration) {
		t.Fatalf("GuardBroadPrune = %v, want ErrForeignRegistration", err)
	}
	reg, err := RemoveOwnRegistration(common, wt)
	if err != nil {
		t.Fatalf("RemoveOwnRegistration: %v", err)
	}
	if reg.Name != "wt" {
		t.Fatalf("removed %q, want own entry wt", reg.Name)
	}
	mustBeGone(t, ownAdmin)
	mustExist(t, filepath.Join(foreignAdmin, "gitdir"))
	mustExist(t, filepath.Join(nativeAdmin, "gitdir"))
	if got, err := os.ReadFile(filepath.Join(foreignAdmin, "gitdir")); err != nil || string(got) != foreignLane()+"\n" {
		t.Fatalf("foreign gitdir changed: %q err=%v", got, err)
	}
}

func TestRemoveOwnRegistrationMatchesGitdirWhenDotGitGone(t *testing.T) {
	common := t.TempDir()
	wt, ownAdmin := addOwnWorktree(t, common, "wt")
	peerAdmin := writeReg(t, common, "ticket-foreign", foreignLane(), false)
	if err := os.Remove(filepath.Join(wt, ".git")); err != nil {
		t.Fatal(err)
	}
	if _, err := RemoveOwnRegistration(common, wt); err != nil {
		t.Fatalf("RemoveOwnRegistration: %v", err)
	}
	mustBeGone(t, ownAdmin)
	mustExist(t, peerAdmin)
}

func TestRemoveOwnRegistrationRefusals(t *testing.T) {
	t.Run("not registered", func(t *testing.T) {
		common := t.TempDir()
		peerAdmin := writeReg(t, common, "ticket-foreign", foreignLane(), false)
		_, err := RemoveOwnRegistration(common, filepath.Join(t.TempDir(), "never-added"))
		if !errors.Is(err, ErrOwnRegistrationNotFound) {
			t.Fatalf("err = %v, want ErrOwnRegistrationNotFound", err)
		}
		mustExist(t, peerAdmin)
	})
	t.Run("locked", func(t *testing.T) {
		common := t.TempDir()
		wt, ownAdmin := addOwnWorktree(t, common, "wt")
		if err := os.WriteFile(filepath.Join(ownAdmin, "locked"), nil, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := RemoveOwnRegistration(common, wt); !errors.Is(err, ErrOwnRegistrationLocked) {
			t.Fatalf("err = %v, want ErrOwnRegistrationLocked", err)
		}
		mustExist(t, ownAdmin)
	})
}

func TestCommonDirResolvesMainAndLinkedCheckouts(t *testing.T) {
	repo := t.TempDir()
	common := filepath.Join(repo, ".git")
	if err := os.MkdirAll(common, 0o755); err != nil {
		t.Fatal(err)
	}
	if got, err := CommonDir(repo); err != nil || !samePath(got, common) {
		t.Fatalf("main CommonDir = %q, %v; want %q", got, err, common)
	}

	lane, admin := addOwnWorktree(t, common, "lane")
	if got, err := CommonDir(lane); err != nil || !samePath(got, common) {
		t.Fatalf("linked CommonDir (heuristic) = %q, %v; want %q", got, err, common)
	}
	if err := os.WriteFile(filepath.Join(admin, "commondir"), []byte("../..\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := CommonDir(lane); err != nil || !samePath(got, common) {
		t.Fatalf("linked CommonDir (commondir file) = %q, %v; want %q", got, err, common)
	}

	if _, err := CommonDir(t.TempDir()); !errors.Is(err, ErrCommonDirUnresolved) {
		t.Fatalf("non-repo err = %v, want ErrCommonDirUnresolved", err)
	}
}

func TestCleanupOwnOutcomes(t *testing.T) {
	newRepo := func(t *testing.T) (repo, common string) {
		repo = t.TempDir()
		common = filepath.Join(repo, ".git")
		if err := os.MkdirAll(common, 0o755); err != nil {
			t.Fatal(err)
		}
		writeReg(t, common, "ticket-foreign", foreignLane(), false)
		return repo, common
	}

	repo, common := newRepo(t)
	wt, ownAdmin := addOwnWorktree(t, common, "wt")
	if got, err := CleanupOwn(repo, wt); got != OutcomeRemoved || err != nil {
		t.Fatalf("CleanupOwn = %q, %v; want %q", got, err, OutcomeRemoved)
	}
	mustBeGone(t, ownAdmin)
	if got, err := CleanupOwn(repo, wt); got != OutcomeAbsent || err != nil {
		t.Fatalf("second CleanupOwn = %q, %v; want %q", got, err, OutcomeAbsent)
	}
	mustExist(t, filepath.Join(common, "worktrees", "ticket-foreign"))

	repo, common = newRepo(t)
	wt, ownAdmin = addOwnWorktree(t, common, "wt")
	if err := os.WriteFile(filepath.Join(ownAdmin, "locked"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	if got, err := CleanupOwn(repo, wt); got != OutcomeKept || !errors.Is(err, ErrOwnRegistrationLocked) {
		t.Fatalf("locked CleanupOwn = %q, %v; want %q + ErrOwnRegistrationLocked", got, err, OutcomeKept)
	}

	if got, err := CleanupOwn(t.TempDir(), wt); got != OutcomeKept || !errors.Is(err, ErrCommonDirUnresolved) {
		t.Fatalf("non-repo CleanupOwn = %q, %v; want %q + ErrCommonDirUnresolved", got, err, OutcomeKept)
	}
}

// TestCleanupOwnAgainstRealGit witnesses the matching against the gitdir/.git
// files real `git worktree add` writes (realpath, slash form) rather than a
// hand-built layout.
func TestCleanupOwnAgainstRealGit(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	runGit(t, repo, "init", "-q")
	runGit(t, repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "base")
	wt := filepath.Join(t.TempDir(), "wt")
	runGit(t, repo, "worktree", "add", "-q", "--detach", wt)
	common := filepath.Join(repo, ".git")
	foreignAdmin := writeReg(t, common, "ticket-foreign", foreignLane(), false)

	if err := GuardBroadPrune(common); !errors.Is(err, ErrForeignRegistration) {
		t.Fatalf("GuardBroadPrune = %v, want ErrForeignRegistration", err)
	}
	if got, err := CleanupOwn(wt, wt); got != OutcomeRemoved || err != nil {
		t.Fatalf("CleanupOwn from the linked checkout = %q, %v; want %q", got, err, OutcomeRemoved)
	}
	mustBeGone(t, filepath.Join(common, "worktrees", "wt"))
	mustExist(t, filepath.Join(foreignAdmin, "gitdir"))
}

func runGit(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
	for _, kv := range os.Environ() {
		if !strings.HasPrefix(kv, "GIT_") {
			cmd.Env = append(cmd.Env, kv)
		}
	}
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
}

// snapshotTree maps every path below dir to its bytes so a test can prove an
// admin entry survived byte-for-byte.
func snapshotTree(t *testing.T, dir string) map[string]string {
	t.Helper()
	out := map[string]string{}
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil {
			return err
		}
		if d.IsDir() {
			out[rel] = "<dir>"
			return nil
		}
		b, err := os.ReadFile(p)
		out[rel] = string(b)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// TestRemoveOwnRegistrationRefusesPointerWithoutBackPointer: a stale or forged
// .git pointer naming a peer's live admin entry must not delete it. git's
// validate_worktree requires the entry's gitdir to point back, and so does
// findOwn; the refusal keeps every entry, the caller's own included.
func TestRemoveOwnRegistrationRefusesPointerWithoutBackPointer(t *testing.T) {
	for _, withOwn := range []bool{false, true} {
		name := "no_own_entry"
		if withOwn {
			name = "own_entry_beside_peer"
		}
		t.Run(name, func(t *testing.T) {
			repo := t.TempDir()
			common := filepath.Join(repo, ".git")
			peerWt, peerAdmin := addOwnWorktree(t, common, "ticket-peer")
			if err := os.WriteFile(filepath.Join(peerAdmin, "HEAD"), []byte("peer head payload\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			wt := filepath.Join(t.TempDir(), "wt")
			if err := os.MkdirAll(wt, 0o755); err != nil {
				t.Fatal(err)
			}
			var ownAdmin string
			if withOwn {
				ownAdmin = writeReg(t, common, "wt", filepath.ToSlash(filepath.Join(wt, ".git")), false)
			}
			if err := os.WriteFile(filepath.Join(wt, ".git"), []byte("gitdir: "+filepath.ToSlash(peerAdmin)+"\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			before := snapshotTree(t, peerAdmin)

			reg, err := RemoveOwnRegistration(common, wt)
			if !errors.Is(err, ErrOwnRegistrationMismatch) {
				t.Fatalf("RemoveOwnRegistration err = %v, want ErrOwnRegistrationMismatch", err)
			}
			if reg.Name != "" {
				t.Fatalf("refusal returned registration %q, want none", reg.Name)
			}
			if got, err := CleanupOwn(repo, wt); got != OutcomeKept || !errors.Is(err, ErrOwnRegistrationMismatch) {
				t.Fatalf("CleanupOwn = %q, %v; want %q + ErrOwnRegistrationMismatch", got, err, OutcomeKept)
			}
			if after := snapshotTree(t, peerAdmin); !maps.Equal(before, after) {
				t.Fatalf("peer admin entry changed:\nbefore %v\nafter  %v", before, after)
			}
			mustExist(t, filepath.Join(peerWt, ".git"))
			if withOwn {
				mustExist(t, filepath.Join(ownAdmin, "gitdir"))
			}
		})
	}
}

// A .git file alias does not make two checkout directories the same worktree.
func TestRemoveOwnRegistrationRefusesAliasedDotGit(t *testing.T) {
	for _, kind := range []string{"hardlink", "symlink"} {
		t.Run(kind, func(t *testing.T) {
			repo := t.TempDir()
			common := filepath.Join(repo, ".git")
			peerWt, peerAdmin := addOwnWorktree(t, common, "peer")
			callerWt := t.TempDir()
			peerDotgit, callerDotgit := filepath.Join(peerWt, ".git"), filepath.Join(callerWt, ".git")
			link := os.Link
			if kind == "symlink" {
				link = os.Symlink
			}
			if err := link(peerDotgit, callerDotgit); err != nil {
				t.Skipf("%s creation unavailable: %v", kind, err)
			}
			before := snapshotTree(t, peerAdmin)
			if got, err := CleanupOwn(repo, callerWt); got != OutcomeKept || !errors.Is(err, ErrOwnRegistrationMismatch) {
				t.Fatalf("CleanupOwn = %q, %v; want kept + mismatch", got, err)
			}
			// An unusable pointer takes the gitdir fallback, which must also
			// reject the shared file identity in distinct checkout directories.
			if err := os.WriteFile(peerDotgit, []byte("invalid pointer\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := RemoveOwnRegistration(common, callerWt); !errors.Is(err, ErrOwnRegistrationNotFound) {
				t.Fatalf("fallback err = %v, want ErrOwnRegistrationNotFound", err)
			}
			if after := snapshotTree(t, peerAdmin); !maps.Equal(before, after) {
				t.Fatalf("peer admin changed: before %v, after %v", before, after)
			}
		})
	}
}

func TestRemoveOwnRegistrationKeepsSymlinkLockMarker(t *testing.T) {
	for _, target := range []string{"locked", "missing-lock-target"} {
		t.Run(target, func(t *testing.T) {
			repo := t.TempDir()
			common := filepath.Join(repo, ".git")
			wt, admin := addOwnWorktree(t, common, "wt")
			before := snapshotTree(t, admin)
			marker := filepath.Join(admin, "locked")
			if err := os.Symlink(target, marker); err != nil {
				t.Skipf("symlink creation unavailable: %v", err)
			}
			regs, err := ReadRegistrations(common)
			if err != nil || len(regs) != 1 || !regs[0].Locked {
				t.Fatalf("ReadRegistrations = %v, %v; want locked registration", regs, err)
			}
			if got, err := CleanupOwn(repo, wt); got != OutcomeKept || !errors.Is(err, ErrOwnRegistrationLocked) {
				t.Fatalf("CleanupOwn = %q, %v; want kept + locked", got, err)
			}
			if got, err := os.Readlink(marker); err != nil || got != target {
				t.Fatalf("lock marker changed: %q, %v", got, err)
			}
			// Remove only this test's marker to compare the remaining admin bytes.
			if err := os.Remove(marker); err != nil {
				t.Fatal(err)
			}
			if after := snapshotTree(t, admin); !maps.Equal(before, after) {
				t.Fatalf("admin changed: before %v, after %v", before, after)
			}
		})
	}
}

// linkDir returns a path that resolves to target: "plain" is target itself,
// "symlink" needs symlink privilege, "junction" is Windows-only. A kind the host
// cannot create skips the test.
func linkDir(t *testing.T, kind, target string) string {
	t.Helper()
	if kind == "plain" {
		return target
	}
	link := filepath.Join(t.TempDir(), "link")
	switch kind {
	case "symlink":
		if err := os.Symlink(target, link); err != nil {
			t.Skipf("symlink creation not permitted here: %v", err)
		}
	case "junction":
		if runtime.GOOS != "windows" {
			t.Skip("directory junctions are Windows-only")
		}
		if out, err := exec.Command("cmd", "/c", "mklink", "/J", link, target).CombinedOutput(); err != nil {
			t.Skipf("mklink /J: %v: %s", err, out)
		}
	default:
		t.Fatalf("unknown link kind %q", kind)
	}
	t.Cleanup(func() { _ = os.Remove(link) })
	return link
}

// TestRemoveOwnRegistrationMatchesThroughLinkedParent: git records the resolved
// real path in gitdir while the caller may name its worktree through a
// symlinked or junctioned parent. Once the .git file, or the whole checkout, is
// gone, the gitdir fallback must still match only the caller's own entry.
func TestRemoveOwnRegistrationMatchesThroughLinkedParent(t *testing.T) {
	for _, kind := range []string{"plain", "symlink", "junction"} {
		for _, gone := range []string{"dotgit", "checkout"} {
			t.Run(kind+"_"+gone, func(t *testing.T) {
				parent, err := filepath.EvalSymlinks(t.TempDir())
				if err != nil {
					t.Fatal(err)
				}
				realWt := filepath.Join(parent, "wt")
				if err := os.MkdirAll(filepath.Join(realWt, "src"), 0o755); err != nil {
					t.Fatal(err)
				}
				common := t.TempDir()
				ownAdmin := writeReg(t, common, "wt", filepath.ToSlash(filepath.Join(realWt, ".git")), false)
				if err := os.WriteFile(filepath.Join(realWt, ".git"), []byte("gitdir: "+filepath.ToSlash(ownAdmin)+"\n"), 0o644); err != nil {
					t.Fatal(err)
				}
				siblingAdmin := writeReg(t, common, "wt-sibling", filepath.ToSlash(filepath.Join(parent, "wt-sibling", ".git")), false)
				foreignAdmin := writeReg(t, common, "ticket-foreign", foreignLane(), false)
				callerWt := filepath.Join(linkDir(t, kind, parent), "wt")

				victim := filepath.Join(realWt, ".git")
				if gone == "checkout" {
					victim = realWt
				}
				if err := os.RemoveAll(victim); err != nil {
					t.Fatal(err)
				}

				reg, err := RemoveOwnRegistration(common, callerWt)
				if err != nil {
					t.Fatalf("RemoveOwnRegistration(%s): %v", callerWt, err)
				}
				if reg.Name != "wt" {
					t.Fatalf("removed %q, want own entry wt", reg.Name)
				}
				mustBeGone(t, ownAdmin)
				mustExist(t, filepath.Join(siblingAdmin, "gitdir"))
				mustExist(t, filepath.Join(foreignAdmin, "gitdir"))
			})
		}
	}
}

// TestCleanupOwnAgainstRealGitThroughLinkedParent witnesses what real git writes
// for a worktree added through a symlinked or junctioned parent (the resolved
// real path) and that cleanup named through the link still pairs with it, by
// .git pointer and back-pointer, and by gitdir once the .git file is gone.
func TestCleanupOwnAgainstRealGitThroughLinkedParent(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	for _, kind := range []string{"symlink", "junction"} {
		for _, mode := range []string{"pointer", "gitdir"} {
			t.Run(kind+"_"+mode, func(t *testing.T) {
				wt := filepath.Join(linkDir(t, kind, t.TempDir()), "wt")
				repo := t.TempDir()
				runGit(t, repo, "init", "-q")
				runGit(t, repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "base")
				runGit(t, repo, "worktree", "add", "-q", "--detach", wt)
				common := filepath.Join(repo, ".git")
				foreignAdmin := writeReg(t, common, "ticket-foreign", foreignLane(), false)
				if mode == "gitdir" {
					if err := os.Remove(filepath.Join(wt, ".git")); err != nil {
						t.Fatal(err)
					}
				}
				if got, err := CleanupOwn(repo, wt); got != OutcomeRemoved || err != nil {
					t.Fatalf("CleanupOwn through %s = %q, %v; want %q", kind, got, err, OutcomeRemoved)
				}
				mustBeGone(t, filepath.Join(common, "worktrees", "wt"))
				mustExist(t, filepath.Join(foreignAdmin, "gitdir"))
			})
		}
	}
}
