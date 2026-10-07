package workerworktree

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

type preservingWorktreeConfigFixture struct {
	reapProofFixture
	parent, peer, workers, source, hooks string
}

func newPreservingWorktreeConfigFixture(t *testing.T, linkedParent bool) preservingWorktreeConfigFixture {
	t.Helper()
	f := preservingWorktreeConfigFixture{reapProofFixture: newQualifiedPreservingFixture(t)}
	f.workers = preservingTempDir(t)
	f.parent = f.repo
	f.peer = Path("cmd", "config-peer", f.workers)
	preservingFixtureGit(t, f.repo, "worktree", "add", "--detach", f.peer, f.base)
	if linkedParent {
		f.parent = filepath.Join(preservingTempDir(t), "source")
		preservingFixtureGit(t, f.repo, "worktree", "add", "--detach", f.parent, f.base)
	}
	preservingFixtureGit(t, f.repo, "config", "extensions.worktreeConfig", "true")
	f.hooks = filepath.Join(preservingTempDir(t), "disabled-hooks")
	if err := os.Mkdir(f.hooks, 0700); err != nil {
		t.Fatal(err)
	}
	preservingFixtureGit(t, f.parent, "config", "--worktree", "core.bare", "false")
	preservingFixtureGit(t, f.parent, "config", "--worktree", "core.hooksPath", f.hooks)
	f.source = preservingWorktreeConfigGitPath(t, f.parent, "config.worktree")
	preservingFixtureGit(t, f.peer, "config", "--worktree", "core.bare", "false")
	preservingFixtureGit(t, f.peer, "config", "--worktree", "core.hooksPath", filepath.Join(f.hooks, "peer"))
	for _, root := range []string{f.parent, f.peer} {
		if err := os.WriteFile(filepath.Join(root, "peer.txt"), []byte("dirty evidence must survive\n"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	return f
}

func preservingWorktreeConfigGitPath(t *testing.T, root, name string) string {
	t.Helper()
	path := strings.TrimSuffix(preservingFixtureGit(t, root, "rev-parse", "--git-path", name), "\n")
	if path == "" || strings.Contains(path, "\n") {
		t.Fatalf("invalid fixture Git path: %q", path)
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(root, path)
	}
	return filepath.Clean(path)
}

func preservingWorktreeConfigRead(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func preservingWorktreeConfigSnapshot(t *testing.T, f preservingWorktreeConfigFixture) map[string]string {
	t.Helper()
	before := map[string]string{}
	for _, root := range []string{f.repo, f.parent, f.peer} {
		for _, name := range []string{"HEAD", "index", "config"} {
			path := preservingWorktreeConfigGitPath(t, root, name)
			before[path] = preservingWorktreeConfigRead(t, path)
		}
		before[filepath.Join(root, "peer.txt")] = preservingWorktreeConfigRead(t, filepath.Join(root, "peer.txt"))
	}
	for _, root := range []string{f.parent, f.peer} {
		path := preservingWorktreeConfigGitPath(t, root, "config.worktree")
		before[path] = preservingWorktreeConfigRead(t, path)
	}
	return before
}

func preservingWorktreeConfigAssertSnapshot(t *testing.T, before map[string]string) {
	t.Helper()
	for path, want := range before {
		if got := preservingWorktreeConfigRead(t, path); got != want {
			t.Fatalf("parent/common/peer evidence changed at %s", path)
		}
	}
}

func preservingWorktreeConfigPrepare(t *testing.T, f preservingWorktreeConfigFixture, key string, git GitRunner) Result {
	t.Helper()
	return preparePreserving(context.Background(), f.parent, "cmd", key, f.base, f.workers,
		OwnerStamp{PID: os.Getpid(), LeaseID: "config-copy-admission"}, git, func(context.Context) error { return nil })
}

func preservingWorktreeConfigIsAdd(args []string, target, base string) bool {
	// Match the complete fixture command against its expected target and base.
	want := []string{
		"-c", "gc.worktreePruneExpire=never", "-c", "core.longpaths=true",
		"worktree", "add", "--detach", target, base,
	}
	return slices.Equal(args, want)
}

func preservingWorktreeConfigIsList(args []string) bool {
	return strings.Join(args, " ") == "config --null --show-origin --list"
}

func preservingWorktreeConfigAssertRefused(t *testing.T, res Result, wantAdded, wantStatus bool, calls []string) {
	t.Helper()
	if res.OK || !strings.Contains(res.Reason, "no ready receipt emitted") {
		t.Fatalf("ready result after refused proof: %+v", res)
	}
	added, status := false, false
	for _, call := range calls {
		added = added || strings.Contains(call, "worktree add")
		status = status || strings.HasPrefix(call, "status ")
	}
	if added != wantAdded || status != wantStatus {
		t.Fatalf("unexpected phase reached: added=%v status=%v calls=%v", added, status, calls)
	}
	if wantAdded {
		if _, err := os.Lstat(filepath.Join(res.Path, ".git")); err != nil || !res.Preserved {
			t.Fatalf("partial target evidence lost: %+v %v", res, err)
		}
	} else if _, err := os.Lstat(res.Path); !os.IsNotExist(err) {
		t.Fatalf("target allocated before policy refusal: %v", err)
	}
}

// Native copy evidence must come from the genuinely qualified executable. The
// existing prerequisite helper skips other versions; a skip is not a witness.
// fak-test:runtime integration est=3s lane=default
func TestPreparePreservingNativeWorktreeConfigCopy(t *testing.T) {
	for _, linked := range []bool{false, true} {
		t.Run(fmt.Sprintf("linked-parent-%v", linked), func(t *testing.T) {
			f := newPreservingWorktreeConfigFixture(t, linked)
			before := preservingWorktreeConfigSnapshot(t, f)
			source := preservingWorktreeConfigRead(t, f.source)
			var calls []string
			git := preservingTestRunner(t, &calls)
			observedCopy := false
			runner := func(dir string, args []string) (int, string) {
				rc, data := git(dir, args)
				if preservingWorktreeConfigIsAdd(args, Path("cmd", "native-config-copy", f.workers), f.base) && rc == 0 {
					// Observe native Git's result before the implementation can run
					// another query or repair a missing/mismatched child config.
					child := preservingWorktreeConfigGitPath(t, args[7], "config.worktree")
					parentInfo, err := os.Lstat(f.source)
					if err != nil {
						t.Fatal(err)
					}
					childInfo, err := os.Lstat(child)
					if err != nil || !childInfo.Mode().IsRegular() || os.SameFile(parentInfo, childInfo) || samePath(child, f.source) {
						t.Fatalf("native copy is not an independent regular file: %s %v", child, err)
					}
					if got := preservingWorktreeConfigRead(t, child); got != source {
						t.Fatalf("native child bytes differ: got=%q want=%q", got, source)
					}
					if got := preservingFixtureGit(t, args[7], "config", "--get", "core.hooksPath"); got != f.hooks+"\n" {
						t.Fatalf("child effective hooks changed: %q", got)
					}
					if got := preservingFixtureGit(t, args[7], "config", "--get", "core.bare"); got != "false\n" {
						t.Fatalf("child effective bare changed: %q", got)
					}
					origin := preservingFixtureGit(t, args[7], "config", "--null", "--show-origin", "--get", "core.hooksPath")
					originRecords, err := preservingNULRecords(origin)
					if err != nil || len(originRecords) != 2 || !strings.HasPrefix(originRecords[0], "file:") ||
						!samePath(strings.TrimPrefix(originRecords[0], "file:"), child) || originRecords[1] != f.hooks {
						t.Fatalf("effective hook did not come from native child origin: %q", origin)
					}
					if got := preservingWorktreeConfigGitPath(t, args[7], "hooks/post-checkout"); got != filepath.Join(f.hooks, "post-checkout") {
						t.Fatalf("child effective hook resolution changed: %q", got)
					}
					observedCopy = true
				}
				return rc, data
			}
			res := preservingWorktreeConfigPrepare(t, f, "native-config-copy", runner)
			if !res.OK || res.Reused || !observedCopy {
				t.Fatalf("exact native copy not admitted: %+v observed=%v", res, observedCopy)
			}
			assertPreservingTreeLive(t, f.repo, res.Path)
			assertPreservingTreeLive(t, f.repo, f.peer)
			if got := strings.TrimSpace(preservingFixtureGit(t, res.Path, "rev-parse", "HEAD")); got != f.base {
				t.Fatalf("pinned child HEAD changed: %q", got)
			}
			if got := preservingWorktreeConfigRead(t, preservingWorktreeConfigGitPath(t, res.Path, "config.worktree")); got != source {
				t.Fatal("child configuration changed after native copy")
			}
			preservingWorktreeConfigAssertSnapshot(t, before)
		})
	}
}

// fak-test:runtime integration est=8s lane=default
func TestPreservingNativeWorktreeConfigRejectsUnsupportedShape(t *testing.T) {
	for _, kind := range []string{"extra", "include", "conditional-include", "duplicate-bare", "duplicate-hooks", "true-bare", "bare-alias", "core-worktree", "missing-bare", "missing-hooks", "indirect", "missing-after-query"} {
		t.Run(kind, func(t *testing.T) {
			f := newPreservingWorktreeConfigFixture(t, false)
			var baselineCalls []string
			if _, err := preservingCheckoutPolicy(f.parent, f.base, preservingTestRunner(t, &baselineCalls)); err != nil {
				t.Fatalf("unmodified source failed positive control: %v", err)
			}
			key, value := "user.name", "not-in-the-approved-shape"
			wantReason := "worktree configuration requires exactly core.bare=false and core.hooksPath"
			switch kind {
			case "include", "conditional-include":
				included := filepath.Join(preservingTempDir(t), "included.gitconfig")
				if err := os.WriteFile(included, []byte("# even an empty include is outside the envelope\n"), 0600); err != nil {
					t.Fatal(err)
				}
				key, value = "include.path", included
				if kind == "conditional-include" {
					key = "includeIf.onbranch:never-match.path"
				}
			case "duplicate-bare":
				key, value = "core.bare", "false"
			case "duplicate-hooks":
				key, value = "core.hooksPath", f.hooks
			case "true-bare":
				key, value = "core.bare", "true"
				wantReason = "worktree configuration requires literal core.bare=false"
			case "bare-alias":
				key, value = "core.bare", "no"
				wantReason = "worktree configuration requires literal core.bare=false"
			case "core-worktree":
				key, value = "core.worktree", f.parent
			}
			switch kind {
			case "missing-bare", "missing-hooks":
				key = "core.bare"
				if kind == "missing-hooks" {
					key = "core.hooksPath"
				}
				preservingFixtureGit(t, f.parent, "config", "--worktree", "--unset", key)
			case "indirect":
				wantReason = "worktree configuration source unavailable or indirect"
				other := filepath.Join(preservingTempDir(t), "source.gitconfig")
				if err := os.Rename(f.source, other); err != nil {
					t.Fatal(err)
				}
				if err := os.Symlink(other, f.source); err != nil {
					t.Skipf("symlink prerequisite unavailable: %v", err)
				}
			case "missing-after-query":
				wantReason = "configuration source unavailable or indirect"
			default:
				operation := "--replace-all"
				if strings.HasPrefix(kind, "duplicate-") {
					operation = "--add"
				}
				preservingFixtureGit(t, f.parent, "config", "--worktree", operation, key, value)
			}
			var calls []string
			git := preservingTestRunner(t, &calls)
			runner := func(dir string, args []string) (int, string) {
				rc, data := git(dir, args)
				if kind == "missing-after-query" && preservingWorktreeConfigIsList(args) && rc == 0 {
					if err := os.Remove(f.source); err != nil {
						t.Fatal(err)
					}
				}
				return rc, data
			}
			if _, err := preservingCheckoutPolicy(f.parent, f.base, runner); err == nil || !strings.Contains(err.Error(), wantReason) {
				t.Fatalf("unsupported %s source did not cause its intended refusal: got=%v want=%q", kind, err, wantReason)
			}
			for _, call := range calls {
				if strings.Contains(call, "worktree add") || strings.HasPrefix(call, "status ") {
					t.Fatalf("proof ran a checkout/helper: %s", call)
				}
			}
		})
	}
}

// Fault injection changes native copy evidence, never the executable version or
// a successful Git result used as positive evidence.
// fak-test:runtime integration est=6s lane=default
func TestPreparePreservingNativeWorktreeConfigRejectsChildCopy(t *testing.T) {
	for _, kind := range []string{"missing", "symlink", "hardlink", "different-bytes", "different-value", "wrong-origin", "wrong-common-origin", "effective-value", "inactive-extension", "copy-error"} {
		t.Run(kind, func(t *testing.T) {
			f := newPreservingWorktreeConfigFixture(t, false)
			before := preservingWorktreeConfigSnapshot(t, f)
			var calls []string
			git := preservingTestRunner(t, &calls)
			child, target, injected := "", "", false
			runner := func(dir string, args []string) (int, string) {
				rc, data := git(dir, args)
				if preservingWorktreeConfigIsAdd(args, Path("cmd", "refuse-child-copy", f.workers), f.base) && rc == 0 {
					target = args[7]
					child = preservingWorktreeConfigGitPath(t, target, "config.worktree")
					switch kind {
					case "missing", "symlink", "hardlink":
						if err := os.Remove(child); err != nil {
							t.Fatal(err)
						}
						var err error
						if kind == "symlink" {
							err = os.Symlink(f.source, child)
						} else if kind == "hardlink" {
							err = os.Link(f.source, child)
						}
						if err != nil {
							t.Skipf("link prerequisite unavailable: %v", err)
						}
					case "different-bytes":
						if err := os.WriteFile(child, []byte(preservingWorktreeConfigRead(t, child)+"# changed copy\n"), 0600); err != nil {
							t.Fatal(err)
						}
					case "different-value":
						preservingFixtureGit(t, target, "config", "--worktree", "core.hooksPath", filepath.Join(f.hooks, "changed"))
					case "copy-error":
						// Model a failed native add that has already allocated a
						// surviving target; no cleanup or retry may follow it.
						injected = true
						return 128, "injected config.worktree copy failure"
					}
					if kind != "wrong-origin" && kind != "wrong-common-origin" && kind != "effective-value" && kind != "inactive-extension" {
						injected = true
					}
				}
				if child != "" && samePath(dir, target) && preservingWorktreeConfigIsList(args) && rc == 0 &&
					(kind == "wrong-origin" || kind == "wrong-common-origin" || kind == "effective-value" || kind == "inactive-extension") {
					records, err := preservingNULRecords(data)
					if err != nil || len(records)%2 != 0 {
						t.Fatalf("native config records malformed: %v", err)
					}
					for i := 0; i < len(records); i += 2 {
						path := strings.TrimPrefix(records[i], "file:")
						if !filepath.IsAbs(path) {
							path = filepath.Join(dir, path)
						}
						key, _, _ := strings.Cut(records[i+1], "\n")
						key = strings.ToLower(key)
						switch {
						case kind == "wrong-origin" && samePath(path, child),
							kind == "wrong-common-origin" && key == "extensions.worktreeconfig":
							other := filepath.Join(preservingTempDir(t), filepath.Base(path))
							if err := os.WriteFile(other, []byte(preservingWorktreeConfigRead(t, path)), 0600); err != nil {
								t.Fatal(err)
							}
							records[i] = "file:" + other
							injected = true
						case kind == "effective-value" && samePath(path, child) && key == "core.hookspath":
							records[i+1] = key + "\n" + filepath.Join(f.hooks, "changed-effective-value")
							injected = true
						case kind == "inactive-extension" && key == "extensions.worktreeconfig":
							records[i+1] = key + "\nfalse"
							injected = true
						}
					}
					data = strings.Join(records, "\x00") + "\x00"
				}
				return rc, data
			}
			res := preservingWorktreeConfigPrepare(t, f, "refuse-child-copy", runner)
			wantCode := "PRESERVATION_CONFIG_REFUSED"
			wantReason := map[string]string{
				"missing":             "configuration copy provenance unavailable",
				"symlink":             "configuration source unavailable or indirect",
				"hardlink":            "copy aliases source",
				"different-bytes":     "not an identical native copy",
				"different-value":     "not an identical native copy",
				"wrong-origin":        "origin or effective entries differ from direct source",
				"wrong-common-origin": "new-worktree checkout policy differs from admitted copy proof",
				"effective-value":     "origin or effective entries differ from direct source",
				"inactive-extension":  "configuration copy provenance unavailable",
				"copy-error":          "injected config.worktree copy failure",
			}[kind]
			if kind == "copy-error" {
				wantCode = "PREPARE_NOT_READY"
			}
			if !injected || res.Code != wantCode || !strings.Contains(res.Reason, wantReason) {
				t.Fatalf("fault was not reached/refused for its intended reason: injected=%v result=%+v want=%q", injected, res, wantReason)
			}
			preservingWorktreeConfigAssertRefused(t, res, true, false, calls)
			preservingWorktreeConfigAssertSnapshot(t, before)
		})
	}
}

// These are parser unit cases, not evidence of native Git behavior.
// fak-test:runtime fast est=10ms lane=default
func TestPreservingWorktreeConfigEntriesExactEnvelope(t *testing.T) {
	valid := "core.bare\nfalse\x00core.hookspath\nsafe-hooks\x00"
	for _, input := range []string{valid, "core.hookspath\nsafe-hooks\x00core.bare\nfalse\x00"} {
		entries, err := preservingWorktreeConfigEntries(input)
		if err != nil || len(entries) != 2 || strings.Join(entries, "\x00")+"\x00" != input {
			t.Fatalf("exact envelope not retained: entries=%q err=%v", entries, err)
		}
	}
	for _, tc := range []struct {
		name, input string
	}{
		{"extra", valid + "user.name\nextra\x00"},
		{"include", "core.bare\nfalse\x00include.path\nempty.gitconfig\x00"},
		{"duplicate-bare", "core.bare\nfalse\x00core.bare\nfalse\x00"},
		{"duplicate-hooks", "core.hookspath\nsafe-hooks\x00core.hookspath\nsafe-hooks\x00"},
		{"duplicate-case", "core.bare\nfalse\x00Core.Bare\nfalse\x00"},
		{"true-bare", strings.Replace(valid, "false", "true", 1)},
		{"bare-alias", strings.Replace(valid, "false", "no", 1)},
		{"core-worktree", "core.worktree\ncheckout\x00core.hookspath\nsafe-hooks\x00"},
		{"missing", "core.bare\nfalse\x00"},
		{"unvalued", "core.bare\x00core.hookspath\nsafe-hooks\x00"},
		{"empty-hooks", "core.bare\nfalse\x00core.hookspath\n\x00"},
		{"unterminated", strings.TrimSuffix(valid, "\x00")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := preservingWorktreeConfigEntries(tc.input); err == nil {
				t.Fatalf("unapproved envelope accepted: %q", tc.input)
			}
		})
	}
}

// fak-test:runtime integration est=4s lane=default
func TestPreparePreservingNativeWorktreeConfigRechecksBothContexts(t *testing.T) {
	for _, phase := range []string{"source-before-add", "source-after-add", "source-before-publication", "child-before-publication"} {
		t.Run(phase, func(t *testing.T) {
			f := newPreservingWorktreeConfigFixture(t, false)
			before := preservingWorktreeConfigSnapshot(t, f)
			var calls []string
			git := preservingTestRunner(t, &calls)
			sourceReads, childReads := 0, 0
			injected := false
			runner := func(dir string, args []string) (int, string) {
				if preservingWorktreeConfigIsList(args) {
					isSource := samePath(dir, f.parent)
					if isSource {
						sourceReads++
					} else {
						childReads++
					}
					mutate := phase == "source-before-add" && isSource && sourceReads == 2 ||
						phase == "source-after-add" && isSource && sourceReads == 3 ||
						phase == "source-before-publication" && isSource && sourceReads == 4 ||
						phase == "child-before-publication" && !isSource && childReads == 2
					if mutate {
						path := preservingWorktreeConfigGitPath(t, dir, "config.worktree")
						changed := preservingWorktreeConfigRead(t, path) + "# drift at " + phase + "\n"
						if err := os.WriteFile(path, []byte(changed), 0600); err != nil {
							t.Fatal(err)
						}
						// Preserve the deliberately injected evidence as well.
						before[path] = changed
						injected = true
					}
				}
				return git(dir, args)
			}
			res := preservingWorktreeConfigPrepare(t, f, "config-drift", runner)
			wantReason := "source checkout policy drifted from admitted proof"
			if phase == "child-before-publication" {
				wantReason = "not an identical native copy"
			}
			if !injected || res.Code != "PRESERVATION_CONFIG_REFUSED" || !strings.Contains(res.Reason, wantReason) {
				t.Fatalf("drift not rechecked for its intended reason: injected=%v source=%d child=%d result=%+v want=%q", injected, sourceReads, childReads, res, wantReason)
			}
			preservingWorktreeConfigAssertRefused(t, res, phase != "source-before-add", strings.HasSuffix(phase, "before-publication"), calls)
			preservingWorktreeConfigAssertSnapshot(t, before)
		})
	}
}
