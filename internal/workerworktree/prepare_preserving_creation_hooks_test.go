package workerworktree

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// These native regression witnesses require the genuine qualified Git binary;
// neither its version output nor hook resolution is replaced. The refusal
// runner stops an unexpected add before Git can execute any fixture helper.
// Only the explicitly named NativeSentinelControls execute native hook aliases.
//
// Git v2.45.0 source for the relevant creation order:
// https://github.com/git/git/blob/v2.45.0/builtin/worktree.c#L477-L539
// https://github.com/git/git/blob/v2.45.0/refs.c#L2180-L2316
// https://github.com/git/git/blob/v2.45.0/read-cache.c#L2947-L2986
// https://github.com/git/git/blob/v2.45.0/hook.c#L110-L150
func preservingCreationHookNames() []string {
	return []string{"reference-transaction", "post-index-change", "post-checkout"}
}

func preservingCreationHookFixture(t *testing.T) reapProofFixture {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("native executable-mode and symlink witness requires POSIX")
	}
	if testing.Short() {
		t.Skip("native creation-hook integration witness")
	}
	return newQualifiedPreservingFixture(t)
}

const preservingCreationHookMagic = "fak-native-creation-hook-fixture-v1"

type preservingCreationHookDispatch struct {
	Magic  string
	Marker string
	Label  string
	Hook   string
}

// Native test-executable aliases dispatch before testing flag parsing. The
// adjacent fixture-only JSON file is pinned with the helper when necessary,
// so parent and target can have distinct labels without ambient environment
// injection, shell programs, go run, or nested compilation.
func init() {
	hook := filepath.Base(os.Args[0])
	if hook != "reference-transaction" && hook != "post-index-change" && hook != "post-checkout" {
		return
	}
	path, err := filepath.Abs(os.Args[0])
	if err != nil {
		preservingCreationHookFatal(err)
	}
	data, err := os.ReadFile(path + ".creation-hook.json")
	if err != nil {
		preservingCreationHookFatal(err)
	}
	var cfg preservingCreationHookDispatch
	if err := json.Unmarshal(data, &cfg); err != nil {
		preservingCreationHookFatal(err)
	}
	if cfg.Magic != preservingCreationHookMagic || cfg.Hook != hook || !filepath.IsAbs(cfg.Marker) || cfg.Label == "" {
		preservingCreationHookFatal(fmt.Errorf("invalid native creation-hook fixture identity"))
	}
	if hook == "reference-transaction" {
		if _, err := io.Copy(io.Discard, os.Stdin); err != nil {
			preservingCreationHookFatal(err)
		}
	}
	cwd, err := os.Getwd()
	if err != nil {
		preservingCreationHookFatal(err)
	}
	cwd, err = filepath.EvalSymlinks(cwd)
	if err != nil {
		preservingCreationHookFatal(err)
	}
	file, err := os.OpenFile(cfg.Marker, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		preservingCreationHookFatal(err)
	}
	if _, err := fmt.Fprintf(file, "%s\t%s\t%s\n", cfg.Label, cwd, strings.Join(os.Args[1:], " ")); err != nil {
		preservingCreationHookFatal(err)
	}
	if err := file.Close(); err != nil {
		preservingCreationHookFatal(err)
	}
	os.Exit(0)
}

func preservingCreationHookFatal(err error) {
	fmt.Fprintln(os.Stderr, "native creation-hook fixture:", err)
	os.Exit(97)
}

func preservingCreationHookExecutable(t *testing.T, path, marker, label, hook string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	source, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}
	if err := os.Link(source, path); err != nil {
		in, err := os.Open(source)
		if err != nil {
			t.Fatal(err)
		}
		defer in.Close()
		out, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o755)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := io.Copy(out, in); err != nil {
			_ = out.Close()
			t.Fatal(err)
		}
		if err := out.Close(); err != nil {
			t.Fatal(err)
		}
	}
	cfg := preservingCreationHookDispatch{Magic: preservingCreationHookMagic, Marker: marker, Label: label, Hook: hook}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path+".creation-hook.json", data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func preservingCreationHookConfigure(t *testing.T, repo, hooksPath string) {
	t.Helper()
	preservingFixtureGit(t, repo, "config", "--local", "extensions.worktreeConfig", "true")
	if hooksPath != "" {
		preservingFixtureGit(t, repo, "config", "--worktree", "core.bare", "false")
		preservingFixtureGit(t, repo, "config", "--worktree", "core.hooksPath", hooksPath)
	}
}

func preservingCreationHookCommit(t *testing.T, f *reapProofFixture, paths ...string) {
	t.Helper()
	staged := append([]string(nil), paths...)
	for _, path := range paths {
		if _, err := os.Stat(filepath.Join(f.repo, path) + ".creation-hook.json"); err == nil {
			staged = append(staged, path+".creation-hook.json")
		} else if !os.IsNotExist(err) {
			t.Fatal(err)
		}
	}
	preservingFixtureGit(t, f.repo, append([]string{"add", "--"}, staged...)...)
	preservingFixtureGit(t, f.repo, "commit", "-q", "-m", "isolated creation-hook witness")
	f.base = strings.TrimSpace(preservingFixtureGit(t, f.repo, "rev-parse", "HEAD"))
}

func preservingCreationHookAssertAbsent(t *testing.T, paths ...string) {
	t.Helper()
	for _, path := range paths {
		if _, err := os.Lstat(path); !os.IsNotExist(err) {
			t.Errorf("unexpected path or uninspectable absence %q: %v", path, err)
		}
	}
}

func preservingCreationHookConfigBytes(t *testing.T, repo string) map[string]string {
	t.Helper()
	out := map[string]string{}
	for _, name := range []string{"config", "config.worktree"} {
		path := strings.TrimSpace(preservingFixtureGit(t, repo, "rev-parse", "--git-path", name))
		if !filepath.IsAbs(path) {
			path = filepath.Join(repo, path)
		}
		data, err := os.ReadFile(path)
		if name == "config.worktree" && os.IsNotExist(err) {
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		out[path] = string(data)
	}
	return out
}

func preservingCreationHookAssertUnchanged(t *testing.T, before map[string]string) {
	t.Helper()
	for path, expected := range before {
		data, err := os.ReadFile(path)
		if err != nil || string(data) != expected {
			t.Errorf("fixture configuration changed: %q: %v", path, err)
		}
	}
}

func preservingCreationHookIsAdd(args []string) bool {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == "worktree" && args[i+1] == "add" {
			return true
		}
	}
	return false
}

// Prevent the failure under test from executing a helper. A production bug
// reaches this tripwire, producing a failed witness rather than an unsafe add.
func preservingCreationHookRequirePreAddRefusal(t *testing.T, f reapProofFixture, workers, key string, markers ...string) {
	t.Helper()
	target := Path("cmd", key, workers)
	preservingCreationHookAssertAbsent(t, append(markers, target)...)
	before := preservingCreationHookConfigBytes(t, f.repo)
	listing := preservingFixtureGit(t, f.repo, "worktree", "list", "--porcelain")
	var calls []string
	native := preservingTestRunner(t, &calls)
	attempts := 0
	runner := func(dir string, args []string) (int, string) {
		if preservingCreationHookIsAdd(args) {
			attempts++
			return 125, "test tripwire: unsafe creation hook reached worktree add"
		}
		if len(args) > 0 && args[0] == "status" {
			t.Error("status attempted before creation-hook refusal")
			return 125, "test tripwire: status is not part of pre-add proof"
		}
		return native(dir, args)
	}
	res := preparePreserving(context.Background(), f.repo, "cmd", key, f.base, workers,
		OwnerStamp{PID: os.Getpid(), LeaseID: "creation-hook-refusal"}, runner,
		func(context.Context) error { return nil })
	if res.OK || res.Code != "PRESERVATION_CONFIG_REFUSED" || !res.Preserved || attempts != 0 {
		t.Errorf("creation hook was not refused before add: result=%+v add attempts=%d calls=%v", res, attempts, calls)
	}
	if !strings.Contains(res.Reason, "creation hook") && !strings.Contains(res.Reason, "hook path") && !strings.Contains(res.Reason, "hooks path") {
		t.Errorf("refusal did not identify the hook boundary: %s", res.Reason)
	}
	preservingCreationHookAssertAbsent(t, append(markers, target)...)
	preservingCreationHookAssertAbsent(t, preservingSidecars(target)...)
	preservingCreationHookAssertUnchanged(t, before)
	if after := preservingFixtureGit(t, f.repo, "worktree", "list", "--porcelain"); after != listing {
		t.Errorf("pre-add refusal changed registrations:\nbefore=%s\nafter=%s", listing, after)
	}
}

// Empty hook locations are admissible. This positive witness prevents a blanket
// refusal of config.worktree/core.hooksPath from making all negatives pass.
// Runtime estimate is unmeasured; native witnesses are skipped under -short.
// fak-test:runtime integration est=5s lane=default
func TestPreparePreservingCreationHooksAllowsAbsentNativeLocations(t *testing.T) {
	for _, location := range []string{"default-administrative", "relative", "absolute"} {
		t.Run(location, func(t *testing.T) {
			f := preservingCreationHookFixture(t)
			hooksPath := "creation-hooks"
			if location == "default-administrative" {
				hooksPath = ""
			} else if location == "absolute" {
				hooksPath = filepath.Join(preservingTempDir(t), "absent-hooks")
			}
			preservingCreationHookConfigure(t, f.repo, hooksPath)
			before := preservingCreationHookConfigBytes(t, f.repo)
			var calls []string
			res := preparePreserving(context.Background(), f.repo, "cmd", "empty-creation-hooks", f.base, preservingTempDir(t),
				OwnerStamp{PID: os.Getpid(), LeaseID: "creation-hook-positive"}, preservingTestRunner(t, &calls),
				func(context.Context) error { return nil })
			if !res.OK {
				t.Fatalf("empty creation-hook locations refused: %+v", res)
			}
			assertPreservingTreeLive(t, f.repo, res.Path)
			if hooksPath == "" {
				info, err := os.Lstat(filepath.Join(res.Path, ".git"))
				if err != nil || !info.Mode().IsRegular() {
					t.Fatalf("default admin-hook witness requires child .git gitfile: %v", err)
				}
			} else if got := strings.TrimSpace(preservingFixtureGit(t, res.Path, "config", "--worktree", "--get", "core.hooksPath")); got != hooksPath {
				t.Fatalf("native hook policy changed: got %q want %q", got, hooksPath)
			}
			for _, call := range calls {
				if strings.Contains(call, "-c core.hooksPath") || strings.Contains(call, "-c core.hookspath") || strings.Contains(call, "--no-checkout") {
					t.Errorf("safe prepare bypassed native hook policy: %s", call)
				}
			}
			preservingCreationHookAssertUnchanged(t, before)
		})
	}
}

// Runtime estimate is unmeasured; native witnesses are skipped under -short.
// fak-test:runtime integration est=15s lane=default
func TestPreparePreservingCreationHooksRefusesParentExecutables(t *testing.T) {
	for _, hook := range preservingCreationHookNames() {
		for _, location := range []string{"default-administrative", "relative", "absolute", "relative-escape"} {
			t.Run(hook+"/"+location, func(t *testing.T) {
				f := preservingCreationHookFixture(t)
				marker := filepath.Join(preservingTempDir(t), "must-not-run")
				hooksPath := "creation-hooks"
				if location == "default-administrative" {
					hooksPath = ""
				} else if location == "absolute" {
					hooksPath = filepath.Join(preservingTempDir(t), "external-hooks")
				} else if location == "relative-escape" {
					outside := filepath.Join(preservingTempDir(t), "external-hooks")
					var err error
					hooksPath, err = filepath.Rel(f.repo, outside)
					if err != nil || !strings.HasPrefix(filepath.ToSlash(hooksPath), "../") {
						t.Fatalf("escaping fixture path: %q: %v", hooksPath, err)
					}
				}
				hookDir := hooksPath
				if location == "default-administrative" {
					hookDir = filepath.Join(f.repo, ".git", "hooks")
				} else if !filepath.IsAbs(hookDir) {
					hookDir = filepath.Join(f.repo, hookDir)
				}
				preservingCreationHookExecutable(t, filepath.Join(hookDir, hook), marker, "parent", hook)
				preservingCreationHookConfigure(t, f.repo, hooksPath)
				preservingCreationHookRequirePreAddRefusal(t, f, preservingTempDir(t), "parent-executable", marker)
			})
		}
	}
}

// Runtime estimate is unmeasured; native witnesses are skipped under -short.
// fak-test:runtime integration est=20s lane=default
func TestPreparePreservingCreationHooksRefusesPinnedTargetExecutables(t *testing.T) {
	for _, hook := range preservingCreationHookNames() {
		for _, location := range []string{"relative", "absolute-future-target"} {
			t.Run(hook+"/"+location, func(t *testing.T) {
				f := preservingCreationHookFixture(t)
				workers := preservingTempDir(t)
				key := "pinned-executable"
				target := Path("cmd", key, workers)
				marker := filepath.Join(preservingTempDir(t), "must-not-run")
				rel := filepath.Join("creation-hooks", hook)
				preservingCreationHookExecutable(t, filepath.Join(f.repo, rel), marker, "target", hook)
				preservingCreationHookCommit(t, &f, rel)
				if err := os.Remove(filepath.Join(f.repo, rel)); err != nil {
					t.Fatal(err)
				}
				hooksPath := "creation-hooks"
				if location == "absolute-future-target" {
					hooksPath = filepath.Join(target, hooksPath)
				}
				preservingCreationHookConfigure(t, f.repo, hooksPath)
				preservingCreationHookRequirePreAddRefusal(t, f, workers, key, marker)
			})
		}
	}
}

// Runtime estimate is unmeasured; native witnesses are skipped under -short.
// fak-test:runtime integration est=10s lane=default
func TestPreparePreservingCreationHooksRefusesPinnedSymlinkAncestors(t *testing.T) {
	for _, hook := range preservingCreationHookNames() {
		for _, location := range []string{"relative", "absolute-future-target"} {
			t.Run(hook+"/"+location, func(t *testing.T) {
				f := preservingCreationHookFixture(t)
				workers := preservingTempDir(t)
				key := "pinned-symlink"
				target := Path("cmd", key, workers)
				marker := filepath.Join(preservingTempDir(t), "must-not-run")
				outside := filepath.Join(workers, "creation-external-hooks")
				preservingCreationHookExecutable(t, filepath.Join(outside, hook), marker, "outside", hook)
				link := filepath.Join(f.repo, "creation-hooks")
				if err := os.Symlink("../creation-external-hooks", link); err != nil {
					t.Fatal(err)
				}
				preservingCreationHookCommit(t, &f, "creation-hooks")
				if err := os.Remove(link); err != nil {
					t.Fatal(err)
				}
				hooksPath := "creation-hooks"
				if location == "absolute-future-target" {
					hooksPath = filepath.Join(target, hooksPath)
				}
				preservingCreationHookConfigure(t, f.repo, hooksPath)
				preservingCreationHookRequirePreAddRefusal(t, f, workers, key, marker)
			})
		}
	}
}

// Runtime estimate is unmeasured; native witnesses are skipped under -short.
// fak-test:runtime integration est=3s lane=default
func TestPreparePreservingCreationHooksRefusesUnmaterializedEscapes(t *testing.T) {
	for _, kind := range []string{"lexical", "symlink-ancestor"} {
		t.Run(kind, func(t *testing.T) {
			f := preservingCreationHookFixture(t)
			outside := preservingTempDir(t)
			hooksPath := "creation-hooks"
			if kind == "lexical" {
				var err error
				hooksPath, err = filepath.Rel(f.repo, filepath.Join(outside, "absent-hooks"))
				if err != nil || !strings.HasPrefix(filepath.ToSlash(hooksPath), "../") {
					t.Fatalf("escaping fixture path: %q: %v", hooksPath, err)
				}
			} else if err := os.Symlink(outside, filepath.Join(f.repo, hooksPath)); err != nil {
				t.Fatal(err)
			}
			preservingCreationHookConfigure(t, f.repo, hooksPath)
			preservingCreationHookRequirePreAddRefusal(t, f, preservingTempDir(t), "empty-escape")
		})
	}
}

// Unicode paths are refused within the deliberately ASCII-qualified envelope.
// This does not pretend that a case-sensitive Linux fixture emulates APFS.
// Runtime estimate is unmeasured; native witnesses are skipped under -short.
// fak-test:runtime integration est=10s lane=default
func TestPreparePreservingCreationHooksRefusesUnicodeAliases(t *testing.T) {
	for _, spelling := range []struct {
		name, configured, pinned string
	}{
		{"nfc-to-nfd", "cr\u00e9ation-hooks", "cre\u0301ation-hooks"},
		{"nfd-to-nfc", "cre\u0301ation-hooks", "cr\u00e9ation-hooks"},
		{"absent-unicode", "cr\u00e9ation-hooks", ""},
	} {
		t.Run(spelling.name, func(t *testing.T) {
			f := preservingCreationHookFixture(t)
			marker := filepath.Join(preservingTempDir(t), "must-not-run")
			if spelling.pinned != "" {
				rel := filepath.Join(spelling.pinned, "post-index-change")
				preservingCreationHookExecutable(t, filepath.Join(f.repo, rel), marker, "unicode-target", "post-index-change")
				preservingCreationHookCommit(t, &f, rel)
				if err := os.Remove(filepath.Join(f.repo, rel)); err != nil {
					t.Fatal(err)
				}
			}
			preservingCreationHookConfigure(t, f.repo, spelling.configured)
			preservingCreationHookRequirePreAddRefusal(t, f, preservingTempDir(t), "unicode-alias", marker)
		})
	}
}

func preservingCreationHookRequireCaseInsensitiveFS(t *testing.T, root string) {
	t.Helper()
	upper := filepath.Join(root, "FaK-Creation-Hook-Case-Probe")
	lower := filepath.Join(root, "fak-creation-hook-case-probe")
	if err := os.WriteFile(upper, []byte("case probe"), 0o600); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(upper)
	a, err := os.Stat(upper)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.Stat(lower)
	if os.IsNotExist(err) {
		t.Skip("case-insensitive filesystem prerequisite for native spelling-alias witness")
	}
	if err != nil || !os.SameFile(a, b) {
		t.Fatalf("case-insensitive fixture identity unproven: %v", err)
	}
}

// Runtime estimate is unmeasured; native witnesses are skipped under -short.
// fak-test:runtime integration est=10s lane=default
func TestPreparePreservingCreationHooksRefusesPinnedCaseAliases(t *testing.T) {
	for _, hook := range preservingCreationHookNames() {
		t.Run(hook, func(t *testing.T) {
			f := preservingCreationHookFixture(t)
			workers := preservingTempDir(t)
			preservingCreationHookRequireCaseInsensitiveFS(t, f.repo)
			preservingCreationHookRequireCaseInsensitiveFS(t, workers)
			marker := filepath.Join(preservingTempDir(t), "must-not-run")
			rel := filepath.Join("CrEaTiOn-HoOkS", hook)
			preservingCreationHookExecutable(t, filepath.Join(f.repo, rel), marker, "case-alias", hook)
			preservingCreationHookCommit(t, &f, rel)
			if err := os.Remove(filepath.Join(f.repo, rel)); err != nil {
				t.Fatal(err)
			}
			preservingCreationHookConfigure(t, f.repo, "creation-hooks")
			preservingCreationHookRequirePreAddRefusal(t, f, workers, "case-alias", marker)
		})
	}
}

// These are deliberate positive controls in isolated temporary repositories.
// Each native helper only appends its label, cwd and args to a private sentinel.
// No hooks are disabled, and no real-user repository or configuration is used.
// Runtime estimate is unmeasured; native witnesses are skipped under -short.
// fak-test:runtime integration est=30s lane=default
func TestPreservingCreationHookNativeSentinelControls(t *testing.T) {
	for _, hook := range preservingCreationHookNames() {
		for _, location := range []string{"absolute-parent", "relative-both", "relative-target-only"} {
			t.Run(hook+"/"+location, func(t *testing.T) {
				f := preservingCreationHookFixture(t)
				workers := preservingTempDir(t)
				target := Path("cmd", "native-control", workers)
				marker := filepath.Join(preservingTempDir(t), "invocation-witness")
				hooksPath := "creation-hooks"
				if location == "absolute-parent" {
					hooksPath = filepath.Join(preservingTempDir(t), "control-hooks")
					preservingCreationHookExecutable(t, filepath.Join(hooksPath, hook), marker, "parent", hook)
				} else {
					rel := filepath.Join(hooksPath, hook)
					preservingCreationHookExecutable(t, filepath.Join(f.repo, rel), marker, "target", hook)
					preservingCreationHookCommit(t, &f, rel)
					if location == "relative-both" {
						preservingCreationHookExecutable(t, filepath.Join(f.repo, rel), marker, "parent", hook)
					} else if err := os.Remove(filepath.Join(f.repo, rel)); err != nil {
						t.Fatal(err)
					}
				}
				preservingCreationHookConfigure(t, f.repo, hooksPath)
				before := preservingCreationHookConfigBytes(t, f.repo)
				preservingCreationHookAssertAbsent(t, marker, target)
				preservingFixtureGit(t, f.repo, "-c", "gc.worktreePruneExpire=never", "-c", "core.longpaths=true", "worktree", "add", "--detach", target, f.base)
				assertPreservingTreeLive(t, f.repo, target)
				if head := strings.TrimSpace(preservingFixtureGit(t, target, "rev-parse", "HEAD")); head != f.base {
					t.Fatalf("native control HEAD=%q want pinned commit %q", head, f.base)
				}
				preservingCreationHookAssertUnchanged(t, before)
				if location != "absolute-parent" {
					info, err := os.Stat(filepath.Join(target, "creation-hooks", hook))
					if err != nil || info.Mode()&0o111 == 0 {
						t.Fatalf("control hook was not materialized executable: %v", err)
					}
				}
				// post-checkout selection belongs to the parent process. Its cwd
				// is target, but a target-only hook must not be selected by it.
				if hook == "post-checkout" && location == "relative-target-only" {
					preservingCreationHookAssertAbsent(t, marker)
					return
				}
				want := map[string]bool{}
				if location == "absolute-parent" {
					want["parent\t"+target] = true
					if hook == "reference-transaction" {
						want["parent\t"+f.repo] = true
					}
				} else {
					label := "target"
					if hook == "post-checkout" {
						label = "parent"
					}
					want[label+"\t"+target] = true
					if hook == "reference-transaction" && location == "relative-both" {
						want["parent\t"+f.repo] = true
					}
				}
				data, err := os.ReadFile(marker)
				if err != nil {
					t.Fatalf("native positive control did not execute its helper: %v", err)
				}
				seen := map[string]bool{}
				referenceStates := map[string]bool{}
				for _, line := range strings.Split(strings.TrimSuffix(string(data), "\n"), "\n") {
					fields := strings.Split(line, "\t")
					if len(fields) != 3 || !want[fields[0]+"\t"+fields[1]] {
						t.Fatalf("unexpected native hook selection/cwd: %q want=%v", line, want)
					}
					seen[fields[0]+"\t"+fields[1]] = true
					if hook == "reference-transaction" {
						// Git 2.45 aborts unnecessary packed-refs subtransactions.
						// Those notifications cannot replace successful updates.
						switch fields[2] {
						case "prepared", "committed", "aborted":
							referenceStates[fields[0]+"\t"+fields[1]+"\t"+fields[2]] = true
						default:
							t.Errorf("unexpected reference transaction state: %q", fields[2])
						}
					}
					if hook == "post-checkout" && fields[2] != strings.Repeat("0", 40)+" "+f.base+" 1" {
						t.Errorf("unexpected checkout arguments: %q", fields[2])
					}
				}
				for expected := range want {
					if !seen[expected] {
						t.Errorf("native control missed %q; events=%s", expected, data)
					}
					if hook == "reference-transaction" && (!referenceStates[expected+"\tprepared"] || !referenceStates[expected+"\tcommitted"]) {
						t.Errorf("native control lacks prepared/committed evidence for %q; events=%s", expected, data)
					}
				}
			})
		}
	}
}

// Runtime estimate is unmeasured; native witnesses are skipped under -short.
// fak-test:runtime integration est=5s lane=default
func TestPreparePreservingCreationHooksRechecksTargetBeforeStatus(t *testing.T) {
	for _, hook := range preservingCreationHookNames() {
		t.Run(hook, func(t *testing.T) {
			f := preservingCreationHookFixture(t)
			workers := preservingTempDir(t)
			key := "target-hook-drift"
			target := Path("cmd", key, workers)
			marker := filepath.Join(preservingTempDir(t), "must-not-run")
			preservingCreationHookConfigure(t, f.repo, "creation-hooks")
			before := preservingCreationHookConfigBytes(t, f.repo)
			var calls []string
			native := preservingTestRunner(t, &calls)
			added, status := false, false
			runner := func(dir string, args []string) (int, string) {
				if len(args) > 0 && args[0] == "status" {
					status = true
					return 125, "test tripwire: target policy must refuse before status"
				}
				rc, output := native(dir, args)
				if preservingCreationHookIsAdd(args) && rc == 0 {
					added = true
					preservingCreationHookExecutable(t, filepath.Join(target, "creation-hooks", hook), marker, "late-target", hook)
				}
				return rc, output
			}
			res := preparePreserving(context.Background(), f.repo, "cmd", key, f.base, workers,
				OwnerStamp{PID: os.Getpid(), LeaseID: "target-hook-drift"}, runner,
				func(context.Context) error { return nil })
			if !added || status || res.OK || res.Code != "PRESERVATION_CONFIG_REFUSED" || !res.Preserved {
				t.Fatalf("target hook requalification: result=%+v added=%v status=%v calls=%v", res, added, status, calls)
			}
			assertPreservingTreeLive(t, f.repo, target)
			preservingCreationHookAssertAbsent(t, marker, filepath.Join(target, WorkerLeaseFileName))
			preservingCreationHookAssertAbsent(t, preservingSidecars(target)...)
			preservingCreationHookAssertUnchanged(t, before)
		})
	}
}

// Runtime estimate is unmeasured; native witnesses are skipped under -short.
// fak-test:runtime integration est=5s lane=default
func TestPreparePreservingCreationHooksRechecksParentBeforeAdd(t *testing.T) {
	for _, hook := range preservingCreationHookNames() {
		t.Run(hook, func(t *testing.T) {
			f := preservingCreationHookFixture(t)
			workers := preservingTempDir(t)
			key := "parent-hook-drift"
			target := Path("cmd", key, workers)
			marker := filepath.Join(preservingTempDir(t), "must-not-run")
			preservingCreationHookConfigure(t, f.repo, "creation-hooks")
			before := preservingCreationHookConfigBytes(t, f.repo)
			var calls []string
			native := preservingTestRunner(t, &calls)
			attempts, checks := 0, 0
			runner := func(dir string, args []string) (int, string) {
				if preservingCreationHookIsAdd(args) {
					attempts++
					return 125, "test tripwire: late parent helper reached add"
				}
				return native(dir, args)
			}
			gate := func(context.Context) error {
				checks++
				if checks == 2 {
					preservingCreationHookExecutable(t, filepath.Join(f.repo, "creation-hooks", hook), marker, "late-parent", hook)
				}
				return nil
			}
			res := preparePreserving(context.Background(), f.repo, "cmd", key, f.base, workers,
				OwnerStamp{PID: os.Getpid(), LeaseID: "parent-hook-drift"}, runner, gate)
			if checks < 2 || attempts != 0 || res.OK || res.Code != "PRESERVATION_CONFIG_REFUSED" || !res.Preserved {
				t.Fatalf("late parent hook was not refused: result=%+v checks=%d adds=%d calls=%v", res, checks, attempts, calls)
			}
			preservingCreationHookAssertAbsent(t, marker, target)
			preservingCreationHookAssertAbsent(t, preservingSidecars(target)...)
			preservingCreationHookAssertUnchanged(t, before)
		})
	}
}
