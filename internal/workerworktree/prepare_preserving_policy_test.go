package workerworktree

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type preservingPolicyFixture struct {
	root, config, base, version, values, tree, attributes string
	calls                                                 []string
}

func newPreservingPolicyFixture(t *testing.T) *preservingPolicyFixture {
	t.Helper()
	root := t.TempDir()
	t.Setenv("HOME", root)
	t.Setenv("XDG_CONFIG_HOME", filepath.Join(root, "xdg"))
	config := filepath.Join(root, "policy.gitconfig")
	if err := os.WriteFile(config, []byte("fixture policy\n"), 0600); err != nil {
		t.Fatal(err)
	}
	f := &preservingPolicyFixture{root: root, config: config, base: strings.Repeat("a", 40), version: "git version 2.45.0\n"}
	f.tree = "100644 blob " + f.base + "\t.gitattributes\x00" + "100644 blob " + f.base + "\ttarget.txt\x00"
	f.attributes = "unspecified"
	return f
}

func (f *preservingPolicyFixture) runner(root string, args []string) (int, string) {
	f.calls = append(f.calls, strings.Join(args, " "))
	switch args[0] {
	case "--version":
		return 0, f.version
	case "config":
		var data string
		origin := f.config
		if root == f.root {
			origin = filepath.Base(origin)
		}
		for _, value := range strings.Split(f.values, "\x00") {
			if value != "" {
				data += "file:" + origin + "\x00" + value + "\x00"
			}
		}
		return 0, data
	case "ls-tree":
		return 0, f.tree
	case "rev-parse":
		return 0, filepath.Join(root, "absent", filepath.Base(args[2])) + "\n"
	case "check-attr":
		if args[1] != "--source="+f.base || args[2] != "-z" || args[3] != "filter" || args[4] != "--" {
			return 128, "unbound query"
		}
		var data string
		for _, path := range args[5:] {
			data += path + "\x00filter\x00" + f.attributes + "\x00"
		}
		return 0, data
	}
	return 128, "unexpected executable operation"
}

func TestPreservingCheckoutPolicyQualifiesDisabledAndDormantSettings(t *testing.T) {
	for _, value := range []string{"false", "no", "off", "0", "FALSE"} {
		t.Run(value, func(t *testing.T) {
			f := newPreservingPolicyFixture(t)
			f.values = "core.fsmonitor\n" + value + "\x00submodule.active\n.\x00filter.lfs.clean\nnever-execute-clean\x00filter.lfs.smudge\nnever-execute-smudge\x00filter.lfs.process\nnever-execute-process\x00filter.lfs.required\ntrue"
			digest, err := preservingCheckoutPolicy(f.root, f.base, f.runner)
			if err != nil || len(digest) != 64 {
				t.Fatalf("digest=%q err=%v", digest, err)
			}
			for _, call := range f.calls {
				for _, forbidden := range []string{"worktree add", "status", "-c ", "never-execute"} {
					if strings.Contains(call, forbidden) {
						t.Fatalf("proof executed %s", call)
					}
				}
			}
		})
	}
}

func TestPreservingCheckoutPolicyRequiresCompletePinnedAttributes(t *testing.T) {
	for _, kind := range []string{"active", "unset", "empty", "partial", "duplicate", "reordered", "wrong-name", "warning", "invalid-utf8", "unterminated", "access"} {
		t.Run(kind, func(t *testing.T) {
			f := newPreservingPolicyFixture(t)
			f.values = "filter.lfs.smudge\nnever-execute"
			runner := func(root string, args []string) (int, string) {
				rc, data := f.runner(root, args)
				if args[0] != "check-attr" {
					return rc, data
				}
				switch kind {
				case "active":
					data = strings.ReplaceAll(data, "unspecified", "lfs")
				case "unset":
					data = strings.ReplaceAll(data, "unspecified", "unset")
				case "empty":
					data = ""
				case "partial":
					data = ".gitattributes\x00filter\x00unspecified\x00"
				case "duplicate":
					data = ".gitattributes\x00filter\x00unspecified\x00.gitattributes\x00filter\x00unspecified\x00"
				case "reordered":
					data = "target.txt\x00filter\x00unspecified\x00.gitattributes\x00filter\x00unspecified\x00"
				case "wrong-name":
					data = strings.ReplaceAll(data, "\x00filter\x00", "\x00diff\x00")
				case "warning":
					data += "warning: attribute source unavailable\n"
				case "invalid-utf8":
					data += string([]byte{255})
				case "unterminated":
					data = strings.TrimSuffix(data, "\x00")
				case "access":
					return 128, "permission denied"
				}
				return 0, data
			}
			if _, err := preservingCheckoutPolicy(f.root, f.base, runner); err == nil {
				t.Fatal("incomplete or active proof accepted")
			}
		})
	}
}

func TestPreservingCheckoutPolicyRefusesUnknownConditionalAndMalformedPolicy(t *testing.T) {
	for _, kind := range []string{"version", "version-warning", "fsmonitor", "bare-fsmonitor", "sparse", "conditional", "submodule", "filter-key", "filter-bool", "gitlink", "gitmodules", "duplicate-tree", "tree-warning", "config-warning", "config-access", "origin", "info-attributes", "global-attributes", "worktree-config", "hook", "tracked-hook", "hook-warning", "hook-escape"} {
		t.Run(kind, func(t *testing.T) {
			f := newPreservingPolicyFixture(t)
			f.values = "filter.lfs.smudge\nnever-execute\x00submodule.active\n."
			switch kind {
			case "version":
				f.version = "git version 2.46.0\n"
			case "version-warning":
				f.version += "warning\n"
			case "fsmonitor":
				f.values += "\x00core.fsmonitor\nactive-program"
			case "bare-fsmonitor":
				f.values += "\x00core.fsmonitor"
			case "sparse":
				f.values += "\x00core.sparsecheckout\nfalse"
			case "conditional":
				f.values += "\x00includeif.gitdir:/future/worker/.path\nnot-loaded-here"
			case "submodule":
				f.values += "\x00submodule.example.update\ncheckout"
			case "filter-key":
				f.values += "\x00filter.lfs.unknown\nvalue"
			case "filter-bool":
				f.values += "\x00filter.lfs.required\nmaybe"
			case "gitlink":
				f.tree += "160000 commit " + f.base + "\tchild\x00"
			case "gitmodules":
				f.tree += "100644 blob " + f.base + "\t.gitmodules\x00"
			case "duplicate-tree":
				f.tree += f.tree
			case "tree-warning":
				f.tree += "warning\n"
			case "worktree-config":
				f.values += "\x00extensions.worktreeconfig\ntrue"
			case "tracked-hook":
				f.tree += "100755 blob " + f.base + "\thooks/post-checkout\x00"
			}
			runner := func(root string, args []string) (int, string) {
				rc, data := f.runner(root, args)
				if args[0] == "config" {
					switch kind {
					case "config-warning":
						data += "warning\n"
					case "config-access":
						return 128, "access"
					case "origin":
						data = strings.ReplaceAll(data, "file:", "command line:")
					}
				}
				if args[0] == "rev-parse" {
					if kind == "hook-escape" && args[2] == "hooks/post-checkout" {
						return 0, "../outside/post-checkout\n"
					}
					if kind == "tracked-hook" && args[2] == "hooks/post-checkout" {
						return 0, "hooks/post-checkout\n"
					}
					if kind == "hook-warning" && args[2] == "hooks/post-checkout" {
						data += "warning\n"
					}
					if kind == "hook" && args[2] == "hooks/post-checkout" || kind == "info-attributes" && args[2] == "info/attributes" || kind == "worktree-config" && args[2] == "config.worktree" {
						path := strings.TrimSpace(data)
						if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
							t.Fatal(err)
						}
						if err := os.WriteFile(path, nil, 0600); err != nil {
							t.Fatal(err)
						}
					}
				}
				if kind == "global-attributes" {
					path := filepath.Join(os.Getenv("XDG_CONFIG_HOME"), "git", "attributes")
					if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(path, nil, 0600); err != nil {
						t.Fatal(err)
					}
				}
				return rc, data
			}
			if _, err := preservingCheckoutPolicy(f.root, f.base, runner); err == nil {
				t.Fatal("unqualified policy accepted")
			}
		})
	}
}

func TestPreservingCheckoutPolicyBindsSourcesAndNewContext(t *testing.T) {
	f := newPreservingPolicyFixture(t)
	f.values = "filter.lfs.process\nnever-execute\x00core.fsmonitor\nfalse\x00extensions.worktreeconfig\ntrue\x00include.path\nempty.gitconfig"
	first, err := preservingCheckoutPolicy(f.root, f.base, f.runner)
	if err != nil {
		t.Fatal(err)
	}
	other := filepath.Join(f.root, "new-context")
	second, err := preservingCheckoutPolicy(other, f.base, f.runner)
	if err != nil || first != second {
		t.Fatalf("context digest differs: %v", err)
	}
	if err := os.WriteFile(f.config, []byte("changed source comment\n"), 0600); err != nil {
		t.Fatal(err)
	}
	third, err := preservingCheckoutPolicy(f.root, f.base, f.runner)
	if err != nil || first == third {
		t.Fatalf("source drift not bound: %v", err)
	}
	if err := os.WriteFile(filepath.Join(f.root, "empty.gitconfig"), []byte("# now present\n"), 0600); err != nil {
		t.Fatal(err)
	}
	fourth, err := preservingCheckoutPolicy(other, f.base, f.runner)
	if err != nil || third == fourth {
		t.Fatalf("empty include source drift not bound: %v", err)
	}
	f.values += "\x00core.fsmonitor\ntrue"
	if _, err := preservingCheckoutPolicy(other, f.base, f.runner); err == nil {
		t.Fatal("new-context active monitor accepted")
	}
}

func TestPreservingCheckoutPolicyCoversEveryPathAcrossBatches(t *testing.T) {
	f := newPreservingPolicyFixture(t)
	f.values = "filter.lfs.smudge\nnever-execute"
	f.tree = ""
	for i := 0; i < 300; i++ {
		f.tree += fmt.Sprintf("100644 blob %s\tpath-%03d.txt\x00", f.base, i)
	}
	seen := map[string]int{}
	runner := func(root string, args []string) (int, string) {
		if args[0] == "check-attr" {
			for _, path := range args[5:] {
				seen[path]++
			}
		}
		return f.runner(root, args)
	}
	if _, err := preservingCheckoutPolicy(f.root, f.base, runner); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 300 {
		t.Fatalf("checked %d paths", len(seen))
	}
	for path, count := range seen {
		if count != 1 {
			t.Fatalf("%s checked %d times", path, count)
		}
	}
}

func TestPreservingGitOutputRefusesSuccessfulReadWarnings(t *testing.T) {
	for _, args := range [][]string{{"config", "--null", "--list"}, {"check-attr", "-z"}, {"ls-tree", "-z"}, {"rev-parse", "--git-path", "hooks/post-checkout"}, {"status", "--porcelain"}} {
		if code, data := preservingGitOutput(args, 0, "valid-output", "warning"); code == 0 || !strings.Contains(data, "warning") {
			t.Fatalf("warning hidden: %d %q", code, data)
		}
	}
	args := []string{"-c", "gc.worktreePruneExpire=never", "-c", "core.longpaths=true", "worktree", "add", "--detach", "target", "base"}
	if code, _ := preservingGitOutput(args, 0, "", "Preparing worktree (detached HEAD abcdef0)\n"); code != 0 {
		t.Fatal("ordinary add progress refused")
	}
	if code, _ := preservingGitOutput(args, 0, "", "Preparing worktree (detached HEAD abcdef0)\nwarning: changed config\n"); code == 0 {
		t.Fatal("successful add warning hidden")
	}
	if code, data := preservingGitOutput(args, 128, "partial", "failure"); code != 128 || data != "partialfailure" {
		t.Fatal("failed add evidence lost")
	}
}

func TestPreparePreservingRechecksPolicyBeforeAddAndStatus(t *testing.T) {
	for _, phase := range []string{"before-add", "new-context", "before-publication"} {
		t.Run(phase, func(t *testing.T) {
			f := newPreservingFixture(t)
			root := t.TempDir()
			var calls []string
			git := preservingTestRunner(t, &calls)
			configs := 0
			runner := func(dir string, args []string) (int, string) {
				rc, data := git(dir, args)
				if args[0] == "config" {
					configs++
					when := 2
					if phase == "new-context" {
						when = 3
					}
					if phase == "before-publication" {
						when = 4
					}
					if configs == when {
						return 128, "policy changed"
					}
				}
				return rc, data
			}
			res := preparePreserving(context.Background(), f.repo, "cmd", "policy-drift", f.base, root, OwnerStamp{PID: os.Getpid(), LeaseID: "fixture-admission"}, runner, func(context.Context) error { return nil })
			if res.OK || res.Code != "PRESERVATION_CONFIG_REFUSED" {
				t.Fatalf("drift=%+v", res)
			}
			added, status := false, false
			for _, call := range calls {
				added = added || strings.Contains(call, "worktree add")
				status = status || strings.HasPrefix(call, "status ")
			}
			if phase == "before-add" && added || phase == "new-context" && status {
				t.Fatalf("mutation/helper after refusal: %v", calls)
			}
			if phase != "before-add" {
				if _, err := os.Stat(res.Path); err != nil || !res.Preserved {
					t.Fatalf("partial evidence lost: %+v %v", res, err)
				}
			}
		})
	}
}

func TestPreparePreservingQualifiesDormantPolicyWithoutExecutingFilters(t *testing.T) {
	f := newPreservingFixture(t)
	for key, value := range map[string]string{"core.fsmonitor": "false", "submodule.active": ".", "filter.lfs.clean": "must-not-execute", "filter.lfs.smudge": "must-not-execute", "filter.lfs.process": "must-not-execute", "filter.lfs.required": "true"} {
		preservingFixtureGit(t, f.repo, "config", key, value)
	}
	var calls []string
	res := preparePreserving(context.Background(), f.repo, "cmd", "dormant-policy", f.base, t.TempDir(), OwnerStamp{PID: os.Getpid(), LeaseID: "fixture-admission"}, preservingTestRunner(t, &calls), func(context.Context) error { return nil })
	if !res.OK {
		t.Fatalf("disabled/dormant config refused: %+v", res)
	}
	assertPreservingTreeLive(t, f.repo, res.Path)
}

func TestPreparePreservingRefusesAttributeDriftBeforeAdd(t *testing.T) {
	f := newPreservingFixture(t)
	preservingFixtureGit(t, f.repo, "config", "filter.lfs.smudge", "must-not-execute")
	var calls []string
	git := preservingTestRunner(t, &calls)
	proofs := 0
	runner := func(dir string, args []string) (int, string) {
		rc, data := git(dir, args)
		if args[0] == "check-attr" {
			proofs++
			if proofs == 2 {
				return rc, strings.ReplaceAll(data, "\x00unspecified\x00", "\x00lfs\x00")
			}
		}
		return rc, data
	}
	res := preparePreserving(context.Background(), f.repo, "cmd", "attribute-drift", f.base, t.TempDir(), OwnerStamp{PID: os.Getpid(), LeaseID: "fixture-admission"}, runner, func(context.Context) error { return nil })
	if res.OK || res.Code != "PRESERVATION_CONFIG_REFUSED" || proofs != 2 {
		t.Fatalf("attribute drift=%+v proofs=%d", res, proofs)
	}
	for _, call := range calls {
		if strings.Contains(call, "worktree add") {
			t.Fatal("checkout after attribute proof changed")
		}
	}
}
