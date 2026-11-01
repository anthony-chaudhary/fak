package safesync

import (
	"context"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestFetchDepthPinnedTransportAndRefusals(t *testing.T) {
	base, target := strings.Repeat("a", 40), strings.Repeat("b", 40)
	head, parent := strings.Repeat("c", 40), strings.Repeat("d", 40)
	for _, fail := range []string{"", "pre-base", "pre-remote", "pre-head-error", "pre-head-race", "pre-graph-error", "pre-graph-warning", "fetch", "tracking", "post-base", "post-remote", "post-head", "post-head-error", "post-graph-error", "lost-node", "lost-edge"} {
		t.Run(fail, func(t *testing.T) {
			var calls [][]string
			bases, remotes, fetches, heads, graphs := 0, 0, 0, 0, 0
			run := func(_ context.Context, _ string, args ...string) RunResult {
				calls = append(calls, append([]string(nil), args...))
				out := ""
				switch args[0] {
				case "check-ref-format":
				case "for-each-ref":
					out = "refs/remotes/origin/main\n"
				case "merge-base":
					bases++
					out = base
					if (bases == 1 && fail == "pre-base") || (bases == 2 && fail == "post-base") {
						return RunResult{Code: 1}
					}
				case "ls-remote":
					remotes++
					if remotes == 2 && fail == "post-remote-error" {
						return RunResult{Code: 1}
					}
					out = target + "\trefs/heads/main\n"
					if (remotes == 1 && fail == "pre-remote") || (remotes == 2 && fail == "post-remote") {
						out = base + "\trefs/heads/main\n"
					}
				case "fetch":
					fetches++
					want := []string{"fetch", "--no-tags", "--no-recurse-submodules", "--depth=30", "--", "origin", target + ":refs/remotes/origin/main"}
					if !reflect.DeepEqual(args, want) {
						t.Fatalf("transport args %q, want %q", args, want)
					}
					if fail == "fetch" {
						return RunResult{Code: 1}
					}
				case "rev-list":
					if !reflect.DeepEqual(args, []string{"rev-list", "--parents", "--all", "HEAD"}) {
						t.Fatalf("incomplete graph scope %q", args)
					}
					graphs++
					if (graphs == 1 && fail == "pre-graph-error") || (graphs == 2 && fail == "post-graph-error") {
						return RunResult{Code: 1}
					}
					if graphs == 1 && fail == "pre-graph-warning" {
						return RunResult{Stderr: []byte("warning: graph incomplete")}
					}
					out = head + " " + base + "\n" + base + " " + parent + "\n" + parent + "\n"
					if graphs == 2 {
						out += target + " " + base + "\n"
						if fail == "lost-node" {
							out = head + " " + base + "\n" + base + "\n" + target + " " + base + "\n"
						}
						if fail == "lost-edge" {
							out = head + " " + base + "\n" + base + "\n" + parent + "\n" + target + " " + base + "\n"
						}
					}
				case "rev-parse":
					if args[len(args)-1] == "HEAD" {
						heads++
						if fail == "pre-head-error" || (fetches > 0 && fail == "post-head-error") {
							return RunResult{Code: 1}
						}
						out = head
						if (heads == 2 && fail == "pre-head-race") || (fetches > 0 && fail == "post-head") {
							out = target
						}
						break
					}
					if fail == "tracking-error" {
						return RunResult{Code: 1}
					}
					out = target
					if fail == "tracking" {
						out = base
					}
				default:
					t.Fatalf("unexpected command %q", args)
				}
				return RunResult{Stdout: []byte(out)}
			}
			err := fetchWithDepth(context.Background(), run, ".", "origin", "main", 30, base, target)
			if (err != nil) != (fail != "") {
				t.Fatalf("failure=%q err=%v calls=%q", fail, err, calls)
			}
			wantFetches := 1
			if strings.HasPrefix(fail, "pre-") {
				wantFetches = 0
			}
			if fetches != wantFetches {
				t.Fatalf("fetches=%d want %d; no fallback allowed", fetches, wantFetches)
			}
			if err != nil && wantFetches == 1 && fail != "fetch" &&
				!strings.Contains(err.Error(), "fetch completed; postcondition failed") {
				t.Fatalf("completed transport effects not distinguished: %v", err)
			}
			if fail == "" && (graphs != 2 || heads != 3) {
				t.Fatalf("missing existing-tracking preservation proof: graphs=%d heads=%d", graphs, heads)
			}
		})
	}
}

func TestFetchDepthDefaultAndInvalidPins(t *testing.T) {
	var calls [][]string
	run := func(_ context.Context, _ string, args ...string) RunResult {
		calls = append(calls, args)
		return RunResult{}
	}
	if err := fetchWithDepth(context.Background(), run, ".", "origin", "main", 0, "", ""); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(calls, [][]string{{"fetch", "origin", "main"}}) {
		t.Fatalf("default changed: %q", calls)
	}
	calls = nil
	if err := fetchWithDepth(context.Background(), run, ".", "origin", "main", 30, "bad", strings.Repeat("b", 40)); err == nil || len(calls) != 0 {
		t.Fatalf("invalid pin reached transport: %v %q", err, calls)
	}
}

// An already-published HEAD may become the unique merge-base after updating a
// stale tracking ref. This is allowed only with the explicit post-base pin.
func TestFetchDepthExplicitPostBaseAdvancement(t *testing.T) {
	base, target, after := strings.Repeat("a", 40), strings.Repeat("b", 40), strings.Repeat("c", 40)
	for _, fail := range []string{"", "pre-base", "base-ancestor", "head-ancestor", "pre-remote", "fetch", "tracking", "tracking-error", "post-base", "post-base-error", "ambiguous-post-base", "post-remote", "post-remote-error", "default"} {
		t.Run(fail, func(t *testing.T) {
			bases, remotes, fetches := 0, 0, 0
			var ancestry [][]string
			run := func(_ context.Context, _ string, args ...string) RunResult {
				out := ""
				switch args[0] {
				case "check-ref-format":
				case "for-each-ref":
					out = "refs/remotes/origin/main\n"
				case "merge-base":
					if args[1] == "--is-ancestor" {
						ancestry = append(ancestry, append([]string(nil), args...))
						if (args[2] == base && fail == "base-ancestor") || (args[3] == "HEAD" && fail == "head-ancestor") {
							return RunResult{Code: 1}
						}
					} else {
						bases++
						out = base
						if bases == 1 && fail == "pre-base" {
							return RunResult{Code: 1}
						}
						if bases == 2 {
							if fail == "post-base-error" {
								return RunResult{Code: 1}
							}
							out = after
							if fail == "ambiguous-post-base" {
								out = after + "\n" + base
							}
							if fail == "post-base" {
								out = base
							}
						}
					}
				case "ls-remote":
					remotes++
					if remotes == 2 && fail == "post-remote-error" {
						return RunResult{Code: 1}
					}
					out = target + "\trefs/heads/main\n"
					if (remotes == 1 && fail == "pre-remote") || (remotes == 2 && fail == "post-remote") {
						out = base + "\trefs/heads/main\n"
					}
				case "fetch":
					fetches++
					if fail == "fetch" {
						return RunResult{Code: 1}
					}
				case "rev-list":
					out = after + " " + base + "\n" + base + "\n"
					if fetches > 0 {
						out += target + " " + after + "\n"
					}
				case "rev-parse":
					if args[len(args)-1] == "HEAD" {
						out = after
						break
					}
					if fail == "tracking-error" {
						return RunResult{Code: 1}
					}
					out = target
					if fail == "tracking" {
						out = base
					}
				default:
					t.Fatalf("unexpected command %q", args)
				}
				return RunResult{Stdout: []byte(out)}
			}
			pin := after
			if fail == "default" {
				pin = ""
			}
			err := fetchWithExpectedBase(context.Background(), run, ".", "origin", "main", 55, base, target, pin)
			if (err != nil) != (fail != "") {
				t.Fatalf("failure=%q err=%v", fail, err)
			}
			wantFetch := 1
			if fail == "pre-base" || fail == "base-ancestor" || fail == "head-ancestor" || fail == "pre-remote" {
				wantFetch = 0
			}
			if fetches != wantFetch {
				t.Fatalf("fetches=%d want%d; no retry/fallback", fetches, wantFetch)
			}
			if fail == "fetch" {
				if !strings.Contains(err.Error(), "effects may be partial; reconcile before retry") || strings.Contains(err.Error(), "fetch completed") {
					t.Fatalf("transport effect warning missing or false: %v", err)
				}
			} else if fail != "" && wantFetch == 1 {
				if !strings.Contains(err.Error(), "fetch completed; postcondition failed") || !strings.Contains(err.Error(), "effects require reconciliation; do not retry blindly") {
					t.Fatalf("completed-fetch effect warning missing: %v", err)
				}
			} else if err != nil && strings.Contains(err.Error(), "fetch completed") {
				t.Fatalf("preflight refusal falsely claims fetch effects: %v", err)
			}
			if fail == "" {
				want := [][]string{{"merge-base", "--is-ancestor", base, after}, {"merge-base", "--is-ancestor", after, "HEAD"}}
				if !reflect.DeepEqual(ancestry, want) {
					t.Fatalf("ancestry checks=%q want%q", ancestry, want)
				}
			}
		})
	}
}

func TestFetchDepthInvalidPostBaseRefusesBeforeTransport(t *testing.T) {
	calls := 0
	run := func(_ context.Context, _ string, _ ...string) RunResult { calls++; return RunResult{} }
	base, target := strings.Repeat("a", 40), strings.Repeat("b", 40)
	for _, after := range []string{"short", strings.Repeat("C", 40)} {
		if err := fetchWithExpectedBase(context.Background(), run, ".", "origin", "main", 55, base, target, after); err == nil {
			t.Fatal("invalid post-base accepted")
		}
	}
	if calls != 0 {
		t.Fatalf("invalid post-base invoked %d commands", calls)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestFetchDepthMissingTrackingBootstrapPreservesVisibleGraph(t *testing.T) {
	base, target := strings.Repeat("a", 40), strings.Repeat("b", 40)
	side, parent, older := strings.Repeat("c", 40), strings.Repeat("d", 40), strings.Repeat("e", 40)
	tracking := "refs/remotes/origin/main"
	for _, fail := range []string{
		"", "missing-base", "missing-target", "missing-depth", "ref-error", "ref-start-error", "ref-warning", "ref-present", "ref-descendant", "ref-race",
		"head-error", "wrong-base", "head-race", "pre-remote", "pre-graph-error", "pre-graph-empty", "pre-graph-invalid", "pre-graph-duplicate", "pre-graph-missing-head", "pre-graph-missing-parent",
		"fetch", "tracking", "post-base", "post-remote", "post-head", "post-graph-error", "lost-node", "lost-edge",
	} {
		t.Run(fail, func(t *testing.T) {
			fetches, bases, refs, heads, remotes, graphs := 0, 0, 0, 0, 0, 0
			run := func(_ context.Context, _ string, args ...string) RunResult {
				out := ""
				switch args[0] {
				case "check-ref-format":
				case "merge-base":
					if !reflect.DeepEqual(args, []string{"merge-base", "--all", "HEAD", tracking}) {
						t.Fatalf("unexpected ancestry probe %q", args)
					}
					bases++
					if bases == 1 || fail == "post-base" {
						return RunResult{Code: 128}
					}
					out = base
				case "for-each-ref":
					if !reflect.DeepEqual(args, []string{"for-each-ref", "--format=%(refname)", "--", tracking}) {
						t.Fatalf("unexpected absence probe %q", args)
					}
					refs++
					switch fail {
					case "ref-error":
						return RunResult{Code: 1}
					case "ref-start-error":
						return RunResult{Err: context.Canceled}
					case "ref-warning":
						return RunResult{Stderr: []byte("warning: broken ref")}
					case "ref-present":
						out = tracking
					case "ref-descendant":
						out = tracking + "/child"
					case "ref-race":
						if refs == 2 {
							out = tracking
						}
					}
				case "rev-parse":
					switch args[len(args)-1] {
					case "HEAD":
						heads++
						if fail == "head-error" {
							return RunResult{Code: 1}
						}
						out = base
						if fail == "wrong-base" || (fail == "head-race" && heads == 2) || (fail == "post-head" && fetches > 0) {
							out = side
						}
					case tracking:
						out = target
						if fail == "tracking" {
							out = side
						}
					default:
						t.Fatalf("unexpected revision probe %q", args)
					}
				case "ls-remote":
					remotes++
					out = target + "\trefs/heads/main\n"
					if fail == "pre-remote" || (fail == "post-remote" && remotes == 2) {
						out = side + "\trefs/heads/main\n"
					}
				case "rev-list":
					if !reflect.DeepEqual(args, []string{"rev-list", "--parents", "--all", "HEAD"}) {
						t.Fatalf("incomplete graph scope %q", args)
					}
					graphs++
					out = base + "\n" + side + " " + parent + "\n" + parent + "\n"
					if graphs == 1 {
						switch fail {
						case "pre-graph-error":
							return RunResult{Code: 1}
						case "pre-graph-empty":
							out = ""
						case "pre-graph-invalid":
							out = base + " short\n"
						case "pre-graph-duplicate":
							out += base + "\n"
						case "pre-graph-missing-head":
							out = side + "\n"
						case "pre-graph-missing-parent":
							out = base + " " + parent + "\n"
						}
					} else {
						out = target + " " + base + "\n" + base + " " + older + "\n" + older + "\n" + side + " " + parent + "\n" + parent + "\n"
						switch fail {
						case "post-graph-error":
							return RunResult{Code: 1}
						case "lost-node":
							out = target + " " + base + "\n" + base + "\n"
						case "lost-edge":
							out = target + " " + base + "\n" + base + "\n" + side + "\n" + parent + "\n"
						}
					}
				case "fetch":
					fetches++
					want := []string{"fetch", "--no-tags", "--no-recurse-submodules", "--depth=30", "--", "origin", target + ":" + tracking}
					if !reflect.DeepEqual(args, want) {
						t.Fatalf("transport args %q, want %q", args, want)
					}
					if fail == "fetch" {
						return RunResult{Code: 1}
					}
				default:
					t.Fatalf("unexpected command %q", args)
				}
				return RunResult{Stdout: []byte(out)}
			}
			depth, basePin, targetPin := 30, base, target
			switch fail {
			case "missing-base":
				basePin = ""
			case "missing-target":
				targetPin = ""
			case "missing-depth":
				depth = 0
			}
			err := fetchWithDepth(context.Background(), run, ".", "origin", "main", depth, basePin, targetPin)
			wantFetch := fail == "" || fail == "fetch" || fail == "tracking" || strings.HasPrefix(fail, "post-") || strings.HasPrefix(fail, "lost-")
			if (err != nil) != (fail != "") || fetches > 1 || (fetches == 1) != wantFetch {
				t.Fatalf("failure=%q err=%v fetches=%d wantFetch=%v", fail, err, fetches, wantFetch)
			}
			if fail == "" && (refs != 2 || heads != 3 || graphs != 2 || remotes != 2 || bases != 2) {
				t.Fatalf("missing proof: refs=%d heads=%d graphs=%d remotes=%d bases=%d", refs, heads, graphs, remotes, bases)
			}
			if fail == "fetch" {
				if !strings.Contains(err.Error(), "effects may be partial; reconcile before retry") || strings.Contains(err.Error(), "fetch completed") {
					t.Fatalf("transport effects not distinguished: %v", err)
				}
			} else if err != nil && (strings.Contains(err.Error(), "fetch completed; postcondition failed") != wantFetch) {
				t.Fatalf("fetch effects misreported: %v", err)
			}
		})
	}
}

// An unchanged merge-base is insufficient: depth two cuts the old base's
// parent edge even though both local and remote tips still reach that base.
// fak-test:runtime fast est=2s lane=default
func TestFetchDepthExistingTrackingRealGitHistory(t *testing.T) {
	for _, tc := range []struct {
		name     string
		depth    int
		wantLoss bool
	}{
		{"cuts-old-parent", 2, true},
		{"preserves-old-parent", 3, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			origin := t.TempDir()
			git(t, origin, "init", "-b", "main")
			git(t, origin, "config", "user.name", "test")
			git(t, origin, "config", "user.email", "test@example.com")
			writeFile(t, filepath.Join(origin, "history.txt"), "old root\n")
			git(t, origin, "add", "history.txt")
			git(t, origin, "commit", "-m", "old root")
			older := revString(t, origin, "HEAD")
			writeFile(t, filepath.Join(origin, "history.txt"), "common base\n")
			git(t, origin, "commit", "-am", "common base")
			base := revString(t, origin, "HEAD")
			cloneRoot := t.TempDir()
			clone := filepath.Join(cloneRoot, "clone")
			git(t, cloneRoot, "clone", "--no-local", "--branch", "main", origin, clone)
			git(t, clone, "config", "user.name", "test")
			git(t, clone, "config", "user.email", "test@example.com")
			writeFile(t, filepath.Join(clone, "local.txt"), "local branch\n")
			git(t, clone, "add", "local.txt")
			git(t, clone, "commit", "-m", "local branch")
			head := revString(t, clone, "HEAD")
			writeFile(t, filepath.Join(origin, "remote.txt"), "remote advance\n")
			git(t, origin, "add", "remote.txt")
			git(t, origin, "commit", "-m", "remote advance")
			target := revString(t, origin, "HEAD")
			if got := strings.TrimSpace(gitOutput(t, clone, "merge-base", "--all", "HEAD", "origin/main")); got != base {
				t.Fatalf("fixture pre-base=%s want %s", got, base)
			}
			before := gitOutput(t, clone, "rev-list", "--parents", "--all", "HEAD")
			if !strings.Contains(before, base+" "+older+"\n") {
				t.Fatalf("fixture old edge absent: %s", before)
			}
			fetches := 0
			runner := func(ctx context.Context, repo string, args ...string) RunResult {
				if len(args) > 0 && args[0] == "fetch" {
					fetches++
				}
				return RealRunner(ctx, repo, args...)
			}
			err := fetchWithDepth(context.Background(), runner, clone, "origin", "main", tc.depth, base, target)
			if fetches != 1 || revString(t, clone, "HEAD") != head || revString(t, clone, "origin/main") != target {
				t.Fatalf("actual transport/readback changed: fetches=%d err=%v", fetches, err)
			}
			if got := strings.TrimSpace(gitOutput(t, clone, "merge-base", "--all", "HEAD", "origin/main")); got != base {
				t.Fatalf("post-base=%s want unchanged %s", got, base)
			}
			after := gitOutput(t, clone, "rev-list", "--parents", "--all", "HEAD")
			lost := !strings.Contains(after, base+" "+older+"\n")
			if lost != tc.wantLoss || (err != nil) != tc.wantLoss {
				t.Fatalf("loss=%v want=%v err=%v graph=%s", lost, tc.wantLoss, err, after)
			}
			if tc.wantLoss && (!strings.Contains(err.Error(), "fetch completed; postcondition failed") ||
				!strings.Contains(err.Error(), "effects require reconciliation; do not retry blindly")) {
				t.Fatalf("partial-effect recovery contract missing: %v", err)
			}
			// The failure deliberately leaves the actual fetch effects visible.
			// No rollback, deeper fetch, retry or destructive shallow-file edit.
		})
	}
}
