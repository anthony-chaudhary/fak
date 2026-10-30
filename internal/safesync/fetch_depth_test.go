package safesync

import (
	"context"
	"reflect"
	"strings"
	"testing"
)

func TestFetchDepthPinnedTransportAndRefusals(t *testing.T) {
	base, target := strings.Repeat("a", 40), strings.Repeat("b", 40)
	for _, fail := range []string{"", "pre-base", "pre-remote", "fetch", "tracking", "post-base", "post-remote"} {
		t.Run(fail, func(t *testing.T) {
			var calls [][]string
			bases, remotes, fetches := 0, 0, 0
			run := func(_ context.Context, _ string, args ...string) RunResult {
				calls = append(calls, append([]string(nil), args...))
				out := ""
				switch args[0] {
				case "check-ref-format":
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
				case "rev-parse":
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
				case "rev-parse":
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
