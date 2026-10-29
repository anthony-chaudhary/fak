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
