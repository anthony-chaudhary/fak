package safecommit

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/wipref"
)

func TestPeerWIPAttributionBounded(t *testing.T) {
	const refs = 10000
	stamp, err := wipref.EncodeStamp(wipref.Stamp{SessionID: "scope", Scope: []string{"scope", "diff"}})
	if err != nil {
		t.Fatal(err)
	}
	selfStamp, err := wipref.EncodeStamp(wipref.Stamp{SessionID: "self", Scope: []string{"self"}})
	if err != nil {
		t.Fatal(err)
	}
	paths := []string{"scope/a", "diff/b", "self/c", "free/d"}
	status := " M scope/a\n M diff/b\n M self/c\n M free/d\n"
	for _, tc := range []struct {
		name, fail   string
		real, cancel bool
	}{
		{name: "many-refs"}, {name: "real-git", real: true},
		{name: "self-shared-object"}, {name: "metadata-record-truncated"}, {name: "delta-record-truncated"},
		{name: "root-failure", fail: "rev-list"},
		{name: "default-deadline"}, {name: "deadline-expiry"}, {name: "already-expired"},
		{name: "status-failure", fail: "status"}, {name: "metadata-failure", fail: "for-each-ref"},
		{name: "delta-failure", fail: "log"}, {name: "metadata-truncated"}, {name: "delta-truncated"},
		{name: "missing-object"}, {name: "callback-cancellation"}, {name: "cancellation", cancel: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			if tc.name == "default-deadline" {
				ctx = context.Background()
			}
			if tc.name == "deadline-expiry" {
				var stop context.CancelFunc
				ctx, stop = context.WithTimeout(ctx, 10*time.Millisecond)
				defer stop()
			}
			if tc.name == "already-expired" {
				var stop context.CancelFunc
				ctx, stop = context.WithDeadline(context.Background(), time.Now().Add(-1*time.Second))
				defer stop()
			}
			fixtureStarted := time.Now()
			dir := ""
			calls := 0
			var deadline time.Time
			run := Runner(func(ctx context.Context, dir string, args ...string) (string, int, error) {
				switch args[0] {
				case "status":
					return status, 0, nil
				case "for-each-ref":
					selfOID := refs + 1
					selfMessage := selfStamp
					if tc.name == "self-shared-object" {
						selfOID = 1
						selfMessage = "unstamped\n"
					}
					var out strings.Builder
					if args[2] == "--format=%(refname) %(objectname)" {
						for i := 0; i < refs; i++ {
							fmt.Fprintf(&out, "refs/fak/wip/peer-%05d %040x\n", i, i+1)
						}
						fmt.Fprintf(&out, "refs/fak/wip/self %040x\n", selfOID)
						return out.String(), 0, nil
					}
					for i := 0; i < refs; i++ {
						msg := "unstamped\n"
						if i == refs-2 {
							msg = "prose with embedded NUL: \x00\n" + stamp + "\n"
						}
						fmt.Fprintf(&out, "refs/fak/wip/peer-%05d\x00%040x\x00commit\x00%d\x00%s\x00\n", i, i+1, len(msg), msg)
					}
					fmt.Fprintf(&out, "refs/fak/wip/self\x00%040x\x00commit\x00%d\x00%s\x00\n", selfOID, len(selfMessage), selfMessage)
					return out.String(), 0, nil
				case "rev-list":
					return fmt.Sprintf("%040x\n", refs+2), 0, nil
				case "-c":
					var out strings.Builder
					ids := args[13:]
					for i, arg := range ids {
						if arg == "--always" {
							ids = ids[:i]
							break
						}
					}
					if len(ids) > 129 {
						t.Fatalf("unbounded argv: %d objects", len(ids))
					}
					for _, oid := range ids {
						n, err := strconv.ParseInt(strings.TrimLeft(oid, "0"), 16, 64)
						if err != nil || n == refs+1 {
							t.Fatalf("invalid or self-only object queried: %s", oid)
						}
						fmt.Fprintf(&out, "%s\x00", oid)
						if n == 1 {
							fmt.Fprintf(&out, "\n:100644 100644 %040x %040x M\x00diff/b\x00", 1, 2)
						}
					}
					return out.String(), 0, nil
				}
				t.Fatalf("unexpected git args: %v", args)
				return "", 1, nil
			})
			if tc.real {
				dir = t.TempDir()
				git := func(args ...string) string {
					t.Helper()
					out, code, err := realRunner(ctx, dir, args...)
					if err != nil || code != 0 {
						t.Fatalf("fixture %v: code=%d err=%v output=%s", args, code, err, out)
					}
					return strings.TrimSpace(out)
				}
				git("init")
				git("config", "user.name", "Fixture")
				git("config", "user.email", "fixture@example.invalid")
				for _, p := range paths {
					if err := os.MkdirAll(filepath.Dir(filepath.Join(dir, p)), 0700); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(filepath.Join(dir, p), []byte("base\n"), 0600); err != nil {
						t.Fatal(err)
					}
				}
				git("add", "--", "scope/a", "diff/b", "self/c", "free/d")
				tree := git("write-tree")
				parent := git("commit-tree", tree, "-m", "base")
				git("update-ref", "HEAD", parent)
				self := git("commit-tree", tree, "-p", parent, "-m", selfStamp)
				git("update-ref", "refs/fak/wip/self", self)
				// Distinct checkpoint objects form a real 10,000-commit history, each
				// carrying a delta. One fast-import process bounds construction overhead.
				var stream strings.Builder
				for i := 0; i < refs; i++ {
					msg := "unstamped"
					if i == refs-2 {
						msg = stamp
					}
					from := parent
					if i > 0 {
						from = fmt.Sprintf(":%d", i)
					}
					fmt.Fprintf(&stream, "commit refs/fak/wip/peer-%05d\nmark :%d\ncommitter Fixture <fixture@example.invalid> %d +0000\ndata %d\n%s\nfrom %s\n", i, i+1, 1700000000+i, len(msg), msg, from)
					body := strconv.Itoa(i)
					fmt.Fprintf(&stream, "M 100644 inline history/noise.txt\ndata %d\n%s\n", len(body), body)
					if i == 0 {
						stream.WriteString("M 100644 inline diff/b\ndata 10\ncheckpoint\n")
					}
					stream.WriteByte('\n')
				}
				stream.WriteString("done\n")
				importCmd := newGitCmd(ctx, dir, "fast-import", "--quiet", "--done")
				importCmd.Stdin = strings.NewReader(stream.String())
				if out, err := importCmd.CombinedOutput(); err != nil {
					t.Fatalf("fixture fast-import: %v: %s", err, out)
				}
				for _, p := range paths {
					if err := os.WriteFile(filepath.Join(dir, p), []byte("dirty\n"), 0600); err != nil {
						t.Fatal(err)
					}
				}
				run = realRunner
				cancel()
				ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
				defer cancel()
			}
			counted := func(ctx context.Context, dir string, args ...string) (string, int, error) {
				calls++
				if calls > 4+(refs+127)/128 {
					return "", 1, fmt.Errorf("attribution subprocess budget exceeded at %d", calls)
				}
				d, ok := ctx.Deadline()
				if !ok {
					t.Fatal("Runner did not receive attribution deadline")
				}
				if deadline.IsZero() {
					deadline = d
				} else if !deadline.Equal(d) {
					t.Fatal("deadline reset between subprocesses")
				}
				command := args[0]
				if command == "-c" {
					command = "log"
				}
				if tc.name == "deadline-expiry" && command == "log" {
					<-ctx.Done()
					return "", -1, nil
				}
				if tc.cancel && command == "log" {
					cancel()
					<-ctx.Done()
					return "", -1, nil
				}
				if command == tc.fail {
					return "lookup failed", 128, nil
				}
				out, code, err := run(ctx, dir, args...)
				if tc.name == "metadata-record-truncated" && command == "for-each-ref" && args[2] != "--format=%(refname) %(objectname)" {
					out = out[:strings.LastIndex(out, "refs/fak/wip/self")]
				}
				if tc.name == "delta-record-truncated" && command == "log" {
					out = out[:len(out)-41]
				}
				if tc.name == "metadata-truncated" && command == "for-each-ref" && args[2] != "--format=%(refname) %(objectname)" {
					out = out[:len(out)-1]
				}
				if tc.name == "delta-truncated" && command == "log" {
					out = out[:len(out)-1]
				}
				if tc.name == "missing-object" && command == "log" {
					out = ""
				}
				return out, code, err
			}
			opts := PathAttributionOptions{SessionID: "self"}
			if tc.name == "callback-cancellation" {
				opts.PeerWIPChecker = func(path string) (string, bool) {
					if path == "self/c" {
						cancel()
					}
					return "", false
				}
			}
			constructionElapsed := time.Since(fixtureStarted)
			attributionStarted := time.Now()
			result, err := ValidatePathAttribution(ctx, counted, dir, paths, opts)
			attributionElapsed := time.Since(attributionStarted)
			failure := tc.fail != "" || tc.cancel || strings.Contains(tc.name, "truncated") || tc.name == "missing-object" || tc.name == "deadline-expiry" || tc.name == "already-expired" || tc.name == "callback-cancellation"
			if failure {
				if err == nil || result.OK {
					t.Fatalf("incomplete attribution allowed: %+v, %v", result, err)
				}
				if (tc.cancel || tc.name == "callback-cancellation") && !errors.Is(err, context.Canceled) {
					t.Fatalf("lost cancellation: %v", err)
				}
				if (tc.name == "deadline-expiry" || tc.name == "already-expired") && !errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("lost deadline: %v", err)
				}
				if tc.name == "already-expired" && calls != 0 {
					t.Fatalf("subprocesses run on expired context: %d calls", calls)
				}
				if calls > 5 {
					t.Fatalf("continued after failure: %d calls", calls)
				}
				return
			}
			if err != nil || result.OK || result.Reason != ReasonPeerWIPCollision ||
				!reflect.DeepEqual(result.CollidingPaths, []string{"diff/b", "scope/a"}) ||
				!reflect.DeepEqual(result.PeerSessions, []string{"peer-00000", "peer-09998"}) {
				t.Fatalf("attribution: %+v err=%v", result, err)
			}
			bound := 4 + (refs+127)/128

			if calls > bound {
				t.Fatalf("subprocess amplification: got %d, bound %d", calls, bound)
			}
			if tc.real && attributionElapsed >= 30*time.Second {
				t.Fatalf("attribution exceeded 30s: %s", attributionElapsed)
			}
			t.Logf("refs=%d unique_peer_objects=%d paths=%d attribution_subprocesses=%d bound=%d construction=%s attribution=%s fixture_total=%s", refs+1, refs, len(paths), calls, bound, constructionElapsed, attributionElapsed, time.Since(fixtureStarted))
		})
	}
}

func TestPeerWIPLiteralFilterPreservesRealGitOwnership(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	dir := t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		out, code, err := realRunner(ctx, dir, args...)
		if err != nil || code != 0 {
			t.Fatalf("git %v: code=%d err=%v output=%s", args, code, err, out)
		}
		return strings.TrimSpace(out)
	}
	write := func(path, body string) {
		t.Helper()
		p := filepath.Join(dir, path)
		if err := os.MkdirAll(filepath.Dir(p), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	git("init")
	git("config", "user.name", "Fixture")
	git("config", "user.email", "fixture@example.invalid")
	for _, p := range []string{"literal/[x].go", "literal/x.go", "deleted.go", "rename-old.go", "scope/claimed.go", "free.go"} {
		write(p, "base\n")
	}
	git("add", "--", ".")
	baseTree := git("write-tree")
	base := git("commit-tree", baseTree, "-m", "base")
	git("update-ref", "HEAD", base)
	stamp, err := wipref.EncodeStamp(wipref.Stamp{SessionID: "a-scope", Scope: []string{"scope"}})
	if err != nil {
		t.Fatal(err)
	}
	scoped := git("commit-tree", baseTree, "-p", base, "-m", stamp)
	git("update-ref", "refs/fak/wip/a-scope", scoped)
	write("literal/[x].go", "owned\n")
	write("literal/x.go", "glob decoy\n")
	if err := os.Remove(filepath.Join(dir, "deleted.go")); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(filepath.Join(dir, "rename-old.go"), filepath.Join(dir, "rename-new.go")); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 128; i++ {
		write(fmt.Sprintf("noise/%03d.go", i), "unrelated\n")
	}
	git("add", "--", ".")
	tree := git("write-tree")
	delta := git("commit-tree", tree, "-p", base, "-m", "delta")
	git("update-ref", "refs/fak/wip/b-delta", delta)
	empty := git("commit-tree", tree, "-p", delta, "-m", "empty")
	git("update-ref", "refs/fak/wip/c-empty", empty)
	paths := []string{"literal/[x].go", "deleted.go", "rename-old.go", "rename-new.go", "scope/claimed.go", "free.go"}
	var filteredRaw string
	var fullBytes int
	runner := func(unfiltered bool) Runner {
		return func(ctx context.Context, dir string, args ...string) (string, int, error) {
			if unfiltered {
				for i, arg := range args {
					if arg == "--always" {
						args = args[:i]
						break
					}
				}
			}
			out, code, err := realRunner(ctx, dir, args...)
			if len(args) > 0 && args[0] == "-c" {
				if unfiltered {
					fullBytes += len(out)
				} else {
					filteredRaw += out
				}
			}
			return out, code, err
		}
	}
	oldOwners, err := resolveGitPeerOwners(ctx, runner(true), dir, paths, "self")
	if err != nil {
		t.Fatal(err)
	}
	owners, err := resolveGitPeerOwners(ctx, runner(false), dir, paths, "self")
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"literal/[x].go": "b-delta", "deleted.go": "b-delta", "rename-old.go": "b-delta", "rename-new.go": "b-delta", "scope/claimed.go": "a-scope"}
	if !reflect.DeepEqual(owners, oldOwners) || !reflect.DeepEqual(owners, want) {
		t.Fatalf("ownership changed: filtered=%v full=%v want=%v", owners, oldOwners, want)
	}
	if strings.Contains(filteredRaw, "noise/") || strings.Contains(filteredRaw, "literal/x.go") || len(filteredRaw)*8 >= fullBytes {
		t.Fatalf("unrelated diff bytes were not excluded: filtered=%d full=%d", len(filteredRaw), fullBytes)
	}
	for _, oid := range []string{scoped, delta, empty, base} {
		if !strings.Contains(filteredRaw, oid+"\x00") {
			t.Fatalf("missing checkpoint or terminator %s", oid)
		}
	}
	none, err := resolveGitPeerOwners(ctx, realRunner, dir, []string{"free.go"}, "self")
	if err != nil || len(none) != 0 {
		t.Fatalf("empty intersection: owners=%v err=%v", none, err)
	}
	t.Logf("full_diff_bytes=%d literal_filtered_bytes=%d identical_owners=%d", fullBytes, len(filteredRaw), len(owners))
}

func TestPeerWIPContextTimeout(t *testing.T) {
	t.Run("ExpiredContextYieldsZeroSubprocesses", func(t *testing.T) {
		ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-1*time.Second))
		defer cancel()
		calls := 0
		run := Runner(func(ctx context.Context, dir string, args ...string) (string, int, error) {
			calls++
			return "", 0, nil
		})
		res, err := ValidatePathAttribution(ctx, run, "", []string{"internal/gateway"}, PathAttributionOptions{})
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("expected context.DeadlineExceeded, got %v", err)
		}
		if res.OK {
			t.Fatalf("expected res.OK to be false, got true")
		}
		if calls != 0 {
			t.Fatalf("expected 0 subprocess calls on expired context, got %d", calls)
		}
	})

	t.Run("CanceledContextYieldsZeroSubprocesses", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		calls := 0
		run := Runner(func(ctx context.Context, dir string, args ...string) (string, int, error) {
			calls++
			return "", 0, nil
		})
		res, err := ValidatePathAttribution(ctx, run, "", []string{"internal/gateway"}, PathAttributionOptions{})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected context.Canceled, got %v", err)
		}
		if res.OK {
			t.Fatalf("expected res.OK to be false, got true")
		}
		if calls != 0 {
			t.Fatalf("expected 0 subprocess calls on canceled context, got %d", calls)
		}
	})

	t.Run("ShortTimeoutAbortsWithoutSubprocessLeakage", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
		defer cancel()
		calls := 0
		run := Runner(func(ctx context.Context, dir string, args ...string) (string, int, error) {
			calls++
			<-ctx.Done()
			return "", -1, ctx.Err()
		})
		res, err := ValidatePathAttribution(ctx, run, "", []string{"internal/gateway"}, PathAttributionOptions{})
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("expected context.DeadlineExceeded, got %v", err)
		}
		if res.OK {
			t.Fatalf("expected res.OK to be false, got true")
		}
		if calls > 1 {
			t.Fatalf("expected at most 1 subprocess call before deadline abort, got %d", calls)
		}
	})

	t.Run("UnboundedContextReceivesDefensiveTimeout", func(t *testing.T) {
		ctx := context.Background()
		calls := 0
		var observedDeadline time.Time
		run := Runner(func(ctx context.Context, dir string, args ...string) (string, int, error) {
			calls++
			d, ok := ctx.Deadline()
			if !ok {
				t.Fatal("expected runner to receive bounded context deadline")
			}
			observedDeadline = d
			return "", 0, nil
		})
		res, err := ValidatePathAttribution(ctx, run, "", []string{"internal/gateway"}, PathAttributionOptions{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !res.OK {
			t.Fatalf("expected res.OK, got %+v", res)
		}
		if calls != 1 {
			t.Fatalf("expected 1 status call, got %d", calls)
		}
		if observedDeadline.IsZero() {
			t.Fatal("no deadline was set on context")
		}
		remaining := time.Until(observedDeadline)
		if remaining <= 0 || remaining > peerWIPAttributionTimeout {
			t.Fatalf("expected deadline within (0, %v], got %v", peerWIPAttributionTimeout, remaining)
		}
	})

	t.Run("SubprocessesShareExactSameDeadline", func(t *testing.T) {
		ctx := context.Background()
		var deadlines []time.Time
		run := Runner(func(ctx context.Context, dir string, args ...string) (string, int, error) {
			d, ok := ctx.Deadline()
			if !ok {
				t.Fatal("expected runner to receive bounded context deadline")
			}
			deadlines = append(deadlines, d)
			switch args[0] {
			case "status":
				return " M internal/gateway/foo.go\n", 0, nil
			case "for-each-ref":
				return "", 0, nil
			default:
				return "", 0, nil
			}
		})
		_, err := ValidatePathAttribution(ctx, run, "", []string{"internal/gateway"}, PathAttributionOptions{SessionID: "self"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(deadlines) < 2 {
			t.Fatalf("expected at least 2 subprocess calls, got %d", len(deadlines))
		}
		for i := 1; i < len(deadlines); i++ {
			if !deadlines[i].Equal(deadlines[0]) {
				t.Fatalf("deadline reset between subprocess %d and 0: %v vs %v", i, deadlines[i], deadlines[0])
			}
		}
	})
}
