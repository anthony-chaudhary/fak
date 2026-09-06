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
		{name: "default-deadline"}, {name: "deadline-expiry"},
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
			failure := tc.fail != "" || tc.cancel || strings.Contains(tc.name, "truncated") || tc.name == "missing-object" || tc.name == "deadline-expiry" || tc.name == "callback-cancellation"
			if failure {
				if err == nil || result.OK {
					t.Fatalf("incomplete attribution allowed: %+v, %v", result, err)
				}
				if (tc.cancel || tc.name == "callback-cancellation") && !errors.Is(err, context.Canceled) {
					t.Fatalf("lost cancellation: %v", err)
				}
				if tc.name == "deadline-expiry" && !errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("lost deadline: %v", err)
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
