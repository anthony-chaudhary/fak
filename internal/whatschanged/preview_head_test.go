package whatschanged

import (
	"context"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestPreviewKeepsResolvedHeadWhenHEADAdvances(t *testing.T) {
	t.Parallel()

	base := strings.Repeat("1", 40)
	head := strings.Repeat("2", 40)
	later := strings.Repeat("3", 40)
	record := func(sha, subject string) string {
		return strings.Join([]string{sha, "fixture", "fixture@example.test", "1", subject}, "\x1f") + "\x1e\n"
	}
	calls := 0
	run := func(_ context.Context, _ string, args ...string) (RunResult, error) {
		calls++
		if reflect.DeepEqual(args, []string{"rev-parse", "--verify", "base^{commit}"}) {
			return RunResult{Stdout: []byte(base + "\n")}, nil
		}
		if reflect.DeepEqual(args, []string{"rev-parse", "--verify", "HEAD^{commit}"}) {
			return RunResult{Stdout: []byte(head + "\n")}, nil
		}
		if len(args) == 6 && args[0] == "log" {
			want := []string{"log", "--reverse", "--format=%H%x1f%an%x1f%ae%x1f%ct%x1f%s%x1e",
				base + ".." + head, "--", "internal/leaf"}
			switch args[3] {
			case base + ".." + head:
			case base + "..HEAD":
				// Simulate HEAD moving after Preview resolved its report snapshot.
				want[3] = base + "..HEAD"
			default:
				t.Fatalf("unexpected log range: %v", args)
			}
			if !reflect.DeepEqual(args, want) {
				t.Fatalf("unexpected log arguments: %v", args)
			}
			out := record(head, "snapshot")
			if args[3] == base+"..HEAD" {
				out += record(later, "arrived later")
			}
			return RunResult{Stdout: []byte(out)}, nil
		}
		for sha, file := range map[string]string{
			head: "internal/leaf/snapshot.go", later: "internal/leaf/later.go",
		} {
			want := []string{"diff-tree", "--no-commit-id", "--name-only", "-r", "-m", "-z",
				sha, "--", "internal/leaf"}
			if reflect.DeepEqual(args, want) {
				return RunResult{Stdout: []byte(file + "\x00")}, nil
			}
		}
		t.Fatalf("unexpected runner call: %v", args)
		return RunResult{}, nil
	}
	got, err := Preview(context.Background(), "unused", Options{
		Since: "base", Paths: []string{"internal/leaf"}, Run: run,
	})
	want := Report{
		Since: base, Head: head, Range: base + ".." + head, Paths: []string{"internal/leaf"},
		ChangedFiles: []string{"internal/leaf/snapshot.go"},
		Commits: []Commit{{
			SHA: head, Short: head[:12], Subject: "snapshot", AuthorName: "fixture",
			AuthorEmail: "fixture@example.test", UnixTime: 1, Files: []string{"internal/leaf/snapshot.go"},
		}},
	}
	if err != nil || !reflect.DeepEqual(got, want) || calls != 4 {
		t.Fatalf("Preview = %#v, %v after %d calls; want %#v after 4 calls", got, err, calls, want)
	}
}
