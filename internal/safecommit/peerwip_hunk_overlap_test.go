package safecommit

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime medium est=3s lane=default
func TestReconcileNarrowOwnershipZeroContextHunks(t *testing.T) {
	rows := func(edit func([]string) []string) string {
		lines := make([]string, 0, 24)
		for i := 1; i <= 20; i++ {
			lines = append(lines, fmt.Sprintf("row%02d", i))
		}
		if edit != nil {
			lines = edit(lines)
		}
		return strings.Join(lines, "\n") + "\n"
	}
	insertAfter := func(n int, text string) func([]string) []string {
		return func(l []string) []string {
			out := append([]string{}, l[:n]...)
			out = append(out, text)
			return append(out, l[n:]...)
		}
	}
	replaceLine := func(n int, text string) func([]string) []string {
		return func(l []string) []string {
			out := append([]string{}, l...)
			out[n-1] = text
			return out
		}
	}

	cases := []struct {
		name   string
		peer   func([]string) []string
		worker func([]string) []string
		want   HunkOwnershipClassification
		wantOK bool
	}{
		{"DisjointInsertsFiveLinesApart", insertAfter(5, "peer-row"), insertAfter(10, "worker-row"), OwnershipAuthoredHunk, true},
		{"DisjointEditsFiveLinesApart", replaceLine(5, "peer"), replaceLine(10, "worker"), OwnershipAuthoredHunk, true},
		{"DifferentEditsSameLine", replaceLine(7, "peer"), replaceLine(7, "worker"), OwnershipConflictingHunk, false},
		{"InsertsAtSameAnchor", insertAfter(7, "peer-row"), insertAfter(7, "worker-row"), OwnershipConflictingHunk, false},
		{"InsertAdjacentToEditedLine", replaceLine(7, "peer"), insertAfter(7, "worker-row"), OwnershipConflictingHunk, false},
		{"IdenticalEditExempt", replaceLine(7, "same"), replaceLine(7, "same"), OwnershipAuthoredHunk, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			dir := t.TempDir()
			git := func(args ...string) string {
				t.Helper()
				out, code, err := realRunner(ctx, dir, args...)
				if err != nil || code != 0 {
					t.Fatalf("fixture git %v: code=%d err=%v output=%s", args, code, err, out)
				}
				return strings.TrimSpace(out)
			}
			const path = "cmd/fak/verbs_gen.go"
			write := func(body string) {
				t.Helper()
				full := filepath.Join(dir, filepath.FromSlash(path))
				if err := os.MkdirAll(filepath.Dir(full), 0o700); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(full, []byte(body), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			git("init")
			git("config", "user.name", "Fixture")
			git("config", "user.email", "fixture@example.invalid")
			git("config", "core.autocrlf", "false")
			write(rows(nil))
			git("add", "--", path)
			base := git("commit-tree", git("write-tree"), "-m", "base")
			git("update-ref", "HEAD", base)

			write(rows(tc.peer))
			git("add", "--", path)
			peer := git("commit-tree", git("write-tree"), "-p", base, "-m", "peer checkpoint")
			git("read-tree", "HEAD")
			write(rows(tc.worker))

			got, ok := reconcileNarrowOwnership(ctx, realRunner, dir, path, "peer-agent", peer, PathAttributionOptions{})
			if got.Classification != tc.want || ok != tc.wantOK {
				t.Fatalf("classification=%q ok=%v, want %q ok=%v", got.Classification, ok, tc.want, tc.wantOK)
			}
		})
	}
}
