package hooks

import (
	"context"
	"strings"
	"testing"
)

type freshDeleteReply struct {
	out  string
	code int
}

type freshDeleteFakeGit map[string]freshDeleteReply

func (f freshDeleteFakeGit) run(_ context.Context, _ string, args ...string) (string, int, error) {
	key := strings.Join(args, "\x00")
	if r, ok := f[key]; ok {
		return r.out, r.code, nil
	}
	return "", 0, nil
}

func freshDeleteReplies(path, addSHA, count string) freshDeleteFakeGit {
	return freshDeleteFakeGit{
		strings.Join([]string{"diff", "--cached", "--name-only", "--diff-filter=D"}, "\x00"): {
			out: path + "\n",
		},
		strings.Join([]string{"log", "--diff-filter=A", "--format=%H", "--max-count=1", "HEAD", "--", path}, "\x00"): {
			out: addSHA + "\n",
		},
		strings.Join([]string{"rev-list", "--count", addSHA + "..HEAD"}, "\x00"): {
			out: count + "\n",
		},
	}
}

func TestFreshDeletionFindingsFlagsFreshDeletionNotNamedInMessage(t *testing.T) {
	p := "docs/notes/SLACK-CONTROL-FOUNDATION-2026-07-02.md"
	g := freshDeleteReplies(p, "18a7a8a6abc", "1")

	findings, err := freshDeletionFindingsWith(context.Background(), g.run, "/repo", "docs(notes): bind no-babysitting doctrine", 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(findings) != 1 {
		t.Fatalf("fresh unnamed deletion should be flagged, got %+v", findings)
	}
	if findings[0].Gate != freshDeletionGateName || findings[0].File != p {
		t.Fatalf("finding = %+v", findings[0])
	}
}

func TestFreshDeletionFindingsAllowsMessageThatNamesDeletedPath(t *testing.T) {
	p := "docs/notes/SLACK-CONTROL-FOUNDATION-2026-07-02.md"
	g := freshDeleteReplies(p, "18a7a8a6abc", "1")

	findings, err := freshDeletionFindingsWith(context.Background(), g.run, "/repo", "docs(notes): remove Slack Control Foundation note", 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(findings) != 0 {
		t.Fatalf("message names the deleted path; got %+v", findings)
	}
}

func TestFreshDeletionFindingsIgnoresOldDeletion(t *testing.T) {
	p := "docs/notes/OLD.md"
	g := freshDeleteReplies(p, "111111111111", "99")

	findings, err := freshDeletionFindingsWith(context.Background(), g.run, "/repo", "docs(notes): prune obsolete note", 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(findings) != 0 {
		t.Fatalf("old deletion should not fire the fresh-peer backstop, got %+v", findings)
	}
}
