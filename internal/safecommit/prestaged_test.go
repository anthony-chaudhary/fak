package safecommit

import (
	"context"
	"strings"
	"testing"
)

func preStagedStatus(porcelain string) map[string]reply {
	rep := onTrunkBase()
	rep["status"] = reply{out: porcelain, code: 0}
	return rep
}

func TestPreStagedPathOverlapRefusesBeforeAdd(t *testing.T) {
	t.Setenv(preStagedPathEnvVar, "")
	g := &fakeGit{reply: preStagedStatus("MM internal/foo/bar.go\n")}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonPreStagedPathOverlap {
		t.Fatalf("want %q, got reason=%q detail=%q", ReasonPreStagedPathOverlap, res.Reason, res.Detail)
	}
	if res.Committed {
		t.Fatalf("pre-staged requested path must not commit, got %+v", res)
	}
	for _, forbidden := range []string{"add", "commit"} {
		if g.sawSubcommand(forbidden) {
			t.Fatalf("pre-staged refusal must not %q; calls=%v", forbidden, g.calls)
		}
	}
	if !strings.Contains(res.Detail, "internal/foo/bar.go") || !strings.Contains(res.Detail, "git restore --staged") {
		t.Fatalf("detail should name the path and unstaging remedy, got %q", res.Detail)
	}
}

func TestPreStagedPathOverlapAllowsUnstagedAndUntracked(t *testing.T) {
	t.Setenv(preStagedPathEnvVar, "")
	detail, fired := preStagedPathOverlapFromStatus(" M internal/foo/bar.go\n?? internal/foo/new.go\n", []string{
		"internal/foo/bar.go",
		"internal/foo/new.go",
	})
	if fired {
		t.Fatalf("unstaged and untracked requested paths must not fire, detail=%q", detail)
	}
}

func TestPreStagedPathOverlapRequestedDirectoryCoversFiles(t *testing.T) {
	t.Setenv(preStagedPathEnvVar, "")
	opts := baseOpts()
	opts.Paths = []string{"internal/foo"}
	g := &fakeGit{reply: preStagedStatus("M  internal/foo/bar.go\n")}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonPreStagedPathOverlap {
		t.Fatalf("directory pathspec must catch staged child path, got reason=%q detail=%q", res.Reason, res.Detail)
	}
	if !strings.Contains(res.Detail, "internal/foo/bar.go") {
		t.Fatalf("detail should name staged child path, got %q", res.Detail)
	}
}

func TestPreStagedPathOverlapWarnProceeds(t *testing.T) {
	t.Setenv(preStagedPathEnvVar, "warn")
	g := &fakeGit{reply: preStagedStatus("M  internal/foo/bar.go\n")}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason == ReasonPreStagedPathOverlap {
		t.Fatalf("warn must not refuse, got reason=%q", res.Reason)
	}
	if !res.Verified {
		t.Fatalf("warn mode should commit and verify, got %+v", res)
	}
	if !strings.Contains(res.Detail, "PRESTAGED_PATH_OVERLAP (warn)") {
		t.Fatalf("warn should record the would-be refusal in Detail, got %q", res.Detail)
	}
}

func TestPreStagedPathOverlapOffSkips(t *testing.T) {
	t.Setenv(preStagedPathEnvVar, "off")
	g := &fakeGit{reply: preStagedStatus("A  internal/foo/bar.go\n")}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason == ReasonPreStagedPathOverlap {
		t.Fatalf("off must skip the guard, got detail=%q", res.Detail)
	}
}

func TestParsePreStagedPathLine(t *testing.T) {
	cases := []struct {
		line   string
		path   string
		staged bool
	}{
		{"M  internal/foo/bar.go", "internal/foo/bar.go", true},
		{"MM internal/foo/bar.go", "internal/foo/bar.go", true},
		{"A  internal/foo/new.go", "internal/foo/new.go", true},
		{"D  internal/foo/bar.go", "internal/foo/bar.go", false},
		{" M internal/foo/bar.go", "internal/foo/bar.go", false},
		{"?? internal/foo/new.go", "internal/foo/new.go", false},
		{"R  old.go -> internal/foo/new.go", "internal/foo/new.go", true},
		{"D", "", false},
		{"", "", false},
	}
	for _, c := range cases {
		path, staged := parsePreStagedPathLine(c.line)
		if path != c.path || staged != c.staged {
			t.Errorf("parsePreStagedPathLine(%q) = (%q,%v), want (%q,%v)", c.line, path, staged, c.path, c.staged)
		}
	}
}
