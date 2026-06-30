package safecommit

import (
	"context"
	"strings"
	"testing"
)

// spuriousStatus overlays onTrunkBase's `status` reply with a porcelain body, so a test can
// drive just the staged-vs-untracked shape the guard reads. The guard issues
// `git status --porcelain -- <paths>`, which the fake harness keys on the bare `status` token.
func spuriousStatus(porcelain string) map[string]reply {
	rep := onTrunkBase()
	rep["status"] = reply{out: porcelain, code: 0}
	return rep
}

// TestSpuriousStagedDeletion_refusesDeleteWithUntrackedTwin is the bug-reproducing test: the
// requested path is staged as a deletion (index `D`) while an untracked copy of the SAME path
// still sits in the working tree. Committing by pathspec would delete a file HEAD carries; the
// guard must refuse BEFORE staging or committing.
func TestSpuriousStagedDeletion_refusesDeleteWithUntrackedTwin(t *testing.T) {
	t.Setenv(spuriousDeleteEnvVar, "") // default = block
	g := &fakeGit{reply: spuriousStatus("D  internal/foo/bar.go\n?? internal/foo/bar.go\n")}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonSpuriousStagedDeletion {
		t.Fatalf("want %q, got reason=%q detail=%q", ReasonSpuriousStagedDeletion, res.Reason, res.Detail)
	}
	if res.Committed {
		t.Fatalf("a spurious-delete refusal must not commit, got %+v", res)
	}
	for _, forbidden := range []string{"add", "commit"} {
		if g.sawSubcommand(forbidden) {
			t.Fatalf("spurious-delete refusal must not %q; calls=%v", forbidden, g.calls)
		}
	}
	if !strings.Contains(res.Detail, "internal/foo/bar.go") || !strings.Contains(res.Detail, "git restore --staged") {
		t.Fatalf("detail should name the path and the restore remedy, got %q", res.Detail)
	}
}

// TestSpuriousStagedDeletion_allowsGenuineDelete confirms the guard does NOT fire on a real
// deletion: the path is staged as `D` with NO untracked twin (the file is genuinely gone). The
// commit proceeds normally.
func TestSpuriousStagedDeletion_allowsGenuineDelete(t *testing.T) {
	t.Setenv(spuriousDeleteEnvVar, "")
	g := &fakeGit{reply: spuriousStatus("D  internal/foo/bar.go\n")}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason == ReasonSpuriousStagedDeletion {
		t.Fatalf("a genuine delete (no untracked twin) must not be refused as spurious; detail=%q", res.Detail)
	}
}

// TestSpuriousStagedDeletion_offSkips confirms FAK_SPURIOUS_DELETE_GUARD=off disables the guard
// even on the spurious shape, for a documented one-shot escape.
func TestSpuriousStagedDeletion_offSkips(t *testing.T) {
	t.Setenv(spuriousDeleteEnvVar, "off")
	g := &fakeGit{reply: spuriousStatus("D  internal/foo/bar.go\n?? internal/foo/bar.go\n")}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason == ReasonSpuriousStagedDeletion {
		t.Fatalf("off must skip the guard, got refusal detail=%q", res.Detail)
	}
}

// TestSpuriousStagedDeletion_warnProceeds confirms warn mode records the would-be refusal in
// Detail but still lets the commit proceed.
func TestSpuriousStagedDeletion_warnProceeds(t *testing.T) {
	t.Setenv(spuriousDeleteEnvVar, "warn")
	g := &fakeGit{reply: spuriousStatus("D  internal/foo/bar.go\n?? internal/foo/bar.go\n")}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), baseOpts())
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason == ReasonSpuriousStagedDeletion {
		t.Fatalf("warn must not refuse, got reason=%q", res.Reason)
	}
	if !strings.Contains(res.Detail, "SPURIOUS_STAGED_DELETION (warn)") {
		t.Fatalf("warn should record the would-be refusal in Detail, got %q", res.Detail)
	}
}

// TestParsePorcelainLine covers the line parser's columns and the rename guard directly.
func TestParsePorcelainLine(t *testing.T) {
	cases := []struct {
		line      string
		path      string
		stagedDel bool
		untracked bool
	}{
		{"D  internal/foo/bar.go", "internal/foo/bar.go", true, false},
		{"?? internal/foo/bar.go", "internal/foo/bar.go", false, true},
		{" M internal/foo/bar.go", "internal/foo/bar.go", false, false},
		{"R  old.go -> new.go", "", false, false}, // rename: not the spurious shape
		{"D", "", false, false},                   // too short
		{"", "", false, false},
	}
	for _, c := range cases {
		path, sd, ut := parsePorcelainLine(c.line)
		if path != c.path || sd != c.stagedDel || ut != c.untracked {
			t.Errorf("parsePorcelainLine(%q) = (%q,%v,%v), want (%q,%v,%v)",
				c.line, path, sd, ut, c.path, c.stagedDel, c.untracked)
		}
	}
}
