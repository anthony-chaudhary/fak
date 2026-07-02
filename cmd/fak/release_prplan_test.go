package main

import (
	"bytes"
	"strings"
	"testing"
)

const prPlanFakeLog = "\x1eccc3333333333333333333333333333333333333\x1fdocs: loose note without a stamp\x1f\x1f\ndocs/notes/x.md\n" +
	"\x1ebbb2222222222222222222222222222222222222\x1ffix(gateway): drop duplicate counter (fak gateway)\x1f\x1f\ninternal/gateway/messages.go\n" +
	"\x1eaaa1111111111111111111111111111111111111\x1ffeat(gateway): treat same-tick ready as positive #1146 (fak gateway)\x1fLong body.\n\nSee #999 for the follow-on.\x1f\ninternal/gateway/gateway.go\ninternal/gateway/messages.go\n"

func TestParsePRPlanLog(t *testing.T) {
	commits := parsePRPlanLog(prPlanFakeLog)
	if len(commits) != 3 {
		t.Fatalf("commits = %d, want 3: %#v", len(commits), commits)
	}
	unstamped := commits[0]
	if unstamped.Leaf != "" || unstamped.Type != "docs" {
		t.Fatalf("unstamped commit parsed wrong: %#v", unstamped)
	}
	feat := commits[2]
	if feat.Leaf != "gateway" || feat.Type != "feat" {
		t.Fatalf("stamped commit parsed wrong: %#v", feat)
	}
	if len(feat.Resolves) != 1 || feat.Resolves[0] != "#1146" {
		t.Fatalf("subject-bound issues = %#v, want [#1146]", feat.Resolves)
	}
	if len(feat.Mentions) != 1 || feat.Mentions[0] != "#999" {
		t.Fatalf("body mentions = %#v, want [#999]", feat.Mentions)
	}
	if len(feat.Files) != 2 {
		t.Fatalf("files = %#v, want 2 paths", feat.Files)
	}
}

func TestFoldPRPlanUnits(t *testing.T) {
	units, unstamped := foldPRPlanUnits(parsePRPlanLog(prPlanFakeLog))
	if len(units) != 1 || len(unstamped) != 1 {
		t.Fatalf("units=%d unstamped=%d, want 1/1", len(units), len(unstamped))
	}
	unit := units[0]
	if unit.Leaf != "gateway" || len(unit.Commits) != 2 {
		t.Fatalf("unit = %#v, want gateway with 2 commits", unit)
	}
	// git log order is newest-first; the PR body reads oldest-first.
	if !strings.HasPrefix(unit.Commits[0].Subject, "feat(gateway)") {
		t.Fatalf("unit commits not chronological: %#v", unit.Commits)
	}
	if unit.Title != "gateway: 2 commits (feat 1, fix 1)" {
		t.Fatalf("unit title = %q", unit.Title)
	}
	if len(unit.Resolves) != 1 || unit.Resolves[0] != "#1146" {
		t.Fatalf("unit resolves = %#v", unit.Resolves)
	}
	if len(unit.Mentions) != 1 || unit.Mentions[0] != "#999" {
		t.Fatalf("unit mentions = %#v", unit.Mentions)
	}
	if len(unit.Files) != 2 {
		t.Fatalf("unit files = %#v, want deduped 2", unit.Files)
	}
}

func TestPRPlanSingleCommitUnitUsesSubjectAsTitle(t *testing.T) {
	units, _ := foldPRPlanUnits([]prPlanCommit{{
		SHA: "aaa1111111111111111111111111111111111111", Subject: "fix(vcache): honor TTL (fak vcache)", Leaf: "vcache", Type: "fix",
	}})
	if len(units) != 1 || units[0].Title != "fix(vcache): honor TTL (fak vcache)" {
		t.Fatalf("single-commit unit title = %#v", units)
	}
}

func prPlanFakeGit(log string) func(string, ...string) string {
	return func(root string, args ...string) string {
		switch args[0] {
		case "rev-parse":
			ref := args[len(args)-1]
			if strings.HasPrefix(ref, "base") {
				return "aaa1111111111111111111111111111111111111"
			}
			if strings.HasPrefix(ref, "head") {
				return "bbb2222222222222222222222222222222222222"
			}
			if strings.HasPrefix(ref, "same") {
				return "ddd4444444444444444444444444444444444444"
			}
			return ""
		case "log":
			return log
		}
		return ""
	}
}

func TestBuildPRPlanFoldsRange(t *testing.T) {
	orig := releasePRPlanGit
	releasePRPlanGit = prPlanFakeGit(prPlanFakeLog)
	defer func() { releasePRPlanGit = orig }()

	plan, err := buildPRPlan(t.TempDir(), prPlanOptions{Base: "baseref", Head: "headref"})
	if err != nil {
		t.Fatalf("buildPRPlan: %v", err)
	}
	if plan["schema"] != releasePRPlanSchema || plan["commit_count"] != 3 || plan["unit_count"] != 1 || plan["unstamped_count"] != 1 {
		t.Fatalf("plan totals wrong: %#v", plan)
	}
	if releaseStatusBool(plan["check_ok"]) {
		t.Fatalf("check_ok = true with an unstamped commit: %#v", plan)
	}
	md := renderPRPlanMarkdown(plan, 20)
	for _, want := range []string{
		"# Promotion PR plan — baseref..headref",
		"## gateway — 2 commit(s)",
		"**Title:** `gateway: 2 commits (feat 1, fix 1)`",
		"Closes #1146.",
		"Mentions #999.",
		"unstamped — 1 commit(s)",
	} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown missing %q:\n%s", want, md)
		}
	}
}

func TestBuildPRPlanEmptyRange(t *testing.T) {
	orig := releasePRPlanGit
	releasePRPlanGit = prPlanFakeGit("")
	defer func() { releasePRPlanGit = orig }()

	plan, err := buildPRPlan(t.TempDir(), prPlanOptions{Base: "sameref", Head: "sameref2"})
	if err != nil {
		t.Fatalf("buildPRPlan: %v", err)
	}
	if plan["commit_count"] != 0 || !releaseStatusBool(plan["check_ok"]) {
		t.Fatalf("empty range plan wrong: %#v", plan)
	}
	if md := renderPRPlanMarkdown(plan, 20); !strings.Contains(md, "The promotion range is empty") {
		t.Fatalf("empty-range markdown wrong:\n%s", md)
	}
}

func TestBuildPRPlanUnresolvableRefErrors(t *testing.T) {
	orig := releasePRPlanGit
	releasePRPlanGit = prPlanFakeGit("")
	defer func() { releasePRPlanGit = orig }()

	if _, err := buildPRPlan(t.TempDir(), prPlanOptions{Base: "nosuchref", Head: "headref"}); err == nil {
		t.Fatal("unresolvable base should error")
	}
}

func TestRunReleasePRPlanCheckGatesUnstamped(t *testing.T) {
	orig := releasePRPlanGit
	releasePRPlanGit = prPlanFakeGit(prPlanFakeLog)
	defer func() { releasePRPlanGit = orig }()

	var stdout, stderr bytes.Buffer
	if code := runReleasePRPlan(&stdout, &stderr, []string{"--check", "--base", "baseref", "--head", "headref"}); code != 1 {
		t.Fatalf("exit = %d, want 1 (unstamped commit present); stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "ship-stamp") {
		t.Fatalf("check refusal should explain the stamp gate: %s", stderr.String())
	}

	stdout.Reset()
	stderr.Reset()
	releasePRPlanGit = prPlanFakeGit("")
	if code := runReleasePRPlan(&stdout, &stderr, []string{"--check", "--base", "sameref", "--head", "sameref2"}); code != 0 {
		t.Fatalf("exit = %d, want 0 on clean empty range; stderr=%s", code, stderr.String())
	}
}

func TestPRPlanFileListFolds(t *testing.T) {
	if got := prPlanFileList([]string{"a", "b", "c"}, 2); got != "a, b (+1 more)" {
		t.Fatalf("file list fold = %q", got)
	}
	if got := prPlanFileList(nil, 2); got != "(none recorded)" {
		t.Fatalf("empty file list = %q", got)
	}
}
