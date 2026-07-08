package ghspam

import "testing"

func TestAnalyzeFlagsUntrustedReleaseArchiveLinks(t *testing.T) {
	comments := []Comment{
		{
			ID:                1,
			NodeID:            "IC_1",
			HTMLURL:           "https://github.com/o/r/issues/1#issuecomment-1",
			AuthorAssociation: "NONE",
			User:              User{Login: "cosuwacu"},
			CreatedAt:         "2026-07-05T09:11:50Z",
			Body:              "(https://github.com/bareneguboko/patch_fix/releases/download/release/patch_fix.rar)",
		},
		{
			ID:                2,
			NodeID:            "IC_2",
			HTMLURL:           "https://github.com/o/r/issues/2#issuecomment-2",
			AuthorAssociation: "NONE",
			User:              User{Login: "darosomowada"},
			CreatedAt:         "2026-07-05T09:16:02Z",
			Body:              "might answer your question (https://github.com/bebusucuku/release_fix_2.1.1/releases/download/release/release_fix_2.1.1.zip)",
		},
	}

	rep := Analyze(comments, DefaultOptions())
	if rep.Counts.Matched != 2 {
		t.Fatalf("matched = %d, want 2", rep.Counts.Matched)
	}
	if rep.Findings[0].ArchiveURL != "https://github.com/bareneguboko/patch_fix/releases/download/release/patch_fix.rar" {
		t.Fatalf("archive URL = %q", rep.Findings[0].ArchiveURL)
	}
	if rep.Findings[1].Reason != "untrusted_github_release_archive_link" {
		t.Fatalf("reason = %q", rep.Findings[1].Reason)
	}
}

func TestAnalyzeIgnoresTrustedInsiders(t *testing.T) {
	comments := []Comment{
		{
			ID:                1,
			NodeID:            "IC_owner",
			AuthorAssociation: "OWNER",
			User:              User{Login: "anthony-chaudhary"},
			Body:              "release witness https://github.com/owner/repo/releases/download/v1/tool.zip",
		},
		{
			ID:                2,
			NodeID:            "IC_collab",
			AuthorAssociation: "COLLABORATOR",
			User:              User{Login: "claude-ai-netra"},
			Body:              "release witness https://github.com/owner/repo/releases/download/v1/tool.zip",
		},
	}

	rep := Analyze(comments, DefaultOptions())
	if rep.Counts.Matched != 0 {
		t.Fatalf("matched = %d, want 0", rep.Counts.Matched)
	}
	if rep.Counts.TrustedSkipped != 2 {
		t.Fatalf("trusted skipped = %d, want 2", rep.Counts.TrustedSkipped)
	}
}

func TestAnalyzeFlagsFakePatchLurePhrasing(t *testing.T) {
	comments := []Comment{
		{
			ID:                7,
			NodeID:            "IC_lure",
			HTMLURL:           "https://github.com/o/r/issues/7#issuecomment-7",
			AuthorAssociation: "NONE",
			User:              User{Login: "driveby"},
			CreatedAt:         "2026-07-06T10:00:00Z",
			// No GitHub release-archive link: the archive family misses and the
			// fake-patch lure family must own this comment instead.
			Body: "Solved it! download the working crack here, password is 1234: https://mega.nz/file/abcd",
		},
		{
			ID:                8,
			NodeID:            "IC_benign",
			HTMLURL:           "https://github.com/o/r/issues/8#issuecomment-8",
			AuthorAssociation: "NONE",
			User:              User{Login: "realbug"},
			CreatedAt:         "2026-07-06T10:05:00Z",
			// Mentions "fix" but names no download/host action: not a lure.
			Body: "The fix should probably clamp the index before the slice read.",
		},
	}

	rep := Analyze(comments, DefaultOptions())
	if rep.Counts.Matched != 1 {
		t.Fatalf("matched = %d, want 1", rep.Counts.Matched)
	}
	if rep.Findings[0].Reason != "fake_patch_fix_lure_phrasing" {
		t.Fatalf("reason = %q, want fake_patch_fix_lure_phrasing", rep.Findings[0].Reason)
	}
	if rep.Findings[0].ArchiveURL != "" {
		t.Fatalf("archive_url = %q, want empty for the lure family", rep.Findings[0].ArchiveURL)
	}
	if rep.Findings[0].Match != "crack/download" {
		t.Fatalf("match = %q, want crack/download", rep.Findings[0].Match)
	}
}

func TestAnalyzeIgnoresTrustedInsiderPatchPhrasing(t *testing.T) {
	comments := []Comment{{
		ID:                9,
		NodeID:            "IC_owner_patch",
		AuthorAssociation: "MEMBER",
		User:              User{Login: "claude-ai-netra"},
		// Same lure shape as the spam family, but from a trusted insider: skip it.
		Body: "download the patch and extract it, password is in the release notes",
	}}

	rep := Analyze(comments, DefaultOptions())
	if rep.Counts.Matched != 0 {
		t.Fatalf("matched = %d, want 0", rep.Counts.Matched)
	}
	if rep.Counts.TrustedSkipped != 1 {
		t.Fatalf("trusted skipped = %d, want 1", rep.Counts.TrustedSkipped)
	}
}

func TestFamiliesCoverAtLeastTwoAbuseFamilies(t *testing.T) {
	fams := Families()
	if len(fams) < 2 {
		t.Fatalf("families = %d, want at least 2 abuse families", len(fams))
	}
	seen := map[string]bool{}
	for _, f := range fams {
		if f.Reason == "" {
			t.Fatalf("family has empty reason")
		}
		if seen[f.Reason] {
			t.Fatalf("duplicate family reason %q", f.Reason)
		}
		seen[f.Reason] = true
	}
}

func TestAnalyzeTrustUserOverridesAssociation(t *testing.T) {
	comments := []Comment{{
		ID:                1,
		NodeID:            "IC_trusted",
		AuthorAssociation: "NONE",
		User:              User{Login: "known-bot"},
		Body:              "https://github.com/owner/repo/releases/download/v1/tool.zip",
	}}

	rep := Analyze(comments, Options{TrustedAssociations: DefaultOptions().TrustedAssociations, TrustedUsers: []string{"known-bot"}})
	if rep.Counts.Matched != 0 {
		t.Fatalf("matched = %d, want 0", rep.Counts.Matched)
	}
}
