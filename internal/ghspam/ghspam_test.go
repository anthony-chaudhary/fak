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
