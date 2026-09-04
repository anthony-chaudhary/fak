package ghspam

import (
	"fmt"
	"testing"
)

func benchmarkCommentCorpus() []Comment {
	return []Comment{
		{
			ID:                1,
			NodeID:            "IC_benign_1",
			HTMLURL:           "https://github.com/org/repo/issues/1#issuecomment-1",
			AuthorAssociation: "NONE",
			User:              User{Login: "external-dev"},
			CreatedAt:         "2026-09-04T10:00:00Z",
			Body:              "Can you provide a minimal reproduction case for the flaky timeout?",
		},
		{
			ID:                2,
			NodeID:            "IC_lure_1",
			HTMLURL:           "https://github.com/org/repo/issues/1#issuecomment-2",
			AuthorAssociation: "NONE",
			User:              User{Login: "driveby-spammer"},
			CreatedAt:         "2026-09-04T10:01:00Z",
			Body:              "Solved it! Download the working patch here, password is 1234: https://mega.nz/file/123",
		},
		{
			ID:                3,
			NodeID:            "IC_archive_1",
			HTMLURL:           "https://github.com/org/repo/issues/1#issuecomment-3",
			AuthorAssociation: "NONE",
			User:              User{Login: "bot-account"},
			CreatedAt:         "2026-09-04T10:02:00Z",
			Body:              "Check release binary: https://github.com/attacker/repo/releases/download/v1.0.0/payload.zip",
		},
		{
			ID:                4,
			NodeID:            "IC_trusted_1",
			HTMLURL:           "https://github.com/org/repo/issues/1#issuecomment-4",
			AuthorAssociation: "MEMBER",
			User:              User{Login: "team-member"},
			CreatedAt:         "2026-09-04T10:03:00Z",
			Body:              "The fix requires installing dependencies and running the unit tests.",
		},
		{
			ID:                5,
			NodeID:            "IC_benign_2",
			HTMLURL:           "https://github.com/org/repo/issues/1#issuecomment-5",
			AuthorAssociation: "CONTRIBUTOR",
			User:              User{Login: "helpful-peer"},
			CreatedAt:         "2026-09-04T10:04:00Z",
			Body:              "Here is the stack trace showing the index out of bounds on line 42.",
		},
		{
			ID:                6,
			NodeID:            "IC_lure_2",
			HTMLURL:           "https://github.com/org/repo/issues/1#issuecomment-6",
			AuthorAssociation: "NONE",
			User:              User{Login: "scam-bot"},
			CreatedAt:         "2026-09-04T10:05:00Z",
			Body:              "Official crack and keygen: grab it from https://anonfiles.com/file/crack",
		},
	}
}

func makeBenchmarkBatch(count int) []Comment {
	corpus := benchmarkCommentCorpus()
	out := make([]Comment, count)
	for i := 0; i < count; i++ {
		c := corpus[i%len(corpus)]
		c.ID = int64(i + 1)
		c.NodeID = fmt.Sprintf("IC_bench_%d", i+1)
		c.CreatedAt = fmt.Sprintf("2026-09-04T12:%02d:%02dZ", (i/60)%60, i%60)
		out[i] = c
	}
	return out
}

// BenchmarkGHSpamClassify measures classification throughput and allocations
// across mixed spam and benign comments.
func BenchmarkGHSpamClassify(b *testing.B) {
	comments := makeBenchmarkBatch(50)
	opts := DefaultOptions()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := Analyze(comments, opts)
		if rep.Counts.Scanned != len(comments) {
			b.Fatalf("scanned = %d, want %d", rep.Counts.Scanned, len(comments))
		}
	}
}

// BenchmarkGHSpamClassifyBatch measures classification scaling across batch sizes.
func BenchmarkGHSpamClassifyBatch(b *testing.B) {
	opts := DefaultOptions()
	for _, size := range []int{10, 100, 500} {
		b.Run(fmt.Sprintf("Size_%d", size), func(b *testing.B) {
			comments := makeBenchmarkBatch(size)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				rep := Analyze(comments, opts)
				if rep.Counts.Scanned != size {
					b.Fatalf("scanned = %d, want %d", rep.Counts.Scanned, size)
				}
			}
		})
	}
}
