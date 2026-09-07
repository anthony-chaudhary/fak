package studyclass

import (
	"testing"
)

func BenchmarkClassify(b *testing.B) {
	corpus := testCorpus()
	digest := testDigest

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Classify(corpus, digest)
	}
}

func BenchmarkValidate(b *testing.B) {
	corpus := testCorpus()
	digest := testDigest
	out, err := Classify(corpus, digest)
	if err != nil {
		b.Fatalf("Classify failed: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Validate(out)
	}
}
