package edgequal

import (
	"testing"
)

func BenchmarkPackBytes(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = PackBytes()
	}
}

func BenchmarkPackSHA256(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = PackSHA256()
	}
}
