package maputil

import (
	"strconv"
	"testing"
)

var (
	benchKeys []string
	benchStr  string
)

func BenchmarkSortedKeysSmall(b *testing.B) {
	m := map[string]int{
		"path":    1,
		"tool":    2,
		"caller":  3,
		"status":  4,
		"latency": 5,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchKeys = SortedKeys(m)
	}
}

func BenchmarkSortedKeysMedium(b *testing.B) {
	m := make(map[string]int, 50)
	for i := 0; i < 50; i++ {
		m["param_"+strconv.Itoa(i)] = i
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchKeys = SortedKeys(m)
	}
}

func BenchmarkSortedKeysNil(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchKeys = SortedKeys[int](nil)
	}
}

func BenchmarkStrHit(b *testing.B) {
	m := map[string]any{
		"role":    "assistant",
		"content": "adjudicated",
		"status":  "ok",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStr = Str(m, "role")
	}
}

func BenchmarkStrMiss(b *testing.B) {
	m := map[string]any{
		"role":    "assistant",
		"content": "adjudicated",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStr = Str(m, "missing_key")
	}
}

func BenchmarkStrWrongType(b *testing.B) {
	m := map[string]any{
		"count":   100,
		"enabled": true,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStr = Str(m, "count")
	}
}
