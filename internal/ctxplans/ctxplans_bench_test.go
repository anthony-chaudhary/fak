package ctxplans

import (
	"path/filepath"
	"testing"
)

func BenchmarkScanFixture(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := Scan(fixtureRoot)
		if err != nil {
			b.Fatalf("Scan: %v", err)
		}
	}
}

func BenchmarkScanDirectives(b *testing.B) {
	dir := filepath.Join(fixtureRoot, "cmd", "fak")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := scanDirectives(dir)
		if err != nil {
			b.Fatalf("scanDirectives: %v", err)
		}
	}
}

func BenchmarkDispatchVerbs(b *testing.B) {
	mainPath := filepath.Join(fixtureRoot, "cmd", "fak", "main.go")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := dispatchVerbs(mainPath)
		if err != nil {
			b.Fatalf("dispatchVerbs: %v", err)
		}
	}
}

func BenchmarkScanSkills(b *testing.B) {
	dir := filepath.Join(fixtureRoot, ".claude", "skills")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := scanSkills(dir)
		if err != nil {
			b.Fatalf("scanSkills: %v", err)
		}
	}
}

func BenchmarkParseDirective(b *testing.B) {
	text := `//fak:ctxplan verb=session enters="the memory store" pages="the active window" warms="the prompt cache"`
	file := "cmd/fak/session.go"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, ok := parseDirective(text, file, 42)
		if !ok {
			b.Fatal("parseDirective failed")
		}
	}
}

func BenchmarkIsContextVerbName(b *testing.B) {
	names := []string{
		"session",
		"vcache",
		"headroom",
		"recall",
		"widget",
		"memory-read",
		"chatrelay",
		"ctxplans",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, name := range names {
			_ = isContextVerbName(name)
		}
	}
}
