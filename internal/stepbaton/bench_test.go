package stepbaton

import (
	"path/filepath"
	"testing"
)

func BenchmarkNew(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = New("trace-1", StepCheckpoint, "token_headroom", "window nearly spent", "guard", "abc123sha", 92831, 48000)
	}
}

func BenchmarkNormalizeStepClass(b *testing.B) {
	classes := []string{StepAny, StepBounded, StepCheckpoint, StepRebuild, StepUnknown, "  checkpoint  ", "invalid"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NormalizeStepClass(classes[i%len(classes)])
	}
}

func BenchmarkValidStepClass(b *testing.B) {
	classes := []string{StepAny, StepBounded, StepCheckpoint, StepRebuild, StepUnknown, "invalid", ""}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ValidStepClass(classes[i%len(classes)])
	}
}

func BenchmarkLine(b *testing.B) {
	s := New("trace-1", StepCheckpoint, "token_headroom", "window nearly spent", "guard", "abc123sha", 92831, 48000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = s.Line()
	}
}

func BenchmarkShouldCarry(b *testing.B) {
	stamps := []Stamp{
		New("t1", StepCheckpoint, "token_headroom", "r1", "guard", "", 100, 200),
		New("t2", StepRebuild, "context_event", "r2", "guard", "", 200, 300),
		New("t3", StepAny, "none", "r3", "guard", "", 0, 0),
		New("t4", StepBounded, "token_headroom", "r4", "guard", "", 50, 100),
		New("t5", StepUnknown, "", "", "", "", 0, 0),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = stamps[i%len(stamps)].ShouldCarry()
	}
}

func BenchmarkMarshal(b *testing.B) {
	s := New("trace-1", StepCheckpoint, "token_headroom", "window nearly spent", "guard", "abc123sha", 92831, 48000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Marshal(s)
	}
}

func BenchmarkUnmarshal(b *testing.B) {
	s := New("trace-1", StepCheckpoint, "token_headroom", "window nearly spent", "guard", "abc123sha", 92831, 48000)
	data, err := Marshal(s)
	if err != nil {
		b.Fatalf("Marshal: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Unmarshal(data)
	}
}

func BenchmarkPath(b *testing.B) {
	sessions := []string{"session-12345", "agent-abc_def.1", "hostile/../path", "safe-id"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Path("/var/run/fak", sessions[i%len(sessions)])
	}
}

func BenchmarkWrite(b *testing.B) {
	dir := b.TempDir()
	s := New("trace-1", StepCheckpoint, "token_headroom", "window nearly spent", "guard", "abc123sha", 92831, 48000)
	path := filepath.Join(dir, "stepadvice-bench.json")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := Write(path, s); err != nil {
			b.Fatalf("Write: %v", err)
		}
	}
}

func BenchmarkRead(b *testing.B) {
	dir := b.TempDir()
	s := New("trace-1", StepCheckpoint, "token_headroom", "window nearly spent", "guard", "abc123sha", 92831, 48000)
	path := filepath.Join(dir, "stepadvice-read.json")
	if err := Write(path, s); err != nil {
		b.Fatalf("Write: %v", err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got, ok, err := Read(path)
		if err != nil || !ok || got.StepClass != StepCheckpoint {
			b.Fatalf("Read failed: ok=%v err=%v", ok, err)
		}
	}
}

func TestBenchmarkSanity(t *testing.T) {
	s := New("trace-sanity", StepCheckpoint, "token_headroom", "window nearly spent", "guard", "sha1", 100, 200)
	if !s.ShouldCarry() {
		t.Errorf("ShouldCarry() = false, want true")
	}
	data, err := Marshal(s)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	got, err := Unmarshal(data)
	if err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got.TraceID != "trace-sanity" {
		t.Errorf("TraceID = %q, want trace-sanity", got.TraceID)
	}
}
