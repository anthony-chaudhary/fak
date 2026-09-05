package stepbatoncapture

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/stepbaton"
)

func BenchmarkCtxvalueURL(b *testing.B) {
	urls := []string{
		"http://localhost:8080",
		"http://localhost:8080/",
		"http://localhost:8080/v1",
		"http://localhost:8080/metrics",
		"https://gateway.internal:9443/base/v1",
		"",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ctxvalueURL(urls[i%len(urls)])
	}
}

func BenchmarkSelectReport(b *testing.B) {
	snap := wireSnapshot{
		Sessions: []wireReport{
			{TraceID: "sess-1", Turns: struct {
				TurnsObserved int `json:"turns_observed"`
			}{TurnsObserved: 5}},
			{TraceID: "sess-2", Turns: struct {
				TurnsObserved int `json:"turns_observed"`
			}{TurnsObserved: 18}},
			{TraceID: "sess-3", Turns: struct {
				TurnsObserved int `json:"turns_observed"`
			}{TurnsObserved: 12}},
			{TraceID: "sess-4", Turns: struct {
				TurnsObserved int `json:"turns_observed"`
			}{TurnsObserved: 2}},
		},
	}
	hints := []string{"sess-3", "", "unknown-sess", "sess-2"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = selectReport(snap, hints[i%len(hints)])
	}
}

func BenchmarkProject(b *testing.B) {
	rep := wireReport{
		TraceID: "trace-bench-1",
		Tokens: struct {
			ResidentTokens int `json:"resident_tokens"`
			BudgetTokens   int `json:"budget_tokens"`
		}{ResidentTokens: 85000, BudgetTokens: 100000},
		Turns: struct {
			TurnsObserved int `json:"turns_observed"`
		}{TurnsObserved: 15},
		Session: struct {
			Phase string `json:"phase"`
		}{Phase: "crowding"},
		StepAdvice: struct {
			StepClass string `json:"step_class"`
			Basis     string `json:"basis"`
			Reason    string `json:"reason"`
		}{
			StepClass: "checkpoint",
			Basis:     "token_headroom",
			Reason:    "resident 85k of 100k budget (85% used)",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = project(rep, "abcdef123456")
	}
}

func BenchmarkCapture(b *testing.B) {
	body := `{"schema":"fak-ctxvalue-report/1","budget_tokens":100000,"sessions":[` +
		`{"trace_id":"trace-1","tokens":{"resident_tokens":80000,"budget_tokens":100000},"turns":{"turns_observed":12},"session":{"phase":"crowding"},"step_advice":{"step_class":"checkpoint","basis":"token_headroom","reason":"80% used"}},` +
		`{"trace_id":"trace-2","tokens":{"resident_tokens":40000,"budget_tokens":100000},"turns":{"turns_observed":5},"session":{"phase":"cruising"},"step_advice":{"step_class":"bounded","basis":"token_headroom","reason":"40% used"}}` +
		`]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	ctx := context.Background()
	opts := Options{
		BaseURL:   srv.URL,
		TraceHint: "trace-1",
		SHA:       "abcdef123456",
		Client:    srv.Client(),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stamp, ok, err := Capture(ctx, opts)
		if err != nil || !ok || stamp.StepClass != stepbaton.StepCheckpoint {
			b.Fatalf("Capture failed: ok=%v err=%v stamp=%+v", ok, err, stamp)
		}
	}
}

func BenchmarkCaptureAndWrite(b *testing.B) {
	body := `{"schema":"fak-ctxvalue-report/1","budget_tokens":100000,"sessions":[` +
		`{"trace_id":"trace-1","tokens":{"resident_tokens":80000,"budget_tokens":100000},"turns":{"turns_observed":12},"session":{"phase":"crowding"},"step_advice":{"step_class":"checkpoint","basis":"token_headroom","reason":"80% used"}}` +
		`]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	dir := b.TempDir()
	ctx := context.Background()
	opts := Options{
		BaseURL:   srv.URL,
		Dir:       dir,
		SessionID: "bench-sess",
		SHA:       "abcdef123456",
		Client:    srv.Client(),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stamp, ok, err := CaptureAndWrite(ctx, opts)
		if err != nil || !ok || stamp.StepClass != stepbaton.StepCheckpoint {
			b.Fatalf("CaptureAndWrite failed: ok=%v err=%v stamp=%+v", ok, err, stamp)
		}
	}
}

func BenchmarkReap(b *testing.B) {
	dir := b.TempDir()
	path := stepbaton.Path(dir, "bench-sess")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		if err := os.WriteFile(path, []byte("{}\n"), 0o600); err != nil {
			b.Fatalf("write file: %v", err)
		}
		b.StartTimer()
		if err := ReapClosedAdvice(dir, "bench-sess"); err != nil {
			b.Fatalf("ReapClosedAdvice: %v", err)
		}
	}
}

func BenchmarkReapAbsent(b *testing.B) {
	dir := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ReapClosedAdvice(dir, "absent-sess"); err != nil {
			b.Fatalf("ReapClosedAdvice: %v", err)
		}
	}
}

func BenchmarkSweepStaleAdvice(b *testing.B) {
	dir := b.TempDir()
	now := time.Now()
	for i := 0; i < 20; i++ {
		p := stepbaton.Path(dir, fmt.Sprintf("sess-%d", i))
		if err := os.WriteFile(p, []byte("{}\n"), 0o600); err != nil {
			b.Fatalf("write sidecar: %v", err)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := SweepStaleAdvice(dir, 24*time.Hour, now); err != nil {
			b.Fatalf("SweepStaleAdvice: %v", err)
		}
	}
}

func BenchmarkSweepStaleAdviceWithOrphans(b *testing.B) {
	dir := b.TempDir()
	now := time.Now()
	old := now.Add(-48 * time.Hour)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		for j := 0; j < 5; j++ {
			p := stepbaton.Path(dir, fmt.Sprintf("orphan-%d", j))
			if err := os.WriteFile(p, []byte("{}\n"), 0o600); err != nil {
				b.Fatalf("write orphan: %v", err)
			}
			if err := os.Chtimes(p, old, old); err != nil {
				b.Fatalf("age orphan: %v", err)
			}
		}
		b.StartTimer()
		n, err := SweepStaleAdvice(dir, 24*time.Hour, now)
		if err != nil || n != 5 {
			b.Fatalf("SweepStaleAdvice: n=%d err=%v", n, err)
		}
	}
}

func TestBenchmarkSanity(t *testing.T) {
	dir := t.TempDir()
	body := `{"schema":"fak-ctxvalue-report/1","budget_tokens":100000,"sessions":[` +
		`{"trace_id":"trace-sanity","tokens":{"resident_tokens":80000,"budget_tokens":100000},"turns":{"turns_observed":12},"session":{"phase":"crowding"},"step_advice":{"step_class":"checkpoint","basis":"token_headroom","reason":"80% used"}}` +
		`]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	ctx := context.Background()
	stamp, ok, err := CaptureAndWrite(ctx, Options{
		BaseURL:   srv.URL,
		Dir:       dir,
		SessionID: "sess-sanity",
		SHA:       "sha1",
		Client:    srv.Client(),
	})
	if err != nil || !ok {
		t.Fatalf("CaptureAndWrite sanity: ok=%v err=%v", ok, err)
	}
	if stamp.StepClass != stepbaton.StepCheckpoint {
		t.Errorf("StepClass = %q, want %q", stamp.StepClass, stepbaton.StepCheckpoint)
	}
	if err := ReapClosedAdvice(dir, "sess-sanity"); err != nil {
		t.Fatalf("ReapClosedAdvice: %v", err)
	}
	swept, err := SweepStaleAdvice(dir, 24*time.Hour, time.Now())
	if err != nil || swept != 0 {
		t.Fatalf("SweepStaleAdvice: swept=%d err=%v", swept, err)
	}
}
