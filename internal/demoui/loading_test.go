package demoui

import (
	"bytes"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestFormatDuration(t *testing.T) {
	tests := []struct {
		d    time.Duration
		want string
	}{
		{d: -5 * time.Second, want: "0.0s"},
		{d: 0, want: "0.0s"},
		{d: 450 * time.Millisecond, want: "0.5s"},
		{d: 4200 * time.Millisecond, want: "4.2s"},
		{d: 59900 * time.Millisecond, want: "59.9s"},
		{d: 60 * time.Second, want: "1m00s"},
		{d: 75 * time.Second, want: "1m15s"},
		{d: 125 * time.Second, want: "2m05s"},
		{d: 3665 * time.Second, want: "1h01m05s"},
	}

	for _, tc := range tests {
		got := formatDuration(tc.d)
		if got != tc.want {
			t.Errorf("formatDuration(%v) = %q, want %q", tc.d, got, tc.want)
		}
	}
}

func TestRenderProgressBar(t *testing.T) {
	// Determinate tests
	determinateTests := []struct {
		current int64
		total   int64
		barLen  int
		wantPct string
	}{
		{current: 0, total: 100, barLen: 10, wantPct: "  0%"},
		{current: 50, total: 100, barLen: 10, wantPct: " 50%"},
		{current: 100, total: 100, barLen: 10, wantPct: "100%"},
		{current: 150, total: 100, barLen: 10, wantPct: "100%"},
		{current: -10, total: 100, barLen: 10, wantPct: "  0%"},
	}

	for _, tc := range determinateTests {
		got := renderProgressBar(tc.current, tc.total, tc.barLen, 0)
		if !strings.Contains(got, tc.wantPct) {
			t.Errorf("renderProgressBar(%d, %d, %d, 0) = %q, want pct %q", tc.current, tc.total, tc.barLen, got, tc.wantPct)
		}
	}

	// 50% on 10 runes should be 5 filled and 5 empty
	bar50 := renderProgressBar(50, 100, 10, 0)
	want50 := "[" + strings.Repeat(FilledGlyph, 5) + strings.Repeat(EmptyGlyph, 5) + "]  50%"
	if bar50 != want50 {
		t.Errorf("renderProgressBar(50, 100, 10, 0) = %q, want %q", bar50, want50)
	}

	// 100% on 10 runes should be 10 filled
	bar100 := renderProgressBar(100, 100, 10, 0)
	want100 := "[" + strings.Repeat(FilledGlyph, 10) + "] 100%"
	if bar100 != want100 {
		t.Errorf("renderProgressBar(100, 100, 10, 0) = %q, want %q", bar100, want100)
	}

	// Indeterminate tests
	barIndet0 := renderProgressBar(0, 0, 10, 0)
	wantIndet0 := "[" + strings.Repeat(FilledGlyph, 4) + strings.Repeat(EmptyGlyph, 6) + "]"
	if barIndet0 != wantIndet0 {
		t.Errorf("renderProgressBar(0, 0, 10, 0) = %q, want %q", barIndet0, wantIndet0)
	}

	barIndet2 := renderProgressBar(0, 0, 10, 2)
	wantIndet2 := "[" + strings.Repeat(EmptyGlyph, 2) + strings.Repeat(FilledGlyph, 4) + strings.Repeat(EmptyGlyph, 4) + "]"
	if barIndet2 != wantIndet2 {
		t.Errorf("renderProgressBar(0, 0, 10, 2) = %q, want %q", barIndet2, wantIndet2)
	}

	// Zero bar length
	if got := renderProgressBar(50, 100, 0, 0); got != "" {
		t.Errorf("renderProgressBar with 0 barLen = %q, want empty string", got)
	}
}

func TestCalculateETA(t *testing.T) {
	// Determinate
	if got := calculateETA(0, 100, 5*time.Second, 0); got != "ETA --" {
		t.Errorf("calculateETA(0, 100) = %q, want ETA --", got)
	}
	if got := calculateETA(100, 100, 5*time.Second, 0); got != "ETA 0s" {
		t.Errorf("calculateETA(100, 100) = %q, want ETA 0s", got)
	}
	// 50 done out of 100 in 5s => rate is 10/s => 50 remaining => 5.0s
	if got := calculateETA(50, 100, 5*time.Second, 0); got != "ETA ~5.0s" {
		t.Errorf("calculateETA(50, 100, 5s) = %q, want ETA ~5.0s", got)
	}

	// Indeterminate
	if got := calculateETA(0, 0, 3*time.Second, 10*time.Second); got != "ETA ~7.0s" {
		t.Errorf("calculateETA(0, 0, 3s, 10s) = %q, want ETA ~7.0s", got)
	}
	if got := calculateETA(0, 0, 12*time.Second, 10*time.Second); got != "ETA ~imminent" {
		t.Errorf("calculateETA(0, 0, 12s, 10s) = %q, want ETA ~imminent", got)
	}
	if got := calculateETA(0, 0, 3*time.Second, 0); got != "" {
		t.Errorf("calculateETA(0, 0, 3s, 0) = %q, want empty", got)
	}
}

func TestFormatLoadingLine(t *testing.T) {
	line := formatLoadingLine(
		'⠋',
		"Loading model",
		50,
		100,
		"tensors",
		"",
		10,
		0,
		5*time.Second,
		0,
		false,
		false,
		false,
	)

	if !strings.Contains(line, "⠋") {
		t.Errorf("line missing glyph: %q", line)
	}
	if !strings.Contains(line, "Loading model") {
		t.Errorf("line missing label: %q", line)
	}
	if !strings.Contains(line, "50%") {
		t.Errorf("line missing percentage: %q", line)
	}
	if !strings.Contains(line, "5.0s") {
		t.Errorf("line missing timer: %q", line)
	}
	if !strings.Contains(line, "ETA ~5.0s") {
		t.Errorf("line missing ETA: %q", line)
	}
	if !strings.Contains(line, "50/100 tensors") {
		t.Errorf("line missing tensor counts: %q", line)
	}
}

func TestLoadingDisplay_Determinate(t *testing.T) {
	var buf bytes.Buffer
	disp := StartLoadingDisplay(LoadingOptions{
		Writer:    &buf,
		Label:     "Loading model (qwen38)",
		Total:     100,
		Current:   10,
		Unit:      "tensors",
		BarLength: 12,
		Cadence:   20 * time.Millisecond,
	})

	time.Sleep(60 * time.Millisecond)
	disp.Update(60, 100)
	time.Sleep(60 * time.Millisecond)

	disp.Done("✓ Loaded qwen38 (0.2s)")

	out := buf.String()
	if !strings.Contains(out, "Loading model (qwen38)") {
		t.Errorf("output missing label: %q", out)
	}
	if !strings.Contains(out, "✓ Loaded qwen38 (0.2s)\n") {
		t.Errorf("output missing completion message: %q", out)
	}
	if !strings.Contains(out, "\r") {
		t.Errorf("output missing carriage return: %q", out)
	}
	if disp.Elapsed() <= 0 {
		t.Errorf("elapsed duration should be > 0, got %v", disp.Elapsed())
	}
}

func TestLoadingDisplay_Indeterminate(t *testing.T) {
	var buf bytes.Buffer
	disp := StartLoadingDisplay(LoadingOptions{
		Writer:            &buf,
		Label:             "Loading model",
		EstimatedDuration: 5 * time.Second,
		Cadence:           20 * time.Millisecond,
	})

	time.Sleep(80 * time.Millisecond)
	disp.SetDetail("reading weights")
	time.Sleep(40 * time.Millisecond)
	disp.Stop()

	out := buf.String()
	if !strings.Contains(out, "Loading model") {
		t.Errorf("output missing label: %q", out)
	}
	if !strings.Contains(out, "\r") {
		t.Errorf("output missing carriage return: %q", out)
	}
}

func TestLoadingDisplay_Quiet(t *testing.T) {
	var buf bytes.Buffer
	disp := StartLoadingDisplay(LoadingOptions{
		Writer:  &buf,
		Label:   "Loading model",
		Quiet:   true,
		Cadence: 10 * time.Millisecond,
	})

	time.Sleep(50 * time.Millisecond)
	disp.Update(50, 100)
	disp.Done("done")

	if buf.Len() != 0 {
		t.Errorf("quiet mode wrote %d bytes, want 0", buf.Len())
	}
}

func TestLoadingDisplay_ParseProgressLine(t *testing.T) {
	disp := &LoadingDisplay{
		total: 100,
	}

	line1 := "fak: loading model 40% (40/100 tensors, 1.2 GB, 2s elapsed, 0.60 GB/s)"
	if !disp.ParseProgressLine(line1) {
		t.Fatalf("ParseProgressLine failed to parse line: %q", line1)
	}
	if disp.current != 40 || disp.total != 100 {
		t.Errorf("got current=%d total=%d, want 40/100", disp.current, disp.total)
	}
	if disp.detail != "0.60 GB/s" {
		t.Errorf("got detail=%q, want 0.60 GB/s", disp.detail)
	}
}

func TestLoadingDisplay_Concurrency(t *testing.T) {
	var buf bytes.Buffer
	disp := StartLoadingDisplay(LoadingOptions{
		Writer:    &buf,
		Label:     "Loading model",
		Total:     1000,
		BarLength: 16,
		Cadence:   10 * time.Millisecond,
	})

	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 50; j++ {
				disp.Tick(1)
				disp.SetDetail("active")
				_ = disp.Elapsed()
				time.Sleep(time.Millisecond)
			}
		}()
	}
	wg.Wait()
	disp.Stop()
}

func TestLoadingDisplay_Options(t *testing.T) {
	var buf bytes.Buffer
	disp := StartLoadingDisplay(LoadingOptions{
		Writer:      &buf,
		Label:       "Minimal",
		Total:       100,
		Current:     50,
		HideTimer:   true,
		HideETA:     true,
		HideGraphic: true,
		Cadence:     15 * time.Millisecond,
	})

	time.Sleep(50 * time.Millisecond)
	disp.Stop()

	out := buf.String()
	if !strings.Contains(out, "Minimal") {
		t.Errorf("expected label Minimal in %q", out)
	}
	if strings.Contains(out, "[") || strings.Contains(out, "]") {
		t.Errorf("HideGraphic failed, found brackets in %q", out)
	}
	if strings.Contains(out, "ETA") {
		t.Errorf("HideETA failed, found ETA in %q", out)
	}
}

func TestLoadingDisplay_DynamicTransition(t *testing.T) {
	var buf bytes.Buffer
	disp := StartLoadingDisplay(LoadingOptions{
		Writer:  &buf,
		Label:   "Dynamic",
		Cadence: 15 * time.Millisecond,
	})

	time.Sleep(30 * time.Millisecond)
	disp.SetLabel("Transitioning")
	disp.SetTotal(200)
	disp.SetCurrent(100)
	time.Sleep(40 * time.Millisecond)

	disp.Donef("✓ Done %s", "all")
	out := buf.String()
	if !strings.Contains(out, "Transitioning") {
		t.Errorf("expected updated label in %q", out)
	}
	if !strings.Contains(out, " 50%") {
		t.Errorf("expected 50%% in %q", out)
	}
	if !strings.Contains(out, "✓ Done all\n") {
		t.Errorf("expected Donef output in %q", out)
	}
}

func TestLoadingDisplay_ProgressWriterPipe(t *testing.T) {
	var buf bytes.Buffer
	disp := NewModelLoadingProgress(&buf, "Testing Pipe", 50)
	defer disp.Stop()

	pw := disp.ProgressWriter()
	_, _ = pw.Write([]byte("fak: loading model 60% (30/50 tensors, 500 MB, 1s elapsed, 500 MB/s)\n"))
	time.Sleep(50 * time.Millisecond)

	disp.mu.Lock()
	cur := disp.current
	tot := disp.total
	detail := disp.detail
	disp.mu.Unlock()

	if cur != 30 || tot != 50 {
		t.Errorf("expected progress 30/50 from pipe, got %d/%d", cur, tot)
	}
	if detail != "500 MB/s" {
		t.Errorf("expected detail 500 MB/s from pipe, got %q", detail)
	}
}

func TestLoadingDisplay_IdempotentStopAndDone(t *testing.T) {
	var buf bytes.Buffer
	disp := StartLoadingDisplay(LoadingOptions{
		Writer:  &buf,
		Label:   "Idempotent",
		Cadence: 15 * time.Millisecond,
	})

	time.Sleep(30 * time.Millisecond)
	disp.Stop()
	disp.Stop()                   // second stop must be safe
	disp.Done("should not crash") // done after stop must be safe
}

func TestModelLoadingSpinner_Helper(t *testing.T) {
	var buf bytes.Buffer
	stop := ModelLoadingSpinner(&buf, "Spinner Helper")
	time.Sleep(40 * time.Millisecond)
	stop()
	stop()

	out := buf.String()
	if !strings.Contains(out, "Spinner Helper") {
		t.Errorf("expected label in %q", out)
	}
}
