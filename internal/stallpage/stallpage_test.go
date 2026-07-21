package stallpage

import (
	"bufio"
	"encoding/json"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
	"github.com/anthony-chaudhary/fak/internal/stallscan"
)

func pageSample(handles, threads int) stallscan.Sample {
	return stallscan.Sample{
		TopHandles: []stallscan.ProcHandles{{PID: 42, Name: "WindowsTerminal.exe", Handles: handles}},
		TopThreads: []stallscan.ProcThreads{{PID: 42, Name: "WindowsTerminal.exe", Threads: threads}},
	}
}

func TestFromSampleCrossingEarnsHumanOperatorPage(t *testing.T) {
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	page, ok := FromSample(pageSample(30_001, 100), now)
	if !ok {
		t.Fatal("handle high-water crossing produced no page")
	}
	if page.Advice.Axis != "handles" || page.Advice.Threshold != 30_000 {
		t.Fatalf("advice = %+v, want unchanged 30000-handle high-water", page.Advice)
	}
	if page.Triage.Disposition != choicetriage.HumanResidual || !page.Triage.NeedsHuman {
		t.Fatalf("triage = %+v, want earned HUMAN_RESIDUAL", page.Triage)
	}
	if !strings.Contains(page.Advice.Reason, "reboot the host before it freezes") {
		t.Fatalf("page dropped AdviseReboot reason: %+v", page)
	}
	if !strings.Contains(page.Action, "do not reboot or kill automatically") {
		t.Fatalf("page action lost read-only safety boundary: %q", page.Action)
	}
}

func TestFromSampleBelowHighWaterIsSilent(t *testing.T) {
	if page, ok := FromSample(pageSample(29_999, 1_999), time.Now()); ok {
		t.Fatalf("below-line sample must not page, got %+v", page)
	}
}

func TestDedupKeyIgnoresPIDButKeepsAxisAndProcess(t *testing.T) {
	a := stallscan.RebootAdvice{Axis: "handle_high_water", Process: "WindowsTerminal.exe", PID: 10}
	b := stallscan.RebootAdvice{Axis: "HANDLE_HIGH_WATER", Process: " windowsterminal.EXE ", PID: 99}
	if got, want := DedupKey(a), DedupKey(b); got != want {
		t.Fatalf("restart changed dedup key: %q != %q", got, want)
	}
	c := stallscan.RebootAdvice{Axis: "thread_high_water", Process: a.Process, PID: a.PID}
	if DedupKey(c) == DedupKey(a) {
		t.Fatal("different high-water axes collapsed to one dedup key")
	}
}

func TestPublishDedupesByAxisAndProcessWithinWindow(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	first, err := Publish(dir, pageSample(31_000, 100), now, DefaultDedupWindow)
	if err != nil || !first.Published {
		t.Fatalf("first Publish = %+v, %v; want published", first, err)
	}
	repeat, err := Publish(dir, pageSample(32_000, 100), now.Add(time.Minute), DefaultDedupWindow)
	if err != nil || repeat.Published {
		t.Fatalf("repeat Publish = %+v, %v; want deduped", repeat, err)
	}
	if repeat.Page.Key != first.Page.Key {
		t.Fatalf("same axis/process changed dedup key: %q != %q", repeat.Page.Key, first.Page.Key)
	}

	pages := readPages(t, PageLogPath(dir))
	if len(pages) != 1 {
		t.Fatalf("page ledger has %d rows, want exactly 1", len(pages))
	}
	if pages[0].Advice.Count != 31_000 {
		t.Fatalf("dedup overwrote first crossing: %+v", pages[0].Advice)
	}

	reminder, err := Publish(dir, pageSample(33_000, 100), now.Add(DefaultDedupWindow+time.Second), DefaultDedupWindow)
	if err != nil || !reminder.Published {
		t.Fatalf("post-window Publish = %+v, %v; want one bounded reminder", reminder, err)
	}
	if got := len(readPages(t, PageLogPath(dir))); got != 2 {
		t.Fatalf("page ledger after reminder has %d rows, want 2", got)
	}
}

func TestPublishBelowLineCreatesNoArtifacts(t *testing.T) {
	dir := t.TempDir()
	got, err := Publish(dir, pageSample(10_000, 500), time.Now(), DefaultDedupWindow)
	if err != nil || got.Published {
		t.Fatalf("below-line Publish = %+v, %v; want silent no-op", got, err)
	}
	if _, err := os.Stat(PageLogPath(dir)); !os.IsNotExist(err) {
		t.Fatalf("below-line sample created page ledger: %v", err)
	}
	if _, err := os.Stat(StatePath(dir)); !os.IsNotExist(err) {
		t.Fatalf("below-line sample created state: %v", err)
	}
}

func TestPublishConcurrentCrossingEmitsExactlyOnce(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	var published atomic.Int32
	var wg sync.WaitGroup
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := Publish(dir, pageSample(31_000, 100), now, DefaultDedupWindow)
			if err != nil {
				t.Errorf("Publish: %v", err)
				return
			}
			if got.Published {
				published.Add(1)
			}
		}()
	}
	wg.Wait()
	if got := published.Load(); got != 1 {
		t.Fatalf("concurrent publishers won %d times, want exactly 1", got)
	}
	if got := len(readPages(t, PageLogPath(dir))); got != 1 {
		t.Fatalf("concurrent page ledger rows = %d, want 1", got)
	}
}

func readPages(t *testing.T, path string) []Page {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open page ledger: %v", err)
	}
	defer f.Close()
	var out []Page
	s := bufio.NewScanner(f)
	for s.Scan() {
		var p Page
		if err := json.Unmarshal(s.Bytes(), &p); err != nil {
			t.Fatalf("decode page: %v", err)
		}
		out = append(out, p)
	}
	if err := s.Err(); err != nil {
		t.Fatalf("scan page ledger: %v", err)
	}
	return out
}
