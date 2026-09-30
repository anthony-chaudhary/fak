//go:build darwin && cgo

// Darwin+cgo host suite: CGO_ENABLED=1 go test ./internal/gateway -run '^TestDebugVarsDarwinRSS' -count=1.
// Included by go test ./internal/gateway and go test ./... on Darwin with cgo.
package gateway

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessres"
	"github.com/anthony-chaudhary/fak/internal/runtimeobs"
)

// fak-test:runtime fast est=200ms lane=default
func TestDebugVarsDarwinRSSMatchesCurrentResidentSet(t *testing.T) {
	reader := harnessres.DarwinSelfRSSReader()
	if reader == nil {
		t.Fatal("Darwin+cgo must expose the native current resident-RSS reader")
	}
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	// Warm the handler and HTTP connection before measuring their RSS overhead.
	for range 3 {
		getDebugVarsRaw(t, ts.URL)
	}

	var loggedReceipt bool
	readRoute := func() uint64 {
		t.Helper()
		before, beforeOK := reader()
		vars, _ := getDebugVarsRaw(t, ts.URL)
		after, afterOK := reader()
		rss := vars.Runtime.GoRuntime.Memory.ProcessRSSBytes
		if !beforeOK || !afterOK || before == 0 || after == 0 || !rss.OK || rss.V == 0 {
			t.Fatalf("current RSS unavailable: native before=%d/%v after=%d/%v route=%+v reasons=%+v", before, beforeOK, after, afterOK, rss, vars.Runtime.GoRuntime.Unavailable)
		}
		low, high := min(before, after), max(before, after)
		// The route sample lies between these native reads. A fixed 2MiB margin
		// allows warmed JSON/HTTP bookkeeping and page faults during the scrape;
		// it is small relative to the 32MiB resident-page witness below.
		const scrapeMargin = 2 << 20
		if (rss.V < low && low-rss.V > scrapeMargin) || (rss.V > high && rss.V-high > scrapeMargin) {
			t.Fatalf("route RSS=%d, want current native RSS bracket [%d,%d] +/- %d bytes", rss.V, low, high, scrapeMargin)
		}
		if !loggedReceipt {
			memoryJSON, err := json.Marshal(vars.Runtime.GoRuntime.Memory)
			if err != nil {
				t.Fatalf("marshal live RSS/heap receipt: %v", err)
			}
			t.Logf("live /debug/vars go_runtime memory receipt: %s", memoryJSON)
			loggedReceipt = true
		}
		return rss.V
	}
	before := readRoute()
	const mappingBytes = 32 << 20
	mapping, err := syscall.Mmap(-1, 0, mappingBytes, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		t.Fatalf("mmap current-RSS witness: %v", err)
	}
	t.Cleanup(func() {
		if mapping != nil {
			if err := syscall.Munmap(mapping); err != nil {
				t.Errorf("release current-RSS witness: %v", err)
			}
		}
	})
	for offset := 0; offset < len(mapping); offset += os.Getpagesize() {
		mapping[offset] = byte(offset/os.Getpagesize()%251 + 1)
	}
	during := readRoute()
	if during < before || during-before < mappingBytes/2 {
		t.Fatalf("touched %d bytes: route RSS before=%d during=%d, want growth >= %d", mappingBytes, before, during, mappingBytes/2)
	}
	if err := syscall.Munmap(mapping); err != nil {
		t.Fatalf("munmap current-RSS witness: %v", err)
	}
	mapping = nil
	after := readRoute()
	t.Logf("/debug/vars current resident RSS bytes: before=%d touched=%d reclaimed=%d", before, during, after)
	// Unlike current RSS, a high-water counter cannot fall after munmap. The
	// native brackets also pin this field to resident RSS, not phys_footprint.
	if after > during || during-after < mappingBytes/2 {
		t.Fatalf("released %d bytes: route RSS during=%d after=%d, want decline >= %d", mappingBytes, during, after, mappingBytes/2)
	}
}

const darwinRSSRouteFixtureEnv = "FAK_TEST_DARWIN_RSS_ROUTE_FIXTURE"

// fak-test:runtime fast est=300ms lane=default
func TestDebugVarsDarwinRSSUsesCurrentSampleAndNamesFailure(t *testing.T) {
	if os.Getenv(darwinRSSRouteFixtureEnv) != "1" {
		// The collector is process-global. A fresh test process keeps the injected
		// reader hermetic even if other route tests gain parallel execution later.
		ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestDebugVarsDarwinRSSUsesCurrentSampleAndNamesFailure$", "-test.count=1", "-test.v")
		cmd.Env = append(os.Environ(), darwinRSSRouteFixtureEnv+"=1")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("isolated RSS route fixture: %v\n%s", err, out)
		}
		t.Logf("isolated RSS route fixture:\n%s", out)
		return
	}

	var reads atomic.Uint32
	collector := runtimeobs.New(runtimeobs.WithRSSReader(func() (uint64, bool) {
		switch reads.Add(1) {
		case 1:
			return 64 << 20, true
		case 2:
			return 8 << 20, true
		default:
			// A failed reader's nonzero output must not leak into the receipt.
			return 12345, false
		}
	}))
	original := goRuntimeCollector
	goRuntimeCollector = func() *runtimeobs.Collector { return collector }
	t.Cleanup(func() { goRuntimeCollector = original })
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	first, _ := getDebugVarsRaw(t, ts.URL)
	second, _ := getDebugVarsRaw(t, ts.URL)
	for i, want := range []uint64{64 << 20, 8 << 20} {
		got := []runtimeobs.Receipt{first.Runtime.GoRuntime, second.Runtime.GoRuntime}[i].Memory.ProcessRSSBytes
		if !got.OK || got.V != want {
			t.Fatalf("scrape %d RSS=%+v, want the current reader sample %d", i+1, got, want)
		}
	}
	failed, raw := getDebugVarsRaw(t, ts.URL)
	if rss := failed.Runtime.GoRuntime.Memory.ProcessRSSBytes; rss.OK || rss.V != 0 {
		t.Fatalf("failed RSS=%+v, want an unavailable value without a stale sample", rss)
	}
	if !strings.Contains(raw, `"process_rss_bytes":null`) {
		t.Fatalf("failed RSS must marshal as null: %.400s", raw)
	}
	var named bool
	for _, unavailable := range failed.Runtime.GoRuntime.Unavailable {
		if unavailable.Field == "memory.process_rss_bytes" && unavailable.Reason == runtimeobs.ReasonReadFailed {
			named = true
		}
	}
	if !named {
		t.Fatalf("failed RSS must name read_failed: %+v", failed.Runtime.GoRuntime.Unavailable)
	}
	if got := reads.Load(); got != 3 {
		t.Fatalf("reader calls=%d, want one current sample per route scrape (3)", got)
	}
}
