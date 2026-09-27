package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/runtimeobs"
)

func getDebugVarsRaw(t *testing.T, url string) (debugVarsResponse, string) {
	t.Helper()
	r, err := http.Get(url + "/debug/vars")
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	body, err := io.ReadAll(r.Body)
	if err != nil {
		t.Fatal(err)
	}
	if r.StatusCode != http.StatusOK {
		t.Fatalf("/debug/vars status = %d", r.StatusCode)
	}
	var vars debugVarsResponse
	if err := json.Unmarshal(body, &vars); err != nil {
		t.Fatalf("decode /debug/vars: %v", err)
	}
	return vars, string(body)
}

// TestDebugVarsCarriesGoRuntimeReceipt proves the live /debug/vars route (not
// just the runtimeobs leaf) serves the typed #10182 receipt and that two
// scrapes are Diff-able within one process epoch.
func TestSelfRSSReaderPlatforms(t *testing.T) {
	for _, goos := range []string{"linux", "windows"} {
		if selfRSSReader(goos) == nil {
			t.Errorf("%s has a harnessres current-RSS reader; want it injected", goos)
		}
	}
	if selfRSSReader("darwin") != nil {
		t.Error("darwin has no current-RSS reader; want nil (no_reader), not a reader that always fails")
	}
}

func TestDebugVarsCarriesGoRuntimeReceipt(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	first, raw := getDebugVarsRaw(t, ts.URL)
	if !strings.Contains(raw, `"go_runtime":{"schema":"`+runtimeobs.Schema+`"`) {
		t.Fatalf("/debug/vars lacks the go_runtime receipt: %.400s", raw)
	}
	g := first.Runtime.GoRuntime
	if g.GoVersion != runtime.Version() || g.Epoch == "" {
		t.Fatalf("receipt header = %+v", g)
	}
	if !g.GC.Cycles.OK || !g.GC.HeapGoalBytes.OK || !g.CPU.GCTotalSeconds.OK || !g.Sched.Latencies.OK || !g.Sched.Threads.OK {
		t.Fatalf("core scheduler/GC axes unavailable: %+v", g.Unavailable)
	}
	if !g.Memory.HeapObjectsBytes.OK || g.Memory.HeapObjectsBytes.V > g.Memory.TotalMappedBytes.V {
		t.Fatalf("go heap %+v must be within total mapped %+v", g.Memory.HeapObjectsBytes, g.Memory.TotalMappedBytes)
	}
	// RSS is read where harnessres has a current-RSS reader; elsewhere it must
	// be an explicit no_reader refusal that marshals as null, never 0.
	rss := g.Memory.ProcessRSSBytes
	if selfRSSReader(runtime.GOOS) != nil {
		if !rss.OK || rss.V == 0 {
			t.Fatalf("%s RSS unavailable: %+v", runtime.GOOS, g.Unavailable)
		}
	} else {
		if rss.OK {
			t.Fatalf("%s has no RSS reader but reported %d", runtime.GOOS, rss.V)
		}
		if !strings.Contains(raw, `"process_rss_bytes":null`) {
			t.Fatalf("unreadable RSS must marshal as null: %.400s", raw)
		}
		named := false
		for _, u := range g.Unavailable {
			named = named || (u.Field == "memory.process_rss_bytes" && u.Reason == runtimeobs.ReasonNoReader)
		}
		if !named {
			t.Fatalf("RSS refusal not named as no_reader: %+v", g.Unavailable)
		}
	}
	if s := g.Sched.LatenciesSummary; !s.Count.OK || s.Count.V != g.Sched.Latencies.Count {
		t.Fatalf("scheduler latency summary = %+v, want the histogram's count %d", s, g.Sched.Latencies.Count)
	}

	second, _ := getDebugVarsRaw(t, ts.URL)
	d, err := runtimeobs.Diff(first.Runtime.GoRuntime, second.Runtime.GoRuntime)
	if err != nil {
		t.Fatalf("successive scrapes must diff within one epoch: %v", err)
	}
	if d.ToSeq <= d.FromSeq || !d.GCCycles.OK || !d.CPUTotalSeconds.OK {
		t.Fatalf("delta = %+v", d)
	}
}
