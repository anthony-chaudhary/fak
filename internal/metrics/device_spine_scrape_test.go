package metrics

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// peerServer stands up an httptest peer that answers GET with the RenderJSON
// bytes of the given snapshot — the real wire shape, not a hand-written
// fixture — and counts how many times it was scraped.
func peerServer(t *testing.T, snap []DeviceMetrics) (*httptest.Server, *atomic.Int64) {
	t.Helper()
	var hits atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		body, err := RenderJSON(snap)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

// localBuffer polls a Buffer once from a fake probe so the scrape tests have a
// real local snapshot to federate onto.
func localBuffer(t *testing.T) *Buffer {
	t.Helper()
	buf := NewBuffer()
	buf.Poll([]Probe{fakeProbe{
		backend: "nvml", detect: true,
		devices: []Device{fakeDevice{id: "gpu0", m: map[string]float64{"power_watts": 100}}},
	}})
	return buf
}

// TestScraperFederatesLivePeers is the end-to-end contract: a scraper pointed
// at two live HTTP peers publishes ONE pane-of-glass snapshot — local rows
// first and untouched, then each peer's rows in configured order, marked
// remote with their origin — and the existing renderers pick up the
// remote/peer labels for free, with no federation-specific render code.
func TestScraperFederatesLivePeers(t *testing.T) {
	peerA, _ := peerServer(t, []DeviceMetrics{
		{Backend: "nvml", DeviceID: "gpu0", TokensPerSecond: f(42)},
	})
	peerB, _ := peerServer(t, []DeviceMetrics{
		{Backend: "engine", DeviceID: "vllm0", QueueDepth: f(3)},
	})

	s := &Scraper{
		Local: localBuffer(t),
		Peers: []PeerEndpoint{
			{Name: "host-a", URL: peerA.URL},
			{Name: "host-b", URL: peerB.URL},
		},
	}

	got, err := s.ScrapeOnce(context.Background())
	if err != nil {
		t.Fatalf("ScrapeOnce with healthy peers: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("want local + 2 peer rows, got %d: %+v", len(got), got)
	}
	if got[0].Remote || got[0].Peer != "" || got[0].DeviceID != "gpu0" || got[0].Backend != "nvml" {
		t.Fatalf("local row should lead the pane unchanged, got %+v", got[0])
	}
	if v, ok := deref(got[0].PowerWatts); !ok || v != 100 {
		t.Fatalf("local metric lost in federation: %v %v", v, ok)
	}
	for i, want := range []struct{ peer, device string }{{"host-a", "gpu0"}, {"host-b", "vllm0"}} {
		row := got[i+1]
		if !row.Remote || row.Peer != want.peer || row.DeviceID != want.device {
			t.Fatalf("row %d: want remote %s/%s, got %+v", i+1, want.peer, want.device, row)
		}
	}

	// The published snapshot is what a reader sees.
	if snap := s.Snapshot(); len(snap) != len(got) {
		t.Fatalf("Snapshot should return the published pane: got %d rows, want %d", len(snap), len(got))
	}

	// Renderers get federation for free: no scrape-specific render path.
	prom, err := RenderProm(s.Snapshot())
	if err != nil {
		t.Fatalf("RenderProm: %v", err)
	}
	text := string(prom)
	for _, want := range []string{`peer="host-a"`, `peer="host-b"`, `remote="true"`} {
		if !strings.Contains(text, want) {
			t.Fatalf("Prom render missing %s:\n%s", want, text)
		}
	}
	if !strings.Contains(text, "fak_device_power_watts") {
		t.Fatalf("Prom render dropped the local family:\n%s", text)
	}
}

// TestScraperSkipsBadPeersNotFatal proves the round's failure contract: an
// unreachable peer, a non-200 peer, and a peer serving malformed JSON are each
// skipped and NAMED in the returned error, while the healthy peer and the
// local rows still publish. A bad peer degrades the pane, never blanks it.
func TestScraperSkipsBadPeersNotFatal(t *testing.T) {
	good, _ := peerServer(t, []DeviceMetrics{{Backend: "nvml", DeviceID: "gpu9", InFlight: f(2)}})

	broken := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	t.Cleanup(broken.Close)

	garbage := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("{not json"))
	}))
	t.Cleanup(garbage.Close)

	// A peer that is not listening at all: stand one up and close it.
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	deadURL := dead.URL
	dead.Close()

	s := &Scraper{
		Local: localBuffer(t),
		Peers: []PeerEndpoint{
			{Name: "sick", URL: broken.URL},
			{Name: "healthy", URL: good.URL},
			{Name: "garbled", URL: garbage.URL},
			{Name: "dead", URL: deadURL},
		},
		Client: &http.Client{Timeout: 2 * time.Second},
	}

	got, err := s.ScrapeOnce(context.Background())
	if err == nil {
		t.Fatalf("three bad peers should be reported, got nil error")
	}
	for _, name := range []string{"sick", "garbled", "dead"} {
		if !strings.Contains(err.Error(), name) {
			t.Fatalf("error should name skipped peer %q, got: %v", name, err)
		}
	}
	if len(got) != 2 {
		t.Fatalf("want local + healthy peer row, got %d: %+v", len(got), got)
	}
	if got[0].Remote || got[0].DeviceID != "gpu0" {
		t.Fatalf("local row lost when peers failed: %+v", got[0])
	}
	if !got[1].Remote || got[1].Peer != "healthy" || got[1].DeviceID != "gpu9" {
		t.Fatalf("healthy peer dropped alongside the bad ones: %+v", got[1])
	}
}

// TestScraperRunPollsOnInterval proves the LIVE half: Run scrapes immediately
// (no first-interval blind spot), keeps scraping on the interval, republishes
// each round, and returns only when the context is cancelled.
func TestScraperRunPollsOnInterval(t *testing.T) {
	peer, hits := peerServer(t, []DeviceMetrics{{Backend: "nvml", DeviceID: "gpu1", PowerWatts: f(7)}})

	s := &Scraper{
		Local:    localBuffer(t),
		Peers:    []PeerEndpoint{{Name: "host-a", URL: peer.URL}},
		Interval: time.Millisecond,
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()

	deadline := time.Now().Add(5 * time.Second)
	for hits.Load() < 3 {
		if time.Now().After(deadline) {
			cancel()
			t.Fatalf("Run scraped only %d times in 5s, want >= 3", hits.Load())
		}
		time.Sleep(time.Millisecond)
	}
	cancel()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Run should return ctx.Err() on cancel, got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("Run did not return after cancel")
	}

	snap := s.Snapshot()
	if len(snap) != 2 || !snap[1].Remote || snap[1].Peer != "host-a" {
		t.Fatalf("Run should keep publishing the federated pane, got %+v", snap)
	}
}

// TestScraperRunSurvivesBadPeer proves a failing peer never stops the loop: Run
// reports it through OnError every round and keeps polling until cancel.
func TestScraperRunSurvivesBadPeer(t *testing.T) {
	var reported atomic.Int64
	s := &Scraper{
		Local:    localBuffer(t),
		Peers:    []PeerEndpoint{{Name: "sick", URL: "http://127.0.0.1:0/metrics"}},
		Interval: time.Millisecond,
		Fetch: func(ctx context.Context, p PeerEndpoint) ([]byte, error) {
			return nil, errors.New("connection refused")
		},
		OnError: func(err error) {
			if strings.Contains(err.Error(), "sick") {
				reported.Add(1)
			}
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()

	deadline := time.Now().Add(5 * time.Second)
	for reported.Load() < 2 {
		if time.Now().After(deadline) {
			cancel()
			t.Fatalf("OnError fired %d times in 5s, want >= 2", reported.Load())
		}
		time.Sleep(time.Millisecond)
	}
	cancel()
	<-done

	// The local pane still publishes with every peer down.
	if snap := s.Snapshot(); len(snap) != 1 || snap[0].Remote {
		t.Fatalf("local-only pane should still publish, got %+v", snap)
	}
}

// TestScrapeOnceCancelledRoundKeepsPane proves an aborted round is not read as
// a measurement of an empty fleet: cancellation fails every in-flight peer at
// once, so the round must publish nothing and leave the standing pane intact
// rather than wiping every peer out of it on the way down.
func TestScrapeOnceCancelledRoundKeepsPane(t *testing.T) {
	peer, _ := peerServer(t, []DeviceMetrics{{Backend: "nvml", DeviceID: "gpu0", PowerWatts: f(7)}})
	s := &Scraper{Local: localBuffer(t), Peers: []PeerEndpoint{{Name: "host-a", URL: peer.URL}}}

	if _, err := s.ScrapeOnce(context.Background()); err != nil {
		t.Fatalf("warm-up scrape: %v", err)
	}
	if len(s.Snapshot()) != 2 {
		t.Fatalf("warm-up should publish local + peer, got %+v", s.Snapshot())
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	got, err := s.ScrapeOnce(ctx)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled round should report cancellation, got %v", err)
	}
	if len(got) != 2 || !got[1].Remote {
		t.Fatalf("cancelled round should return the standing pane, got %+v", got)
	}
	if snap := s.Snapshot(); len(snap) != 2 || !snap[1].Remote || snap[1].Peer != "host-a" {
		t.Fatalf("cancelled round clobbered the published pane: %+v", snap)
	}
}

// TestScraperUnnamedPeerFallsBackToURL proves an unnamed peer is still
// identifiable: its URL becomes the `peer` label and appears in a skip error.
func TestScraperUnnamedPeerFallsBackToURL(t *testing.T) {
	peer, _ := peerServer(t, []DeviceMetrics{{Backend: "nvml", DeviceID: "gpu0"}})
	s := &Scraper{Peers: []PeerEndpoint{{URL: peer.URL}}}

	got, err := s.ScrapeOnce(context.Background())
	if err != nil {
		t.Fatalf("ScrapeOnce: %v", err)
	}
	if len(got) != 1 || got[0].Peer != peer.URL {
		t.Fatalf("unnamed peer should be labelled by URL, got %+v", got)
	}
}

// TestScraperDoesNotAliasLocalBuffer proves publishing the federated pane
// cannot corrupt the local single-writer double buffer: the published slice is
// a fresh allocation, and a later local Poll neither sees nor carries forward
// the remote rows (seed-back-from-front stays local-only).
func TestScraperDoesNotAliasLocalBuffer(t *testing.T) {
	peer, _ := peerServer(t, []DeviceMetrics{{Backend: "engine", DeviceID: "vllm0", QueueDepth: f(1)}})
	buf := localBuffer(t)
	s := &Scraper{Local: buf, Peers: []PeerEndpoint{{Name: "host-a", URL: peer.URL}}}

	if _, err := s.ScrapeOnce(context.Background()); err != nil {
		t.Fatalf("ScrapeOnce: %v", err)
	}
	if local := buf.Snapshot(); len(local) != 1 {
		t.Fatalf("federation leaked into the local Buffer: %+v", local)
	}

	next := buf.Poll([]Probe{fakeProbe{
		backend: "nvml", detect: true,
		devices: []Device{fakeDevice{id: "gpu0", m: map[string]float64{"power_watts": 110}}},
	}})
	if len(next) != 1 {
		t.Fatalf("local Poll carried forward remote rows: %+v", next)
	}
	for _, row := range next {
		if row.Remote {
			t.Fatalf("remote row survived a local Poll: %+v", row)
		}
	}
}
