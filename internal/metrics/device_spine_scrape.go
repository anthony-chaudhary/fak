package metrics

// device_spine_scrape.go — the LIVE fleet-federation scrape loop for the
// device-telemetry spine (issue #4365, parent #3237, epic #3236). The reader
// half landed transport-agnostic (`ParsePeerSnapshot` + `Federate`,
// 88ca9533533a); this is the loop that actually drives it: poll each configured
// peer's RenderJSON endpoint on an interval, parse the bytes, federate them
// onto the local Buffer's snapshot, and publish one pane-of-glass snapshot that
// every renderer (Prom/CSV/JSON/text) already reads — so the `remote`/`peer`
// labels come for free, with no new render surface.
//
// Two contracts carry over from the spine and are load-bearing here:
//
//   - skip-not-fatal: one unreachable, non-200, oversized, or malformed peer
//     contributes nothing to this round and is NAMED in the returned error,
//     while every healthy peer still publishes. A scrape round has no failure
//     mode that blanks the pane.
//   - single-writer: the Scraper publishes into its OWN atomic pointer, not
//     into the local Buffer. Writing federated rows back into the Buffer would
//     both add a second writer to a single-writer double buffer AND poison the
//     next local Poll — seed-back-from-front would see the remote rows as
//     local devices that "vanished" and carry them forward forever. Local Poll
//     owns Buffer.front; the loop owns Scraper.front.
//
// Generation: gen/next. This loop IS the promotion evidence the spine named
// ("wire a live scraper loop that polls peer /metrics endpoints and feeds
// Federate"). Demotion evidence: if the fleet never runs a second host, the
// scrape loop and the reader under it are dead weight — retire both. An
// invalidating assumption: a peer is assumed to answer a plain GET with a
// RenderJSON-shaped body; a peer exposing only Prometheus TEXT needs a Prom
// parser, and a peer behind auth needs a Fetch hook rather than the default
// client. A second: a failed peer drops out of the pane for that round rather
// than holding last-good the way local metrics do, so a flapping peer blinks;
// per-peer last-good is a follow-on, not this loop's contract.

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync/atomic"
	"time"
)

// DefaultScrapeInterval is the poll period used when a Scraper leaves Interval
// unset — slow enough to be cheap on a fleet, fast enough that a pane-of-glass
// readout is not stale.
const DefaultScrapeInterval = 15 * time.Second

// defaultScrapeTimeout bounds one peer fetch when the caller supplies no
// client, so a hung peer cannot stall the whole round.
const defaultScrapeTimeout = 5 * time.Second

// maxPeerSnapshotBytes caps how much of a peer's response is read. A peer that
// answers with an endless or absurd body is a skipped peer, not an OOM.
const maxPeerSnapshotBytes = 8 << 20

// PeerEndpoint is one configured peer of the fleet pane-of-glass: the name that
// becomes the `peer` label on every row scraped from it (unless the row already
// carries its own origin — multi-hop federation keeps that), and the URL of its
// RenderJSON-shaped snapshot endpoint.
type PeerEndpoint struct {
	Name string
	URL  string
}

// FetchFunc fetches one peer's RenderJSON bytes. The seam keeps the loop as
// transport-agnostic as the reader: the default is an HTTP GET, and a caller
// that speaks a unix socket, ssh, a file drop, or an authenticated endpoint
// substitutes its own func without touching the federation path.
type FetchFunc func(ctx context.Context, peer PeerEndpoint) ([]byte, error)

// Scraper polls the configured peers on an interval and publishes the local
// snapshot federated with theirs. The zero value is not useful on its own —
// set Local and Peers — but every other field has a working default.
type Scraper struct {
	// Local is the local device Buffer whose snapshot leads the pane. A nil
	// Local federates peers alone (a collector-only host).
	Local *Buffer
	// Peers are the configured peers, scraped in order; the published snapshot
	// preserves that order, so the pane is deterministic.
	Peers []PeerEndpoint
	// Interval is the poll period for Run; <= 0 means DefaultScrapeInterval.
	Interval time.Duration
	// Fetch overrides the transport; nil means an HTTP GET via Client.
	Fetch FetchFunc
	// Client is the HTTP client for the default transport; nil means a client
	// bounded by defaultScrapeTimeout.
	Client *http.Client
	// OnError, when set, is called with each round's joined peer error so a
	// live loop can log which peers are down. Run otherwise drops the error,
	// because a bad peer must never stop the loop.
	OnError func(error)

	front atomic.Pointer[[]DeviceMetrics]
}

// Snapshot returns the last published federated snapshot, or nil before the
// first scrape. As with Buffer.Snapshot the slice is read-only: it is the same
// slice handed to every reader, and it is never mutated after publication.
func (s *Scraper) Snapshot() []DeviceMetrics {
	p := s.front.Load()
	if p == nil {
		return nil
	}
	return *p
}

// ScrapeOnce runs exactly one round: read the local snapshot, fetch and parse
// every peer, federate the healthy ones onto the local rows, and publish. It
// returns the published snapshot along with the joined errors of the peers it
// skipped — a non-nil error is a partial pane, never a missing one, so callers
// should use the snapshot and report the error rather than choosing between
// them.
//
// A round whose context is cancelled publishes NOTHING and returns the pane
// still standing. Cancellation fails every in-flight peer at once, so
// publishing that round would read as "the fleet went away" and wipe every
// peer from the pane at the exact moment a reader is shutting down — an
// aborted measurement is not a measurement of an empty fleet.
func (s *Scraper) ScrapeOnce(ctx context.Context) ([]DeviceMetrics, error) {
	var local []DeviceMetrics
	if s.Local != nil {
		local = s.Local.Snapshot()
	}
	peers := make([][]DeviceMetrics, 0, len(s.Peers))
	var errs []error
	for _, p := range s.Peers {
		rows, err := s.scrapePeer(ctx, p)
		if err != nil {
			errs = append(errs, err) // skip-not-fatal; the peer is named in err
			continue
		}
		peers = append(peers, rows)
	}
	if err := ctx.Err(); err != nil {
		return s.Snapshot(), errors.Join(append(errs, err)...)
	}
	// Federate always returns a fresh slice, so the published snapshot never
	// aliases the local Buffer's — publishing here cannot tear a local reader.
	published := Federate(local, peers...)
	s.front.Store(&published)
	return published, errors.Join(errs...)
}

// Run scrapes immediately and then every Interval until ctx is done, returning
// ctx.Err(). Peer failures never stop the loop — they are reported through
// OnError and the round still publishes whatever was healthy.
func (s *Scraper) Run(ctx context.Context) error {
	interval := s.Interval
	if interval <= 0 {
		interval = DefaultScrapeInterval
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		_, err := s.ScrapeOnce(ctx)
		if ctx.Err() != nil {
			// Cancelled mid-round: the round published nothing, and its
			// errors are shutdown noise, not peer health.
			return ctx.Err()
		}
		if err != nil && s.OnError != nil {
			s.OnError(err)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

// scrapePeer fetches and parses one peer. Every error path names the peer, so
// the caller's skip is diagnosable from the joined error alone.
func (s *Scraper) scrapePeer(ctx context.Context, p PeerEndpoint) ([]DeviceMetrics, error) {
	name := peerName(p)
	fetch := s.Fetch
	if fetch == nil {
		fetch = s.httpFetch
	}
	data, err := fetch(ctx, p)
	if err != nil {
		return nil, fmt.Errorf("peer %s scrape: %w", name, err)
	}
	// ParsePeerSnapshot already names the peer on a malformed body.
	return ParsePeerSnapshot(data, name)
}

// httpFetch is the default transport: a plain GET of the peer's snapshot
// endpoint, bounded in both time and body size.
func (s *Scraper) httpFetch(ctx context.Context, p PeerEndpoint) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.URL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	client := s.Client
	if client == nil {
		client = &http.Client{Timeout: defaultScrapeTimeout}
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("http status %s", resp.Status)
	}
	return io.ReadAll(io.LimitReader(resp.Body, maxPeerSnapshotBytes))
}

// peerName falls back to the URL so an unnamed peer is still identifiable in
// both the `peer` label and a skip error.
func peerName(p PeerEndpoint) string {
	if p.Name != "" {
		return p.Name
	}
	return p.URL
}
