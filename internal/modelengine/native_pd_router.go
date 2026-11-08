package modelengine

// native_pd_router.go — the P/D gateway peer router.
//
// The prefill role fronts N decode peers instead of one: every handoff first
// reads each peer's active-imported-lane counter from its control plane
// /metrics, selects by the routing key (fewest active imported lanes, tie ->
// lowest peer index), ships, and on handoff failure retries ONCE on a
// different peer. The second failure refuses with ErrPDHandoffFailed, which
// the front door surfaces as the typed 502 handoff_failed. Every admission
// step records its per-peer lane counts in the route ledger — the
// admission-skew witness the disaggregation campaign reads.
//
// Evidence scope: [SW-VERIFIED] loopback/TCP. Nothing here claims multi-GPU or
// hardware performance.

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// PDDecodePeer identifies one remote decode role reachable from the prefill
// role: its wire data-plane address plus the optional HTTP control plane the
// router scrapes for the active-imported-lane counter. A peer without a
// control plane keeps a permanently unknown count and sorts after every
// readable peer.
type PDDecodePeer struct {
	WireAddr    string
	ControlAddr string // host:port; "" = no control plane
}

// PDPeerLaneFetcher reports one decode peer's active imported-lane count from
// its control plane. The shipped fetcher scrapes /metrics; tests inject a
// deterministic stand-in. A non-nil error marks the count UNKNOWN for this
// step: the router never refuses a handoff because a scrape failed.
type PDPeerLaneFetcher func(ctx context.Context, controlAddr string) (int, error)

// ErrPDPeerNoControlPlane marks a peer configured without a control-plane
// address: its lane count is permanently unknown. Callers may use errors.Is.
var ErrPDPeerNoControlPlane = errors.New("modelengine: P/D peer has no control-plane address")

// DefaultPDPeerLaneFetcher scrapes http://<controlAddr>/metrics and sums the
// decode-role active-lane counters out of the exposition. timeout bounds each
// scrape; zero takes the two-second default.
func DefaultPDPeerLaneFetcher(timeout time.Duration) PDPeerLaneFetcher {
	if timeout <= 0 {
		timeout = 2 * time.Second
	}
	client := &http.Client{Timeout: timeout}
	return func(ctx context.Context, controlAddr string) (int, error) {
		if controlAddr == "" {
			return 0, ErrPDPeerNoControlPlane
		}
		url := "http://" + controlAddr + "/metrics"
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return 0, err
		}
		resp, err := client.Do(req)
		if err != nil {
			return 0, err
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			return 0, fmt.Errorf("modelengine: P/D peer metrics scrape %s: status %d", controlAddr, resp.StatusCode)
		}
		body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		if err != nil {
			return 0, err
		}
		return ParseActiveLanes(string(body)), nil
	}
}

// pdActiveLanesMetric is the exposition prefix the decode control plane emits
// per worker; ParseActiveLanes sums its decode-role values.
const pdActiveLanesMetric = "fak_native_pd_active_lanes{"

// ParseActiveLanes sums the decode-role active-lane gauges in one Prometheus
// text exposition. Lines that do not carry a well-formed
// fak_native_pd_active_lanes{...role="decode"...} <int> line are ignored, so a
// partial or foreign exposition degrades to an undercount, never a panic.
func ParseActiveLanes(exposition string) int {
	total := 0
	for _, line := range strings.Split(exposition, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, pdActiveLanesMetric) {
			continue
		}
		rest := line[len(pdActiveLanesMetric):]
		closeIdx := strings.IndexByte(rest, '}')
		if closeIdx < 0 {
			continue
		}
		if !strings.Contains(rest[:closeIdx], `role="decode"`) {
			continue
		}
		value, err := strconv.Atoi(strings.TrimSpace(rest[closeIdx+1:]))
		if err != nil || value < 0 {
			continue
		}
		total += value
	}
	return total
}

// ErrPDHandoffFailed marks a P/D handoff that exhausted its bounded retry
// budget on every attempted decode peer. Callers may use errors.Is; the front
// door maps it to the typed 502 handoff_failed.
var ErrPDHandoffFailed = errors.New("modelengine: P/D handoff failed on every attempted decode peer")

// PDHandoffAttempt records one peer attempt inside a bounded-retry handoff.
type PDHandoffAttempt struct {
	PeerIndex int
	LaneCount int // -1 when unknown at selection time
	Err       error
}

// PDHandoffFailure is the typed refusal minted after the retry budget is
// spent; it carries every attempt so the operator can see which peers were
// tried and why each lost. It matches ErrPDHandoffFailed via errors.Is.
type PDHandoffFailure struct {
	Attempts []PDHandoffAttempt
}

func (e *PDHandoffFailure) Error() string {
	parts := make([]string, len(e.Attempts))
	for i, a := range e.Attempts {
		detail := "no attempt recorded"
		if a.Err != nil {
			detail = a.Err.Error()
		}
		parts[i] = fmt.Sprintf("peer %d: %s", a.PeerIndex, detail)
	}
	return fmt.Sprintf("%s (%s)", ErrPDHandoffFailed.Error(), strings.Join(parts, "; "))
}

// Unwrap matches every PDHandoffFailure against ErrPDHandoffFailed so callers
// can use errors.Is on the sentinel.
func (e *PDHandoffFailure) Unwrap() error { return ErrPDHandoffFailed }

// PDRouteReceipt is one admission step's routing witness: the per-peer lane
// counts observed at selection time, the chosen peer, and the retry outcome.
// The router's ledger of these receipts is the admission-skew record: the
// campaign's ADMISSION-SKEW class reads max/min spread across peers over a C
// step from exactly these rows.
type PDRouteReceipt struct {
	Step       uint64
	LaneCounts []int // per peer index; -1 unknown
	Chosen     int   // first-attempt peer index
	RetryTo    int   // retry target index; -1 unless Retried
	ServedBy   int   // peer that served; -1 when no peer served
}

// pdRouterLedgerCap bounds the retained route receipts; older steps drop off
// the tail once the cap is exceeded.
const pdRouterLedgerCap = 1024

// PDPeerRouter selects a decode peer per handoff and owns the bounded retry.
// It implements both PDHandoffTransport and PDHandoffTransportStream, so the
// front door swaps a single-address PDWireClient for the router without
// touching the chat handler. Configure it once and do not mutate it while it
// serves.
type PDPeerRouter struct {
	cluster *NativePDCluster
	peers   []PDDecodePeer
	clients []*PDWireClient
	fetcher PDPeerLaneFetcher

	scrapeTimeout time.Duration
	dialTimeout   time.Duration

	mu             sync.Mutex
	step           uint64
	routes         int64
	retries        int64
	failures       int64
	lastLaneCounts []int // per peer; -1 unknown
	ledger         []PDRouteReceipt
}

// NewPDPeerRouter builds the gateway router over cluster toward the given
// decode peers, in peer-index order. fetcher supplies the lane counters; nil
// takes the shipped /metrics scraper. peers must be non-empty with valid
// wire addresses; the serve verb validates that surface before construction.
func NewPDPeerRouter(cluster *NativePDCluster, peers []PDDecodePeer, fetcher PDPeerLaneFetcher) *PDPeerRouter {
	if fetcher == nil {
		fetcher = DefaultPDPeerLaneFetcher(0)
	}
	r := &PDPeerRouter{
		cluster:        cluster,
		peers:          append([]PDDecodePeer(nil), peers...),
		fetcher:        fetcher,
		scrapeTimeout:  2 * time.Second,
		dialTimeout:    10 * time.Second,
		lastLaneCounts: make([]int, len(peers)),
	}
	for i := range r.lastLaneCounts {
		r.lastLaneCounts[i] = -1
	}
	r.clients = make([]*PDWireClient, len(peers))
	for i, p := range peers {
		r.clients[i] = NewPDWireClient(cluster, p.WireAddr, r.dialTimeout)
	}
	return r
}

// Peers reports the configured decode peers in peer-index order.
func (r *PDPeerRouter) Peers() []PDDecodePeer {
	if r == nil {
		return nil
	}
	return append([]PDDecodePeer(nil), r.peers...)
}

// selectPeerOrder orders peer indices by the routing key: fewest active
// imported lanes first, ties broken by lowest peer index. Unknown counts
// (-1) sort after every known count, so a readable peer always beats an
// unreadable one; an all-unknown snapshot falls back to index order.
func selectPeerOrder(counts []int) []int {
	order := make([]int, len(counts))
	for i := range order {
		order[i] = i
	}
	rank := func(i int) int {
		if counts[i] < 0 {
			return 1 << 30
		}
		return counts[i]
	}
	sort.SliceStable(order, func(a, b int) bool {
		ra, rb := rank(order[a]), rank(order[b])
		if ra != rb {
			return ra < rb
		}
		return order[a] < order[b]
	})
	return order
}

// scrapeLaneCounts reads every peer's active-lane counter concurrently under
// one bounded context. A scrape failure (or a peer without a control plane)
// leaves that peer's count unknown (-1); scraping never blocks the handoff
// past scrapeTimeout.
func (r *PDPeerRouter) scrapeLaneCounts(ctx context.Context) []int {
	counts := make([]int, len(r.peers))
	scrapeCtx, cancel := context.WithTimeout(ctx, r.scrapeTimeout)
	defer cancel()
	var wg sync.WaitGroup
	for i := range r.peers {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			n, err := r.fetcher(scrapeCtx, r.peers[i].ControlAddr)
			if err != nil || n < 0 {
				counts[i] = -1
				return
			}
			counts[i] = n
		}(i)
	}
	wg.Wait()
	return counts
}

// Ship runs one non-streaming handoff through the router: lane-count scrape,
// fewest-active-lanes selection, ship, and one bounded retry on a different
// peer. After the budget it refuses with *PDHandoffFailure (errors.Is
// ErrPDHandoffFailed).
func (r *PDPeerRouter) Ship(ctx context.Context, req PDShipRequest) (PDShipResult, error) {
	return r.run(ctx, req, false, nil)
}

// ShipStream is Ship with incremental token delivery to sink. A sink error
// aborts the handoff with ErrPDStreamConsumerStopped and spends NO retry: the
// caller's request is already over.
func (r *PDPeerRouter) ShipStream(ctx context.Context, req PDShipRequest, sink PDTokenSink) (PDShipResult, error) {
	if sink == nil {
		return PDShipResult{}, errors.New("modelengine: P/D peer router requires a token sink for streaming handoffs")
	}
	return r.run(ctx, req, true, sink)
}

// run executes the shared admit -> route -> hand-off -> retry sequence.
func (r *PDPeerRouter) run(ctx context.Context, req PDShipRequest, stream bool, sink PDTokenSink) (PDShipResult, error) {
	if r == nil || len(r.peers) == 0 {
		return PDShipResult{}, errors.New("modelengine: P/D peer router has no decode peers")
	}
	counts := r.scrapeLaneCounts(ctx)
	order := selectPeerOrder(counts)

	attempts := make([]PDHandoffAttempt, 0, 2)
	receipt := PDRouteReceipt{
		LaneCounts: append([]int(nil), counts...),
		Chosen:     order[0],
		RetryTo:    -1,
		ServedBy:   -1,
	}
	for depth, idx := range order {
		if depth >= 2 {
			break // bounded budget: first pick + one retry
		}
		attempt := PDHandoffAttempt{PeerIndex: idx, LaneCount: counts[idx]}
		var res PDShipResult
		var err error
		if stream {
			res, err = r.clients[idx].ShipStream(ctx, req, sink)
		} else {
			res, err = r.clients[idx].Ship(ctx, req)
		}
		if err == nil {
			receipt.ServedBy = idx
			if depth > 0 {
				receipt.RetryTo = idx
			}
			r.recordReceipt(receipt, false)
			return res, nil
		}
		attempt.Err = err
		attempts = append(attempts, attempt)
		// A dead caller, a spent deadline, or a stopped stream consumer owes
		// no retry: the request itself is over, so spend no second peer on
		// it and surface the closed reason unchanged.
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) || errors.Is(err, ErrPDStreamConsumerStopped) {
			r.recordReceipt(receipt, false)
			return PDShipResult{}, err
		}
	}
	r.recordReceipt(receipt, true)
	return PDShipResult{}, &PDHandoffFailure{Attempts: attempts}
}

// recordReceipt stamps the step number, updates the cumulative counters and
// last-known lane counts, and appends the receipt to the bounded ledger.
func (r *PDPeerRouter) recordReceipt(rec PDRouteReceipt, failed bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.step++
	rec.Step = r.step
	r.routes++
	if rec.RetryTo >= 0 {
		r.retries++
	}
	if failed {
		r.failures++
	}
	copy(r.lastLaneCounts, rec.LaneCounts)
	r.ledger = append(r.ledger, rec)
	if len(r.ledger) > pdRouterLedgerCap {
		r.ledger = r.ledger[len(r.ledger)-pdRouterLedgerCap:]
	}
}

// RouteLedger returns a copy of the recorded route receipts, oldest first.
// The empty slice means no handoff has run yet.
func (r *PDPeerRouter) RouteLedger() []PDRouteReceipt {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]PDRouteReceipt(nil), r.ledger...)
}

// Metrics renders the router counters in the same Prometheus text shape as
// the cluster metrics; the prefill front door splices it into its /metrics.
// The per-peer last-known lane gauges are the standing admission-skew
// witness; the full per-step history lives in the route ledger.
func (r *PDPeerRouter) Metrics() string {
	if r == nil {
		return ""
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	var b strings.Builder
	b.WriteString("# HELP fak_pd_router_routes_total P/D gateway handoff admissions through the peer router.\n")
	b.WriteString("# TYPE fak_pd_router_routes_total counter\n")
	fmt.Fprintf(&b, "fak_pd_router_routes_total %d\n", r.routes)
	b.WriteString("# HELP fak_pd_router_retries_total P/D handoffs that needed the bounded second peer.\n")
	b.WriteString("# TYPE fak_pd_router_retries_total counter\n")
	fmt.Fprintf(&b, "fak_pd_router_retries_total %d\n", r.retries)
	b.WriteString("# HELP fak_pd_router_failures_total P/D handoffs refused after the retry budget.\n")
	b.WriteString("# TYPE fak_pd_router_failures_total counter\n")
	fmt.Fprintf(&b, "fak_pd_router_failures_total %d\n", r.failures)
	b.WriteString("# HELP fak_pd_router_peer_lane_count Last observed active imported lanes per decode peer.\n")
	b.WriteString("# TYPE fak_pd_router_peer_lane_count gauge\n")
	for i, n := range r.lastLaneCounts {
		if n < 0 {
			continue // never observed: unobservable values stay absent, never zero
		}
		fmt.Fprintf(&b, "fak_pd_router_peer_lane_count{peer=%q} %d\n", strconv.Itoa(i), n)
	}
	return b.String()
}
