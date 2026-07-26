package servicelease

// The LIVE half of the #4752 acceptance witness. partition_sim_test.go proves
// the fencing contract deterministically with a logical clock and an in-memory
// transport; this file re-runs the SAME fault script against two REAL node
// processes, a REAL HTTP control plane, real on-disk durable state, and the
// wall clock — the "only the transport and clock change, never the table"
// promise partition_sim.go makes, actually collected.
//
// Shape:
//
//   - The parent test IS the control plane: one mutex-guarded Table behind an
//     httptest server, plus a reconcile loop that folds the three evidence
//     channels (pull heartbeat, native-manager read-back, its own durable boot
//     record) into a BuildPlan every tick.
//   - Each node is this test binary re-exec'd with an env guard (see
//     TestLiveNodeHelper). It acquires/renews over real HTTP, persists its own
//     lease copy, and publishes a CLAIM file for as long as it believes it is
//     running the workload.
//   - A claim file deliberately OUTLIVES a hard-killed node. That is the whole
//     point: the safety invariant is not "only one process wrote a claim", it
//     is "at most one claim would be ACCEPTED by the table" — running, not
//     valid. The sampler checks exactly that, every 25ms, for the whole run.
//
// Faults covered, each of which must converge: delayed heartbeat (no
// reassignment), process crash (local restart under the SAME incarnation,
// offline, ownership unchanged), host reboot (new boot ID supersedes and
// fences the old incarnation), and true node loss (takeover only after the
// held lease actually expires).

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// Env contract between the parent control plane and the re-exec'd node
// children. liveNodeEnv is the guard that turns a plain test run of
// TestLiveNodeHelper into a node process.
const (
	liveNodeEnv     = "FAK_SERVICELEASE_LIVE_NODE"
	liveURLEnv      = "FAK_SERVICELEASE_LIVE_URL"
	liveNodeNameEnv = "FAK_SERVICELEASE_LIVE_NODE_NAME"
	liveBootEnv     = "FAK_SERVICELEASE_LIVE_BOOT"
	liveDirEnv      = "FAK_SERVICELEASE_LIVE_DIR"
	liveTickEnv     = "FAK_SERVICELEASE_LIVE_TICK_MS"
)

// Live timing budget. Real wall clock, so every window is sized for a loaded
// Windows/CI box: the lease TTL is long enough that a process restart never
// costs ownership, and the heartbeat timeout is short enough that a 1s silence
// is unambiguously a partition.
const (
	liveTTLMS         int64 = 3000
	liveHBTimeoutMS   int64 = 500
	liveTickMS              = 120
	liveReconcileTick       = 40 * time.Millisecond
	liveSampleTick          = 25 * time.Millisecond
	liveWaitTimeout         = 30 * time.Second
	liveChildDeadline       = 3 * time.Minute
)

// liveClaim is the on-disk artifact a node writes while it believes it is
// RUNNING the workload. It is intentionally not cleaned up on a hard kill.
type liveClaim struct {
	Node  string       `json:"node"`
	Boot  string       `json:"boot"`
	Token FencingToken `json:"token"`
}

// Incarnation returns the claim's identity.
func (c liveClaim) Incarnation() Incarnation { return Incarnation{Node: c.Node, BootID: c.Boot} }

// liveReq / liveResp are the real wire format between node and control plane.
type liveReq struct {
	Op    string       `json:"op"` // "acquire" | "renew"
	Node  string       `json:"node"`
	Boot  string       `json:"boot"`
	Token FencingToken `json:"token"`
}

type liveResp struct {
	OK    bool   `json:"ok"`
	Lease *Lease `json:"lease,omitempty"`
	Class string `json:"class,omitempty"`
}

// -------------------------------------------------------------------------
// control plane (parent process)
// -------------------------------------------------------------------------

// liveObs is the pull-heartbeat channel for one node: when it last spoke, and
// which boot it claimed to be.
type liveObs struct {
	lastMS int64
	boot   string
}

// liveCP is the single authority. Table is pure state with no internal
// locking, so every touch goes through mu.
type liveCP struct {
	mu       sync.Mutex
	tbl      *Table
	workload string
	seen     map[string]liveObs
	known    map[string]string // control plane's own durable boot record (lags one reconcile tick)
	refusals map[string]int
	conds    map[Condition]int
	actions  map[ActionKind]int
}

func liveNowMS() int64 { return time.Now().UnixMilli() }

// liveRefusalClass maps a fencing error onto the closed refusal vocabulary so
// assertions bind on the class, never on message text.
func liveRefusalClass(err error) string {
	switch {
	case err == nil:
		return ""
	case errors.Is(err, ErrStaleIncarnation):
		return "stale-incarnation"
	case errors.Is(err, ErrFenced):
		return "fenced"
	case errors.Is(err, ErrLeaseHeld):
		return "lease-held"
	case errors.Is(err, ErrNotHolder):
		return "not-holder"
	case errors.Is(err, ErrLeaseExpired):
		return "lease-expired"
	}
	return "unknown"
}

// serveRPC adjudicates one node claim. Every request doubles as that node's
// pull heartbeat — the control plane learns liveness only from real traffic.
func (c *liveCP) serveRPC(w http.ResponseWriter, r *http.Request) {
	var req liveReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	now := liveNowMS()
	c.seen[req.Node] = liveObs{lastMS: now, boot: req.Boot}

	inc := Incarnation{Node: req.Node, BootID: req.Boot}
	var (
		l   *Lease
		err error
	)
	switch req.Op {
	case "acquire":
		l, err = c.tbl.Acquire(c.workload, inc, now)
	case "renew":
		l, err = c.tbl.Renew(c.workload, inc, req.Token, now)
	default:
		err = fmt.Errorf("unknown op %q", req.Op)
	}
	resp := liveResp{OK: err == nil, Lease: l}
	if err != nil {
		resp.Class = liveRefusalClass(err)
		c.refusals[resp.Class]++
	}
	_ = json.NewEncoder(w).Encode(resp)
}

// reconcile is one control-plane tick: sync the fencing registry from the
// heartbeat channel, then plan for the workload's current holder from all
// three evidence channels. It records the plan vocabulary it witnesses so the
// live run can prove it exercised the same conditions/actions as the sim.
func (c *liveCP) reconcile(nodes []string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	now := liveNowMS()

	// The control plane's PREVIOUS boot record is what makes a reboot
	// classifiable; capture it before the heartbeat channel overwrites it.
	prevKnown := map[string]string{}
	for _, n := range nodes {
		prevKnown[n] = c.known[n]
	}
	for _, n := range nodes {
		o := c.seen[n]
		if o.lastMS == 0 || o.boot == "" || now-o.lastMS > liveHBTimeoutMS {
			continue
		}
		// A fresh heartbeat carrying a new boot ID supersedes the old
		// incarnation immediately — fencing must not wait for a plan.
		c.tbl.RecordIncarnation(Incarnation{Node: n, BootID: o.boot})
		c.known[n] = o.boot
	}

	lease, ok := c.tbl.Leases[c.workload]
	if !ok {
		return
	}
	holder := lease.Holder.Node
	o := c.seen[holder]
	e := Evidence{
		NowMS:              now,
		LastHeartbeatMS:    o.lastMS,
		HeartbeatBootID:    o.boot,
		HeartbeatTimeoutMS: liveHBTimeoutMS,
		KnownBootID:        prevKnown[holder],
		// Native-manager read-back channel: a node that is talking to us is
		// reporting its unit ready. Silence is handled by the heartbeat rung.
		ReadBack: &servicespec.Observed{Schema: servicespec.ObservedSchemaV1, Phase: servicespec.PhaseReady},
	}
	p := BuildPlan(c.tbl, c.workload, e)
	c.conds[p.Condition]++
	c.actions[p.Action]++
}

// -------------------------------------------------------------------------
// harness (parent process)
// -------------------------------------------------------------------------

// liveSample is one observed ownership transition on the wall clock.
type liveSample struct {
	atMS  int64
	owner string
}

type liveHarness struct {
	t     *testing.T
	dir   string
	srv   *httptest.Server
	cp    *liveCP
	nodes []string
	stop  chan struct{}
	wg    sync.WaitGroup

	mu         sync.Mutex
	timeline   []liveSample
	violations []string
	maxOwners  int
	procs      map[string]*exec.Cmd
}

func newLiveHarness(t *testing.T) *liveHarness {
	t.Helper()
	cp := &liveCP{
		tbl:      NewTable(liveTTLMS),
		workload: wl,
		seen:     map[string]liveObs{},
		known:    map[string]string{},
		refusals: map[string]int{},
		conds:    map[Condition]int{},
		actions:  map[ActionKind]int{},
	}
	h := &liveHarness{
		t:     t,
		dir:   t.TempDir(),
		cp:    cp,
		nodes: []string{"alpha", "beta"},
		stop:  make(chan struct{}),
		procs: map[string]*exec.Cmd{},
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/rpc", cp.serveRPC)
	h.srv = httptest.NewServer(mux)

	h.wg.Add(2)
	go h.reconcileLoop()
	go h.sampleLoop()
	t.Cleanup(h.shutdown)
	return h
}

func (h *liveHarness) shutdown() {
	h.mu.Lock()
	live := make([]string, 0, len(h.procs))
	for n := range h.procs {
		live = append(live, n)
	}
	h.mu.Unlock()
	for _, n := range live {
		h.kill(n)
	}
	close(h.stop)
	h.wg.Wait()
	h.srv.Close()
}

func (h *liveHarness) reconcileLoop() {
	defer h.wg.Done()
	tk := time.NewTicker(liveReconcileTick)
	defer tk.Stop()
	for {
		select {
		case <-h.stop:
			return
		case <-tk.C:
			h.cp.reconcile(h.nodes)
		}
	}
}

// sampleLoop is the safety witness: every tick it asks the table which of the
// on-disk claims would ACTUALLY be accepted right now. More than one at any
// instant is overlapping ownership and fails the run.
func (h *liveHarness) sampleLoop() {
	defer h.wg.Done()
	tk := time.NewTicker(liveSampleTick)
	defer tk.Stop()
	for {
		select {
		case <-h.stop:
			return
		case <-tk.C:
			h.sampleOnce()
		}
	}
}

func (h *liveHarness) sampleOnce() {
	var valid []string
	h.cp.mu.Lock()
	now := liveNowMS()
	for _, n := range h.nodes {
		c, ok := readLiveClaim(h.claimPath(n))
		if !ok {
			continue
		}
		if h.cp.tbl.WouldAccept(h.cp.workload, c.Incarnation(), c.Token, now) {
			valid = append(valid, n)
		}
	}
	h.cp.mu.Unlock()

	h.mu.Lock()
	defer h.mu.Unlock()
	if len(valid) > h.maxOwners {
		h.maxOwners = len(valid)
	}
	if len(valid) > 1 {
		h.violations = append(h.violations,
			fmt.Sprintf("at %d: %d simultaneously valid owners %v", now, len(valid), valid))
	}
	owner := strings.Join(valid, "+")
	if n := len(h.timeline); n == 0 || h.timeline[n-1].owner != owner {
		h.timeline = append(h.timeline, liveSample{atMS: now, owner: owner})
	}
}

func (h *liveHarness) claimPath(node string) string {
	return filepath.Join(h.dir, "claim-"+node+".json")
}
func (h *liveHarness) pausePath(node string) string {
	return filepath.Join(h.dir, "pause-"+node)
}

// start launches a real node process for (node, boot).
func (h *liveHarness) start(node, boot string) {
	h.t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestLiveNodeHelper$")
	cmd.Env = append(os.Environ(),
		liveNodeEnv+"=1",
		liveURLEnv+"="+h.srv.URL,
		liveNodeNameEnv+"="+node,
		liveBootEnv+"="+boot,
		liveDirEnv+"="+h.dir,
		liveTickEnv+"="+strconv.Itoa(liveTickMS),
	)
	cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
	if err := cmd.Start(); err != nil {
		h.t.Fatalf("start node %s@%s: %v", node, boot, err)
	}
	h.mu.Lock()
	h.procs[node] = cmd
	h.mu.Unlock()
}

// kill hard-kills a node process without letting it clean up — its claim file
// survives on purpose.
func (h *liveHarness) kill(node string) {
	h.mu.Lock()
	cmd := h.procs[node]
	delete(h.procs, node)
	h.mu.Unlock()
	if cmd == nil || cmd.Process == nil {
		return
	}
	_ = cmd.Process.Kill()
	_ = cmd.Wait()
}

func (h *liveHarness) pause(node string) {
	if err := os.WriteFile(h.pausePath(node), []byte("1"), 0o644); err != nil {
		h.t.Fatalf("pause %s: %v", node, err)
	}
}
func (h *liveHarness) resume(node string) { _ = os.Remove(h.pausePath(node)) }

// ownerNow reports the single currently-valid owner node ("" if none).
func (h *liveHarness) ownerNow() string {
	h.cp.mu.Lock()
	defer h.cp.mu.Unlock()
	return h.cp.tbl.ValidOwner(h.cp.workload, liveNowMS()).Node
}

// leaseNow returns a copy of the workload's current lease (ok=false if none).
func (h *liveHarness) leaseNow() (Lease, bool) {
	h.cp.mu.Lock()
	defer h.cp.mu.Unlock()
	l, ok := h.cp.tbl.Leases[h.cp.workload]
	if !ok {
		return Lease{}, false
	}
	return *l, true
}

func (h *liveHarness) counts() (map[string]int, map[Condition]int, map[ActionKind]int) {
	h.cp.mu.Lock()
	defer h.cp.mu.Unlock()
	r := map[string]int{}
	for k, v := range h.cp.refusals {
		r[k] = v
	}
	c := map[Condition]int{}
	for k, v := range h.cp.conds {
		c[k] = v
	}
	a := map[ActionKind]int{}
	for k, v := range h.cp.actions {
		a[k] = v
	}
	return r, c, a
}

// waitOwner blocks until node is the valid owner, or fails the test.
func (h *liveHarness) waitOwner(node, why string) {
	h.t.Helper()
	h.waitFor(why, func() bool { return h.ownerNow() == node })
}

func (h *liveHarness) waitFor(why string, cond func() bool) {
	h.t.Helper()
	deadline := time.Now().Add(liveWaitTimeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		h.assertNoOverlap()
		time.Sleep(20 * time.Millisecond)
	}
	h.t.Fatalf("timed out waiting for %s (owner=%q timeline=%v)", why, h.ownerNow(), h.ownerTimeline())
}

// assertNoOverlap fails the instant the safety invariant has ever broken.
func (h *liveHarness) assertNoOverlap() {
	h.t.Helper()
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.violations) > 0 {
		h.t.Fatalf("overlapping valid ownership: %v", h.violations)
	}
}

func (h *liveHarness) ownerTimeline() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := make([]string, 0, len(h.timeline))
	for _, s := range h.timeline {
		out = append(out, fmt.Sprintf("%d:%s", s.atMS, orNone(s.owner)))
	}
	return out
}

// mark returns the current end of the ownership timeline, so a later query can
// ask "since this fault", not "ever".
func (h *liveHarness) mark() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return len(h.timeline)
}

// firstOwnedAtMSSince reports when node was first observed as a valid owner at
// or after the given timeline mark (0 = never). Scoping to the mark matters:
// both nodes legitimately own the workload at different points in the run, so
// an unscoped search would return a sighting from a previous phase.
func (h *liveHarness) firstOwnedAtMSSince(node string, mark int) int64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	for i := mark; i < len(h.timeline); i++ {
		if h.timeline[i].owner == node {
			return h.timeline[i].atMS
		}
	}
	return 0
}

func orNone(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func readLiveClaim(path string) (liveClaim, bool) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return liveClaim{}, false
	}
	var c liveClaim
	if json.Unmarshal(raw, &c) != nil || c.Node == "" {
		return liveClaim{}, false
	}
	return c, true
}

// writeLiveJSON publishes atomically so a sampler never reads a half-written
// claim.
func writeLiveJSON(path string, v any) {
	raw, err := json.Marshal(v)
	if err != nil {
		return
	}
	tmp := path + ".tmp"
	if os.WriteFile(tmp, raw, 0o644) != nil {
		return
	}
	_ = os.Rename(tmp, path)
}

// -------------------------------------------------------------------------
// the acceptance test
// -------------------------------------------------------------------------

// TestLiveTwoNodeFencedOwnership is the live half of the #4752 acceptance
// witness: two real node processes against a real HTTP control plane, driven
// through delayed heartbeat, crash, reboot, and node loss. No instant of the
// run may have two simultaneously valid owners, and every fault must converge
// back to exactly one.
func TestLiveTwoNodeFencedOwnership(t *testing.T) {
	if testing.Short() {
		t.Skip("live two-node witness: spawns real child processes and uses the wall clock")
	}
	h := newLiveHarness(t)

	// --- steady state: alpha acquires, beta is refused while the lease stands.
	h.start("alpha", "alpha-boot-1")
	h.waitOwner("alpha", "alpha to acquire the workload lease")
	h.start("beta", "beta-boot-1")
	h.waitFor("beta to be refused by the held lease", func() bool {
		r, _, _ := h.counts()
		return r["lease-held"] > 0
	})
	first, ok := h.leaseNow()
	if !ok {
		t.Fatal("no lease after alpha acquired")
	}
	if h.ownerNow() != "alpha" {
		t.Fatalf("owner = %q, want alpha (beta must not steal a valid lease)", h.ownerNow())
	}

	// --- delayed heartbeat: alpha goes silent INSIDE its lease. The control
	// plane must classify a partition and WAIT, never reassign.
	h.pause("alpha")
	h.waitFor("the silence to be classified network-partitioned", func() bool {
		_, c, a := h.counts()
		return c[CondNetworkPartitioned] > 0 && a[ActionWaitLease] > 0
	})
	if got := h.ownerNow(); got != "alpha" {
		t.Fatalf("owner during delayed heartbeat = %q, want alpha (no reassignment inside a valid lease)", got)
	}
	h.resume("alpha")
	h.waitFor("alpha to resume renewing", func() bool {
		l, ok := h.leaseNow()
		return ok && l.ExpiresMS > liveNowMS()+liveTTLMS/2
	})
	after, _ := h.leaseNow()
	if after.Token != first.Token {
		t.Fatalf("delayed heartbeat moved the token %+v -> %+v: silence must not reassign", first.Token, after.Token)
	}
	h.assertNoOverlap()

	// --- crash: alpha's process dies and restarts under the SAME incarnation.
	// It resumes from its durable lease copy with no controller round-trip
	// (offline local recovery) and ownership never moves.
	h.kill("alpha")
	h.start("alpha", "alpha-boot-1")
	h.waitFor("alpha to renew after the crash restart", func() bool {
		l, ok := h.leaseNow()
		return ok && l.Holder.Node == "alpha" && l.ExpiresMS > liveNowMS()+liveTTLMS/2
	})
	crashed, _ := h.leaseNow()
	if crashed.Token != first.Token {
		t.Fatalf("crash restart moved the token %+v -> %+v: a local restart must not change ownership", first.Token, crashed.Token)
	}
	if h.ownerNow() != "alpha" {
		t.Fatalf("post-crash owner = %q, want alpha", h.ownerNow())
	}

	// --- reboot: alpha comes back as a NEW incarnation. The old boot is
	// superseded, its token is dead, and ownership is re-granted under a
	// strictly newer lease sequence.
	h.kill("alpha")
	h.start("alpha", "alpha-boot-2")
	h.waitFor("the reboot to be classified and reassignment to be permitted", func() bool {
		_, c, a := h.counts()
		return c[CondHostRebooted] > 0 && a[ActionReassign] > 0
	})
	h.waitFor("a fresh lease under the post-reboot incarnation", func() bool {
		l, ok := h.leaseNow()
		return ok && l.Token.LeaseSeq > crashed.Token.LeaseSeq && h.ownerNow() != ""
	})
	rebooted, _ := h.leaseNow()
	if rebooted.Token.LeaseSeq <= crashed.Token.LeaseSeq {
		t.Fatalf("post-reboot seq %d must exceed pre-reboot %d", rebooted.Token.LeaseSeq, crashed.Token.LeaseSeq)
	}
	// The dead incarnation can never publish completion, live or not.
	h.cp.mu.Lock()
	err := h.cp.tbl.PublishCompletion(wl, Incarnation{Node: "alpha", BootID: "alpha-boot-1"}, crashed.Token, Checkpoint{Seq: 99})
	h.cp.mu.Unlock()
	if !errors.Is(err, ErrStaleIncarnation) && !errors.Is(err, ErrFenced) && !errors.Is(err, ErrNotHolder) {
		t.Fatalf("dead-incarnation publish = %v, want a fencing refusal", err)
	}
	h.assertNoOverlap()

	// --- node loss: the current owner is killed for good. Takeover is allowed
	// only once the held lease has ACTUALLY expired — never a moment sooner.
	lost := h.ownerNow()
	if lost == "" {
		t.Fatal("no valid owner before the node-loss fault")
	}
	survivor := "beta"
	if lost == "beta" {
		survivor = "alpha"
	}
	held, _ := h.leaseNow()
	mark := h.mark()
	h.kill(lost)
	h.waitOwner(survivor, "the surviving node to take over after lease expiry")
	if at := h.firstOwnedAtMSSince(survivor, mark); at != 0 && at < held.ExpiresMS {
		t.Fatalf("%s became valid owner at %d, before the held lease expired at %d: premature reassignment",
			survivor, at, held.ExpiresMS)
	}

	// --- converged: one owner, and the invariant held for the whole run.
	h.assertNoOverlap()
	if got := h.maxOwnersSeen(); got > 1 {
		t.Fatalf("max simultaneously valid owners = %d, want <= 1", got)
	}
	if h.ownerNow() != survivor {
		t.Fatalf("converged owner = %q, want %s", h.ownerNow(), survivor)
	}

	// The run must have exercised the real vocabulary, not just idled: both
	// nodes owned at some point, and both fencing refusals were witnessed over
	// the wire.
	owners := map[string]bool{}
	for _, s := range h.ownerTimeline() {
		if i := strings.IndexByte(s, ':'); i >= 0 && s[i+1:] != "-" {
			owners[s[i+1:]] = true
		}
	}
	if len(owners) < 2 {
		t.Fatalf("only %v ever owned: the live run never witnessed a cross-node handoff", owners)
	}
	refusals, conds, actions := h.counts()
	for _, want := range []string{"lease-held", "stale-incarnation"} {
		if refusals[want] == 0 {
			t.Fatalf("refusal %q never witnessed over the wire (saw %v)", want, refusals)
		}
	}
	for _, want := range []Condition{CondNetworkPartitioned, CondHostRebooted} {
		if conds[want] == 0 {
			t.Fatalf("condition %q never classified live (saw %v)", want, conds)
		}
	}
	for _, want := range []ActionKind{ActionWaitLease, ActionReassign} {
		if actions[want] == 0 {
			t.Fatalf("action %q never planned live (saw %v)", want, actions)
		}
	}
}

func (h *liveHarness) maxOwnersSeen() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.maxOwners
}

// -------------------------------------------------------------------------
// node child (re-exec'd process)
// -------------------------------------------------------------------------

// TestLiveNodeHelper is not a test: under the liveNodeEnv guard this test
// binary IS one of the two live nodes. Without the guard it is a no-op skip,
// so a normal `go test` run never enters the loop.
func TestLiveNodeHelper(t *testing.T) {
	if os.Getenv(liveNodeEnv) != "1" {
		t.Skip("not a live-node child process")
	}
	runLiveNode()
}

// runLiveNode is one node's whole lifetime: resume durable state offline if it
// legitimately can, then acquire/renew over real HTTP until killed. It writes
// its claim file for exactly as long as it believes it owns the workload, and
// drops it the moment the control plane fences it.
func runLiveNode() {
	node := os.Getenv(liveNodeNameEnv)
	boot := os.Getenv(liveBootEnv)
	dir := os.Getenv(liveDirEnv)
	url := os.Getenv(liveURLEnv) + "/rpc"
	tick, err := strconv.Atoi(os.Getenv(liveTickEnv))
	if err != nil || tick <= 0 {
		tick = liveTickMS
	}
	self := Incarnation{Node: node, BootID: boot}
	claimPath := filepath.Join(dir, "claim-"+node+".json")
	leasePath := filepath.Join(dir, "lease-"+node+".json")
	pausePath := filepath.Join(dir, "pause-"+node)
	client := &http.Client{Timeout: 2 * time.Second}

	// Offline recovery (#4752): a restart under the SAME incarnation resumes
	// the durable lease copy immediately, with no controller round-trip. A
	// restart under a NEW boot ID fails LocalRestartAllowed and must go back
	// through the table — that is the node-side half of reboot fencing.
	var lease *Lease
	if raw, rerr := os.ReadFile(leasePath); rerr == nil {
		var l Lease
		if json.Unmarshal(raw, &l) == nil && LocalRestartAllowed(&l, self) {
			lease = &l
			writeLiveJSON(claimPath, liveClaim{Node: node, Boot: boot, Token: l.Token})
		}
	}
	if lease == nil {
		_ = os.Remove(claimPath)
		_ = os.Remove(leasePath)
	}

	deadline := time.Now().Add(liveChildDeadline)
	for time.Now().Before(deadline) {
		time.Sleep(time.Duration(tick) * time.Millisecond)
		// A paused node keeps RUNNING (its claim stands) but stops talking:
		// the control plane sees only silence, exactly like a partition.
		if _, serr := os.Stat(pausePath); serr == nil {
			continue
		}
		req := liveReq{Op: "acquire", Node: node, Boot: boot}
		if lease != nil {
			req.Op, req.Token = "renew", lease.Token
		}
		resp, rerr := liveNodeRPC(client, url, req)
		if rerr != nil {
			continue // transport blip: keep the current belief, retry
		}
		if resp.OK && resp.Lease != nil {
			lease = resp.Lease
			writeLiveJSON(leasePath, lease)
			writeLiveJSON(claimPath, liveClaim{Node: node, Boot: boot, Token: lease.Token})
			continue
		}
		// Refused: this node is not (or is no longer) the valid owner. It
		// stops running the workload and drops its durable copy.
		lease = nil
		_ = os.Remove(claimPath)
		_ = os.Remove(leasePath)
	}
	os.Exit(0)
}

func liveNodeRPC(client *http.Client, url string, req liveReq) (liveResp, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return liveResp{}, err
	}
	resp, err := client.Post(url, "application/json", strings.NewReader(string(body)))
	if err != nil {
		return liveResp{}, err
	}
	defer resp.Body.Close()
	var out liveResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return liveResp{}, err
	}
	return out, nil
}
