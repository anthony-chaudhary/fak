package gateway

// a2a_pushnotify.go — the A2A pushNotificationConfig surface + best-effort webhook
// delivery (the disconnected-consumer half of the A2A task lifecycle).
//
// The A2A spec lets a caller register a push-notification webhook for a task so a peer
// that is not holding the connection open still learns when the task's state changes,
// instead of long-polling GetTask. This file adds that surface WITHOUT reopening the
// self-report door the rest of the A2A stack keeps shut:
//
//   - The outbound body is a RE-FETCH POINTER, never a fak-asserted terminal status.
//     a2aPushNotification carries only the task id, the GetTask URL the receiver
//     re-reads to re-verify, and the no-`claimed`-field VerifiedProgress cursor
//     (internal/relay/progress.go — the same fold GetTask projects). There is
//     deliberately NO state/status/completed/success/claimed field: the receiver learns
//     "something changed, go re-verify", not "fak says you are done".
//     TestA2APushNotificationHasNoClaimedField asserts this reflectively.
//
//   - Delivery RIDES AN ADMISSION FLOOR before any dial (the exfil floor discipline the
//     gateway uses everywhere else): a2aWebhookAdmit requires an http(s) scheme, a host
//     on the configured allowlist, and — riding the same net.ParseIP/IsLoopback SSRF
//     floor as requestFromLoopback — refuses an IP-literal target in a loopback/private/
//     link-local range unless loopback delivery was explicitly enabled. A target that
//     does not pass is refused with the CLOSED reason WEBHOOK_URL_NOT_ALLOWLISTED and is
//     NEVER dialed (refuse, don't guess). Deny-by-default: an unconfigured allowlist
//     refuses every URL.
//
//   - The per-transition POST overhead is METERED (meter-the-meter): a2aCheckWebhookOverhead
//     reads the dial's wall-clock back against a declared turntaxmeter.Budget and names the
//     same closed turntaxmeter.OverheadBudgetExceeded token a lifecycle-rung breach names,
//     so a webhook that turns pathologically slow reads back as a structured breach rather
//     than silent latency.
//
// Live host-wiring (calling SetA2APushWebhookAllowlist from `fak serve` so a deployment
// enables webhooks) is the next rung, in the same spirit as a2a_progress.go's not-yet-wired
// LedgerReader: here the surface, the admission floor, the meter, and their contract tests
// are the artifact. Until a host configures an allowlist, delivery is inert (deny-by-default).

import (
	"bytes"
	"encoding/json"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/relay"
	"github.com/anthony-chaudhary/fak/internal/turntaxmeter"
)

// a2aReasonWebhookNotAllowlisted is the closed-vocabulary refusal token a push-notification
// target names when it is not admissible — a non-allowlisted host, a non-http(s) scheme, a
// malformed URL, or an SSRF-range IP literal that was not explicitly permitted. It MUST stay
// byte-identical to the dos.toml [reasons.WEBHOOK_URL_NOT_ALLOWLISTED] declaration so the same
// token a producer stamps is the one `dos man wedge <TOKEN> --explain` verifies and a deny-loopback routes.
const a2aReasonWebhookNotAllowlisted = "WEBHOOK_URL_NOT_ALLOWLISTED"

// a2aWebhookOverheadBudget is the DECLARED per-transition webhook-POST overhead envelope —
// the "expected" baseline a2aCheckWebhookOverhead measures a breach against, in the same
// turntaxmeter grammar the lifecycle rungs use. A notify POST is network-bound (its cost is
// dominated by the receiver, like the witness gate is dominated by spawning git), so the
// envelope is deliberately wide: it exists to catch a GROSS regression (a hung or
// pathologically slow receiver), not to promise a fast round-trip.
var a2aWebhookOverheadBudget = turntaxmeter.Budget{
	Rung:            "a2a_webhook",
	Method:          "notify",
	MaxNS:           5_000_000_000, // 5s: a single best-effort notify POST over the network
	MaxTokenDelta:   0,             // a pointer body adds no model tokens
	SubprocessBound: true,          // network-bound, like the git-spawning witness gate
}

// a2aCheckWebhookOverhead reads one delivered POST's wall-clock back against the declared
// webhook overhead budget and returns the closed turntaxmeter.OverheadBudgetExceeded token
// on a breach — meter-the-meter over the exact same predicate turntaxmeter.CheckSpan uses,
// so a slow webhook is a structured breach, never silent latency.
func a2aCheckWebhookOverhead(elapsed time.Duration) (breach bool, reason string) {
	span := turntaxmeter.Span{Rung: "a2a_webhook", Method: "notify", ElapsedNS: elapsed.Nanoseconds()}
	b := a2aWebhookOverheadBudget
	if span.ElapsedNS > b.MaxNS || span.TokenDelta > b.MaxTokenDelta {
		return true, turntaxmeter.OverheadBudgetExceeded
	}
	return false, ""
}

// a2aPushNotification is the outbound webhook body: a RE-FETCH POINTER, never a fak-asserted
// terminal status. It carries (a) the task id, (b) the relative GetTask path the receiver
// re-reads to re-verify, and (c) the no-`claimed`-field VerifiedProgress cursor. There is
// deliberately NO state/status/completed/success/claimed field anywhere in this type tree
// (TestA2APushNotificationHasNoClaimedField asserts it reflectively): the receiver is told
// "something changed, go re-verify at RefetchURL", not handed a completion it did not check.
type a2aPushNotification struct {
	// TaskID is the task whose transition triggered this notification.
	TaskID string `json:"task_id"`
	// RefetchURL is the GetTask resource path the receiver GETs to re-verify the task's
	// current state from the authoritative store — the evidence pointer, not the evidence.
	RefetchURL string `json:"refetch_url"`
	// Progress is the re-verifiable VerifiedProgress cursor (verdict verified|unknown), the
	// same fold GetTask projects. It has no `claimed` field by construction (relay invariant).
	Progress relay.VerifiedProgress `json:"progress"`
	// SentAt is a display-only send timestamp; it is not a status a receiver consumes.
	SentAt string `json:"sent_at,omitempty"`
}

// a2aPushConfig is one registered webhook target, scoped to a task id (and the caller that
// owns the task). It is what a2aPushNotificationConfig/set stores and /get echoes back.
type a2aPushConfig struct {
	URL       string    `json:"url"`
	CallerID  string    `json:"caller_id,omitempty"`
	CreatedAt time.Time `json:"created_at"`
}

// a2aPushStore holds the per-task webhook registrations plus the delivery-side admission
// configuration (the host allowlist, the loopback opt-in, and the bounded outbound client).
// Deny-by-default: a zero allowlist refuses every delivery. Counters back the meter.
type a2aPushStore struct {
	mu      sync.RWMutex
	configs map[string]a2aPushConfig // keyed by task id
	allow   map[string]bool          // allowlisted hostnames (lowercased); empty ⇒ deny all
	// allowLoopback opts delivery into loopback/private IP-literal targets (for an operator
	// who deliberately fronts an internal receiver, and for the test HTTP server). Off by
	// default: the SSRF floor refuses private targets even if a host was allowlisted.
	allowLoopback bool
	client        *http.Client
	// meter counters (atomic): deliveries fired, admission refusals, overhead breaches.
	delivered uint64
	refused   uint64
	breaches  uint64
}

func newA2APushStore() *a2aPushStore {
	return &a2aPushStore{
		configs: make(map[string]a2aPushConfig),
		allow:   make(map[string]bool),
		client:  &http.Client{Timeout: 10 * time.Second},
	}
}

var (
	a2aPushMu        sync.Mutex
	a2aPushSingleton *a2aPushStore
)

// getA2APushStore returns the singleton push-notification store, building it on first use.
// Uses a plain guarded lazy-init (not sync.Once) so a test can reset the singleton.
func getA2APushStore() *a2aPushStore {
	a2aPushMu.Lock()
	defer a2aPushMu.Unlock()
	if a2aPushSingleton == nil {
		a2aPushSingleton = newA2APushStore()
	}
	return a2aPushSingleton
}

// SetA2APushWebhookAllowlist configures which webhook hosts push-notification delivery may
// dial (deny-by-default until called) and whether loopback/private targets are permitted.
// This is the host-facing seam a deployment calls to ENABLE webhooks; without it delivery
// stays inert. Hosts are matched case-insensitively by hostname.
func (s *Server) SetA2APushWebhookAllowlist(hosts []string, allowLoopback bool) {
	ps := getA2APushStore()
	ps.mu.Lock()
	defer ps.mu.Unlock()
	ps.allow = make(map[string]bool, len(hosts))
	for _, h := range hosts {
		if h = strings.ToLower(strings.TrimSpace(h)); h != "" {
			ps.allow[h] = true
		}
	}
	ps.allowLoopback = allowLoopback
}

// a2aBlockedIP reports whether an IP literal is in an SSRF-sensitive range (loopback,
// private, link-local, or unspecified) — the ranges the exfil floor refuses to dial by
// default. Classifying by IP VALUE (not string prefix) is the same discipline
// requestFromLoopback uses, so "127.0.0.1.evil.com" cannot masquerade as internal.
func a2aBlockedIP(ip net.IP) bool {
	return ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsUnspecified()
}

// a2aWebhookSyntacticOK reports whether a URL is a well-formed http(s) target with a host —
// the light validation the set surface applies at registration time (a malformed target is
// refused up front) before the full allowlist admission runs at delivery time.
func a2aWebhookSyntacticOK(rawURL string) bool {
	u, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil || u == nil {
		return false
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return false
	}
	return u.Hostname() != ""
}

// a2aWebhookAdmit decides whether a webhook target may be DIALED, riding the exfil/SSRF
// floor. It returns the parsed URL and reason=="" on admit; otherwise nil and the closed
// token a2aReasonWebhookNotAllowlisted. No dial happens here — a refusal means the outbound
// client is never invoked. Gates, in order: parseable http(s) URL with a host; the host is
// on the allowlist (deny-by-default); and an IP-literal target in an SSRF range is refused
// unless loopback delivery was explicitly enabled.
func a2aWebhookAdmit(rawURL string, allow map[string]bool, allowLoopback bool) (*url.URL, string) {
	u, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil || u == nil || u.Host == "" {
		return nil, a2aReasonWebhookNotAllowlisted
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, a2aReasonWebhookNotAllowlisted
	}
	host := u.Hostname()
	if host == "" {
		return nil, a2aReasonWebhookNotAllowlisted
	}
	if len(allow) == 0 || !allow[strings.ToLower(host)] {
		return nil, a2aReasonWebhookNotAllowlisted
	}
	if ip := net.ParseIP(host); ip != nil && a2aBlockedIP(ip) && !allowLoopback {
		return nil, a2aReasonWebhookNotAllowlisted
	}
	return u, ""
}

// allowSnapshot copies the delivery config under the read lock so a dial runs against a
// stable allowlist without holding the store lock across the network POST.
func (ps *a2aPushStore) allowSnapshot() (allow map[string]bool, allowLoopback bool, client *http.Client) {
	ps.mu.RLock()
	defer ps.mu.RUnlock()
	allow = make(map[string]bool, len(ps.allow))
	for h := range ps.allow {
		allow[h] = true
	}
	client = ps.client
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Second}
	}
	return allow, ps.allowLoopback, client
}

// deliver fires AT MOST ONE POST for a task's registered webhook. It returns fired=true only
// when a request was actually dialed. A task with no registered config is a no-op (fired
// false, no reason). A config whose URL is not admissible is refused with the closed reason
// and NEVER dialed. The single POST's overhead is metered against the declared budget. All
// delivery is best-effort: a transport error or non-2xx response is logged, not retried, and
// does not fail the transition that triggered it.
func (ps *a2aPushStore) deliver(taskID string, note a2aPushNotification, logf func(string, ...any)) (fired bool, reason string) {
	ps.mu.RLock()
	cfg, ok := ps.configs[taskID]
	ps.mu.RUnlock()
	if !ok {
		return false, "" // no webhook registered for this task — nothing to deliver
	}
	allow, allowLoopback, client := ps.allowSnapshot()

	u, refuse := a2aWebhookAdmit(cfg.URL, allow, allowLoopback)
	if refuse != "" {
		atomic.AddUint64(&ps.refused, 1)
		a2aPushLogf(logf, "gateway: A2A push refused task=%s reason=%s url=%q (never dialed)", taskID, refuse, cfg.URL)
		return false, refuse
	}

	body, err := json.Marshal(note)
	if err != nil {
		a2aPushLogf(logf, "gateway: A2A push marshal failed task=%s: %v", taskID, err)
		return false, ""
	}
	req, err := http.NewRequest(http.MethodPost, u.String(), bytes.NewReader(body))
	if err != nil {
		a2aPushLogf(logf, "gateway: A2A push build request failed task=%s: %v", taskID, err)
		return false, ""
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "fak-a2a-push/1")

	start := time.Now()
	resp, err := client.Do(req)
	elapsed := time.Since(start)

	if breach, tok := a2aCheckWebhookOverhead(elapsed); breach {
		atomic.AddUint64(&ps.breaches, 1)
		a2aPushLogf(logf, "gateway: A2A push %s task=%s elapsed=%s budget=%dns", tok, taskID, elapsed, a2aWebhookOverheadBudget.MaxNS)
	}
	if err != nil {
		a2aPushLogf(logf, "gateway: A2A push delivery error task=%s url=%q: %v (best-effort, not retried)", taskID, u.String(), err)
		// A POST was dialed even though the transport erred — count it as fired so the
		// exactly-one-per-transition contract holds regardless of receiver behavior.
		atomic.AddUint64(&ps.delivered, 1)
		return true, ""
	}
	// Drain+close so the connection can be reused; the body is not consumed as progress.
	if resp.Body != nil {
		_ = resp.Body.Close()
	}
	atomic.AddUint64(&ps.delivered, 1)
	return true, ""
}

// a2aPushLogf calls logf only when it is wired (a bare Server has none).
func a2aPushLogf(logf func(string, ...any), format string, args ...any) {
	if logf != nil {
		logf(format, args...)
	}
}

// a2aOnTaskTransition is the single fire point for a witnessed terminal task transition: it
// delivers at most one re-fetch pointer to the task's registered webhook (if any). It fires
// ONLY on a terminal state (a2aTerminalState) — the transitions a disconnected consumer
// waits on — and builds the no-`claimed`-field pointer body from the same VerifiedProgress
// fold GetTask projects. Called after the store lock is released so no network POST runs
// under the task-store lock. A task with no registered webhook is a cheap no-op.
func (s *Server) a2aOnTaskTransition(task *a2aTask) {
	if task == nil || !a2aTerminalState(task.State) {
		return
	}
	ps := getA2APushStore()
	// Fast no-op when nothing is registered — avoid building a body for the common case.
	ps.mu.RLock()
	_, registered := ps.configs[task.TaskID]
	ps.mu.RUnlock()
	if !registered {
		return
	}
	note := a2aPushNotification{
		TaskID:     task.TaskID,
		RefetchURL: "/a2a/v1/tasks/" + task.TaskID,
		Progress:   a2aVerifiedProgress(task, nil),
		SentAt:     time.Now().UTC().Format(time.RFC3339),
	}
	ps.deliver(task.TaskID, note, s.logf)
}

// a2aSetPushRequest is the pushNotificationConfig/set body. It accepts both a flat
// {"url": "..."} and the spec's nested {"pushNotificationConfig": {"url": "..."}}.
type a2aSetPushRequest struct {
	URL                    string `json:"url"`
	PushNotificationConfig *struct {
		URL string `json:"url"`
	} `json:"pushNotificationConfig"`
}

func (r a2aSetPushRequest) url() string {
	if u := strings.TrimSpace(r.URL); u != "" {
		return u
	}
	if r.PushNotificationConfig != nil {
		return strings.TrimSpace(r.PushNotificationConfig.URL)
	}
	return ""
}

// handleA2ASetPushConfig implements POST /a2a/v1/tasks/{task_id}/pushNotificationConfig:
// register (or replace) the webhook a witnessed terminal transition of {task_id} notifies.
// The caller must own the task. The URL is syntactically validated up front (a malformed
// target is refused with the closed reason); the full allowlist admission runs at delivery.
func (s *Server) handleA2ASetPushConfig(w http.ResponseWriter, r *http.Request, taskID string) {
	callerID := r.Header.Get("X-Caller-ID")
	if callerID == "" {
		writeErr(w, http.StatusUnauthorized, "caller identity required")
		return
	}
	store := getA2AStore()
	store.mu.RLock()
	task, exists := store.tasks[taskID]
	var owner string
	if exists {
		owner = task.CallerID
	}
	store.mu.RUnlock()
	if !exists {
		writeErr(w, http.StatusNotFound, "task not found")
		return
	}
	if owner != "" && owner != callerID {
		writeErr(w, http.StatusForbidden, "task not owned by caller")
		return
	}

	var req a2aSetPushRequest
	if !decodeRequestBody(w, r, &req) {
		return // decodeRequestBody already wrote the 400
	}
	target := req.url()
	if target == "" {
		writeErr(w, http.StatusBadRequest, "url required")
		return
	}
	if !a2aWebhookSyntacticOK(target) {
		writeErrCode(w, http.StatusBadRequest, a2aReasonWebhookNotAllowlisted, "webhook url must be a well-formed http(s) URL with a host")
		return
	}

	cfg := a2aPushConfig{URL: target, CallerID: callerID, CreatedAt: time.Now().UTC()}
	ps := getA2APushStore()
	ps.mu.Lock()
	ps.configs[taskID] = cfg
	ps.mu.Unlock()

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"task_id":    taskID,
		"url":        cfg.URL,
		"caller_id":  cfg.CallerID,
		"created_at": cfg.CreatedAt.Format(time.RFC3339),
	})
}

// handleA2AGetPushConfig implements GET /a2a/v1/tasks/{task_id}/pushNotificationConfig:
// read back the registered webhook. The caller must own the task.
func (s *Server) handleA2AGetPushConfig(w http.ResponseWriter, r *http.Request, taskID string) {
	callerID := r.Header.Get("X-Caller-ID")
	if callerID == "" {
		writeErr(w, http.StatusUnauthorized, "caller identity required")
		return
	}
	store := getA2AStore()
	store.mu.RLock()
	task, exists := store.tasks[taskID]
	var owner string
	if exists {
		owner = task.CallerID
	}
	store.mu.RUnlock()
	if !exists {
		writeErr(w, http.StatusNotFound, "task not found")
		return
	}
	if owner != "" && owner != callerID {
		writeErr(w, http.StatusForbidden, "task not owned by caller")
		return
	}

	ps := getA2APushStore()
	ps.mu.RLock()
	cfg, ok := ps.configs[taskID]
	ps.mu.RUnlock()
	if !ok {
		writeErr(w, http.StatusNotFound, "no push notification config registered for task")
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"task_id":    taskID,
		"url":        cfg.URL,
		"caller_id":  cfg.CallerID,
		"created_at": cfg.CreatedAt.Format(time.RFC3339),
	})
}
