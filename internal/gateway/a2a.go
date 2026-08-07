package gateway

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

// A2A protocol version
const a2aVersion = "1.0"

// a2aMaxTasks bounds the in-memory task store. handleA2ASendMessage inserts one task per
// POST /a2a/v1/messages and nothing ever deleted them, so a long-lived `fak serve` grew the
// map without bound — monotonic heap growth over a multi-day process (the same unbounded-
// accumulation class as the self-update swap-aside leak, in memory instead of on disk). The
// cap is far above any realistic in-flight working set, so a busy but healthy gateway never
// evicts a task a caller is still polling; it only reclaims the long tail of completed tasks
// that no one will read again.
const a2aMaxTasks = 4096

// a2aTaskStore holds A2A tasks with stable IDs and audit logging. It is size-bounded: see
// insertLocked, which evicts the least-useful entry once the map exceeds a2aMaxTasks.
type a2aTaskStore struct {
	mu    sync.RWMutex
	tasks map[string]*a2aTask
}

// insertLocked adds task under taskID, evicting one entry first if the store is at capacity.
// The caller must hold s.mu. Eviction prefers the oldest TERMINAL task (completed/canceled/
// failed — no one is waiting on its result), and only falls back to the oldest task overall
// when every entry is still in-flight (a pathological all-pending state). This keeps the
// common case — a flood of immediately-completed send-message tasks — from ever displacing a
// task a caller is actively polling.
func (s *a2aTaskStore) insertLocked(taskID string, task *a2aTask) {
	if len(s.tasks) >= a2aMaxTasks {
		if victim := s.evictionVictimLocked(); victim != "" {
			delete(s.tasks, victim)
		}
	}
	s.tasks[taskID] = task
}

// evictionVictimLocked picks the task ID to drop when at capacity, or "" if the map is empty.
// The caller must hold s.mu (read or write). It scans once, tracking the oldest terminal task
// and the oldest task overall, and returns the terminal one when present.
func (s *a2aTaskStore) evictionVictimLocked() string {
	var oldestTerminal, oldestAny string
	var terminalAt, anyAt time.Time
	for id, t := range s.tasks {
		if oldestAny == "" || t.CreatedAt.Before(anyAt) {
			oldestAny, anyAt = id, t.CreatedAt
		}
		if a2aTerminalState(t.State) && (oldestTerminal == "" || t.CreatedAt.Before(terminalAt)) {
			oldestTerminal, terminalAt = id, t.CreatedAt
		}
	}
	if oldestTerminal != "" {
		return oldestTerminal
	}
	return oldestAny
}

// a2aTerminalState reports whether an A2A task state is final (no further updates expected),
// so its entry is safe to evict under memory pressure without dropping live work.
func a2aTerminalState(state string) bool {
	switch state {
	case "completed", "canceled", "cancelled", "failed", "error":
		return true
	default:
		return false
	}
}

type a2aTask struct {
	TaskID    string                 `json:"task_id"`
	Title     string                 `json:"title,omitempty"`
	State     string                 `json:"state"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
	Method    string                 `json:"method,omitempty"`
	Params    map[string]interface{} `json:"params,omitempty"`
	Result    interface{}            `json:"result,omitempty"`
	Error     string                 `json:"error,omitempty"`
	// LedgerRef optionally anchors this task to an intent/run-ledger row so the
	// GetTask read can project RE-VERIFIABLE progress (internal/relay/progress.go)
	// instead of only the self-reported State/Result. Empty until a run is bound —
	// a2aVerifiedProgress then fails closed to verdict "unknown" (a2a_progress.go).
	LedgerRef    string                 `json:"ledger_ref,omitempty"`
	CallerID     string                 `json:"caller_id,omitempty"`
	TenantID     string                 `json:"tenant_id,omitempty"`
	AgentCardURL string                 `json:"agent_card_url,omitempty"`
	Message      map[string]interface{} `json:"message,omitempty"`
	// SessionTrace binds this task to a LIVE served session (the message's
	// session_id), making cancel real (#2758): a cancel of a bound task drives the
	// session's drive-state to Draining through the same injected control seam the
	// /v1/fak/session route uses, instead of only flipping this record's State
	// field. Empty = unbound (a record-only task; cancel stays record-only).
	SessionTrace string `json:"session_trace,omitempty"`
}

// a2aMessage represents an A2A message
type a2aMessage struct {
	MessageID string                 `json:"message_id"`
	From      string                 `json:"from"`
	To        string                 `json:"to"`
	Content   map[string]interface{} `json:"content"`
	ContextID string                 `json:"context_id,omitempty"`
	Timestamp time.Time              `json:"timestamp"`
}

// a2aAgentCard represents an A2A Agent Card
type a2aAgentCard struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Version     string                 `json:"version"`
	Endpoint    string                 `json:"endpoint"`
	Skills      []a2aSkill             `json:"skills"`
	Security    a2aSecurity            `json:"security"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

type a2aSkill struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Scope       string                 `json:"scope"`
	Inputs      map[string]interface{} `json:"inputs,omitempty"`
	Outputs     map[string]interface{} `json:"outputs,omitempty"`
}

type a2aSecurity struct {
	Schemes         []a2aSecurityScheme `json:"schemes"`
	Authorization   string              `json:"authorization,omitempty"`
	TenantRequired  bool                `json:"tenant_required"`
	AuditEnabled    bool                `json:"audit_enabled"`
	QuarantineAware bool                `json:"quarantine_aware"`
}

type a2aSecurityScheme struct {
	Type        string `json:"type"`
	Description string `json:"description,omitempty"`
}

// a2aSelfBaseURL derives the absolute origin ("scheme://authority") a peer must dial
// to reach THIS process, from the request this process is answering right now.
//
// Before #5642 the served card carried a literal `https://fleet.example.com/...`.
// example.com is reserved for documentation, so it always resolves and never to fak:
// a peer that fetched the card and dialed what it said failed as a timeout or a DNS
// error attributed to the peer's own config, which is worse than serving no card.
//
// Which source wins, most authoritative first:
//
//  1. r.Host — the authority the caller actually dialed to arrive here. For a direct
//     bind this IS the listener, including the ephemeral-port case: a gateway on
//     127.0.0.1:0 advertises the port the kernel handed it, because that is the port
//     the request came in on. It is also the only source that survives a bare
//     Handler with no listener of our own.
//  2. boundAddr — the address Serve bound, used only when the request carries no
//     authority at all (an HTTP/1.0 client may omit Host). Emitting "http:///a2a"
//     there would be the same undialable-descriptor bug in a new costume.
//
// The scheme comes from the connection: https when the request arrived over TLS,
// http otherwise. We never upgrade on a guess.
//
// NOT a source: X-Forwarded-Host / X-Forwarded-Proto. This gateway deliberately does
// not trust forwarded headers (see isLoopbackRequest's note on X-Forwarded-For), and
// honoring them here would let any direct caller dictate the authority in its own
// card. Behind a reverse proxy r.Host is the proxy's Host header, which is usually
// already the public name; a deployment that terminates TLS at the proxy and needs
// https advertised wants an explicit operator override, which is a stated follow-on
// on #5642 rather than a guess made here.
func (s *Server) a2aSelfBaseURL(r *http.Request) string {
	scheme := "http"
	host := ""
	if r != nil {
		if r.TLS != nil {
			scheme = "https"
		}
		host = strings.TrimSpace(r.Host)
	}
	if !validAuthority(host) {
		host = ""
	}
	if host == "" {
		if p := s.boundAddr.Load(); p != nil {
			host = dialableAuthority(*p)
		}
	}
	if host == "" {
		// Nothing told us who we are and we own no listener. Loopback is the one
		// authority that is true of every live process, so it beats an empty one.
		host = "127.0.0.1"
	}
	return scheme + "://" + host
}

// dialableAuthority turns a bound listener address into one a peer can dial. A
// wildcard bind (":8080", "0.0.0.0:8080", "[::]:8080") names no reachable host, so
// advertising it verbatim would trade one undialable descriptor for another; the
// process is always reachable on loopback at that port, so that is what we say.
func dialableAuthority(addr string) string {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		if validAuthority(addr) {
			return addr
		}
		return ""
	}
	switch host {
	case "", "0.0.0.0", "::", "[::]":
		host = "127.0.0.1"
	}
	return net.JoinHostPort(host, port)
}

// validAuthority rejects an authority that cannot appear in a URL — a Host header is
// attacker-supplied, and splicing a stray space, control byte, or path separator into
// the endpoint would emit a malformed descriptor rather than a wrong-but-parseable one.
func validAuthority(host string) bool {
	if host == "" || len(host) > 253 {
		return false
	}
	return !strings.ContainsAny(host, "/?#\\ \t\r\n\x00")
}

// a2aIdentity is the card's (id, name) for THIS process. An A2A id is how a peer
// tells two agents apart, so a gateway that was configured with an engine identity
// says so instead of every process in the fleet answering to the same "fleet-fak"
// (#5642). An unconfigured gateway keeps the historical literals, so a peer holding
// today's expectations sees no change it did not ask for.
func (s *Server) a2aIdentity() (id, name string) {
	id, name = "fleet-fak", "Fleet fak Agent"
	if eng := strings.TrimSpace(s.engineID); eng != "" {
		id = "fleet-fak-" + eng
		name = "Fleet fak Agent (" + eng + ")"
	}
	return id, name
}

// a2aEndpointFor and a2aCardURLFor keep the card's two URL fields derived from ONE
// origin and pinned to the routes actually registered in Server.routeTable(): the
// A2A methods live under /a2a/v1, and the card is served at /a2a/v1/agent-card. The
// old literals advertised a /a2a prefix and a /a2a/agent-card path, neither of which
// this gateway has ever served.
func (s *Server) a2aEndpointFor(r *http.Request) string { return s.a2aSelfBaseURL(r) + "/a2a/v1" }

func (s *Server) a2aCardURLFor(r *http.Request) string { return s.a2aEndpointFor(r) + "/agent-card" }

// a2aAuditLog represents an audit log entry for task transitions
type a2aAuditLog struct {
	TaskID        string    `json:"task_id"`
	ContextID     string    `json:"context_id,omitempty"`
	CallerID      string    `json:"caller_id"`
	TenantID      string    `json:"tenant_id,omitempty"`
	Method        string    `json:"method,omitempty"`
	ParamsHash    string    `json:"params_hash,omitempty"`
	Transition    string    `json:"transition"` // "created", "running", "completed", "failed", "canceled"
	ArtifactPaths []string  `json:"artifact_paths,omitempty"`
	DenialReason  string    `json:"denial_reason,omitempty"`
	Timestamp     time.Time `json:"timestamp"`
}

// a2aMethodSpec represents a reviewed method in the registry
type a2aMethodSpec struct {
	Name        string
	Scope       string // "read" or "act"
	Description string
}

// a2aMethodRegistry is the reviewed method registry (#1019)
// Methods are reviewed in tools/fleet_agent_link.py build_registry()
var a2aMethodRegistry = map[string]a2aMethodSpec{
	"agent.info": {
		Name:        "agent.info",
		Scope:       "read",
		Description: "Return host, repo, tool, and method metadata.",
	},
	"agent.ping": {
		Name:        "agent.ping",
		Scope:       "read",
		Description: "Cheap liveness check.",
	},
	"protocol.manifest": {
		Name:        "protocol.manifest",
		Scope:       "read",
		Description: "Return the Fleet Agent Link method manifest.",
	},
	"laptop.check": {
		Name:        "laptop.check",
		Scope:       "act",
		Description: "Run tools/fak_laptop_test.py check with reviewed parameters.",
	},
	"laptop.status": {
		Name:        "laptop.status",
		Scope:       "read",
		Description: "Run tools/fak_laptop_test.py status against saved proof reports.",
	},
	"laptop.verify": {
		Name:        "laptop.verify",
		Scope:       "act",
		Description: "Run tools/fak_laptop_test.py verify against saved proof reports.",
	},
	"laptop.accept": {
		Name:        "laptop.accept",
		Scope:       "act",
		Description: "Run tools/fak_laptop_test.py accept.",
	},
}

var (
	a2aStore *a2aTaskStore
	a2aOnce  sync.Once
)

// A2AMethodSpecForResolver is the exported type for the capindex A2A resolver.
// This exposes the reviewed method registry as generic Capabilities, proving
// the loader is protocol-blind (issue #1108, C5).
type A2AMethodSpecForResolver struct {
	Name        string
	Scope       string // "read" or "act"
	Description string
}

// A2AMethodRegistryForResolver returns the reviewed method registry for use
// by the protocol-generic capindex A2A resolver.
func A2AMethodRegistryForResolver() []A2AMethodSpecForResolver {
	methods := make([]A2AMethodSpecForResolver, 0, len(a2aMethodRegistry))
	for _, spec := range a2aMethodRegistry {
		methods = append(methods, A2AMethodSpecForResolver{
			Name:        spec.Name,
			Scope:       spec.Scope,
			Description: spec.Description,
		})
	}
	return methods
}

// getA2AStore returns the singleton A2A task store
func getA2AStore() *a2aTaskStore {
	a2aOnce.Do(func() {
		a2aStore = &a2aTaskStore{
			tasks: make(map[string]*a2aTask),
		}
	})
	return a2aStore
}

// generateTaskID generates a stable task ID using crypto/rand
func generateTaskID() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("failed to generate task ID: %w", err)
	}
	return "a2a_" + hex.EncodeToString(b), nil
}

// hashParams creates a hash of params for audit logging (not raw secrets)
func hashParams(params map[string]interface{}) string {
	if len(params) == 0 {
		return ""
	}
	// Use JSON serialization for hashing (simple but effective for audit purposes)
	data, _ := json.Marshal(params)
	// This is a simple hash - in production you'd use a proper cryptographic hash
	return fmt.Sprintf("%x", len(data))
}

// logAuditEntry logs an audit entry for task state transitions
func (s *Server) logAuditEntry(log a2aAuditLog) {
	if s.auditLog == nil {
		return
	}
	s.auditLog(log)
}

// validateMethodAgainstRegistry validates a method name against the reviewed registry (#1019)
// Returns the method spec if valid, otherwise an error
func validateMethodAgainstRegistry(method string) (a2aMethodSpec, error) {
	spec, ok := a2aMethodRegistry[method]
	if !ok {
		return a2aMethodSpec{}, fmt.Errorf("method not in reviewed registry: %s", method)
	}
	return spec, nil
}

// handleA2ASendMessage implements POST /a2a/v1/messages
// Parses a single skill invocation, validates params, dispatches short method or creates task
func (s *Server) handleA2ASendMessage(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}

	var msg a2aMessage
	if err := decodeJSON(w, r, &msg); err != nil {
		writeErr(w, http.StatusBadRequest, "malformed message: "+err.Error())
		return
	}

	// Extract caller identity from auth headers (simplified)
	callerID := r.Header.Get("X-Caller-ID")
	if callerID == "" {
		callerID = msg.From
	}
	if callerID == "" {
		writeErr(w, http.StatusUnauthorized, "caller identity required")
		return
	}

	// #2439: resolve the TARGET before anything is dispatched. A `to` carrying the
	// kernel's lease prefix is a kernel-minted lease identity and is resolved strictly —
	// an id the kernel never minted refuses LEASE_UNKNOWN, one past its expiry refuses
	// LEASE_EXPIRED. Neither refusal falls back to name routing: that fallback IS the
	// misroute this closes, where an expired lease id is delivered to whichever agent
	// holds the underlying NAME now. A plain name still routes as before.
	if _, reason, ok := s.resolveLeaseTarget(msg.To, time.Now()); !ok {
		writeErrCode(w, http.StatusGone, strings.ToLower(reason),
			reason+": the message addresses a kernel-minted lease that is no longer deliverable; "+
				"it is refused rather than routed to the name's current holder — re-resolve the peer's live lease id")
		return
	}

	// Extract tenant from headers
	tenantID := r.Header.Get("X-Tenant-ID")

	// Validate message content
	if msg.Content == nil {
		writeErr(w, http.StatusBadRequest, "message content required")
		return
	}

	// Extract method name from content
	method, ok := msg.Content["method"].(string)
	if !ok {
		writeErr(w, http.StatusBadRequest, "method name required")
		return
	}

	// Validate method against reviewed registry (#1019)
	methodSpec, err := validateMethodAgainstRegistry(method)
	if err != nil {
		writeErr(w, http.StatusForbidden, err.Error())
		return
	}

	// Extract params from content
	var params map[string]interface{}
	if paramsRaw, ok := msg.Content["params"]; ok {
		if paramsMap, ok := paramsRaw.(map[string]interface{}); ok {
			params = paramsMap
		} else {
			writeErr(w, http.StatusBadRequest, "params must be an object")
			return
		}
	} else {
		params = make(map[string]interface{})
	}

	// Generate stable task ID
	taskID, err := generateTaskID()
	if err != nil {
		writeErr(w, http.StatusInternalServerError, "failed to generate task ID")
		return
	}

	// A session_id in the message binds the task to a live served session: the task
	// then REPRESENTS that run (it stays "running" until the run ends), and a cancel
	// of the task cancels the run itself, not just this record.
	sessionTrace, _ := msg.Content["session_id"].(string)
	sessionTrace = strings.TrimSpace(sessionTrace)

	now := time.Now()
	task := &a2aTask{
		TaskID:       taskID,
		Title:        fmt.Sprintf("A2A: %s", method),
		State:        "created",
		CreatedAt:    now,
		UpdatedAt:    now,
		Method:       method,
		Params:       params,
		CallerID:     callerID,
		TenantID:     tenantID,
		AgentCardURL: s.a2aCardURLFor(r),
		Message:      msg.Content,
		SessionTrace: sessionTrace,
	}

	// Store task (bounded: insertLocked evicts the oldest terminal task at capacity).
	store := getA2AStore()
	store.mu.Lock()
	store.insertLocked(taskID, task)
	store.mu.Unlock()

	// Log audit entry for task creation
	s.logAuditEntry(a2aAuditLog{
		TaskID:     taskID,
		ContextID:  msg.ContextID,
		CallerID:   callerID,
		TenantID:   tenantID,
		Method:     method,
		ParamsHash: hashParams(params),
		Transition: "created",
		Timestamp:  now,
	})

	// Dispatch the method call (validated against method registry #1019)
	// For now, just mark as running and complete
	task.State = "running"
	task.UpdatedAt = time.Now()

	s.logAuditEntry(a2aAuditLog{
		TaskID:     taskID,
		CallerID:   callerID,
		TenantID:   tenantID,
		Method:     method,
		Transition: "running",
		Timestamp:  task.UpdatedAt,
	})

	if sessionTrace == "" {
		// Unbound record-only task: no live run backs it, so the method dispatch is
		// still simulated (the record completes immediately).
		task.State = "completed"
		task.UpdatedAt = time.Now()
		task.Result = map[string]interface{}{
			"success": true,
			"message": fmt.Sprintf("Method %s executed successfully", method),
			"scope":   methodSpec.Scope,
		}

		s.logAuditEntry(a2aAuditLog{
			TaskID:     taskID,
			CallerID:   callerID,
			TenantID:   tenantID,
			Method:     method,
			Transition: "completed",
			Timestamp:  task.UpdatedAt,
		})
	}

	s.logf("gateway: A2A SendMessage task %s created for method %s", taskID, method)
	writeJSON(w, http.StatusCreated, map[string]interface{}{
		"task_id":    taskID,
		"state":      task.State,
		"created_at": task.CreatedAt.UTC().Format(time.RFC3339),
		"message_id": msg.MessageID,
	})
}

// handleA2AListTasks implements GET /a2a/v1/tasks
// List tasks by context/caller/tenant
func (s *Server) handleA2AListTasks(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}

	// Get query parameters
	query := r.URL.Query()
	callerID := query.Get("caller_id")
	tenantID := query.Get("tenant_id")
	contextID := query.Get("context_id")

	store := getA2AStore()
	store.mu.RLock()
	defer store.mu.RUnlock()

	var tasks []*a2aTask
	for _, task := range store.tasks {
		// Filter by query parameters
		if callerID != "" && task.CallerID != callerID {
			continue
		}
		if tenantID != "" && task.TenantID != tenantID {
			continue
		}
		if contextID != "" {
			// Check if message has context_id
			if task.Message != nil {
				if ctx, ok := task.Message["context_id"].(string); !ok || ctx != contextID {
					continue
				}
			}
		}
		tasks = append(tasks, task)
	}

	// Return task list
	taskList := make([]map[string]interface{}, 0, len(tasks))
	for _, task := range tasks {
		taskList = append(taskList, map[string]interface{}{
			"task_id":    task.TaskID,
			"title":      task.Title,
			"state":      task.State,
			"created_at": task.CreatedAt.UTC().Format(time.RFC3339),
			"updated_at": task.UpdatedAt.UTC().Format(time.RFC3339),
			"caller_id":  task.CallerID,
			"tenant_id":  task.TenantID,
		})
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"tasks": taskList,
		"count": len(taskList),
	})
}

// handleA2ATask is the subtree handler for /a2a/v1/tasks/{id}.
// GET /a2a/v1/tasks/{id} reads one task.
// POST /a2a/v1/tasks/{id}/cancel cancels one task.
// requirePathIDVerb is the routing rule every id-keyed subtree obeys: parse
// "{prefix}{id}[/{verb}]" into its leading id segment and optional verb — stripping the
// prefix, dropping a trailing slash, splitting on "/", trimming whitespace off each of the
// first two segments — and refuse an id-less request with a 400 rather than routing on the
// verb with an empty id. missing is the resource's own 400 text, so each subtree still
// names the field IT wanted; ok false means the response is already written and the
// handler must return without touching w.
func requirePathIDVerb(w http.ResponseWriter, r *http.Request, prefix, missing string) (id, verb string, ok bool) {
	rest := strings.TrimSuffix(strings.TrimPrefix(r.URL.Path, prefix), "/")
	parts := strings.Split(rest, "/")
	if len(parts) == 0 || strings.TrimSpace(parts[0]) == "" {
		writeErr(w, http.StatusBadRequest, missing)
		return "", "", false
	}
	id = strings.TrimSpace(parts[0])
	if len(parts) >= 2 {
		verb = strings.TrimSpace(parts[1])
	}
	return id, verb, true
}

func (s *Server) handleA2ATask(w http.ResponseWriter, r *http.Request) {
	// Extract path after /a2a/v1/tasks/
	taskID, verb, ok := requirePathIDVerb(w, r, "/a2a/v1/tasks/", "task_id required")
	if !ok {
		return
	}

	switch r.Method {
	case http.MethodGet:
		// GET reads one task, or reads back a registered push-notification webhook.
		switch verb {
		case "":
			s.handleA2AGetTaskByID(w, r, taskID)
		case "pushNotificationConfig":
			s.handleA2AGetPushConfig(w, r, taskID)
		default:
			writeErr(w, http.StatusMethodNotAllowed, "use GET /a2a/v1/tasks/{task_id} or GET /a2a/v1/tasks/{task_id}/pushNotificationConfig")
		}
	case http.MethodPost:
		// POST applies a verb: cancel the task, or register a push-notification webhook.
		switch verb {
		case "cancel":
			s.handleA2ACancelTaskByID(w, r, taskID)
		case "pushNotificationConfig":
			s.handleA2ASetPushConfig(w, r, taskID)
		default:
			writeErr(w, http.StatusBadRequest, "supported verbs: cancel, pushNotificationConfig")
		}
	default:
		writeErr(w, http.StatusMethodNotAllowed, "use GET or POST")
	}
}

// handleA2AGetTaskByID implements GET /a2a/v1/tasks/{task_id}
// Read task store by id
func (s *Server) handleA2AGetTaskByID(w http.ResponseWriter, r *http.Request, taskID string) {
	store := getA2AStore()
	store.mu.RLock()
	task, exists := store.tasks[taskID]
	store.mu.RUnlock()

	if !exists {
		writeErr(w, http.StatusNotFound, "task not found")
		return
	}

	// Return task snapshot. `state`/`result` are the task's own imperative record;
	// `progress` is the RE-VERIFIABLE cursor a foreign peer can trust — the
	// no-`claimed`-field VerifiedProgress shape projected across the edge. It fails
	// closed to verdict "unknown" until a run's intent-ledger anchor is bound, so a
	// peer is never handed a self-report as progress (see a2a_progress.go). No live
	// LedgerReader is wired onto the task store yet (the named next rung), so pass nil.
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"task_id":        task.TaskID,
		"title":          task.Title,
		"state":          task.State,
		"created_at":     task.CreatedAt.UTC().Format(time.RFC3339),
		"updated_at":     task.UpdatedAt.UTC().Format(time.RFC3339),
		"method":         task.Method,
		"result":         task.Result,
		"error":          task.Error,
		"caller_id":      task.CallerID,
		"tenant_id":      task.TenantID,
		"agent_card_url": task.AgentCardURL,
		"progress":       a2aVerifiedProgress(task, nil),
	})
}

// handleA2ACancelTaskByID implements POST /a2a/v1/tasks/{task_id}/cancel
// A task bound to a live served session (SessionTrace) is canceled FOR REAL: the
// session's drive-state is driven to Draining through the injected control seam (the
// same seam the /v1/fak/session route uses), so the running arm stops at its next
// boundary — the task record flips to "canceled" only after the run's cancel took.
// An unbound task keeps the record-only cancel. Fail closed: a bound task whose
// session cancel cannot be applied (no seam wired, refused, or errored) is NOT
// marked canceled — a canceled record over a still-running session is the simulation
// this handler no longer performs.
func (s *Server) handleA2ACancelTaskByID(w http.ResponseWriter, r *http.Request, taskID string) {
	store := getA2AStore()
	store.mu.Lock()
	task, exists := store.tasks[taskID]
	var sessionTrace string
	if exists {
		sessionTrace = task.SessionTrace
		// Check if task can be canceled
		if task.State == "completed" || task.State == "canceled" || task.State == "failed" {
			store.mu.Unlock()
			writeErr(w, http.StatusConflict, "task cannot be canceled in current state")
			return
		}
	}
	store.mu.Unlock()
	if !exists {
		writeErr(w, http.StatusNotFound, "task not found")
		return
	}

	// Bound task: cancel the real run first, outside the store lock (the control
	// seam may perform durable writes). Only a consumed drain justifies the record
	// flip below.
	var sessionState *SessionState
	if sessionTrace != "" {
		if s.controlSession == nil {
			writeErr(w, http.StatusConflict, "task is bound to session "+sessionTrace+" but session control is not configured; refusing a record-only cancel")
			return
		}
		st, ok, err := s.controlSession(r.Context(), sessionTrace, "run", SessionControlRequest{Run: "draining"})
		if err != nil {
			writeErr(w, http.StatusBadRequest, "session cancel failed: "+err.Error())
			return
		}
		if !ok {
			// Terminal session or lost CAS race — the closed control refusal, not a
			// silent record flip.
			writeErr(w, http.StatusConflict, "session cancel refused (terminal or stale rev); task left unchanged")
			return
		}
		sessionState = &st
	}

	store.mu.Lock()
	// notifySnap carries a copy of the just-canceled task out of the locked region so the
	// push-notification POST fires AFTER the store lock is released (this defer is registered
	// before the unlock defer, so LIFO runs it last). Nil ⇒ an early-return path committed no
	// transition, so nothing fires. See a2a_pushnotify.go.
	var notifySnap *a2aTask
	defer func() {
		if notifySnap != nil {
			s.a2aOnTaskTransition(notifySnap)
		}
	}()
	defer store.mu.Unlock()
	task, exists = store.tasks[taskID]
	if !exists {
		writeErr(w, http.StatusNotFound, "task not found")
		return
	}

	// Mark task as canceled
	task.State = "canceled"
	task.UpdatedAt = time.Now()
	task.Error = "Task canceled by request"
	if sessionState != nil {
		task.Result = map[string]interface{}{
			"session_canceled": true,
			"session_run":      sessionState.Run,
			"session_rev":      sessionState.Rev,
		}
	}
	snap := *task
	notifySnap = &snap

	// Log audit entry
	s.logAuditEntry(a2aAuditLog{
		TaskID:     taskID,
		CallerID:   task.CallerID,
		TenantID:   task.TenantID,
		Method:     task.Method,
		Transition: "canceled",
		Timestamp:  task.UpdatedAt,
	})

	s.logf("gateway: A2A task %s canceled", taskID)
	resp := map[string]interface{}{
		"task_id":    taskID,
		"state":      task.State,
		"updated_at": task.UpdatedAt.UTC().Format(time.RFC3339),
		"canceled":   true,
	}
	if sessionState != nil {
		// The proof the cancel was real: the run's post-cancel drive state, read from
		// the control seam's answer, never synthesized from the task record.
		resp["session"] = sessionState
	}
	writeJSON(w, http.StatusOK, resp)
}

// handleA2AGetExtendedAgentCard implements GET /a2a/v1/agent-card
// Return the authenticated/private card when allowed
// Skills are projected from the reviewed method registry (#1019)
func (s *Server) handleA2AGetExtendedAgentCard(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}

	// Extract caller identity from auth headers
	callerID := r.Header.Get("X-Caller-ID")
	if callerID == "" {
		writeErr(w, http.StatusUnauthorized, "caller identity required")
		return
	}

	// Get tenant from headers
	tenantID := r.Header.Get("X-Tenant-ID")

	// Build skills from the reviewed method registry (#1019)
	skills := make([]a2aSkill, 0, len(a2aMethodRegistry))
	for _, spec := range a2aMethodRegistry {
		skills = append(skills, a2aSkill{
			ID:          spec.Name,
			Name:        spec.Name,
			Description: spec.Description,
			Scope:       spec.Scope,
		})
	}

	// Generate Agent Card with skills from method registry. Identity and endpoint
	// come from THIS process — the configured engine and the request we are
	// answering — not from literals (#5642).
	cardID, cardName := s.a2aIdentity()
	card := a2aAgentCard{
		ID:          cardID,
		Name:        cardName,
		Description: "A Fleet agent with reviewed method registry and policy-scoped skills",
		Version:     a2aVersion,
		Endpoint:    s.a2aEndpointFor(r),
		Skills:      skills,
		Security: a2aSecurity{
			Schemes: []a2aSecurityScheme{
				{
					Type:        "bearer",
					Description: "Bearer token authentication",
				},
			},
			Authorization:   "Bearer token required for all operations",
			TenantRequired:  tenantID != "",
			AuditEnabled:    true,
			QuarantineAware: true,
		},
		Metadata: map[string]interface{}{
			"caller_id":       callerID,
			"tenant_id":       tenantID,
			"method_registry": "fleet",
			"policy_scopes":   []string{"read", "act"},
		},
	}

	s.logf("gateway: A2A extended agent card requested by %s (tenant: %s)", callerID, tenantID)
	writeJSON(w, http.StatusOK, card)
}

// AuditLogFunc is a function type for audit logging
type AuditLogFunc func(log a2aAuditLog)

// SetA2AAuditLog sets the audit logging function for A2A operations
func (s *Server) SetA2AAuditLog(fn func(log a2aAuditLog)) {
	s.auditLog = fn
}
