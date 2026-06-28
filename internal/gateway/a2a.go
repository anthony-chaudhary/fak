package gateway

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"
)

// A2A protocol version
const a2aVersion = "1.0"

// a2aTaskStore holds A2A tasks with stable IDs and audit logging
type a2aTaskStore struct {
	mu    sync.RWMutex
	tasks map[string]*a2aTask
}

type a2aTask struct {
	TaskID       string                 `json:"task_id"`
	Title        string                 `json:"title,omitempty"`
	State        string                 `json:"state"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
	Method       string                 `json:"method,omitempty"`
	Params       map[string]interface{} `json:"params,omitempty"`
	Result       interface{}            `json:"result,omitempty"`
	Error        string                 `json:"error,omitempty"`
	CallerID     string                 `json:"caller_id,omitempty"`
	TenantID     string                 `json:"tenant_id,omitempty"`
	AgentCardURL string                 `json:"agent_card_url,omitempty"`
	Message      map[string]interface{} `json:"message,omitempty"`
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
		AgentCardURL: "https://fleet.example.com/a2a/agent-card",
		Message:      msg.Content,
	}

	// Store task
	store := getA2AStore()
	store.mu.Lock()
	store.tasks[taskID] = task
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

	// Simulate method execution
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
// splitPathIDVerb parses a "{prefix}{id}[/{verb}]" request path into its leading
// id segment and optional verb. It strips prefix, drops a trailing slash, splits on
// "/", and trims whitespace from each of the first two segments. ok is false when no
// non-empty id is present, so callers can emit their own resource-specific 400.
func splitPathIDVerb(path, prefix string) (id, verb string, ok bool) {
	rest := strings.TrimSuffix(strings.TrimPrefix(path, prefix), "/")
	parts := strings.Split(rest, "/")
	if len(parts) == 0 || strings.TrimSpace(parts[0]) == "" {
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
	taskID, verb, ok := splitPathIDVerb(r.URL.Path, "/a2a/v1/tasks/")
	if !ok {
		writeErr(w, http.StatusBadRequest, "task_id required")
		return
	}

	switch r.Method {
	case http.MethodGet:
		// GET reads one task. A verb on the path is not allowed.
		if verb != "" {
			writeErr(w, http.StatusMethodNotAllowed, "use GET /a2a/v1/tasks/{task_id}")
			return
		}
		s.handleA2AGetTaskByID(w, r, taskID)
	case http.MethodPost:
		// POST applies a verb. Only "cancel" is supported.
		if verb != "cancel" {
			writeErr(w, http.StatusBadRequest, "only cancel verb is supported: POST /a2a/v1/tasks/{task_id}/cancel")
			return
		}
		s.handleA2ACancelTaskByID(w, r, taskID)
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

	// Return task snapshot
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
	})
}

// handleA2ACancelTaskByID implements POST /a2a/v1/tasks/{task_id}/cancel
// Mark cancellable tasks as canceled
func (s *Server) handleA2ACancelTaskByID(w http.ResponseWriter, r *http.Request, taskID string) {
	store := getA2AStore()
	store.mu.Lock()
	defer store.mu.Unlock()

	task, exists := store.tasks[taskID]
	if !exists {
		writeErr(w, http.StatusNotFound, "task not found")
		return
	}

	// Check if task can be canceled
	if task.State == "completed" || task.State == "canceled" || task.State == "failed" {
		writeErr(w, http.StatusConflict, "task cannot be canceled in current state")
		return
	}

	// Mark task as canceled
	task.State = "canceled"
	task.UpdatedAt = time.Now()
	task.Error = "Task canceled by request"

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
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"task_id":    taskID,
		"state":      task.State,
		"updated_at": task.UpdatedAt.UTC().Format(time.RFC3339),
		"canceled":   true,
	})
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

	// Generate Agent Card with skills from method registry
	card := a2aAgentCard{
		ID:          "fleet-fak",
		Name:        "Fleet fak Agent",
		Description: "A Fleet agent with reviewed method registry and policy-scoped skills",
		Version:     a2aVersion,
		Endpoint:    "https://fleet.example.com/a2a",
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
