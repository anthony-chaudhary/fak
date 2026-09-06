package mcpbroker

import (
	"context"
	"encoding/json"
	"errors"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Common error definitions returned by the broker.
var (
	// ErrBrokerClosed is returned when attempting an operation on a closed broker.
	ErrBrokerClosed = errors.New("mcpbroker: broker is closed")

	// ErrInvalidToolName is returned when a tool registration or call has an empty or invalid tool name.
	ErrInvalidToolName = errors.New("mcpbroker: invalid or empty tool name")

	// ErrToolNotFound is returned when attempting to route a call to an unregistered tool.
	ErrToolNotFound = errors.New("mcpbroker: tool not found")

	// ErrToolAlreadyRegistered is returned when a tool name conflicts with an existing registration.
	ErrToolAlreadyRegistered = errors.New("mcpbroker: tool already registered")

	// ErrToolDenied is returned when a tool is blocked by server-level denylist policy.
	ErrToolDenied = errors.New("mcpbroker: tool is denied by server policy")

	// ErrToolNotAllowed is returned when a tool is not present in a server-level allowlist policy.
	ErrToolNotAllowed = errors.New("mcpbroker: tool is not permitted by server allowlist")

	// ErrServerReadOnly is returned when a mutating tool is registered under a read-only server.
	ErrServerReadOnly = errors.New("mcpbroker: cannot register mutating tool on read-only server")

	// ErrServerNotFound is returned when referencing a server ID that does not exist.
	ErrServerNotFound = errors.New("mcpbroker: server not found")
)

// ServerConfig configures an upstream MCP server and defines policy boundaries
// enforced by the broker.
type ServerConfig struct {
	// ID is the unique identifier for the upstream MCP server.
	ID string `json:"id"`

	// Name is a human-readable display name for the server.
	Name string `json:"name"`

	// Command is the executable command or endpoint URL for the MCP server.
	Command string `json:"command,omitempty"`

	// Args contains optional command-line arguments passed to the server process.
	Args []string `json:"args,omitempty"`

	// Env specifies optional environment variables in KEY=VALUE format.
	Env []string `json:"env,omitempty"`

	// Timeout specifies the maximum duration allowed for tool calls to this server.
	// If zero or negative, no server-specific deadline is applied.
	Timeout time.Duration `json:"timeout,omitempty"`

	// AllowedTools specifies an explicit allowlist of tool names. If non-empty,
	// only listed tools may be registered or called for this server.
	AllowedTools []string `json:"allowed_tools,omitempty"`

	// DeniedTools specifies a denylist of tool names that are prohibited from
	// being registered or executed.
	DeniedTools []string `json:"denied_tools,omitempty"`

	// ReadOnly indicates whether the server is restricted to non-mutating operations.
	ReadOnly bool `json:"read_only,omitempty"`

	// MaxRestarts specifies the maximum number of restart attempts on unexpected crashes.
	MaxRestarts *int `json:"max_restarts,omitempty"`
}

// ToolHandler defines the execution signature for an MCP tool call.
type ToolHandler func(ctx context.Context, req CallRequest) (*CallResponse, error)

// SecurityFilter defines a security inspection hook that evaluates whether a tool call
// is permitted under active security policies. It returns allowed=true if permitted, or
// allowed=false along with a human-readable rejection reason.
type SecurityFilter func(ctx context.Context, req CallRequest, reg ToolRegistration) (allowed bool, reason string)

// ToolRegistration defines an MCP tool registered with the broker, including its
// metadata, schema, security annotations, and execution handler.
type ToolRegistration struct {
	// Name is the unique name of the tool as exposed to callers.
	Name string `json:"name"`

	// ServerID identifies the upstream MCP server hosting or providing this tool.
	ServerID string `json:"server_id"`

	// Description is a human-readable summary of what the tool does.
	Description string `json:"description"`

	// InputSchema contains the JSON Schema specification for the tool's parameters.
	InputSchema json.RawMessage `json:"input_schema,omitempty"`

	// ReadOnly marks this specific tool as a read-only (non-mutating) tool.
	ReadOnly bool `json:"read_only,omitempty"`

	// Handler executes the tool call. If nil, a default echo/mock response is returned.
	Handler ToolHandler `json:"-"`

	// SecurityFilter is an optional per-tool security filter executed prior to invocation.
	SecurityFilter SecurityFilter `json:"-"`
}

// CallRequest represents a request to invoke an MCP tool through the broker.
type CallRequest struct {
	// SessionID identifies the client session making the call.
	SessionID string `json:"session_id,omitempty"`

	// ServerID optionally identifies the target server; if empty, the broker routes by Tool name.
	ServerID string `json:"server_id,omitempty"`

	// Tool is the name of the tool to be invoked.
	Tool string `json:"tool"`

	// Arguments contains the raw JSON-encoded parameters for the tool invocation.
	Arguments json.RawMessage `json:"arguments,omitempty"`

	// Metadata carries optional arbitrary contextual tags or attributes.
	Metadata map[string]string `json:"metadata,omitempty"`

	// Compression optionally specifies the compression policy for this call (e.g. CompressionIdentity).
	// When set to CompressionIdentity, structured compression is bypassed for this call.
	Compression CompressionPolicy `json:"compression,omitempty"`
}

// CallResponse represents the outcome of an MCP tool invocation routed through the broker.
type CallResponse struct {
	// Tool is the name of the tool that was invoked.
	Tool string `json:"tool"`

	// ServerID is the upstream server that executed the tool call.
	ServerID string `json:"server_id"`

	// Content contains the output payload or text returned by the tool.
	Content json.RawMessage `json:"content,omitempty"`

	// IsError indicates whether the tool execution returned an error condition.
	IsError bool `json:"is_error,omitempty"`

	// Filtered indicates whether the call was blocked by security policy filtering.
	Filtered bool `json:"filtered,omitempty"`

	// FilterReason explains why the call was blocked when Filtered is true.
	FilterReason string `json:"filter_reason,omitempty"`

	// ErrorMessage contains the error message if the call failed or encountered a runtime error.
	ErrorMessage string `json:"error_message,omitempty"`

	// Latency is the measured execution time of the call within the broker.
	Latency time.Duration `json:"latency"`

	// CompressionReceipt carries the structured compression decision receipt, if evaluated.
	CompressionReceipt *CompressionReceipt `json:"compression_receipt,omitempty"`

	// Metadata carries optional arbitrary contextual tags or attributes.
	Metadata map[string]string `json:"metadata,omitempty"`
}

// BrokerStats captures running operational counters and metrics for the broker.
type BrokerStats struct {
	// RegisteredTools is the current count of registered tools.
	RegisteredTools int `json:"registered_tools"`

	// RegisteredServers is the current count of configured upstream servers.
	RegisteredServers int `json:"registered_servers"`

	// TotalCalls is the cumulative number of tool call requests received.
	TotalCalls int64 `json:"total_calls"`

	// AllowedCalls is the number of tool calls that passed policy and were executed.
	AllowedCalls int64 `json:"allowed_calls"`

	// FilteredCalls is the number of tool calls denied or filtered by security policies.
	FilteredCalls int64 `json:"filtered_calls"`

	// ErrorCalls is the number of tool calls that resulted in execution errors.
	ErrorCalls int64 `json:"error_calls"`

	// ActiveSessions is the count of distinct active sessions tracked by the broker.
	ActiveSessions int `json:"active_sessions"`
}

// BrokerOption defines functional configuration options for a Broker.
type BrokerOption func(*Broker)

// WithGlobalSecurityFilter sets a global security policy filter evaluated on all tool calls.
func WithGlobalSecurityFilter(filter SecurityFilter) BrokerOption {
	return func(b *Broker) {
		b.globalFilter = filter
	}
}

// WithDefaultTimeout sets a fallback execution timeout applied when no server-specific
// timeout is configured.
func WithDefaultTimeout(d time.Duration) BrokerOption {
	return func(b *Broker) {
		b.defaultTimeout = d
	}
}

// Broker mediates access to MCP tool servers, enforcing security policies, routing
// execution requests, and collecting operational statistics.
type Broker struct {
	mu                 sync.RWMutex
	closed             bool
	servers            map[string]ServerConfig
	tools              map[string]ToolRegistration
	sessions           map[string]time.Time
	sessionCompression map[string]CompressionPolicy
	supervisors        map[string]*ProcessSupervisor
	globalFilter       SecurityFilter
	defaultTimeout     time.Duration

	// Atomic telemetry counters
	totalCalls    int64
	allowedCalls  int64
	filteredCalls int64
	errorCalls    int64
}

// NewBroker initializes and returns a new Broker ready to accept registrations and calls.
func NewBroker(opts ...BrokerOption) *Broker {
	b := &Broker{
		servers:            make(map[string]ServerConfig),
		tools:              make(map[string]ToolRegistration),
		sessions:           make(map[string]time.Time),
		sessionCompression: make(map[string]CompressionPolicy),
		supervisors:        make(map[string]*ProcessSupervisor),
	}
	for _, opt := range opts {
		opt(b)
	}
	return b
}

// RegisterServer configures an upstream MCP server in the broker, establishing its
// policy constraints (allowlist, denylist, read-only status, and execution timeout).
func (b *Broker) RegisterServer(cfg ServerConfig) error {
	if cfg.ID == "" {
		return errors.New("mcpbroker: server ID cannot be empty")
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return ErrBrokerClosed
	}

	b.servers[cfg.ID] = cfg
	return nil
}

// RegisterTool adds a tool registration to the broker. If the tool specifies a ServerID,
// it is validated against the server's policy constraints (denylist, allowlist, and read-only).
// Returns an error if the tool name is empty, if the broker is closed, or if registration
// violates server policy.
func (b *Broker) RegisterTool(reg ToolRegistration) error {
	if reg.Name == "" {
		return ErrInvalidToolName
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return ErrBrokerClosed
	}

	if _, exists := b.tools[reg.Name]; exists {
		return ErrToolAlreadyRegistered
	}

	// Validate against server policy if server is registered.
	if reg.ServerID != "" {
		if srv, exists := b.servers[reg.ServerID]; exists {
			_, rawName, isNamespaced := ParseNamespacedTool(reg.Name)
			// Check denylist
			for _, denied := range srv.DeniedTools {
				if denied == reg.Name || (isNamespaced && denied == rawName) {
					return ErrToolDenied
				}
			}

			// Check allowlist
			if len(srv.AllowedTools) > 0 {
				allowed := false
				for _, allow := range srv.AllowedTools {
					if allow == reg.Name || (isNamespaced && allow == rawName) {
						allowed = true
						break
					}
				}
				if !allowed {
					return ErrToolNotAllowed
				}
			}

			// Check read-only server restriction
			if srv.ReadOnly && !reg.ReadOnly {
				return ErrServerReadOnly
			}
		}
	}

	b.tools[reg.Name] = reg
	return nil
}

// ListTools returns a slice of all currently registered tools, sorted lexicographically
// by tool name for deterministic output. Returns an empty slice if no tools are registered
// or if the broker is closed.
func (b *Broker) ListTools() []ToolRegistration {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if b.closed || len(b.tools) == 0 {
		return []ToolRegistration{}
	}

	out := make([]ToolRegistration, 0, len(b.tools))
	for _, t := range b.tools {
		out = append(out, t)
	}

	sort.Slice(out, func(i, j int) bool {
		return out[i].Name < out[j].Name
	})

	return out
}

// RouteCall inspects, filters, and routes an incoming tool call request.
//
// Processing order:
//  1. Validate broker state and request parameters.
//  2. Resolve target tool registration.
//  3. Apply server-level and global execution timeouts.
//  4. Evaluate global security filter. If denied, return filtered response immediately.
//  5. Evaluate tool-specific security filter. If denied, return filtered response immediately.
//  6. Execute tool handler (or return default echo response if handler is nil).
//  7. Record execution latency and update operational metrics.
func (b *Broker) RouteCall(ctx context.Context, req CallRequest) (*CallResponse, error) {
	if ctx == nil {
		ctx = context.Background()
	}

	start := time.Now()

	b.mu.RLock()
	if b.closed {
		b.mu.RUnlock()
		return nil, ErrBrokerClosed
	}

	if req.Tool == "" {
		b.mu.RUnlock()
		return nil, ErrInvalidToolName
	}

	tool, found := b.tools[req.Tool]
	var srv ServerConfig
	var hasSrv bool
	if found && tool.ServerID != "" {
		srv, hasSrv = b.servers[tool.ServerID]
	}
	b.mu.RUnlock()

	atomic.AddInt64(&b.totalCalls, 1)

	// Record session activity and resolve session compression policy
	var sessionPolicy CompressionPolicy
	if req.SessionID != "" {
		b.mu.Lock()
		b.sessions[req.SessionID] = start
		if b.sessionCompression == nil {
			b.sessionCompression = make(map[string]CompressionPolicy)
		}
		for k, v := range req.Metadata {
			if strings.EqualFold(strings.TrimSpace(k), "session_compression") {
				if IsCompressionOptOut(v) {
					b.sessionCompression[req.SessionID] = CompressionIdentity
				} else {
					b.sessionCompression[req.SessionID] = CompressionAuto
				}
				break
			}
		}
		sessionPolicy = b.sessionCompression[req.SessionID]
		b.mu.Unlock()
	}

	// Forward metadata and effective compression policy onto context so downstream
	// handlers, filters, and transports observe the caller's compression preference.
	ctx = WithCallMetadata(ctx, req.Metadata)
	effPolicy := ResolveEffectiveCompression(ctx, req, sessionPolicy)
	ctx = WithCompressionPolicy(ctx, effPolicy)

	if !found {
		atomic.AddInt64(&b.errorCalls, 1)
		resp := &CallResponse{
			Tool:         req.Tool,
			IsError:      true,
			ErrorMessage: "tool not found: " + req.Tool,
			Latency:      time.Since(start),
		}
		return resp, ErrToolNotFound
	}

	// Determine timeout deadline
	timeout := b.defaultTimeout
	if hasSrv && srv.Timeout > 0 {
		timeout = srv.Timeout
	}

	var cancel context.CancelFunc
	if timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}

	// Evaluate global security filter
	if b.globalFilter != nil {
		if allowed, reason := b.globalFilter(ctx, req, tool); !allowed {
			atomic.AddInt64(&b.filteredCalls, 1)
			return &CallResponse{
				Tool:         tool.Name,
				ServerID:     tool.ServerID,
				Filtered:     true,
				FilterReason: reason,
				Latency:      time.Since(start),
			}, nil
		}
	}

	// Evaluate tool-specific security filter
	if tool.SecurityFilter != nil {
		if allowed, reason := tool.SecurityFilter(ctx, req, tool); !allowed {
			atomic.AddInt64(&b.filteredCalls, 1)
			return &CallResponse{
				Tool:         tool.Name,
				ServerID:     tool.ServerID,
				Filtered:     true,
				FilterReason: reason,
				Latency:      time.Since(start),
			}, nil
		}
	}

	// Check context cancellation before invocation
	if err := ctx.Err(); err != nil {
		atomic.AddInt64(&b.errorCalls, 1)
		return &CallResponse{
			Tool:         tool.Name,
			ServerID:     tool.ServerID,
			IsError:      true,
			ErrorMessage: err.Error(),
			Latency:      time.Since(start),
		}, err
	}

	// Execute handler
	if tool.Handler != nil {
		resp, err := tool.Handler(ctx, req)
		latency := time.Since(start)
		if err != nil {
			atomic.AddInt64(&b.errorCalls, 1)
			if resp == nil {
				resp = &CallResponse{
					Tool:         tool.Name,
					ServerID:     tool.ServerID,
					IsError:      true,
					ErrorMessage: err.Error(),
					Latency:      latency,
				}
			} else {
				resp.IsError = true
				if resp.ErrorMessage == "" {
					resp.ErrorMessage = err.Error()
				}
				resp.Latency = latency
			}
			return resp, err
		}

		atomic.AddInt64(&b.allowedCalls, 1)
		if resp == nil {
			resp = &CallResponse{}
		}
		if resp.Tool == "" {
			resp.Tool = tool.Name
		}
		if resp.ServerID == "" {
			resp.ServerID = tool.ServerID
		}
		resp.Latency = latency
		return resp, nil
	}

	// Default fallback handler for stub/echo execution
	atomic.AddInt64(&b.allowedCalls, 1)
	content := req.Arguments
	if len(content) == 0 {
		content = json.RawMessage(`{"status":"ok"}`)
	}

	return &CallResponse{
		Tool:     tool.Name,
		ServerID: tool.ServerID,
		Content:  content,
		Latency:  time.Since(start),
	}, nil
}

// Stats returns a point-in-time snapshot of the broker's operational counters.
func (b *Broker) Stats() BrokerStats {
	b.mu.RLock()
	regTools := len(b.tools)
	regServers := len(b.servers)
	activeCount := len(b.sessions)
	b.mu.RUnlock()

	return BrokerStats{
		RegisteredTools:   regTools,
		RegisteredServers: regServers,
		TotalCalls:        atomic.LoadInt64(&b.totalCalls),
		AllowedCalls:      atomic.LoadInt64(&b.allowedCalls),
		FilteredCalls:     atomic.LoadInt64(&b.filteredCalls),
		ErrorCalls:        atomic.LoadInt64(&b.errorCalls),
		ActiveSessions:    activeCount,
	}
}

// SetSessionCompression configures the default compression policy for a session ID.
func (b *Broker) SetSessionCompression(sessionID string, policy CompressionPolicy) {
	if sessionID == "" {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.sessionCompression == nil {
		b.sessionCompression = make(map[string]CompressionPolicy)
	}
	b.sessionCompression[sessionID] = policy
}

// GetSessionCompression retrieves the configured compression policy for a session ID, if any.
func (b *Broker) GetSessionCompression(sessionID string) (CompressionPolicy, bool) {
	if sessionID == "" {
		return "", false
	}
	b.mu.RLock()
	defer b.mu.RUnlock()
	if b.sessionCompression == nil {
		return "", false
	}
	p, ok := b.sessionCompression[sessionID]
	return p, ok
}

// UnregisterTool removes a registered tool by name. Returns true if the tool was found and removed.
func (b *Broker) UnregisterTool(name string) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return false
	}
	_, exists := b.tools[name]
	if exists {
		delete(b.tools, name)
	}
	return exists
}

// UnregisterServerTools removes all tools associated with serverID. Returns the count of removed tools.
func (b *Broker) UnregisterServerTools(serverID string) int {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return 0
	}
	count := 0
	for name, tool := range b.tools {
		if tool.ServerID == serverID {
			delete(b.tools, name)
			count++
		}
	}
	return count
}

// RegisterSupervisor registers a configured ProcessSupervisor with the broker under its server ID.
func (b *Broker) RegisterSupervisor(sup *ProcessSupervisor) error {
	if sup == nil {
		return errors.New("mcpbroker: nil supervisor")
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.closed {
		return ErrBrokerClosed
	}
	b.supervisors[sup.cfg.ID] = sup
	return nil
}

// LaunchSupervisor creates, starts, and supervises an MCP server using ServerConfig,
// automatically discovering and registering its tools into the broker with namespacing.
func (b *Broker) LaunchSupervisor(ctx context.Context, cfg ServerConfig) (*ProcessSupervisor, error) {
	if cfg.ID == "" {
		return nil, errors.New("mcpbroker: server ID cannot be empty")
	}

	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil, ErrBrokerClosed
	}
	b.servers[cfg.ID] = cfg
	b.mu.Unlock()

	sup := NewProcessSupervisor(cfg, b)
	if err := sup.Start(ctx); err != nil {
		return nil, err
	}

	b.mu.Lock()
	b.supervisors[cfg.ID] = sup
	b.mu.Unlock()

	return sup, nil
}

// GetSupervisor retrieves the supervisor for the given server ID, if any.
func (b *Broker) GetSupervisor(serverID string) (*ProcessSupervisor, bool) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	sup, ok := b.supervisors[serverID]
	return sup, ok
}

// StopSupervisor gracefully stops and removes the supervisor for the given server ID,
// and unregisters its associated tools.
func (b *Broker) StopSupervisor(serverID string) error {
	b.mu.Lock()
	sup, ok := b.supervisors[serverID]
	if ok {
		delete(b.supervisors, serverID)
	}
	b.mu.Unlock()

	if !ok {
		return ErrServerNotFound
	}

	_ = sup.Stop()
	b.UnregisterServerTools(serverID)
	return nil
}

// Close gracefully closes the broker, preventing subsequent registrations and routing,
// and terminates all running supervised MCP server processes. Calling Close on an already
// closed broker is a no-op.
func (b *Broker) Close() error {
	b.mu.Lock()
	if b.closed {
		b.mu.Unlock()
		return nil
	}

	b.closed = true

	sups := make([]*ProcessSupervisor, 0, len(b.supervisors))
	for _, s := range b.supervisors {
		sups = append(sups, s)
	}
	b.supervisors = make(map[string]*ProcessSupervisor)
	b.mu.Unlock()

	for _, s := range sups {
		_ = s.Stop()
	}

	cleanupBrokerRestore(b)

	return nil
}
