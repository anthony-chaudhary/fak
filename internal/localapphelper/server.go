package localapphelper

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/localappcontract"
)

// TaskRequest is the authenticated helper's stable app-facing request.
type TaskRequest struct {
	Schema  string          `json:"schema"`
	TaskID  string          `json:"task_id"`
	Task    string          `json:"task"`
	Payload json.RawMessage `json:"payload"`
}

type TaskResult struct {
	Events  []localappcontract.Event `json:"events"`
	Receipt localappcontract.Receipt `json:"receipt"`
}

type Executor interface {
	Execute(context.Context, TaskRequest) (TaskResult, error)
}

// Server owns only authenticated loopback transport and cancellation. The injected
// executor remains responsible for proving fak-native model execution in its receipt.
type Server struct {
	Binding    Binding
	Host       HostIdentity
	Capability []byte
	Executor   Executor
	mu         sync.Mutex
	cancels    map[string]context.CancelFunc
}

func (s *Server) Handler() (http.Handler, error) {
	if s.Executor == nil {
		return nil, errors.New("localapphelper: executor is required")
	}
	if err := s.Binding.Authorize(s.Host, s.Capability); err != nil {
		return nil, err
	}
	s.mu.Lock()
	if s.cancels == nil {
		s.cancels = map[string]context.CancelFunc{}
	}
	s.mu.Unlock()
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/tasks", s.run)
	mux.HandleFunc("DELETE /v1/tasks/{id}", s.cancel)
	return s.authenticate(mux), nil
}

func ListenLocal(addr string) (net.Listener, error) {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return nil, fmt.Errorf("localapphelper: invalid listen address: %w", err)
	}
	ip := net.ParseIP(host)
	if host != "localhost" && (ip == nil || !ip.IsLoopback()) {
		return nil, errors.New("localapphelper: listener must be loopback")
	}
	return net.Listen("tcp", addr)
}

func (s *Server) authenticate(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		if token == r.Header.Get("Authorization") || s.Binding.Authorize(s.Host, []byte(token)) != nil {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (s *Server) run(w http.ResponseWriter, r *http.Request) {
	var req TaskRequest
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20))
	dec.DisallowUnknownFields()
	if dec.Decode(&req) != nil || req.Schema != localappcontract.Schema || strings.TrimSpace(req.TaskID) == "" || strings.TrimSpace(req.Task) == "" {
		http.Error(w, "invalid task request", http.StatusBadRequest)
		return
	}
	ctx, cancel := context.WithCancel(r.Context())
	s.mu.Lock()
	if _, exists := s.cancels[req.TaskID]; exists {
		s.mu.Unlock()
		cancel()
		http.Error(w, "task already active", http.StatusConflict)
		return
	}
	s.cancels[req.TaskID] = cancel
	s.mu.Unlock()
	defer func() { cancel(); s.mu.Lock(); delete(s.cancels, req.TaskID); s.mu.Unlock() }()
	result, err := s.Executor.Execute(ctx, req)
	if err != nil {
		http.Error(w, "task execution failed", http.StatusBadGateway)
		return
	}
	if err := result.Receipt.Validate(); err != nil || result.Receipt.TaskID != req.TaskID || result.Receipt.Engine != "fak-native" {
		http.Error(w, "invalid fak-native receipt", http.StatusBadGateway)
		return
	}
	if err := localappcontract.Replay(result.Events); err != nil {
		http.Error(w, "invalid event stream", http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func (s *Server) cancel(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	cancel, ok := s.cancels[r.PathValue("id")]
	s.mu.Unlock()
	if !ok {
		http.Error(w, "task not active", http.StatusNotFound)
		return
	}
	cancel()
	w.WriteHeader(http.StatusNoContent)
}
