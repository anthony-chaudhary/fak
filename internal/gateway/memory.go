package gateway

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/memq"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

// The memq surface over MCP/HTTP: an agent self-navigates its own memory by listing
// the composable strategies, EXPLAINing a plan, and running a driver or an authored
// Query against a recall core image (or the in-memory demo corpus). This is the
// "build SQL, not a query" substrate exposed to a NON-Go agent — the agent emits the
// strategy, the kernel executes it under the same fail-closed posture the CLI uses.

// MemoryRequest is the arguments object for fak_memory_explain / fak_memory_run. Supply
// either a built-in `driver` name or an inline authored `query`; parameterize with
// intent/k/budget; point `run` at a recall image with `image_dir` (default: the
// in-memory demo corpus). `apply` is honored only by run and only enacts the safe
// negative-only / storage mutations (tombstone, prune); the default is dry-run.
type MemoryRequest struct {
	Driver   string      `json:"driver,omitempty"`
	Query    *memq.Query `json:"query,omitempty"`
	Intent   string      `json:"intent,omitempty"`
	K        int         `json:"k,omitempty"`
	Budget   int64       `json:"budget,omitempty"`
	ImageDir string      `json:"image_dir,omitempty"`
	Apply    bool        `json:"apply,omitempty"`
}

func (r MemoryRequest) resolveQuery() (memq.Query, error) {
	if r.Query != nil {
		q := *r.Query
		if q.Intent == "" {
			q.Intent = r.Intent
		}
		return q, nil
	}
	if strings.TrimSpace(r.Driver) == "" {
		return memq.Query{}, errors.New("memory request needs a driver or an inline query")
	}
	d, ok := memq.Get(r.Driver)
	if !ok {
		return memq.Query{}, fmt.Errorf("unknown memory driver %q", r.Driver)
	}
	return d.Build(memq.Params{Intent: r.Intent, K: r.K, Budget: r.Budget}), nil
}

// MemoryDriverInfo is one registered strategy plus its compiled plan (so a client can
// see the pipeline, not just the name).
type MemoryDriverInfo struct {
	Name string    `json:"name"`
	Doc  string    `json:"doc"`
	Plan memq.Plan `json:"plan"`
}

// memoryDrivers lists every registered strategy with an example compiled plan.
func (s *Server) memoryDrivers() []MemoryDriverInfo {
	ds := memq.Drivers()
	out := make([]MemoryDriverInfo, 0, len(ds))
	for _, d := range ds {
		out = append(out, MemoryDriverInfo{Name: d.Name, Doc: d.Doc, Plan: memq.Explain(d.Build(memq.Params{Intent: "the task at hand"}))})
	}
	return out
}

// memoryExplain renders a request's query as a plan without touching any backend.
func (s *Server) memoryExplain(req MemoryRequest) (memq.Plan, error) {
	q, err := req.resolveQuery()
	if err != nil {
		return memq.Plan{}, err
	}
	return memq.Explain(q), nil
}

// memoryRun executes a request's query against the chosen backend. Mutations apply
// only when req.Apply is set (and then only the safe tombstone/prune effects); the
// default is a dry-run proposal — the same fail-closed default the CLI enforces.
func (s *Server) memoryRun(ctx context.Context, req MemoryRequest) (memq.Result, error) {
	q, err := req.resolveQuery()
	if err != nil {
		return memq.Result{}, err
	}
	var backend memq.Backend
	if dir := strings.TrimSpace(req.ImageDir); dir != "" {
		sess, err := recall.Load(dir)
		if err != nil {
			return memq.Result{}, fmt.Errorf("load core image: %w", err)
		}
		backend = memq.NewRecallBackend(sess, dir)
	} else {
		backend = memq.NewDemoStore()
	}
	caps := memq.Caps{}
	if req.Apply {
		caps = memq.AllowAll()
	}
	res, err := memq.Run(ctx, backend, q, caps)
	if err != nil {
		return memq.Result{}, err
	}
	s.logf("gateway: memory run driver=%q apply=%v image=%q rendered=%d effects=%d/%d",
		req.Driver, req.Apply, req.ImageDir, res.Stats.Rendered, res.Stats.EffectsApplied, res.Stats.EffectsProposed)
	return res, nil
}
