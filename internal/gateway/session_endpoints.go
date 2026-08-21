package gateway

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardvars"
)

// session_endpoints.go — the live "who + where" of a guarded session, exposed on
// /debug/vars so an operator pane can show which Claude ACCOUNTS and which serving
// NODES the session is actually using (fak guard, epic "status area"). The gateway
// discards the account identity and the upstream base URL after New (it keeps only
// engineID/model), so the host RE-supplies them through a pull provider — the same
// seam pattern SetHarnessMetricsProvider uses, but rendered as a structured
// /debug/vars block (like ModelLoad) rather than Prometheus text.
//
// It is a PULL provider (called on each /debug/vars scrape) on purpose: a session's
// ACTIVE account can change mid-run when an account-scoped 403 forces a failover onto
// a sibling seat, so a boot-time snapshot would go stale. Re-reading the roster each
// scrape (a cheap local FS glob on the host side) keeps the active/walled marks live.
// Everything here is display metadata — seat names, an email, a login word, an
// endpoint host — never a token or a request payload, so the payload-free /debug/vars
// contract holds.

// SessionEndpoints is the accounts + nodes a live guarded session is using.
type SessionEndpoints struct {
	Accounts []SessionAccount `json:"accounts,omitempty"`
	Nodes    []SessionNode    `json:"nodes,omitempty"`
}

// SessionAccount is one Claude subscription seat in the on-box roster, marked with how
// this session relates to it: Active (the seat currently serving turns), Walled (a
// failover proved this seat's org/region/billing walled this session and skipped it),
// and the login readiness. Email is advisory display identity, never the credential.
type SessionAccount struct {
	Name        string `json:"name"`
	Email       string `json:"email,omitempty"`
	Active      bool   `json:"active,omitempty"`
	Walled      bool   `json:"walled,omitempty"`
	CanServe    bool   `json:"can_serve"`
	LoginStatus string `json:"login_status,omitempty"`
}

// SessionNode is one place the session runs. Every guarded session has at least two —
// the KERNEL node (this host, where fak guard + the agent + adjudication run) and a
// SERVING node (where inference runs) — so "multiple nodes" is honest by construction.
// Kind distinguishes the serving posture: proxy (a provider API), remote-serve (a lab
// `fak serve` box), local-server (a detected Ollama/LM Studio/llama.cpp), or in-kernel
// (fak decodes on this box itself); host is the kernel node.
type SessionNode struct {
	Role   string `json:"role"` // "kernel" | "serving"
	ID     string `json:"id"`   // host / endpoint / "in-kernel"
	Kind   string `json:"kind"` // "host" | "proxy" | "remote-serve" | "local-server" | "in-kernel"
	Detail string `json:"detail,omitempty"`
}

// SetSessionEndpointsProvider installs the pull source for the live accounts+nodes
// /debug/vars block. fak guard wires it to a closure over the discovered account roster
// (marked active/walled from its failover state) and the resolved serving nodes; the
// default `fak serve` path never sets it, so the block stays absent there. Passing nil
// detaches it. Safe for concurrent use and on a nil Server.
func (s *Server) SetSessionEndpointsProvider(fn func() SessionEndpoints) {
	if s == nil {
		return
	}
	s.endpointsMu.Lock()
	s.endpointsProvider = fn
	s.endpointsMu.Unlock()
}

// sessionEndpoints pulls the current accounts+nodes snapshot for /debug/vars. It
// returns ok=false when no provider is set OR the provider has nothing to report, so
// the debug block is omitted (cold gateway / plain serve) rather than emitted empty.
func (s *Server) sessionEndpoints() (SessionEndpoints, bool) {
	if s == nil {
		return SessionEndpoints{}, false
	}
	s.endpointsMu.Lock()
	fn := s.endpointsProvider
	s.endpointsMu.Unlock()
	if fn == nil {
		return SessionEndpoints{}, false
	}
	ep := fn()
	if len(ep.Accounts) == 0 && len(ep.Nodes) == 0 {
		return SessionEndpoints{}, false
	}
	return ep, true
}

// SessionHarness is a compact, structured projection of the guard harness-resource
// sampler for the live /debug/vars "harness" block — the twin of the /metrics-only
// fak_harness_* family (SetHarnessMetricsProvider), so a pane reading /debug/vars can
// show the kernel/agent CPU/RSS/IO/net the exit summary prints, live. Fields are
// zero-valued (and omitted) when their axis was not observed. Defined here as a
// gateway-local shape so the gateway keeps no dependency on internal/harnessres; the
// host converts its own Snapshot into this.
type SessionHarness struct {
	Samples            int     `json:"samples"`
	ElapsedSeconds     float64 `json:"elapsed_seconds,omitempty"`
	KernelCPUSeconds   float64 `json:"kernel_cpu_seconds,omitempty"`
	KernelCPUPercent   float64 `json:"kernel_cpu_percent,omitempty"`
	KernelRSSBytes     uint64  `json:"kernel_rss_bytes,omitempty"`
	KernelIOReadBytes  uint64  `json:"kernel_io_read_bytes,omitempty"`
	KernelIOWriteBytes uint64  `json:"kernel_io_write_bytes,omitempty"`
	NetRxBytes         uint64  `json:"net_rx_bytes,omitempty"`
	NetTxBytes         uint64  `json:"net_tx_bytes,omitempty"`
	GoroutinesPeak     int     `json:"goroutines_peak,omitempty"`
	GoHeapSysBytes     uint64  `json:"go_heap_sys_bytes,omitempty"`
	GPUVRAMUsedBytes   uint64  `json:"gpu_vram_used_bytes,omitempty"`
	GPUVRAMTotalBytes  uint64  `json:"gpu_vram_total_bytes,omitempty"`
	HaveGPU            bool    `json:"have_gpu,omitempty"`
}

// SessionHarnessObservation is the typed pull result used by
// /v1/fak/observation. Availability distinguishes an observed sample from an
// empty, stale, unavailable, or unsupported source without exposing process
// payloads. ObservedAt and Revision are optional source clocks; the gateway
// supplies the HTTP snapshot boundary when both are absent.
type SessionHarnessObservation struct {
	Snapshot     SessionHarness
	Availability guardvars.Availability
	ObservedAt   time.Time
	Revision     string
}

// SetSessionHarnessProvider installs the pull source for the live harness-resource
// /debug/vars block. fak guard wires it to its running *harnessres.Sampler (converted
// to a SessionHarness per scrape); the default serve path leaves it unset so the block
// is absent. Passing nil detaches it. Safe for concurrent use and on a nil Server.
func (s *Server) SetSessionHarnessProvider(fn func() SessionHarness) {
	if s == nil {
		return
	}
	s.harnessSnapshotMu.Lock()
	if fn == nil {
		s.harnessSnapshotProvider = nil
	} else {
		s.harnessSnapshotProvider = func() SessionHarnessObservation {
			snapshot := fn()
			availability := guardvars.AvailabilityObserved
			if snapshot.Samples <= 0 {
				availability = guardvars.AvailabilityEmpty
			}
			return SessionHarnessObservation{
				Snapshot:     snapshot,
				Availability: availability,
			}
		}
	}
	s.harnessSnapshotMu.Unlock()
}

// SetSessionHarnessObservationProvider installs the status-aware harness pull
// source used by /v1/fak/observation. It is the richer sibling of
// SetSessionHarnessProvider: hosts that can distinguish a stale sample from a
// failed or unsupported sampler can report that state directly. The legacy
// /debug/vars block still emits only an OBSERVED sample.
func (s *Server) SetSessionHarnessObservationProvider(fn func() SessionHarnessObservation) {
	if s == nil {
		return
	}
	s.harnessSnapshotMu.Lock()
	s.harnessSnapshotProvider = fn
	s.harnessSnapshotMu.Unlock()
}

// sessionHarnessObservation pulls the typed harness-source result. ok is false
// only when no provider is configured; the caller owns status normalization and
// panic containment so one bad optional source cannot fail the whole snapshot.
func (s *Server) sessionHarnessObservation() (SessionHarnessObservation, bool) {
	if s == nil {
		return SessionHarnessObservation{}, false
	}
	s.harnessSnapshotMu.Lock()
	fn := s.harnessSnapshotProvider
	s.harnessSnapshotMu.Unlock()
	if fn == nil {
		return SessionHarnessObservation{}, false
	}
	return fn(), true
}

// sessionHarness pulls the current harness snapshot for /debug/vars. ok is false when
// no provider is set or nothing has been sampled yet, so the block is omitted rather
// than emitted all-zero.
func (s *Server) sessionHarness() (SessionHarness, bool) {
	observation, ok := s.sessionHarnessObservation()
	if !ok {
		return SessionHarness{}, false
	}
	availability := observation.Availability
	if availability == "" {
		availability = guardvars.AvailabilityObserved
		if observation.Snapshot.Samples <= 0 {
			availability = guardvars.AvailabilityEmpty
		}
	}
	if availability != guardvars.AvailabilityObserved || observation.Snapshot.Samples <= 0 {
		return SessionHarness{}, false
	}
	return observation.Snapshot, true
}
