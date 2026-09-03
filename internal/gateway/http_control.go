package gateway

import (
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/control"
)

var serverControlManagers sync.Map // map[*Server]*control.Manager

// ControlManager returns the control plane manager for the server, initializing it on first use.
func (s *Server) ControlManager() *control.Manager {
	if s == nil {
		return nil
	}
	if v, ok := serverControlManagers.Load(s); ok {
		return v.(*control.Manager)
	}

	s.controlConfigMu.Lock()
	defer s.controlConfigMu.Unlock()

	if v, ok := serverControlManagers.Load(s); ok {
		return v.(*control.Manager)
	}

	cur := s.VersionedConfig()
	initCtrlCfg := control.DefaultConfig()
	initCtrlCfg.CompletionDeadlineMs = cur.Config.CompletionDeadlineMs
	initCtrlCfg.StreamProgressTimeoutMs = cur.Config.StreamProgressTimeoutMs
	initCtrlCfg.MaxWaitingSeqs = cur.Config.MaxWaitingSeqs
	initCtrlCfg.CompactHistoryBudget = cur.Config.CompactHistoryBudget
	initCtrlCfg.CompactAnchorHead = cur.Config.CompactAnchorHead
	if cur.Config.LogLevel != "" {
		initCtrlCfg.LogLevel = cur.Config.LogLevel
	}

	qfn := func() int {
		s.admissionMu.RLock()
		ctl := s.admissionCtl
		s.admissionMu.RUnlock()
		if ctl != nil {
			st := ctl.Stats()
			return st.Waiting + st.Running
		}
		return 0
	}

	mgr, err := control.NewManager(initCtrlCfg, control.DefaultWatchdogConfig(), qfn)
	if err != nil {
		return nil
	}

	mgr.RegisterObserver(func(vc control.VersionedConfig) {
		s.versionedConfig.Store(&VersionedScalarConfig{
			Epoch: vc.Epoch,
			Config: ScalarConfig{
				CompletionDeadlineMs:    vc.Config.CompletionDeadlineMs,
				StreamProgressTimeoutMs: vc.Config.StreamProgressTimeoutMs,
				MaxWaitingSeqs:          vc.Config.MaxWaitingSeqs,
				CompactHistoryBudget:    vc.Config.CompactHistoryBudget,
				CompactAnchorHead:       vc.Config.CompactAnchorHead,
				LogLevel:                vc.Config.LogLevel,
			},
		})
		s.admissionMu.RLock()
		ctl := s.admissionCtl
		s.admissionMu.RUnlock()
		if ctl != nil {
			ctl.SetMaxWaiting(int(vc.Config.MaxWaitingSeqs))
		}
	})

	serverControlManagers.Store(s, mgr)
	return mgr
}

// handleControlConfig serves GET /v1/control/config and PATCH /v1/control/config (and their /v1/fak/... twins).
// GET returns the active ScalarConfig and current ConfigEpoch with X-Fak-Config-Epoch.
// PATCH validates and applies partial updates atomically, increments ConfigEpoch, and returns
// the updated configuration with X-Fak-Config-Epoch and X-Fak-Witness headers.
func (s *Server) handleControlConfig(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		vc := s.VersionedConfig()
		w.Header().Set("X-Fak-Config-Epoch", strconv.FormatUint(vc.Epoch, 10))
		writeJSON(w, http.StatusOK, ControlConfigResponse{
			ConfigEpoch: vc.Epoch,
			Config:      vc.Config,
		})
	case http.MethodPatch:
		dec := json.NewDecoder(r.Body)
		var patch ScalarConfigPatch
		if err := dec.Decode(&patch); err != nil {
			writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
			return
		}
		updated, epoch, err := s.PatchScalarConfig(patch)
		if err != nil {
			writeErr(w, http.StatusBadRequest, err.Error())
			return
		}
		w.Header().Set("X-Fak-Config-Epoch", strconv.FormatUint(epoch, 10))
		w.Header().Set("X-Fak-Witness", "verified-atomic-swap")
		writeJSON(w, http.StatusOK, ControlConfigResponse{
			Status:      "applied",
			ConfigEpoch: epoch,
			Config:      *updated,
		})
	default:
		writeErr(w, http.StatusMethodNotAllowed, "use GET or PATCH")
	}
}

// ControlApplyRequest carries a candidate configuration update for apply or dry-run evaluation.
type ControlApplyRequest struct {
	DryRun                         *bool    `json:"dry_run,omitempty"`
	CompletionDeadlineMs           *uint32  `json:"completion_deadline_ms,omitempty"`
	StreamProgressTimeoutMs        *uint32  `json:"stream_progress_timeout_ms,omitempty"`
	MaxWaitingSeqs                 *uint32  `json:"max_waiting_seqs,omitempty"`
	CompactHistoryBudget           *int     `json:"compact_history_budget,omitempty"`
	CompactAnchorHead              *int     `json:"compact_anchor_head,omitempty"`
	LogLevel                       *string  `json:"log_level,omitempty"`
	SpeculativeDraftDepth          *uint32  `json:"speculative_draft_depth,omitempty"`
	SpeculativeAcceptanceThreshold *float64 `json:"speculative_acceptance_threshold,omitempty"`

	MaxBatchTokens     *uint32 `json:"max_batch_tokens,omitempty"`
	MaxModelLen        *uint32 `json:"max_model_len,omitempty"`
	MaxNumSeqs         *uint32 `json:"max_num_seqs,omitempty"`
	PriorityStrategy   *string `json:"priority_strategy,omitempty"`
	PreemptionStrategy *string `json:"preemption_strategy,omitempty"`

	TargetKVBlocks            *uint32 `json:"target_kv_blocks,omitempty"`
	BlockSizeBytes            *uint32 `json:"block_size_bytes,omitempty"`
	MaxPreallocatedDraftLimit *uint32 `json:"max_preallocated_draft_slots,omitempty"`
	AvailableVRAMBytes        *uint64 `json:"available_vram_bytes,omitempty"`
	ModelWeightsBytes         *uint64 `json:"model_weights_bytes,omitempty"`
	ActivationHeadroomBytes   *uint64 `json:"activation_headroom_bytes,omitempty"`

	DeclaredLatencySLAMS *float64 `json:"declared_latency_sla_ms,omitempty"`
}

func (req ControlApplyRequest) toControlPatch() control.ConfigPatch {
	return control.ConfigPatch{
		CompletionDeadlineMs:           req.CompletionDeadlineMs,
		StreamProgressTimeoutMs:        req.StreamProgressTimeoutMs,
		MaxWaitingSeqs:                 req.MaxWaitingSeqs,
		CompactHistoryBudget:           req.CompactHistoryBudget,
		CompactAnchorHead:              req.CompactAnchorHead,
		LogLevel:                       req.LogLevel,
		SpeculativeDraftDepth:          req.SpeculativeDraftDepth,
		SpeculativeAcceptanceThreshold: req.SpeculativeAcceptanceThreshold,

		MaxBatchTokens:     req.MaxBatchTokens,
		MaxModelLen:        req.MaxModelLen,
		MaxNumSeqs:         req.MaxNumSeqs,
		PriorityStrategy:   req.PriorityStrategy,
		PreemptionStrategy: req.PreemptionStrategy,

		TargetKVBlocks:            req.TargetKVBlocks,
		BlockSizeBytes:            req.BlockSizeBytes,
		MaxPreallocatedDraftLimit: req.MaxPreallocatedDraftLimit,
		AvailableVRAMBytes:        req.AvailableVRAMBytes,
		ModelWeightsBytes:         req.ModelWeightsBytes,
		ActivationHeadroomBytes:   req.ActivationHeadroomBytes,

		DeclaredLatencySLAMS: req.DeclaredLatencySLAMS,
	}
}

// handleControlApply handles POST /v1/control/apply and POST /v1/fak/control/apply.
// Supports ?dry_run=true query parameter or dry_run field in payload.
func (s *Server) handleControlApply(w http.ResponseWriter, r *http.Request) {
	mgr := s.ControlManager()
	if mgr == nil {
		writeErr(w, http.StatusInternalServerError, "control manager not initialized")
		return
	}
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return
	}

	dryRunQuery := r.URL.Query().Get("dry_run")
	dryRun := dryRunQuery == "true" || dryRunQuery == "1"

	var req ControlApplyRequest
	if r.Body != nil {
		dec := json.NewDecoder(r.Body)
		if err := dec.Decode(&req); err != nil && err != io.EOF {
			writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
			return
		}
	}
	if req.DryRun != nil && *req.DryRun {
		dryRun = true
	}

	res, err := mgr.Apply(req.toControlPatch(), dryRun)
	if err != nil {
		if ve, ok := err.(control.ValidationErrors); ok {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"error":   "validation failed",
				"details": ve,
			})
			return
		}
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}

	w.Header().Set("X-Fak-Config-Epoch", strconv.FormatUint(res.ConfigEpoch, 10))
	if dryRun {
		w.Header().Set("X-Fak-Witness", "verified-dry-run")
	} else {
		w.Header().Set("X-Fak-Witness", "verified-atomic-swap")
	}
	writeJSON(w, http.StatusOK, res)
}

// handleControlEvents handles GET /v1/control/events and GET /v1/fak/control/events.
func (s *Server) handleControlEvents(w http.ResponseWriter, r *http.Request) {
	mgr := s.ControlManager()
	if mgr == nil {
		writeErr(w, http.StatusInternalServerError, "control manager not initialized")
		return
	}
	if r.Method != http.MethodGet {
		writeErr(w, http.StatusMethodNotAllowed, "use GET")
		return
	}
	stream := mgr.EventStream()
	if stream == nil {
		writeJSON(w, http.StatusOK, map[string]any{"events": []control.AuditEvent{}})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"events": stream.Snapshot(),
	})
}

// handleControlTelemetry handles POST /v1/control/telemetry and POST /v1/fak/control/telemetry.
func (s *Server) handleControlTelemetry(w http.ResponseWriter, r *http.Request) {
	mgr := s.ControlManager()
	if mgr == nil {
		writeErr(w, http.StatusInternalServerError, "control manager not initialized")
		return
	}
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	var sample control.TelemetrySample
	if err := json.NewDecoder(r.Body).Decode(&sample); err != nil {
		writeErr(w, http.StatusBadRequest, "invalid json: "+err.Error())
		return
	}
	triggered, triggerName, detail := mgr.IngestTelemetry(sample)
	active := mgr.Active()
	w.Header().Set("X-Fak-Config-Epoch", strconv.FormatUint(active.Epoch, 10))
	writeJSON(w, http.StatusOK, map[string]any{
		"triggered":    triggered,
		"trigger":      triggerName,
		"detail":       detail,
		"config_epoch": active.Epoch,
		"config":       active.Config,
	})
}
