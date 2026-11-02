package gateway

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"time"
)

// This interval binds current samples, not request-local counter deltas.
type debugRuntimeSnapshot struct {
	Schema             string                    `json:"schema"`
	Source             string                    `json:"source"`
	Scope              string                    `json:"scope"`
	State              string                    `json:"state"`
	Reason             string                    `json:"reason,omitempty"`
	StartedAt          time.Time                 `json:"started_at"`
	CompletedAt        time.Time                 `json:"completed_at"`
	LoadID             string                    `json:"load_id,omitempty"`
	ConfigSHA256       string                    `json:"config_sha256,omitempty"`
	ConfigIdentityMode string                    `json:"config_identity_mode,omitempty"`
	ConfigEpoch        uint64                    `json:"config_epoch"`
	Model              string                    `json:"model,omitempty"`
	Planner            string                    `json:"planner,omitempty"`
	Backend            string                    `json:"backend,omitempty"`
	Host               string                    `json:"host,omitempty"`
	Version            string                    `json:"fak_version,omitempty"`
	KVState            string                    `json:"kv_state"`
	MoEState           string                    `json:"moe_state"`
	CounterScope       string                    `json:"counter_scope"`
	Warm               *debugRuntimeWarmSnapshot `json:"warm,omitempty"`
}

type debugRuntimeWarmSnapshot struct {
	Scope             string `json:"scope"`
	LoadBinding       string `json:"load_binding"`
	LoadBindingReason string `json:"load_binding_reason"`
	Status            string `json:"status"`
	Identity          string `json:"identity,omitempty"`
	RuntimeGeneration string `json:"runtime_generation,omitempty"`
	Ready             bool   `json:"ready"`
}

type runtimeSnapshotCapture struct {
	started time.Time
	profile *ModelLoadProfile
	config  VersionedScalarConfig
}

func (s *Server) beginRuntimeSnapshot() runtimeSnapshotCapture {
	return runtimeSnapshotCapture{started: time.Now().UTC(), profile: s.modelLoadProfile(), config: s.VersionedConfig()}
}

func (s *Server) finishRuntimeSnapshot(c runtimeSnapshotCapture, response *debugVarsResponse) *debugRuntimeSnapshot {
	id := s.perfIdentity()
	out := &debugRuntimeSnapshot{
		Schema: "fak.runtime-snapshot.v1", Source: "/debug/vars", Scope: "current_server_observation",
		State: "UNKNOWN", StartedAt: c.started, ConfigEpoch: c.config.Epoch,
		Model: s.model, Planner: id.Planner, Backend: id.Backend, Host: id.Host, Version: id.Version,
		KVState: "UNKNOWN", MoEState: "UNKNOWN", CounterScope: "process_model_lifetime",
	}
	if response.KVMemory != nil {
		out.KVState = "OBSERVED"
	}
	if response.MoEResidency != nil {
		out.MoEState = "OBSERVED"
	}
	status, _, receipt, configured := s.agentWarm.snapshot()
	if configured {
		out.Warm = &debugRuntimeWarmSnapshot{Scope: "current_warm_gate", Status: status, LoadBinding: "UNKNOWN", LoadBindingReason: "warm_receipt_has_no_load_id"}
		if receipt != nil {
			out.Warm.Identity, out.Warm.RuntimeGeneration, out.Warm.Ready = receipt.Identity, receipt.RuntimeGeneration, receipt.Ready
		}
	}
	config, err := json.Marshal(c.config.Config)
	if err == nil {
		sum := sha256.Sum256(config)
		out.ConfigSHA256, out.ConfigIdentityMode = hex.EncodeToString(sum[:]), "scalar_v1"
	}
	current := s.modelLoadProfile()
	currentConfig := s.VersionedConfig()
	out.CompletedAt = time.Now().UTC()
	switch {
	case c.profile == nil || c.profile.LoadID == "" || current == nil:
		out.Reason = "load_identity_unavailable"
	case current.LoadID != c.profile.LoadID:
		out.Reason = "load_replaced_during_capture"
	case currentConfig.Epoch != c.config.Epoch || currentConfig.Config != c.config.Config:
		out.Reason = "config_replaced_during_capture"
	case err != nil:
		out.Reason = "config_identity_unavailable"
	default:
		out.State, out.LoadID = "OBSERVED", c.profile.LoadID
	}
	return out
}
