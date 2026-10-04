package model

import "fmt"

// Qwen35MetalGDNDecodeForwardPath is emitted only after a finalized prompt has
// promoted every linear layer to a session-owned fak-native Metal state owner.
const Qwen35MetalGDNDecodeForwardPath = "fak-native/metal/qwen35-gdn-resident-decode-v1"

type qwen35GDNStateSeeder interface {
	SeedQwen35GDNAuxState(Qwen35GDNAuxState, []float32, []float32) error
}

type qwen35GDNLayerSnapshot struct {
	layer           int
	conv, recurrent []float32
}

// promoteQwen35MetalGDNDecode seeds every owner before selecting the decode
// path. A missing capability or seed failure releases all owners and leaves the
// historical host cache as the sole state authority.
func (s *Session) promoteQwen35MetalGDNDecode(snapshots []qwen35GDNLayerSnapshot) (bool, error) {
	q := s.qwen35HAL
	if len(snapshots) == 0 {
		q.freeSequence()
		q.sequenceAccepted = false
		return false, nil
	}
	seeder, ok := q.sequenceBackend.(qwen35GDNStateSeeder)
	if !ok {
		q.freeSequence()
		q.sequenceAccepted = false
		return false, nil
	}
	for _, snapshot := range snapshots {
		state := q.sequenceLayers[snapshot.layer]
		if err := seeder.SeedQwen35GDNAuxState(state, snapshot.conv, snapshot.recurrent); err != nil {
			q.freeSequence()
			q.sequenceAccepted = false
			return false, fmt.Errorf("model: seed resident Metal GDN layer %d: %w", snapshot.layer, err)
		}
	}
	q.sequenceAccepted = false
	q.decodeAccepted = true
	q.decodePath = Qwen35MetalGDNDecodeForwardPath
	return true, nil
}

func (s *Session) tryQwen35MetalGDNDecode(layer int, mixed, z, b, a, convWeight, aLog, dtBias, norm []float32, eps float32) ([]float32, bool, error) {
	if s == nil || s.qwen35HAL == nil || !s.qwen35HAL.decodeAccepted {
		return nil, false, nil
	}
	q := s.qwen35HAL
	if layer < 0 || layer >= len(q.sequenceLayers) || !q.sequenceLayers[layer].valid() {
		return nil, false, nil
	}
	result, err := q.sequenceBackend.Qwen35GDNPreprojectedSequence(Qwen35GDNPreprojectedSequenceRequest{
		Layer: layer, Tokens: 1, Geometry: s.qwen35GDNSequenceGeometry(),
		Mixed: mixed, Z: z, B: b, A: a, Conv1D: convWeight, ALog: aLog, DTBias: dtBias, Norm: norm, RMSNormEpsilon: eps,
		State: q.sequenceLayers[layer],
	})
	s.recordQwen35ResidentGDNAccepted()
	if err != nil {
		return nil, true, s.failQwen35GDNSequence(layer, "resident decode", err)
	}
	_, nV, _, vHd, _, _, _ := s.M.Cfg.linearAttnDims()
	if len(result.Core) != nV*vHd {
		return nil, true, s.failQwen35GDNSequence(layer, "resident decode shape", fmt.Errorf("core elements=%d, want %d", len(result.Core), nV*vHd))
	}
	return result.Core, true, nil
}

// ResetQwen35MetalGDNDecode releases all resident owners and clears path
// identity. A later prompt must explicitly admit and seed a new owner set.
func (s *Session) ResetQwen35MetalGDNDecode() {
	if s == nil || s.qwen35HAL == nil {
		return
	}
	s.qwen35HAL.freeSequence()
	s.qwen35HAL.decodeAccepted = false
	s.qwen35HAL.decodePath = ""
	s.qwen35HAL.decodeHandoff = Qwen35DecodeHandoffReceipt{}
}

// Qwen35GDNDecodePath reports selected execution identity without inferring it
// from configuration. The false result means decode remains on the host path.
func (s *Session) Qwen35GDNDecodePath() (string, bool) {
	if s == nil || s.qwen35HAL == nil || !s.qwen35HAL.decodeAccepted {
		return "", false
	}
	return s.qwen35HAL.decodePath, true
}

// qwen35GDNOwnersHoldLiveState reports whether session-owned resident GDN owners
// (admitted sequence or promoted decode) are the authority for linear-attention
// state, so host Cache.linear may be stale until explicitly synchronized.
func (s *Session) qwen35GDNOwnersHoldLiveState() bool {
	if s == nil || s.qwen35HAL == nil || s.Backend != nil {
		return false
	}
	q := s.qwen35HAL
	if q.sequenceFailure != nil || q.sequenceBackend == nil || !(q.sequenceAccepted || q.decodeAccepted) {
		return false
	}
	for _, state := range q.sequenceLayers {
		if state.valid() {
			return true
		}
	}
	return false
}

// snapshotQwen35GDNOwners reads and shape-checks every live resident owner. It
// never mutates host or owner state; on error it names the failing layer/stage so
// the caller can apply its own failure policy.
func (s *Session) snapshotQwen35GDNOwners() ([]qwen35GDNLayerSnapshot, int, string, error) {
	q := s.qwen35HAL
	snapshotter, ok := q.sequenceBackend.(qwen35GDNSequenceSnapshotter)
	if !ok {
		return nil, -1, "final state synchronization", fmt.Errorf("admitted backend cannot snapshot auxiliary state")
	}
	cfg := s.M.Cfg
	_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	convElems := (cfg.LinearConvKernelDim - 1) * convDim
	recurrentElems := nV * kHd * vHd
	snapshots := make([]qwen35GDNLayerSnapshot, 0, len(q.sequenceLayers))
	for layer, state := range q.sequenceLayers {
		if !state.valid() {
			continue
		}
		conv, recurrent, err := snapshotter.SnapshotQwen35GDNAuxState(state)
		if err != nil {
			return nil, layer, "final state synchronization", err
		}
		if len(conv) != convElems || len(recurrent) != recurrentElems {
			return nil, layer, "final state shape", fmt.Errorf("conv/recurrent elements=%d/%d, want %d/%d", len(conv), len(recurrent), convElems, recurrentElems)
		}
		snapshots = append(snapshots, qwen35GDNLayerSnapshot{layer: layer, conv: conv, recurrent: recurrent})
	}
	return snapshots, -1, "", nil
}

// writeQwen35GDNSnapshotsToHost installs flat owner snapshots into the host
// linear-attention cache (conv rows oldest first, recurrent per value head).
func writeQwen35GDNSnapshotsToHost(cfg Config, cache *KVCache, snapshots []qwen35GDNLayerSnapshot) {
	if cache == nil || len(snapshots) == 0 {
		return
	}
	_, _, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	if cache.linear == nil {
		cache.linear = newLinearAttnCache(cfg)
	}
	for _, snapshot := range snapshots {
		state := cache.linear.layer(cfg, snapshot.layer)
		state.conv = make([][]float32, cfg.LinearConvKernelDim-1)
		for row := range state.conv {
			start := row * convDim
			state.conv[row] = append([]float32(nil), snapshot.conv[start:start+convDim]...)
		}
		for head := range state.recurrent {
			start := head * kHd * vHd
			copy(state.recurrent[head], snapshot.recurrent[start:start+kHd*vHd])
		}
	}
}

// SyncQwen35ResidentGDNStateToHost copies live resident GDN owner state into the
// host Cache.linear so a bare host KV clone (or PrefixSnapshot) carries the
// current recurrent/convolution state. Owners stay authoritative and unchanged;
// it is a no-op when no resident owner holds live state.
func (s *Session) SyncQwen35ResidentGDNStateToHost() error {
	if !s.qwen35GDNOwnersHoldLiveState() {
		if s != nil && s.Backend == nil && s.qwen35HAL != nil && s.qwen35HAL.sequenceFailure != nil {
			return fmt.Errorf("model: resident GDN state is unrecoverable after failure: %w", s.qwen35HAL.sequenceFailure)
		}
		return nil
	}
	snapshots, layer, stage, err := s.snapshotQwen35GDNOwners()
	if err != nil {
		return fmt.Errorf("model: host sync cannot read resident GDN layer %d (%s): %w", layer, stage, err)
	}
	writeQwen35GDNSnapshotsToHost(s.M.Cfg, s.Cache, snapshots)
	return nil
}

// qwen35HostGDNSeedSnapshots flattens a restored host Cache.linear into the
// owner seed layout for every linear-attention layer. It returns a decline
// reason (and no snapshots) when the host state is not a complete prefix state.
func (s *Session) qwen35HostGDNSeedSnapshots() ([]qwen35GDNLayerSnapshot, string) {
	cfg := s.M.Cfg
	_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	keep := cfg.LinearConvKernelDim - 1
	if keep < 0 {
		return nil, "invalid convolution kernel geometry"
	}
	c := s.Cache.linear
	if c == nil || len(c.layers) != cfg.NumLayers {
		return nil, "restored prefix carries no host linear-attention state"
	}
	wantConv := min(s.Cache.Len(), keep)
	var out []qwen35GDNLayerSnapshot
	for layer := 0; layer < cfg.NumLayers; layer++ {
		if !cfg.isLinearAttnLayer(layer) {
			continue
		}
		st := &c.layers[layer]
		if len(st.recurrent) != nV {
			return nil, fmt.Sprintf("restored layer %d recurrent heads=%d, want %d", layer, len(st.recurrent), nV)
		}
		for head, row := range st.recurrent {
			if len(row) != kHd*vHd {
				return nil, fmt.Sprintf("restored layer %d recurrent head %d elements=%d, want %d", layer, head, len(row), kHd*vHd)
			}
		}
		if len(st.conv) < wantConv || len(st.conv) > keep {
			return nil, fmt.Sprintf("restored layer %d convolution rows=%d, want %d..%d", layer, len(st.conv), wantConv, keep)
		}
		for row, values := range st.conv {
			if len(values) != convDim {
				return nil, fmt.Sprintf("restored layer %d convolution row %d elements=%d, want %d", layer, row, len(values), convDim)
			}
		}
		out = append(out, flattenQwen35HostGDNLayer(st, keep, convDim, layer))
	}
	return out, ""
}

// seedQwen35GDNOwnersFromHost seeds the freshly admitted sequence owners from
// flattened host state. It mutates only owner state.
func (s *Session) seedQwen35GDNOwnersFromHost(snapshots []qwen35GDNLayerSnapshot) error {
	q := s.qwen35HAL
	seeder, ok := q.sequenceBackend.(qwen35GDNStateSeeder)
	if !ok {
		return fmt.Errorf("admitted backend cannot seed auxiliary state")
	}
	for _, snapshot := range snapshots {
		if snapshot.layer < 0 || snapshot.layer >= len(q.sequenceLayers) || !q.sequenceLayers[snapshot.layer].valid() {
			return fmt.Errorf("layer %d has no admitted owner", snapshot.layer)
		}
		if err := seeder.SeedQwen35GDNAuxState(q.sequenceLayers[snapshot.layer], snapshot.conv, snapshot.recurrent); err != nil {
			return fmt.Errorf("layer %d: %w", snapshot.layer, err)
		}
	}
	return nil
}

// demoteQwen35ResidentGDNDecodeToHost hands a promoted decode owner set back to
// the host cache before a batched host prefill walk, which only consults owners
// in the admitted-sequence phase. The owners are synchronized into Cache.linear
// and released; a caller may then re-admit (and re-seed) from that host state.
func (s *Session) demoteQwen35ResidentGDNDecodeToHost() error {
	if s == nil || s.qwen35HAL == nil || s.qwen35HAL.sequenceAccepted || !s.qwen35HAL.decodeAccepted {
		return nil
	}
	if err := s.SyncQwen35ResidentGDNStateToHost(); err != nil {
		return err
	}
	s.ResetQwen35MetalGDNDecode()
	return nil
}

// flattenQwen35HostGDNLayer converts one host linear-attention layer to the flat
// owner layout: conv is keep rows right-aligned (missing oldest rows are the
// causal zero padding the host step applies), recurrent is head-major.
func flattenQwen35HostGDNLayer(st *linearAttnLayerState, keep, convDim, layer int) qwen35GDNLayerSnapshot {
	conv := make([]float32, keep*convDim)
	start := keep - len(st.conv)
	for i, row := range st.conv {
		copy(conv[(start+i)*convDim:(start+i+1)*convDim], row)
	}
	var recurrent []float32
	for _, head := range st.recurrent {
		recurrent = append(recurrent, head...)
	}
	return qwen35GDNLayerSnapshot{layer: layer, conv: conv, recurrent: recurrent}
}
