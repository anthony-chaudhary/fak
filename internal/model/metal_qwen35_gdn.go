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
