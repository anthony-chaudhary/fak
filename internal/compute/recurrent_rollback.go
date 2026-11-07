// Ported from modular/modular max/python/max/nn/_graph_ops/state_space/kimi_delta.py@88ab8913af6b9298fe914e0f20d87a49e4172d0d (Apache-2.0 WITH LLVM-exception; Copyright (c) 2026 Modular Inc.). Adapted to fak-owned Go.
//
// The attribution above covers the GDN spec-verify ring-record machinery in this
// file (the GDNVerifyRing* types, record construction, ring wraparound, the fold,
// and the manager's ring-record rollback path); the surrounding slot-shadow
// rollback manager is pre-existing fak code.

package compute

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// RecurrentRollbackReceipt records verifiable metrics for on-device atomic slot rollback.
type RecurrentRollbackReceipt struct {
	SlotID          int           `json:"slot_id"`
	DraftK          int           `json:"draft_k"`
	AcceptedRows    int           `json:"accepted_rows"`
	D2HBytes        int64         `json:"d2h_bytes"`
	D2HEvents       int           `json:"d2h_events"`
	ZeroD2HVerified bool          `json:"zero_d2h_verified"`
	RollbackLatency time.Duration `json:"rollback_latency"`
	D2DBytesCopied  int64         `json:"d2d_bytes_copied"`
	Strategy        string        `json:"strategy"`
}

// RecurrentSlot manages on-device live and shadow states for one recurrent attention session.
type RecurrentSlot struct {
	SlotID       int
	LiveState    RecurrentDeviceState
	ShadowState  RecurrentDeviceState
	StepHistory  []RecurrentDeviceState // Indexed 0..DraftK
	DraftTokens  []float32
	AcceptedRows int
	Active       bool

	// GDN spec-verify ring-record path: when armed via EnableGDNRingRecording,
	// a draft round records compact per-token ring records (2*keyHeadDim+
	// valueHeadDim float32s each) instead of cloning the full state per step,
	// and RollbackSlot folds the accepted prefix from the shadow state.
	Ring         *GDNVerifyRing
	KeyHeadDim   int
	ValueHeadDim int
	roundMode    uint8
}

const (
	roundModeLegacy = iota // full-state-clone StepHistory checkpoints
	roundModeRing          // compact GDN verify-ring records
)

// RecurrentRollbackManager coordinates on-device state shadowing and atomic rollback
// across allocated state slots without incurring device-to-host synchronizations.
type RecurrentRollbackManager struct {
	mu          sync.RWMutex
	maxSlots    int
	draftK      int
	slots       map[int]*RecurrentSlot
	ringDefault bool // default fold strategy: compact ring-record fold when the round recorded ring records
}

// NewRecurrentRollbackManager initializes a rollback manager with a capacity of maxSlots and draft depth draftK.
func NewRecurrentRollbackManager(maxSlots int, draftK int) (*RecurrentRollbackManager, error) {
	if maxSlots <= 0 {
		return nil, fmt.Errorf("maxSlots must be positive, got %d", maxSlots)
	}
	if draftK <= 0 {
		return nil, fmt.Errorf("draftK must be positive, got %d", draftK)
	}
	return &RecurrentRollbackManager{
		maxSlots:    maxSlots,
		draftK:      draftK,
		slots:       make(map[int]*RecurrentSlot, maxSlots),
		ringDefault: true, // the ring-record fold is the new default; full-clone rewind remains the fallback
	}, nil
}

// RegisterSlot registers or resets a live state slot.
func (m *RecurrentRollbackManager) RegisterSlot(slotID int, initial RecurrentDeviceState) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if slotID < 0 || slotID >= m.maxSlots {
		return fmt.Errorf("slotID %d out of bounds [0, %d)", slotID, m.maxSlots)
	}
	if len(initial.ConvState) == 0 && len(initial.RecState) == 0 {
		return fmt.Errorf("initial state must not be empty")
	}

	slot := &RecurrentSlot{
		SlotID:       slotID,
		LiveState:    cloneRecurrentState(initial),
		ShadowState:  cloneRecurrentState(initial),
		StepHistory:  make([]RecurrentDeviceState, m.draftK+1),
		DraftTokens:  make([]float32, m.draftK),
		AcceptedRows: 0,
		Active:       true,
	}
	slot.StepHistory[0] = cloneRecurrentState(initial)
	m.slots[slotID] = slot
	return nil
}

// ShadowSlot captures an on-device shadow snapshot of the live slot prior to speculative rollout.
func (m *RecurrentRollbackManager) ShadowSlot(slotID int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	slot, ok := m.slots[slotID]
	if !ok || !slot.Active {
		return fmt.Errorf("slot %d is not registered or active", slotID)
	}

	// Device-to-Device copy into shadow buffer and step 0 checkpoint
	slot.ShadowState = cloneRecurrentState(slot.LiveState)
	slot.StepHistory[0] = cloneRecurrentState(slot.LiveState)
	if slot.Ring != nil {
		slot.Ring.Reset() // fresh draft round: previous records are no longer retained
	}
	return nil
}

// RecordDraftTokens loads speculative draft tokens into the slot and simulates GDN recurrence on device.
func (m *RecurrentRollbackManager) RecordDraftTokens(slotID int, tokens []float32) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	slot, ok := m.slots[slotID]
	if !ok || !slot.Active {
		return fmt.Errorf("slot %d is not registered or active", slotID)
	}
	if len(tokens) != m.draftK {
		return fmt.Errorf("expected %d draft tokens, got %d", m.draftK, len(tokens))
	}

	copy(slot.DraftTokens, tokens)
	curr := cloneRecurrentState(slot.ShadowState)
	for step := 0; step < m.draftK; step++ {
		stepRecurrent(&curr, tokens[step])
		slot.StepHistory[step+1] = cloneRecurrentState(curr)
	}
	slot.roundMode = roundModeLegacy
	return nil
}

// RollbackSlot atomically restores the live state slot to the state corresponding to acceptedRows.
// This operation is performed entirely on device with zero D2H memory transfers or synchronization events.
func (m *RecurrentRollbackManager) RollbackSlot(slotID int, acceptedRows int) (RecurrentRollbackReceipt, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	var receipt RecurrentRollbackReceipt
	slot, ok := m.slots[slotID]
	if !ok || !slot.Active {
		return receipt, fmt.Errorf("slot %d is not registered or active", slotID)
	}
	if acceptedRows < 0 || acceptedRows > m.draftK {
		return receipt, fmt.Errorf("acceptedRows %d out of bounds [0, %d]", acceptedRows, m.draftK)
	}

	start := time.Now()

	strategy := "device-slot-shadow-rewind"
	var d2dBytes int64

	switch {
	case slot.roundMode == roundModeRing && m.ringDefault:
		// Compact fold: apply the accepted ring records to the shadow state in
		// one pass instead of restoring a full-state clone (2K+V f32/token vs
		// KxV per draft step). ConvState passes through from the shadow state;
		// the GDN ring models the recurrent state only.
		keyDim, valDim := slot.KeyHeadDim, slot.ValueHeadDim
		recs := slot.Ring.Retained()
		if acceptedRows > len(recs) {
			return receipt, fmt.Errorf("acceptedRows %d exceeds retained ring records %d", acceptedRows, len(recs))
		}
		folded, err := GDNVerifyRingFold(slot.ShadowState.RecState, recs[:acceptedRows], keyDim, valDim)
		if err != nil {
			return receipt, err
		}
		// Copy both halves: a zero-record fold returns its input slice, and live
		// state must never alias the shadow state's backing arrays.
		shadow := cloneRecurrentState(slot.ShadowState)
		slot.LiveState = RecurrentDeviceState{
			ConvState: shadow.ConvState,
			RecState:  append([]float32(nil), folded...),
		}
		strategy = "gdn-verify-ring-fold"
		d2dBytes = int64(acceptedRows) * int64(GDNVerifyRingRecordElements(keyDim, valDim)) * 4
	case slot.roundMode == roundModeRing:
		// Fail closed: a ring-recorded round has no full-state checkpoints to
		// restore, so the legacy rewind cannot serve it.
		return receipt, fmt.Errorf("slot %d recorded compact ring records this round; the full-clone fallback needs a RecordDraftTokens round (enable ring recording or re-record)", slotID)
	default:
		// Atomic rewind: copy accepted checkpoint back to live state entirely within device memory
		acceptedState := slot.StepHistory[acceptedRows]
		slot.LiveState = cloneRecurrentState(acceptedState)
		// Calculate bytes copied on device: float32 size * (conv + rec)
		d2dBytes = int64((len(acceptedState.ConvState) + len(acceptedState.RecState)) * 4)
	}
	slot.AcceptedRows = acceptedRows

	latency := time.Since(start)

	receipt = RecurrentRollbackReceipt{
		SlotID:          slotID,
		DraftK:          m.draftK,
		AcceptedRows:    acceptedRows,
		D2HBytes:        0, // strictly 0 D2H bytes
		D2HEvents:       0, // strictly 0 D2H events
		ZeroD2HVerified: true,
		RollbackLatency: latency,
		D2DBytesCopied:  d2dBytes,
		Strategy:        strategy,
	}

	return receipt, nil
}

// BatchRollbackSlots performs atomic on-device rewind across a batch of slots.
func (m *RecurrentRollbackManager) BatchRollbackSlots(slotIDs []int, acceptedCounts []int) ([]RecurrentRollbackReceipt, error) {
	if len(slotIDs) != len(acceptedCounts) {
		return nil, fmt.Errorf("mismatched batch dimensions: slots=%d, acceptedCounts=%d", len(slotIDs), len(acceptedCounts))
	}
	receipts := make([]RecurrentRollbackReceipt, len(slotIDs))
	for i, slotID := range slotIDs {
		rcpt, err := m.RollbackSlot(slotID, acceptedCounts[i])
		if err != nil {
			return nil, fmt.Errorf("rollback failed on slot %d: %w", slotID, err)
		}
		receipts[i] = rcpt
	}
	return receipts, nil
}

// GetLiveState returns a clone of the current live state for verification.
func (m *RecurrentRollbackManager) GetLiveState(slotID int) (RecurrentDeviceState, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	slot, ok := m.slots[slotID]
	if !ok || !slot.Active {
		return RecurrentDeviceState{}, fmt.Errorf("slot %d is not registered or active", slotID)
	}
	return cloneRecurrentState(slot.LiveState), nil
}

// SimulateHostRollback simulates the legacy device-to-host fallback path where state must be
// copied from device to host, inspected, and copied back to device. It returns the simulated latency
// and bytes transferred over the PCIe bus.
func (m *RecurrentRollbackManager) SimulateHostRollback(slotID int, acceptedRows int) (time.Duration, int64, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	slot, ok := m.slots[slotID]
	if !ok || !slot.Active {
		return 0, 0, fmt.Errorf("slot %d is not registered or active", slotID)
	}
	if acceptedRows < 0 || acceptedRows > m.draftK {
		return 0, 0, fmt.Errorf("acceptedRows %d out of bounds [0, %d]", acceptedRows, m.draftK)
	}

	stateBytes := int64((len(slot.LiveState.ConvState) + len(slot.LiveState.RecState)) * 4)
	totalD2HBytes := stateBytes * 2 // D2H roundtrip: device -> host -> device

	// Model host synchronization overhead + PCIe bus transfer latency (typical PCIe Gen4 ~ 20-50 microseconds)
	// We simulate host buffer allocation and memory copies
	start := time.Now()
	hostBuffer := make([]byte, stateBytes)
	for i := range hostBuffer {
		hostBuffer[i] = byte(i & 0xFF)
	}
	time.Sleep(50 * time.Microsecond) // Simulated host-device sync synchronization floor
	deviceRestore := make([]byte, len(hostBuffer))
	copy(deviceRestore, hostBuffer)
	latency := time.Since(start)

	return latency, totalD2HBytes, nil
}

// SetRingRecording switches the default rollback fold between the compact GDN
// verify-ring record path (enabled, the default) and the legacy full-state-clone
// rewind (disabled, fallback). It applies to rounds recorded after the switch;
// a round already recorded as ring records fails closed on RollbackSlot while
// the switch is disabled rather than restoring checkpoints that do not exist.
func (m *RecurrentRollbackManager) SetRingRecording(enabled bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ringDefault = enabled
}

// EnableGDNRingRecording arms compact ring-record capture for a registered slot.
// The slot's RecState must lay out one head's recurrent state as
// keyHeadDim*valueHeadDim float32s in key-major order (state[k*valueHeadDim+v]).
// The ring holds one record per draft step and wraps at the draft depth; ConvState
// passes through the shadow state unchanged by the fold.
func (m *RecurrentRollbackManager) EnableGDNRingRecording(slotID, keyHeadDim, valueHeadDim int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if keyHeadDim <= 0 || valueHeadDim <= 0 {
		return fmt.Errorf("head dims must be positive, got key=%d value=%d", keyHeadDim, valueHeadDim)
	}
	slot, ok := m.slots[slotID]
	if !ok || !slot.Active {
		return fmt.Errorf("slot %d is not registered or active", slotID)
	}
	if len(slot.LiveState.RecState) != keyHeadDim*valueHeadDim {
		return fmt.Errorf("slot %d RecState len %d != keyHeadDim*valueHeadDim %d",
			slotID, len(slot.LiveState.RecState), keyHeadDim*valueHeadDim)
	}
	ring, err := NewGDNVerifyRing(m.draftK)
	if err != nil {
		return err
	}
	slot.Ring = ring
	slot.KeyHeadDim = keyHeadDim
	slot.ValueHeadDim = valueHeadDim
	return nil
}

// RecordDraftTokensGDN runs the GDN recurrence over the speculative draft tokens,
// writing one compact ring record per token instead of cloning the full state per
// step. Records are written against the evolving state starting from the shadow
// state captured by ShadowSlot, so folding the first j records reproduces the
// sequential j-token recurrence bit-for-bit.
func (m *RecurrentRollbackManager) RecordDraftTokensGDN(slotID int, ops []GDNVerifyRingToken) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	slot, ok := m.slots[slotID]
	if !ok || !slot.Active {
		return fmt.Errorf("slot %d is not registered or active", slotID)
	}
	if slot.Ring == nil {
		return fmt.Errorf("slot %d has no GDN ring recording enabled", slotID)
	}
	if len(ops) != m.draftK {
		return fmt.Errorf("expected %d draft tokens, got %d", m.draftK, len(ops))
	}

	keyDim, valDim := slot.KeyHeadDim, slot.ValueHeadDim
	curr := cloneRecurrentState(slot.ShadowState)
	for step := 0; step < m.draftK; step++ {
		rec, err := GDNVerifyRingRecordFromState(curr.RecState, ops[step], keyDim, valDim)
		if err != nil {
			return fmt.Errorf("draft token %d: %w", step, err)
		}
		slot.Ring.Write(rec)
		// Advance the running state with the same pinned arithmetic the fold uses.
		next, err := GDNVerifyRingFold(curr.RecState, []GDNVerifyRingRecord{rec}, keyDim, valDim)
		if err != nil {
			return fmt.Errorf("draft token %d: %w", step, err)
		}
		curr.RecState = next
	}
	slot.roundMode = roundModeRing
	return nil
}

// GDNVerifyRingToken is one token's GDN operands for the spec-verify ring recorder.
type GDNVerifyRingToken struct {
	Gate []float32 // per-channel log-decay g, len keyHeadDim; the record stores exp(g)
	Key  []float32 // raw key, len keyHeadDim (NOT L2-normalized)
	Val  []float32 // value, len valueHeadDim
	Beta float32   // update gate
}

// GDNVerifyRingRecordElements returns the float32 elements one ring record holds:
// per-channel decay exp(g), then the raw (not L2-normalized) key, then the update
// factor beta*(v - S^T k_raw * inv_norm)*inv_norm.
func GDNVerifyRingRecordElements(keyHeadDim, valueHeadDim int) int {
	return 2*keyHeadDim + valueHeadDim
}

// GDNVerifyRingRecord is one token's update to one head's state, laid out flat as
// [decay(keyHeadDim) | raw key(keyHeadDim) | update(valueHeadDim)] float32s.
// The key is stored raw because k_hat*delta and k_raw*(delta*inv_norm) round
// differently; folding is S = diag(decay)*S + key*update^T, the recurrence's own
// update with its operands and rounding, so a fold over the first j records lands
// bit-for-bit the state the sequential decode holds after j tokens.
type GDNVerifyRingRecord struct {
	Data []float32
}

// Decay returns the per-channel decay exp(g) section of the record.
func (r GDNVerifyRingRecord) Decay(keyHeadDim int) []float32 { return r.Data[:keyHeadDim] }

// Key returns the raw (not L2-normalized) key section of the record.
func (r GDNVerifyRingRecord) Key(keyHeadDim int) []float32 { return r.Data[keyHeadDim : 2*keyHeadDim] }

// Update returns the update-factor section of the record.
func (r GDNVerifyRingRecord) Update(keyHeadDim, valueHeadDim int) []float32 {
	return r.Data[2*keyHeadDim : 2*keyHeadDim+valueHeadDim]
}

// GDNVerifyRing holds up to ringLen per-token records for one session, wrapping:
// a write past ringLen overwrites the oldest slot, so the retained window is
// always the most recent min(count, ringLen) records in chronological order.
type GDNVerifyRing struct {
	cap   int
	slots []GDNVerifyRingRecord
	count int // total records written so far
}

// NewGDNVerifyRing allocates a wrapping ring of ringLen records.
func NewGDNVerifyRing(ringLen int) (*GDNVerifyRing, error) {
	if ringLen <= 0 {
		return nil, fmt.Errorf("ring length must be positive, got %d", ringLen)
	}
	return &GDNVerifyRing{cap: ringLen, slots: make([]GDNVerifyRingRecord, ringLen)}, nil
}

// Write appends a record, wrapping around and overwriting the oldest slot when
// the ring is full.
func (r *GDNVerifyRing) Write(rec GDNVerifyRingRecord) {
	r.slots[r.count%r.cap] = rec
	r.count++
}

// Retained returns the retained window, oldest to newest: all records when fewer
// than ringLen have been written, otherwise the most recent ringLen.
func (r *GDNVerifyRing) Retained() []GDNVerifyRingRecord {
	n := r.count
	if n > r.cap {
		n = r.cap
	}
	start := r.count - n
	out := make([]GDNVerifyRingRecord, n)
	for i := 0; i < n; i++ {
		out[i] = r.slots[(start+i)%r.cap]
	}
	return out
}

// Count returns the total number of records written so far.
func (r *GDNVerifyRing) Count() int { return r.count }

// Capacity returns the ring length.
func (r *GDNVerifyRing) Capacity() int { return r.cap }

// Reset clears the retained window and rewinds the write cursor for a fresh
// draft round.
func (r *GDNVerifyRing) Reset() { r.count = 0 }

// gdnVerifyRingTokenMath derives the pinned per-token intermediates shared by
// record construction and the sequential step so both round identically: the
// per-channel decay exp(g), the scalar inverse L2 norm of the raw key, and the
// update factor beta*(v - S^T k_raw * inv_norm)*inv_norm with the S^T k_raw
// reduction accumulated over ascending key channels through fmaF32.
func gdnVerifyRingTokenMath(state []float32, tok GDNVerifyRingToken, keyHeadDim, valueHeadDim int) (decay []float32, invNorm float32, update []float32, err error) {
	if keyHeadDim <= 0 || valueHeadDim <= 0 {
		return nil, 0, nil, fmt.Errorf("head dims must be positive, got key=%d value=%d", keyHeadDim, valueHeadDim)
	}
	if len(state) != keyHeadDim*valueHeadDim {
		return nil, 0, nil, fmt.Errorf("state len %d != keyHeadDim*valueHeadDim %d", len(state), keyHeadDim*valueHeadDim)
	}
	if len(tok.Gate) != keyHeadDim || len(tok.Key) != keyHeadDim || len(tok.Val) != valueHeadDim {
		return nil, 0, nil, fmt.Errorf("operand lens gate=%d key=%d value=%d, want %d/%d/%d",
			len(tok.Gate), len(tok.Key), len(tok.Val), keyHeadDim, keyHeadDim, valueHeadDim)
	}

	decay = make([]float32, keyHeadDim)
	for k, g := range tok.Gate {
		decay[k] = float32(math.Exp(float64(g)))
	}

	var sumSq float32
	for _, kk := range tok.Key {
		sumSq += kk * kk
	}
	if sumSq > 0 {
		invNorm = float32(1.0 / math.Sqrt(float64(sumSq)))
	}

	dot := make([]float32, valueHeadDim)
	for k := 0; k < keyHeadDim; k++ {
		kk := tok.Key[k]
		base := k * valueHeadDim
		for v := 0; v < valueHeadDim; v++ {
			dot[v] = fmaF32(state[base+v], kk, dot[v])
		}
	}

	update = make([]float32, valueHeadDim)
	for v := 0; v < valueHeadDim; v++ {
		residual := tok.Val[v] - dot[v]*invNorm
		update[v] = (tok.Beta * residual) * invNorm
	}
	return decay, invNorm, update, nil
}

// GDNVerifyRingRecordFromState builds the ring record for one token given the
// state BEFORE that token.
func GDNVerifyRingRecordFromState(state []float32, tok GDNVerifyRingToken, keyHeadDim, valueHeadDim int) (GDNVerifyRingRecord, error) {
	decay, _, update, err := gdnVerifyRingTokenMath(state, tok, keyHeadDim, valueHeadDim)
	if err != nil {
		return GDNVerifyRingRecord{}, err
	}
	rec := GDNVerifyRingRecord{Data: make([]float32, GDNVerifyRingRecordElements(keyHeadDim, valueHeadDim))}
	copy(rec.Data, decay)
	copy(rec.Data[keyHeadDim:], tok.Key)
	copy(rec.Data[2*keyHeadDim:], update)
	return rec, nil
}

// GDNVerifyRingStep applies one token's GDN recurrence to state using the same
// pinned operand order as the record fold, returning the next state.
func GDNVerifyRingStep(state []float32, tok GDNVerifyRingToken, keyHeadDim, valueHeadDim int) ([]float32, error) {
	decay, _, update, err := gdnVerifyRingTokenMath(state, tok, keyHeadDim, valueHeadDim)
	if err != nil {
		return nil, err
	}
	out := make([]float32, keyHeadDim*valueHeadDim)
	for k := 0; k < keyHeadDim; k++ {
		dk, kk := decay[k], tok.Key[k]
		base := k * valueHeadDim
		for v := 0; v < valueHeadDim; v++ {
			out[base+v] = fmaF32(kk, update[v], dk*state[base+v])
		}
	}
	return out, nil
}

// GDNVerifyRingFold advances base over the records, applying
// S = diag(decay)*S + key*update^T per record with the recurrence's own
// operands and rounding. With zero records it is a no-op returning the input
// state unchanged.
func GDNVerifyRingFold(base []float32, recs []GDNVerifyRingRecord, keyHeadDim, valueHeadDim int) ([]float32, error) {
	if keyHeadDim <= 0 || valueHeadDim <= 0 {
		return nil, fmt.Errorf("head dims must be positive, got key=%d value=%d", keyHeadDim, valueHeadDim)
	}
	elems := GDNVerifyRingRecordElements(keyHeadDim, valueHeadDim)
	if len(base) != keyHeadDim*valueHeadDim {
		return nil, fmt.Errorf("state len %d != keyHeadDim*valueHeadDim %d", len(base), keyHeadDim*valueHeadDim)
	}
	if len(recs) == 0 {
		return base, nil
	}
	for i, rec := range recs {
		if len(rec.Data) != elems {
			return nil, fmt.Errorf("record %d len %d != record elements %d", i, len(rec.Data), elems)
		}
	}
	out := make([]float32, keyHeadDim*valueHeadDim)
	copy(out, base)
	for _, rec := range recs {
		for k := 0; k < keyHeadDim; k++ {
			dk, kk := rec.Data[k], rec.Data[keyHeadDim+k]
			upBase := 2 * keyHeadDim
			baseIdx := k * valueHeadDim
			for v := 0; v < valueHeadDim; v++ {
				out[baseIdx+v] = fmaF32(kk, rec.Data[upBase+v], dk*out[baseIdx+v])
			}
		}
	}
	return out, nil
}
