package model

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
)

const (
	inkling2StateSnapshotMagic   = "FAKI2S"
	inkling2StateSnapshotVersion = uint16(1)

	// Inkling2PrefillSnapshotChunk is the default prefill chunk boundary (512 tokens)
	// at which an Inkling-2 session may capture a full state snapshot.
	Inkling2PrefillSnapshotChunk = 512

	Inkling2SnapshotBoundaryPrefillChunk = "prefill_chunk"
	Inkling2SnapshotBoundaryTurnEnd      = "turn_end"
)

var (
	ErrInkling2SnapshotBoundary     = errors.New("model: inkling2 state snapshot requires a snapshot boundary")
	ErrInkling2StateSnapshotScope   = errors.New("model: inkling2 state snapshot model identity mismatch")
	ErrInkling2StateSnapshotInvalid = errors.New("model: invalid inkling2 state snapshot")
)

// Inkling2SnapshotBoundary records the exact boundary where a snapshot was captured.
type Inkling2SnapshotBoundary struct {
	Kind      string `json:"kind"`
	ChunkSize int    `json:"chunk_size,omitempty"`
	Pos       int    `json:"pos"`
}

// Validate reports whether b is a valid snapshot boundary at pos.
func (b Inkling2SnapshotBoundary) Validate(pos int) error {
	if pos <= 0 || b.Pos != pos {
		return fmt.Errorf("%w: boundary pos=%d session pos=%d", ErrInkling2SnapshotBoundary, b.Pos, pos)
	}
	switch b.Kind {
	case Inkling2SnapshotBoundaryTurnEnd:
		return nil
	case Inkling2SnapshotBoundaryPrefillChunk:
		if b.ChunkSize <= 0 || pos%b.ChunkSize != 0 {
			return fmt.Errorf("%w: pos=%d is not aligned to chunk=%d", ErrInkling2SnapshotBoundary, pos, b.ChunkSize)
		}
		return nil
	default:
		return fmt.Errorf("%w: unknown boundary kind %q", ErrInkling2SnapshotBoundary, b.Kind)
	}
}

// Inkling2SlotKVSnapshot stores one global slot's KV rows up to Pos.
type Inkling2SlotKVSnapshot struct {
	Slot      int
	Positions []int
	K         [][]float32
	V         [][]float32
}

// Inkling2RingKVSnapshot stores one local sliding-window ring slot's rows and bounds.
type Inkling2RingKVSnapshot struct {
	Slot      int
	Cap       int
	Lo        int
	Hi        int
	Positions []int
	K         [][]float32
	V         [][]float32
}

// Inkling2StateSnapshot is a complete, portable host-RAM snapshot of an Inkling-2
// session at a snapshot boundary: global and local KV rows up to pos, conv history,
// ring rows, horizon metadata, and model identity.
type Inkling2StateSnapshot struct {
	ModelIdentity  string
	HP             Inkling2LoopHParams
	LastLoopKV     bool
	Pos            int
	Boundary       Inkling2SnapshotBoundary
	Horizon        int
	Prompt         []int
	Conv           [][]float32
	GlobalSlots    []Inkling2SlotKVSnapshot
	LocalRingSlots []Inkling2RingKVSnapshot
}

func inkling2HostSlotLayer(hp Inkling2LoopHParams, slot int) int {
	if slot < hp.Layers {
		return slot
	}
	core := hp.CoreEnd - hp.CoreStart
	if core <= 0 {
		return slot
	}
	return hp.CoreStart + (slot-hp.Layers)%core
}

func cloneFloat32Rows(rows [][]float32) [][]float32 {
	out := make([][]float32, len(rows))
	for i, r := range rows {
		out[i] = append([]float32(nil), r...)
	}
	return out
}

// StateSnapshot captures the full Inkling-2 host state at a snapshot boundary.
func (h *Inkling2LoopHost) StateSnapshot(boundary Inkling2SnapshotBoundary, horizon int, modelIdentity string) (*Inkling2StateSnapshot, error) {
	if h == nil {
		return nil, fmt.Errorf("%w: nil host", ErrInkling2StateSnapshotInvalid)
	}
	if err := boundary.Validate(h.pos); err != nil {
		return nil, err
	}
	snap := &Inkling2StateSnapshot{
		ModelIdentity: modelIdentity,
		HP:            h.HP,
		LastLoopKV:    h.lastLoopKV,
		Pos:           h.pos,
		Boundary:      boundary,
		Horizon:       max(horizon, 0),
		Conv:          cloneFloat32Rows(h.conv),
	}
	for slot := range h.conv {
		layer := inkling2HostSlotLayer(h.HP, slot)
		isLocal := layer >= 0 && layer < len(h.HP.LocalLayer) && h.HP.LocalLayer[layer]
		if !isLocal {
			snap.GlobalSlots = append(snap.GlobalSlots, Inkling2SlotKVSnapshot{
				Slot:      slot,
				Positions: append([]int(nil), h.positions[slot]...),
				K:         cloneFloat32Rows(h.keys[slot]),
				V:         cloneFloat32Rows(h.values[slot]),
			})
			continue
		}
		lo := max(0, h.pos-h.HP.SlidingWindow)
		hi := h.pos
		start := 0
		pos := h.positions[slot]
		for start < len(pos) && pos[start] < lo {
			start++
		}
		snap.LocalRingSlots = append(snap.LocalRingSlots, Inkling2RingKVSnapshot{
			Slot:      slot,
			Cap:       h.HP.SlidingWindow,
			Lo:        lo,
			Hi:        hi,
			Positions: append([]int(nil), pos[start:]...),
			K:         cloneFloat32Rows(h.keys[slot][start:]),
			V:         cloneFloat32Rows(h.values[slot][start:]),
		})
	}
	return snap, nil
}

// RestoreStateSnapshot validates and installs snap into h.
func (h *Inkling2LoopHost) RestoreStateSnapshot(snap *Inkling2StateSnapshot) error {
	if h == nil {
		return fmt.Errorf("%w: nil host", ErrInkling2StateSnapshotInvalid)
	}
	if err := snap.Validate(); err != nil {
		return err
	}
	wantHP, err := json.Marshal(h.HP)
	if err != nil {
		return fmt.Errorf("%w: marshal host hparams: %v", ErrInkling2StateSnapshotInvalid, err)
	}
	gotHP, err := json.Marshal(snap.HP)
	if err != nil {
		return fmt.Errorf("%w: marshal snapshot hparams: %v", ErrInkling2StateSnapshotInvalid, err)
	}
	if !bytes.Equal(wantHP, gotHP) {
		return fmt.Errorf("%w: hparams mismatch", ErrInkling2StateSnapshotScope)
	}
	if len(snap.Conv) != len(h.conv) {
		return fmt.Errorf("%w: conv slots=%d, want %d", ErrInkling2StateSnapshotScope, len(snap.Conv), len(h.conv))
	}
	h.Reset()
	for i := range h.conv {
		if len(snap.Conv[i]) != len(h.conv[i]) {
			h.Reset()
			return fmt.Errorf("%w: conv slot %d len=%d, want %d", ErrInkling2StateSnapshotInvalid, i, len(snap.Conv[i]), len(h.conv[i]))
		}
		copy(h.conv[i], snap.Conv[i])
	}
	for _, gs := range snap.GlobalSlots {
		if gs.Slot < 0 || gs.Slot >= len(h.keys) {
			h.Reset()
			return fmt.Errorf("%w: global slot %d out of range", ErrInkling2StateSnapshotInvalid, gs.Slot)
		}
		h.positions[gs.Slot] = append([]int(nil), gs.Positions...)
		h.keys[gs.Slot] = cloneFloat32Rows(gs.K)
		h.values[gs.Slot] = cloneFloat32Rows(gs.V)
	}
	for _, ls := range snap.LocalRingSlots {
		if ls.Slot < 0 || ls.Slot >= len(h.keys) {
			h.Reset()
			return fmt.Errorf("%w: local slot %d out of range", ErrInkling2StateSnapshotInvalid, ls.Slot)
		}
		h.positions[ls.Slot] = append([]int(nil), ls.Positions...)
		h.keys[ls.Slot] = cloneFloat32Rows(ls.K)
		h.values[ls.Slot] = cloneFloat32Rows(ls.V)
	}
	h.pos = snap.Pos
	h.lastLoopKV = snap.LastLoopKV
	return nil
}

// Validate checks internal consistency of s.
func (s *Inkling2StateSnapshot) Validate() error {
	if s == nil {
		return fmt.Errorf("%w: nil snapshot", ErrInkling2StateSnapshotInvalid)
	}
	if err := s.Boundary.Validate(s.Pos); err != nil {
		return err
	}
	if s.Horizon < 0 {
		return fmt.Errorf("%w: negative horizon %d", ErrInkling2StateSnapshotInvalid, s.Horizon)
	}
	for i, gs := range s.GlobalSlots {
		if len(gs.K) != len(gs.Positions) || len(gs.V) != len(gs.Positions) {
			return fmt.Errorf("%w: global slot %d row/position mismatch", ErrInkling2StateSnapshotInvalid, i)
		}
	}
	for i, ls := range s.LocalRingSlots {
		if ls.Lo < 0 || ls.Hi < ls.Lo || ls.Hi > s.Pos {
			return fmt.Errorf("%w: local slot %d bounds [%d,%d) exceed pos=%d", ErrInkling2StateSnapshotInvalid, i, ls.Lo, ls.Hi, s.Pos)
		}
		if len(ls.K) != len(ls.Positions) || len(ls.V) != len(ls.Positions) {
			return fmt.Errorf("%w: local slot %d row/position mismatch", ErrInkling2StateSnapshotInvalid, i)
		}
	}
	return nil
}

// ResidentBytes returns the approximate host RAM footprint of s in bytes.
func (s *Inkling2StateSnapshot) ResidentBytes() int64 {
	if s == nil {
		return 0
	}
	var total int64
	total += int64(len(s.Prompt)) * 8
	for _, c := range s.Conv {
		total += int64(len(c)) * 4
	}
	for _, gs := range s.GlobalSlots {
		total += int64(len(gs.Positions)) * 8
		for _, r := range gs.K {
			total += int64(len(r)) * 4
		}
		for _, r := range gs.V {
			total += int64(len(r)) * 4
		}
	}
	for _, ls := range s.LocalRingSlots {
		total += int64(len(ls.Positions)) * 8
		for _, r := range ls.K {
			total += int64(len(r)) * 4
		}
		for _, r := range ls.V {
			total += int64(len(r)) * 4
		}
	}
	return total
}

// MarshalBinary encodes s into a deterministic, SHA-256-verified binary image
// using the prefix_snapshot_codec envelope header.
func (s *Inkling2StateSnapshot) MarshalBinary() ([]byte, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	hpBytes, err := json.Marshal(s.HP)
	if err != nil {
		return nil, fmt.Errorf("model: encode inkling2 hparams: %w", err)
	}
	var e snapshotEncoder
	e.bytes([]byte(s.ModelIdentity))
	e.bytes(hpBytes)
	e.boolean(s.LastLoopKV)
	e.integer(s.Pos)
	e.bytes([]byte(s.Boundary.Kind))
	e.integer(s.Boundary.ChunkSize)
	e.integer(s.Boundary.Pos)
	e.integer(s.Horizon)
	e.ints(s.Prompt)
	e.f32rows(s.Conv)

	e.count(len(s.GlobalSlots))
	for _, gs := range s.GlobalSlots {
		e.integer(gs.Slot)
		e.ints(gs.Positions)
		e.f32rows(gs.K)
		e.f32rows(gs.V)
	}
	e.count(len(s.LocalRingSlots))
	for _, ls := range s.LocalRingSlots {
		e.integer(ls.Slot)
		e.integer(ls.Cap)
		e.integer(ls.Lo)
		e.integer(ls.Hi)
		e.ints(ls.Positions)
		e.f32rows(ls.K)
		e.f32rows(ls.V)
	}
	if e.err != nil {
		return nil, e.err
	}
	return marshalSnapshotEnvelope(inkling2StateSnapshotMagic, inkling2StateSnapshotVersion, e.Bytes()), nil
}

// DecodeInkling2StateSnapshot verifies integrity, version, and model identity,
// returning the decoded Inkling2StateSnapshot.
func DecodeInkling2StateSnapshot(data []byte, expectedModelID string, expectedHP *Inkling2LoopHParams) (*Inkling2StateSnapshot, error) {
	body, err := unmarshalSnapshotEnvelope(data, inkling2StateSnapshotMagic, inkling2StateSnapshotVersion)
	if err != nil {
		return nil, err
	}
	d := snapshotDecoder{data: body}
	modelID := string(d.bytes())
	if d.err == nil && expectedModelID != "" && modelID != expectedModelID {
		return nil, fmt.Errorf("%w: got %q, want %q", ErrInkling2StateSnapshotScope, modelID, expectedModelID)
	}
	hpBytes := d.bytes()
	var hp Inkling2LoopHParams
	if d.err == nil {
		if err := json.Unmarshal(hpBytes, &hp); err != nil {
			return nil, fmt.Errorf("model: decode inkling2 hparams: %w", err)
		}
		if expectedHP != nil {
			wantHP, err := json.Marshal(*expectedHP)
			if err != nil {
				return nil, fmt.Errorf("model: encode expected inkling2 hparams: %w", err)
			}
			if !bytes.Equal(hpBytes, wantHP) {
				return nil, fmt.Errorf("%w: hparams mismatch", ErrInkling2StateSnapshotScope)
			}
		}
	}
	s := &Inkling2StateSnapshot{
		ModelIdentity: modelID,
		HP:            hp,
		LastLoopKV:    d.boolean(),
		Pos:           d.integer(),
		Boundary: Inkling2SnapshotBoundary{
			Kind:      string(d.bytes()),
			ChunkSize: d.integer(),
			Pos:       d.integer(),
		},
		Horizon: d.integer(),
		Prompt:  d.ints(),
		Conv:    d.f32rows(),
	}
	nGlobal := d.count()
	if d.err == nil && nGlobal > (len(d.data)-d.off)/8 {
		d.err = errors.New("oversized global slot count")
	}
	if d.err == nil {
		s.GlobalSlots = make([]Inkling2SlotKVSnapshot, nGlobal)
		for i := range s.GlobalSlots {
			s.GlobalSlots[i] = Inkling2SlotKVSnapshot{
				Slot:      d.integer(),
				Positions: d.ints(),
				K:         d.f32rows(),
				V:         d.f32rows(),
			}
		}
	}
	nLocal := d.count()
	if d.err == nil && nLocal > (len(d.data)-d.off)/8 {
		d.err = errors.New("oversized local slot count")
	}
	if d.err == nil {
		s.LocalRingSlots = make([]Inkling2RingKVSnapshot, nLocal)
		for i := range s.LocalRingSlots {
			s.LocalRingSlots[i] = Inkling2RingKVSnapshot{
				Slot:      d.integer(),
				Cap:       d.integer(),
				Lo:        d.integer(),
				Hi:        d.integer(),
				Positions: d.ints(),
				K:         d.f32rows(),
				V:         d.f32rows(),
			}
		}
	}
	if d.err != nil {
		return nil, fmt.Errorf("model: decode inkling2 state snapshot: %w", d.err)
	}
	if d.off != len(d.data) {
		return nil, fmt.Errorf("model: decode inkling2 state snapshot: %d trailing bytes", len(d.data)-d.off)
	}
	if err := s.Validate(); err != nil {
		return nil, err
	}
	return s, nil
}
