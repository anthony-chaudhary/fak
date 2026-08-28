package model

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"hash"
	"math"
	"time"
)

const (
	Qwen35MetalStateIdentitySchema = "fak.qwen35-metal-state-identity/1"

	Qwen35MetalStateAuthorityControl  = "metal-host-control"
	Qwen35MetalStateAuthoritySequence = "metal-sequence"

	Qwen35MetalStateRoleKRaw         = "full_attention_k_raw"
	Qwen35MetalStateRoleKPost        = "full_attention_k_post"
	Qwen35MetalStateRoleV            = "full_attention_v"
	Qwen35MetalStateRoleGDNConv      = "gdn_convolution"
	Qwen35MetalStateRoleGDNRecurrent = "gdn_recurrent"
)

// Qwen35MetalStateDigest identifies one already-host state slice without
// exposing its contents. Equality is meaningful only within one arm and token
// lineage; the hardware campaign retains the tolerant cross-arm oracle.
type Qwen35MetalStateDigest struct {
	Layer    int    `json:"layer"`
	Role     string `json:"role"`
	Elements int    `json:"elements"`
	SHA256   string `json:"sha256"`
}

// Qwen35MetalStateIdentityReceipt binds a fresh exact-P32 Metal observation to
// one opaque session owner and exact token lineage. Digest accounting covers
// every canonical byte fed to SHA-256. GDN transfer accounting names the
// existing final snapshot/seed work; hashing never adds a device readback.
type Qwen35MetalStateIdentityReceipt struct {
	Schema              string                   `json:"schema"`
	Available           bool                     `json:"available"`
	Authority           string                   `json:"authority"`
	OwnerGeneration     string                   `json:"owner_generation"`
	Tokens              int                      `json:"tokens"`
	TokenLineageSHA256  string                   `json:"token_lineage_sha256"`
	FullAttentionLayers int                      `json:"full_attention_layers"`
	GDNLayers           int                      `json:"gdn_layers"`
	StateCount          int                      `json:"state_count"`
	States              []Qwen35MetalStateDigest `json:"states"`
	GDNSnapshotOps      int                      `json:"gdn_snapshot_ops"`
	GDNSeedOps          int                      `json:"gdn_seed_ops"`
	GDNStateD2HBytes    uint64                   `json:"gdn_state_d2h_bytes"`
	GDNStateH2DBytes    uint64                   `json:"gdn_state_h2d_bytes"`
	DigestOperations    int                      `json:"digest_operations"`
	DigestInputBytes    uint64                   `json:"digest_input_bytes"`
	DigestNanoseconds   int64                    `json:"digest_nanoseconds"`
	BindingSHA256       string                   `json:"binding_sha256"`
}

type qwen35MetalStateIdentityObservation struct {
	ownerGeneration string
	tokenIDs        []int
	receipt         *Qwen35MetalStateIdentityReceipt
}

type qwen35MetalStateDigestSource struct {
	Layer  int
	Role   string
	Values []float32
	Chunks [][]float32
}

func (s qwen35MetalStateDigestSource) elements() int {
	n := len(s.Values)
	for _, chunk := range s.Chunks {
		n += len(chunk)
	}
	return n
}

type qwen35MetalStateIdentityAccounting struct {
	GDNSnapshotOps, GDNSeedOps         int
	GDNStateD2HBytes, GDNStateH2DBytes uint64
}

// Qwen35MetalStateIdentityUnavailableError is a pre-execution omission reason.
type Qwen35MetalStateIdentityUnavailableError struct{ Reason string }

func (e *Qwen35MetalStateIdentityUnavailableError) Error() string {
	return "model: Qwen Metal state identity unavailable: " + e.Reason
}

func newQwen35MetalStateIdentityObservation(tokenIDs []int) (*qwen35MetalStateIdentityObservation, error) {
	if len(tokenIDs) != 32 {
		return nil, &Qwen35MetalStateIdentityUnavailableError{Reason: fmt.Sprintf("requires fresh exact P32 tokens, got %d", len(tokenIDs))}
	}
	var nonce [32]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return nil, fmt.Errorf("model: mint Qwen Metal state owner generation: %w", err)
	}
	return &qwen35MetalStateIdentityObservation{
		ownerGeneration: hex.EncodeToString(nonce[:]),
		tokenIDs:        append([]int(nil), tokenIDs...),
	}, nil
}

// EnableQwen35MetalStateIdentityReceipt opts one fresh exact-P32 Metal session
// into state observation. It is independent of the sequence selector.
func (s *Session) EnableQwen35MetalStateIdentityReceipt(tokenIDs []int) error {
	if s == nil || s.M == nil || !s.M.Cfg.IsQwen35Hybrid() {
		return &Qwen35MetalStateIdentityUnavailableError{Reason: "session is not a Qwen hybrid"}
	}
	if s.Backend != nil || !s.Q4K || !s.MetalQ4K || newQwen35MetalGDNSequenceBackend == nil {
		return &Qwen35MetalStateIdentityUnavailableError{Reason: "requires a native resident-Q4_K Metal session"}
	}
	if s.Cache == nil || s.Cache.Len() != 0 {
		return &Qwen35MetalStateIdentityUnavailableError{Reason: "requires base=0 before prompt-state mutation"}
	}
	if s.qwen35MetalStateIdentity != nil {
		return &Qwen35MetalStateIdentityUnavailableError{Reason: "observation is already enabled for this session generation"}
	}
	observation, err := newQwen35MetalStateIdentityObservation(tokenIDs)
	if err != nil {
		return err
	}
	s.qwen35MetalStateIdentity = observation
	return nil
}

// ResetQwen35MetalStateIdentityReceipt clears request observation only. It does
// not reset KV, GDN owners, selection, or execution state.
func (s *Session) ResetQwen35MetalStateIdentityReceipt() {
	if s != nil {
		s.qwen35MetalStateIdentity = nil
	}
}

// Qwen35MetalStateIdentityReceipt returns an independently owned receipt copy.
func (s *Session) Qwen35MetalStateIdentityReceipt() (Qwen35MetalStateIdentityReceipt, bool) {
	if s == nil || s.qwen35MetalStateIdentity == nil || s.qwen35MetalStateIdentity.receipt == nil || !s.qwen35MetalStateIdentity.receipt.Available {
		return Qwen35MetalStateIdentityReceipt{}, false
	}
	return cloneQwen35MetalStateIdentityReceipt(*s.qwen35MetalStateIdentity.receipt), true
}

func cloneQwen35MetalStateIdentityReceipt(src Qwen35MetalStateIdentityReceipt) Qwen35MetalStateIdentityReceipt {
	src.States = append([]Qwen35MetalStateDigest(nil), src.States...)
	return src
}

// FinalizeQwen35MetalStateIdentityReceipt seals the selector-off control arm
// from its already-host KV and GDN state. The sequence arm is sealed by the GDN
// finalizer so it can reuse those existing snapshots without another readback.
func (s *Session) FinalizeQwen35MetalStateIdentityReceipt() (bool, error) {
	if s == nil || s.qwen35MetalStateIdentity == nil {
		return false, nil
	}
	if s.qwen35MetalStateIdentity.receipt != nil {
		return true, nil
	}
	if s.qwen35HAL != nil && (s.qwen35HAL.sequenceAccepted || s.qwen35HAL.decodeAccepted) {
		return true, fmt.Errorf("model: Qwen Metal sequence state identity must finalize at the existing GDN snapshot boundary")
	}
	receipt, err := s.buildQwen35MetalStateIdentityReceipt(Qwen35MetalStateAuthorityControl, nil, qwen35MetalStateIdentityAccounting{})
	if err != nil {
		return true, err
	}
	s.installQwen35MetalStateIdentityReceipt(receipt)
	return true, nil
}

func (s *Session) buildQwen35MetalStateIdentityReceipt(authority string, snapshots []qwen35GDNLayerSnapshot, accounting qwen35MetalStateIdentityAccounting) (Qwen35MetalStateIdentityReceipt, error) {
	observation := s.qwen35MetalStateIdentity
	if observation == nil {
		return Qwen35MetalStateIdentityReceipt{}, nil
	}
	cacheLen := 0
	if s.Cache != nil {
		cacheLen = s.Cache.Len()
	}
	if cacheLen != 32 {
		return Qwen35MetalStateIdentityReceipt{}, fmt.Errorf("model: Qwen Metal state identity requires fresh P32 cache length, got %d", cacheLen)
	}
	if _, err := s.Cache.lineage.verify(s.Cache.pos, observation.tokenIDs); err != nil {
		return Qwen35MetalStateIdentityReceipt{}, err
	}
	cfg := s.M.Cfg
	kvElements := 32 * cfg.NumKVHeads * cfg.HeadDim
	snapshotByLayer := make(map[int]qwen35GDNLayerSnapshot, len(snapshots))
	for _, snapshot := range snapshots {
		if _, duplicate := snapshotByLayer[snapshot.layer]; duplicate {
			return Qwen35MetalStateIdentityReceipt{}, fmt.Errorf("model: duplicate GDN snapshot for layer %d", snapshot.layer)
		}
		snapshotByLayer[snapshot.layer] = snapshot
	}
	sources := make([]qwen35MetalStateDigestSource, 0, cfg.NumLayers*3)
	for layer := 0; layer < cfg.NumLayers; layer++ {
		if !cfg.isLinearAttnLayer(layer) {
			for _, state := range []struct {
				role   string
				values []float32
			}{
				{Qwen35MetalStateRoleKRaw, s.Cache.Kraw[layer]},
				{Qwen35MetalStateRoleKPost, s.Cache.K[layer]},
				{Qwen35MetalStateRoleV, s.Cache.V[layer]},
			} {
				if len(state.values) != kvElements {
					return Qwen35MetalStateIdentityReceipt{}, fmt.Errorf("model: layer %d role %s elements=%d, want %d", layer, state.role, len(state.values), kvElements)
				}
				sources = append(sources, qwen35MetalStateDigestSource{Layer: layer, Role: state.role, Values: state.values})
			}
			continue
		}
		if snapshot, ok := snapshotByLayer[layer]; ok {
			sources = append(sources,
				qwen35MetalStateDigestSource{Layer: layer, Role: Qwen35MetalStateRoleGDNConv, Values: snapshot.conv},
				qwen35MetalStateDigestSource{Layer: layer, Role: Qwen35MetalStateRoleGDNRecurrent, Values: snapshot.recurrent},
			)
			continue
		}
		if s.Cache.linear == nil || layer >= len(s.Cache.linear.layers) {
			return Qwen35MetalStateIdentityReceipt{}, fmt.Errorf("model: missing host GDN state for layer %d", layer)
		}
		state := &s.Cache.linear.layers[layer]
		sources = append(sources,
			qwen35MetalStateDigestSource{Layer: layer, Role: Qwen35MetalStateRoleGDNConv, Chunks: state.conv},
			qwen35MetalStateDigestSource{Layer: layer, Role: Qwen35MetalStateRoleGDNRecurrent, Chunks: state.recurrent},
		)
	}
	return buildQwen35MetalStateIdentityReceipt(observation.ownerGeneration, observation.tokenIDs, authority, sources, accounting)
}

func (s *Session) installQwen35MetalStateIdentityReceipt(receipt Qwen35MetalStateIdentityReceipt) {
	if s == nil || s.qwen35MetalStateIdentity == nil {
		return
	}
	cloned := cloneQwen35MetalStateIdentityReceipt(receipt)
	s.qwen35MetalStateIdentity.receipt = &cloned
	if s.qwen35HAL != nil {
		if binder, ok := s.qwen35HAL.sequenceBackend.(qwen35MetalStateIdentityBinder); ok {
			binder.bindQwen35MetalStateIdentity(receipt)
		}
	}
}

type countedSHA256 struct {
	h     hash.Hash
	bytes uint64
}

func newCountedSHA256() *countedSHA256 { return &countedSHA256{h: sha256.New()} }

func (h *countedSHA256) raw(p []byte) {
	_, _ = h.h.Write(p)
	h.bytes += uint64(len(p))
}

func (h *countedSHA256) u64(v uint64) {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], v)
	h.raw(b[:])
}

func (h *countedSHA256) text(v string) {
	h.u64(uint64(len(v)))
	h.raw([]byte(v))
}

func (h *countedSHA256) digest() string { return hex.EncodeToString(h.h.Sum(nil)) }

func buildQwen35MetalStateIdentityReceipt(owner string, tokenIDs []int, authority string, sources []qwen35MetalStateDigestSource, accounting qwen35MetalStateIdentityAccounting) (Qwen35MetalStateIdentityReceipt, error) {
	started := time.Now()
	if len(tokenIDs) != 32 {
		return Qwen35MetalStateIdentityReceipt{}, fmt.Errorf("model: state identity token lineage length=%d, want 32", len(tokenIDs))
	}
	if _, err := decodeQwen35IdentityDigest(owner); err != nil {
		return Qwen35MetalStateIdentityReceipt{}, fmt.Errorf("model: state identity owner generation: %w", err)
	}
	lineageHash := newCountedSHA256()
	lineageHash.text("fak/qwen35-metal-token-lineage/v1")
	lineageHash.u64(uint64(len(tokenIDs)))
	for _, tokenID := range tokenIDs {
		if tokenID < 0 || uint64(tokenID) > math.MaxUint32 {
			return Qwen35MetalStateIdentityReceipt{}, fmt.Errorf("model: token id %d is outside exact uint32 lineage", tokenID)
		}
		lineageHash.u64(uint64(uint32(tokenID)))
	}
	lineageDigest := lineageHash.digest()
	lineageRaw, _ := hex.DecodeString(lineageDigest)
	receipt := Qwen35MetalStateIdentityReceipt{
		Schema: Qwen35MetalStateIdentitySchema, Available: true, Authority: authority,
		OwnerGeneration: owner, Tokens: len(tokenIDs), TokenLineageSHA256: lineageDigest,
		GDNSnapshotOps: accounting.GDNSnapshotOps, GDNSeedOps: accounting.GDNSeedOps,
		GDNStateD2HBytes: accounting.GDNStateD2HBytes, GDNStateH2DBytes: accounting.GDNStateH2DBytes,
	}
	digestBytes := lineageHash.bytes
	for _, source := range sources {
		elements := source.elements()
		if elements <= 0 {
			return Qwen35MetalStateIdentityReceipt{}, fmt.Errorf("model: state layer %d role %s is empty", source.Layer, source.Role)
		}
		h := newCountedSHA256()
		h.text("fak/qwen35-metal-state/v1")
		h.u64(uint64(source.Layer))
		h.text(source.Role)
		h.u64(uint64(elements))
		h.raw(lineageRaw)
		writeValues := func(values []float32) {
			for _, value := range values {
				var bits [4]byte
				binary.BigEndian.PutUint32(bits[:], math.Float32bits(value))
				h.raw(bits[:])
			}
		}
		writeValues(source.Values)
		for _, chunk := range source.Chunks {
			writeValues(chunk)
		}
		receipt.States = append(receipt.States, Qwen35MetalStateDigest{Layer: source.Layer, Role: source.Role, Elements: elements, SHA256: h.digest()})
		digestBytes += h.bytes
	}
	receipt.StateCount = len(receipt.States)
	full, gdn, err := qwen35MetalStateLayerCounts(receipt.States)
	if err != nil {
		return Qwen35MetalStateIdentityReceipt{}, err
	}
	receipt.FullAttentionLayers, receipt.GDNLayers = full, gdn
	receipt.DigestOperations = len(receipt.States) + 2
	binding, bindingBytes, err := qwen35MetalStateBindingDigest(receipt)
	if err != nil {
		return Qwen35MetalStateIdentityReceipt{}, err
	}
	receipt.BindingSHA256 = binding
	receipt.DigestInputBytes = digestBytes + bindingBytes
	receipt.DigestNanoseconds = max(int64(1), time.Since(started).Nanoseconds())
	return receipt, nil
}

func qwen35MetalStateLayerCounts(states []Qwen35MetalStateDigest) (full, gdn int, err error) {
	roles := make(map[int]map[string]struct{})
	for _, state := range states {
		if state.Layer < 0 || state.Elements <= 0 {
			return 0, 0, fmt.Errorf("model: invalid state layer/elements %d/%d", state.Layer, state.Elements)
		}
		switch state.Role {
		case Qwen35MetalStateRoleKRaw, Qwen35MetalStateRoleKPost, Qwen35MetalStateRoleV, Qwen35MetalStateRoleGDNConv, Qwen35MetalStateRoleGDNRecurrent:
		default:
			return 0, 0, fmt.Errorf("model: unknown state role %q", state.Role)
		}
		if roles[state.Layer] == nil {
			roles[state.Layer] = make(map[string]struct{})
		}
		if _, duplicate := roles[state.Layer][state.Role]; duplicate {
			return 0, 0, fmt.Errorf("model: duplicate state layer %d role %q", state.Layer, state.Role)
		}
		roles[state.Layer][state.Role] = struct{}{}
	}
	for layer, got := range roles {
		_, kraw := got[Qwen35MetalStateRoleKRaw]
		_, kpost := got[Qwen35MetalStateRoleKPost]
		_, value := got[Qwen35MetalStateRoleV]
		_, conv := got[Qwen35MetalStateRoleGDNConv]
		_, recurrent := got[Qwen35MetalStateRoleGDNRecurrent]
		switch {
		case len(got) == 3 && kraw && kpost && value:
			full++
		case len(got) == 2 && conv && recurrent:
			gdn++
		default:
			return 0, 0, fmt.Errorf("model: incomplete or mixed state roles at layer %d", layer)
		}
	}
	return full, gdn, nil
}

func decodeQwen35IdentityDigest(value string) ([]byte, error) {
	decoded, err := hex.DecodeString(value)
	if err != nil || len(decoded) != sha256.Size {
		return nil, fmt.Errorf("want lowercase SHA-256 hex")
	}
	if hex.EncodeToString(decoded) != value {
		return nil, fmt.Errorf("want canonical lowercase SHA-256 hex")
	}
	return decoded, nil
}

func qwen35MetalStateBindingDigest(receipt Qwen35MetalStateIdentityReceipt) (string, uint64, error) {
	owner, err := decodeQwen35IdentityDigest(receipt.OwnerGeneration)
	if err != nil {
		return "", 0, err
	}
	lineage, err := decodeQwen35IdentityDigest(receipt.TokenLineageSHA256)
	if err != nil {
		return "", 0, err
	}
	h := newCountedSHA256()
	h.text("fak/qwen35-metal-state-binding/v1")
	h.text(receipt.Schema)
	h.u64(boolU64(receipt.Available))
	h.text(receipt.Authority)
	h.raw(owner)
	h.u64(uint64(receipt.Tokens))
	h.raw(lineage)
	h.u64(uint64(receipt.FullAttentionLayers))
	h.u64(uint64(receipt.GDNLayers))
	h.u64(uint64(receipt.StateCount))
	for _, state := range receipt.States {
		digest, decodeErr := decodeQwen35IdentityDigest(state.SHA256)
		if decodeErr != nil {
			return "", 0, decodeErr
		}
		h.u64(uint64(state.Layer))
		h.text(state.Role)
		h.u64(uint64(state.Elements))
		h.raw(digest)
	}
	h.u64(uint64(receipt.GDNSnapshotOps))
	h.u64(uint64(receipt.GDNSeedOps))
	h.u64(receipt.GDNStateD2HBytes)
	h.u64(receipt.GDNStateH2DBytes)
	h.u64(uint64(receipt.DigestOperations))
	return h.digest(), h.bytes, nil
}

func boolU64(v bool) uint64 {
	if v {
		return 1
	}
	return 0
}

// ValidateQwen35MetalStateIdentityReceipt fails closed on structural mutation
// without requiring or exposing the tensor contents used to author the digests.
func ValidateQwen35MetalStateIdentityReceipt(receipt Qwen35MetalStateIdentityReceipt) error {
	if !receipt.Available || receipt.Schema != Qwen35MetalStateIdentitySchema || receipt.Tokens != 32 {
		return fmt.Errorf("model: invalid Qwen Metal state identity header")
	}
	if receipt.Authority != Qwen35MetalStateAuthorityControl && receipt.Authority != Qwen35MetalStateAuthoritySequence {
		return fmt.Errorf("model: invalid Qwen Metal state authority %q", receipt.Authority)
	}
	if _, err := decodeQwen35IdentityDigest(receipt.OwnerGeneration); err != nil {
		return fmt.Errorf("model: invalid owner generation: %w", err)
	}
	if _, err := decodeQwen35IdentityDigest(receipt.TokenLineageSHA256); err != nil {
		return fmt.Errorf("model: invalid token lineage digest: %w", err)
	}
	full, gdn, err := qwen35MetalStateLayerCounts(receipt.States)
	if err != nil {
		return err
	}
	if receipt.StateCount != len(receipt.States) || receipt.FullAttentionLayers != full || receipt.GDNLayers != gdn || full+gdn == 0 {
		return fmt.Errorf("model: Qwen Metal state count/layer mismatch")
	}
	for _, state := range receipt.States {
		if _, err := decodeQwen35IdentityDigest(state.SHA256); err != nil {
			return fmt.Errorf("model: invalid layer %d role %s digest: %w", state.Layer, state.Role, err)
		}
	}
	gdnBytes := uint64(0)
	for _, state := range receipt.States {
		if state.Role == Qwen35MetalStateRoleGDNConv || state.Role == Qwen35MetalStateRoleGDNRecurrent {
			gdnBytes += uint64(state.Elements) * 4
		}
	}
	if receipt.Authority == Qwen35MetalStateAuthorityControl {
		if receipt.GDNSnapshotOps != 0 || receipt.GDNSeedOps != 0 || receipt.GDNStateD2HBytes != 0 || receipt.GDNStateH2DBytes != 0 {
			return fmt.Errorf("model: control identity reported device state transfers")
		}
	} else if receipt.GDNSnapshotOps != gdn || receipt.GDNSeedOps != gdn || receipt.GDNStateD2HBytes != gdnBytes || receipt.GDNStateH2DBytes != gdnBytes {
		return fmt.Errorf("model: sequence GDN snapshot/seed accounting mismatch")
	}
	if receipt.DigestOperations != len(receipt.States)+2 || receipt.DigestNanoseconds <= 0 {
		return fmt.Errorf("model: invalid digest operation/time accounting")
	}
	binding, bindingBytes, err := qwen35MetalStateBindingDigest(receipt)
	if err != nil {
		return err
	}
	lineageBytes := uint64(8 + len("fak/qwen35-metal-token-lineage/v1") + 8 + receipt.Tokens*8)
	stateBytes := uint64(0)
	for _, state := range receipt.States {
		stateBytes += uint64(8 + len("fak/qwen35-metal-state/v1") + 8 + 8 + len(state.Role) + 8 + sha256.Size + state.Elements*4)
	}
	if receipt.DigestInputBytes != lineageBytes+stateBytes+bindingBytes {
		return fmt.Errorf("model: digest input-byte accounting mismatch")
	}
	if receipt.BindingSHA256 != binding {
		return fmt.Errorf("model: state identity binding digest mismatch")
	}
	return nil
}
