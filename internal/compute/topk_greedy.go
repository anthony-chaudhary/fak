package compute

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"sync"
)

// topk_greedy.go — Candidate-filtered top-k logit reduction for tensor-parallel greedy decode
// (borrowed from wkljohn/ds4-strix-halo-tp-odinlink DS4_TP_FEATURE_GREEDY_TOP2; Issue #10756).
//
// PROBLEM / CONTEXT:
// In tensor-parallel language model generation, the output vocabulary head (e.g. 128,000 tokens)
// is sharded row-wise across ranks (e.g. 64,000 tokens per rank for TP=2). Under standard collective
// AllGather or AllReduce operations, transmitting the entire 256 KiB FP32 partial logit vector across
// the interconnect on every single decoded token creates substantial wire volume and serialization
// latency (~5 ms/token on unified-memory APUs / USB4 / PCIe interconnects).
//
// DS4 GREEDY TOP-2 REDUCTION:
// Under greedy decoding (--temp 0), token selection is purely the global argmax. Worker ranks
// evaluate a local top-k (specifically top-2) argmax reduction across their vocabulary partition
// and transmit only 24 bytes (struct ds4_tp_logits_top2: 2 × {int32 id, float value} + rank/seq metadata)
// over the network. Rank 0 merges the candidate lists to determine the global argmax.
//
// MATHEMATICAL PARITY & ZERO DIVERGENCE:
// For any rank r with local partition V_r, the local maximum M_r = max_{i in V_r} logits[i] is
// guaranteed to be candidate #1 in rank r's local top-k (for any k >= 1).
// Since the global maximum max_i logits[i] = max_r M_r, merging the top-k candidates from all ranks
// guarantees finding the exact global argmax with 100% numerical parity and zero divergence vs
// exhaustive AllGather + argmax, while reducing wire volume by >99.99%.

// TopKLogit represents an extracted logit candidate token ID and its floating-point value.
type TopKLogit struct {
	ID    int32
	Value float32
}

// Top2LogitsPacketSize is the exact wire size (24 bytes) matching struct ds4_tp_logits_top2.
const Top2LogitsPacketSize = 24

// Top2LogitsPacket represents the fixed 24-byte wire structure transmitted across ranks
// during tensor-parallel greedy decode (struct ds4_tp_logits_top2).
// Layout:
//   - Candidates[0]: 8 bytes (int32 ID, float32 Value)
//   - Candidates[1]: 8 bytes (int32 ID, float32 Value)
//   - Rank:          4 bytes (int32 rank ID)
//   - SeqID:         4 bytes (uint32 sequence/step counter)
//     Total: 24 bytes.
type Top2LogitsPacket struct {
	Candidates [2]TopKLogit
	Rank       int32
	SeqID      uint32
}

// MarshalBinary serializes the packet into a fixed 24-byte array (little-endian).
func (p Top2LogitsPacket) MarshalBinary() [Top2LogitsPacketSize]byte {
	var b [Top2LogitsPacketSize]byte
	binary.LittleEndian.PutUint32(b[0:4], uint32(p.Candidates[0].ID))
	binary.LittleEndian.PutUint32(b[4:8], math.Float32bits(p.Candidates[0].Value))
	binary.LittleEndian.PutUint32(b[8:12], uint32(p.Candidates[1].ID))
	binary.LittleEndian.PutUint32(b[12:16], math.Float32bits(p.Candidates[1].Value))
	binary.LittleEndian.PutUint32(b[16:20], uint32(p.Rank))
	binary.LittleEndian.PutUint32(b[20:24], p.SeqID)
	return b
}

// UnmarshalTop2LogitsPacket deserializes a 24-byte array into a Top2LogitsPacket.
func UnmarshalTop2LogitsPacket(b [Top2LogitsPacketSize]byte) Top2LogitsPacket {
	return Top2LogitsPacket{
		Candidates: [2]TopKLogit{
			{
				ID:    int32(binary.LittleEndian.Uint32(b[0:4])),
				Value: math.Float32frombits(binary.LittleEndian.Uint32(b[4:8])),
			},
			{
				ID:    int32(binary.LittleEndian.Uint32(b[8:12])),
				Value: math.Float32frombits(binary.LittleEndian.Uint32(b[12:16])),
			},
		},
		Rank:  int32(binary.LittleEndian.Uint32(b[16:20])),
		SeqID: binary.LittleEndian.Uint32(b[20:24]),
	}
}

// UnmarshalTop2LogitsBytes parses a byte slice into a Top2LogitsPacket, enforcing exact length and valid boundaries.
func UnmarshalTop2LogitsBytes(b []byte) (Top2LogitsPacket, error) {
	if len(b) != Top2LogitsPacketSize {
		return Top2LogitsPacket{}, fmt.Errorf("compute: Top2LogitsPacket length = %d, want %d", len(b), Top2LogitsPacketSize)
	}
	var arr [Top2LogitsPacketSize]byte
	copy(arr[:], b)
	pkt := UnmarshalTop2LogitsPacket(arr)
	if pkt.Rank < 0 {
		return Top2LogitsPacket{}, fmt.Errorf("compute: Top2LogitsPacket invalid negative rank %d", pkt.Rank)
	}
	return pkt, nil
}

// ExtractTopK extracts the top-k candidates from a local partition of logits.
// Token IDs are mapped by adding vocabOffset to the local index.
// In the case of equal logit values, the candidate with the lower token ID wins (deterministic tie-breaking).
func ExtractTopK(logits []float32, k int, vocabOffset int) []TopKLogit {
	n := len(logits)
	if n == 0 || k <= 0 {
		return nil
	}
	if n <= k {
		out := make([]TopKLogit, n)
		for i, v := range logits {
			out[i] = TopKLogit{ID: int32(vocabOffset + i), Value: v}
		}
		sortTopK(out)
		return out
	}

	// Fast inline path for k=1 (pure argmax).
	if k == 1 {
		bestIdx := 0
		bestVal := logits[0]
		for i := 1; i < n; i++ {
			v := logits[i]
			if v > bestVal {
				bestVal = v
				bestIdx = i
			}
		}
		return []TopKLogit{{ID: int32(vocabOffset + bestIdx), Value: bestVal}}
	}

	// Fast inline path for k=2 (the canonical DS4_TP_FEATURE_GREEDY_TOP2 configuration).
	if k == 2 {
		c0 := TopKLogit{ID: int32(vocabOffset), Value: logits[0]}
		c1 := TopKLogit{ID: int32(vocabOffset + 1), Value: logits[1]}
		if c1.Value > c0.Value || (c1.Value == c0.Value && c1.ID < c0.ID) {
			c0, c1 = c1, c0
		}

		for i := 2; i < n; i++ {
			v := logits[i]
			id := int32(vocabOffset + i)
			if v > c0.Value || (v == c0.Value && id < c0.ID) {
				c1 = c0
				c0 = TopKLogit{ID: id, Value: v}
			} else if v > c1.Value || (v == c1.Value && id < c1.ID) {
				c1 = TopKLogit{ID: id, Value: v}
			}
		}
		return []TopKLogit{c0, c1}
	}

	// General k > 2: maintain a sorted buffer of k elements.
	buf := make([]TopKLogit, k)
	for i := 0; i < k; i++ {
		buf[i] = TopKLogit{ID: int32(vocabOffset + i), Value: logits[i]}
	}
	sortTopK(buf)

	for i := k; i < n; i++ {
		v := logits[i]
		id := int32(vocabOffset + i)
		// Check if it qualifies for inclusion in the top-k buffer.
		if v > buf[k-1].Value || (v == buf[k-1].Value && id < buf[k-1].ID) {
			// Find insertion position
			pos := k - 1
			for pos > 0 && (v > buf[pos-1].Value || (v == buf[pos-1].Value && id < buf[pos-1].ID)) {
				buf[pos] = buf[pos-1]
				pos--
			}
			buf[pos] = TopKLogit{ID: id, Value: v}
		}
	}

	return buf
}

func sortTopK(s []TopKLogit) {
	sort.Slice(s, func(i, j int) bool {
		if s[i].Value != s[j].Value {
			return s[i].Value > s[j].Value
		}
		return s[i].ID < s[j].ID
	})
}

// MergeTopKCandidates reduces the candidate lists gathered from all ranks and determines
// the single global argmax winner: token ID and logit value.
// It fails closed if candidates is empty or contains no candidates.
func MergeTopKCandidates(candidates [][]TopKLogit) (int, float32, error) {
	if len(candidates) == 0 {
		return 0, 0, fmt.Errorf("compute: MergeTopKCandidates got no candidate lists")
	}

	var bestID int32 = -1
	var bestVal float32 = -math.MaxFloat32
	found := false

	for _, rankList := range candidates {
		for _, cand := range rankList {
			if !found {
				bestID = cand.ID
				bestVal = cand.Value
				found = true
				continue
			}
			if cand.Value > bestVal || (cand.Value == bestVal && cand.ID < bestID) {
				bestVal = cand.Value
				bestID = cand.ID
			}
		}
	}

	if !found {
		return 0, 0, fmt.Errorf("compute: MergeTopKCandidates found zero valid candidates")
	}
	return int(bestID), bestVal, nil
}

// MergeFlatTopKCandidates reduces a flat slice of candidates across all ranks.
func MergeFlatTopKCandidates(candidates []TopKLogit) (int, float32, error) {
	if len(candidates) == 0 {
		return 0, 0, fmt.Errorf("compute: MergeFlatTopKCandidates got empty candidates")
	}
	bestID := candidates[0].ID
	bestVal := candidates[0].Value
	for i := 1; i < len(candidates); i++ {
		cand := candidates[i]
		if cand.Value > bestVal || (cand.Value == bestVal && cand.ID < bestID) {
			bestVal = cand.Value
			bestID = cand.ID
		}
	}
	return int(bestID), bestVal, nil
}

// MergeTopK merges candidate lists from all ranks and returns the top-k global candidates
// sorted in descending order of value (with lower token ID winning on tie).
func MergeTopK(candidates [][]TopKLogit, k int) []TopKLogit {
	if k <= 0 {
		return nil
	}
	total := 0
	for _, list := range candidates {
		total += len(list)
	}
	if total == 0 {
		return nil
	}

	all := make([]TopKLogit, 0, total)
	for _, list := range candidates {
		all = append(all, list...)
	}
	sortTopK(all)

	if len(all) > k {
		return all[:k]
	}
	return all
}

// SimulateMultiRankTopKGreedy simulates multi-rank tensor-parallel greedy reduction on CPU.
// Each element of parts represents a rank's local vocabulary slice (in rank order 0..P-1).
// Each rank evaluates its local top-k candidates, and Rank 0 merges them to produce the global argmax.
func SimulateMultiRankTopKGreedy(ctx context.Context, parts []Tensor, k int) (int, float32, error) {
	if ctx != nil && ctx.Err() != nil {
		return 0, 0, ctx.Err()
	}
	if len(parts) == 0 {
		return 0, 0, fmt.Errorf("compute: SimulateMultiRankTopKGreedy got no rank parts")
	}
	if k <= 0 {
		return 0, 0, fmt.Errorf("compute: SimulateMultiRankTopKGreedy k must be >= 1, got %d", k)
	}

	views := make([][]float32, len(parts))
	for r, p := range parts {
		if !p.Ready() {
			return 0, 0, fmt.Errorf("compute: SimulateMultiRankTopKGreedy rank %d tensor is not ready", r)
		}
		if p.Dtype != F32 {
			return 0, 0, fmt.Errorf("compute: SimulateMultiRankTopKGreedy rank %d dtype = %s, want f32", r, p.Dtype)
		}
		b := p.Backend()
		if b == nil {
			return 0, 0, fmt.Errorf("compute: SimulateMultiRankTopKGreedy rank %d has nil backend", r)
		}
		v, ok := b.Host(p)
		if !ok {
			return 0, 0, fmt.Errorf("compute: SimulateMultiRankTopKGreedy rank %d tensor is not host-readable", r)
		}
		if len(v) == 0 {
			return 0, 0, fmt.Errorf("compute: SimulateMultiRankTopKGreedy rank %d got empty partition", r)
		}
		views[r] = v
	}

	// Compute per-rank vocabulary offsets and extract top-k candidates.
	offset := 0
	rankCandidates := make([][]TopKLogit, len(views))
	for r, v := range views {
		rankCandidates[r] = ExtractTopK(v, k, offset)
		offset += len(v)
	}

	return MergeTopKCandidates(rankCandidates)
}

// ExhaustiveAllGatherArgmax performs standard AllGather over all rank parts to assemble
// the complete vocabulary logit tensor, and then computes the standard argmax.
// This serves as the independent numerical oracle against which candidate-filtered
// reduction must have 100% numerical parity and zero divergence.
func ExhaustiveAllGatherArgmax(cb CollectiveBackend, parts []Tensor) (int, float32, error) {
	if cb == nil {
		return 0, 0, fmt.Errorf("compute: ExhaustiveAllGatherArgmax got nil CollectiveBackend")
	}
	gathered, err := cb.AllGather(parts)
	if err != nil {
		return 0, 0, fmt.Errorf("compute: ExhaustiveAllGatherArgmax AllGather: %w", err)
	}
	v, ok := cb.Host(gathered)
	if !ok {
		return 0, 0, fmt.Errorf("compute: ExhaustiveAllGatherArgmax gathered tensor is not host-readable")
	}
	if len(v) == 0 {
		return 0, 0, fmt.Errorf("compute: ExhaustiveAllGatherArgmax gathered empty tensor")
	}

	bestIdx := 0
	bestVal := v[0]
	for i := 1; i < len(v); i++ {
		if v[i] > bestVal {
			bestVal = v[i]
			bestIdx = i
		}
	}
	return bestIdx, bestVal, nil
}

// MultiRankGreedySimulator simulates concurrent ranks exchanging candidate packets over channels.
// Worker ranks 1..P-1 pack their local top-k into 24-byte Top2LogitsPacket structures and transmit
// them to Rank 0. Rank 0 merges all candidates to output the global argmax.
type MultiRankGreedySimulator struct {
	ranks int
	k     int
}

// NewMultiRankGreedySimulator creates a concurrent multi-rank simulator.
func NewMultiRankGreedySimulator(ranks int, k int) (*MultiRankGreedySimulator, error) {
	if ranks <= 0 {
		return nil, fmt.Errorf("compute: ranks must be >= 1, got %d", ranks)
	}
	if k <= 0 {
		return nil, fmt.Errorf("compute: k must be >= 1, got %d", k)
	}
	return &MultiRankGreedySimulator{ranks: ranks, k: k}, nil
}

// Run executes concurrent candidate extraction across simulated worker goroutines, transmits
// 24-byte wire packets to Rank 0, and returns the winning token ID, logit value, and total bytes
// transmitted over the interconnect.
func (s *MultiRankGreedySimulator) Run(ctx context.Context, parts []Tensor) (int, float32, int, error) {
	if ctx != nil && ctx.Err() != nil {
		return 0, 0, 0, ctx.Err()
	}
	if len(parts) != s.ranks {
		return 0, 0, 0, fmt.Errorf("compute: simulator ranks=%d, but got %d parts", s.ranks, len(parts))
	}

	views := make([][]float32, s.ranks)
	offsets := make([]int, s.ranks)
	currOffset := 0
	for r, p := range parts {
		if !p.Ready() {
			return 0, 0, 0, fmt.Errorf("compute: rank %d tensor is not ready", r)
		}
		if p.Dtype != F32 {
			return 0, 0, 0, fmt.Errorf("compute: rank %d dtype = %s, want f32", r, p.Dtype)
		}
		b := p.Backend()
		if b == nil {
			return 0, 0, 0, fmt.Errorf("compute: rank %d nil backend", r)
		}
		v, ok := b.Host(p)
		if !ok {
			return 0, 0, 0, fmt.Errorf("compute: rank %d tensor is not host-readable", r)
		}
		views[r] = v
		offsets[r] = currOffset
		currOffset += len(v)
	}

	if s.ranks == 1 {
		cands := ExtractTopK(views[0], s.k, 0)
		if len(cands) == 0 {
			return 0, 0, 0, fmt.Errorf("compute: rank 0 failed to extract candidates")
		}
		return int(cands[0].ID), cands[0].Value, 0, nil
	}

	// Channel for worker ranks 1..P-1 to send serialized 24-byte packets to Rank 0.
	type rankMsg struct {
		rank int
		wire [Top2LogitsPacketSize]byte
		err  error
	}

	msgCh := make(chan rankMsg, s.ranks-1)
	var wg sync.WaitGroup

	for r := 1; r < s.ranks; r++ {
		wg.Add(1)
		go func(rankID int) {
			defer wg.Done()
			if ctx != nil && ctx.Err() != nil {
				msgCh <- rankMsg{rank: rankID, err: ctx.Err()}
				return
			}
			cands := ExtractTopK(views[rankID], s.k, offsets[rankID])
			var pkt Top2LogitsPacket
			pkt.Rank = int32(rankID)
			if len(cands) > 0 {
				pkt.Candidates[0] = cands[0]
			}
			if len(cands) > 1 {
				pkt.Candidates[1] = cands[1]
			}
			wire := pkt.MarshalBinary()
			msgCh <- rankMsg{rank: rankID, wire: wire}
		}(r)
	}

	// Rank 0 extracts its own local candidates directly (no network transmission).
	rank0Cands := ExtractTopK(views[0], s.k, offsets[0])

	wg.Wait()
	close(msgCh)

	allCandidates := make([][]TopKLogit, s.ranks)
	allCandidates[0] = rank0Cands

	totalBytesTransmitted := 0
	for msg := range msgCh {
		if msg.err != nil {
			return 0, 0, 0, msg.err
		}
		pkt := UnmarshalTop2LogitsPacket(msg.wire)
		totalBytesTransmitted += Top2LogitsPacketSize
		allCandidates[msg.rank] = []TopKLogit{pkt.Candidates[0], pkt.Candidates[1]}
	}

	bestID, bestVal, err := MergeTopKCandidates(allCandidates)
	if err != nil {
		return 0, 0, 0, err
	}
	return bestID, bestVal, totalBytesTransmitted, nil
}

// WireSavingsReport details the interconnect traffic savings of candidate-filtered top-2
// reduction over exhaustive AllGather.
type WireSavingsReport struct {
	VocabSize          int     `json:"vocab_size"`
	NumRanks           int     `json:"num_ranks"`
	AllGatherBytes     int     `json:"all_gather_bytes"`
	Top2ReductionBytes int     `json:"top2_reduction_bytes"`
	SavedBytes         int     `json:"saved_bytes"`
	SavingsPercentage  float64 `json:"savings_percentage"`
}

// ComputeGreedyWireSavings calculates the wire volume and savings percentage for a given vocabulary
// size and rank count under top-2 candidate reduction.
func ComputeGreedyWireSavings(vocabSize int, numRanks int) WireSavingsReport {
	allGatherBytes := vocabSize * 4 // FP32 partials gathered across interconnect
	top2Bytes := (numRanks - 1) * Top2LogitsPacketSize
	if numRanks <= 1 {
		top2Bytes = 0
	}
	saved := allGatherBytes - top2Bytes
	savingsPct := 0.0
	if allGatherBytes > 0 {
		savingsPct = (float64(saved) / float64(allGatherBytes)) * 100.0
	}
	return WireSavingsReport{
		VocabSize:          vocabSize,
		NumRanks:           numRanks,
		AllGatherBytes:     allGatherBytes,
		Top2ReductionBytes: top2Bytes,
		SavedBytes:         saved,
		SavingsPercentage:  savingsPct,
	}
}

// ReduceTopKGreedy extracts top-k candidates from a local partition and returns the top candidate
// (token ID mapped by vocabOffset, and its logit value).
func (c *cpuBackend) ReduceTopKGreedy(ctx context.Context, t Tensor, k int, vocabOffset int) (int, float32, error) {
	if ctx != nil && ctx.Err() != nil {
		return 0, 0, ctx.Err()
	}
	if k <= 0 {
		return 0, 0, fmt.Errorf("compute: ReduceTopKGreedy k must be >= 1, got %d", k)
	}
	if vocabOffset < 0 {
		return 0, 0, fmt.Errorf("compute: ReduceTopKGreedy negative vocabOffset %d", vocabOffset)
	}
	if !t.Ready() {
		return 0, 0, fmt.Errorf("compute: ReduceTopKGreedy tensor is not ready")
	}
	if t.Backend() != c {
		return 0, 0, fmt.Errorf("compute: ReduceTopKGreedy foreign backend")
	}
	if t.Dtype != F32 {
		return 0, 0, fmt.Errorf("compute: ReduceTopKGreedy tensor dtype %s, want f32", t.Dtype)
	}
	v, ok := c.Host(t)
	if !ok {
		return 0, 0, fmt.Errorf("compute: ReduceTopKGreedy tensor is not host-readable")
	}
	if len(v) == 0 {
		return 0, 0, fmt.Errorf("compute: ReduceTopKGreedy empty tensor")
	}
	cands := ExtractTopK(v, k, vocabOffset)
	if len(cands) == 0 {
		return 0, 0, fmt.Errorf("compute: ReduceTopKGreedy failed to extract candidates")
	}
	return int(cands[0].ID), cands[0].Value, nil
}
