package hwutil

import (
	"sort"
	"strings"
	"sync"
	"time"
)

// Recorder is the in-process tier-residency emitter: a thread-safe, bounded
// sink that movement paths (the model weight loader, the fabric transport
// plane, future offload engines) record into while they move bytes between
// memory tiers. Collect drains it into Snapshot.Tiers / Snapshot.Transfers,
// so each snapshot carries the residency state plus the movements observed
// since the previous drain.
//
// The wire rows follow the frozen fak.hwutil.v1 contract: placements are
// TierRow bands keyed by (model, tier, device); transfers are TransferRow
// movements between two tiers. Nothing here may fail or block a caller:
// malformed rows are dropped, and the transfer ring evicts its oldest row
// when full rather than growing without bound.

// DefaultTransferCapacity is the standard transfer-ring size.
const DefaultTransferCapacity = 1024

// maxTransferCapacity bounds explicit capacities so a misconfigured caller
// cannot pin unbounded history.
const maxTransferCapacity = 4096

type placementKey struct {
	model  string
	kind   TierKind
	device string
}

// Recorder holds tier-residency bands and a FIFO ring of cross-tier
// transfers. All methods are safe for concurrent use; locking is a single
// mutex held only for map/slice mutation, so recorders stay cheap on hot
// paths.
type Recorder struct {
	mu sync.Mutex

	transferCap int
	transfers   []TransferRow // ring buffer, index 0 is oldest when full
	transNext   int           // next slot to overwrite
	transFull   bool

	placements map[placementKey]TierRow
}

// NewRecorder returns a Recorder whose transfer ring keeps at most
// capacity rows. capacity <= 0 selects DefaultTransferCapacity; values above
// maxTransferCapacity clamp down to it.
func NewRecorder(capacity int) *Recorder {
	if capacity <= 0 {
		capacity = DefaultTransferCapacity
	}
	if capacity > maxTransferCapacity {
		capacity = maxTransferCapacity
	}
	return &Recorder{
		transferCap: capacity,
		transfers:   make([]TransferRow, 0, capacity),
		placements:  make(map[placementKey]TierRow),
	}
}

var defaultRecorder = NewRecorder(DefaultTransferCapacity)

// Default returns the process-wide recorder that Collect drains when
// Options.Recorder is nil. Movement paths that have no other channel to the
// plane (weight uploads, fabric hops) record here.
func Default() *Recorder { return defaultRecorder }

// RecordPlacement upserts one residency band and adds bytes to the band's
// running total: every call asserts that bytes more of modelID now live in
// kind on device across layers [layersStart, layersEnd]. The loader records
// the host RAM band once at load time and each weight upload adds its tensor
// bytes to the device VRAM band, so the band total tracks real residency
// without callers maintaining their own counters. LastTransferAt stamps this
// event; TransferBytesPerS stays nil because a placement alone carries no
// rate (RecordTransfer does). Malformed rows (empty identity, inverted layer
// range, negative bytes/rank) are dropped silently.
func (r *Recorder) RecordPlacement(modelID string, kind TierKind, device string, layersStart, layersEnd int, bytes int64, hotnessRank int) {
	if r == nil {
		return
	}
	if strings.TrimSpace(modelID) == "" || strings.TrimSpace(device) == "" {
		return
	}
	if !tierKinds[kind] || layersStart < 0 || layersEnd < layersStart || bytes < 0 || hotnessRank < 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	key := placementKey{model: modelID, kind: kind, device: device}
	row, ok := r.placements[key]
	if !ok {
		row = TierRow{Kind: kind, Device: device, ModelID: modelID, LayersStart: layersStart, LayersEnd: layersEnd, HotnessRank: hotnessRank}
	}
	row.Bytes += bytes
	now := time.Now().UTC()
	row.LastTransferAt = &now
	r.placements[key] = row
}

// RecordTransfer appends one cross-tier movement. duration <= 0 records the
// row with rate_gib_s 0 (the wire field is not nullable in v1); a positive
// duration yields the measured GiB/s rate. Malformed rows (unknown tier pair
// outside the closed vocabulary is impossible by type, but empty reason or
// negative bytes/duration) are dropped silently. When the ring is full the
// oldest row is evicted.
func (r *Recorder) RecordTransfer(from, to TierKind, bytes int64, duration time.Duration, reason TransferReason) {
	if r == nil {
		return
	}
	if !tierKinds[from] || !tierKinds[to] || from == to || bytes < 0 || duration < 0 {
		return
	}
	if !reasons[reason] {
		reason = TransferUnknown
	}
	rate := 0.0
	if duration > 0 {
		const gib = 1 << 30
		rate = round6(float64(bytes) / duration.Seconds() / gib)
	}
	row := TransferRow{FromTier: string(from), ToTier: string(to), Bytes: bytes, DurationS: round6(duration.Seconds()), RateGiBS: rate, Reason: reason}
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.transfers) < r.transferCap {
		r.transfers = append(r.transfers, row)
		return
	}
	r.transfers[r.transNext] = row
	r.transNext = (r.transNext + 1) % r.transferCap
	r.transFull = true
}

// Drain returns the recorded bands and movements as deterministic wire rows
// and clears the recorder: tiers sorted by (model, tier, device, layer
// start), transfers in FIFO order. Both results are nil when nothing was
// recorded since the last drain, which is how Collect keeps an idle process
// honest (empty arrays on the wire, never invented zeros). Concurrent
// drains partition the FIFO fairly: each row is delivered exactly once.
func (r *Recorder) Drain() ([]TierRow, []TransferRow) {
	if r == nil {
		return nil, nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	var tiers []TierRow
	for _, row := range r.placements {
		tiers = append(tiers, row)
	}
	sort.Slice(tiers, func(i, j int) bool {
		a, b := tiers[i], tiers[j]
		if a.ModelID != b.ModelID {
			return a.ModelID < b.ModelID
		}
		if a.Kind != b.Kind {
			return a.Kind < b.Kind
		}
		if a.Device != b.Device {
			return a.Device < b.Device
		}
		return a.LayersStart < b.LayersStart
	})
	var out []TransferRow
	if len(r.transfers) > 0 {
		out = make([]TransferRow, 0, len(r.transfers))
		if r.transFull {
			start := r.transNext
			for i := 0; i < len(r.transfers); i++ {
				out = append(out, r.transfers[(start+i)%len(r.transfers)])
			}
		} else {
			out = append(out, r.transfers...)
		}
	}
	r.placements = make(map[placementKey]TierRow)
	r.transfers = r.transfers[:0]
	r.transNext = 0
	r.transFull = false
	return tiers, out
}

// TransferLen reports how many transfer rows are buffered right now.
func (r *Recorder) TransferLen() int {
	if r == nil {
		return 0
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.transfers)
}
