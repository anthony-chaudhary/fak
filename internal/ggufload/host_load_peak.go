package ggufload

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// The host peak estimate folds walkQ4KLoadStorage so admission uses the same
// packed, Q8, and F32 classification as the loader before reading tensor data.

// HostLoadPeakOverrideEnv is the operator override for the native serve host-peak admission gate:
// "off" (or 0/false/no) skips the refusal and loads anyway.
const HostLoadPeakOverrideEnv = "FAK_SERVE_HOST_PEAK_ADMISSION"

// HostLoadPeakMarginEnv overrides the default safety margin, in GiB (fractional values allowed).
const HostLoadPeakMarginEnv = "FAK_SERVE_HOST_PEAK_MARGIN_GIB"

// HostLoadPeakDefaultMarginBytes is the MemAvailable the gate leaves unclaimed after the predicted
// peak: Go GC slack over the live heap, the session's KV/scratch, driver staging and the
// page cache other processes are actively using.
const HostLoadPeakDefaultMarginBytes int64 = 2 << 30

// ErrHostLoadPeakTooBig is the sentinel every HostLoadPeakError matches with errors.Is.
var ErrHostLoadPeakTooBig = errors.New("HOST_LOAD_PEAK_TOO_BIG")

// HostLoadTypeRow is one GGML tensor type's share of the host load: its on-disk payload and where
// those tensors land in host memory.
type HostLoadTypeRow struct {
	Type         string `json:"type"`
	Tensors      int    `json:"tensors"`
	PayloadBytes int64  `json:"payload_bytes"` // packed bytes on disk
	PackedBytes  int64  `json:"packed_bytes"`  // retained packed, anonymous heap copy
	MappedBytes  int64  `json:"mapped_bytes"`  // retained packed as file-backed views (not anonymous)
	Q8Bytes      int64  `json:"q8_bytes"`      // dequantized -> native Q8 (codes + f32 scales)
	F32Bytes     int64  `json:"f32_bytes"`     // dequantized -> F32 kept in the builder arena
	StreamBytes  int64  `json:"stream_bytes"`  // bounded streamed-dense working-set candidates
}

// AnonBytes is the steady anonymous host residency this type contributes.
func (r HostLoadTypeRow) AnonBytes() int64 { return r.PackedBytes + r.Q8Bytes + r.F32Bytes }

// HostLoadPeak is the predicted host-resident peak of one resident-Q4K load.
type HostLoadPeak struct {
	Rows []HostLoadTypeRow `json:"rows"` // sorted by AnonBytes, largest first
	// SteadyAnonBytes is every anonymous byte the built model holds on the host before the
	// first session uploads it: packed copies, Q8, F32 and the bounded streamed working set.
	SteadyAnonBytes int64 `json:"steady_anon_bytes"`
	// MappedBytes are packed weights served as views into the read-only checkpoint map: page
	// cache the kernel can reclaim, so they are reported but not charged.
	MappedBytes int64 `json:"mapped_bytes"`
	// WorkerWindowBytes is the largest sum of per-tensor conversion windows over any Workers
	// consecutive tensors (the loader admits at most Workers tensors, in header order, before the
	// collector applies the oldest; see parallelQuantLoadContextBudget).
	WorkerWindowBytes int64 `json:"worker_window_bytes"`
	Workers           int   `json:"workers"`
	// ArenaCopyBytes is QuantBuilder.Build's coalescing copy of the F32 chunks into one arena:
	// both copies are live until the chunks are dropped.
	ArenaCopyBytes int64 `json:"arena_copy_bytes"`
	// TransientBytes is max(WorkerWindowBytes, ArenaCopyBytes): the two phases are disjoint.
	TransientBytes int64 `json:"transient_bytes"`
	// PeakAnonBytes is SteadyAnonBytes + TransientBytes, the number the admission gate charges.
	PeakAnonBytes int64 `json:"peak_anon_bytes"`
}

// Dominant returns up to n type rows with the largest anonymous host contribution.
func (p HostLoadPeak) Dominant(n int) []HostLoadTypeRow {
	out := make([]HostLoadTypeRow, 0, n)
	for _, r := range p.Rows {
		if len(out) >= n {
			break
		}
		if r.AnonBytes() > 0 {
			out = append(out, r)
		}
	}
	return out
}

// EstimateQ4KHostLoadPeak predicts the host-resident peak of the resident-Q4K load for opts.
// mapped selects the mapped-resident entry (LoadModelQ4KMappedResident): eligible dense Q4_K and
// the lazily held k-quants (Q2_K/Q3_K/Q5_K/Q6_K) become file-backed views, while i-quants, native
// rows, MTP roles and packed embeddings stay owned heap copies exactly as that loader keeps them.
// It returns ErrQ4KLoadEstimateUnsupported for the routes walkQ4KLoadStorage does not qualify
// (MoE, split fused projections, unbounded streaming, expert shards, W3).
func (s *WeightSource) EstimateQ4KHostLoadPeak(mapped bool, opts ...Q4KLoadOption) (HostLoadPeak, error) {
	byType := map[TensorType]*HostLoadTypeRow{}
	var windows []int64
	var out HostLoadPeak
	var hostDenseStreamed int64
	add := func(dst *int64, n int64) error {
		if n < 0 || *dst > math.MaxInt64-n {
			return fmt.Errorf("gguf: host load peak estimate overflows int64")
		}
		*dst += n
		return nil
	}
	if s == nil || s.File == nil {
		return HostLoadPeak{}, fmt.Errorf("gguf: Q4K estimate requires a weight source")
	}
	cfgSeen, err := s.File.Config()
	if err != nil {
		return HostLoadPeak{}, err
	}
	// The residency predicates depend only on the option list; the walk resolves (and validates)
	// the same list again before it visits anything.
	residency, err := resolveQ4KLoadOptions(cfgSeen, opts)
	if err != nil {
		return HostLoadPeak{}, err
	}
	_, loadOpts, err := s.walkQ4KLoadStorage(opts, func(st q4kTensorStorage) error {
		row := byType[st.info.Type]
		if row == nil {
			row = &HostLoadTypeRow{Type: st.info.Type.String()}
			byType[st.info.Type] = row
		}
		payload, err := checkedEstimateUint64(st.payload, "payload", st.info.Name)
		if err != nil {
			return err
		}
		stored, err := checkedEstimateUint64(st.bytes, "stored bytes", st.info.Name)
		if err != nil {
			return err
		}
		f32, err := checkedEstimateMulUint64(st.elems, 4, "f32 bytes", st.info.Name)
		if err != nil {
			return err
		}
		row.Tensors++
		if err := add(&row.PayloadBytes, payload); err != nil {
			return err
		}
		window := int64(0)
		switch {
		case st.streamedBounded:
			// A range descriptor at load; the bounded working set is charged once below.
			if err := add(&row.StreamBytes, stored); err != nil {
				return err
			}
			if err := add(&hostDenseStreamed, stored); err != nil {
				return err
			}
		case st.kind == q4kStoragePacked && mapped && mappedResidentFileBacked(cfgSeen, st, residency):
			if err := add(&row.MappedBytes, stored); err != nil {
				return err
			}
		case st.kind == q4kStoragePacked || st.kind == q4kStoragePackedEmbedding:
			// The worker's read buffer becomes (or is copied into) the resident tensor.
			if err := add(&row.PackedBytes, stored); err != nil {
				return err
			}
			window = payload
		case st.kind == q4kStorageQ8:
			if err := add(&row.Q8Bytes, stored); err != nil {
				return err
			}
			// raw read + f32 dequant + the normalized f32 the canonicalizer may return.
			window, err = sumWindow(payload, f32, f32)
			if err != nil {
				return err
			}
		default: // q4kStorageF32
			if err := add(&row.F32Bytes, stored); err != nil {
				return err
			}
			if err := add(&out.ArenaCopyBytes, stored); err != nil {
				return err
			}
			// raw read + the f32 dequant; the builder's appended chunk is the steady F32 above.
			window, err = sumWindow(payload, f32)
			if err != nil {
				return err
			}
		}
		if st.tiedHeadQ8 > 0 {
			head, err := checkedEstimateUint64(st.tiedHeadQ8, "tied head bytes", st.info.Name)
			if err != nil {
				return err
			}
			if err := add(&row.Q8Bytes, head); err != nil {
				return err
			}
		}
		windows = append(windows, window)
		return nil
	})
	if err != nil {
		return HostLoadPeak{}, err
	}

	for _, r := range byType {
		out.Rows = append(out.Rows, *r)
		if err := add(&out.SteadyAnonBytes, r.AnonBytes()); err != nil {
			return HostLoadPeak{}, err
		}
		if err := add(&out.MappedBytes, r.MappedBytes); err != nil {
			return HostLoadPeak{}, err
		}
	}
	sort.Slice(out.Rows, func(i, j int) bool {
		if a, b := out.Rows[i].AnonBytes(), out.Rows[j].AnonBytes(); a != b {
			return a > b
		}
		return out.Rows[i].Type < out.Rows[j].Type
	})
	if loadOpts.streamedDenseBounded {
		resident := hostDenseStreamed
		if loadOpts.streamedDenseBytes < resident {
			resident = loadOpts.streamedDenseBytes
		}
		if err := add(&out.SteadyAnonBytes, resident); err != nil {
			return HostLoadPeak{}, err
		}
	}

	out.Workers = loadWorkers()
	out.WorkerWindowBytes, err = maxContiguousWindowSum(windows, out.Workers)
	if err != nil {
		return HostLoadPeak{}, err
	}
	out.TransientBytes = max(out.WorkerWindowBytes, out.ArenaCopyBytes)
	out.PeakAnonBytes = out.SteadyAnonBytes
	if err := add(&out.PeakAnonBytes, out.TransientBytes); err != nil {
		return HostLoadPeak{}, err
	}
	return out, nil
}

// mappedResidentFileBacked mirrors computeQ4KTensorWork under the WithStreamedDenseQ4K(true) the
// mapped-resident entry adds: only an identity-layout Q4_K matmul and a retained Q2_K/Q3_K/Q5_K/
// Q6_K the lazy k-quant store can hold become range descriptors (and then views into the map).
// MTP roles, native-row reorders and i-quants are read into owned copies on that route too.
func mappedResidentFileBacked(cfg model.Config, st q4kTensorStorage, o q4kLoadOptions) bool {
	if st.qwenMTP || st.kind != q4kStoragePacked {
		return false
	}
	if st.info.Type == TensorQ4_K {
		return model.ResidentQ4KEligible(cfg, st.canon)
	}
	return denseKQuantRetained(o, st.info.Type) && lazyDenseKQuantBoundedEligible(cfg, st.info.Type, st.canon)
}

func sumWindow(parts ...int64) (int64, error) {
	var total int64
	for _, p := range parts {
		if p < 0 || total > math.MaxInt64-p {
			return 0, fmt.Errorf("gguf: host load peak window overflows int64")
		}
		total += p
	}
	return total, nil
}

// maxContiguousWindowSum is the largest sum over any w consecutive entries (all of them when
// w >= len).
func maxContiguousWindowSum(windows []int64, w int) (int64, error) {
	if w < 1 {
		w = 1
	}
	var cur, best int64
	for i, n := range windows {
		if cur > math.MaxInt64-n {
			return 0, fmt.Errorf("gguf: host load peak window overflows int64")
		}
		cur += n
		if i >= w {
			cur -= windows[i-w]
		}
		if cur > best {
			best = cur
		}
	}
	return best, nil
}

// HostLoadPeakError is the typed, actionable refusal of a load whose predicted host peak does not
// fit the host's allocatable memory with the margin.
type HostLoadPeakError struct {
	NeedBytes   int64             // predicted peak anonymous bytes
	AvailBytes  int64             // MemAvailable (or the platform equivalent) when probed
	MarginBytes int64             // safety margin left unclaimed
	Peak        HostLoadPeak      // the estimate, for reports
	Dominant    []HostLoadTypeRow // the largest per-type contributors
}

func (e *HostLoadPeakError) Unwrap() error { return ErrHostLoadPeakTooBig }

func (e *HostLoadPeakError) Error() string {
	const gib = float64(1 << 30)
	var b strings.Builder
	fmt.Fprintf(&b, "%s: native resident load needs ~%.2f GiB of host RAM at peak (steady %.2f GiB + transient %.2f GiB",
		ErrHostLoadPeakTooBig, float64(e.NeedBytes)/gib, float64(e.Peak.SteadyAnonBytes)/gib, float64(e.Peak.TransientBytes)/gib)
	if e.Peak.MappedBytes > 0 {
		fmt.Fprintf(&b, "; %.2f GiB more stays file-backed", float64(e.Peak.MappedBytes)/gib)
	}
	fmt.Fprintf(&b, ") plus a %.2f GiB margin, but the host has %.2f GiB available; refusing before any tensor is read to avoid host memory exhaustion",
		float64(e.MarginBytes)/gib, float64(e.AvailBytes)/gib)
	if len(e.Dominant) > 0 {
		b.WriteString("; dominant host types:")
		for i, r := range e.Dominant {
			if i > 0 {
				b.WriteString(",")
			}
			fmt.Fprintf(&b, " %s %.2f GiB (%s)", r.Type, float64(r.AnonBytes())/gib, hostLoadRowStorage(r))
		}
	}
	fmt.Fprintf(&b, "; free host memory, stop the co-resident model, use a smaller quant, or set %s=off to load anyway", HostLoadPeakOverrideEnv)
	return b.String()
}

func hostLoadRowStorage(r HostLoadTypeRow) string {
	const gib = float64(1 << 30)
	var parts []string
	if r.F32Bytes > 0 {
		parts = append(parts, fmt.Sprintf("f32 %.2f", float64(r.F32Bytes)/gib))
	}
	if r.Q8Bytes > 0 {
		parts = append(parts, fmt.Sprintf("q8 %.2f", float64(r.Q8Bytes)/gib))
	}
	if r.PackedBytes > 0 {
		parts = append(parts, fmt.Sprintf("packed %.2f", float64(r.PackedBytes)/gib))
	}
	return strings.Join(parts, " + ")
}

// AdmitHostLoadPeak is the pure admission decision: nil when the host's available memory is
// unknown (fail open, like every other capacity rung) or when peak + margin fits it, else a
// *HostLoadPeakError naming the bytes and the dominant tensor types.
func AdmitHostLoadPeak(peak HostLoadPeak, availBytes int64, availKnown bool, marginBytes int64) error {
	if !availKnown || availBytes < 0 || peak.PeakAnonBytes <= 0 {
		return nil
	}
	if marginBytes < 0 {
		marginBytes = 0
	}
	need := peak.PeakAnonBytes
	if need <= math.MaxInt64-marginBytes && need+marginBytes <= availBytes {
		return nil
	}
	return &HostLoadPeakError{
		NeedBytes:   need,
		AvailBytes:  availBytes,
		MarginBytes: marginBytes,
		Peak:        peak,
		Dominant:    peak.Dominant(3),
	}
}
