package headroom

import (
	"context"
	"strconv"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// AdmitRank places the compression gate AFTER the de-obfuscation rescan (normgate,
// rank 5) and as a peer just ahead of the context-MMU (ctxmmu, rank 10). At rank 8
// a clean BENIGN result is offered to the selected compressor before ctxmmu would
// page an oversize one out to an OPAQUE pointer — compression keeps the bytes
// MODEL-READABLE ("same answer, fewer tokens"), where a page-out makes the model
// retrieve. Because Transform (the compress verdict) outranks Allow but is
// outranked by Quarantine in the fold, a poisoned result still loses to the
// security gates; and because this gate SCREENS the raw bytes itself (ScreenBytes)
// and declines to touch poison, it can never hide an injection from the gates that
// screen raw bytes downstream.
const AdmitRank = 8

// Gate is the result-admission driver that folds the selected Compressor into the
// kernel result path (it is also reached from the gateway's inbound result
// admission, so `fak guard -- claude` compresses tool results in-stream with the
// same one registration). Construct with NewGate; Default is the registered one.
type Gate struct {
	considered int64
	compressed int64
	bytesIn    int64
	bytesOut   int64
}

// NewGate builds a fresh gate (its counters are independent).
func NewGate() *Gate { return &Gate{} }

// Default is the registered gate; its counters are the process-wide compression
// KPI surfaced by `fak headroom status`.
var Default = NewGate()

func (g *Gate) Caps() []abi.Capability { return nil }

// Admit offers a benign tool result to the selected Compressor and, on a genuine
// saving, rewrites the payload to the smaller MODEL-READABLE rendering
// (VerdictTransform) while preserving the original in the shared CAS so it stays
// retrievable (the reversible-compression / CCR promise). It returns Allow
// (admit-as-is — the ResultAdmitter fold identity) whenever it must not act:
// compression is off (noop selected), the bytes screen as poison (left for the
// security gates), the result is empty, or the compressor found no saving.
func (g *Gate) Admit(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	if r == nil {
		return admitAsIs()
	}
	comp := Selected()
	if comp.Name() == NoopName {
		return admitAsIs() // compression disabled -> zero overhead, no resolve
	}
	atomic.AddInt64(&g.considered, 1)

	body := resolveBytes(ctx, r.Payload)
	if len(body) == 0 {
		return admitAsIs()
	}
	// Security FIRST: never compress what the gates would quarantine. A poisoned
	// result is left raw so ctxmmu/normgate screen the REAL bytes and seal it.
	if _, poison := ctxmmu.ScreenBytes(body); poison {
		return admitAsIs()
	}

	out, err := comp.Compress(ctx, Input{
		Tool:  toolName(c),
		Kind:  Detect(body),
		Model: modelHint(c),
		Bytes: body,
	})
	if err != nil || !out.Compressed || len(out.Bytes) == 0 || len(out.Bytes) >= len(body) {
		return admitAsIs()
	}

	// Preserve the original for retrieval (reversible CCR), then rewrite the
	// payload to the compressed, model-readable rendering.
	origin := preserveOriginal(ctx, body)
	atomic.AddInt64(&g.compressed, 1)
	atomic.AddInt64(&g.bytesIn, int64(len(body)))
	atomic.AddInt64(&g.bytesOut, int64(len(out.Bytes)))

	meta := map[string]string{
		"compressed":  "true",
		"compressor":  comp.Name(),
		"codec":       out.Codec,
		"saved_ratio": strconv.FormatFloat(out.SavedRatio(), 'f', 3, 64),
		"orig_len":    strconv.Itoa(len(body)),
		"new_len":     strconv.Itoa(len(out.Bytes)),
	}
	if out.Retrieval != "" {
		meta["ccr"] = out.Retrieval // external-service retrieval handle(s)
	}
	if origin != "" {
		meta["origin"] = origin // in-CAS digest of the original bytes
	}
	ref := abi.Ref{Kind: abi.RefInline, Inline: out.Bytes, Len: int64(len(out.Bytes))}
	return abi.Verdict{
		Kind:    abi.VerdictTransform,
		By:      "headroom",
		Payload: abi.TransformPayload{NewArgs: ref},
		Meta:    meta,
	}
}

// Stats is the gate's lifetime compression KPI.
type Stats struct {
	Considered int64
	Compressed int64
	BytesIn    int64
	BytesOut   int64
}

// Stats snapshots the gate's counters.
func (g *Gate) Stats() Stats {
	return Stats{
		Considered: atomic.LoadInt64(&g.considered),
		Compressed: atomic.LoadInt64(&g.compressed),
		BytesIn:    atomic.LoadInt64(&g.bytesIn),
		BytesOut:   atomic.LoadInt64(&g.bytesOut),
	}
}

func admitAsIs() abi.Verdict { return abi.Verdict{Kind: abi.VerdictAllow, By: "headroom"} }

func resolveBytes(ctx context.Context, r abi.Ref) []byte {
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(ctx, r); err == nil {
			return b
		}
	}
	return nil
}

// preserveOriginal stores the pre-compression bytes in the shared content-
// addressed store and returns their digest, so a future read can retrieve the
// exact original. Best-effort: if no blob codec is registered it returns "" and
// the compression still proceeds (the external plugin's own CCR may still hold it).
func preserveOriginal(ctx context.Context, body []byte) string {
	b, ok := abi.PageOut("blob")
	if !ok {
		return ""
	}
	h, err := b.PageOut(ctx, abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))})
	if err != nil {
		return ""
	}
	return h.Digest
}

func toolName(c *abi.ToolCall) string {
	if c == nil {
		return ""
	}
	return c.Tool
}

func modelHint(c *abi.ToolCall) string {
	if c == nil || c.Meta == nil {
		return ""
	}
	return c.Meta["model"]
}

func init() {
	abi.RegisterResultAdmitter(AdmitRank, Default)
	abi.RegisterCapability("headroom.v1")
}
