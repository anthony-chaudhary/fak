package wirescreen

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"  // register decoder so image.Decode handles GIF
	_ "image/jpeg" // register decoder so image.Decode handles JPEG
	_ "image/png"  // register decoder so image.Decode handle PNG
	"math"
	"math/bits"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// phash.go is the rung-3 multi-modal screenshot-triage arm of the local-model-on-
// the-wire spine (doc.go RUNG 3; issue #571): a perceptual-hash (pHash) frame DEDUP
// proposer. It is the same witnessed-lossy-proposer contract as the Digester
// (digester.go) — bounded by the original pinned in the CAS, so a wrong "unchanged"
// call costs one demand-page fault, never a lost frame; strictly additive and
// one-sided; default-inert — but it emits a DEDUP POINTER ("unchanged, see frame#k")
// instead of a summary, because multi-modal triage collapses a redundant frame to a
// reference, not a gist.
//
// What it does: when selected (FAK_WIRE_SCREEN=phash, or a host registers PhashScreen),
// Summarize decodes an image block from the body, computes a pure-Go DCT perceptual
// hash, and compares it against a bounded store of recently-seen frames. On a hit (a
// near-identical frame already seen) it returns "unchanged, see frame#k"; the
// screenAdapter maps that onto an abi.ScreenDigest advisory, and ctxmmu.MMU.Admit pages
// the duplicate pixels out to a stub that carries the dedup pointer, with the original
// retained in the CAS and a witness Clear + PageIn restoring it byte-exact. On a miss
// (a new frame, an undecodable or sub-threshold image) it declines, so the body falls
// through to today's opaque oversize page-out / allow path — strictly one-sided, never a
// new refusal.
//
// It is ZERO model (pure-Go image hashing, stdlib only) — so unlike the model-backed
// rungs it carries no weights/latency gate, only the default-inert opt-in. The vision
// arms (OCR/VLM collapse-to-text, crop-to-ROI) are BLOCKED on a vision encoder
// (internal/model is text-only; the compute HAL is f32-only) and are filed as future
// sub-tasks; this ships the buildable phash arm only.
//
// Default-inert: with FAK_WIRE_SCREEN unset, ActiveDigester() returns nil so this
// digester is never consulted, and the leaf adds no ABI registration itself
// (screenAdapter, registered only when FAK_WIRE_SCREEN is set, is the bridge). The
// pure-Go binary is unchanged until an operator opts in.

const (
	// phashGrid is the side length of the grayscale grid the hash is computed over.
	// 32x32 feeds the canonical 8x8 low-frequency DCT block.
	phashGrid = 32
	// phashBlock is the low-frequency DCT block kept after the transform (8x8 = 64 bits
	// minus the DC term, which carries brightness not structure).
	phashBlock = 8
	// phashDedupThreshold is the max Hamming distance (of 64 bits) at which two frames
	// are judged "perceptually identical" — a redundant re-send worth collapsing. pHash
	// near-duplicates sit well under 10/64 and identical frames are 0; kept tight so a
	// genuinely different screen is never collapsed.
	phashDedupThreshold = 8
	// phashFrameCapacity bounds the remembered-frame store (process-lifetime, FIFO) so a
	// long-running agent cannot grow it without bound — the same discipline as ctxmmu's
	// held ledger (DefaultMaxHeld).
	phashFrameCapacity = 256
)

var (
	phmu     sync.Mutex
	phframes = map[uint64]int{} // perceptual hash -> assigned frame number
	phorder  []uint64           // FIFO insertion order (bounded eviction)
	phnext   int                // next frame number to assign (starts at 1)
	phdedup  int64              // lifetime count of dedup collapses (observability)
)

// phashDigester is the rung-3 perceptual-hash frame-dedup proposer (issue #571). It is a
// Digester sibling of heuristicDigester — same ScreenDigest disposition, same witness —
// but it authors a dedup pointer only for a frame it judges redundant. It is NOT a
// model: pure-Go image hashing, stdlib only. Inert unless selected by FAK_WIRE_SCREEN=phash.
type phashDigester struct{}

func (phashDigester) Name() string { return "phash" }

// Summarize authors a dedup pointer ("unchanged, see frame#k") when body decodes to an
// image perceptually identical to a recently-seen frame; otherwise it declines
// (ok=false) so the MMU falls through to the opaque oversize page-out / allow path. An
// undecodable or sub-threshold image always declines — strictly one-sided, never a new
// refusal. A NEW frame is remembered (so a later re-send dedups) but not summarized: the
// full frame is strictly better than a pointer the first time it is seen.
func (phashDigester) Summarize(_ context.Context, body []byte, _ string) (string, bool) {
	img := decodeImageBlock(body)
	if img == nil {
		return "", false // not a decodable image -> decline (fall through)
	}
	h, ok := perceptualHash(img)
	if !ok {
		return "", false // image too small/blank to hash -> decline
	}
	phmu.Lock()
	defer phmu.Unlock()
	if frame, found := lookupFrame(h); found {
		atomic.AddInt64(&phdedup, 1)
		return fmt.Sprintf("unchanged, see frame#%d", frame), true
	}
	rememberFrame(h)
	return "", false // new frame -> decline (remember for a later re-send)
}

// Dedups reports how many frames this leaf has collapsed to a dedup pointer over its
// lifetime — the phash peer of Digests() (the digester), Flags() (the screener), and
// ctxmmu.MMU.Digested() (the page-outs the MMU actually used).
func Dedups() int64 { return atomic.LoadInt64(&phdedup) }

// PhashScreen returns an abi.SemanticScreen that advises a dedup pointer for a redundant
// screenshot frame via the perceptual-hash arm (issue #571). It is the PROGRAMMATIC
// opt-in peer of the FAK_WIRE_SCREEN=phash env gate: a host that wants phash dedup
// without the init-time env may register this directly. Default-inert either way —
// registering it (or setting the env) is the operator's explicit choice; otherwise
// nothing fires and the MMU is the bare regex floor.
func PhashScreen() abi.SemanticScreen { return phashScreen{} }

// phashScreen bridges the phash dedup proposer straight to the abi.SemanticScreen seam
// the MMU consults, without going through the FAK_WIRE_SCREEN-selected ActiveDigester.
// It maps a dedup hit to abi.ScreenDigest carrying the "unchanged, see frame#k" pointer,
// exactly as screenAdapter does for the env-selected path; the original pixels are then
// pinned in the CAS by ctxmmu.digestToPointer so a witness Clear + PageIn restores them
// byte-exact.
type phashScreen struct{}

func (phashScreen) ScreenResult(ctx context.Context, c *abi.ToolCall, body []byte) abi.ScreenAdvice {
	tool := ""
	if c != nil {
		tool = c.Tool
	}
	if digest, ok := (phashDigester{}).Summarize(ctx, body, tool); ok && digest != "" {
		return abi.ScreenAdvice{Disposition: abi.ScreenDigest, Digest: digest, By: "wirescreen:phash"}
	}
	return abi.ScreenAdvice{}
}

// rememberFrame records a new perceptual hash and returns its assigned frame number,
// evicting the oldest once the store exceeds capacity (FIFO). The caller holds phmu.
func rememberFrame(h uint64) int {
	if phnext == 0 {
		phnext = 1
	}
	n := phnext
	phnext++
	phframes[h] = n
	phorder = append(phorder, h)
	for len(phframes) > phashFrameCapacity && len(phorder) > 0 {
		old := phorder[0]
		phorder = phorder[1:]
		delete(phframes, old)
	}
	return n
}

// lookupFrame returns the frame number of the closest remembered hash within the dedup
// threshold, and whether one was found. O(frames) over a bounded store — the store is
// capped at phashFrameCapacity, so this never scans an unbounded set. The caller holds phmu.
func lookupFrame(h uint64) (frame int, found bool) {
	bestDist := phashDedupThreshold + 1
	bestFrame := -1
	for stored, n := range phframes {
		if d := hamming(h, stored); d < bestDist {
			bestDist = d
			bestFrame = n
		}
	}
	if bestFrame >= 0 {
		return bestFrame, true
	}
	return 0, false
}

// resetPhashStoreForTest clears the remembered-frame store for deterministic tests. It is
// unexported on purpose — it is not part of the operator surface, only the test harness.
func resetPhashStoreForTest() {
	phmu.Lock()
	phframes = map[uint64]int{}
	phorder = nil
	phnext = 0
	phmu.Unlock()
}

// ---------------------------------------------------------------------------
// Image extraction: handle the shapes a screenshot tool actually emits on the wire.
// ---------------------------------------------------------------------------

// decodeImageBlock extracts a decodable image from a result body, handling raw
// PNG/JPEG/GIF bytes, a base64 encoding of them, or a JSON content array carrying an
// Anthropic {"type":"image","source":{"data":...}} or OpenAI
// {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}} block. It
// returns nil when no image can be extracted (the caller declines).
func decodeImageBlock(body []byte) image.Image {
	if len(body) == 0 {
		return nil
	}
	if img, ok := tryDecodeImage(body); ok {
		return img
	}
	if b, err := base64.StdEncoding.DecodeString(stripWhitespace(string(body))); err == nil && len(b) > 0 {
		if img, ok := tryDecodeImage(b); ok {
			return img
		}
	}
	if data := extractImageBlockData(body); len(data) > 0 {
		if b, err := base64.StdEncoding.DecodeString(string(data)); err == nil && len(b) > 0 {
			if img, ok := tryDecodeImage(b); ok {
				return img
			}
		}
	}
	return nil
}

func tryDecodeImage(b []byte) (image.Image, bool) {
	if len(b) == 0 {
		return nil, false
	}
	img, _, err := image.Decode(bytes.NewReader(b))
	if err != nil {
		return nil, false
	}
	return img, true
}

// extractImageBlockData pulls the base64 payload out of an Anthropic or OpenAI image
// content block embedded in body. Returns nil when no image block is found.
func extractImageBlockData(body []byte) []byte {
	var blocks []map[string]json.RawMessage
	if err := json.Unmarshal(body, &blocks); err != nil {
		return nil
	}
	for _, blk := range blocks {
		switch rawString(blk["type"]) {
		case "image": // Anthropic: {"source":{"type":"base64","media_type":"...","data":"..."}}
			if src := rawObject(blk["source"]); src != nil {
				if d := rawString(src["data"]); d != "" {
					return []byte(d)
				}
			}
		case "image_url": // OpenAI: {"image_url":{"url":"data:image/png;base64,..."}}
			if iu := rawObject(blk["image_url"]); iu != nil {
				if _, d, ok := parseDataURI(rawString(iu["url"])); ok {
					return d
				}
			}
		}
	}
	return nil
}

func rawString(r json.RawMessage) string {
	if len(r) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(r, &s) == nil {
		return s
	}
	return ""
}

func rawObject(r json.RawMessage) map[string]json.RawMessage {
	if len(r) == 0 {
		return nil
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(r, &m) == nil {
		return m
	}
	return nil
}

func parseDataURI(u string) (media string, data []byte, ok bool) {
	const marker = ";base64,"
	if !strings.HasPrefix(u, "data:") {
		return "", nil, false
	}
	rest := u[len("data:"):]
	i := strings.Index(rest, marker)
	if i < 0 {
		return "", nil, false
	}
	return rest[:i], []byte(rest[i+len(marker):]), true
}

func stripWhitespace(s string) string {
	return strings.Map(func(r rune) rune {
		if r == '\n' || r == '\r' || r == ' ' || r == '\t' {
			return -1
		}
		return r
	}, s)
}

// ---------------------------------------------------------------------------
// Pure-Go perceptual hash (pHash): downsample -> grayscale -> 2D DCT -> low-frequency
// block -> median threshold. Stdlib only, no dependencies.
// ---------------------------------------------------------------------------

// perceptualHash computes a 64-bit DCT perceptual hash of img (the canonical pHash). It
// downsamples to phashGrid x phashGrid grayscale, takes the 2D DCT, keeps the top-left
// phashBlock x phashBlock low-frequency coefficients (dropping the DC term), and
// thresholds at their median. Two perceptually identical frames hash within a few bits;
// identical frames hash equal. ok is false for an image too small to downsample.
func perceptualHash(img image.Image) (uint64, bool) {
	gray := downsampleGray(img, phashGrid)
	if gray == nil {
		return 0, false
	}
	dct := dct2D(gray, phashGrid)
	coeffs := make([]float64, 0, phashBlock*phashBlock)
	for u := 0; u < phashBlock; u++ {
		for v := 0; v < phashBlock; v++ {
			if u == 0 && v == 0 {
				continue // DC term: overall brightness, not structure
			}
			coeffs = append(coeffs, dct[u][v])
		}
	}
	med := median(coeffs)
	var h uint64
	for i, c := range coeffs {
		if c > med {
			h |= uint64(1) << uint(i)
		}
	}
	return h, true
}

// downsampleGray box-filters img to an n x n grid of luma values in the 0-255 range. The
// box average is dependency-free and bounded; nil for a zero-size image.
func downsampleGray(img image.Image, n int) [][]float64 {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w < 1 || h < 1 {
		return nil
	}
	out := make([][]float64, n)
	for y := 0; y < n; y++ {
		row := make([]float64, n)
		y0 := b.Min.Y + y*h/n
		y1 := b.Min.Y + (y+1)*h/n
		if y1 <= y0 {
			y1 = y0 + 1
		}
		for x := 0; x < n; x++ {
			x0 := b.Min.X + x*w/n
			x1 := b.Min.X + (x+1)*w/n
			if x1 <= x0 {
				x1 = x0 + 1
			}
			var sum float64
			var cnt int
			for yy := y0; yy < y1; yy++ {
				for xx := x0; xx < x1; xx++ {
					r, g, bv, _ := img.At(xx, yy).RGBA() // 16-bit channels
					sum += 0.299*float64(r) + 0.587*float64(g) + 0.114*float64(bv)
					cnt++
				}
			}
			if cnt > 0 {
				row[x] = sum / float64(cnt) / 257.0 // back to 0-255
			}
		}
		out[y] = row
	}
	return out
}

// dct2D returns the 2D DCT-II of an n x n grid (separable: rows then columns). The
// pHash only compares coefficients against their median, so the standard normalization
// factors are omitted without affecting the resulting bits.
func dct2D(g [][]float64, n int) [][]float64 {
	rows := make([][]float64, n)
	for i := 0; i < n; i++ {
		rows[i] = dct1D(g[i], n)
	}
	out := make([][]float64, n)
	for i := range out {
		out[i] = make([]float64, n)
	}
	for j := 0; j < n; j++ {
		col := make([]float64, n)
		for i := 0; i < n; i++ {
			col[i] = rows[i][j]
		}
		colDCT := dct1D(col, n)
		for i := 0; i < n; i++ {
			out[i][j] = colDCT[i]
		}
	}
	return out
}

func dct1D(x []float64, n int) []float64 {
	out := make([]float64, n)
	for k := 0; k < n; k++ {
		var sum float64
		kf := float64(k)
		nf := float64(n)
		for i := 0; i < n; i++ {
			sum += x[i] * math.Cos(math.Pi*(float64(i)+0.5)*kf/nf)
		}
		out[k] = sum
	}
	return out
}

func median(a []float64) float64 {
	if len(a) == 0 {
		return 0
	}
	s := append([]float64(nil), a...)
	sort.Float64s(s)
	return s[len(s)/2]
}

func hamming(a, b uint64) int { return bits.OnesCount64(a ^ b) }
