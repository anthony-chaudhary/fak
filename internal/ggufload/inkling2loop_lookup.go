package ggufload

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Draft sources the speculative host loop can verify.
const (
	inkling2DraftLookup = "lookup"
	inkling2DraftDSpark = "dspark"
	inkling2DraftDFlash = "dflash2"
	inkling2DraftAuto   = "auto"
)

// inkling2LookupMinMatch is the shortest repeated suffix worth a verify: one-
// and two-token matches fire on common tokens and mostly draft misses, even
// behind the cost gate.
const inkling2LookupMinMatch = 3

// inkling2LookupMaxMatch bounds the suffix scan; longer matches rarely change
// which occurrence wins.
const inkling2LookupMaxMatch = 64

// inkling2LookupVerifyCost is a verify's wall time in plain decode steps by
// width (index), on an M3 Pro under shared load; the drafter replaces each
// entry with the request's own timing once it has one.
var inkling2LookupVerifyCost = [inkling2SpecMaxDraft + 2]float64{0, 1, 1.34, 1.74, 2.2, 2.66, 3.18, 3.7, 4.22, 4.75}

// inkling2LookupLongMatch splits the acceptance tracker: a suffix this long is
// usually a verbatim copy and keeps far more of its draft than a short one.
const inkling2LookupLongMatch = 8

// inkling2LookupPrior is the per-token accept:reject prior of a fresh request,
// for short and long matches.
var inkling2LookupPrior = [2][2]float64{{2, 1}, {8, 1}}

// Old outcomes fade so a missed phase does not shut drafting off for the rest
// of the reply.
const (
	inkling2LookupDecay = 0.85
	// a verify must beat plain decode by this much to pay for the gamble
	inkling2LookupMargin  = 1.05
	inkling2LookupMaxSkip = 16
)

// inkling2LookupDrafter drafts by prompt lookup: the tokens that followed the
// most recent earlier occurrence of the context's longest repeated suffix. It
// owns no weights and no device state. It drafts only when the expected tokens
// per verify, from this request's acceptance and verify timings, beat plain
// decode, so a miss or a poor bet runs the plain step instead of a verify.
type inkling2LookupDrafter struct {
	ctx    []int // prompt followed by every emitted token seen so far
	prompt int

	acc, rej [2]float64 // decayed per-token accepts and rejects, short and long matches
	bucket   int
	skip     int // plain steps left before the next draft
	backoff  int // skip length after the next zero-accept verify

	ms     [inkling2SpecMaxDraft + 2]float64 // measured wall ms by width; 1 is a plain step
	timed  [inkling2SpecMaxDraft + 2]bool
	gamble bool // the last proposal was cost-gated, so its outcome trains the tracker
}

func newInkling2LookupDrafter(prompt []int) *inkling2LookupDrafter {
	ctx := make([]int, len(prompt), len(prompt)+256)
	copy(ctx, prompt)
	return &inkling2LookupDrafter{ctx: ctx, prompt: len(prompt), backoff: 1}
}

// sync extends the context with the history it has not seen; the loop's
// history only grows, so the context is appended to rather than rebuilt.
func (d *inkling2LookupDrafter) sync(history []int) {
	seen := len(d.ctx) - d.prompt
	if len(history) < seen || (seen > 0 && d.ctx[len(d.ctx)-1] != history[seen-1]) {
		d.ctx = d.ctx[:d.prompt]
		seen = 0
	}
	d.ctx = append(d.ctx, history[seen:]...)
}

// lookup returns up to k tokens after the most recent earlier occurrence of the
// context's longest repeated suffix (at most inkling2LookupMaxMatch), and that
// suffix's length; a suffix shorter than inkling2LookupMinMatch drafts nothing.
func (d *inkling2LookupDrafter) lookup(k int) ([]int, int) {
	c := d.ctx
	last := len(c) - 1
	best, at := 0, -1
	for i := last - 1; i >= 0; i-- {
		n := 0
		for n < inkling2LookupMaxMatch && i-n >= 0 && c[i-n] == c[last-n] {
			n++
		}
		if n > best {
			best, at = n, i
			if n == inkling2LookupMaxMatch {
				break
			}
		}
	}
	if best < inkling2LookupMinMatch || k < 1 {
		return nil, best
	}
	return c[at+1 : min(len(c), at+1+k)], best
}

// cost is the wall time of a verify of width w in plain steps.
func (d *inkling2LookupDrafter) cost(w int) float64 {
	if !d.timed[1] || !d.timed[w] {
		return inkling2LookupVerifyCost[w]
	}
	return d.ms[w] / d.ms[1]
}

// depth picks the draft length with the most expected tokens per plain-step
// time, or 0 when no length beats plain decode. A short match caps the depth:
// its continuation diverges sooner.
func (d *inkling2LookupDrafter) depth(avail, match int) int {
	b := d.bucketOf(match)
	pa, pr := inkling2LookupPrior[b][0], inkling2LookupPrior[b][1]
	q := (d.rej[b] + pr) / (d.acc[b] + d.rej[b] + pa + pr)
	// a longer repeated suffix diverges less often per token
	p := 1 - q
	best, bestRate := 0, inkling2LookupMargin
	expect, pi := 1.0, 1.0
	for n := 1; n <= min(avail, match-1, inkling2SpecMaxDraft); n++ {
		pi *= p
		expect += pi
		if r := expect / d.cost(n+1); r > bestRate {
			best, bestRate = n, r
		}
	}
	return best
}

func (d *inkling2LookupDrafter) Draft(history []int, k int) []int {
	d.sync(history)
	d.gamble = false
	if k < 1 {
		return nil
	}
	if d.skip > 0 {
		d.skip--
		return nil
	}
	cont, match := d.lookup(k)
	n := d.depth(len(cont), match)
	if n == 0 {
		if cont != nil {
			// a refused match forgets a little, so a bad phase is probed again
			b := d.bucketOf(match)
			d.acc[b], d.rej[b] = d.acc[b]*inkling2LookupDecay, d.rej[b]*inkling2LookupDecay
		}
		return nil
	}
	d.gamble, d.bucket = true, d.bucketOf(match)
	return append([]int(nil), cont[:n]...)
}

// Observe trains the acceptance tracker on each verify: keep-1 drafted tokens
// were accepted and, short of the full draft, the next was rejected. A verify
// that kept no draft token backs drafting off for a doubling run of plain steps.
func (d *inkling2LookupDrafter) Observe(_ *inkling2Resident, fed []int, keep int) error {
	if !d.gamble || len(fed) < 2 {
		return nil
	}
	d.gamble = false
	acc := float64(keep - 1)
	b := d.bucket
	d.acc[b], d.rej[b] = d.acc[b]*inkling2LookupDecay+acc, d.rej[b]*inkling2LookupDecay
	if keep < len(fed) {
		d.rej[b]++
	}
	if keep == 1 {
		d.skip = d.backoff
		d.backoff = min(2*d.backoff, inkling2LookupMaxSkip)
	} else {
		d.backoff = 1
	}
	return nil
}

// ObserveCost records the wall time of a plain step (width 1) or a verify of
// width tokens, as a moving average that rides out one-off load spikes.
func (d *inkling2LookupDrafter) ObserveCost(width int, took time.Duration) {
	if width < 1 || width >= len(d.ms) {
		return
	}
	ms := float64(took.Microseconds()) / 1000
	if !d.timed[width] {
		d.ms[width], d.timed[width] = ms, true
		return
	}
	d.ms[width] += 0.25 * (ms - d.ms[width])
}

// inkling2SpecDraftKind is the draft source from FAK_INKLING2_SPEC_DRAFT
// (lookup, dspark, dflash2 or auto; unset is auto).
func inkling2SpecDraftKind() (string, error) {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("FAK_INKLING2_SPEC_DRAFT")))
	switch v {
	case "":
		return inkling2DraftAuto, nil
	case inkling2DraftLookup, inkling2DraftDSpark, inkling2DraftDFlash, inkling2DraftAuto:
		return v, nil
	}
	return "", fmt.Errorf("inkling2 spec: FAK_INKLING2_SPEC_DRAFT=%q, want lookup, dspark, dflash2 or auto", v)
}

// inkling2ResolveSpecDraft turns auto into the neural draft whose weights are
// on disk, else prompt lookup, so speculation never needs a draft model.
// dflashDir is the DFlash2 checkpoint beside the target, "" when it is absent
// or the backend has no DFlash2 ops.
func inkling2ResolveSpecDraft(kind, mtpDir, dflashDir string) string {
	if kind != inkling2DraftAuto {
		return kind
	}
	if mtpDir != "" {
		if fi, err := os.Stat(filepath.Join(mtpDir, inkling2DSparkWeightsFile)); err == nil && fi.Mode().IsRegular() {
			return inkling2DraftDSpark
		}
	}
	if dflashDir != "" {
		if fi, err := os.Stat(filepath.Join(dflashDir, inkling2DFlash2Weights)); err == nil && fi.Mode().IsRegular() {
			return inkling2DraftDFlash
		}
	}
	return inkling2DraftLookup
}

// inkling2DFlash2DirFor is the DFlash2 checkpoint beside m's GGUF when be can
// run it, else "".
func inkling2DFlash2DirFor(m *Inkling2LoopModel, be compute.Backend) string {
	if _, ok := be.(compute.Inkling2DFlash2Device); !ok || m.path == "" {
		return ""
	}
	dir, err := Inkling2DFlash2AdjacentDir(m.path)
	if err != nil {
		return ""
	}
	return dir
}

// bucketOf is the acceptance tracker a match of this length trains and reads.
func (d *inkling2LookupDrafter) bucketOf(match int) int {
	if match >= inkling2LookupLongMatch {
		return 1
	}
	return 0
}
