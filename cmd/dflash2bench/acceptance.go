// acceptance.go prices speculative decoding before a drafter is built.
//
// Adapted from wandb/senpai (Apache-2.0),
// literature_and_guidance/LLM-INFERENCE-OPTIMIZATION-SENPAI-GUIDE.md:294-321,515-555
// @5406c6195372251ce9b9ef7bc9133b5f9c92939c: expected accepted draft tokens are
// the numerator, drafter cost plus roofline-priced verifier width are the
// denominator, the best draft length is picked by priced throughput rather than
// maximized, and acceptance is mocked before any drafter build. The idea is
// re-expressed in this package; no upstream code was copied.

package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

var (
	// ErrInvalidAcceptanceTable rejects a mocked acceptance curve that cannot
	// be produced by any drafter: a row outside [0, L] or a decrease in L.
	ErrInvalidAcceptanceTable = errors.New("invalid acceptance table")
	// ErrInvalidRoofline rejects non-positive sustained rates or negative
	// byte/FLOP amounts in the verifier roofline.
	ErrInvalidRoofline = errors.New("invalid roofline parameters")
	// ErrInvalidDrafterCost rejects negative drafter cost components.
	ErrInvalidDrafterCost = errors.New("invalid drafter cost")
	// ErrInvalidAmberMargin rejects a negative green/amber/red margin.
	ErrInvalidAmberMargin = errors.New("invalid amber margin")
)

const (
	acceptanceSchema   = "fak-dflash2-acceptance-economics/1"
	defaultAmberMargin = 0.05
	decisionGreen      = "GREEN"
	decisionAmber      = "AMBER"
	decisionRed        = "RED"
)

// acceptanceMock holds expected accepted draft tokens per requested draft
// length: table[L] = E[accepted | request L] for L in 0..len(table)-1. It is
// the mocking half of the economics: the curve may come from traces of an
// existing speculator or from a closed-form assumption, and the build/no-build
// decision must not depend on a drafter that does not exist yet.
type acceptanceMock []float64

// GeometricAcceptance returns the acceptance table for independent
// per-position acceptance probability p: E[accepted | L] = p*(1-p^L)/(1-p).
func GeometricAcceptance(p float64, maxL int) (acceptanceMock, error) {
	if maxL < 0 || math.IsNaN(p) || math.IsInf(p, 0) || p < 0 || p > 1 {
		return nil, fmt.Errorf("%w: p=%v maxL=%d", ErrInvalidAcceptanceTable, p, maxL)
	}
	table := make(acceptanceMock, maxL+1)
	switch {
	case p == 0:
	case p == 1:
		for l := range table {
			table[l] = float64(l)
		}
	default:
		for l := 1; l <= maxL; l++ {
			table[l] = p * (1 - math.Pow(p, float64(l))) / (1 - p)
		}
	}
	return table, nil
}

// validate enforces the closed contract every honest acceptance mock obeys:
// each row is finite, within [0, L], and the expectation is non-decreasing in
// L because accepting k positions requires the first k to be correct.
func (a acceptanceMock) validate() error {
	prev := 0.0
	for l, value := range a {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > float64(l) {
			return fmt.Errorf("%w: row %d value %v outside [0,%d]", ErrInvalidAcceptanceTable, l, value, l)
		}
		if value < prev {
			return fmt.Errorf("%w: row %d decreases from %v", ErrInvalidAcceptanceTable, l, prev)
		}
		prev = value
	}
	return nil
}

// rooflineVerifier prices one native verification pass over w positions with a
// roofline: weight bytes move once per pass regardless of width (the
// bandwidth floor), while KV bytes and matmul FLOPs scale with width. Small
// widths sit on the bandwidth plateau, which is what makes speculation pay;
// past the knee the pass is compute-bound and grows linearly in width, which
// is what bounds the useful draft length.
type rooflineVerifier struct {
	WeightBytes   float64 // bytes moved once per pass
	KVBytesPerPos float64 // KV bytes read per verified position
	FlopsPerPos   float64 // matmul FLOPs per verified position
	Bandwidth     float64 // sustained bytes/sec
	Compute       float64 // sustained FLOPs/sec
}

func (v rooflineVerifier) valid() error {
	if !(v.Bandwidth > 0) || !(v.Compute > 0) || v.WeightBytes < 0 || v.KVBytesPerPos < 0 || v.FlopsPerPos < 0 {
		return ErrInvalidRoofline
	}
	// A pass with no priced work would price infinite throughput; at least
	// one byte or FLOP source must move.
	if v.WeightBytes == 0 && v.KVBytesPerPos == 0 && v.FlopsPerPos == 0 {
		return fmt.Errorf("%w: no verifier work priced", ErrInvalidRoofline)
	}
	return nil
}

func (v rooflineVerifier) seconds(w int) float64 {
	if w < 1 {
		return 0
	}
	bytesMoved := v.WeightBytes + v.KVBytesPerPos*float64(w)
	flops := v.FlopsPerPos * float64(w)
	return math.Max(bytesMoved/v.Bandwidth, flops/v.Compute)
}

// drafterCost prices proposing L draft tokens.
type drafterCost struct {
	Fixed    float64 // per-call overhead, charged only when L > 0
	PerToken float64 // per drafted position
}

func (d drafterCost) valid() error {
	if d.Fixed < 0 || d.PerToken < 0 {
		return ErrInvalidDrafterCost
	}
	return nil
}

func (d drafterCost) seconds(l int) float64 {
	if l == 0 {
		return 0
	}
	return d.Fixed + d.PerToken*float64(l)
}

// specDecodeRound is the priced economics of one speculation round at draft
// length L. CommittedTokens counts the anchor plus accepted drafts and no
// bonus token, matching this bench's receipt accounting. VerifierWidth is the
// anchor plus the whole proposed chain: panel verification pays for every
// proposed position even though only the accepted prefix is committed.
type specDecodeRound struct {
	DraftLength         int     `json:"draft_length"`
	AcceptedDraftTokens float64 `json:"expected_accepted_draft_tokens"`
	CommittedTokens     float64 `json:"expected_committed_tokens_per_round"`
	VerifierWidth       int     `json:"verifier_width"`
	VerifierSeconds     float64 `json:"verifier_seconds"`
	DrafterSeconds      float64 `json:"drafter_seconds"`
	RoundSeconds        float64 `json:"round_seconds"`
	TokensPerSecond     float64 `json:"committed_tokens_per_second"`
}

// specDecodeVerdict is the break-even pricing of one drafter candidate: every
// draft length 0..max priced against plain decode, the throughput-best length,
// and a green/amber/red build decision.
type specDecodeVerdict struct {
	Schema      string            `json:"schema"`
	Plain       specDecodeRound   `json:"plain_decode"`
	Rounds      []specDecodeRound `json:"draft_lengths"`
	Best        specDecodeRound   `json:"best"`
	Ratio       float64           `json:"best_over_plain_tokens_per_second"`
	Decision    string            `json:"decision"`
	AmberMargin float64           `json:"amber_margin"`
}

// priceSpecDecode prices every draft length under the mocked acceptance curve
// and returns the verdict. Ties keep the smaller draft length, so the result
// is deterministic for identical inputs.
func priceSpecDecode(v rooflineVerifier, d drafterCost, accept acceptanceMock, amberMargin float64) (specDecodeVerdict, error) {
	if err := v.valid(); err != nil {
		return specDecodeVerdict{}, err
	}
	if err := d.valid(); err != nil {
		return specDecodeVerdict{}, err
	}
	if len(accept) == 0 {
		return specDecodeVerdict{}, fmt.Errorf("%w: empty table", ErrInvalidAcceptanceTable)
	}
	if err := accept.validate(); err != nil {
		return specDecodeVerdict{}, err
	}
	if amberMargin < 0 || math.IsNaN(amberMargin) || math.IsInf(amberMargin, 0) {
		return specDecodeVerdict{}, ErrInvalidAmberMargin
	}
	plainSeconds := v.seconds(1)
	plain := specDecodeRound{CommittedTokens: 1, VerifierWidth: 1, VerifierSeconds: plainSeconds, RoundSeconds: plainSeconds, TokensPerSecond: 1 / plainSeconds}
	verdict := specDecodeVerdict{Schema: acceptanceSchema, Plain: plain, Best: plain, AmberMargin: amberMargin, Rounds: []specDecodeRound{}}
	for l := 1; l < len(accept); l++ {
		round := specDecodeRound{
			DraftLength:         l,
			AcceptedDraftTokens: accept[l],
			CommittedTokens:     1 + accept[l],
			VerifierWidth:       1 + l,
			VerifierSeconds:     v.seconds(1 + l),
			DrafterSeconds:      d.seconds(l),
		}
		round.RoundSeconds = round.VerifierSeconds + round.DrafterSeconds
		round.TokensPerSecond = round.CommittedTokens / round.RoundSeconds
		verdict.Rounds = append(verdict.Rounds, round)
		if round.TokensPerSecond > verdict.Best.TokensPerSecond {
			verdict.Best = round
		}
	}
	verdict.Ratio = verdict.Best.TokensPerSecond / plain.TokensPerSecond
	switch {
	case verdict.Ratio <= 1:
		verdict.Decision = decisionRed
	case verdict.Ratio < 1+amberMargin:
		verdict.Decision = decisionAmber
	default:
		verdict.Decision = decisionGreen
	}
	return verdict, nil
}

func parseAcceptanceTable(raw string, wantLen int) (acceptanceMock, error) {
	parts := strings.Split(raw, ",")
	if len(parts) != wantLen {
		return nil, fmt.Errorf("%w: %d values, want %d", ErrInvalidAcceptanceTable, len(parts), wantLen)
	}
	table := make(acceptanceMock, wantLen)
	for i, part := range parts {
		value, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil {
			return nil, fmt.Errorf("%w: row %d: %v", ErrInvalidAcceptanceTable, i, err)
		}
		table[i] = value
	}
	return table, nil
}

// runAcceptanceCLI prices a drafter candidate from flags alone, without a
// device, a model, or a built drafter: this is the acceptance-mocking step
// that must precede any drafter build.
func runAcceptanceCLI(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("dflash2bench acceptance", flag.ContinueOnError)
	fs.SetOutput(stderr)
	maxDraft := fs.Int("max-draft", 8, "maximum requested draft length L (1..64)")
	acceptanceP := fs.Float64("acceptance-p", -1, "geometric per-position acceptance probability in [0,1]; exactly one of --acceptance-p or --acceptance-table")
	acceptanceTable := fs.String("acceptance-table", "", "comma-separated E[accepted | request L] for L=0..max-draft; exactly one of --acceptance-p or --acceptance-table")
	weightBytes := fs.Float64("weight-bytes", 0, "weight bytes moved once per verification pass")
	kvBytesPerPos := fs.Float64("kv-bytes-per-pos", 0, "KV bytes read per verified position")
	flopsPerPos := fs.Float64("flops-per-pos", 0, "matmul FLOPs per verified position")
	bandwidth := fs.Float64("bandwidth", 0, "sustained verifier bandwidth in bytes/sec (required > 0)")
	compute := fs.Float64("compute", 0, "sustained verifier compute in FLOPs/sec (required > 0)")
	drafterFixed := fs.Float64("drafter-fixed", 0, "per-call drafter overhead in seconds")
	drafterPerToken := fs.Float64("drafter-per-token", 0, "drafter seconds per drafted position")
	amberMargin := fs.Float64("amber-margin", defaultAmberMargin, "minimum best/plain ratio above 1 for GREEN")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() != 0 || *maxDraft < 1 || *maxDraft > 64 || (*acceptanceTable != "") == (*acceptanceP >= 0) {
		fmt.Fprintln(stderr, "usage: dflash2bench acceptance --bandwidth B --compute R [--weight-bytes W] [--kv-bytes-per-pos K] [--flops-per-pos F] (--acceptance-p P | --acceptance-table V,...) [--max-draft L] [--drafter-fixed S] [--drafter-per-token S] [--amber-margin M]")
		return 2
	}
	var accept acceptanceMock
	var err error
	if *acceptanceTable != "" {
		accept, err = parseAcceptanceTable(*acceptanceTable, *maxDraft+1)
	} else {
		accept, err = GeometricAcceptance(*acceptanceP, *maxDraft)
	}
	if err != nil {
		fmt.Fprintf(stderr, "acceptance: %v\n", err)
		return 1
	}
	verdict, err := priceSpecDecode(
		rooflineVerifier{WeightBytes: *weightBytes, KVBytesPerPos: *kvBytesPerPos, FlopsPerPos: *flopsPerPos, Bandwidth: *bandwidth, Compute: *compute},
		drafterCost{Fixed: *drafterFixed, PerToken: *drafterPerToken},
		accept, *amberMargin)
	if err != nil {
		fmt.Fprintf(stderr, "acceptance: %v\n", err)
		return 1
	}
	if err := json.NewEncoder(stdout).Encode(verdict); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}
