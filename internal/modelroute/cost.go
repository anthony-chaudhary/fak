package modelroute

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// ---------------------------------------------------------------------------
// THE COST LENS — a ROUGH "usage saved vs SOTA" estimate for a routing Decision.
// ---------------------------------------------------------------------------
//
// Model routing earns its keep by NOT sending every aspect to one big model. The
// SOTA / naive default — what a plain agent loop does, and the floor a
// request-level router (RouteLLM, Martian, …) reduces FROM — is one frontier model
// for the whole request. fak routes the cheap aspects (a short interactive turn, a
// read-shaped tool call, a retrieval) to a cheap tier and pays the frontier price
// only on the hard ones. This file makes that trade visible: for a routed
// Decision it prints, roughly, how much cheaper (or — for a guard/best-of ENSEMBLE
// — how much MORE) the chosen Plan is than always-frontier.
//
// HONESTY (the load-bearing part — a savings claim is only worth as much as the
// evidence under it):
//   - Every figure is a ROUGH order-of-magnitude ESTIMATE, anchored to the repo's
//     published price convention (Opus-class $3/Mtok in, $15/Mtok out — see
//     experiments/parity and cmd/fanbench's --dollars-in/--dollars-out). It is a
//     cost LENS for choosing a policy, never a bill.
//   - The whole price book is overridable (`fak route --prices`), so the number is
//     a transparent function of stated assumptions, not a hidden claim.
//   - An ENSEMBLE runs more compute than one frontier call, so its "savings" is
//     NEGATIVE — reported as a PREMIUM (a deliberate spend on a high-stakes write),
//     never dressed up as a saving.
//   - A model with no price is charged at the FRONTIER rate (the conservative
//     assumption: it makes a saving harder to claim and a premium an upper bound)
//     and listed in Assumed so an operator can supply the real number.

// Price is a rough $/Mtok price for a model, split input/output to match the
// repo's canonical cost convention. Both figures are estimates for the cost lens,
// never billing inputs.
type Price struct {
	In  float64 `json:"in"`  // $/Mtok input
	Out float64 `json:"out"` // $/Mtok output
}

// PriceBook maps a model id to its rough Price.
type PriceBook map[string]Price

// FrontierAnchor is the SOTA baseline: one Opus-class frontier model for
// everything — what an un-routed agent loop pays per token, and the price a
// request-level router reduces from. Anchored to the repo's published $3 in /
// $15 out convention. fak's per-aspect routing is measured AGAINST this.
var FrontierAnchor = Price{In: 3, Out: 15}

// DefaultPrices is the built-in rough ladder, keyed by the conventional tier
// names fak's manifests use. Every tier shares the frontier's 1:5 in:out ratio,
// so the saved fraction is identical on input and output tokens — i.e. it does
// NOT depend on a workload's prompt:completion mix. Override any of it with
// `fak route --prices model=in/out,...`.
func DefaultPrices() PriceBook {
	return PriceBook{
		"frontier": {3, 15}, "large": {3, 15}, // Opus-class — the SOTA baseline tier
		"mid": {1, 5}, "medium": {1, 5}, "default": {1, 5}, // balanced mid tier
		"small": {0.25, 1.25}, "tiny": {0.25, 1.25}, "mini": {0.25, 1.25}, "nano": {0.25, 1.25}, // small/fast tier
		"local": {0, 0}, "in-kernel": {0, 0}, "on-device": {0, 0}, "kernel": {0, 0}, // no marginal $
	}
}

// Savings is the rough cost lens on a routing Decision: what the routed Plan costs
// vs the frontier-everything baseline. Frac fields are (frontier-routed)/frontier
// per token-direction — positive is a saving, negative is an ensemble premium. All
// dollars are $/Mtok rough estimates.
type Savings struct {
	Frontier     string   `json:"frontier"`      // baseline model id (or "frontier" for the anchor)
	FrontierIn   float64  `json:"frontier_in"`   // $/Mtok in of the baseline
	FrontierOut  float64  `json:"frontier_out"`  // $/Mtok out of the baseline
	RoutedIn     float64  `json:"routed_in"`     // $/Mtok in the routed plan pays (sum over members)
	RoutedOut    float64  `json:"routed_out"`    // $/Mtok out the routed plan pays (sum over members)
	Members      int      `json:"members"`       // models the plan runs (>1 == ensemble: pays them all)
	SavedInFrac  float64  `json:"saved_in_frac"` // fraction saved on input tokens (<0 == premium)
	SavedOutFrac float64  `json:"saved_out_frac"`
	Assumed      []string `json:"assumed,omitempty"` // members with no price (charged at the frontier rate)
	Estimable    bool     `json:"estimable"`         // false => baseline rate is $0; no fraction to report
}

// EstimateSavings prices a routing Decision against the SOTA frontier baseline.
// book supplies known prices (nil => DefaultPrices); frontier names the baseline
// model (empty => the FrontierAnchor). A member with no price is charged at the
// frontier rate and recorded in Assumed. Pure and deterministic — same Decision,
// same book, same result.
func EstimateSavings(d Decision, book PriceBook, frontier string) Savings {
	if book == nil {
		book = DefaultPrices()
	}
	fIn, fOut, fname := FrontierAnchor.In, FrontierAnchor.Out, "frontier"
	if frontier != "" {
		fname = frontier
		if p, ok := book[frontier]; ok {
			fIn, fOut = p.In, p.Out
		}
	}
	s := Savings{Frontier: fname, FrontierIn: fIn, FrontierOut: fOut, Members: len(d.Plan.Members)}
	for _, m := range d.Plan.Members {
		p, ok := book[m.Model]
		if !ok {
			p = Price{In: fIn, Out: fOut} // unpriced -> conservative frontier rate
			s.Assumed = append(s.Assumed, m.Model)
		}
		s.RoutedIn += p.In
		s.RoutedOut += p.Out
	}
	sort.Strings(s.Assumed)
	if fIn > 0 {
		s.SavedInFrac = (fIn - s.RoutedIn) / fIn
	}
	if fOut > 0 {
		s.SavedOutFrac = (fOut - s.RoutedOut) / fOut
		s.Estimable = true
	}
	return s
}

// Headline renders the one-line rough usage note for a human. It reads SAVED for a
// cheaper route, PREMIUM for an ensemble that runs more compute than one frontier
// call, and BASELINE when the routed cost ties the frontier. Always tagged rough +
// overridable so it cannot be misread as a bill. ASCII only (matches printRoute).
func (s Savings) Headline() string {
	const tag = "usage (rough public list prices, overridable; not a bill): "
	if !s.Estimable {
		return tag + fmt.Sprintf("not estimated (baseline %s has a $0 rate in this price book)", s.Frontier)
	}
	frac := s.SavedOutFrac
	var msg string
	switch {
	case frac > 0.005:
		msg = fmt.Sprintf("~%.0f%% cheaper than always-%s -- plan ~$%s vs $%s /Mtok-out (saves ~$%s/Mtok-out)",
			frac*100, s.Frontier, money(s.RoutedOut), money(s.FrontierOut), money(s.FrontierOut-s.RoutedOut))
	case frac < -0.005:
		msg = fmt.Sprintf("+%.0f%% vs one %s call -- %d-model ensemble ~$%s vs $%s /Mtok-out (a deliberate reliability spend)",
			-frac*100, s.Frontier, s.Members, money(s.RoutedOut), money(s.FrontierOut))
	default:
		msg = fmt.Sprintf("~ the %s baseline -- no routing saving on this aspect", s.Frontier)
	}
	if len(s.Assumed) > 0 {
		msg += fmt.Sprintf(" [unpriced, charged at frontier: %s -- pass --prices]", strings.Join(s.Assumed, ", "))
	}
	return tag + msg
}

// ParsePrices reads a --prices spec into a PriceBook overlay: comma-separated
// "model=in/out" pairs (e.g. "small=0.25/1.25,large=3/15"); a single value
// "model=N" sets both directions to N. Fails loud on a malformed pair, mirroring
// the manifest's DisallowUnknownFields discipline. The caller layers the result on
// top of DefaultPrices so a spec need only name the models it overrides.
func ParsePrices(spec string) (PriceBook, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return PriceBook{}, nil
	}
	out := PriceBook{}
	for _, pair := range strings.Split(spec, ",") {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue
		}
		kv := strings.SplitN(pair, "=", 2)
		if len(kv) != 2 || strings.TrimSpace(kv[0]) == "" {
			return nil, fmt.Errorf("modelroute: bad --prices pair %q (want model=in/out or model=N)", pair)
		}
		model := strings.TrimSpace(kv[0])
		in, outv, err := parseInOut(strings.TrimSpace(kv[1]))
		if err != nil {
			return nil, fmt.Errorf("modelroute: --prices %q: %w", pair, err)
		}
		out[model] = Price{In: in, Out: outv}
	}
	return out, nil
}

// Overlay returns a copy of book with every entry of over applied on top.
func (book PriceBook) Overlay(over PriceBook) PriceBook {
	merged := make(PriceBook, len(book)+len(over))
	for k, v := range book {
		merged[k] = v
	}
	for k, v := range over {
		merged[k] = v
	}
	return merged
}

// parseInOut parses "in/out" (two prices) or "N" (one price for both directions).
func parseInOut(s string) (float64, float64, error) {
	if i := strings.IndexByte(s, '/'); i >= 0 {
		in, err := strconv.ParseFloat(strings.TrimSpace(s[:i]), 64)
		if err != nil {
			return 0, 0, fmt.Errorf("input price %q: %w", s[:i], err)
		}
		outv, err := strconv.ParseFloat(strings.TrimSpace(s[i+1:]), 64)
		if err != nil {
			return 0, 0, fmt.Errorf("output price %q: %w", s[i+1:], err)
		}
		return in, outv, nil
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, 0, fmt.Errorf("price %q: %w", s, err)
	}
	return v, v, nil
}

// money formats a $/Mtok figure with up to two decimals, trailing zeros trimmed.
func money(v float64) string {
	s := strconv.FormatFloat(v, 'f', 2, 64)
	if strings.Contains(s, ".") {
		s = strings.TrimRight(s, "0")
		s = strings.TrimRight(s, ".")
	}
	if s == "" || s == "-0" {
		s = "0"
	}
	return s
}
