package main

import (
	"fmt"
	"io"
	"math"
	"math/big"
	"strconv"
	"strings"
)

// defaultMaxRSSHeadroomPct is the margin `--max-rss auto` (and a stale explicit
// ceiling's raise) adds above the measured idle footprint plus one session's KV.
const defaultMaxRSSHeadroomPct = 15.0

// maxRSSSpec is the parsed --max-rss / FAK_UP_MAX_RSS value.
// Bytes==0 && !Auto means the guard is disabled.
type maxRSSSpec struct {
	Bytes uint64
	Auto  bool
}

// disabled reports whether the spec arms no ceiling at all.
func (s maxRSSSpec) disabled() bool { return s.Bytes == 0 && !s.Auto }

// maxRSSSuffixShift maps a lower-cased binary size suffix to its power-of-two shift.
var maxRSSSuffixShift = map[string]uint{
	"k": 10, "kib": 10,
	"m": 20, "mib": 20,
	"g": 30, "gib": 30,
	"t": 40, "tib": 40,
}

// parseMaxRSSSpec parses a --max-rss value: "" or "0" disables the guard, "auto"
// derives the ceiling from the measured footprint, a plain decimal integer is bytes,
// and a number with a binary suffix (K/KiB/M/MiB/G/GiB/T/TiB, 1024-based,
// case-insensitive, optional space) is scaled to bytes.
func parseMaxRSSSpec(s string) (maxRSSSpec, error) {
	v := strings.ToLower(strings.TrimSpace(s))
	if v == "" {
		return maxRSSSpec{}, nil
	}
	if v == "auto" {
		return maxRSSSpec{Auto: true}, nil
	}
	// Split the leading number from the suffix.
	i := 0
	for i < len(v) && (v[i] >= '0' && v[i] <= '9' || v[i] == '.') {
		i++
	}
	num, suffix := v[:i], strings.TrimSpace(v[i:])
	if num == "" {
		return maxRSSSpec{}, fmt.Errorf("invalid --max-rss %q: want bytes, a size like 30GiB, or \"auto\"", s)
	}
	if suffix == "" {
		n, err := strconv.ParseUint(num, 10, 64)
		if err != nil {
			return maxRSSSpec{}, fmt.Errorf("invalid --max-rss %q: want bytes, a size like 30GiB, or \"auto\"", s)
		}
		return maxRSSSpec{Bytes: n}, nil
	}
	shift, ok := maxRSSSuffixShift[suffix]
	if !ok {
		return maxRSSSpec{}, fmt.Errorf("invalid --max-rss %q: unknown size suffix %q (want K, KiB, M, MiB, G, GiB, T, or TiB)", s, suffix)
	}
	if strings.Count(num, ".") > 1 || strings.HasPrefix(num, ".") || strings.HasSuffix(num, ".") {
		return maxRSSSpec{}, fmt.Errorf("invalid --max-rss %q: malformed number", s)
	}
	r, ok := new(big.Rat).SetString(num)
	if !ok {
		return maxRSSSpec{}, fmt.Errorf("invalid --max-rss %q: malformed number", s)
	}
	r.Mul(r, new(big.Rat).SetInt(new(big.Int).Lsh(big.NewInt(1), shift)))
	// Round up so a fractional size never yields a ceiling below what was asked.
	q := new(big.Int).Quo(r.Num(), r.Denom())
	if new(big.Rat).SetInt(q).Cmp(r) < 0 {
		q.Add(q, big.NewInt(1))
	}
	if !q.IsUint64() {
		return maxRSSSpec{}, fmt.Errorf("invalid --max-rss %q: overflows 64-bit bytes", s)
	}
	return maxRSSSpec{Bytes: q.Uint64()}, nil
}

// deriveMaxRSSCeiling returns ceil((resident+kvReserve) * (1 + headroomPct/100)).
// headroomPct <= 0 uses defaultMaxRSSHeadroomPct. The result is always strictly above
// resident+kvReserve when that sum is positive, and 0 when resident == 0 (unknown
// footprint). It saturates at math.MaxUint64 instead of wrapping.
func deriveMaxRSSCeiling(resident, kvReserve uint64, headroomPct float64) uint64 {
	if resident == 0 {
		return 0
	}
	if headroomPct <= 0 || math.IsNaN(headroomPct) || math.IsInf(headroomPct, 0) {
		headroomPct = defaultMaxRSSHeadroomPct
	}
	base := resident + kvReserve
	if base < resident { // overflow
		return math.MaxUint64
	}
	// Exact rational arithmetic: base * (100 + pct) / 100, rounded up.
	pct := new(big.Rat).SetFloat64(headroomPct)
	if pct == nil {
		pct = new(big.Rat).SetFloat64(defaultMaxRSSHeadroomPct)
	}
	factor := new(big.Rat).Add(big.NewRat(1, 1), new(big.Rat).Quo(pct, big.NewRat(100, 1)))
	r := new(big.Rat).Mul(new(big.Rat).SetInt(new(big.Int).SetUint64(base)), factor)
	q := new(big.Int).Quo(r.Num(), r.Denom())
	if new(big.Rat).SetInt(q).Cmp(r) < 0 {
		q.Add(q, big.NewInt(1))
	}
	if !q.IsUint64() {
		return math.MaxUint64
	}
	out := q.Uint64()
	if out <= base {
		if base == math.MaxUint64 {
			return base
		}
		out = base + 1
	}
	return out
}

// maxRSSResolution is the boot-time decision for the memory ceiling.
type maxRSSResolution struct {
	Ceiling   uint64 // 0 = guard disabled
	Source    string // "disabled" | "explicit" | "auto" | "auto_raised" | "auto_unmeasured"
	Requested uint64 // explicit bytes asked for (0 if none)
	Resident  uint64
	SessionKV uint64
}

// resolveMaxRSSCeiling turns the operator's --max-rss spec plus the measured idle
// footprint into the ceiling the memory guard and host admission budget arm with.
// An explicit ceiling that has gone stale (below resident + one session's KV) is
// raised to the derived ceiling unless strict, in which case boot is refused with a
// *MaxRSSBelowResidentError that names the ceiling that would work.
func resolveMaxRSSCeiling(spec maxRSSSpec, resident, sessionKV uint64, headroomPct float64, strict bool) (maxRSSResolution, error) {
	res := maxRSSResolution{Resident: resident, SessionKV: sessionKV}
	switch {
	case spec.disabled():
		res.Source = "disabled"
		return res, nil
	case spec.Auto:
		if resident == 0 {
			res.Source = "auto_unmeasured"
			return res, nil
		}
		res.Ceiling = deriveMaxRSSCeiling(resident, sessionKV, headroomPct)
		res.Source = "auto"
		return res, nil
	}
	res.Requested = spec.Bytes
	if resident == 0 || resident+sessionKV < spec.Bytes {
		res.Ceiling = spec.Bytes
		res.Source = "explicit"
		return res, nil
	}
	suggested := deriveMaxRSSCeiling(resident, sessionKV, headroomPct)
	if strict {
		return res, &MaxRSSBelowResidentError{Ceiling: spec.Bytes, ResidentBytes: resident, SessionKV: sessionKV, Suggested: suggested}
	}
	res.Ceiling = suggested
	res.Source = "auto_raised"
	return res, nil
}

// upMaxRSSOptions is the resolved --max-rss family (flag over env).
type upMaxRSSOptions struct {
	spec        maxRSSSpec
	headroomPct float64
	strict      bool
}

// resolveUpMaxRSSOptions folds --max-rss / --max-rss-headroom / --max-rss-strict with
// their FAK_UP_MAX_RSS* env fallbacks; an explicitly set flag wins over the env. An
// unparsable value is an error (a usage error for the caller), never a silent 0.
func resolveUpMaxRSSOptions(f upFlagSet, explicit map[string]bool, getenv func(string) string) (upMaxRSSOptions, error) {
	var opts upMaxRSSOptions
	raw := ""
	if f.maxRSS != nil {
		raw = *f.maxRSS
	}
	if !explicit["max-rss"] {
		if v := strings.TrimSpace(getenv("FAK_UP_MAX_RSS")); v != "" {
			raw = v
		}
	}
	spec, err := parseMaxRSSSpec(raw)
	if err != nil {
		return opts, err
	}
	opts.spec = spec

	opts.headroomPct = defaultMaxRSSHeadroomPct
	if f.maxRSSHeadroom != nil {
		opts.headroomPct = *f.maxRSSHeadroom
	}
	if !explicit["max-rss-headroom"] {
		if v := strings.TrimSpace(getenv("FAK_UP_MAX_RSS_HEADROOM")); v != "" {
			pct, err := strconv.ParseFloat(strings.TrimSuffix(v, "%"), 64)
			if err != nil || math.IsNaN(pct) || math.IsInf(pct, 0) {
				return opts, fmt.Errorf("invalid FAK_UP_MAX_RSS_HEADROOM %q: want a percentage like 15", v)
			}
			opts.headroomPct = pct
		}
	}
	if math.IsNaN(opts.headroomPct) || math.IsInf(opts.headroomPct, 0) || opts.headroomPct < 0 {
		return opts, fmt.Errorf("invalid --max-rss-headroom %v: want a non-negative percentage", opts.headroomPct)
	}
	if opts.headroomPct == 0 {
		opts.headroomPct = defaultMaxRSSHeadroomPct
	}

	if f.maxRSSStrict != nil {
		opts.strict = *f.maxRSSStrict
	}
	if !explicit["max-rss-strict"] {
		switch strings.ToLower(strings.TrimSpace(getenv("FAK_UP_MAX_RSS_STRICT"))) {
		case "1", "true", "yes", "on":
			opts.strict = true
		}
	}
	return opts, nil
}

// resolveMaxRSS measures the idle footprint and one session's KV reserve and resolves
// the ceiling the memory guard and host admission budget arm with. A disabled spec
// measures nothing.
func (s *turnkeyServer) resolveMaxRSS(opts upMaxRSSOptions) (maxRSSResolution, error) {
	var resident, sessionKV uint64
	if s != nil && !opts.spec.disabled() {
		resident = upResidentFootprint()
		sessionKV = s.capacity().PerSessionKVBytes
	}
	return resolveMaxRSSCeiling(opts.spec, resident, sessionKV, opts.headroomPct, opts.strict)
}

// logMaxRSSResolution writes the operator-visible ceiling line (only when a ceiling is
// armed) and a loud WARN when a stale explicit ceiling was raised.
func logMaxRSSResolution(w io.Writer, res maxRSSResolution, headroomPct float64) {
	if w == nil {
		return
	}
	if headroomPct <= 0 {
		headroomPct = defaultMaxRSSHeadroomPct
	}
	if res.Source == "auto_raised" {
		fmt.Fprintf(w, "fak up: WARN --max-rss %s (%d bytes) is at or below the measured footprint %s (resident %s + session KV %s); raised the memory ceiling to %s (%d bytes). This hardcoded ceiling is stale: change it to `--max-rss auto` (or pass --max-rss-strict to refuse to start instead)\n",
			formatBytes(res.Requested), res.Requested, formatBytes(res.Resident+res.SessionKV), formatBytes(res.Resident), formatBytes(res.SessionKV), formatBytes(res.Ceiling), res.Ceiling)
	}
	if res.Source == "auto_unmeasured" {
		fmt.Fprintln(w, "fak up: --max-rss auto could not measure the resident footprint on this host; the memory guard stays disabled")
	}
	if res.Ceiling == 0 {
		return
	}
	fmt.Fprintf(w, "fak up: memory ceiling %s (source %s; resident %s + session KV %s, headroom %g%%)\n",
		formatBytes(res.Ceiling), res.Source, formatBytes(res.Resident), formatBytes(res.SessionKV), headroomPct)
}
