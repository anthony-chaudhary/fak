// setupparity.go gates a warm-path gain on non-cache SETUP PARITY (issue #5704, a
// leaf of #1279). It answers the question a stopwatch cannot: was the warm trial the
// SAME treatment as the cold one, with only the cache state changed?
//
// The defect it closes is a silent treatment swap. A cold/warm pair is supposed to
// isolate ONE variable — cache state — and attribute the whole timing delta to reuse.
// But nothing stops the warm trial from also running with fewer threads, a shorter
// max_tokens, a different model, or a smaller input slice. The stopwatch reports the
// same speedup either way, so a genuine cache gain and a quietly retuned benchmark are
// indistinguishable from timing alone. So every comparison is bound to a machine-
// readable parity receipt that fingerprints EVERY declared setup field, and a gain is
// reported only when the non-cache fingerprints of the two trials are identical.
//
// Three rules keep it honest.
//
// First, the permitted delta is an EXPLICIT allowlist, not a convention: only field
// names the caller declared as cache state may differ. Every receipt publishes the
// allowlist AND a binding digest over it, so a receipt judged under a quietly widened
// allowlist cannot be mistaken for one judged under the shipped policy.
//
// Second, ABSENT is not EQUAL. A non-cache field the cold trial declared and the warm
// trial omits is a refusal (setup_field_omitted), never a silent match — the classic
// hole, because a field that stops being reported looks like a field that stopped
// differing. A field present with an empty value is a different thing from an absent
// one, and the receipt shows it: an absent side reads "absent" where a present side
// carries a digest.
//
// Third, the pair must actually BE a cold/warm pair. If nothing on the allowlist
// changed, no cache state was exercised and the timing delta is noise or some other
// effect — refused as no_cache_state_delta rather than banked as a cache gain.
//
// Canonicalization is declared, not implicit (CanonicalizationRule): outer whitespace
// is trimmed, fields sort by name, and the digest is LENGTH-PREFIXED so no field name
// or value can impersonate a field boundary — {"a=b": "c"} and {"a": "b=c"} must never
// fingerprint alike. A setup that names the same field twice after canonicalization is
// ambiguous and is refused rather than resolved by iteration order.
//
// The receipt is scrubbed by construction — field NAMES appear in the clear (they are
// the audit surface), values only as field-salted digests — so it publishes without
// leaking the host, model, or path a setup field may quote. It is deterministic (no
// clock, no randomness, no map iteration in output):
//
//	go test ./internal/bench -run TestSetupParity -count=1
//
// (the golden artifact regenerates into testdata/setup_parity.json with UPDATE_GOLDEN=1).
package bench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
)

// CanonicalizationRule is the PUBLISHED canonical form every comparison runs under.
// It rides in each receipt so an auditor reads the policy rather than inferring it
// from this implementation.
const CanonicalizationRule = "trim-outer-whitespace; sort-by-field-name; length-prefixed-digest; duplicate-name-refused"

// The closed typed reason vocabulary. A reason names ONE field's delta, and the
// dominant one becomes the pair's verdict — the same word in both places, so a verdict
// is always traceable back to the field that produced it.
const (
	// ReasonCacheState: an ALLOWLISTED field differs (in value or in presence). This
	// is the declared treatment under test — the only permitted delta.
	ReasonCacheState = "cache_state_changed"
	// ReasonValueChanged: a non-allowlisted field is present on both sides with
	// different canonical values. The warm trial was a different treatment.
	ReasonValueChanged = "setup_value_changed"
	// ReasonFieldOmitted: a non-allowlisted field the cold trial declared is ABSENT
	// from the warm trial. Never folded into "equal".
	ReasonFieldOmitted = "setup_field_omitted"
	// ReasonFieldAdded: a non-allowlisted field only the warm trial declares.
	ReasonFieldAdded = "setup_field_added"
)

// Pair verdicts that no single field produces.
const (
	// VerdictParityOK is the ONLY verdict under which a cache gain may be reported.
	VerdictParityOK = "setup_parity_ok"
	// VerdictSetupNotWitnessed: a trial declared no setup fields at all. Parity over
	// nothing is not parity, so no gain is reported.
	VerdictSetupNotWitnessed = "setup_not_witnessed"
	// VerdictFieldAmbiguous: two declared fields canonicalize to the same name, so
	// which value governs depends on input order. Refused, not resolved.
	VerdictFieldAmbiguous = "setup_field_ambiguous"
	// VerdictNoCacheStateDelta: nothing on the allowlist changed, so this pair is not
	// a cold/warm pair and its delta is not a cache gain.
	VerdictNoCacheStateDelta = "no_cache_state_delta"
)

// Run-level verdicts.
const (
	VerdictSetupParityClean   = "setup_parity_clean"
	VerdictSetupParityFlagged = "setup_parity_flagged"
)

// setupAbsent is the sentinel a receipt carries where a trial did not declare a field
// at all. It is deliberately NOT a digest: an absent field can never be confused with
// a field whose value happens to be empty.
const setupAbsent = "absent"

// SetupField is one declared setup, input, or execution-policy field of a single
// trial — anything whose change could explain a timing delta that is not cache reuse.
type SetupField struct {
	Name  string
	Value string
}

// Trial is one measured leg of a cold/warm comparison: its declared setup and the
// wall time it took.
type Trial struct {
	// Label names the leg ("cold", "warm").
	Label string
	// Setup is every field this leg declares. A field the harness does not report
	// here is not fingerprinted, and so cannot be proven equal — which is why an
	// omitted field is a refusal rather than a match.
	Setup []SetupField
	// DurationMS is the measured wall time in milliseconds.
	DurationMS float64
}

// TrialPair is one cold/warm comparison awaiting a parity verdict.
type TrialPair struct {
	Name string
	Cold Trial
	Warm Trial
}

// CacheStateAllowlist is the EXPLICIT set of setup field names whose difference
// between the cold and the warm trial is the treatment under test. Every other
// declared field must match, so widening this list is the one way to license a new
// delta — and the widening is visible, because each receipt binds to the list's digest.
type CacheStateAllowlist struct {
	fields []string
}

// NewCacheStateAllowlist declares the cache-state fields. Names are canonicalized,
// deduped, and sorted, so the resulting binding does not depend on argument order.
func NewCacheStateAllowlist(fields ...string) CacheStateAllowlist {
	seen := map[string]bool{}
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		n := strings.TrimSpace(f)
		if n == "" || seen[n] {
			continue
		}
		seen[n] = true
		out = append(out, n)
	}
	sort.Strings(out)
	return CacheStateAllowlist{fields: out}
}

// Declared returns the canonical allowlist, for publication in the receipt.
func (a CacheStateAllowlist) Declared() []string {
	out := make([]string, len(a.fields))
	copy(out, a.fields)
	return out
}

// Allows reports whether a canonical field name is declared cache state.
func (a CacheStateAllowlist) Allows(field string) bool {
	for _, f := range a.fields {
		if f == field {
			return true
		}
	}
	return false
}

// Binding is the non-reversible digest over the declared allowlist. Every receipt
// carries it, so a comparison judged under a quietly widened allowlist cannot pass as
// one judged under the shipped policy.
func (a CacheStateAllowlist) Binding() string {
	h := sha256.New()
	h.Write([]byte("fak-cache-state-allowlist/1"))
	for _, f := range a.fields {
		writeLenPrefixed(h, f)
	}
	return "allow:" + hex.EncodeToString(h.Sum(nil))[:16]
}

// TrialFingerprint is one leg's canonical setup identity: the whole setup, and the two
// halves the verdict actually turns on. Non-cache parity is the one-line machine check
// (the two legs' NonCache digests must be equal); a cache-state delta exists exactly
// when the two legs' CacheState digests differ.
type TrialFingerprint struct {
	Label string `json:"label"`
	// Fields is the number of canonical fields the leg declared.
	Fields int `json:"fields"`
	// Setup digests every canonical field — the leg's full setup identity.
	Setup string `json:"setup_fingerprint"`
	// NonCache digests only the fields NOT on the allowlist. Parity holds iff the two
	// legs agree here.
	NonCache string `json:"non_cache_fingerprint"`
	// CacheState digests only the allowlisted fields — the declared treatment.
	CacheState string `json:"cache_state_fingerprint"`
}

// SetupDelta is one field that differs between the legs: the field name in the clear,
// its typed reason, whether that reason is permitted, and each side's value as a
// field-salted digest (or "absent").
type SetupDelta struct {
	Field   string `json:"field"`
	Reason  string `json:"reason"`
	Allowed bool   `json:"allowed"`
	Cold    string `json:"cold_value_digest"`
	Warm    string `json:"warm_value_digest"`
}

// TrialPairReceipt is the machine-readable parity receipt for one cold/warm pair: the
// policy it was judged under, both canonical fingerprints, the exact differing fields,
// and the single dominant verdict that decides whether a gain may be reported at all.
type TrialPairReceipt struct {
	Pair             string   `json:"pair"`
	Canonicalization string   `json:"canonicalization"`
	AllowlistBinding string   `json:"allowlist_binding"`
	Allowlist        []string `json:"cache_state_allowlist"`

	Cold   TrialFingerprint `json:"cold"`
	Warm   TrialFingerprint `json:"warm"`
	Deltas []SetupDelta     `json:"deltas"`

	// NonCacheParity is the headline invariant: the two legs' non-cache fingerprints
	// are identical, so every undeclared setup field was held fixed.
	NonCacheParity bool `json:"non_cache_parity"`
	// CacheStateChanged says the declared treatment actually moved. Without it the
	// pair is not a cold/warm pair at all.
	CacheStateChanged bool `json:"cache_state_changed"`

	Verdict string `json:"verdict"`
	Finding string `json:"finding"`

	ColdMS float64 `json:"cold_ms"`
	WarmMS float64 `json:"warm_ms"`
	// GainReported is false on every refusal: the speedup is withheld, not merely
	// annotated, so a refused comparison cannot be quoted as a cache gain.
	GainReported bool `json:"gain_reported"`
	// SpeedupPct is populated ONLY when GainReported is true.
	SpeedupPct float64 `json:"speedup_pct"`
}

// ParityHolds reports whether this pair may back a cache-gain claim.
func (r TrialPairReceipt) ParityHolds() bool { return r.Verdict == VerdictParityOK }

// RefusedPair names one excluded pair and the typed reason it was excluded.
type RefusedPair struct {
	Pair   string `json:"pair"`
	Reason string `json:"reason"`
}

// SetupParityReport folds a run of cold/warm pairs: which may back a cache-gain claim,
// which were refused and why, and the pooled speedup computed over the admitted pairs
// ONLY — the "exclude or fail closed before reporting a cache gain" half of the contract.
type SetupParityReport struct {
	Schema           string             `json:"schema"`
	Provenance       Provenance         `json:"provenance"`
	Canonicalization string             `json:"canonicalization"`
	AllowlistBinding string             `json:"allowlist_binding"`
	Allowlist        []string           `json:"cache_state_allowlist"`
	Pairs            []TrialPairReceipt `json:"pairs"`
	// GainEligible lists the pair names whose speedup may be reported, in input order.
	GainEligible []string      `json:"gain_eligible"`
	Refused      []RefusedPair `json:"refused"`
	// PooledSpeedupPct is derived from the admitted pairs only; a run with none is 0.
	PooledSpeedupPct float64 `json:"pooled_speedup_pct"`
	Verdict          string  `json:"verdict"`
	Finding          string  `json:"finding"`
}

// JSON renders the report as stable, indented JSON — a re-derivable witness artifact.
func (r SetupParityReport) JSON() ([]byte, error) { return json.MarshalIndent(r, "", "  ") }

// writeLenPrefixed writes a length-prefixed field to a digest. The prefix is what makes
// the fingerprint injection-proof: without it a field NAMED "a=b" and a field named "a"
// VALUED "b=..." could serialize to the same bytes.
func writeLenPrefixed(w io.Writer, s string) {
	fmt.Fprintf(w, "\x00%d\x00%s", len(s), s)
}

// setupValueDigest is the field-salted value digest: the same value under two different
// fields does not produce the same digest, so the receipt cannot be used to correlate
// values across fields, and no raw value ever appears.
func setupValueDigest(field, value string) string {
	sum := sha256.Sum256([]byte("fak-setup-value/1\x00" + field + "\x00" + value))
	return "sha256:" + hex.EncodeToString(sum[:])[:16]
}

// setupFingerprint digests a canonical, sorted field slice under a tagged domain.
func setupFingerprint(tag string, fields []SetupField) string {
	h := sha256.New()
	h.Write([]byte(tag))
	for _, f := range fields {
		writeLenPrefixed(h, f.Name)
		writeLenPrefixed(h, f.Value)
	}
	return "sha256:" + hex.EncodeToString(h.Sum(nil))[:16]
}

// canonicalSetup projects a declared setup onto its canonical form: outer whitespace
// trimmed, sorted by field name. It returns the offending name when the setup is
// AMBIGUOUS — an unnamed field, or two fields that canonicalize to one name — because
// resolving that by iteration order would make the fingerprint depend on input order.
func canonicalSetup(fields []SetupField) ([]SetupField, string) {
	out := make([]SetupField, 0, len(fields))
	for _, f := range fields {
		out = append(out, SetupField{
			Name:  strings.TrimSpace(f.Name),
			Value: strings.TrimSpace(f.Value),
		})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	for i, f := range out {
		if f.Name == "" {
			return out, "(unnamed field)"
		}
		if i > 0 && f.Name == out[i-1].Name {
			return out, f.Name
		}
	}
	return out, ""
}

// fingerprintTrial canonicalizes one leg and digests it three ways: whole, non-cache
// only, and cache-state only.
func fingerprintTrial(t Trial, allow CacheStateAllowlist) (TrialFingerprint, []SetupField, string) {
	canon, ambiguous := canonicalSetup(t.Setup)
	var nonCache, cacheState []SetupField
	for _, f := range canon {
		if allow.Allows(f.Name) {
			cacheState = append(cacheState, f)
		} else {
			nonCache = append(nonCache, f)
		}
	}
	return TrialFingerprint{
		Label:      t.Label,
		Fields:     len(canon),
		Setup:      setupFingerprint("fak-setup/1", canon),
		NonCache:   setupFingerprint("fak-setup-noncache/1", nonCache),
		CacheState: setupFingerprint("fak-setup-cachestate/1", cacheState),
	}, canon, ambiguous
}

// setupDeltas walks the UNION of both legs' canonical field names — the union, not the
// intersection, so a field only one leg declares is a delta rather than a blind spot.
func setupDeltas(cold, warm []SetupField, allow CacheStateAllowlist) []SetupDelta {
	coldBy := map[string]string{}
	warmBy := map[string]string{}
	var names []string
	seen := map[string]bool{}
	add := func(n string) {
		if !seen[n] {
			seen[n] = true
			names = append(names, n)
		}
	}
	for _, f := range cold {
		coldBy[f.Name] = f.Value
		add(f.Name)
	}
	for _, f := range warm {
		warmBy[f.Name] = f.Value
		add(f.Name)
	}
	sort.Strings(names)

	out := make([]SetupDelta, 0, len(names))
	for _, n := range names {
		cv, cok := coldBy[n]
		wv, wok := warmBy[n]
		if cok && wok && cv == wv {
			continue // held fixed: not a delta
		}
		d := SetupDelta{Field: n, Cold: setupAbsent, Warm: setupAbsent}
		if cok {
			d.Cold = setupValueDigest(n, cv)
		}
		if wok {
			d.Warm = setupValueDigest(n, wv)
		}
		switch {
		case allow.Allows(n):
			// A declared cache-state field may differ in value OR in presence: whether
			// a cache entry is reported at all is itself cache state.
			d.Reason, d.Allowed = ReasonCacheState, true
		case cok && !wok:
			d.Reason = ReasonFieldOmitted
		case !cok && wok:
			d.Reason = ReasonFieldAdded
		default:
			d.Reason = ReasonValueChanged
		}
		out = append(out, d)
	}
	return out
}

// CompareTrialPair is the PURE parity fold for one cold/warm pair. It fingerprints both
// legs, enumerates the exact differing fields, and folds them into one dominant verdict
// — withholding the speedup entirely on every refusal, so a comparison this gate turned
// away cannot be quoted as a cache gain.
//
// The precedence is fail-closed: a setup that could not be judged at all (unwitnessed,
// ambiguous) outranks everything; a field-SET disagreement outranks a value change,
// because two legs that do not even describe the same fields cannot be trusted over
// their intersection; and any undeclared delta outranks the "this was not a cold/warm
// pair" finding, since the undeclared delta is the headline defect.
func CompareTrialPair(pair TrialPair, allow CacheStateAllowlist) TrialPairReceipt {
	coldFP, coldCanon, coldAmb := fingerprintTrial(pair.Cold, allow)
	warmFP, warmCanon, warmAmb := fingerprintTrial(pair.Warm, allow)

	r := TrialPairReceipt{
		Pair:             pair.Name,
		Canonicalization: CanonicalizationRule,
		AllowlistBinding: allow.Binding(),
		Allowlist:        allow.Declared(),
		Cold:             coldFP,
		Warm:             warmFP,
		Deltas:           setupDeltas(coldCanon, warmCanon, allow),
		ColdMS:           pair.Cold.DurationMS,
		WarmMS:           pair.Warm.DurationMS,
	}
	r.NonCacheParity = coldFP.NonCache == warmFP.NonCache
	r.CacheStateChanged = coldFP.CacheState != warmFP.CacheState

	var omitted, added, changed []string
	for _, d := range r.Deltas {
		switch d.Reason {
		case ReasonFieldOmitted:
			omitted = append(omitted, d.Field)
		case ReasonFieldAdded:
			added = append(added, d.Field)
		case ReasonValueChanged:
			changed = append(changed, d.Field)
		}
	}

	switch {
	case coldFP.Fields == 0 || warmFP.Fields == 0:
		r.Verdict = VerdictSetupNotWitnessed
		r.Finding = "a trial declared no setup fields, so nothing was held fixed and nothing can be " +
			"proven equal; parity over an unwitnessed setup is not parity."
	case coldAmb != "" || warmAmb != "":
		r.Verdict = VerdictFieldAmbiguous
		r.Finding = fmt.Sprintf(
			"setup field %q is declared more than once after canonicalization, so which value governs "+
				"depends on input order; refused rather than resolved arbitrarily.",
			firstNonEmpty(coldAmb, warmAmb))
	case len(omitted) > 0:
		r.Verdict = ReasonFieldOmitted
		r.Finding = fmt.Sprintf(
			"the warm trial did not declare %s, which the cold trial did: an omitted field is UNPROVEN, "+
				"never equal, so the comparison is refused before any gain is reported.",
			strings.Join(omitted, ", "))
	case len(added) > 0:
		r.Verdict = ReasonFieldAdded
		r.Finding = fmt.Sprintf(
			"the warm trial declared %s, which the cold trial did not: the two legs are not the same "+
				"treatment, so the comparison is refused.",
			strings.Join(added, ", "))
	case len(changed) > 0:
		r.Verdict = ReasonValueChanged
		r.Finding = fmt.Sprintf(
			"undeclared non-cache setup changed between the legs (%s); the timing delta is not "+
				"attributable to cache reuse, so no gain is reported.",
			strings.Join(changed, ", "))
	case !r.CacheStateChanged:
		r.Verdict = VerdictNoCacheStateDelta
		r.Finding = "no declared cache-state field changed, so this is not a cold/warm pair: whatever the " +
			"timing delta measures, it is not a cache gain."
	default:
		r.Verdict = VerdictParityOK
		r.GainReported = true
		r.SpeedupPct = speedupPct(pair.Cold.DurationMS, pair.Warm.DurationMS)
		r.Finding = fmt.Sprintf(
			"every non-cache setup field is identical under the declared canonical form (non-cache "+
				"fingerprint %s on both legs) and only allowlisted cache state moved; the gain is attributable.",
			coldFP.NonCache)
	}
	return r
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// speedupPct is the warm-vs-cold speedup, rounded to 4 decimals so a committed witness
// re-derives byte-for-byte. A non-positive cold time cannot ground a percentage.
func speedupPct(coldMS, warmMS float64) float64 {
	if coldMS <= 0 {
		return 0
	}
	return math.Round((coldMS-warmMS)/coldMS*100*1e4) / 1e4
}

// SetupParityBlocks is the boolean gate a live harness calls before banking a cache
// gain, mirroring rsiloop's ForkParityBlocks: it returns (true, reason) iff this pair
// may NOT back a cache-gain claim, and (false, passNote) otherwise.
func SetupParityBlocks(pair TrialPair, allow CacheStateAllowlist) (bool, string) {
	r := CompareTrialPair(pair, allow)
	return !r.ParityHolds(), r.Verdict + ": " + r.Finding
}

// CompareTrialPairs folds a run of pairs: each is judged independently, refused pairs
// are EXCLUDED from the pooled gain, and the report says exactly which ones and why.
func CompareTrialPairs(pairs []TrialPair, allow CacheStateAllowlist) SetupParityReport {
	out := SetupParityReport{
		Schema:           "setup-parity.v1",
		Canonicalization: CanonicalizationRule,
		AllowlistBinding: allow.Binding(),
		Allowlist:        allow.Declared(),
		Pairs:            []TrialPairReceipt{},
		GainEligible:     []string{},
		Refused:          []RefusedPair{},
		Provenance: simulatedProvenance(
			"go test ./internal/bench -run TestSetupParity -count=1",
			"fak/internal/bench.CompareTrialPairs",
			"Pairs are labeled fixture setups (declared field names + values, never a live host readout): the "+
				"run witnesses canonicalization, the allowlist binding, the omitted-field refusal, and the typed "+
				"mismatch vocabulary. A real harness feeds the same receipt shape by reporting its own setup fields "+
				"for each leg.",
		),
	}

	var coldSum, warmSum float64
	for _, p := range pairs {
		r := CompareTrialPair(p, allow)
		out.Pairs = append(out.Pairs, r)
		if r.ParityHolds() {
			out.GainEligible = append(out.GainEligible, r.Pair)
			coldSum += r.ColdMS
			warmSum += r.WarmMS
			continue
		}
		out.Refused = append(out.Refused, RefusedPair{Pair: r.Pair, Reason: r.Verdict})
	}
	out.PooledSpeedupPct = speedupPct(coldSum, warmSum)

	if len(out.Refused) == 0 {
		out.Verdict = VerdictSetupParityClean
		out.Finding = fmt.Sprintf(
			"all %d pair(s) held every non-cache setup field fixed and moved only declared cache state; "+
				"pooled speedup %.4f%% is attributable to cache reuse.",
			len(out.Pairs), out.PooledSpeedupPct)
		return out
	}
	names := make([]string, 0, len(out.Refused))
	for _, rp := range out.Refused {
		names = append(names, rp.Pair+"="+rp.Reason)
	}
	out.Verdict = VerdictSetupParityFlagged
	out.Finding = fmt.Sprintf(
		"%d of %d pair(s) were REFUSED and excluded from the gain (%s); the pooled %.4f%% is computed over "+
			"the %d admitted pair(s) only.",
		len(out.Refused), len(out.Pairs), strings.Join(names, ", "), out.PooledSpeedupPct, len(out.GainEligible))
	return out
}

// ---------------------------------------------------------------------------
// The committed fixtures.
// ---------------------------------------------------------------------------

// DefaultCacheStateAllowlist is the shipped allowlist: the three fields that describe
// cache state and nothing else. Everything a harness reports outside this list — the
// input, the execution policy, the host — must be held fixed across the pair.
func DefaultCacheStateAllowlist() CacheStateAllowlist {
	return NewCacheStateAllowlist(
		"cache.state",
		"cache.entries_present",
		"cache.prefix_tokens_reused",
	)
}

// baseSetup is the non-cache half every fixture leg shares: the input, the execution
// policy, and the host. A pair that changes any of these is not measuring cache reuse.
func baseSetup() []SetupField {
	return []SetupField{
		{Name: "input.workload_hash", Value: "wh-7f2c"},
		{Name: "input.trace_slice", Value: "slice-a"},
		{Name: "policy.model", Value: "reference-small"},
		{Name: "policy.max_tokens", Value: "4096"},
		{Name: "exec.threads", Value: "8"},
		{Name: "host.arch", Value: "reference-x64"},
	}
}

// withSetup returns baseSetup with the given fields appended, and any listed name
// dropped — the fixture builder for a leg that adds cache state, overrides one setup
// field, or omits one entirely.
func withSetup(drop []string, extra ...SetupField) []SetupField {
	dropped := map[string]bool{}
	for _, d := range drop {
		dropped[d] = true
	}
	pending := map[string]string{}
	for _, e := range extra {
		pending[e.Name] = e.Value
	}
	base := baseSetup()
	out := make([]SetupField, 0, len(base)+len(extra))
	for _, f := range base {
		if dropped[f.Name] {
			continue
		}
		if v, ok := pending[f.Name]; ok {
			f.Value = v
			delete(pending, f.Name)
		}
		out = append(out, f)
	}
	for _, e := range extra {
		if _, still := pending[e.Name]; still {
			out = append(out, e)
			delete(pending, e.Name)
		}
	}
	return out
}

// coldLeg is the reference cold leg: the shared non-cache setup plus a cold cache.
func coldLeg(drop []string, extra ...SetupField) Trial {
	return Trial{
		Label: "cold",
		Setup: withSetup(drop, append([]SetupField{
			{Name: "cache.state", Value: "cold"},
			{Name: "cache.entries_present", Value: "0"},
			{Name: "cache.prefix_tokens_reused", Value: "0"},
		}, extra...)...),
		DurationMS: 1000,
	}
}

// warmLeg is the reference warm leg: the same non-cache setup with a warm cache.
func warmLeg(ms float64, drop []string, extra ...SetupField) Trial {
	return Trial{
		Label: "warm",
		Setup: withSetup(drop, append([]SetupField{
			{Name: "cache.state", Value: "warm"},
			{Name: "cache.entries_present", Value: "1"},
			{Name: "cache.prefix_tokens_reused", Value: "3180"},
		}, extra...)...),
		DurationMS: ms,
	}
}

// DefaultSetupParityPairs is the committed fixture set. It carries the known-positive
// case and three distinct ways a warm-path gain lies, each landing on its own typed
// refusal:
//
//   - honest-warm-reuse      — only cache state moved; parity PASSES and the gain stands.
//   - retuned-thread-count   — the warm leg also dropped exec.threads 8 -> 1, so its far
//     larger "gain" is a different treatment, not reuse. REFUSED.
//   - dropped-policy-field   — the warm leg stopped reporting policy.max_tokens: the
//     field is UNPROVEN, not equal. REFUSED.
//   - cache-never-moved      — nothing on the allowlist changed, so the pair is not a
//     cold/warm pair at all and its delta is not a cache gain. REFUSED.
//
// The thread-count leg is the sharp one: it shows the largest speedup of the four, which
// is exactly why a stopwatch-only comparison would have promoted it.
func DefaultSetupParityPairs() []TrialPair {
	return []TrialPair{
		{
			Name: "honest-warm-reuse",
			Cold: coldLeg(nil),
			Warm: warmLeg(640, nil),
		},
		{
			Name: "retuned-thread-count",
			Cold: coldLeg(nil),
			Warm: warmLeg(210, nil, SetupField{Name: "exec.threads", Value: "1"}),
		},
		{
			Name: "dropped-policy-field",
			Cold: coldLeg(nil),
			Warm: warmLeg(655, []string{"policy.max_tokens"}),
		},
		{
			Name: "cache-never-moved",
			Cold: coldLeg(nil),
			Warm: Trial{Label: "warm", Setup: coldLeg(nil).Setup, DurationMS: 968},
		},
	}
}

// DefaultSetupParityReport is the committed witness: one deterministic run over the
// fixture set, showing the canonical fingerprints, the allowed cache-state deltas, the
// three typed refusals, and a pooled gain computed over the admitted pair alone.
func DefaultSetupParityReport() SetupParityReport {
	return CompareTrialPairs(DefaultSetupParityPairs(), DefaultCacheStateAllowlist())
}
