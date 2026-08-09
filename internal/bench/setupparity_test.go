package bench

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestSetupParity is the re-runnable witness for issue #5704. One deterministic run
// over the committed fixture set demonstrates every half the proof artifacts name:
//
//   - the KNOWN-POSITIVE case — only cache state moves, parity passes, and the gain
//     is reported;
//   - the ADVERSARIAL case — the warm leg also retuned exec.threads, so its LARGER
//     apparent speedup is refused rather than banked; and
//   - the pooled gain is computed over the admitted pair alone, so a refused pair
//     cannot leak into the run-level number.
//
// The run is pinned to a committed golden (the scrubbed, re-derivable receipt).
func TestSetupParity(t *testing.T) {
	got := DefaultSetupParityReport()

	if got.Provenance.Kind != ProvenanceSimulated {
		t.Fatalf("provenance = %q, want %q", got.Provenance.Kind, ProvenanceSimulated)
	}
	if got.Verdict != VerdictSetupParityFlagged {
		t.Fatalf("verdict = %q, want %q", got.Verdict, VerdictSetupParityFlagged)
	}
	if len(got.Pairs) != 4 {
		t.Fatalf("pairs = %d, want 4", len(got.Pairs))
	}

	// --- the known-positive fixture ---
	ok := parityPair(t, got, "honest-warm-reuse")
	if !ok.ParityHolds() || ok.Verdict != VerdictParityOK {
		t.Errorf("known-positive pair = %q, want %q", ok.Verdict, VerdictParityOK)
	}
	if !ok.NonCacheParity || !ok.CacheStateChanged {
		t.Errorf("known-positive parity/cache-moved = %v/%v, want true/true", ok.NonCacheParity, ok.CacheStateChanged)
	}
	if !ok.GainReported || ok.SpeedupPct <= 0 {
		t.Errorf("known-positive gain_reported=%v speedup=%v, want a reported positive gain", ok.GainReported, ok.SpeedupPct)
	}
	// Every delta it DOES carry must be an allowlisted cache-state one.
	for _, d := range ok.Deltas {
		if !d.Allowed || d.Reason != ReasonCacheState {
			t.Errorf("known-positive carries a non-cache delta %q (%s)", d.Field, d.Reason)
		}
	}

	// --- the adversarial fixture: the BIGGEST speedup is the one that is refused ---
	bad := parityPair(t, got, "retuned-thread-count")
	if bad.ParityHolds() || bad.Verdict != ReasonValueChanged {
		t.Errorf("adversarial pair = %q, want %q", bad.Verdict, ReasonValueChanged)
	}
	if bad.GainReported || bad.SpeedupPct != 0 {
		t.Errorf("adversarial pair reported a gain (%v%%); a refused comparison must withhold the speedup", bad.SpeedupPct)
	}
	if bad.NonCacheParity {
		t.Errorf("adversarial pair claims non-cache parity while exec.threads changed")
	}
	if rawSpeedup := (bad.ColdMS - bad.WarmMS) / bad.ColdMS; rawSpeedup <= (ok.ColdMS-ok.WarmMS)/ok.ColdMS {
		t.Errorf("fixture is toothless: the refused pair must LOOK better than the honest one (%v vs %v)",
			rawSpeedup, (ok.ColdMS-ok.WarmMS)/ok.ColdMS)
	}
	if d := parityDelta(t, bad, "exec.threads"); d.Allowed || d.Reason != ReasonValueChanged {
		t.Errorf("exec.threads delta = %s allowed=%v, want %s not-allowed", d.Reason, d.Allowed, ReasonValueChanged)
	}

	// --- the pooled gain excludes every refused pair ---
	if len(got.GainEligible) != 1 || got.GainEligible[0] != "honest-warm-reuse" {
		t.Fatalf("gain_eligible = %v, want only the honest pair", got.GainEligible)
	}
	if len(got.Refused) != 3 {
		t.Errorf("refused = %d, want 3", len(got.Refused))
	}
	if want := ok.SpeedupPct; got.PooledSpeedupPct != want {
		t.Errorf("pooled speedup = %v, want %v (the admitted pair alone)", got.PooledSpeedupPct, want)
	}

	// The receipt binds to the allowlist it was judged under, on every pair.
	binding := DefaultCacheStateAllowlist().Binding()
	for _, p := range got.Pairs {
		if p.AllowlistBinding != binding {
			t.Errorf("pair %q binding = %q, want %q", p.Pair, p.AllowlistBinding, binding)
		}
		if p.Canonicalization != CanonicalizationRule {
			t.Errorf("pair %q does not publish the canonicalization rule", p.Pair)
		}
	}

	// The artifact is publishable: no raw setup VALUE appears anywhere in it.
	gotJSON, err := got.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	for _, secret := range []string{"reference-small", "wh-7f2c", "reference-x64", "slice-a"} {
		if bytes.Contains(gotJSON, []byte(secret)) {
			t.Errorf("receipt leaks the raw setup value %q; values must appear only as digests", secret)
		}
	}
	// ...but the field NAMES do, because they are the audit surface.
	if !bytes.Contains(gotJSON, []byte("exec.threads")) {
		t.Errorf("receipt does not name the field that broke parity")
	}

	golden := filepath.Join("testdata", "setup_parity.json")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(golden, append(gotJSON, '\n'), 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		t.Logf("updated golden %s", golden)
		return
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if !bytes.Equal(bytes.TrimRight(want, "\n"), bytes.TrimRight(gotJSON, "\n")) {
		t.Errorf("receipt drifted from golden %s; re-run with UPDATE_GOLDEN=1 if intended", golden)
	}
}

// TestSetupParityDeterministic pins that the report is re-derivable byte-for-byte: no
// clock, no randomness, and no map iteration reaching the output.
func TestSetupParityDeterministic(t *testing.T) {
	a, err := DefaultSetupParityReport().JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	for i := 0; i < 8; i++ {
		b, err := DefaultSetupParityReport().JSON()
		if err != nil {
			t.Fatalf("JSON: %v", err)
		}
		if !bytes.Equal(a, b) {
			t.Fatalf("report is not deterministic (run %d differs)", i)
		}
	}
}

// TestSetupParityCanonicalization covers the canonical form the whole gate rests on.
// Two setups that mean the same thing must fingerprint alike no matter how they were
// serialized; two setups that mean DIFFERENT things must never collide.
func TestSetupParityCanonicalization(t *testing.T) {
	allow := NewCacheStateAllowlist("cache.state")

	// Field ORDER and outer whitespace are serialization artifacts, not setup.
	scrambled := Trial{Label: "warm", Setup: []SetupField{
		{Name: "  cache.state ", Value: " warm"},
		{Name: "exec.threads", Value: "8 "},
		{Name: " input.workload_hash", Value: "wh "},
	}, DurationMS: 500}
	tidy := Trial{Label: "warm", Setup: []SetupField{
		{Name: "input.workload_hash", Value: "wh"},
		{Name: "cache.state", Value: "warm"},
		{Name: "exec.threads", Value: "8"},
	}, DurationMS: 500}
	a, _, _ := fingerprintTrial(scrambled, allow)
	b, _, _ := fingerprintTrial(tidy, allow)
	if a.Setup != b.Setup || a.NonCache != b.NonCache || a.CacheState != b.CacheState {
		t.Errorf("canonicalization did not absorb field order + outer whitespace:\n %+v\n %+v", a, b)
	}

	// Delimiter injection: a field NAMED "a=b" must not fingerprint like a field named
	// "a" whose VALUE is "b=c". This is what the length prefix buys.
	inject := Trial{Setup: []SetupField{{Name: "a=b", Value: "c"}}}
	plain := Trial{Setup: []SetupField{{Name: "a", Value: "b=c"}}}
	if x, _, _ := fingerprintTrial(inject, allow); x.Setup == mustFingerprint(t, plain, allow) {
		t.Errorf("two structurally different setups collided: the fingerprint is not injection-proof")
	}

	// A setup that names one field twice is ambiguous, and is REFUSED rather than
	// resolved by whichever entry sorted first.
	dup := TrialPair{
		Name: "dup",
		Cold: Trial{Label: "cold", Setup: []SetupField{
			{Name: "exec.threads", Value: "8"},
			{Name: "exec.threads ", Value: "1"},
			{Name: "cache.state", Value: "cold"},
		}, DurationMS: 100},
		Warm: Trial{Label: "warm", Setup: []SetupField{
			{Name: "exec.threads", Value: "8"},
			{Name: "cache.state", Value: "warm"},
		}, DurationMS: 50},
	}
	r := CompareTrialPair(dup, allow)
	if r.Verdict != VerdictFieldAmbiguous || r.GainReported {
		t.Errorf("duplicate field verdict = %q gain=%v, want %q with no gain", r.Verdict, r.GainReported, VerdictFieldAmbiguous)
	}
	if !strings.Contains(r.Finding, "exec.threads") {
		t.Errorf("ambiguity finding does not name the offending field: %q", r.Finding)
	}

	// A value digest is salted by its field: the same value under two field names must
	// not produce the same digest.
	if setupValueDigest("exec.threads", "8") == setupValueDigest("policy.max_tokens", "8") {
		t.Errorf("value digests are not field-salted")
	}
}

// TestSetupParityOmittedField is the honesty case the contract turns on: a non-cache
// field the warm leg stops reporting is UNPROVEN, never equal. An absent field must
// also stay distinguishable from a field present with an empty value.
func TestSetupParityOmittedField(t *testing.T) {
	allow := DefaultCacheStateAllowlist()

	omitted := parityPair(t, DefaultSetupParityReport(), "dropped-policy-field")
	if omitted.Verdict != ReasonFieldOmitted || omitted.GainReported {
		t.Errorf("omitted-field verdict = %q gain=%v, want %q with no gain", omitted.Verdict, omitted.GainReported, ReasonFieldOmitted)
	}
	d := parityDelta(t, omitted, "policy.max_tokens")
	if d.Reason != ReasonFieldOmitted || d.Allowed {
		t.Errorf("policy.max_tokens delta = %s allowed=%v, want %s not-allowed", d.Reason, d.Allowed, ReasonFieldOmitted)
	}
	if d.Warm != setupAbsent {
		t.Errorf("omitted side digest = %q, want the %q sentinel", d.Warm, setupAbsent)
	}
	if d.Cold == setupAbsent {
		t.Errorf("present side reads as absent")
	}

	// ABSENT is not the same as PRESENT-BUT-EMPTY. If it were, a harness could silence a
	// mismatch by reporting the field as "".
	emptyWarm := TrialPair{
		Name: "empty-vs-absent",
		Cold: coldLeg(nil),
		Warm: warmLeg(600, []string{"policy.max_tokens"}, SetupField{Name: "policy.max_tokens", Value: ""}),
	}
	er := CompareTrialPair(emptyWarm, allow)
	if er.Verdict != ReasonValueChanged {
		t.Errorf("present-but-empty verdict = %q, want %q (a value change, not an omission)", er.Verdict, ReasonValueChanged)
	}
	ed := parityDelta(t, er, "policy.max_tokens")
	if ed.Warm == setupAbsent {
		t.Errorf("a field present with an empty value was recorded as ABSENT; the two must stay distinct")
	}

	// A field only the WARM leg declares is its own typed reason.
	added := CompareTrialPair(TrialPair{
		Name: "added",
		Cold: coldLeg([]string{"exec.threads"}),
		Warm: warmLeg(600, nil),
	}, allow)
	if added.Verdict != ReasonFieldAdded {
		t.Errorf("added-field verdict = %q, want %q", added.Verdict, ReasonFieldAdded)
	}

	// An ALLOWLISTED field may differ in PRESENCE too — whether a cache entry is
	// reported at all is itself cache state, so this stays a pass.
	presence := CompareTrialPair(TrialPair{
		Name: "cache-presence",
		Cold: coldLeg([]string{}, SetupField{Name: "cache.prefix_tokens_reused", Value: "0"}),
		Warm: Trial{Label: "warm", Setup: withSetup(nil,
			SetupField{Name: "cache.state", Value: "warm"},
			SetupField{Name: "cache.entries_present", Value: "1"},
		), DurationMS: 600},
	}, allow)
	if presence.Verdict != VerdictParityOK {
		t.Errorf("allowlisted presence change verdict = %q, want %q: %s", presence.Verdict, VerdictParityOK, presence.Finding)
	}
}

// TestSetupParityTypedMismatchReasons pins the CLOSED reason vocabulary and the
// fail-closed precedence between reasons. A caller switching on these strings must
// never meet one this test does not name.
func TestSetupParityTypedMismatchReasons(t *testing.T) {
	allow := DefaultCacheStateAllowlist()
	known := map[string]bool{
		ReasonCacheState: true, ReasonValueChanged: true,
		ReasonFieldOmitted: true, ReasonFieldAdded: true,
	}
	knownVerdicts := map[string]bool{
		VerdictParityOK: true, VerdictSetupNotWitnessed: true, VerdictFieldAmbiguous: true,
		VerdictNoCacheStateDelta: true, ReasonValueChanged: true,
		ReasonFieldOmitted: true, ReasonFieldAdded: true,
	}

	cases := []struct {
		name string
		pair TrialPair
		want string
	}{
		{
			name: "no cache state moved is not a cold/warm pair",
			pair: TrialPair{Name: "p", Cold: coldLeg(nil), Warm: Trial{Label: "warm", Setup: coldLeg(nil).Setup, DurationMS: 10}},
			want: VerdictNoCacheStateDelta,
		},
		{
			name: "an unwitnessed setup cannot be judged",
			pair: TrialPair{Name: "p", Cold: Trial{Label: "cold", DurationMS: 100}, Warm: warmLeg(50, nil)},
			want: VerdictSetupNotWitnessed,
		},
		{
			name: "a field-set disagreement outranks a value change",
			pair: TrialPair{
				Name: "p",
				Cold: coldLeg(nil),
				Warm: warmLeg(50, []string{"policy.max_tokens"}, SetupField{Name: "exec.threads", Value: "1"}),
			},
			want: ReasonFieldOmitted,
		},
		{
			name: "an undeclared delta outranks the no-cache-delta finding",
			pair: TrialPair{
				Name: "p",
				Cold: coldLeg(nil),
				Warm: Trial{Label: "warm", Setup: withSetup(nil,
					SetupField{Name: "cache.state", Value: "cold"},
					SetupField{Name: "cache.entries_present", Value: "0"},
					SetupField{Name: "cache.prefix_tokens_reused", Value: "0"},
					SetupField{Name: "exec.threads", Value: "1"},
				), DurationMS: 50},
			},
			want: ReasonValueChanged,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := CompareTrialPair(tc.pair, allow)
			if r.Verdict != tc.want {
				t.Errorf("verdict = %q, want %q (%s)", r.Verdict, tc.want, r.Finding)
			}
			if !knownVerdicts[r.Verdict] {
				t.Errorf("verdict %q is outside the closed vocabulary", r.Verdict)
			}
			if r.GainReported || r.SpeedupPct != 0 {
				t.Errorf("a refused comparison reported a %v%% gain", r.SpeedupPct)
			}
			if r.Finding == "" {
				t.Errorf("refusal carries no finding")
			}
			for _, d := range r.Deltas {
				if !known[d.Reason] {
					t.Errorf("delta on %q carries the untyped reason %q", d.Field, d.Reason)
				}
				if d.Allowed != (d.Reason == ReasonCacheState) {
					t.Errorf("delta %q allowed=%v under reason %s", d.Field, d.Allowed, d.Reason)
				}
			}
			// The gate verb agrees with the receipt, and quotes the typed reason.
			blocked, why := SetupParityBlocks(tc.pair, allow)
			if !blocked || !strings.HasPrefix(why, r.Verdict) {
				t.Errorf("SetupParityBlocks = %v/%q, want blocked with the %q reason", blocked, why, r.Verdict)
			}
		})
	}

	// The happy path agrees too.
	if blocked, _ := SetupParityBlocks(DefaultSetupParityPairs()[0], allow); blocked {
		t.Errorf("SetupParityBlocks refused the known-positive pair")
	}
}

// TestSetupParityNonCacheFingerprintIsTheInvariant cross-checks the two independent
// derivations of the same fact: the published non-cache fingerprint equality and the
// per-field delta walk must never disagree. If they can, one of them is decorative.
func TestSetupParityNonCacheFingerprintIsTheInvariant(t *testing.T) {
	allow := DefaultCacheStateAllowlist()
	for _, p := range append(DefaultSetupParityPairs(), TrialPair{
		Name: "widened",
		Cold: coldLeg(nil),
		Warm: warmLeg(600, nil, SetupField{Name: "policy.model", Value: "reference-large"}),
	}) {
		r := CompareTrialPair(p, allow)
		undeclared := false
		for _, d := range r.Deltas {
			if !d.Allowed {
				undeclared = true
			}
		}
		if r.NonCacheParity == undeclared {
			t.Errorf("pair %q: non_cache_parity=%v but undeclared-delta=%v — the fingerprint and the "+
				"delta walk disagree", p.Name, r.NonCacheParity, undeclared)
		}
	}

	// Widening the allowlist is VISIBLE: the same pair judged under a wider policy
	// carries a different binding, so a laundered pass cannot pass as a shipped one.
	wide := NewCacheStateAllowlist(append(DefaultCacheStateAllowlist().Declared(), "exec.threads")...)
	if wide.Binding() == allow.Binding() {
		t.Fatalf("a widened allowlist produced the shipped binding")
	}
	laundered := CompareTrialPair(DefaultSetupParityPairs()[1], wide)
	if !laundered.ParityHolds() {
		t.Errorf("widening the allowlist did not admit the pair; the allowlist is not the control")
	}
	if laundered.AllowlistBinding == allow.Binding() {
		t.Errorf("the laundered pass carries the shipped allowlist binding")
	}

	// Allowlist declaration is order- and duplicate-insensitive, so the binding is a
	// property of the POLICY, not of how it was typed.
	if NewCacheStateAllowlist("b", "a", "a", " b ").Binding() != NewCacheStateAllowlist("a", "b").Binding() {
		t.Errorf("allowlist binding depends on argument order or duplicates")
	}
}

func parityPair(t *testing.T, r SetupParityReport, name string) TrialPairReceipt {
	t.Helper()
	for _, p := range r.Pairs {
		if p.Pair == name {
			return p
		}
	}
	t.Fatalf("report has no pair %q", name)
	return TrialPairReceipt{}
}

func parityDelta(t *testing.T, r TrialPairReceipt, field string) SetupDelta {
	t.Helper()
	for _, d := range r.Deltas {
		if d.Field == field {
			return d
		}
	}
	t.Fatalf("pair %q has no delta on %q: %+v", r.Pair, field, r.Deltas)
	return SetupDelta{}
}

func mustFingerprint(t *testing.T, tr Trial, allow CacheStateAllowlist) string {
	t.Helper()
	fp, _, _ := fingerprintTrial(tr, allow)
	return fp.Setup
}
