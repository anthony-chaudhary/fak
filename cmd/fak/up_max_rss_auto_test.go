package main

import (
	"errors"
	"strings"
	"testing"
)

const (
	rssTestKiB = uint64(1) << 10
	rssTestMiB = uint64(1) << 20
	rssTestGiB = uint64(1) << 30
	rssTestTiB = uint64(1) << 40
)

func TestParseMaxRSSSpec(t *testing.T) {
	cases := []struct {
		in   string
		want maxRSSSpec
	}{
		{"", maxRSSSpec{}},
		{"   ", maxRSSSpec{}},
		{"0", maxRSSSpec{}},
		{" 0 ", maxRSSSpec{}},
		{"auto", maxRSSSpec{Auto: true}},
		{"AUTO", maxRSSSpec{Auto: true}},
		{"Auto", maxRSSSpec{Auto: true}},
		{" auto ", maxRSSSpec{Auto: true}},
		{"1", maxRSSSpec{Bytes: 1}},
		{"25769803776", maxRSSSpec{Bytes: 25769803776}},
		{" 25769803776 ", maxRSSSpec{Bytes: 25769803776}},
		{"1K", maxRSSSpec{Bytes: rssTestKiB}},
		{"1KiB", maxRSSSpec{Bytes: rssTestKiB}},
		{"4M", maxRSSSpec{Bytes: 4 * rssTestMiB}},
		{"512MiB", maxRSSSpec{Bytes: 512 * rssTestMiB}},
		{"512 MiB", maxRSSSpec{Bytes: 512 * rssTestMiB}},
		{"30G", maxRSSSpec{Bytes: 30 * rssTestGiB}},
		{"30GiB", maxRSSSpec{Bytes: 30 * rssTestGiB}},
		{"30 GiB", maxRSSSpec{Bytes: 30 * rssTestGiB}},
		{"30gib", maxRSSSpec{Bytes: 30 * rssTestGiB}},
		{"30g", maxRSSSpec{Bytes: 30 * rssTestGiB}},
		{"2T", maxRSSSpec{Bytes: 2 * rssTestTiB}},
		{"2TiB", maxRSSSpec{Bytes: 2 * rssTestTiB}},
	}
	for _, tc := range cases {
		t.Run(tc.in, func(t *testing.T) {
			got, err := parseMaxRSSSpec(tc.in)
			if err != nil {
				t.Fatalf("parseMaxRSSSpec(%q) error = %v, want nil", tc.in, err)
			}
			if got != tc.want {
				t.Fatalf("parseMaxRSSSpec(%q) = %+v, want %+v", tc.in, got, tc.want)
			}
		})
	}
}

func TestParseMaxRSSSpecRejectsMalformed(t *testing.T) {
	bad := []string{
		"-1",
		"-30GiB",
		"abc",
		"30XB",
		"30 XB",
		"GiB",
		"autox",
		"18446744073709551616",  // 2^64: overflows uint64
		"17179869184GiB",        // 2^34 * 2^30 = 2^64: overflows after the suffix multiply
		"99999999999999999999T", // overflows before and after the multiply
	}
	for _, in := range bad {
		t.Run(in, func(t *testing.T) {
			got, err := parseMaxRSSSpec(in)
			if err == nil {
				t.Fatalf("parseMaxRSSSpec(%q) = %+v, want error", in, got)
			}
		})
	}
}

func TestDeriveMaxRSSCeiling(t *testing.T) {
	cases := []struct {
		name         string
		resident, kv uint64
		pct          float64
		want         uint64
	}{
		{"simple_10pct", 100, 0, 10, 110},
		{"kv_counts_toward_base", 60, 40, 10, 110},
		{"fifty_pct", 200, 0, 50, 300},
		{"rounds_up", 101, 0, 10, 112}, // 111.1 -> 112
		{"resident_zero_is_unmeasured", 0, 0, 15, 0},
		{"resident_zero_ignores_kv", 0, rssTestGiB, 15, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := deriveMaxRSSCeiling(tc.resident, tc.kv, tc.pct); got != tc.want {
				t.Fatalf("deriveMaxRSSCeiling(%d, %d, %v) = %d, want %d", tc.resident, tc.kv, tc.pct, got, tc.want)
			}
		})
	}
}

func TestDeriveMaxRSSCeilingNonPositiveHeadroomUsesDefault(t *testing.T) {
	if defaultMaxRSSHeadroomPct != 15.0 {
		t.Fatalf("defaultMaxRSSHeadroomPct = %v, want 15", defaultMaxRSSHeadroomPct)
	}
	const resident, kv = uint64(1000), uint64(0)
	want := deriveMaxRSSCeiling(resident, kv, defaultMaxRSSHeadroomPct)
	if want < 1150 || want > 1151 {
		t.Fatalf("deriveMaxRSSCeiling(1000, 0, 15) = %d, want ~1150", want)
	}
	for _, pct := range []float64{0, -5} {
		if got := deriveMaxRSSCeiling(resident, kv, pct); got != want {
			t.Fatalf("deriveMaxRSSCeiling(1000, 0, %v) = %d, want default-headroom %d", pct, got, want)
		}
	}
}

func TestDeriveMaxRSSCeilingStrictlyAboveFootprint(t *testing.T) {
	cases := []struct {
		resident, kv uint64
		pct          float64
	}{
		{1, 0, 15},
		{1, 0, 0.0001},
		{7, 3, 1},
		{29066260070, rssTestGiB, 15},
		{29066260070, 0, 0.000001},
		{64 * rssTestGiB, 4 * rssTestGiB, 0},
	}
	for _, tc := range cases {
		got := deriveMaxRSSCeiling(tc.resident, tc.kv, tc.pct)
		if got <= tc.resident+tc.kv {
			t.Fatalf("deriveMaxRSSCeiling(%d, %d, %v) = %d, want > %d", tc.resident, tc.kv, tc.pct, got, tc.resident+tc.kv)
		}
	}
}

func TestResolveMaxRSSCeiling(t *testing.T) {
	const (
		resident = 10 * rssTestGiB
		kv       = rssTestGiB
	)
	derived := deriveMaxRSSCeiling(resident, kv, defaultMaxRSSHeadroomPct)
	cases := []struct {
		name         string
		spec         maxRSSSpec
		resident, kv uint64
		strict       bool
		wantCeiling  uint64
		wantSource   string
		wantReq      uint64
	}{
		{"disabled", maxRSSSpec{}, resident, kv, false, 0, "disabled", 0},
		{"disabled_strict", maxRSSSpec{}, resident, kv, true, 0, "disabled", 0},
		{"auto_unmeasured", maxRSSSpec{Auto: true}, 0, kv, false, 0, "auto_unmeasured", 0},
		{"auto_unmeasured_strict", maxRSSSpec{Auto: true}, 0, kv, true, 0, "auto_unmeasured", 0},
		{"auto", maxRSSSpec{Auto: true}, resident, kv, false, derived, "auto", 0},
		{"auto_strict", maxRSSSpec{Auto: true}, resident, kv, true, derived, "auto", 0},
		{"explicit_fits", maxRSSSpec{Bytes: 32 * rssTestGiB}, resident, kv, false, 32 * rssTestGiB, "explicit", 32 * rssTestGiB},
		{"explicit_fits_strict", maxRSSSpec{Bytes: 32 * rssTestGiB}, resident, kv, true, 32 * rssTestGiB, "explicit", 32 * rssTestGiB},
		{"explicit_one_byte_over", maxRSSSpec{Bytes: resident + kv + 1}, resident, kv, true, resident + kv + 1, "explicit", resident + kv + 1},
		{"explicit_unmeasured_keeps_ceiling", maxRSSSpec{Bytes: rssTestGiB}, 0, kv, true, rssTestGiB, "explicit", rssTestGiB},
		{"explicit_stale_raised", maxRSSSpec{Bytes: 8 * rssTestGiB}, resident, kv, false, derived, "auto_raised", 8 * rssTestGiB},
		{"explicit_equal_is_stale_raised", maxRSSSpec{Bytes: resident + kv}, resident, kv, false, derived, "auto_raised", resident + kv},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := resolveMaxRSSCeiling(tc.spec, tc.resident, tc.kv, defaultMaxRSSHeadroomPct, tc.strict)
			if err != nil {
				t.Fatalf("resolveMaxRSSCeiling error = %v, want nil", err)
			}
			if got.Ceiling != tc.wantCeiling {
				t.Fatalf("Ceiling = %d, want %d", got.Ceiling, tc.wantCeiling)
			}
			if got.Source != tc.wantSource {
				t.Fatalf("Source = %q, want %q", got.Source, tc.wantSource)
			}
			if got.Requested != tc.wantReq {
				t.Fatalf("Requested = %d, want %d", got.Requested, tc.wantReq)
			}
			if got.Source == "auto_raised" && got.Ceiling <= tc.resident+tc.kv {
				t.Fatalf("auto_raised Ceiling %d not above footprint %d", got.Ceiling, tc.resident+tc.kv)
			}
			// A resolved non-zero ceiling must pass the existing strict boot check.
			if got.Ceiling != 0 {
				if err := checkMaxRSSAgainstResident(got.Ceiling, tc.resident, tc.kv); err != nil {
					t.Fatalf("resolved ceiling %d fails checkMaxRSSAgainstResident: %v", got.Ceiling, err)
				}
			}
		})
	}
}

func TestResolveMaxRSSCeilingHonorsHeadroomPct(t *testing.T) {
	got, err := resolveMaxRSSCeiling(maxRSSSpec{Auto: true}, 100, 0, 10, false)
	if err != nil {
		t.Fatalf("resolve auto error = %v", err)
	}
	if got.Ceiling != 110 || got.Source != "auto" {
		t.Fatalf("resolve auto = %+v, want Ceiling 110 Source auto", got)
	}
	got, err = resolveMaxRSSCeiling(maxRSSSpec{Bytes: 50}, 100, 0, 10, false)
	if err != nil {
		t.Fatalf("resolve stale explicit error = %v", err)
	}
	if got.Ceiling != 110 || got.Source != "auto_raised" || got.Requested != 50 {
		t.Fatalf("resolve stale explicit = %+v, want Ceiling 110 Source auto_raised Requested 50", got)
	}
}

func TestResolveMaxRSSCeilingStrictRefusesStaleExplicit(t *testing.T) {
	const resident, kv, ceiling = 100, 10, 100
	_, err := resolveMaxRSSCeiling(maxRSSSpec{Bytes: ceiling}, resident, kv, 10, true)
	if err == nil {
		t.Fatal("strict resolve of a stale explicit ceiling returned nil error")
	}
	if !errors.Is(err, ErrMaxRSSBelowResident) {
		t.Fatalf("err = %v, want errors.Is ErrMaxRSSBelowResident", err)
	}
	var typed *MaxRSSBelowResidentError
	if !errors.As(err, &typed) {
		t.Fatalf("errors.As(*MaxRSSBelowResidentError) failed on %v", err)
	}
	if want := deriveMaxRSSCeiling(resident, kv, 10); typed.Suggested != want {
		t.Fatalf("Suggested = %d, want %d", typed.Suggested, want)
	}
}

// TestMaxRSSStaleMacPlistCeilingDoesNotRefuseBoot is the regression for the Mac launchd
// plist that hardcodes --max-rss 25769803776 (24 GiB) while the idle phys_footprint is
// ~27.07 GiB. On the parent commit (d32d2e89b9) this configuration refused boot with
// exit 78 (ErrMaxRSSBelowResident) and launchd crash-looped it; the non-strict default
// must now raise the stale ceiling to a derived one instead.
func TestMaxRSSStaleMacPlistCeilingDoesNotRefuseBoot(t *testing.T) {
	const (
		plistMaxRSS = uint64(25769803776)
		resident    = uint64(29066260070)
		sessionKV   = rssTestGiB
	)
	spec, err := parseMaxRSSSpec("25769803776")
	if err != nil {
		t.Fatalf("parseMaxRSSSpec(plist value) error = %v", err)
	}
	if spec.Auto || spec.Bytes != plistMaxRSS {
		t.Fatalf("parseMaxRSSSpec(plist value) = %+v, want Bytes %d", spec, plistMaxRSS)
	}
	// Sanity: the old strict boot check refuses these numbers.
	if !errors.Is(checkMaxRSSAgainstResident(plistMaxRSS, resident, sessionKV), ErrMaxRSSBelowResident) {
		t.Fatal("precondition: checkMaxRSSAgainstResident should refuse the stale plist ceiling")
	}

	got, err := resolveMaxRSSCeiling(spec, resident, sessionKV, defaultMaxRSSHeadroomPct, false)
	if err != nil {
		t.Fatalf("resolveMaxRSSCeiling(stale Mac plist) error = %v, want nil (boot must not be refused)", err)
	}
	if got.Source != "auto_raised" {
		t.Fatalf("Source = %q, want auto_raised", got.Source)
	}
	if got.Ceiling <= resident+sessionKV {
		t.Fatalf("Ceiling = %d, want > resident+sessionKV %d", got.Ceiling, resident+sessionKV)
	}
	if got.Ceiling != deriveMaxRSSCeiling(resident, sessionKV, defaultMaxRSSHeadroomPct) {
		t.Fatalf("Ceiling = %d, want derived %d", got.Ceiling, deriveMaxRSSCeiling(resident, sessionKV, defaultMaxRSSHeadroomPct))
	}
	if got.Requested != plistMaxRSS {
		t.Fatalf("Requested = %d, want %d", got.Requested, plistMaxRSS)
	}
	if got.Resident != resident || got.SessionKV != sessionKV {
		t.Fatalf("Resident/SessionKV = %d/%d, want %d/%d", got.Resident, got.SessionKV, resident, sessionKV)
	}
	if err := checkMaxRSSAgainstResident(got.Ceiling, resident, sessionKV); err != nil {
		t.Fatalf("raised ceiling still fails the boot check: %v", err)
	}
}

func TestMaxRSSStaleMacPlistCeilingStrictStillRefuses(t *testing.T) {
	const (
		plistMaxRSS = uint64(25769803776)
		resident    = uint64(29066260070)
		sessionKV   = rssTestGiB
	)
	spec, err := parseMaxRSSSpec("25769803776")
	if err != nil {
		t.Fatalf("parseMaxRSSSpec(plist value) error = %v", err)
	}
	_, err = resolveMaxRSSCeiling(spec, resident, sessionKV, defaultMaxRSSHeadroomPct, true)
	if err == nil {
		t.Fatal("strict resolve of the stale Mac plist ceiling returned nil error, want refusal")
	}
	if !errors.Is(err, ErrMaxRSSBelowResident) {
		t.Fatalf("err = %v, want errors.Is ErrMaxRSSBelowResident", err)
	}
	var typed *MaxRSSBelowResidentError
	if !errors.As(err, &typed) {
		t.Fatalf("errors.As(*MaxRSSBelowResidentError) failed on %v", err)
	}
	suggested := deriveMaxRSSCeiling(resident, sessionKV, defaultMaxRSSHeadroomPct)
	if typed.Suggested != suggested {
		t.Fatalf("Suggested = %d, want %d", typed.Suggested, suggested)
	}
	if typed.Suggested <= resident+sessionKV {
		t.Fatalf("Suggested = %d, want > resident+sessionKV %d", typed.Suggested, resident+sessionKV)
	}
	msg := err.Error()
	if !strings.Contains(msg, formatBytes(suggested)) {
		t.Fatalf("error %q does not name the suggested ceiling %s", msg, formatBytes(suggested))
	}
	if !strings.Contains(msg, "auto") {
		t.Fatalf("error %q does not point at --max-rss auto", msg)
	}
}
