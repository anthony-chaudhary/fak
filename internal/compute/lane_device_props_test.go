package compute

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// devicePropsFixture opens a probe document from the committed testdata directory.
func devicePropsFixture(t *testing.T, name string) DeviceProps {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", "lane_device_props", name))
	if err != nil {
		t.Fatalf("open fixture %s: %v", name, err)
	}
	defer f.Close()
	p, err := ParseDeviceProps(f)
	if err != nil {
		t.Fatalf("parse fixture %s: %v", name, err)
	}
	return p
}

// devicePropsFromString parses an inline probe document.
func devicePropsFromString(doc string) (DeviceProps, error) {
	return ParseDeviceProps(strings.NewReader(doc))
}

// devicePropsCheckField returns the named field comparison, failing the test if the check has no such row.
func devicePropsCheckField(t *testing.T, c DevicePropsCheck, field string) DevicePropsFieldCheck {
	t.Helper()
	for _, f := range c.Fields {
		if f.Field == field {
			return f
		}
	}
	t.Fatalf("check has no field %q: %+v", field, c.Fields)
	return DevicePropsFieldCheck{}
}

func mustSM120(t *testing.T) NVArch {
	t.Helper()
	row, ok := LookupNVArch(SM120)
	if !ok {
		t.Fatal("sm_120 row missing from the NVArch table")
	}
	return row
}

func TestDevicePropsRealFixture(t *testing.T) {
	p := devicePropsFixture(t, "rtx5080.json")

	sms, err := p.DeviceSMs()
	if err != nil || sms != 84 {
		t.Fatalf("DeviceSMs() = %d, %v; want 84, nil", sms, err)
	}
	arch, err := p.Arch()
	if err != nil {
		t.Fatalf("Arch() error: %v", err)
	}
	if arch.Label != "sm_120" || arch.Cap != SM120 {
		t.Fatalf("Arch() = %s/%d, want sm_120/SM120", arch.Label, arch.Cap)
	}
	if p.Name != "NVIDIA GeForce RTX 5080" {
		t.Errorf("Name = %q", p.Name)
	}
	if want := devicePropsSourceBase + ": NVIDIA GeForce RTX 5080"; p.Provenance != want {
		t.Errorf("Provenance = %q, want %q", p.Provenance, want)
	}
	if !p.DriverVersion.Present || p.DriverVersion.Value != 13050 {
		t.Errorf("DriverVersion = %+v, want 13050 present", p.DriverVersion)
	}
	wantExtra := []string{"attributes.memoryBusWidth", "attributes.memoryClockRate_kHz", "attributes.l2CacheSize"}
	if len(p.Extra) != len(wantExtra) {
		t.Fatalf("Extra has %d keys, want %d: %v", len(p.Extra), len(wantExtra), p.Extra)
	}
	for _, k := range wantExtra {
		if _, ok := p.Extra[k]; !ok {
			t.Errorf("Extra missing %q; keys = %v", k, p.Extra)
		}
	}
}

func TestDevicePropsRealFixtureConfirmed(t *testing.T) {
	p := devicePropsFixture(t, "rtx5080.json")
	arch, err := p.Arch()
	if err != nil {
		t.Fatalf("Arch() error: %v", err)
	}
	c := p.CheckArch(arch)

	if c.Verdict != DevicePropsConfirmed {
		t.Fatalf("verdict = %s, want confirmed; check = %+v", c.Verdict, c)
	}
	if c.Capability != DevicePropsMatch {
		t.Errorf("Capability = %s, want MATCH", c.Capability)
	}
	for _, field := range []string{"RegsPerSM", "MaxThreadsPerSM", "MaxBlocksPerSM", "SmemPerSM", "WarpSize", "MaxWarpsPerSM"} {
		if f := devicePropsCheckField(t, c, field); f.Status != DevicePropsMatch {
			t.Errorf("%s status = %s, want MATCH (%+v)", field, f.Status, f)
		}
	}
	warps := devicePropsCheckField(t, c, "MaxWarpsPerSM")
	if !warps.Derived || !warps.Device.Present || warps.Device.Value != 48 || warps.Row != 48 {
		t.Errorf("MaxWarpsPerSM = %+v, want derived device 48 vs row 48", warps)
	}
	if len(c.Unverifiable) != 3 {
		t.Fatalf("unverifiable rows = %d, want 3: %+v", len(c.Unverifiable), c.Unverifiable)
	}
	for _, f := range c.Unverifiable {
		if f.Status != DevicePropsUnverifiable || f.Device.Present {
			t.Errorf("unverifiable row %+v must carry UNVERIFIABLE_FROM_DEVICE and no device value", f)
		}
	}
}

func TestDevicePropsFlatSpellingMatchesNested(t *testing.T) {
	flat := devicePropsFixture(t, "rtx5080_flat.json")
	nested := devicePropsFixture(t, "rtx5080.json")
	if flat.SMCount != nested.SMCount || flat.RegsPerSM != nested.RegsPerSM ||
		flat.MaxBlocksPerSM != nested.MaxBlocksPerSM || flat.SmemPerSMBytes != nested.SmemPerSMBytes {
		t.Fatalf("flat spelling parsed differently:\nflat   %+v\nnested %+v", flat, nested)
	}
	if len(flat.Extra) != 0 {
		t.Errorf("flat fixture has no unknown keys but Extra = %v", flat.Extra)
	}
	arch, err := flat.Arch()
	if err != nil {
		t.Fatalf("Arch() error: %v", err)
	}
	if c := flat.CheckArch(arch); c.Verdict != DevicePropsConfirmed {
		t.Errorf("flat verdict = %s, want confirmed", c.Verdict)
	}
}

func TestDevicePropsRefusesMissingSMCountFixture(t *testing.T) {
	f, err := os.Open(filepath.Join("testdata", "lane_device_props", "rtx5080_missing_sm_count.json"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := ParseDeviceProps(f); !errors.Is(err, ErrDevicePropsNoSMCount) {
		t.Fatalf("ParseDeviceProps error = %v, want ErrDevicePropsNoSMCount", err)
	}
}

func TestDevicePropsParseRefusals(t *testing.T) {
	tests := []struct {
		name string
		doc  string
		want error
	}{
		{"not json", "{", ErrDevicePropsMalformed},
		{"top level is an array", "[]", ErrDevicePropsMalformed},
		{"fractional value", `{"multiProcessorCount": 84.5}`, ErrDevicePropsMalformed},
		{"attributes not an object", `{"attributes": 5}`, ErrDevicePropsMalformed},
		{"probe call failed", `{"multiProcessorCount": {"value": 84, "rc": 1}}`, ErrDevicePropsAttributeFailed},
		{"no SM count", `{"name": "x"}`, ErrDevicePropsNoSMCount},
		{"zero SM count", `{"multiProcessorCount": 0}`, ErrDevicePropsNonPositive},
		{"negative warp size", `{"multiProcessorCount": 84, "warpSize": -32}`, ErrDevicePropsNonPositive},
		{"zero warp size", `{"multiProcessorCount": 84, "warpSize": 0}`, ErrDevicePropsNonPositive},
		{"nested and flat spellings both present", `{"multiProcessorCount": 84, "attributes": {"multiProcessorCount": 84}}`, ErrDevicePropsDuplicate},
		{"null SM count is not a zero", `{"multiProcessorCount": null}`, ErrDevicePropsMalformed},
		{"null capability minor is not a zero", `{"multiProcessorCount": 84, "computeCapabilityMajor": 12, "computeCapabilityMinor": null}`, ErrDevicePropsMalformed},
		{"null in the nested spelling", `{"attributes": {"clockRate_kHz": null}}`, ErrDevicePropsMalformed},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := devicePropsFromString(tc.doc)
			if !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestDevicePropsArchRefusals(t *testing.T) {
	tests := []struct {
		name string
		doc  string
		want error
	}{
		{"capability absent", `{"multiProcessorCount": 84}`, ErrDevicePropsNoCapability},
		{"neighbour of sm_120 is not guessed", `{"multiProcessorCount": 84, "computeCapabilityMajor": 12, "computeCapabilityMinor": 1}`, ErrDevicePropsUnknownCapability},
		{"no row for 8.6", `{"multiProcessorCount": 84, "computeCapabilityMajor": 8, "computeCapabilityMinor": 6}`, ErrDevicePropsUnknownCapability},
		{"minor outside one digit", `{"multiProcessorCount": 84, "computeCapabilityMajor": 1, "computeCapabilityMinor": 10}`, ErrDevicePropsUnknownCapability},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			p, err := devicePropsFromString(tc.doc)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			if _, err := p.Arch(); !errors.Is(err, tc.want) {
				t.Fatalf("Arch() error = %v, want %v", err, tc.want)
			}
		})
	}
}

func TestDevicePropsArchMapsRegisteredRows(t *testing.T) {
	tests := []struct {
		major, minor int64
		label        string
	}{
		{8, 0, "sm_80"},
		{8, 9, "sm_89"},
		{9, 0, "sm_90"},
		{10, 0, "sm_100"},
		{12, 0, "sm_120"},
	}
	for _, tc := range tests {
		p := DeviceProps{
			SMCount: DevicePropsField{Value: 1, Present: true},
			CCMajor: DevicePropsField{Value: tc.major, Present: true},
			CCMinor: DevicePropsField{Value: tc.minor, Present: true},
		}
		arch, err := p.Arch()
		if err != nil || arch.Label != tc.label {
			t.Errorf("Arch() for %d.%d = %s, %v; want %s", tc.major, tc.minor, arch.Label, err, tc.label)
		}
	}
}

func TestDevicePropsRealFixtureUnknownCapabilityRefused(t *testing.T) {
	p := devicePropsFixture(t, "rtx5080_unknown_capability.json")
	if _, err := p.Arch(); !errors.Is(err, ErrDevicePropsUnknownCapability) {
		t.Fatalf("Arch() error = %v, want ErrDevicePropsUnknownCapability", err)
	}
	if sms, err := p.DeviceSMs(); err != nil || sms != 84 {
		t.Errorf("DeviceSMs() = %d, %v; the SM count stays readable without a capability row", sms, err)
	}
}

func TestDevicePropsDeviceSMsRefusesUnreported(t *testing.T) {
	var p DeviceProps
	if _, err := p.DeviceSMs(); !errors.Is(err, ErrDevicePropsNoSMCount) {
		t.Fatalf("DeviceSMs() on an unreported count = %v, want ErrDevicePropsNoSMCount", err)
	}
	p.SMCount = DevicePropsField{Value: 0, Present: true}
	if _, err := p.DeviceSMs(); !errors.Is(err, ErrDevicePropsNonPositive) {
		t.Fatalf("DeviceSMs() on a zero count = %v, want ErrDevicePropsNonPositive", err)
	}
}

func TestDevicePropsCheckRefutedOnMismatch(t *testing.T) {
	p := devicePropsFixture(t, "rtx5080_blocks_mismatch.json")
	arch, err := p.Arch()
	if err != nil {
		t.Fatalf("Arch() error: %v", err)
	}
	c := p.CheckArch(arch)
	if c.Verdict != DevicePropsRefuted {
		t.Fatalf("verdict = %s, want refuted", c.Verdict)
	}
	blocks := devicePropsCheckField(t, c, "MaxBlocksPerSM")
	if blocks.Status != DevicePropsMismatch || blocks.Device.Value != 32 || blocks.Row != 24 {
		t.Errorf("MaxBlocksPerSM = %+v, want MISMATCH device 32 vs row 24", blocks)
	}
	if f := devicePropsCheckField(t, c, "RegsPerSM"); f.Status != DevicePropsMatch {
		t.Errorf("RegsPerSM status = %s, want MATCH", f.Status)
	}
}

func TestDevicePropsCheckIncompleteWhenFieldUnreported(t *testing.T) {
	p := devicePropsFixture(t, "rtx5080_missing_threads_per_sm.json")
	arch, err := p.Arch()
	if err != nil {
		t.Fatalf("Arch() error: %v", err)
	}
	c := p.CheckArch(arch)
	if c.Verdict != DevicePropsIncomplete {
		t.Fatalf("verdict = %s, want incomplete", c.Verdict)
	}
	threads := devicePropsCheckField(t, c, "MaxThreadsPerSM")
	if threads.Status != DevicePropsNotReported || threads.Device.Present {
		t.Errorf("MaxThreadsPerSM = %+v, want NOT_REPORTED with no device value", threads)
	}
	warps := devicePropsCheckField(t, c, "MaxWarpsPerSM")
	if warps.Status != DevicePropsNotReported || !warps.Derived || warps.Device.Present {
		t.Errorf("MaxWarpsPerSM = %+v, want derived NOT_REPORTED (its input is absent)", warps)
	}
}

func TestDevicePropsRefutedOutranksIncomplete(t *testing.T) {
	p := DeviceProps{
		CCMajor:        DevicePropsField{Value: 12, Present: true},
		CCMinor:        DevicePropsField{Value: 0, Present: true},
		SMCount:        DevicePropsField{Value: 84, Present: true},
		MaxBlocksPerSM: DevicePropsField{Value: 32, Present: true},
	}
	arch := mustSM120(t)
	if c := p.CheckArch(arch); c.Verdict != DevicePropsRefuted {
		t.Fatalf("verdict = %s, want refuted (a mismatch outranks unreported fields)", c.Verdict)
	}
}

func TestDevicePropsCapabilityDistinguishesSharedLimits(t *testing.T) {
	p := devicePropsFixture(t, "rtx5080.json")
	sm89, ok := LookupNVArch(SM89)
	if !ok {
		t.Fatal("sm_89 row missing")
	}
	c := p.CheckArch(sm89)
	if c.Capability != DevicePropsMismatch {
		t.Errorf("Capability against sm_89 = %s, want MISMATCH", c.Capability)
	}
	if c.Verdict != DevicePropsRefuted {
		t.Errorf("verdict against sm_89 = %s, want refuted; sm_89 and sm_120 share every per-SM limit", c.Verdict)
	}
	if f := devicePropsCheckField(t, c, "MaxBlocksPerSM"); f.Status != DevicePropsMatch {
		t.Errorf("per-SM MaxBlocksPerSM against sm_89 = %s, want MATCH: the shared limit is not the discriminator", f.Status)
	}
}

func TestDevicePropsUnknownKeysAreKept(t *testing.T) {
	p, err := devicePropsFromString(`{"hostProbe": "x", "attributes": {"multiProcessorCount": 4, "extraAttr": 9}}`)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if _, ok := p.Extra["hostProbe"]; !ok {
		t.Errorf("top-level unknown key not kept: %v", p.Extra)
	}
	if _, ok := p.Extra["attributes.extraAttr"]; !ok {
		t.Errorf("nested unknown attribute not kept: %v", p.Extra)
	}
	if len(p.Extra) != 2 {
		t.Errorf("Extra has %d keys, want 2", len(p.Extra))
	}
}

func TestDevicePropsSentinelsDistinct(t *testing.T) {
	sentinels := []error{
		ErrDevicePropsMalformed, ErrDevicePropsNoSMCount, ErrDevicePropsNonPositive,
		ErrDevicePropsAttributeFailed, ErrDevicePropsDuplicate, ErrDevicePropsNoCapability,
		ErrDevicePropsUnknownCapability,
	}
	for i, a := range sentinels {
		for j, b := range sentinels {
			if i != j && errors.Is(a, b) {
				t.Errorf("sentinel %d matches sentinel %d under errors.Is", i, j)
			}
		}
	}
}
