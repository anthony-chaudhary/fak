package compute

// lane_device_props.go reads the device properties a CUDA driver probe measured on a GPU, so the lane
// ledger's SM count and the NVArch per-SM limits can be checked against silicon rather than typed in.
//
// Accepted file shape: top-level name, api, driverVersion, totalGlobalMem, and the attributes. The
// attributes may sit under an "attributes" object or at top level. Each is a bare integer or an object
// {"value": n, "rc": r}; a non-zero rc refuses the file because the probe call itself failed. Keys this
// package does not read are kept in Extra and counted, never dropped.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strconv"
)

// Each refusal is a distinct sentinel so a caller can branch on the cause with errors.Is.
var (
	ErrDevicePropsMalformed         = errors.New("device properties: malformed probe document")
	ErrDevicePropsNoSMCount         = errors.New("device properties: multiProcessorCount is not reported; the SM count is required")
	ErrDevicePropsNonPositive       = errors.New("device properties: a reported value must be positive where the arithmetic divides or sizes by it, and non-negative otherwise")
	ErrDevicePropsAttributeFailed   = errors.New("device properties: a probe attribute reported a non-zero return code")
	ErrDevicePropsDuplicate         = errors.New("device properties: an attribute is given twice (nested and top-level spellings)")
	ErrDevicePropsNoCapability      = errors.New("device properties: compute capability is not reported")
	ErrDevicePropsUnknownCapability = errors.New("device properties: compute capability has no registered NVArch row")
)

// devicePropsSourceBase names where every DeviceProps value came from. The device name is appended when
// the probe reports one.
const devicePropsSourceBase = "device properties parsed from caller-supplied file"

// DevicePropsField is one probe value. Present distinguishes a reported zero from an absent field, so an
// unreported value is never read as zero.
type DevicePropsField struct {
	Value   int64 `json:"value"`
	Present bool  `json:"present"`
}

// DeviceProps is the device-attribute record of one GPU. Units: SM-count, warps, threads, registers,
// bytes and kHz are separate fields; no field is converted into another.
type DeviceProps struct {
	Name                string                     `json:"name"`
	API                 string                     `json:"api,omitempty"`
	Provenance          string                     `json:"provenance"`
	DriverVersion       DevicePropsField           `json:"driver_version"`
	TotalGlobalMemBytes DevicePropsField           `json:"total_global_mem_bytes"`
	CCMajor             DevicePropsField           `json:"cc_major"`
	CCMinor             DevicePropsField           `json:"cc_minor"`
	SMCount             DevicePropsField           `json:"sm_count"`
	WarpSize            DevicePropsField           `json:"warp_size"`
	RegsPerSM           DevicePropsField           `json:"regs_per_sm"`
	RegsPerBlock        DevicePropsField           `json:"regs_per_block"`
	MaxThreadsPerSM     DevicePropsField           `json:"max_threads_per_sm"`
	MaxBlocksPerSM      DevicePropsField           `json:"max_blocks_per_sm"`
	MaxThreadsPerBlock  DevicePropsField           `json:"max_threads_per_block"`
	SmemPerSMBytes      DevicePropsField           `json:"smem_per_sm_bytes"`
	SmemPerBlockBytes   DevicePropsField           `json:"smem_per_block_bytes"`
	SmemPerBlockOptin   DevicePropsField           `json:"smem_per_block_optin_bytes"`
	ClockRateKHz        DevicePropsField           `json:"clock_rate_khz"`
	Extra               map[string]json.RawMessage `json:"extra,omitempty"`
}

// devicePropsAccessors maps every probe key this package reads to the DeviceProps field it fills.
var devicePropsAccessors = map[string]func(*DeviceProps) *DevicePropsField{
	"driverVersion":               func(p *DeviceProps) *DevicePropsField { return &p.DriverVersion },
	"totalGlobalMem":              func(p *DeviceProps) *DevicePropsField { return &p.TotalGlobalMemBytes },
	"computeCapabilityMajor":      func(p *DeviceProps) *DevicePropsField { return &p.CCMajor },
	"computeCapabilityMinor":      func(p *DeviceProps) *DevicePropsField { return &p.CCMinor },
	"multiProcessorCount":         func(p *DeviceProps) *DevicePropsField { return &p.SMCount },
	"warpSize":                    func(p *DeviceProps) *DevicePropsField { return &p.WarpSize },
	"regsPerMultiprocessor":       func(p *DeviceProps) *DevicePropsField { return &p.RegsPerSM },
	"regsPerBlock":                func(p *DeviceProps) *DevicePropsField { return &p.RegsPerBlock },
	"maxThreadsPerMultiProcessor": func(p *DeviceProps) *DevicePropsField { return &p.MaxThreadsPerSM },
	"maxBlocksPerMultiProcessor":  func(p *DeviceProps) *DevicePropsField { return &p.MaxBlocksPerSM },
	"maxThreadsPerBlock":          func(p *DeviceProps) *DevicePropsField { return &p.MaxThreadsPerBlock },
	"sharedMemPerMultiprocessor":  func(p *DeviceProps) *DevicePropsField { return &p.SmemPerSMBytes },
	"sharedMemPerBlock":           func(p *DeviceProps) *DevicePropsField { return &p.SmemPerBlockBytes },
	"sharedMemPerBlockOptin":      func(p *DeviceProps) *DevicePropsField { return &p.SmemPerBlockOptin },
	"clockRate_kHz":               func(p *DeviceProps) *DevicePropsField { return &p.ClockRateKHz },
}

// devicePropsKeys is devicePropsAccessors in sorted order, so refusals and Extra handling are deterministic.
var devicePropsKeys = func() []string {
	keys := make([]string, 0, len(devicePropsAccessors))
	for k := range devicePropsAccessors {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}()

// ParseDeviceProps reads a probe document in the accepted shape. It refuses a document with no SM count,
// since the lane ledger cannot price a launch without it.
func ParseDeviceProps(r io.Reader) (DeviceProps, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return DeviceProps{}, fmt.Errorf("%w: %v", ErrDevicePropsMalformed, err)
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(data, &top); err != nil {
		return DeviceProps{}, fmt.Errorf("%w: %v", ErrDevicePropsMalformed, err)
	}
	p := DeviceProps{Extra: map[string]json.RawMessage{}}
	seen := map[string]bool{}
	for _, key := range sortedRawKeys(top) {
		if err := p.loadTopLevel(key, top[key], seen); err != nil {
			return DeviceProps{}, err
		}
	}
	if err := p.validate(); err != nil {
		return DeviceProps{}, err
	}
	p.Provenance = devicePropsSourceBase
	if p.Name != "" {
		p.Provenance += ": " + p.Name
	}
	return p, nil
}

func sortedRawKeys(m map[string]json.RawMessage) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// loadTopLevel routes one top-level key: the nested attributes object, a string field, a known flat
// attribute, or an unknown key kept in Extra.
func (p *DeviceProps) loadTopLevel(key string, raw json.RawMessage, seen map[string]bool) error {
	switch key {
	case "attributes":
		return p.loadAttributes(raw, seen)
	case "name":
		return unmarshalDevicePropsString(raw, &p.Name)
	case "api":
		return unmarshalDevicePropsString(raw, &p.API)
	}
	if _, ok := devicePropsAccessors[key]; ok {
		return p.setAttribute(key, raw, seen)
	}
	p.Extra[key] = raw
	return nil
}

func (p *DeviceProps) loadAttributes(raw json.RawMessage, seen map[string]bool) error {
	var attrs map[string]json.RawMessage
	if err := json.Unmarshal(raw, &attrs); err != nil {
		return fmt.Errorf("%w: attributes must be an object: %v", ErrDevicePropsMalformed, err)
	}
	for _, key := range sortedRawKeys(attrs) {
		if _, ok := devicePropsAccessors[key]; ok {
			if err := p.setAttribute(key, attrs[key], seen); err != nil {
				return err
			}
			continue
		}
		p.Extra["attributes."+key] = attrs[key]
	}
	return nil
}

func (p *DeviceProps) setAttribute(key string, raw json.RawMessage, seen map[string]bool) error {
	if seen[key] {
		return fmt.Errorf("%w: %s", ErrDevicePropsDuplicate, key)
	}
	value, err := devicePropsScalar(raw)
	if err != nil {
		return fmt.Errorf("attribute %s: %w", key, err)
	}
	seen[key] = true
	*devicePropsAccessors[key](p) = DevicePropsField{Value: value, Present: true}
	return nil
}

func unmarshalDevicePropsString(raw json.RawMessage, dst *string) error {
	if err := json.Unmarshal(raw, dst); err != nil {
		return fmt.Errorf("%w: want a string: %v", ErrDevicePropsMalformed, err)
	}
	return nil
}

// devicePropsScalar reads a bare integer, or an object {"value": n, "rc": r} whose rc is zero. A
// missing rc is accepted; a non-zero rc means the probe call failed, so the value is refused. A JSON null
// is refused too: Unmarshal would read it as zero, which is an unreported value, not a measured one.
func devicePropsScalar(raw json.RawMessage) (int64, error) {
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return 0, fmt.Errorf("%w: null is not a reported value", ErrDevicePropsMalformed)
	}
	var bare int64
	if err := json.Unmarshal(raw, &bare); err == nil {
		return bare, nil
	}
	var obj struct {
		Value *int64 `json:"value"`
		RC    *int64 `json:"rc"`
	}
	if err := json.Unmarshal(raw, &obj); err != nil || obj.Value == nil {
		return 0, fmt.Errorf("%w: want an integer or {value, rc}", ErrDevicePropsMalformed)
	}
	if obj.RC != nil && *obj.RC != 0 {
		return 0, fmt.Errorf("%w: rc=%d", ErrDevicePropsAttributeFailed, *obj.RC)
	}
	return *obj.Value, nil
}

// validate refuses an absent SM count, a zero SM count or warp size (both divide or size the ledger),
// and any negative reported value.
func (p *DeviceProps) validate() error {
	for _, key := range devicePropsKeys {
		if f := devicePropsAccessors[key](p); f.Present && f.Value < 0 {
			return fmt.Errorf("%w: %s=%d", ErrDevicePropsNonPositive, key, f.Value)
		}
	}
	if !p.SMCount.Present {
		return ErrDevicePropsNoSMCount
	}
	if p.SMCount.Value == 0 {
		return fmt.Errorf("%w: multiProcessorCount=0", ErrDevicePropsNonPositive)
	}
	if p.WarpSize.Present && p.WarpSize.Value == 0 {
		return fmt.Errorf("%w: warpSize=0", ErrDevicePropsNonPositive)
	}
	return nil
}

// DeviceSMs returns the reported SM count. It refuses an unreported or non-positive count rather than
// returning a default, because a wrong count silently rescales every lane-ledger share.
func (p DeviceProps) DeviceSMs() (int, error) {
	if !p.SMCount.Present {
		return 0, ErrDevicePropsNoSMCount
	}
	if p.SMCount.Value <= 0 {
		return 0, fmt.Errorf("%w: multiProcessorCount=%d", ErrDevicePropsNonPositive, p.SMCount.Value)
	}
	return int(p.SMCount.Value), nil
}

// devicePropsLabel spells a compute capability the way NVArch.Label does, e.g. 12.0 -> "sm_120".
func devicePropsLabel(major, minor int64) string {
	return "sm_" + strconv.FormatInt(major, 10) + strconv.FormatInt(minor, 10)
}

// Arch maps the reported compute capability to its registered NVArch row. Only an exact label match is
// accepted: a neighbouring capability such as 12.1 is refused, never mapped onto sm_120.
func (p DeviceProps) Arch() (NVArch, error) {
	if !p.CCMajor.Present || !p.CCMinor.Present {
		return NVArch{}, ErrDevicePropsNoCapability
	}
	if p.CCMinor.Value < 0 || p.CCMinor.Value > 9 {
		return NVArch{}, fmt.Errorf("%w: %d.%d", ErrDevicePropsUnknownCapability, p.CCMajor.Value, p.CCMinor.Value)
	}
	label := devicePropsLabel(p.CCMajor.Value, p.CCMinor.Value)
	for _, row := range KnownNVArches() {
		if row.Label == label {
			return row, nil
		}
	}
	return NVArch{}, fmt.Errorf("%w: %s", ErrDevicePropsUnknownCapability, label)
}

// Status values of a DevicePropsFieldCheck and of DevicePropsCheck.Capability.
const (
	DevicePropsMatch        = "MATCH"
	DevicePropsMismatch     = "MISMATCH"
	DevicePropsNotReported  = "NOT_REPORTED"
	DevicePropsUnverifiable = "UNVERIFIABLE_FROM_DEVICE"
)

// DevicePropsVerdict is the overall outcome of CheckArch.
type DevicePropsVerdict string

const (
	// DevicePropsConfirmed: every checkable field matches the row.
	DevicePropsConfirmed DevicePropsVerdict = "confirmed"
	// DevicePropsRefuted: at least one checkable field disagrees with the row. A mismatch outranks an
	// unreported field.
	DevicePropsRefuted DevicePropsVerdict = "refuted"
	// DevicePropsIncomplete: no field disagrees, but at least one checkable field is not reported.
	DevicePropsIncomplete DevicePropsVerdict = "incomplete"
)

// DevicePropsFieldCheck compares one NVArch field with the device value. Device is present only when the
// probe reported it; a NOT_REPORTED field carries no device value at all.
type DevicePropsFieldCheck struct {
	Field   string           `json:"field"`
	Source  string           `json:"source"`
	Status  string           `json:"status"`
	Derived bool             `json:"derived,omitempty"`
	Device  DevicePropsField `json:"device"`
	Row     int64            `json:"row"`
}

// DevicePropsCheck is the field-by-field comparison of a device with one NVArch row. Capability compares
// the reported compute capability label with the row label, because two rows can share their per-SM
// limits (sm_89 and sm_120) and the limits alone would not tell them apart.
type DevicePropsCheck struct {
	Arch         string                  `json:"arch"`
	Capability   string                  `json:"capability"`
	Verdict      DevicePropsVerdict      `json:"verdict"`
	Fields       []DevicePropsFieldCheck `json:"fields"`
	Unverifiable []DevicePropsFieldCheck `json:"unverifiable_from_device"`
}

// CheckArch compares the device's per-SM limits with row a field by field.
func (p DeviceProps) CheckArch(a NVArch) DevicePropsCheck {
	c := DevicePropsCheck{
		Arch:       a.Label,
		Capability: p.capabilityStatus(a),
		Fields: []DevicePropsFieldCheck{
			compareDevicePropsField("RegsPerSM", "regsPerMultiprocessor", p.RegsPerSM, int64(a.RegsPerSM)),
			compareDevicePropsField("MaxThreadsPerSM", "maxThreadsPerMultiProcessor", p.MaxThreadsPerSM, int64(a.MaxThreadsPerSM)),
			p.maxWarpsCheck(a),
			compareDevicePropsField("MaxBlocksPerSM", "maxBlocksPerMultiProcessor", p.MaxBlocksPerSM, int64(a.MaxBlocksPerSM)),
			compareDevicePropsField("SmemPerSM", "sharedMemPerMultiprocessor", p.SmemPerSMBytes, int64(a.SmemPerSM)),
			compareDevicePropsField("WarpSize", "warpSize", p.WarpSize, int64(a.WarpSize)),
		},
		Unverifiable: []DevicePropsFieldCheck{
			unverifiableDevicePropsField("RegAllocUnit", int64(a.RegAllocUnit)),
			unverifiableDevicePropsField("WarpAllocGranularity", int64(a.WarpAllocGranularity)),
			unverifiableDevicePropsField("SmemAllocUnit", int64(a.SmemAllocUnit)),
		},
	}
	c.Verdict = c.verdict()
	return c
}

func (p DeviceProps) capabilityStatus(a NVArch) string {
	if !p.CCMajor.Present || !p.CCMinor.Present {
		return DevicePropsNotReported
	}
	if devicePropsLabel(p.CCMajor.Value, p.CCMinor.Value) == a.Label {
		return DevicePropsMatch
	}
	return DevicePropsMismatch
}

// maxWarpsCheck derives the device warp ceiling as resident threads over warp size. It is reported only
// when both inputs are present, so an absent input is never turned into a derived zero.
func (p DeviceProps) maxWarpsCheck(a NVArch) DevicePropsFieldCheck {
	check := DevicePropsFieldCheck{
		Field:   "MaxWarpsPerSM",
		Source:  "derived: maxThreadsPerMultiProcessor / warpSize",
		Derived: true,
		Row:     int64(a.MaxWarpsPerSM),
	}
	// A zero warp size only reaches here through a hand-built value; ParseDeviceProps refuses it.
	if !p.MaxThreadsPerSM.Present || !p.WarpSize.Present || p.WarpSize.Value <= 0 {
		check.Status = DevicePropsNotReported
		return check
	}
	check.Device = DevicePropsField{Value: p.MaxThreadsPerSM.Value / p.WarpSize.Value, Present: true}
	check.Status = devicePropsStatus(check.Device, check.Row)
	return check
}

func compareDevicePropsField(field, source string, dev DevicePropsField, row int64) DevicePropsFieldCheck {
	return DevicePropsFieldCheck{
		Field:  field,
		Source: source,
		Status: devicePropsStatus(dev, row),
		Device: dev,
		Row:    row,
	}
}

func unverifiableDevicePropsField(field string, row int64) DevicePropsFieldCheck {
	return DevicePropsFieldCheck{
		Field:  field,
		Source: "not exposed by CUDA device attributes",
		Status: DevicePropsUnverifiable,
		Row:    row,
	}
}

func devicePropsStatus(dev DevicePropsField, row int64) string {
	switch {
	case !dev.Present:
		return DevicePropsNotReported
	case dev.Value == row:
		return DevicePropsMatch
	default:
		return DevicePropsMismatch
	}
}

// verdict applies the precedence refuted > incomplete > confirmed. Unverifiable fields never affect it.
func (c DevicePropsCheck) verdict() DevicePropsVerdict {
	statuses := []string{c.Capability}
	for _, f := range c.Fields {
		statuses = append(statuses, f.Status)
	}
	incomplete := false
	for _, s := range statuses {
		switch s {
		case DevicePropsMismatch:
			return DevicePropsRefuted
		case DevicePropsNotReported:
			incomplete = true
		}
	}
	if incomplete {
		return DevicePropsIncomplete
	}
	return DevicePropsConfirmed
}
