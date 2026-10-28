// Package hwutil is the public hardware-usage telemetry plane: a closed,
// versioned counter schema (node / device / link / tier) covering CPU, DRAM,
// disk/IO, network, USB4/TB links, GPU devices, and memory-tier residency.
//
// The wire contract is frozen as schema fak.hwutil.v1 (goal
// GOAL-hw-utilization-telemetry): every metric is a *float64 or *int64 where
// nil means unsupported/unread on this host — never zero-filled — while
// identity fields are always set. Null-on-error survives to the wire: JSON
// rendering keeps nulls for unread metrics instead of dropping them.
package hwutil

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

// Schema is the frozen snapshot wire version. Changing any tag below is a
// contract change that requires a new schema value.
const Schema = "fak.hwutil.v1"

// FleetSchema is the frozen multi-node roll-up wire version.
const FleetSchema = "fak.hwutil.fleet.v1"

// Closed error-code vocabulary. Code strings cross the wire and are matched
// by exact value; prose around them is not part of the contract.
type ErrorCode string

const (
	// CodeCollectorUnreadable marks a family whose source does not exist on
	// this host (no /proc file, no sysfs class, no driver surface).
	CodeCollectorUnreadable ErrorCode = "HWUTIL_COLLECTOR_UNREADABLE"
	// CodePartial marks a snapshot in which some families are unread while
	// the rest are valid.
	CodePartial ErrorCode = "HWUTIL_PARTIAL"
	// CodeHostUnreachable marks a remote node that did not answer.
	CodeHostUnreachable ErrorCode = "HWUTIL_HOST_UNREACHABLE"
	// CodeSnapshotMalformed marks a document that violates the frozen shape.
	CodeSnapshotMalformed ErrorCode = "HWUTIL_SNAPSHOT_MALFORMED"
	// CodeStoreFull marks a durable store that refused a write.
	CodeStoreFull ErrorCode = "HWUTIL_STORE_FULL"
)

var (
	ErrCollectorUnreadable = &codeError{code: CodeCollectorUnreadable}
	ErrPartial             = &codeError{code: CodePartial}
	ErrHostUnreachable     = &codeError{code: CodeHostUnreachable}
	ErrSnapshotMalformed   = &codeError{code: CodeSnapshotMalformed}
	ErrStoreFull           = &codeError{code: CodeStoreFull}
)

// codeError carries one closed error code. Sentinels are the zero-detail
// instances; dynamic errors wrap a sentinel so errors.Is keeps working.
type codeError struct {
	code     ErrorCode
	msg      string
	sentinel error
}

func (e *codeError) Error() string {
	if e.msg == "" {
		return string(e.code)
	}
	return string(e.code) + ": " + e.msg
}

// Code returns the closed code carried by this error.
func (e *codeError) Code() ErrorCode { return e.code }

// NewErr builds a dynamic error carrying code with detail msg. It wraps the
// code's sentinel, so errors.Is(err, ErrX) holds for the matching code.
func NewErr(code ErrorCode, format string, args ...any) error {
	var sentinel error
	switch code {
	case CodeCollectorUnreadable:
		sentinel = ErrCollectorUnreadable
	case CodePartial:
		sentinel = ErrPartial
	case CodeHostUnreachable:
		sentinel = ErrHostUnreachable
	case CodeSnapshotMalformed:
		sentinel = ErrSnapshotMalformed
	case CodeStoreFull:
		sentinel = ErrStoreFull
	default:
		return fmt.Errorf("hwutil: unknown error code %q", code)
	}
	return &codeError{code: code, msg: fmt.Sprintf(format, args...), sentinel: sentinel}
}

// Unwrap exposes the wrapped sentinel for errors.Is.
func (e *codeError) Unwrap() error { return e.sentinel }

// CodeOf walks err's chain and returns the first closed hwutil code it finds,
// or "" when err carries none.
func CodeOf(err error) ErrorCode {
	for err != nil {
		if ce, ok := err.(*codeError); ok {
			return ce.code
		}
		err = errors.Unwrap(err)
	}
	return ""
}

// Snapshot is one node sample in the frozen fak.hwutil.v1 wire shape. Every
// metric pointer is nil when the host cannot read it; identity fields
// (Schema, Host, Arch, Os, At) are always set.
type Snapshot struct {
	Schema          string        `json:"schema"`
	Host            string        `json:"host"`
	Arch            string        `json:"arch"`
	Os              string        `json:"os"`
	At              time.Time     `json:"at"`
	DurationSeconds float64       `json:"duration_seconds"`
	Error           string        `json:"error"`
	CPU             *CPU          `json:"cpu"`
	DRAM            *DRAM         `json:"dram"`
	Disks           []Disk        `json:"disks"`
	Net             []NetIfc      `json:"net"`
	Links           []Link        `json:"links"`
	GPUs            []GPU         `json:"gpus"`
	Tiers           []TierRow     `json:"tiers"`
	Transfers       []TransferRow `json:"transfers"`
}

// CPU is the processor family. Pct fields are 0..100.
type CPU struct {
	Cores     int      `json:"cores"`
	UtilPct   *float64 `json:"util_pct"`
	StealPct  *float64 `json:"steal_pct"`
	IOWaitPct *float64 `json:"iowait_pct"`
	IntervalS float64  `json:"interval_s"`
	Load1     *float64 `json:"load1"`
}

// DRAM is system memory. Every metric is nil-capable; bandwidth_gib_s has no
// stdlib-only source and stays nil unless a future collector fills it.
type DRAM struct {
	TotalBytes       *int64   `json:"total_bytes"`
	UsedBytes        *int64   `json:"used_bytes"`
	AvailableBytes   *int64   `json:"available_bytes"`
	PageFaultsPerSec *float64 `json:"page_faults_per_s"`
	SwapUsedBytes    *int64   `json:"swap_used_bytes"`
	BandwidthGiBS    *float64 `json:"bandwidth_gib_s"`
}

// DiskKind is the closed disk-transport vocabulary.
type DiskKind string

const (
	DiskNVMe    DiskKind = "nvme"
	DiskSATA    DiskKind = "sata"
	DiskUSB     DiskKind = "usb"
	DiskRamdisk DiskKind = "ramdisk"
	DiskUnknown DiskKind = "unknown"
)

// Disk is one storage device row. Rates are deltas over the sample window;
// free/total come from the mount point attributed to the device when one
// exists.
type Disk struct {
	Name           string   `json:"name"`
	Kind           DiskKind `json:"kind"`
	ReadBytesPerS  *float64 `json:"read_bytes_per_s"`
	WriteBytesPerS *float64 `json:"write_bytes_per_s"`
	IopsRead       *float64 `json:"iops_read"`
	IopsWrite      *float64 `json:"iops_write"`
	QueueDepth     *int64   `json:"queue_depth"`
	LatencyMSAvg   *float64 `json:"latency_ms_avg"`
	FreeBytes      *int64   `json:"free_bytes"`
	TotalBytes     *int64   `json:"total_bytes"`
}

// NetIfc is one network interface row. ErrorsTotal is cumulative, not a rate.
type NetIfc struct {
	Iface         string   `json:"iface"`
	RxBytesPerS   *float64 `json:"rx_bytes_per_s"`
	TxBytesPerS   *float64 `json:"tx_bytes_per_s"`
	RxPacketsPerS *float64 `json:"rx_packets_per_s"`
	TxPacketsPerS *float64 `json:"tx_packets_per_s"`
	RxDropsPerS   *float64 `json:"rx_drops_per_s"`
	TxDropsPerS   *float64 `json:"tx_drops_per_s"`
	ErrorsTotal   *int64   `json:"errors_total"`
}

// LinkKind is the closed fabric-link vocabulary.
type LinkKind string

const (
	LinkPCIE        LinkKind = "pcie"
	LinkNVLink      LinkKind = "nvlink"
	LinkUSB4        LinkKind = "usb4"
	LinkThunderbolt LinkKind = "thunderbolt"
	LinkTCP         LinkKind = "tcp"
	LinkRDMA        LinkKind = "rdma"
	LinkUnknown     LinkKind = "unknown"
)

// LinkState is the closed link-state vocabulary.
type LinkState string

const (
	LinkUp           LinkState = "up"
	LinkDown         LinkState = "down"
	LinkStateUnknown LinkState = "unknown"
)

// Link is one host fabric link (PCIe/NVLink/USB4/Thunderbolt/…); only state
// and speed are required, byte rates stay nil unless measured.
type Link struct {
	Kind        LinkKind  `json:"kind"`
	ID          string    `json:"id"`
	SpeedGBps   *float64  `json:"speed_gbps"`
	RxBytesPerS *float64  `json:"rx_bytes_per_s"`
	TxBytesPerS *float64  `json:"tx_bytes_per_s"`
	State       LinkState `json:"state"`
}

// GPU is one accelerator row. All metric pointers are nil when the vendor
// surface does not expose them; ThrottleReasons is an empty slice when the
// surface reports nothing active.
type GPU struct {
	UUID                string   `json:"uuid"`
	Name                string   `json:"name"`
	PCI                 string   `json:"pci"`
	UtilPct             *float64 `json:"util_pct"`
	MemTotalBytes       *int64   `json:"mem_total_bytes"`
	MemUsedBytes        *int64   `json:"mem_used_bytes"`
	PowerW              *float64 `json:"power_w"`
	SMClockMHz          *float64 `json:"sm_clock_mhz"`
	MaxSMClockMHz       *float64 `json:"max_sm_clock_mhz"`
	PCIERxBytesPerS     *float64 `json:"pcie_rx_bytes_per_s"`
	PCIETxBytesPerS     *float64 `json:"pcie_tx_bytes_per_s"`
	ThrottleReasons     []string `json:"throttle_reasons"`
	ECCUncorrectedTotal *int64   `json:"ecc_uncorrected_total"`
}

// TierKind is the closed memory-tier vocabulary.
type TierKind string

const (
	TierVRAM TierKind = "vram"
	TierRAM  TierKind = "ram"
	TierSSD  TierKind = "ssd"
)

// TransferReason is the closed transfer-reason vocabulary.
type TransferReason string

const (
	TransferLoad        TransferReason = "load"
	TransferEvict       TransferReason = "evict"
	TransferPPStageMove TransferReason = "pp_stage_move"
	TransferUnknown     TransferReason = "unknown"
)

// TierRow is one model-layer residency band: which layers of which model live
// in which tier on which device.
type TierRow struct {
	Kind              TierKind   `json:"kind"`
	Device            string     `json:"device"`
	ModelID           string     `json:"model_id"`
	LayersStart       int        `json:"layers_start"`
	LayersEnd         int        `json:"layers_end"`
	Bytes             int64      `json:"bytes"`
	HotnessRank       int        `json:"hotness_rank"`
	LastTransferAt    *time.Time `json:"last_transfer_at"`
	TransferBytesPerS *float64   `json:"transfer_bytes_per_s"`
}

// TransferRow is one observed cross-tier byte movement.
type TransferRow struct {
	FromTier  string         `json:"from_tier"`
	ToTier    string         `json:"to_tier"`
	Bytes     int64          `json:"bytes"`
	DurationS float64        `json:"duration_s"`
	RateGiBS  float64        `json:"rate_gib_s"`
	Reason    TransferReason `json:"reason"`
}

var (
	diskKinds  = map[DiskKind]bool{DiskNVMe: true, DiskSATA: true, DiskUSB: true, DiskRamdisk: true, DiskUnknown: true}
	linkKinds  = map[LinkKind]bool{LinkPCIE: true, LinkNVLink: true, LinkUSB4: true, LinkThunderbolt: true, LinkTCP: true, LinkRDMA: true, LinkUnknown: true}
	linkStates = map[LinkState]bool{LinkUp: true, LinkDown: true, LinkStateUnknown: true}
	tierKinds  = map[TierKind]bool{TierVRAM: true, TierRAM: true, TierSSD: true}
	reasons    = map[TransferReason]bool{TransferLoad: true, TransferEvict: true, TransferPPStageMove: true, TransferUnknown: true}
	arches     = map[string]bool{"amd64": true, "arm64": true}
	oses       = map[string]bool{"linux": true, "windows": true}
)

// Validate checks the frozen shape: schema identity, closed vocabularies,
// ranges, and non-negative counters. It returns errors wrapping
// ErrSnapshotMalformed (errors.Is-friendly) naming the offending field.
func (s *Snapshot) Validate() error {
	if s.Schema != Schema {
		return NewErr(CodeSnapshotMalformed, "schema %q is not %q", s.Schema, Schema)
	}
	if strings.TrimSpace(s.Host) == "" {
		return NewErr(CodeSnapshotMalformed, "host identity is empty")
	}
	if !arches[s.Arch] {
		return NewErr(CodeSnapshotMalformed, "arch %q outside amd64|arm64", s.Arch)
	}
	if !oses[s.Os] {
		return NewErr(CodeSnapshotMalformed, "os %q outside linux|windows", s.Os)
	}
	if s.At.IsZero() {
		return NewErr(CodeSnapshotMalformed, "at is not a parseable RFC3339 timestamp")
	}
	if s.DurationSeconds < 0 {
		return NewErr(CodeSnapshotMalformed, "duration_seconds is negative")
	}
	if s.CPU != nil {
		if s.CPU.Cores <= 0 {
			return NewErr(CodeSnapshotMalformed, "cpu.cores must be positive")
		}
		if err := checkPct("cpu.util_pct", s.CPU.UtilPct); err != nil {
			return err
		}
		if err := checkPct("cpu.steal_pct", s.CPU.StealPct); err != nil {
			return err
		}
		if err := checkPct("cpu.iowait_pct", s.CPU.IOWaitPct); err != nil {
			return err
		}
		if s.CPU.IntervalS < 0 {
			return NewErr(CodeSnapshotMalformed, "cpu.interval_s is negative")
		}
		if s.CPU.Load1 != nil && *s.CPU.Load1 < 0 {
			return NewErr(CodeSnapshotMalformed, "cpu.load1 is negative")
		}
	}
	if s.DRAM != nil {
		for _, p := range []*int64{s.DRAM.TotalBytes, s.DRAM.UsedBytes, s.DRAM.AvailableBytes, s.DRAM.SwapUsedBytes} {
			if p != nil && *p < 0 {
				return NewErr(CodeSnapshotMalformed, "dram byte counter is negative")
			}
		}
		if s.DRAM.PageFaultsPerSec != nil && *s.DRAM.PageFaultsPerSec < 0 {
			return NewErr(CodeSnapshotMalformed, "dram.page_faults_per_s is negative")
		}
		if s.DRAM.SwapUsedBytes != nil && *s.DRAM.SwapUsedBytes < 0 {
			return NewErr(CodeSnapshotMalformed, "dram.swap_used_bytes is negative")
		}
		if s.DRAM.BandwidthGiBS != nil && *s.DRAM.BandwidthGiBS < 0 {
			return NewErr(CodeSnapshotMalformed, "dram.bandwidth_gib_s is negative")
		}
	}
	for i, d := range s.Disks {
		if strings.TrimSpace(d.Name) == "" {
			return NewErr(CodeSnapshotMalformed, "disks[%d].name is empty", i)
		}
		if !diskKinds[d.Kind] {
			return NewErr(CodeSnapshotMalformed, "disks[%d].kind %q outside closed set", i, d.Kind)
		}
		if err := checkNonNegFloats(fmt.Sprintf("disks[%d]", i), d.ReadBytesPerS, d.WriteBytesPerS, d.IopsRead, d.IopsWrite, d.LatencyMSAvg); err != nil {
			return err
		}
		if d.QueueDepth != nil && *d.QueueDepth < 0 {
			return NewErr(CodeSnapshotMalformed, "disks[%d].queue_depth is negative", i)
		}
		if d.FreeBytes != nil && *d.FreeBytes < 0 {
			return NewErr(CodeSnapshotMalformed, "disks[%d].free_bytes is negative", i)
		}
		if d.TotalBytes != nil && *d.TotalBytes < 0 {
			return NewErr(CodeSnapshotMalformed, "disks[%d].total_bytes is negative", i)
		}
	}
	for i, n := range s.Net {
		if strings.TrimSpace(n.Iface) == "" {
			return NewErr(CodeSnapshotMalformed, "net[%d].iface is empty", i)
		}
		if err := checkNonNegFloats(fmt.Sprintf("net[%d]", i), n.RxBytesPerS, n.TxBytesPerS, n.RxPacketsPerS, n.TxPacketsPerS, n.RxDropsPerS, n.TxDropsPerS); err != nil {
			return err
		}
		if n.ErrorsTotal != nil && *n.ErrorsTotal < 0 {
			return NewErr(CodeSnapshotMalformed, "net[%d].errors_total is negative", i)
		}
	}
	for i, l := range s.Links {
		if !linkKinds[l.Kind] {
			return NewErr(CodeSnapshotMalformed, "links[%d].kind %q outside closed set", i, l.Kind)
		}
		if strings.TrimSpace(l.ID) == "" {
			return NewErr(CodeSnapshotMalformed, "links[%d].id is empty", i)
		}
		if !linkStates[l.State] {
			return NewErr(CodeSnapshotMalformed, "links[%d].state %q outside up|down|unknown", i, l.State)
		}
		if err := checkNonNegFloats(fmt.Sprintf("links[%d]", i), l.SpeedGBps, l.RxBytesPerS, l.TxBytesPerS); err != nil {
			return err
		}
	}
	for i, g := range s.GPUs {
		if strings.TrimSpace(g.UUID) == "" {
			return NewErr(CodeSnapshotMalformed, "gpus[%d].uuid is empty", i)
		}
		if err := checkPct(fmt.Sprintf("gpus[%d].util_pct", i), g.UtilPct); err != nil {
			return err
		}
		if g.MemTotalBytes != nil && *g.MemTotalBytes < 0 {
			return NewErr(CodeSnapshotMalformed, "gpus[%d].mem_total_bytes is negative", i)
		}
		if g.MemUsedBytes != nil && *g.MemUsedBytes < 0 {
			return NewErr(CodeSnapshotMalformed, "gpus[%d].mem_used_bytes is negative", i)
		}
		if err := checkNonNegFloats(fmt.Sprintf("gpus[%d]", i), g.PowerW, g.SMClockMHz, g.MaxSMClockMHz, g.PCIERxBytesPerS, g.PCIETxBytesPerS); err != nil {
			return err
		}
		if g.ECCUncorrectedTotal != nil && *g.ECCUncorrectedTotal < 0 {
			return NewErr(CodeSnapshotMalformed, "gpus[%d].ecc_uncorrected_total is negative", i)
		}
	}
	for i, t := range s.Tiers {
		if !tierKinds[t.Kind] {
			return NewErr(CodeSnapshotMalformed, "tiers[%d].kind %q outside vram|ram|ssd", i, t.Kind)
		}
		if strings.TrimSpace(t.Device) == "" {
			return NewErr(CodeSnapshotMalformed, "tiers[%d].device is empty", i)
		}
		if t.LayersStart < 0 || t.LayersEnd < t.LayersStart {
			return NewErr(CodeSnapshotMalformed, "tiers[%d] layer range invalid: [%d,%d]", i, t.LayersStart, t.LayersEnd)
		}
		if t.Bytes < 0 {
			return NewErr(CodeSnapshotMalformed, "tiers[%d].bytes is negative", i)
		}
		if t.HotnessRank < 0 {
			return NewErr(CodeSnapshotMalformed, "tiers[%d].hotness_rank is negative", i)
		}
		if t.TransferBytesPerS != nil && *t.TransferBytesPerS < 0 {
			return NewErr(CodeSnapshotMalformed, "tiers[%d].transfer_bytes_per_s is negative", i)
		}
	}
	for i, tr := range s.Transfers {
		if !tierKinds[TierKind(tr.FromTier)] || !tierKinds[TierKind(tr.ToTier)] {
			return NewErr(CodeSnapshotMalformed, "transfers[%d] tier %q/%q outside vram|ram|ssd", i, tr.FromTier, tr.ToTier)
		}
		if !reasons[tr.Reason] {
			return NewErr(CodeSnapshotMalformed, "transfers[%d].reason %q outside load|evict|pp_stage_move|unknown", i, tr.Reason)
		}
		if tr.Bytes < 0 {
			return NewErr(CodeSnapshotMalformed, "transfers[%d].bytes is negative", i)
		}
		if tr.DurationS < 0 {
			return NewErr(CodeSnapshotMalformed, "transfers[%d].duration_s is negative", i)
		}
		if tr.RateGiBS < 0 {
			return NewErr(CodeSnapshotMalformed, "transfers[%d].rate_gib_s is negative", i)
		}
	}
	return nil
}

func checkPct(name string, v *float64) error {
	if v != nil && (*v < 0 || *v > 100) {
		return NewErr(CodeSnapshotMalformed, "%s out of [0,100]: %v", name, *v)
	}
	return nil
}

func checkNonNegFloats(prefix string, vals ...*float64) error {
	for i, v := range vals {
		if v != nil && *v < 0 {
			return NewErr(CodeSnapshotMalformed, "%s field %d is negative", prefix, i)
		}
	}
	return nil
}

// UnreadCount counts the nil metric pointers across the five host/device
// families (cpu, dram, disks, net, links, gpus). Tiers and transfers are
// emitter-driven, not collectors, so an empty tier plane is not "unread".
// Identity and always-set scalar fields do not count.
func (s *Snapshot) UnreadCount() int {
	n := 0
	if s.CPU == nil {
		n += 4 // util, steal, iowait, load1
	} else {
		for _, p := range []*float64{s.CPU.UtilPct, s.CPU.StealPct, s.CPU.IOWaitPct, s.CPU.Load1} {
			if p == nil {
				n++
			}
		}
	}
	if s.DRAM == nil {
		n += 6 // total, used, available, faults, swap, bandwidth
	} else {
		n += nilsI(s.DRAM.TotalBytes, s.DRAM.UsedBytes, s.DRAM.AvailableBytes, s.DRAM.SwapUsedBytes)
		n += nilsF(s.DRAM.PageFaultsPerSec, s.DRAM.BandwidthGiBS)
	}
	for _, d := range s.Disks {
		n += nilsF(d.ReadBytesPerS, d.WriteBytesPerS, d.IopsRead, d.IopsWrite, d.LatencyMSAvg)
		n += nilsI(d.QueueDepth, d.FreeBytes, d.TotalBytes)
	}
	for _, x := range s.Net {
		n += nilsF(x.RxBytesPerS, x.TxBytesPerS, x.RxPacketsPerS, x.TxPacketsPerS, x.RxDropsPerS, x.TxDropsPerS)
		n += nilsI(x.ErrorsTotal)
	}
	for _, l := range s.Links {
		n += nilsF(l.SpeedGBps, l.RxBytesPerS, l.TxBytesPerS)
	}
	for _, g := range s.GPUs {
		n += nilsF(g.UtilPct, g.PowerW, g.SMClockMHz, g.MaxSMClockMHz, g.PCIERxBytesPerS, g.PCIETxBytesPerS)
		n += nilsI(g.MemTotalBytes, g.MemUsedBytes, g.ECCUncorrectedTotal)
	}
	return n
}

// nilsF / nilsI count nil metric pointers within one concrete type so the
// unread tally stays allocation-free and typed-nil-safe.
func nilsF(vals ...*float64) int {
	n := 0
	for _, v := range vals {
		if v == nil {
			n++
		}
	}
	return n
}

func nilsI(vals ...*int64) int {
	n := 0
	for _, v := range vals {
		if v == nil {
			n++
		}
	}
	return n
}

// NumericCounterCount counts the distinct numeric metric values present in
// this sample: every non-nil *float64/*int64 metric plus the always-set
// numerics (cores, interval_s, duration_seconds, tier/transfer numerics).
// This is the goal's climb metric: distinct observed numeric counters per
// node sample.
func (s *Snapshot) NumericCounterCount() int {
	n := 0
	countF := func(v *float64) {
		if v != nil {
			n++
		}
	}
	countI := func(v *int64) {
		if v != nil {
			n++
		}
	}
	if s.CPU != nil {
		n++ // cores
		n++ // interval_s
		countF(s.CPU.UtilPct)
		countF(s.CPU.StealPct)
		countF(s.CPU.IOWaitPct)
		countF(s.CPU.Load1)
	}
	if s.DRAM != nil {
		countI(s.DRAM.TotalBytes)
		countI(s.DRAM.UsedBytes)
		countI(s.DRAM.AvailableBytes)
		countF(s.DRAM.PageFaultsPerSec)
		countI(s.DRAM.SwapUsedBytes)
		countF(s.DRAM.BandwidthGiBS)
	}
	for _, d := range s.Disks {
		countF(d.ReadBytesPerS)
		countF(d.WriteBytesPerS)
		countF(d.IopsRead)
		countF(d.IopsWrite)
		countI(d.QueueDepth)
		countF(d.LatencyMSAvg)
		countI(d.FreeBytes)
		countI(d.TotalBytes)
	}
	for _, x := range s.Net {
		countF(x.RxBytesPerS)
		countF(x.TxBytesPerS)
		countF(x.RxPacketsPerS)
		countF(x.TxPacketsPerS)
		countF(x.RxDropsPerS)
		countF(x.TxDropsPerS)
		countI(x.ErrorsTotal)
	}
	for _, l := range s.Links {
		countF(l.SpeedGBps)
		countF(l.RxBytesPerS)
		countF(l.TxBytesPerS)
	}
	for _, g := range s.GPUs {
		countF(g.UtilPct)
		countI(g.MemTotalBytes)
		countI(g.MemUsedBytes)
		countF(g.PowerW)
		countF(g.SMClockMHz)
		countF(g.MaxSMClockMHz)
		countF(g.PCIERxBytesPerS)
		countF(g.PCIETxBytesPerS)
		countI(g.ECCUncorrectedTotal)
	}
	for _, t := range s.Tiers {
		n += 4 // layers_start, layers_end, bytes, hotness_rank
		countF(t.TransferBytesPerS)
	}
	for range s.Transfers {
		n += 3 // bytes, duration_s, rate_gib_s
	}
	return n
}
