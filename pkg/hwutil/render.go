package hwutil

// Bounded readers over one Snapshot. RenderJSON preserves null-on-error on
// the wire; RenderText is a bounded drill-down with a hard line cap.

import (
	"encoding/json"
	"fmt"
	"strings"
)

// DrillLevel selects which sections RenderText expands.
type DrillLevel string

const (
	DrillNode   DrillLevel = "node"   // summary lines only
	DrillDevice DrillLevel = "device" // node + cpu/dram/disk/net/gpu detail
	DrillLink   DrillLevel = "link"   // node + host fabric links
	DrillTier   DrillLevel = "tier"   // node + tier residency and transfers
)

// ParseDrillLevel validates the closed drill vocabulary.
func ParseDrillLevel(s string) (DrillLevel, error) {
	switch DrillLevel(s) {
	case DrillNode, DrillDevice, DrillLink, DrillTier:
		return DrillLevel(s), nil
	default:
		return "", fmt.Errorf("hwutil: drill level %q outside node|device|link|tier", s)
	}
}

// MaxTextLines is the hard text output cap; explicit caps clamp to it.
const MaxTextLines = 200

// TextOptions tunes RenderText.
type TextOptions struct {
	Drill    DrillLevel
	MaxLines int
}

// normalize fills defaults and clamps the cap into [1, MaxTextLines].
func (o TextOptions) normalize() TextOptions {
	if o.Drill == "" {
		o.Drill = DrillNode
	}
	if o.MaxLines <= 0 || o.MaxLines > MaxTextLines {
		o.MaxLines = MaxTextLines
	}
	return o
}

// RenderJSON renders the frozen wire shape. Nil metric pointers stay null
// (never dropped, never zero-filled); empty array families render as [] so
// the document always carries arrays where the schema has arrays.
func RenderJSON(s *Snapshot) ([]byte, error) {
	if s == nil {
		return json.Marshal(nil)
	}
	doc := *s
	if doc.Disks == nil {
		doc.Disks = []Disk{}
	}
	if doc.Net == nil {
		doc.Net = []NetIfc{}
	}
	if doc.Links == nil {
		doc.Links = []Link{}
	}
	if doc.GPUs == nil {
		doc.GPUs = []GPU{}
	}
	if doc.Tiers == nil {
		doc.Tiers = []TierRow{}
	}
	if doc.Transfers == nil {
		doc.Transfers = []TransferRow{}
	}
	return json.Marshal(doc)
}

// RenderText renders a bounded human readout. The result never exceeds
// opts.MaxLines (hard-capped at MaxTextLines); when truncated, the final
// line names the cut and counts toward the cap.
func RenderText(s *Snapshot, opts TextOptions) (string, error) {
	if s == nil {
		return "", fmt.Errorf("hwutil: nil snapshot")
	}
	o := opts.normalize()
	lines := make([]string, 0, 32)
	add := func(format string, args ...any) {
		lines = append(lines, fmt.Sprintf(format, args...))
	}
	add("%s %s/%s @ %s (%.3fs)%s", s.Host, s.Os, s.Arch, s.At.Format("2006-01-02T15:04:05Z"), s.DurationSeconds, errTag(s))
	nodeSummary(s, add)
	switch o.Drill {
	case DrillDevice:
		deviceDetail(s, add)
	case DrillLink:
		linkDetail(s, add)
	case DrillTier:
		tierDetail(s, add)
	}
	if len(lines) > o.MaxLines {
		kept := lines[:o.MaxLines-1]
		kept = append(kept, fmt.Sprintf("(truncated: showing %d of %d lines)", o.MaxLines-1, len(lines)))
		lines = kept
	}
	return strings.Join(lines, "\n"), nil
}

func errTag(s *Snapshot) string {
	if s.Error == "" {
		return ""
	}
	return " [" + s.Error + "]"
}

func nodeSummary(s *Snapshot, add func(string, ...any)) {
	if c := s.CPU; c != nil {
		add("cpu: %d cores, util %s%%, iowait %s%%, steal %s%%, load1 %s",
			c.Cores, pct(c.UtilPct), pct(c.IOWaitPct), pct(c.StealPct), num(c.Load1))
	} else {
		add("cpu: unread")
	}
	if d := s.DRAM; d != nil {
		add("dram: total %s, used %s, available %s, faults/s %s, swap %s",
			bytes(d.TotalBytes), bytes(d.UsedBytes), bytes(d.AvailableBytes), num(d.PageFaultsPerSec), bytes(d.SwapUsedBytes))
	} else {
		add("dram: unread")
	}
	if len(s.Disks) > 0 {
		add("disks: %d device(s)", len(s.Disks))
	} else {
		add("disks: none reported")
	}
	if len(s.Net) > 0 {
		add("net: %d interface(s)", len(s.Net))
	} else {
		add("net: none reported")
	}
	if len(s.Links) > 0 {
		add("links: %d", len(s.Links))
	}
	if len(s.GPUs) > 0 {
		parts := make([]string, 0, len(s.GPUs))
		for _, g := range s.GPUs {
			parts = append(parts, fmt.Sprintf("%s util %s%%", shortName(g.Name), pct(g.UtilPct)))
		}
		add("gpus: %s", strings.Join(parts, ", "))
	} else {
		add("gpus: none")
	}
	if len(s.Tiers) > 0 {
		add("tiers: %d row(s), %d transfer(s)", len(s.Tiers), len(s.Transfers))
	}
}

func deviceDetail(s *Snapshot, add func(string, ...any)) {
	for _, d := range s.Disks {
		add("  disk %-14s %-8s r %s w %s iops r/w %s/%s qd %s lat %sms free %s/total %s",
			d.Name, d.Kind, bps(d.ReadBytesPerS), bps(d.WriteBytesPerS),
			num(d.IopsRead), num(d.IopsWrite), numI(d.QueueDepth), num(d.LatencyMSAvg),
			bytes(d.FreeBytes), bytes(d.TotalBytes))
	}
	for _, x := range s.Net {
		add("  net %-14s rx %s tx %s pkts r/w %s/%s drops r/w %s/%s errs %s",
			x.Iface, bps(x.RxBytesPerS), bps(x.TxBytesPerS),
			num(x.RxPacketsPerS), num(x.TxPacketsPerS),
			num(x.RxDropsPerS), num(x.TxDropsPerS), numI(x.ErrorsTotal))
	}
	for _, g := range s.GPUs {
		add("  gpu %-20s util %s%% mem %s/%s power %sW smclk %s/%sMHz pcie rx/tx %s/%s ecc_uncorr %s throttle %s",
			shortName(g.Name), pct(g.UtilPct), bytes(g.MemUsedBytes), bytes(g.MemTotalBytes),
			num(g.PowerW), num(g.SMClockMHz), num(g.MaxSMClockMHz),
			bps(g.PCIERxBytesPerS), bps(g.PCIETxBytesPerS), numI(g.ECCUncorrectedTotal),
			strings.Join(g.ThrottleReasons, ","))
	}
}

func linkDetail(s *Snapshot, add func(string, ...any)) {
	if len(s.Links) == 0 {
		add("  links: none reported")
		return
	}
	for _, l := range s.Links {
		add("  link %-12s %-10s speed %sGbps state %s", l.Kind, l.ID, num(l.SpeedGBps), l.State)
	}
}

func tierDetail(s *Snapshot, add func(string, ...any)) {
	if len(s.Tiers) == 0 && len(s.Transfers) == 0 {
		add("  tiers: nothing recorded")
		return
	}
	for _, t := range s.Tiers {
		last := "never"
		if t.LastTransferAt != nil {
			last = t.LastTransferAt.UTC().Format("2006-01-02T15:04:05Z")
		}
		add("  tier %-4s %-12s model=%s layers[%d..%d] %s rank %d last %s rate %s",
			t.Kind, t.Device, t.ModelID, t.LayersStart, t.LayersEnd,
			bytes64(t.Bytes), t.HotnessRank, last, num(t.TransferBytesPerS))
	}
	for _, tr := range s.Transfers {
		add("  transfer %s -> %s %s in %.3fs (%s GiB/s) reason=%s",
			tr.FromTier, tr.ToTier, bytes64(tr.Bytes), tr.DurationS, numF(tr.RateGiBS), tr.Reason)
	}
}

func shortName(name string) string {
	if i := strings.IndexAny(name, " ("); i > 0 {
		return name[:i]
	}
	return name
}

func pct(v *float64) string { return num(v) }

func num(v *float64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%.2f", *v)
}

func numF(v float64) string { return fmt.Sprintf("%.2f", v) }

func numI(v *int64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%d", *v)
}

func bytes(v *int64) string {
	if v == nil {
		return "-"
	}
	return humanBytes(float64(*v))
}

func bytes64(v int64) string { return humanBytes(float64(v)) }

func bps(v *float64) string {
	if v == nil {
		return "-"
	}
	return humanBytes(*v) + "/s"
}

func humanBytes(v float64) string {
	const unit = 1024
	if v < unit {
		return fmt.Sprintf("%.0fB", v)
	}
	div, exp := float64(unit), 0
	for n := v / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f%c", v/div, "KMGTPE"[exp])
}
