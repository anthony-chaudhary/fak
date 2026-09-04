package alloc

import (
	"fmt"
	"strings"
)

// MemoryMapConfig configures the memory map output.
type MemoryMapConfig struct {
	NumShards      int
	ModelPageBytes uint64
}

// MemoryMap renders a human-readable slab memory distribution table.
func MemoryMap(a Allocator, cfg MemoryMapConfig) string {
	regions := a.Regions()
	numClasses := a.NumClasses()
	if numClasses == 0 || len(regions) == 0 {
		return ""
	}

	// Collect per-class info
	type classInfo struct {
		size         uint64
		regionBytes  uint64
		slots        uint64
		isModel      bool   // **
		isDerivative bool   // *
		divFactor    int    // /2, /4, /8
		hugepage     string // "H" for MAP_HUGETLB, "T" for THP, "" for regular
	}

	classes := make([]classInfo, numClasses)
	var totalRegionBytes uint64
	for i := 0; i < numClasses; i++ {
		r := regions[i]
		ci := classInfo{
			size:        r.SlotSize,
			regionBytes: r.Region.Size(),
			slots:       r.Region.Size() / r.SlotSize,
		}
		switch {
		case r.Region.GotHugePages():
			ci.hugepage = "H"
		case r.Region.THPHinted():
			ci.hugepage = "T"
		}
		totalRegionBytes += ci.regionBytes
		classes[i] = ci
	}

	// Classify model / derivative
	if cfg.ModelPageBytes > 0 {
		mpb := cfg.ModelPageBytes
		derivs := map[uint64]int{}
		if v := mpb / 2; v > 0 {
			derivs[v] = 2
		}
		if v := mpb / 4; v > 0 {
			derivs[v] = 4
		}
		if v := mpb / 8; v > 0 {
			derivs[v] = 8
		}
		for i := range classes {
			if classes[i].size == mpb {
				classes[i].isModel = true
			} else if d, ok := derivs[classes[i].size]; ok {
				classes[i].isDerivative = true
				classes[i].divFactor = d
			}
		}
	}

	ns := cfg.NumShards
	if ns < 1 {
		ns = 1
	}
	perShardBytes := totalRegionBytes

	// Build rows
	type row struct {
		label     string
		marker    string // "**", "*", ""
		regionStr string // per-shard region size
		slotsStr  string // slots/shard
		totalStr  string // total slots across shards
		fitStr    string // worst-case overhead for values landing in this class
		pctMem    float64
		bar       string
	}

	type classFit struct {
		prevSize    uint64
		overheadPct float64
	}
	fits := make([]classFit, len(classes))
	var largestGapFrom, largestGapTo uint64
	var largestGapPct float64
	for i := range classes {
		if i == 0 {
			fits[i] = classFit{prevSize: 0, overheadPct: 0}
		} else {
			prev := classes[i-1].size
			fits[i].prevSize = prev
			if classes[i].size > prev {
				overhead := float64(classes[i].size-prev) / float64(classes[i].size) * 100
				fits[i].overheadPct = overhead
				if overhead > largestGapPct {
					largestGapPct = overhead
					largestGapFrom = prev
					largestGapTo = classes[i].size
				}
			}
		}
	}

	var rows []row

	// Group: <64K classes
	var smallRegion uint64
	var smallSlots uint64
	var smallCount int
	for i := range classes {
		if classes[i].size < 65536 {
			smallRegion += classes[i].regionBytes
			smallSlots += classes[i].slots
			smallCount++
		}
	}
	if smallCount > 0 {
		pct := float64(smallRegion) / float64(totalRegionBytes) * 100
		rows = append(rows, row{
			label:     fmt.Sprintf("<64K      (%d)", smallCount),
			regionStr: fmtSize(smallRegion),
			slotsStr:  fmtSlots(smallSlots),
			totalStr:  fmtSlots(smallSlots * uint64(ns)),
			pctMem:    pct,
		})
	}

	// Group: 64K-256K classes
	var medRegion uint64
	var medSlots uint64
	var medCount int
	for i := range classes {
		if classes[i].size >= 65536 && classes[i].size <= 262144 {
			medRegion += classes[i].regionBytes
			medSlots += classes[i].slots
			medCount++
		}
	}
	if medCount > 0 {
		pct := float64(medRegion) / float64(totalRegionBytes) * 100
		rows = append(rows, row{
			label:     fmt.Sprintf("64K-256K   (%d)", medCount),
			regionStr: fmtSize(medRegion),
			slotsStr:  fmtSlots(medSlots),
			totalStr:  fmtSlots(medSlots * uint64(ns)),
			pctMem:    pct,
		})
	}

	// Individual rows for classes >= 512K (after 256K)
	type largeClass struct {
		idx int
		ci  classInfo
	}
	var larges []largeClass
	for i := range classes {
		if classes[i].size > 262144 {
			larges = append(larges, largeClass{idx: i, ci: classes[i]})
		}
	}

	i := 0
	for i < len(larges) {
		lc := larges[i]
		if lc.ci.isModel || lc.ci.isDerivative {
			marker := ""
			label := fmtSize(lc.ci.size)
			if lc.ci.isModel {
				marker = "**"
			} else if lc.ci.isDerivative {
				marker = " *"
				label = fmt.Sprintf("%s   /%d", fmtSize(lc.ci.size), lc.ci.divFactor)
			}
			pct := float64(lc.ci.regionBytes) / float64(totalRegionBytes) * 100
			fitS := ""
			if fits[lc.idx].overheadPct > 0 {
				fitS = fmt.Sprintf("%.0f%%", fits[lc.idx].overheadPct)
			}
			rows = append(rows, row{
				label:     label,
				marker:    marker,
				regionStr: fmtSize(lc.ci.regionBytes),
				slotsStr:  fmtSlots(lc.ci.slots),
				totalStr:  fmtSlots(lc.ci.slots * uint64(ns)),
				fitStr:    fitS,
				pctMem:    pct,
			})
			i++
			continue
		}

		pct := float64(lc.ci.regionBytes) / float64(totalRegionBytes) * 100
		if pct < 1.5 {
			j := i
			var rangeRegion uint64
			var rangeSlots uint64
			for j < len(larges) && !larges[j].ci.isModel && !larges[j].ci.isDerivative {
				p := float64(larges[j].ci.regionBytes) / float64(totalRegionBytes) * 100
				if p >= 1.5 {
					break
				}
				rangeRegion += larges[j].ci.regionBytes
				rangeSlots += larges[j].ci.slots
				j++
			}
			count := j - i
			if count > 1 {
				rangePct := float64(rangeRegion) / float64(totalRegionBytes) * 100
				lo := fmtSizeShort(larges[i].ci.size)
				hi := fmtSizeShort(larges[j-1].ci.size)
				rows = append(rows, row{
					label:     fmt.Sprintf("%s-%s    (%d)", lo, hi, count),
					regionStr: fmtSize(rangeRegion),
					slotsStr:  fmtSlots(rangeSlots),
					totalStr:  fmtSlots(rangeSlots * uint64(ns)),
					pctMem:    rangePct,
				})
				i = j
			} else {
				fitS := ""
				if fits[lc.idx].overheadPct > 0 {
					fitS = fmt.Sprintf("%.0f%%", fits[lc.idx].overheadPct)
				}
				rows = append(rows, row{
					label:     fmtSize(lc.ci.size),
					regionStr: fmtSize(lc.ci.regionBytes),
					slotsStr:  fmtSlots(lc.ci.slots),
					totalStr:  fmtSlots(lc.ci.slots * uint64(ns)),
					fitStr:    fitS,
					pctMem:    pct,
				})
				i++
			}
		} else {
			fitS := ""
			if fits[lc.idx].overheadPct > 0 {
				fitS = fmt.Sprintf("%.0f%%", fits[lc.idx].overheadPct)
			}
			rows = append(rows, row{
				label:     fmtSize(lc.ci.size),
				regionStr: fmtSize(lc.ci.regionBytes),
				slotsStr:  fmtSlots(lc.ci.slots),
				totalStr:  fmtSlots(lc.ci.slots * uint64(ns)),
				fitStr:    fitS,
				pctMem:    pct,
			})
			i++
		}
	}

	var maxPct float64
	for _, r := range rows {
		if r.pctMem > maxPct {
			maxPct = r.pctMem
		}
	}
	if maxPct > 0 {
		for i := range rows {
			n := int(rows[i].pctMem / maxPct * 30)
			if n < 1 && rows[i].pctMem > 0.3 {
				n = 1
			}
			rows[i].bar = strings.Repeat("=", n)
		}
	}

	var b strings.Builder
	totalGB := float64(totalRegionBytes) * float64(ns) / (1 << 30)
	perShardGB := float64(perShardBytes) / (1 << 30)

	sep := strings.Repeat("\u2500", 80)

	fmt.Fprintf(&b, "slab memory map  (%.1f GB/shard \u00d7 %d shards = %.1f GB total, %d classes)\n",
		perShardGB, ns, totalGB, numClasses)
	b.WriteString(sep + "\n")
	fmt.Fprintf(&b, "%-15s %9s %15s %9s  %%mem  fit\n", "class", "region", "slots/shard", "total")
	b.WriteString(sep + "\n")

	for _, r := range rows {
		prefix := "  "
		if r.marker == "**" {
			prefix = "**"
		} else if r.marker == " *" {
			prefix = " *"
		}
		fmt.Fprintf(&b, "%s%-13s %9s %15s %9s %5.1f  %-4s %s\n",
			prefix, r.label, r.regionStr, r.slotsStr, r.totalStr, r.pctMem, r.fitStr, r.bar)
	}

	b.WriteString(sep + "\n")
	if cfg.ModelPageBytes > 0 {
		fmt.Fprintf(&b, "** model page (%s)   * derivative (/2, /4, /8)\n", fmtSize(cfg.ModelPageBytes))
	}
	if largestGapPct > 0 {
		fmt.Fprintf(&b, "largest gap: %s\u2192%s (%.0f%% overhead)\n",
			fmtSize(largestGapFrom), fmtSize(largestGapTo), largestGapPct)
	}

	return b.String()
}

func fmtSize(b uint64) string {
	switch {
	case b >= 10*1024*1024*1024:
		return fmt.Sprintf("%.0f GB", float64(b)/(1<<30))
	case b >= 1024*1024*1024:
		return fmt.Sprintf("%.2f GB", float64(b)/(1<<30))
	case b >= 10*1024*1024:
		return fmt.Sprintf("%.0f MB", float64(b)/(1<<20))
	case b >= 1024*1024:
		return fmt.Sprintf("%.2f MB", float64(b)/(1<<20))
	case b >= 1024:
		return fmt.Sprintf("%.0f KB", float64(b)/1024)
	default:
		return fmt.Sprintf("%d B", b)
	}
}

func fmtSizeShort(b uint64) string {
	switch {
	case b >= 1024*1024 && b%(1024*1024) == 0:
		return fmt.Sprintf("%dM", b/(1024*1024))
	case b >= 1024 && b%1024 == 0:
		return fmt.Sprintf("%dK", b/1024)
	default:
		return fmt.Sprintf("%d", b)
	}
}

func fmtSlots(n uint64) string {
	s := fmt.Sprintf("%d", n)
	if len(s) <= 3 {
		return s
	}
	var b strings.Builder
	start := len(s) % 3
	if start == 0 {
		start = 3
	}
	b.WriteString(s[:start])
	for i := start; i < len(s); i += 3 {
		b.WriteByte(',')
		b.WriteString(s[i : i+3])
	}
	return b.String()
}
