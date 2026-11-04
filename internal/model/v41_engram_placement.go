package model

// v41_engram_placement.go — host-table placement/layout policy for a V4.1
// Engram RESIDENT host table.
//
// WHY THIS EXISTS. The verified row source (v41_engram_binding.go) and the
// bounded set-associative cache (v41_engram_stream.go) are file-backed: rows
// stream from checkpoint shards into a budgeted resident window, so no single
// decision ever charges the whole table against host memory. A resident host
// table is the opposite discipline: the entire bound extent stays in host RAM
// for the process lifetime, backed either by one shared mapping every rank
// reads in place or by one private shard per rank, and its lookup cost hinges
// on whether the kernel actually backed those pages with huge pages. This file
// adapts SGLang's Engram _HostTable placement discipline (sgl-project/sglang,
// python/sglang/srt/layers/engram.py:547-706 @ 3c6a6c8068658b88e40bcaa5a0bf48b9a94ebdba,
// Apache-2.0; ADAPT verdict): layout select with fail-closed validation of the
// two accepted spellings, admission accounting that charges the full resident
// copy count against an explicit host budget, and a huge-page-backing check
// parsed from smaps-format evidence.
//
// WHAT THIS IS NOT. No kernel work and no lookup all-reduce yet: selection and
// accounting only. The sharded layout records that lookups must reduce across
// ranks (LookupReduce); performing that reduction is a later leaf. Nothing here
// touches a real /proc/self/smaps or a real mapping — the probe parses
// smaps-FORMAT content so the rule is deterministic and testable on any host.

import (
	"errors"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// Sentinel errors for host-table placement. Callers match these with
// errors.Is; the wrapped messages carry the offending value.
var (
	// ErrV41EngramHostTableUnknownLayout refuses a layout spelling outside the
	// closed {shared, per_rank} vocabulary ("" is the shared default).
	ErrV41EngramHostTableUnknownLayout = errors.New("v41 engram host table: unknown layout")
	// ErrV41EngramHostTableGeometry refuses malformed admission inputs: a
	// non-positive table extent, a non-positive rank count, or byte math that
	// overflows int64.
	ErrV41EngramHostTableGeometry = errors.New("v41 engram host table: invalid geometry")
	// ErrV41EngramHostTableNotAdmitted refuses a selected layout whose full
	// resident charge exceeds the declared host budget.
	ErrV41EngramHostTableNotAdmitted = errors.New("v41 engram host table: not admitted under selected layout")
	// ErrV41EngramHugePageProbe fails closed when smaps-format content cannot
	// establish the VMA holding the probed address: no VMA contains it, or a
	// recognized field is malformed.
	ErrV41EngramHugePageProbe = errors.New("v41 engram host table: smaps probe failed")
	// ErrV41EngramHugePageBacking fails closed when the probed VMA has no
	// resident pages at all, so backing cannot be established.
	ErrV41EngramHugePageBacking = errors.New("v41 engram host table: no huge-page backing evidence")
)

// V41EngramHostTableLayout names how one resident Engram host table is backed.
type V41EngramHostTableLayout int

const (
	// V41EngramHostTableShared backs the whole table with ONE shared mapping
	// (upstream: memfd + MAP_SHARED, opened by every rank through rank 0's
	// descriptor). Each rank reads its own rows in place, so a lookup needs no
	// cross-rank reduction.
	V41EngramHostTableShared V41EngramHostTableLayout = iota
	// V41EngramHostTableSharded backs the table with one PRIVATE per-rank shard
	// (upstream: MAP_PRIVATE | MAP_ANONYMOUS per rank). A rank sees only its
	// shard, so a lookup must reduce across ranks.
	V41EngramHostTableSharded
)

// String renders the layout in its accepted spelling.
func (l V41EngramHostTableLayout) String() string {
	switch l {
	case V41EngramHostTableShared:
		return "shared"
	case V41EngramHostTableSharded:
		return "per_rank"
	default:
		return fmt.Sprintf("layout(%d)", int(l))
	}
}

// ParseV41EngramHostTableLayout selects the layout from its spelling. The
// vocabulary is closed: "" defaults to shared (the upstream default), exactly
// "shared" and "per_rank" parse, and anything else — including case variants
// and other layouts — is refused fail-closed.
func ParseV41EngramHostTableLayout(spelling string) (V41EngramHostTableLayout, error) {
	switch spelling {
	case "", "shared":
		return V41EngramHostTableShared, nil
	case "per_rank":
		return V41EngramHostTableSharded, nil
	}
	return 0, fmt.Errorf("%w: %q", ErrV41EngramHostTableUnknownLayout, spelling)
}

// V41EngramHostTableRequest is the admission input for one resident host
// table. TableBytes is the FULL resident extent (rows * row width), not a
// streaming window: that is what makes this accounting distinct from the
// file-backed row cache, which budgets sets*ways rows instead.
type V41EngramHostTableRequest struct {
	Layout     string // requested layout spelling; "" selects the shared default
	TableBytes int64  // full resident extent of one copy
	Ranks      int    // TP ranks that will read the table
	HostBudget int64  // host bytes available to charge the table
}

// V41EngramHostTablePlacement is the selected, accounted placement.
type V41EngramHostTablePlacement struct {
	Layout   V41EngramHostTableLayout
	Copies   int   // shared: 1; sharded: Ranks
	Resident int64 // Copies*TableBytes, charged against HostBudget
	// LookupReduce is true when a lookup must reduce across ranks because each
	// rank holds only its shard. Shared tables need no reduction.
	LookupReduce bool
}

// SelectV41EngramHostTablePlacement validates the request and accounts the
// resident charge fail-closed: the FULL copy count (one shared copy, or one
// private shard per rank) times the table extent must fit the declared host
// budget, or admission is refused. It performs no allocation and no syscall.
func SelectV41EngramHostTablePlacement(req V41EngramHostTableRequest) (V41EngramHostTablePlacement, error) {
	if req.TableBytes <= 0 || req.HostBudget < 0 {
		return V41EngramHostTablePlacement{}, fmt.Errorf("%w: table=%d budget=%d",
			ErrV41EngramHostTableGeometry, req.TableBytes, req.HostBudget)
	}
	if req.Ranks < 1 {
		return V41EngramHostTablePlacement{}, fmt.Errorf("%w: ranks=%d",
			ErrV41EngramHostTableGeometry, req.Ranks)
	}
	layout, err := ParseV41EngramHostTableLayout(req.Layout)
	if err != nil {
		return V41EngramHostTablePlacement{}, err
	}
	copies := 1
	if layout == V41EngramHostTableSharded {
		copies = req.Ranks
	}
	if int64(copies) > math.MaxInt64/req.TableBytes {
		return V41EngramHostTablePlacement{}, fmt.Errorf("%w: copies=%d x table=%d overflows",
			ErrV41EngramHostTableGeometry, copies, req.TableBytes)
	}
	resident := int64(copies) * req.TableBytes
	if resident > req.HostBudget {
		return V41EngramHostTablePlacement{}, fmt.Errorf("%w: layout=%s copies=%d resident=%d exceeds host budget=%d",
			ErrV41EngramHostTableNotAdmitted, layout.String(), copies, resident, req.HostBudget)
	}
	return V41EngramHostTablePlacement{
		Layout:       layout,
		Copies:       copies,
		Resident:     resident,
		LookupReduce: layout == V41EngramHostTableSharded,
	}, nil
}

// V41EngramHugePageMinRatio is the minimum fraction of a SHARDED table's
// resident pages that must be huge-backed before the layout stays admitted
// (upstream threshold: 98%). Checked in exact integer arithmetic as
// 100*HugeKB >= 98*MappedKB.
const V41EngramHugePageMinRatio = 0.98

// V41EngramHugePageProbe is the parsed smaps evidence for the VMA holding one
// host-table address. MappedKB is the VMA's Rss; HugeKB is the sum of its
// AnonHugePages, ShmemPmdMapped, and FilePmdMapped fields — the only evidence
// that the kernel really handed out huge pages for that region.
type V41EngramHugePageProbe struct {
	MappedKB int64
	HugeKB   int64
}

var v41EngramSmapsVMAPattern = regexp.MustCompile(`^([0-9a-f]+)-([0-9a-f]+) `)

// ProbeV41EngramHugePages parses /proc/self/smaps-format content and returns
// the probe for the VMA containing addr. It stops at the next VMA header after
// the containing one, so fields of later VMAs never leak into the result. It
// fails closed with ErrV41EngramHugePageProbe when no VMA contains addr or a
// recognized field does not parse.
func ProbeV41EngramHugePages(smaps string, addr uint64) (V41EngramHugePageProbe, error) {
	var probe V41EngramHugePageProbe
	inside := false
	for _, line := range strings.Split(smaps, "\n") {
		if m := v41EngramSmapsVMAPattern.FindStringSubmatch(line); m != nil {
			if inside {
				break // left the containing VMA
			}
			start, err1 := strconv.ParseUint(m[1], 16, 64)
			end, err2 := strconv.ParseUint(m[2], 16, 64)
			if err1 != nil || err2 != nil {
				return V41EngramHugePageProbe{}, fmt.Errorf("%w: malformed VMA header %q",
					ErrV41EngramHugePageProbe, line)
			}
			inside = start <= addr && addr < end
			continue
		}
		if !inside {
			continue
		}
		key, rest, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		fields := strings.Fields(rest)
		if len(fields) == 0 {
			return V41EngramHugePageProbe{}, fmt.Errorf("%w: empty value for %q",
				ErrV41EngramHugePageProbe, key)
		}
		value, err := strconv.ParseInt(fields[0], 10, 64)
		if err != nil {
			return V41EngramHugePageProbe{}, fmt.Errorf("%w: bad value %q for %q",
				ErrV41EngramHugePageProbe, fields[0], key)
		}
		switch key {
		case "Rss":
			probe.MappedKB = value
		case "AnonHugePages", "ShmemPmdMapped", "FilePmdMapped":
			probe.HugeKB += value
		}
	}
	if !inside {
		return V41EngramHugePageProbe{}, fmt.Errorf("%w: no VMA contains address 0x%x",
			ErrV41EngramHugePageProbe, addr)
	}
	return probe, nil
}

// Ratio reports HugeKB/MappedKB clamped to [0,1]; a probe with no mapped pages
// reports 0.
func (p V41EngramHugePageProbe) Ratio() float64 {
	if p.MappedKB <= 0 {
		return 0
	}
	r := float64(p.HugeKB) / float64(p.MappedKB)
	if r < 0 {
		return 0
	}
	if r > 1 {
		return 1
	}
	return r
}

// HugePageBacked applies the layout's huge-page rule to the probe:
//   - sharded: fail-closed — admitted only when at least
//     V41EngramHugePageMinRatio of the resident pages are huge-backed;
//   - shared: advisory — the kernel collapses shared memory only while
//     shmem_enabled allows it, so observed backing reports without refusal.
//
// A probe with no mapped pages is never backed: ErrV41EngramHugePageBacking.
func (p V41EngramHugePageProbe) HugePageBacked(layout V41EngramHostTableLayout) (bool, error) {
	if p.MappedKB <= 0 {
		return false, fmt.Errorf("%w: probed VMA has no resident pages", ErrV41EngramHugePageBacking)
	}
	if layout != V41EngramHostTableSharded {
		return true, nil
	}
	return 100*p.HugeKB >= 98*p.MappedKB, nil
}

// V41EngramHostTableRequestFromBinding derives the host-table admission
// request geometry from a verified artifact binding instead of
// caller-declared numbers: the binding's validated Size is exactly the full
// resident extent (rows * row width). This is the seam between the verified
// file-backed row source and the resident host table's admission accounting.
func V41EngramHostTableRequestFromBinding(binding V41EngramArtifactBinding, layout string, ranks int, hostBudget int64) (V41EngramHostTableRequest, error) {
	size, err := binding.HostTableExtent()
	if err != nil {
		return V41EngramHostTableRequest{}, fmt.Errorf("host table request from binding: %w", err)
	}
	return V41EngramHostTableRequest{Layout: layout, TableBytes: size, Ranks: ranks, HostBudget: hostBudget}, nil
}
