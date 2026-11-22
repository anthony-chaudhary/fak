// DIRECT-PORT of NVIDIA/nccl src/ras/diagnostics.h, src/ras/diagnostics.cc
// (dispatch table) and src/ras/diagnostics_gpu.cc (GPU model, ECC, NVLink
// summarize rules) at 3e46c089f4ae4732722a419eb6096aead7e47a30.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 (upstream LICENSE.txt).
// Ported: the closed RAS diagnostics check-ID vocabulary in upstream order,
// with each check pairing a per-node collect with a cross-node summarize that
// reports consensus against the lowest-sorted node or names the outliers.
// fak adds: the catalog is a derived surface over existing Snapshot data (no
// new wire fields), ECC falls back to the core uncorrected total when the
// volatile counter is absent, NVLink also faults on CRC/replay error totals,
// and every check without an hwutil data source reports RASDiagUnobserved,
// never ok.

package hwutil

import (
	"fmt"
	"sort"
	"strings"
)

// RASDiagCheckID mirrors rasDiagnosticsCheckId; the ordinals are the upstream
// wire values, so they must not be renumbered.
type RASDiagCheckID int

const (
	RASDiagGPUModel            RASDiagCheckID = 0
	RASDiagCUDADriverVersion   RASDiagCheckID = 1
	RASDiagECC                 RASDiagCheckID = 2
	RASDiagNVLink              RASDiagCheckID = 3
	RASDiagNCCLEnv             RASDiagCheckID = 4
	RASDiagRDMATopo            RASDiagCheckID = 5
	RASDiagIOMMUMode           RASDiagCheckID = 6
	RASDiagATS                 RASDiagCheckID = 7
	RASDiagXIDSXID             RASDiagCheckID = 8
	RASDiagNVIDIADriverVersion RASDiagCheckID = 9
	RASDiagPaths               RASDiagCheckID = 10
	// RASDiagCheckCount is the sentinel; it is not a check.
	RASDiagCheckCount RASDiagCheckID = 11
)

var rasDiagNames = [RASDiagCheckCount]string{
	"gpu_model", "cuda_driver_version", "ecc", "nvlink", "nccl_env",
	"rdma_topo", "iommu_mode", "ats", "xid_sxid", "nvidia_driver_version", "paths",
}

func (id RASDiagCheckID) String() string {
	if id < 0 || id >= RASDiagCheckCount {
		return fmt.Sprintf("rasdiag(%d)", int(id))
	}
	return rasDiagNames[id]
}

// RASDiagStatus is the closed verdict vocabulary for one record or summary.
type RASDiagStatus string

const (
	RASDiagOK       RASDiagStatus = "ok"
	RASDiagMismatch RASDiagStatus = "mismatch"
	RASDiagFault    RASDiagStatus = "fault"
	// RASDiagUnobserved means no data was read; it is never a pass.
	RASDiagUnobserved RASDiagStatus = "unobserved"
)

// RASDiagRecord is one node's flat collect result for one check.
type RASDiagRecord struct {
	Check  RASDiagCheckID `json:"check"`
	Host   string         `json:"host"`
	Value  string         `json:"value"`
	Status RASDiagStatus  `json:"status"`
}

// RASDiagSummary is the cross-node verdict for one check. Reference is the
// value of the lowest-sorted observed host, matching upstream's use of the
// lowest rank as the expected value.
type RASDiagSummary struct {
	Check      RASDiagCheckID `json:"check"`
	Status     RASDiagStatus  `json:"status"`
	Reference  string         `json:"reference"`
	Outliers   []string       `json:"outliers"`
	Unobserved []string       `json:"unobserved"`
}

// RASDiagOptions carries upstream's tunables.
type RASDiagOptions struct {
	// ECCCorrectedThreshold mirrors NCCL_DIAGNOSTICS_ECC_THRESHOLD: corrected
	// errors at or above it fault the node; 0 disables the corrected check.
	ECCCorrectedThreshold int64
}

// RASDiagCheck pairs a per-node collect with a cross-node summarize.
type RASDiagCheck struct {
	ID        RASDiagCheckID
	Collect   func(s *Snapshot, opt RASDiagOptions) RASDiagRecord
	Summarize func(id RASDiagCheckID, recs []RASDiagRecord) RASDiagSummary
}

// RASDiagCatalog returns the closed check table in upstream dispatch order,
// one entry per ID; entry i always has ID i.
func RASDiagCatalog() []RASDiagCheck {
	cat := make([]RASDiagCheck, RASDiagCheckCount)
	for id := RASDiagCheckID(0); id < RASDiagCheckCount; id++ {
		cat[id] = RASDiagCheck{ID: id, Collect: collectUnobserved(id), Summarize: summarizeAgainstReference}
	}
	cat[RASDiagGPUModel].Collect = collectGPUModel
	cat[RASDiagECC].Collect = collectECC
	cat[RASDiagECC].Summarize = summarizeFaults
	cat[RASDiagNVLink].Collect = collectNVLink
	return cat
}

// RunRASDiag collects every check on every snapshot (sorted by host so the
// reference node is deterministic) and summarizes each check across nodes.
func RunRASDiag(snaps []*Snapshot, opt RASDiagOptions) []RASDiagSummary {
	sorted := make([]*Snapshot, 0, len(snaps))
	for _, s := range snaps {
		if s != nil {
			sorted = append(sorted, s)
		}
	}
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Host < sorted[j].Host })
	cat := RASDiagCatalog()
	out := make([]RASDiagSummary, 0, len(cat))
	for _, c := range cat {
		recs := make([]RASDiagRecord, 0, len(sorted))
		for _, s := range sorted {
			r := c.Collect(s, opt)
			r.Check, r.Host = c.ID, s.Host
			recs = append(recs, r)
		}
		out = append(out, c.Summarize(c.ID, recs))
	}
	return out
}

func collectUnobserved(id RASDiagCheckID) func(*Snapshot, RASDiagOptions) RASDiagRecord {
	return func(s *Snapshot, _ RASDiagOptions) RASDiagRecord {
		return RASDiagRecord{Check: id, Host: s.Host, Status: RASDiagUnobserved}
	}
}

const rasDiagGPUModelUnknown = "unknown"

// collectGPUModel records GPU count and the sorted model list; upstream
// compares both against the reference rank.
func collectGPUModel(s *Snapshot, _ RASDiagOptions) RASDiagRecord {
	r := RASDiagRecord{Check: RASDiagGPUModel, Host: s.Host}
	if len(s.GPUs) == 0 {
		r.Status = RASDiagUnobserved
		return r
	}
	names := make([]string, 0, len(s.GPUs))
	known := false
	for _, g := range s.GPUs {
		n := strings.TrimSpace(g.Name)
		if n == "" {
			n = rasDiagGPUModelUnknown
		} else {
			known = true
		}
		names = append(names, n)
	}
	sort.Strings(names)
	r.Value = fmt.Sprintf("%dx %s", len(names), strings.Join(names, ","))
	r.Status = RASDiagOK
	if !known {
		// Count alone is still comparable, but an all-unknown model list must
		// not read as a model match.
		r.Status = RASDiagUnobserved
	}
	return r
}

// collectECC sums uncorrected volatile errors across the node's GPUs, as
// upstream does across SRAM and DRAM, and checks corrected errors only when a
// threshold is set.
func collectECC(s *Snapshot, opt RASDiagOptions) RASDiagRecord {
	r := RASDiagRecord{Check: RASDiagECC, Host: s.Host, Status: RASDiagUnobserved}
	var unc, cor int64
	avail, corAvail := false, false
	for _, g := range s.GPUs {
		u := g.ECCUncorrectableVolatile
		if u == nil {
			u = g.ECCUncorrectedTotal
		}
		if u != nil {
			avail = true
			unc += *u
		}
		if g.ECCCorrectableVolatile != nil {
			corAvail = true
			cor += *g.ECCCorrectableVolatile
		}
	}
	if !avail {
		return r
	}
	r.Value = fmt.Sprintf("uncorrected=%d", unc)
	if corAvail {
		r.Value += fmt.Sprintf(" corrected=%d", cor)
	}
	r.Status = RASDiagOK
	if unc > 0 || (opt.ECCCorrectedThreshold > 0 && cor >= opt.ECCCorrectedThreshold) {
		r.Status = RASDiagFault
	}
	return r
}

// collectNVLink records link count, inactive count and the speed range from
// the node's NVLink fabric links; a link is active only when up at a known
// non-zero speed, as upstream requires.
func collectNVLink(s *Snapshot, _ RASDiagOptions) RASDiagRecord {
	r := RASDiagRecord{Check: RASDiagNVLink, Host: s.Host, Status: RASDiagUnobserved}
	n, inactive, unreadable := 0, 0, 0
	var lo, hi float64
	for _, l := range s.Links {
		if l.Kind != LinkNVLink {
			continue
		}
		n++
		if l.State != LinkUp && l.State != LinkDown {
			unreadable++
		}
		if l.State != LinkUp || l.SpeedGBps == nil || *l.SpeedGBps <= 0 {
			inactive++
			continue
		}
		v := *l.SpeedGBps
		if lo == 0 || v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	// Upstream treats a device whose every link state is unreadable as having
	// no NVLink rather than as all-inactive.
	if unreadable == n {
		n, inactive = 0, 0
	}
	var errs int64
	errAvail := false
	for _, g := range s.GPUs {
		for _, p := range []*int64{g.NVLinkCRCErrorTotal, g.NVLinkReplayErrorTotal} {
			if p != nil {
				errAvail = true
				errs += *p
			}
		}
	}
	if n == 0 && !errAvail {
		return r
	}
	r.Value = fmt.Sprintf("links=%d speed_gbps=%g", n, hi)
	r.Status = RASDiagOK
	if inactive > 0 || lo != hi || errs > 0 {
		r.Status = RASDiagFault
	}
	return r
}

// summarizeAgainstReference implements upstream's consistency rule: the
// lowest-sorted observed node is the reference and every node whose value
// differs is an outlier. A node-local fault also lands in Outliers.
func summarizeAgainstReference(id RASDiagCheckID, recs []RASDiagRecord) RASDiagSummary {
	sum := RASDiagSummary{Check: id, Status: RASDiagUnobserved}
	haveRef, mismatch, fault := false, false, false
	for _, r := range recs {
		if r.Status == RASDiagUnobserved {
			sum.Unobserved = append(sum.Unobserved, r.Host)
			continue
		}
		if !haveRef {
			sum.Reference, haveRef = r.Value, true
		}
		switch {
		case r.Status == RASDiagFault:
			fault = true
			sum.Outliers = append(sum.Outliers, r.Host)
		case r.Value != sum.Reference:
			mismatch = true
			sum.Outliers = append(sum.Outliers, r.Host)
		}
	}
	switch {
	case !haveRef:
	case fault:
		sum.Status = RASDiagFault
	case mismatch:
		sum.Status = RASDiagMismatch
	default:
		sum.Status = RASDiagOK
	}
	return sum
}

// summarizeFaults is for counter checks such as ECC, where per-node values
// legitimately differ and only nodes over the fault line are outliers.
func summarizeFaults(id RASDiagCheckID, recs []RASDiagRecord) RASDiagSummary {
	sum := RASDiagSummary{Check: id, Status: RASDiagUnobserved}
	observed := false
	for _, r := range recs {
		switch r.Status {
		case RASDiagUnobserved:
			sum.Unobserved = append(sum.Unobserved, r.Host)
		case RASDiagFault:
			observed = true
			sum.Outliers = append(sum.Outliers, r.Host)
		default:
			observed = true
		}
	}
	switch {
	case !observed:
	case len(sum.Outliers) > 0:
		sum.Status = RASDiagFault
	default:
		sum.Status = RASDiagOK
	}
	return sum
}
