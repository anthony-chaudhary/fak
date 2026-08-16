package main

// fak doctor serve — the inference-hardware serve-readiness doctor (ktransformers
// `kt doctor` borrow, sibling epic #3900). It answers the operator's pre-flight
// question "will this model serve here, and optimally?" as a small table of
// green/yellow/red rows, each carrying an actionable remediation hint:
//
//   - cpu-isa       : the CPU SIMD tier the decode kernel will run on
//                     (AMX/AVX-512 > AVX2 > scalar; NEON/ASIMD on arm64).
//   - model-fit     : does the target model's resident weight size fit free
//                     device/host memory with headroom for KV + activations?
//   - numa-topology : the online NUMA node layout + a placement hint.
//
// The classification is a PURE fold over an injected host profile
// (serveHostFacts) so it is unit-witnessable with no GPU and no live hardware —
// the same posture as internal/kvbudget. The live host probe (probeServeHost) is
// a thin, best-effort seam over runtime + the compute package's NUMA accessor;
// tests never touch it. Read-only, off the hot path, no session or gateway.
//
// Exit 0 = ready (no red rows), 1 = at least one Unready (red) row, 2 = usage
// error — so `fak doctor serve --json` also composes as a serve-side CI gate.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"
	"runtime"
	"strings"

	"golang.org/x/sys/cpu"
)

// sevFail is the red rung of the serve-readiness table (doctor.go already defines
// sevOK/sevWarn for the green/yellow rungs, which this file reuses).
const sevFail = "fail"

// serveHostFacts is the injected host profile the readiness rows classify. Every
// field is a plain measured fact so the classification stays pure and tests can
// drive any green/yellow/red combination without real hardware.
type serveHostFacts struct {
	Arch       string  `json:"arch"`        // GOARCH: "amd64", "arm64", …
	ISA        string  `json:"isa"`         // best detected SIMD feature: "amx","avx512","avx2","sse","neon","asimd","scalar",""
	ModelBytes int64   `json:"model_bytes"` // resident weight bytes the target model needs (0 = unknown)
	FreeBytes  int64   `json:"free_bytes"`  // free device/host memory the model would load into
	MemKnown   bool    `json:"mem_known"`   // whether FreeBytes was actually probeable
	NUMANodes  int     `json:"numa_nodes"`  // online NUMA node count (0 = topology unreadable)
	Headroom   float64 `json:"headroom"`    // fit headroom fraction reserved for KV/activations (0..1)
}

// serveReadinessRow is one row of the serve-readiness table: a named check, its
// green/yellow/red status, the human tier label, what it found, and the operator
// action to take when it is not green.
type serveReadinessRow struct {
	Check       string `json:"check"`
	Status      string `json:"status"` // "ok" | "warn" | "fail"
	Tier        string `json:"tier"`   // "Ready" | "Marginal" | "Unready"
	Finding     string `json:"finding"`
	Remediation string `json:"remediation,omitempty"`
}

// serveReadinessReport is the whole table plus the rolled-up worst tier and a
// count of non-green rows.
type serveReadinessReport struct {
	Facts      serveHostFacts          `json:"facts"`
	Rows       []serveReadinessRow     `json:"rows"`
	Durability *serveDurabilityPosture `json:"durability,omitempty"`
	Rollup     string                  `json:"rollup"` // "Ready" | "Marginal" | "Unready"
	Findings   int                     `json:"findings"`
}

// serveStatusRank orders the three rungs so the report can roll up the worst one.
func serveStatusRank(status string) int {
	switch status {
	case sevFail:
		return 2
	case sevWarn:
		return 1
	default:
		return 0
	}
}

// serveTierLabel maps a green/yellow/red status to the operator-facing tier word.
func serveTierLabel(status string) string {
	switch status {
	case sevFail:
		return "Unready"
	case sevWarn:
		return "Marginal"
	default:
		return "Ready"
	}
}

// serveHumanBytes renders a byte count in binary GiB/MiB for the memory rows (the
// package humanBytes tops out at MiB, too coarse for multi-GiB weights).
func serveHumanBytes(n int64) string {
	const (
		kib = int64(1) << 10
		mib = int64(1) << 20
		gib = int64(1) << 30
	)
	switch {
	case n >= gib:
		return fmt.Sprintf("%.1f GiB", float64(n)/float64(gib))
	case n >= mib:
		return fmt.Sprintf("%.1f MiB", float64(n)/float64(mib))
	case n >= kib:
		return fmt.Sprintf("%.1f KiB", float64(n)/float64(kib))
	default:
		return fmt.Sprintf("%d B", n)
	}
}

func isaOrNone(isa string) string {
	if strings.TrimSpace(isa) == "" {
		return "none"
	}
	return isa
}

// serveISARow classifies the CPU SIMD decode tier: on x86 the fused Q8 kernel
// wants AVX-512/AMX (green), degrades to an AVX2 lane (yellow), and falls to the
// pure-Go scalar reference below that (red). On arm64 NEON/ASIMD/SVE is the
// serving lane; anything else is the scalar fallback. An unrecognized arch has no
// known SIMD tier and runs the portable scalar kernel (yellow, informational).
func serveISARow(f serveHostFacts) serveReadinessRow {
	isa := strings.ToLower(strings.TrimSpace(f.ISA))
	row := serveReadinessRow{Check: "cpu-isa"}
	switch f.Arch {
	case "amd64", "386":
		switch isa {
		case "amx", "avx512":
			row.Status = sevOK
			row.Finding = "x86 " + isa + " detected — top decode tier; the fused SIMD Q8 kernel is engaged"
		case "avx2":
			row.Status = sevWarn
			row.Finding = "x86 AVX2 only — decode uses the AVX2 kernel, not the faster AVX-512/AMX lane"
			row.Remediation = "serve on an AVX-512 or AMX host for peak tok/s; AVX2 still serves, just slower"
		default:
			row.Status = sevFail
			row.Finding = "no AVX2 detected (isa=" + isaOrNone(isa) + ") — decode falls back to the pure-Go scalar reference kernel"
			row.Remediation = "serve on a CPU with at least AVX2 (ideally AVX-512/AMX); the scalar path is correctness-only, far below serving throughput"
		}
	case "arm64":
		switch isa {
		case "neon", "asimd", "sve":
			row.Status = sevOK
			row.Finding = "arm64 " + isa + " detected — SIMD decode lane available"
		default:
			row.Status = sevWarn
			row.Finding = "arm64 without a detected NEON/ASIMD lane (isa=" + isaOrNone(isa) + ") — decode may fall back to scalar"
			row.Remediation = "expected on virtually every arm64 chip; a warn here means the ISA probe could not read CPU features"
		}
	default:
		row.Status = sevWarn
		row.Finding = "unrecognized arch " + arch(f) + " — no known SIMD decode tier; the portable scalar kernel is used"
		row.Remediation = "throughput is correctness-only on this arch; use an x86 AVX-512/AMX or arm64 NEON host to serve"
	}
	row.Tier = serveTierLabel(row.Status)
	return row
}

// arch renders the arch field for a message, guarding the empty case.
func arch(f serveHostFacts) string {
	if strings.TrimSpace(f.Arch) == "" {
		return "(unknown)"
	}
	return f.Arch
}

// serveFitRow classifies whether the target model's resident weights fit free
// memory. It reserves Headroom for the KV cache + activations that do NOT pass
// through the weight fit: weights inside the headroom budget are green; weights
// that fit raw free memory but not the budget are yellow (likely OOM under load);
// weights larger than free memory are red (will not load). Missing model size or
// unprobeable memory is a yellow "cannot verify" row, never a false red.
func serveFitRow(f serveHostFacts) serveReadinessRow {
	row := serveReadinessRow{Check: "model-fit"}
	switch {
	case f.ModelBytes <= 0:
		row.Status = sevWarn
		row.Finding = "no target model size given — cannot check memory headroom"
		row.Remediation = "pass --model-bytes (resident weight bytes) to verify the model fits free VRAM/RAM"
	case !f.MemKnown:
		row.Status = sevWarn
		row.Finding = fmt.Sprintf("model needs %s but free device/host memory is not probeable here", serveHumanBytes(f.ModelBytes))
		row.Remediation = "run this doctor on the target serve host so free VRAM/RAM can be measured"
	default:
		budget := f.FreeBytes
		if f.Headroom > 0 && f.Headroom < 1 {
			budget = int64(float64(f.FreeBytes) * (1 - f.Headroom))
		}
		switch {
		case f.ModelBytes <= budget:
			row.Status = sevOK
			row.Finding = fmt.Sprintf("model %s fits the headroom budget %s of %s free", serveHumanBytes(f.ModelBytes), serveHumanBytes(budget), serveHumanBytes(f.FreeBytes))
		case f.ModelBytes <= f.FreeBytes:
			row.Status = sevWarn
			row.Finding = fmt.Sprintf("model %s fits raw free %s but exceeds the %.0f%% headroom budget %s — no room left for KV/activations", serveHumanBytes(f.ModelBytes), serveHumanBytes(f.FreeBytes), f.Headroom*100, serveHumanBytes(budget))
			row.Remediation = "reduce context length, quantize the KV cache, or free memory; the weights fit but the run may OOM under load"
		default:
			row.Status = sevFail
			row.Finding = fmt.Sprintf("model %s exceeds free memory %s — it will not load", serveHumanBytes(f.ModelBytes), serveHumanBytes(f.FreeBytes))
			row.Remediation = "quantize the weights, shard across more devices, or add memory/GPUs; the model does not fit as configured"
		}
	}
	row.Tier = serveTierLabel(row.Status)
	return row
}

// serveNUMARow reports the online NUMA node layout. An unreadable topology is a
// benign yellow (single-socket boxes and non-Linux hosts land here); one node is
// green; two or more is green with a placement hint (size the decode worker set to
// the node count and interleave weight placement for cross-socket bandwidth).
func serveNUMARow(f serveHostFacts) serveReadinessRow {
	row := serveReadinessRow{Check: "numa-topology"}
	switch {
	case f.NUMANodes <= 0:
		row.Status = sevWarn
		row.Finding = "NUMA topology unreadable — cannot advise thread/interleave placement"
		row.Remediation = "on Linux ensure /sys/devices/system/node is present; on a single-socket box this is benign"
	case f.NUMANodes == 1:
		row.Status = sevOK
		row.Finding = "single NUMA node — no cross-socket placement concern"
	default:
		row.Status = sevOK
		row.Finding = fmt.Sprintf("%d NUMA nodes online — cross-socket memory bandwidth applies", f.NUMANodes)
		row.Remediation = fmt.Sprintf("size the decode worker threadpool to the %d nodes and interleave weight placement for best bandwidth", f.NUMANodes)
	}
	row.Tier = serveTierLabel(row.Status)
	return row
}

// buildServeReadiness folds the injected host facts into the full readiness table:
// the three rows, the rolled-up worst tier, and the count of non-green rows. Pure
// — no I/O — so tests assert the whole report directly.
func buildServeReadiness(f serveHostFacts) serveReadinessReport {
	rep := serveReadinessReport{Facts: f}
	rep.Rows = []serveReadinessRow{serveISARow(f), serveFitRow(f), serveNUMARow(f)}
	worst := sevOK
	for _, r := range rep.Rows {
		if serveStatusRank(r.Status) > serveStatusRank(worst) {
			worst = r.Status
		}
		if r.Status != sevOK {
			rep.Findings++
		}
	}
	rep.Rollup = serveTierLabel(worst)
	return rep
}

// probeServeHost is the live, best-effort host probe. It reads the arch and CPU
// SIMD feature bits (via golang.org/x/sys/cpu) and the online NUMA node count (via
// the compute package's shared accessor). Free device VRAM needs a live backend,
// which a read-only CPU-host probe cannot supply, so MemKnown stays false and the
// model-fit row reports "not probeable here" rather than a false verdict. The
// caller supplies the target model size + headroom.
func probeServeHost(modelBytes int64, headroom float64) serveHostFacts {
	return serveHostFacts{
		Arch:       runtime.GOARCH,
		ISA:        probeISAName(),
		ModelBytes: modelBytes,
		FreeBytes:  0,
		MemKnown:   false,
		NUMANodes:  probeNUMANodeCount(),
		Headroom:   headroom,
	}
}

// numaNodeDir matches the per-node directories the Linux kernel exposes under
// /sys/devices/system/node (node0, node1, …).
var numaNodeDir = regexp.MustCompile(`^node[0-9]+$`)

// probeNUMANodeCount counts the online NUMA nodes the host exposes by reading the
// Linux /sys topology, returning 0 when the topology is unreadable (non-Linux
// hosts, containers without the sysfs, single-socket boxes that expose none). The
// model-fit and NUMA rows treat 0 as a benign "unreadable" yellow, never a red.
func probeNUMANodeCount() int {
	entries, err := os.ReadDir("/sys/devices/system/node")
	if err != nil {
		return 0
	}
	n := 0
	for _, e := range entries {
		if e.IsDir() && numaNodeDir.MatchString(e.Name()) {
			n++
		}
	}
	return n
}

// probeISAName returns the highest SIMD feature name detected on the running host,
// or "scalar" when none of the serving lanes are present.
func probeISAName() string {
	switch runtime.GOARCH {
	case "amd64", "386":
		switch {
		case cpu.X86.HasAVX512F:
			return "avx512"
		case cpu.X86.HasAVX2:
			return "avx2"
		case cpu.X86.HasSSE2:
			return "sse"
		default:
			return "scalar"
		}
	case "arm64":
		if cpu.ARM64.HasASIMD {
			return "asimd"
		}
		return "scalar"
	default:
		return "scalar"
	}
}

// runServeDoctor is the testable core of `fak doctor serve`: it parses flags,
// probes the live host (unless facts are injected via the parsed flags), and
// renders the readiness table. Exit 1 when any row is Unready (red).
func runServeDoctor(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("doctor serve", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "doctor serve")
	modelBytes := fs.Int64("model-bytes", 0, "resident weight bytes of the target model, to check it fits free memory (0 = skip the fit check)")
	headroom := fs.Float64("headroom", 0.15, "fraction of free memory reserved for KV + activations in the model-fit check (0..1)")
	asJSON := fs.Bool("json", false, "emit the readiness report as JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak doctor serve: unexpected args: %v\n", fs.Args())
		return 2
	}

	facts := probeServeHost(*modelBytes, *headroom)
	rep := withServeDurabilityRow(buildServeReadiness(facts), resolveServeSessionState("", os.Getenv))
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak doctor serve")
	}
	writeServeReadinessHuman(stdout, rep)
	for _, r := range rep.Rows {
		if r.Status == sevFail {
			return 1
		}
	}
	return 0
}

// writeServeReadinessHuman renders the readiness table for a terminal.
func writeServeReadinessHuman(w io.Writer, rep serveReadinessReport) {
	fmt.Fprintln(w, "== fak doctor: serve readiness ==")
	fmt.Fprintf(w, "host: arch=%s isa=%s numa-nodes=%d\n", arch(rep.Facts), isaOrNone(rep.Facts.ISA), rep.Facts.NUMANodes)
	for _, r := range rep.Rows {
		fmt.Fprintf(w, "[%-8s] %-14s %s\n", r.Tier, r.Check, r.Finding)
		if r.Remediation != "" {
			fmt.Fprintf(w, "           remediation: %s\n", r.Remediation)
		}
	}
	if rep.Findings == 0 {
		fmt.Fprintf(w, "serve readiness: %s (0 findings)\n", rep.Rollup)
	} else {
		fmt.Fprintf(w, "serve readiness: %s (%d finding(s))\n", rep.Rollup, rep.Findings)
	}
}
