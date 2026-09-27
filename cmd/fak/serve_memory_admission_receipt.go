package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/harnessres"
	"github.com/anthony-chaudhary/fak/internal/localadmission"
	"github.com/anthony-chaudhary/fak/internal/memgate"
)

// metalAdmissionReceiptPrefix starts every native-admission receipt line on
// stderr, so an operator (or a live-witness capture) can grep one JSON object
// per lifecycle stage out of a serve log.
const metalAdmissionReceiptPrefix = "fak native admission receipt: "

// serveAdmissionReceiptOut is where admission receipts are written. Tests swap
// it to capture the lifecycle.
var serveAdmissionReceiptOut io.Writer = os.Stderr

// serveUnifiedMemoryTopology probes whether the Metal device draws from the
// host's physical memory pool. probed=false means no device answered, so the
// receipt reports the topology as unprobed rather than assuming it.
var serveUnifiedMemoryTopology = func() (unified, probed bool) {
	l := compute.DarwinWorkingSetLimits()
	return l.HasUnifiedMemory, l.RecommendedMaxWorkingSet > 0 || l.MaxBufferLength > 0
}

// serveSwapUsedBytes measures host swap in use (Darwin vm.swapusage). ok=false
// means it was not measured; the receipt then omits it instead of reporting 0.
var serveSwapUsedBytes = func() (int64, bool) {
	if runtime.GOOS != "darwin" {
		return 0, false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "sysctl", "vm.swapusage").Output()
	if err != nil {
		return 0, false
	}
	used, err := parseModelCanarySwap(out)
	if err != nil {
		return 0, false
	}
	return used, true
}

// serveProcessRSS measures this process's current and peak RSS through the
// self sampler (getrusage peak on darwin/linux; current RSS where the platform
// exposes it). Each value is reported only when the reader actually produced
// it. The per-PID reader is not used here: on darwin it leaves self RSS absent.
var serveProcessRSS = func() (rss, peak int64, haveRSS, havePeak bool) {
	s := harnessres.New()
	s.Start(time.Hour) // one immediate reading; Stop takes the final one
	k := s.Stop().Kernel
	return int64(k.RSSBytes), int64(k.PeakRSSBytes), k.HaveRSS, k.HavePeakRSS
}

// serveAdmissionHostSample reads the byte-precise host sample that admission
// reserves against. known=false means the host could not be probed; the sample
// then carries unknown pressure and zero capacity so the reservation store
// refuses it with a typed reason instead of the loader running unreserved.
func serveAdmissionHostSample() (localadmission.AdmissionSample, bool) {
	mem, err := serveReadMemory()
	if err != nil || mem.TotalBytes <= 0 {
		return localadmission.AdmissionSample{Pressure: localadmission.PressureUnknown}, false
	}
	s := memgate.AdmissionSampleFor(mem)
	return localadmission.AdmissionSample{
		TotalBytes:       s.TotalBytes,
		AllocatableBytes: s.AllocatableBytes,
		CompressedBytes:  s.CompressedBytes,
		WiredBytes:       s.WiredBytes,
		Pressure:         localadmission.Pressure(s.Pressure),
	}, true
}

// metalAdmissionReceipt accumulates one reservation's lifecycle receipt; each
// emit writes the current state as a single JSON line.
type metalAdmissionReceipt struct {
	localadmission.MetalReservationReceipt
	// Measured names the measurement fields this stage actually sampled
	// (swap_bytes, rss_bytes, peak_rss_bytes), so an absent value is never
	// mistaken for a measured zero.
	Measured []string `json:"measured,omitempty"`
	// out overrides serveAdmissionReceiptOut when non-nil.
	out io.Writer
}

func newMetalAdmissionReceipt(ggufPath, mode string, plan localadmission.MemoryPlan, env *localadmission.SessionResidencyPlan) *metalAdmissionReceipt {
	unified, probed := serveUnifiedMemoryTopology()
	topology := "unprobed"
	if probed {
		topology = "discrete"
		if unified {
			topology = "apple-unified-memory"
		}
	}
	rc := &metalAdmissionReceipt{MetalReservationReceipt: localadmission.MetalReservationReceipt{
		Schema:         localadmission.MetalReservationReceiptSchema,
		Engine:         "fak-native",
		Model:          filepath.Base(ggufPath),
		Topology:       topology,
		HostUnified:    unified,
		TopologyProbed: probed,
		// A unified physical pool never makes a device buffer host-dereferenceable.
		HostAddressable: false,
		PlannedBytes:    plan,
		OwnerPID:        os.Getpid(),
		AdmissionMode:   mode,
		Cleanup:         "none",
	}}
	if env != nil {
		rc.Classes = memoryClassesToStrings(env.Classes())
	}
	return rc
}

// observeHost records the live host sample admission read.
func (rc *metalAdmissionReceipt) observeHost(host localadmission.AdmissionSample) {
	rc.TotalBytes = host.TotalBytes
	rc.LiveAllocatableBytes = host.AllocatableBytes
	rc.AllocatableBytes = host.AllocatableBytes
	rc.AvailableBytes = host.AllocatableBytes
	rc.CompressedBytes = host.CompressedBytes
	rc.Pressure = host.Pressure
}

// observeDecision records the reservation store's decision. ReservedBytes is
// the aggregate held across every FAK reservation after the decision (this
// one included when admitted); AvailableBytes is what remains of the admitted
// capacity.
func (rc *metalAdmissionReceipt) observeDecision(dec localadmission.ReservationDecision) {
	rc.AllocatableBytes = dec.CapacityBytes
	rc.Pressure = dec.Pressure
	rc.Reason = dec.Reason
	rc.ReservedBytes = dec.ReservedBytes
	if dec.Admit && dec.Reservation != nil {
		rc.ReservationID = dec.Reservation.ID
		rc.Phase = dec.Reservation.Phase
		rc.ReservedBytes += dec.Reservation.HeldBytes
		rc.Cleanup = "active"
	}
	rc.AvailableBytes = max(dec.CapacityBytes-rc.ReservedBytes, 0)
}

// observeRelease records the post-teardown host: the reservation is gone, so
// ReservedBytes is what the remaining FAK owners hold and the live sample shows
// the capacity this release restored. The caller stamps the stage. A nil store
// (exclusive mode) leaves the shared ledger untouched.
func (rc *metalAdmissionReceipt) observeRelease(store *localadmission.ReservationStore, released bool) {
	rc.Phase = "released"
	rc.Cleanup = "released"
	if !released {
		rc.Cleanup = "release_failed"
	}
	if host, ok := serveAdmissionHostSample(); ok {
		rc.observeHost(host)
	}
	if store != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		reserved, err := store.TotalReservedBytes(ctx)
		cancel()
		if err == nil {
			rc.ReservedBytes = reserved
			rc.AvailableBytes = max(rc.AllocatableBytes-reserved, 0)
		}
	}
}

// stage stamps the lifecycle stage and verdict and measures swap and this
// process's RSS at that moment.
func (rc *metalAdmissionReceipt) stage(stage, verdict, reason string) {
	rc.Stage, rc.Verdict, rc.Reason = stage, verdict, reason
	rc.Timestamp = time.Now().UTC()
	rc.Measured = rc.Measured[:0]
	rc.SwapBytes, rc.RSSBytes, rc.PeakRSSBytes = 0, 0, 0
	if swap, ok := serveSwapUsedBytes(); ok {
		rc.SwapBytes = swap
		rc.Measured = append(rc.Measured, "swap_bytes")
	}
	rss, peak, haveRSS, havePeak := serveProcessRSS()
	if haveRSS {
		rc.RSSBytes = rss
		rc.Measured = append(rc.Measured, "rss_bytes")
	}
	if havePeak {
		rc.PeakRSSBytes = peak
		rc.Measured = append(rc.Measured, "peak_rss_bytes")
	}
}

// emit writes the receipt as one prefixed JSON line.
func (rc *metalAdmissionReceipt) emit() {
	out := rc.out
	if out == nil {
		out = serveAdmissionReceiptOut
	}
	b, err := json.Marshal(rc)
	if err != nil {
		fmt.Fprintf(out, "%s{\"stage\":%q,\"error\":%q}\n", metalAdmissionReceiptPrefix, rc.Stage, err.Error())
		return
	}
	fmt.Fprintf(out, "%s%s\n", metalAdmissionReceiptPrefix, b)
}
