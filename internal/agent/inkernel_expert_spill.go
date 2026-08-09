package agent

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkernel_expert_spill.go — the served seam for the GRADED MoE expert spill (#5612, epic #5606,
// docs/MOE-ACTIVATED-OFFLOAD-PLAN.md).
//
// What was missing. The model package can now size a graded placement and install it
// (internal/model/expert_spill_placement.go: ResolveExpertSpillPlacement /
// ApplyExpertSpillPlacement), and the bounded routed-expert ring (#5611) can hold the device side
// to the ACTIVATED working set rather than the stored expert bulk. Neither was reachable from a
// serve: every session the in-kernel planner built took the ungraded all-or-nothing split
// (cpuOffloadExperts) and left the ring at 0. This file is the reachable half — one resolve at
// setup, one install per session.
//
// Why the resolve happens ONCE. Sizing walks every resident tensor name to find the model's real
// MoE layer ordinals and tallies its resident bytes; on a 753B checkpoint that is ~70k names. The
// device path builds a session per REQUEST, so resolving there would pay that walk per request for
// an answer that cannot change while the model is loaded. SetExpertSpill resolves at setup and each
// session install is then three field writes plus one already-built predicate.

// ExpertSpillAuto is the grade that means SIZE IT: AutoFitExpertSpill picks the smallest number of
// MoE layers to spill so the device-resident remainder fits the measured budget. It is the value
// `--n-cpu-moe auto` resolves to. Any n >= 0 is an explicit operator count and is honored exactly
// (or refused when out of range) — the auto search is bypassed.
const ExpertSpillAuto = -1

// expertSpillDeviceHeadroom is the fraction of measured free device memory the expert placement
// does NOT budget for weights. The KV cache, activations and the allocator's own fragmentation come
// out of the same pool and are not in ExpertSpillBudget's byte terms, so sizing the ring against
// every free byte would hand the experts memory the decode then needs and turn a graceful spill
// into an OOM. 0.15 is the same reserve `fak serve`'s device preflight already holds back
// (serveGGUFDeviceHeadroom), so the two admissions agree instead of each admitting what the other
// refuses.
const expertSpillDeviceHeadroom = 0.15

// SetExpertSpill resolves the operator's `--n-cpu-moe` grade against this planner's model and
// device, and installs the result so every session built afterwards runs it. n is either
// ExpertSpillAuto or an explicit count of MoE layers to spill to host RAM.
//
// deviceBudgetBytes is the device byte budget the resident remainder must fit. Pass <= 0 to MEASURE
// it from the backend (expertSpillDeviceBudget below); a caller that already sized the device — a
// serve that ran its own preflight — passes its own figure so both use one number.
//
// It REFUSES rather than degrading, in three cases an operator can actually hit:
//
//   - an explicit n outside [0, MoELayers] — the typed *model.ExpertSpillRangeError, never a
//     silent clamp into a residency nobody asked for;
//   - ExpertSpillAuto with no measurable budget (no backend, or a backend with no capacity probe)
//     — auto-fit against a zero budget would "fit" by spilling every layer, which is the ungraded
//     offload wearing the word auto;
//   - an explicit spill of n > 0 on a model with no routed-expert residency (a dense model, or an
//     MoE whose experts are not in any resident store) — there is nothing to spill, and silently
//     serving the unchanged placement would let an operator believe a spill they asked for happened.
//
// n == 0 and ExpertSpillAuto on such a model are NOT errors: neither asked to move anything, so the
// placement is simply left as it was.
func (p *InKernelPlanner) SetExpertSpill(n int, deviceBudgetBytes int64) error {
	if p == nil || p.m == nil {
		return nil
	}
	if deviceBudgetBytes <= 0 {
		deviceBudgetBytes = p.expertSpillDeviceBudget()
	}
	if n < 0 && deviceBudgetBytes <= 0 {
		return fmt.Errorf("agent: --n-cpu-moe auto needs a measurable device budget, but backend %s reports none; pass an explicit layer count instead", expertSpillBackendName(p.backend))
	}
	plan, ok, err := p.m.ResolveExpertSpillPlacement(deviceBudgetBytes, n)
	if err != nil {
		return fmt.Errorf("agent: --n-cpu-moe: %w", err)
	}
	if !ok {
		if n > 0 {
			return fmt.Errorf("agent: --n-cpu-moe %d: this model has no routed-expert residency to spill (a dense model, or its experts are not resident)", n)
		}
		return nil
	}
	p.expertSpill = &plan
	return nil
}

// ExpertSpillPlacement reports the resolved graded placement, or ok=false when this planner runs
// the ungraded default. It is exported so a serve can REPORT what it admitted — the spill count,
// the resulting device residency, and whether it Fits — instead of the operator having to infer the
// placement from throughput.
func (p *InKernelPlanner) ExpertSpillPlacement() (model.ExpertSpillPlacement, bool) {
	if p == nil || p.expertSpill == nil {
		return model.ExpertSpillPlacement{}, false
	}
	return *p.expertSpill, true
}

// expertSpillDeviceBudget measures the device bytes this placement may size against, or 0 when the
// backend cannot say.
//
// It measures FREE rather than total. The CUDA context, any co-tenant process and whatever the
// model load already uploaded are all charged against the device before this runs, and total would
// silently count every one of those bytes as available to the experts. Free is the honest ceiling;
// the headroom reserve above then keeps the KV cache and activations out of the experts' budget.
func (p *InKernelPlanner) expertSpillDeviceBudget() int64 {
	if p == nil || p.backend == nil {
		return 0
	}
	total, free, known := compute.DeviceMemoryInfo(p.backend)
	if !known {
		return 0
	}
	avail := free
	if avail <= 0 {
		avail = total
	}
	if avail <= 0 {
		return 0
	}
	return int64(float64(avail) * (1 - expertSpillDeviceHeadroom))
}

// expertSpillBackendName names the backend for a refusal message without asserting a non-nil one.
func expertSpillBackendName(b compute.Backend) string {
	if b == nil {
		return "(none)"
	}
	return b.Name()
}

// ExpertSpillEnv is the environment knob that grades the expert spill on an already-running serve.
// It spells llama.cpp's flag so an operator carrying a working `--n-cpu-moe` number types the same
// thing here and gets the same placement.
const ExpertSpillEnv = "FAK_N_CPU_MOE"

// ParseExpertSpillGrade parses an `--n-cpu-moe` value into a grade for SetExpertSpill.
//
//	""  / "off"  -> set=false: the ungraded default, exactly the pre-#5612 all-or-nothing split
//	"auto"       -> ExpertSpillAuto: size N against the measured device budget
//	"0".."N"     -> that many MoE layers spilled to host, honored exactly
//
// Anything else is REFUSED. A misspelled grade must not fall back to a placement the operator did
// not choose: silently serving "atuo" as off is how a 424 GB expert bulk ends up back on a device
// that cannot hold it, with nothing in the log saying so.
func ParseExpertSpillGrade(s string) (n int, set bool, err error) {
	v := strings.ToLower(strings.TrimSpace(s))
	switch v {
	case "", "off", "none":
		return 0, false, nil
	case "auto":
		return ExpertSpillAuto, true, nil
	}
	n, err = strconv.Atoi(v)
	if err != nil || n < 0 {
		return 0, false, fmt.Errorf("n-cpu-moe: %q is not a layer count; want auto, off, or a count >= 0", s)
	}
	return n, true, nil
}

// setExpertSpillFromEnv applies ExpertSpillEnv, if the operator set it. Unset — every serve that
// has not opted in — it is a no-op and the placement stays exactly what cpuOffloadExperts made it.
//
// A refusal here is LOGGED, not fatal: this is the opportunistic env door on an already-constructed
// planner, and taking a serve down at construction time for a mistyped optional knob trades a
// degraded placement for no serve at all. The refusal names the knob and says which placement is
// actually being served, so it cannot be mistaken for the spill having happened. The strict door is
// SetExpertSpill itself — a caller that parses an operator FLAG gets the error back and can refuse
// the launch outright, which is the right posture when the operator is still at the terminal.
func (p *InKernelPlanner) setExpertSpillFromEnv() {
	n, set, err := ParseExpertSpillGrade(os.Getenv(ExpertSpillEnv))
	if err == nil && !set {
		return
	}
	if err == nil {
		err = p.SetExpertSpill(n, 0)
	}
	if err != nil {
		log.Printf("fak: %s=%q REFUSED: %v — serving the ungraded expert placement instead", ExpertSpillEnv, os.Getenv(ExpertSpillEnv), err)
		return
	}
	if plan, ok := p.ExpertSpillPlacement(); ok {
		log.Printf("fak: %s=%q -> spill %d of %d MoE layers to host (%d MiB), device-resident %d MiB, expert ring %d MiB, fits=%v",
			ExpertSpillEnv, os.Getenv(ExpertSpillEnv), plan.Fit.SpillLayers, plan.Budget.MoELayers,
			plan.Fit.HostSpillBytes>>20, plan.Fit.DeviceResidentBytes>>20, plan.RingBytes>>20, plan.Fit.Fits)
	}
}

// applyExpertSpill installs the resolved placement on a session the planner just built. With no
// resolved placement (the default, and every planner whose operator never passed `--n-cpu-moe`)
// it is a no-op, so the session keeps byte-for-byte the placement cpuOffloadExperts alone gave it.
func (p *InKernelPlanner) applyExpertSpill(s *model.Session) {
	if p == nil || p.expertSpill == nil || s == nil {
		return
	}
	s.ApplyExpertSpillPlacement(*p.expertSpill)
}
