package main

// serveCPUOffloadStreamedResidentMargin is the fraction of the headroom-adjusted host budget the
// bounded-resident streamed expert bound DECLINES to charge, so the streamed plan's host total lands
// STRICTLY below the budget that judges it. Without a margin the bound IS fit.avail(), so
// fitServeStreamedCPUOffloadPathOnHost compares the budget against itself: any non-expert host row
// (KV/scratch/activation) or the int64 truncation in compute.BudgetAfterHeadroom
// (int64(float64(budget)*(1-headroom))) pushes the plan total over by a few bytes and fails closed.
// That is the physical strix3 refusal (plan needs 48.01 GiB, host has 47.99 GiB, FitTooBig) with no
// real wall, on the exact host class the streamed arm exists to serve (fak#13140). 0.10 also leaves
// real slack for the resident-page jitter the host arm's own headroom exists to absorb; it is
// deliberately conservative rather than a razor-thin epsilon, and still a genuine bounded working
// set (10% of the budget), never zero.
const serveCPUOffloadStreamedResidentMargin = 0.10

// serveCPUOffloadStreamedResidentBound derives the bounded host-resident expert working set for the
// streamed policy from the SAME measured fit budget the resident arm is judged against: the
// headroom-adjusted host budget (fit.avail()) with serveCPUOffloadStreamedResidentMargin declared
// BELOW it, so the streamed plan's host total is strictly less than the budget it is judged against
// (fak#13140). An unprobeable host yields zero (stream-through) -- the honest floor, because an
// unmeasurable host must not be promised residency it may not have.
func serveCPUOffloadStreamedResidentBound(fit serveFitBudget) int64 {
	avail := fit.avail()
	if avail <= 0 {
		return 0
	}
	bound := int64(float64(avail) * (1 - serveCPUOffloadStreamedResidentMargin))
	if bound >= avail {
		// A razor-thin budget must still land strictly below avail, and never become negative.
		bound = avail - 1
	}
	if bound < 0 {
		return 0
	}
	return bound
}

// serveStreamedDenseQ4KWorkingSetBound derives the bounded host-resident DENSE working set for the
// streamed dense route (fak#13205) from the SAME measured fit budget the calling sizing path judges
// its plan against. The device-scoped dense transit is NOT born in VRAM: staging materializes it in
// host RAM (read -> dequant/transcode -> device upload -> free), so the fit guard must judge the
// bounded working set the loader actually RETAINS (ggufload.WithStreamedDenseQ4KWorkingSet, landed
// fak#13194) rather than the full on-disk dense side -- otherwise the ~63.22 GiB V4.1 dense side is
// ALSO charged as a transient host demand and the kernel OOM-kills the serve at staging
// (`staging-host-charge=63.223GiB host-budget=48.542GiB`, then OOM via filemap_fault; fak#13171).
//
// The arithmetic is identical to the bounded streamed-EXPERT precedent
// (serveCPUOffloadStreamedResidentBound, fak#13121/#13140): the headroom-adjusted host budget
// (fit.avail()) less serveCPUOffloadStreamedResidentMargin, clamped so the bound lands strictly
// below avail and never goes negative. An unprobeable host yields zero (stream-through) -- the
// honest floor, because an unmeasurable host must not be promised residency it may not have. The
// SAME value must flow to both the estimate and the load option list (one measurement).
func serveStreamedDenseQ4KWorkingSetBound(fit serveFitBudget) int64 {
	avail := fit.avail()
	if avail <= 0 {
		return 0
	}
	bound := int64(float64(avail) * (1 - serveCPUOffloadStreamedResidentMargin))
	if bound >= avail {
		// A razor-thin budget must still land strictly below avail, and never become negative.
		bound = avail - 1
	}
	if bound < 0 {
		return 0
	}
	return bound
}
