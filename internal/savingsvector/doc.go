// Package savingsvector re-projects a turnbench Report's FLAT saving into the FOUR
// orthogonal accounts named by docs/explainers/compounding-benefits-of-a-saved-call.md.
//
// THE PROBLEM IT FIXES. internal/turnbench ships a FLAT saving: Net prices one
// integer, turns_saved, in tokens, dollars, and latency -- the SAME saving in three
// currencies. That under-models the benefit two ways the doc names:
//
//	(1) it has NO local-CPU axis (the boundary tax / in-process serve cost the gate
//	    actually saved is invisible to Net, even though the Report MEASURES it as
//	    local_serve_ns), and
//	(2) it treats dollars/tokens/latency as interchangeable, when they are SEPARATE
//	    budgets with separate ceilings -- and the binding one on a real run is rarely
//	    dollars (a laptop agent is context/wall-clock-bound; a GPU fleet is prefill-
//	    bound; a hooked CI gate is CPU-bound on the gate itself).
//
// WHAT THIS PACKAGE DOES. It reads a turnbench Report (the artifact `fak turntax
// --out report.json` writes) and re-projects its ALREADY-COMPUTED, ALREADY-MEASURED
// fields into the four-account savings VECTOR:
//
//	account        what a RUN call draws        per-axis provenance
//	-----------    -------------------------    --------------------------------
//	local_cpu      adjudication + maybe a spawn  MEASURED (Report.local_serve_ns,
//	                                             baseline = spawned-hook floor)
//	gpu_prefill    a forward pass               MODELED (turns_saved x prompt tokens;
//	                                             a token proxy, not FLOPs/wall-clock)
//	context_window a permanent window slot      MEASURED-as-a-rate where a ctxmmu
//	                                             pollution figure is supplied; else
//	                                             MODELED from turns_saved
//	wall_clock     a model round-trip           MODELED (cost_model.ModelTurnLatencyMs,
//	                                             a knob, never a measured wall-clock)
//
// It does NOT invent the horizon multiplier (r/d). It ships the MEASURED INPUTS to d
// (the per-call cost on each axis) and stops there -- the same discipline the
// webbench-number correction established (publish the structure + measured parts;
// never the invented single number).
//
// ANTI-OVERCLAIM (Selfcheck). The vector DECOMPOSES one event; it must never INFLATE
// it. The hard invariants Selfcheck asserts:
//   - the dollar-axis saving equals Net.dollars_saved to the cent (a re-projection,
//     not a new claim);
//   - every account saving is >= 0 and 0 exactly when turns_saved == 0 (the
//     happy-path control saves nothing on every axis);
//   - the binding account is whichever has the TIGHTEST headroom under the supplied
//     profile, never hard-coded to dollars.
//
// The profile only changes WHICH account is reported as binding (its ceiling), never
// the saving amounts. It defaults to "laptop" (context/wall-clock-bound), the common
// long-agent shape.
package savingsvector
