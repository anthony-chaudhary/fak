// SM90 pull-style FP4 MegaMoE task-shape contract and K128 stage-ordering
// invariants.
//
// ADAPTED from flashinfer-ai/flashinfer PR #7961 "perf(moe_ep): faster SM90
// MXFP4/NVFP4 pull MegaMoE (N72/N80 token tiles, measured defaults, K128
// pipelining)", merged 2026-11-06 at 74c8d6bd0943165f8ec85c898908f3cebcef875a:
//   flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/shim/hopper_mxfp4.py
//   flashinfer/moe_ep/kernel_src/sm90/pull_style_cutedsl_megakernel/shim/mxfp4_optimization.py
// Apache-2.0 upstream. This fragment re-expresses the task-shape contract
// and pipelining invariants as a host/device-shared skeleton; no upstream
// code block is copied and no BSD device kernel is ported. The closed
// in-memory oracle over tile packing and stage ordering lives in
// mega_moe_fp4_test.go (package compute); the constexpr mirrors below keep
// both sides on one contract.
//
// Structure and oracle only: this fragment defines no kernels, launches
// nothing, and adds no ABI surface. The upstream H200 wall-clock band
// (0.84-0.95x vs main) is UPSTREAM-REPORTED provenance data, never fak
// performance evidence.

#pragma once

// Swap-AB task shape: an M256 expert-row tile with token-tile width N72 or
// N80, so one task holds an expert's 65-80 rows instead of a near-empty
// second N64 tile whose activation-scale pool is padded to a tile multiple.
constexpr int kMegaMoeFP4M = 256;
constexpr int kMegaMoeFP4TokenTileN72 = 72;
constexpr int kMegaMoeFP4TokenTileN80 = 80;
constexpr int kMegaMoeFP4MaxRowsPerTask = 80;
constexpr int kMegaMoeFP4KStage = 128;
constexpr int kMegaMoeFP4PipelineDepth = 2;
constexpr int kMegaMoeFP4MaxInFlightRows = 4;

// Reference envelopes carrying upstream-measured untuned-launch tactics.
// The 0.84-0.95x wall-clock band was reported by upstream on H200 for these
// shapes; recorded here for provenance only, never asserted as fak
// performance.
constexpr int kMegaMoeFP4RefHiddenA = 7168;
constexpr int kMegaMoeFP4RefIntermediateA = 3072;
constexpr int kMegaMoeFP4RefExpertsA = 384;
constexpr int kMegaMoeFP4RefExpertParallelA = 4;
constexpr int kMegaMoeFP4RefHiddenB = 2560;
constexpr int kMegaMoeFP4RefIntermediateB = 640;
constexpr int kMegaMoeFP4RefExpertsB = 512;
constexpr int kMegaMoeFP4RefExpertParallelB = 2;

// select_token_tile_width picks the contracted swap-AB token-tile width for
// rows in [1, 80]: N72 up to 72 rows, N80 above. Out-of-range row counts
// return 0 so callers fail closed; multi-task splits above the cap are
// outside this skeleton's scope.
constexpr int mega_moe_fp4_select_token_tile_width(int rows) {
    if (rows < 1 || rows > kMegaMoeFP4MaxRowsPerTask) return 0;
    return rows <= kMegaMoeFP4TokenTileN72 ? kMegaMoeFP4TokenTileN72 : kMegaMoeFP4TokenTileN80;
}

// scale_pool_rows sizes the activation-scale pool to the exact row count.
// The rejected N64 alternative padded the pool to a tile multiple; that
// padding is exactly the cost this contract removes.
constexpr int mega_moe_fp4_scale_pool_rows(int rows) { return rows; }

// stage_buffer maps a K128 stage onto its double buffer.
constexpr int mega_moe_fp4_stage_buffer(int stage) {
    return stage % kMegaMoeFP4PipelineDepth;
}

// pipeline_makespan replays S K128 stages in abstract time units under the
// contracted dependencies: load(s) may issue once the WGMMA that last
// consumed its double buffer finished (stage s-2 at depth 2), and wgmma(s)
// starts no earlier than load(s)'s end or the previous stage's wgmma end.
// This is the wait-for-and-LDSM-load-the-next-stage invariant while current
// WGMMA tiles run.
constexpr int mega_moe_fp4_pipeline_makespan(const int *load_dur, const int *wgmma_dur, int stages) {
    int w_end[kMegaMoeFP4PipelineDepth] = {0, 0};
    for (int s = 0; s < stages; ++s) {
        const int l_start = (s >= kMegaMoeFP4PipelineDepth) ? w_end[s % kMegaMoeFP4PipelineDepth] : 0;
        const int l_end = l_start + load_dur[s];
        int w_start = l_end;
        if (s > 0 && w_end[(s - 1) % kMegaMoeFP4PipelineDepth] > w_start)
            w_start = w_end[(s - 1) % kMegaMoeFP4PipelineDepth];
        w_end[s % kMegaMoeFP4PipelineDepth] = w_start + wgmma_dur[s];
    }
    return w_end[(stages - 1) % kMegaMoeFP4PipelineDepth];
}

// pull_slot_for_row gives the pull-buffer slot for row r on pass p under the
// per-pass row rotation: each pass advances the base offset by one slot.
// slots must be >= kMegaMoeFP4MaxInFlightRows; the caller owns that check.
constexpr int mega_moe_fp4_pull_slot_for_row(int row, int pass, int slots) {
    return (row + pass) % slots;
}

// Fixed-duration oracle vectors pinning the constexpr pipeline replay.
constexpr int kMegaMoeFP4OracleLoadsEven[4] = {10, 10, 10, 10};
constexpr int kMegaMoeFP4OracleWgmmaEven[4] = {10, 10, 10, 10};
constexpr int kMegaMoeFP4OracleLoadsSlow[4] = {20, 20, 20, 20};
constexpr int kMegaMoeFP4OracleWgmmaFast[4] = {10, 10, 10, 10};

// Closed-contract pins: the Go oracle in mega_moe_fp4_test.go asserts the
// same values, so the two sides cannot drift apart silently.
static_assert(kMegaMoeFP4TokenTileN72 < kMegaMoeFP4TokenTileN80, "tile ladder order");
static_assert(kMegaMoeFP4TokenTileN80 == kMegaMoeFP4MaxRowsPerTask, "wide tile covers the whole task range");
static_assert(mega_moe_fp4_select_token_tile_width(65) == 72, "65 rows take the N72 tile");
static_assert(mega_moe_fp4_select_token_tile_width(72) == 72, "the 72-row boundary stays N72");
static_assert(mega_moe_fp4_select_token_tile_width(73) == 80, "73 rows take the N80 tile");
static_assert(mega_moe_fp4_select_token_tile_width(80) == 80, "80 rows stay one task");
static_assert(mega_moe_fp4_select_token_tile_width(81) == 0, "out-of-range rows fail closed");
static_assert(mega_moe_fp4_scale_pool_rows(65) == 65, "scale pool is exact, not padded");
static_assert(mega_moe_fp4_stage_buffer(4) == 0, "double-buffer wrap");
static_assert(mega_moe_fp4_pull_slot_for_row(0, 1, 8) == 1, "per-pass rotation advances one slot");
static_assert(mega_moe_fp4_pipeline_makespan(kMegaMoeFP4OracleLoadsEven, kMegaMoeFP4OracleWgmmaEven, 4) == 50,
              "balanced K128 pipeline hides loads behind WGMMA");
static_assert(mega_moe_fp4_pipeline_makespan(kMegaMoeFP4OracleLoadsSlow, kMegaMoeFP4OracleWgmmaFast, 4) == 70,
              "slow loads serialize through double-buffer reuse");
