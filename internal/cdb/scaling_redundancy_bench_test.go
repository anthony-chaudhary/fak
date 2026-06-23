package cdb

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

// This file extends the #515 workflow-memory harness (#551) with the two EMPIRICAL
// measurements the analytic ctxplan scaling model (internal/ctxplan/scaling.go) only
// ASSERTS from Params:
//
//   - empirical scaling — as a recorded core image grows (N turns), the resident
//     working set a FIXED query demand-pages stays Θ(1)-bounded (byte-identical
//     image to image) while the lossless store grows Θ(N). scaling.go's
//     Model(Planned, …) draws that bend from Params; TestEmpiricalScalingBoundedResident
//     draws it from real recorded core images through Attach + WorkingSet, and
//     cross-checks the empirical curve against that analytic one.
//   - resident-redundancy — the content-addressed CAS collapses identical pages, so a
//     recorded image with repeated content carries fewer DISTINCT bytes than pages.
//     Info already exposes DedupSaved; TestResidentRedundancyContentAddressed makes it
//     a measured, asserted curve (not a field) and ties it to the sha256 content
//     address recall.Digest.
//
// Both run on the SAME recorded-core-image path the #540 utility ablation drives
// (record → Persist → Attach → WorkingSet); they add the scaling + redundancy axes
// that venue was missing. They touch no production code and assert fail-closed, so a
// regression that made resident grow with N or broke content-addressed dedup is caught.

// benchQuery is the fixed follow-up the working set is assembled for at every scale.
// Its content tokens {refund, fee, ledger} are the reference string.
const benchQuery = "refund fee ledger"

// benchRelevant are the byte-fixed pages the query references — held CONSTANT across
// every scale so the resident working set is identical image to image. Each is distinct
// (a different content address) and each carries all three query tokens, so the working
// set is exactly these pages at every N and nothing else.
var benchRelevant = [...]string{
	"refund fee ledger entry confirmed balance 2500 dollars account alpha",
	"refund fee ledger waiver premium tier credit 300 dollars account beta",
	"refund fee ledger reconciliation statement closing period gamma",
	"refund fee ledger policy override supervisor approval delta",
}

// fillerBody is a benign page DISJOINT from the query vocab — so it is never a
// working-set candidate (a page fault AVOIDED, growing the store without growing the
// resident set). When fillerDistinct > 1 the bodies CYCLE through fillerDistinct fixed
// strings, so identical bytes recur and the content-addressed CAS dedups them; when
// fillerDistinct <= 1 each slot is unique (no dedup).
func fillerBody(i, fillerDistinct int) string {
	if fillerDistinct <= 1 {
		return fmt.Sprintf("weather climate forecast pressure humidity temperature wind velocity report number %d", i)
	}
	return fmt.Sprintf("weather climate forecast pressure humidity temperature wind velocity precipitation cloud batch %d", i%fillerDistinct)
}

// recordScalingImage records a core image of nRelevant query-relevant pages plus nFiller
// disjoint filler pages (cycled over fillerDistinct distinct bodies) and persists it
// under dir. It returns the distinct-content count the image SHOULD carry, so the
// redundancy test can assert the CAS collapsed identical bytes to exactly that count.
func recordScalingImage(t *testing.T, dir string, nRelevant, nFiller, fillerDistinct int) int {
	t.Helper()
	ctx := context.Background()
	r := recall.NewRecorder("scaling-551")
	for _, body := range benchRelevant[:nRelevant] {
		r.Record(ctx, "Ledger", []byte(body))
	}
	for i := 0; i < nFiller; i++ {
		r.Record(ctx, "Weather", []byte(fillerBody(i, fillerDistinct)))
	}
	if err := r.Persist(dir); err != nil {
		t.Fatalf("persist scaling image: %v", err)
	}
	distinctFiller := fillerDistinct
	if distinctFiller <= 1 {
		distinctFiller = nFiller // every slot unique
	}
	return nRelevant + distinctFiller
}

// TestEmpiricalScalingBoundedResident is the #551 empirical-scaling witness. As a
// recorded core image grows 16×→256× turns, the resident working set a fixed query
// demand-pages stays byte-identical (Θ(1)) while the lossless store grows ~linearly
// (Θ(N)) — the bend ctxplan.Model(Planned) only draws from Params, here measured over
// real recorded core images and cross-checked against that analytic curve.
func TestEmpiricalScalingBoundedResident(t *testing.T) {
	const nRelevant = len(benchRelevant)
	scales := []int{16, 64, 256}

	// The bounded resident the analytic model pins at W = the relevant bytes (a real
	// measured constant, not magic), and the per-turn growth rate b = one filler page.
	relevantBytes := 0
	for _, body := range benchRelevant {
		relevantBytes += len(body)
	}
	bytesPerTurn := float64(len(fillerBody(0, 0)))

	type row struct {
		n                int
		pagesTouched     int
		bytesPagedIn     int64
		storeBytes       int64
		residencyPct     float64
		analyticResident int64
		analyticStore    int64
	}
	rows := make([]row, 0, len(scales))
	ctx := context.Background()

	for _, n := range scales {
		dir := t.TempDir()
		recordScalingImage(t, dir, nRelevant, n-nRelevant, 0) // unique filler — isolates scaling from redundancy

		im, err := Attach(dir)
		if err != nil {
			t.Fatalf("attach N=%d: %v", n, err)
		}
		info := im.Info()
		ws := im.WorkingSet(ctx, benchQuery, 0)

		// Analytic ctxplan Planned curve at this N, params mirrored to the empirical
		// byte sizes so the two curves are in the SAME byte units (not abstract tokens).
		pt := ctxplan.Model(ctxplan.Planned, ctxplan.Params{
			TokensPerTurn: bytesPerTurn, WorkingSet: relevantBytes, ForecastHit: 0.9, Retain: 0.7,
		}, []int{n})[0]

		rows = append(rows, row{n, ws.PagesTouched, ws.BytesPagedIn, info.RawBytes, ws.ResidencyPct, int64(pt.Resident), pt.Store})

		// (1) The working set is EXACTLY the relevant set at every scale: filler is never
		// a candidate (zero query overlap), so demand-paging touches nRelevant pages only.
		if ws.PagesTouched != nRelevant {
			t.Fatalf("N=%d: working set touched %d pages, want exactly the %d relevant (filler must never be a candidate)",
				n, ws.PagesTouched, nRelevant)
		}
		// (2) Empirical resident is byte-identical image to image AND equals the analytic
		// Planned resident: both pin at W once b*N > W. Θ(1) in N, measured not asserted.
		if ws.BytesPagedIn != int64(relevantBytes) {
			t.Fatalf("N=%d: resident bytes %d, want exactly the %d relevant bytes (the working set must be byte-identical across scales)",
				n, ws.BytesPagedIn, relevantBytes)
		}
		if int64(pt.Resident) != int64(relevantBytes) {
			t.Fatalf("N=%d: analytic Planned resident %d diverged from the empirical bounded resident %d",
				n, pt.Resident, relevantBytes)
		}
	}

	first, last := rows[0], rows[len(rows)-1]

	// (3) The lossless store grows ~linearly with N (Θ(N)) while resident stays flat.
	if last.storeBytes <= first.storeBytes {
		t.Fatalf("store did not grow with N: N=%d store=%d <= N=%d store=%d",
			first.n, first.storeBytes, last.n, last.storeBytes)
	}
	// Filler dominates the store at large N, so raw bytes scale ~N/nSmall (≈16×); the
	// relevant bytes are a rounding fraction at N=256, so >10× is a robust linear signal.
	if first.storeBytes*10 >= last.storeBytes {
		t.Fatalf("store did not grow ~linearly: %d (N=%d) not > 10× %d (N=%d)",
			last.storeBytes, last.n, first.storeBytes, first.n)
	}
	// (4) Residency % FALLS with N — the query touches a vanishing fraction of the image.
	if last.residencyPct >= first.residencyPct {
		t.Fatalf("residency %% did not fall with N: N=%d %.2f%% not < N=%d %.2f%%",
			last.n, last.residencyPct, first.n, first.residencyPct)
	}
	// (5) Analytic store is the same order of magnitude as the empirical store (shape
	// check — both are Θ(N) in bytes; exact equality is not expected because the
	// empirical store also carries the relevant bytes and real filler lengths vary).
	for _, r := range rows {
		if r.analyticStore > 0 && (float64(r.storeBytes) < 0.5*float64(r.analyticStore) || float64(r.storeBytes) > 2.0*float64(r.analyticStore)) {
			t.Fatalf("N=%d: empirical store %d outside [0.5×, 2×] analytic Planned store %d",
				r.n, r.storeBytes, r.analyticStore)
		}
	}

	var b strings.Builder
	fmt.Fprintf(&b, "\nempirical scaling #551 (recorded core images, query=%q, W=%d relevant bytes):\n", benchQuery, relevantBytes)
	fmt.Fprintf(&b, "  %-6s %-10s %-12s %-12s %-12s | %-12s %-12s\n",
		"N", "touched", "emp-resident", "emp-store", "residency%", "ana-resident", "ana-store")
	for _, r := range rows {
		fmt.Fprintf(&b, "  %-6d %-10d %-12d %-12d %-12.2f | %-12d %-12d\n",
			r.n, r.pagesTouched, r.bytesPagedIn, r.storeBytes, r.residencyPct, r.analyticResident, r.analyticStore)
	}
	b.WriteString("  -> resident stays byte-identical (Θ(1)) while the store grows ~linearly (Θ(N));\n")
	b.WriteString("     empirical confirms the ctxplan Planned-regime bend that Model() draws from Params.")
	t.Log(b.String())
}

// TestResidentRedundancyContentAddressed is the #551 resident-redundancy witness. A
// recorded core image with repeated content carries fewer DISTINCT bytes than pages,
// because recall.Digest is sha256(body) and the CAS is content-addressed. As the filler
// recurrences tighten (fewer distinct bodies over the same page count) the dedup win
// grows and the distinct-blob count falls to EXACTLY the distinct bodies written — a
// measured, asserted redundancy curve over a real recorded image.
func TestResidentRedundancyContentAddressed(t *testing.T) {
	const nRelevant = len(benchRelevant)
	const nFiller = 124
	const pages = nRelevant + nFiller

	arms := []struct {
		name           string
		fillerDistinct int
	}{
		{"unique", nFiller}, // every filler body distinct -> no dedup
		{"moderate", 16},    // 16 bodies cycled over 124 slots
		{"high", 4},         // 4 bodies cycled over 124 slots
	}

	type rrow struct {
		name          string
		distinctBlobs int
		dedupSaved    int64
		rawBytes      int64
		redundancyPct float64
	}
	rows := make([]rrow, 0, len(arms))
	for _, a := range arms {
		dir := t.TempDir()
		wantDistinct := recordScalingImage(t, dir, nRelevant, nFiller, a.fillerDistinct)

		im, err := Attach(dir)
		if err != nil {
			t.Fatalf("attach %s: %v", a.name, err)
		}
		info := im.Info()
		if info.Pages != pages {
			t.Fatalf("%s: image has %d pages, want %d", a.name, info.Pages, pages)
		}
		var redundancyPct float64
		if info.RawBytes > 0 {
			redundancyPct = 100 * float64(info.DedupSaved) / float64(info.RawBytes)
		}
		rows = append(rows, rrow{a.name, info.DistinctBlobs, info.DedupSaved, info.RawBytes, redundancyPct})

		// The CAS is content-addressed (sha256), so identical bytes collapse to one
		// address: the distinct-blob count is EXACTLY the distinct bodies written.
		if info.DistinctBlobs != wantDistinct {
			t.Fatalf("%s: distinct blobs %d, want exactly %d (nRelevant %d + fillerDistinct %d) — content addressing must collapse identical bodies",
				a.name, info.DistinctBlobs, wantDistinct, nRelevant, a.fillerDistinct)
		}
	}

	uniq := rows[0]
	// Unique arm: no two pages share content, so there is nothing to dedup.
	if uniq.dedupSaved != 0 {
		t.Fatalf("unique arm: expected zero dedup, got %d bytes saved", uniq.dedupSaved)
	}
	// Redundant arms: dedup is real, and strictly more recurrence saves strictly more.
	for _, r := range rows[1:] {
		if r.dedupSaved <= 0 {
			t.Fatalf("%s: expected positive dedup, got %d", r.name, r.dedupSaved)
		}
		if r.distinctBlobs >= uniq.distinctBlobs {
			t.Fatalf("%s: distinct blobs %d not below unique %d", r.name, r.distinctBlobs, uniq.distinctBlobs)
		}
	}
	if rows[2].dedupSaved <= rows[1].dedupSaved {
		t.Fatalf("dedup did not grow with recurrence: moderate=%d high=%d", rows[1].dedupSaved, rows[2].dedupSaved)
	}
	if rows[2].distinctBlobs >= rows[1].distinctBlobs {
		t.Fatalf("distinct blobs did not fall with recurrence: moderate=%d high=%d", rows[1].distinctBlobs, rows[2].distinctBlobs)
	}

	var b strings.Builder
	fmt.Fprintf(&b, "\nresident-redundancy #551 (recorded core image, %d pages):\n", pages)
	fmt.Fprintf(&b, "  %-10s %-16s %-16s %-12s %-14s\n", "arm", "distinct_blobs", "dedup_saved_B", "raw_bytes", "redundancy%")
	for _, r := range rows {
		fmt.Fprintf(&b, "  %-10s %-16d %-16d %-12d %-14.1f\n", r.name, r.distinctBlobs, r.dedupSaved, r.rawBytes, r.redundancyPct)
	}
	b.WriteString("  -> identical bytes collapse under the sha256 content address, so the\n")
	b.WriteString("     recorded image's cold-store footprint shrinks with redundancy, measured.")
	t.Log(b.String())
}
