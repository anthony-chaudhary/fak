package stallscan

// reboot_unmeasured_test.go — the binary/source skew guard on the reboot gate
// (issue #3668 remedy 1).
//
// The live defect these tests pin: `fak stallscan` records are tagged
// `fak.stallscan.v1` and every per-process array is `omitempty`, and the tag did
// NOT change when the thread axis was added. So a record written by a deployed
// `fak.exe` built before that axis is indistinguishable, field for field, from a
// record written by a current one on a host with no thread hog — and AdviseReboot,
// reading `TopThreads` off that record, returned `Advised:false`. On 2026-07-09 a
// WindowsTerminal at 3,206 threads read as "no reboot needed" for exactly this
// reason. The guard is that an empty census on an enabled axis is named, not
// silently folded into the calm verdict.

import (
	"encoding/json"
	"testing"
	"time"
)

// TestAdviseReboot_staleProducerThreadAxis_namedNotCalm is the regression proper:
// the shape a pre-thread-axis binary actually emits — a populated handle census
// and NO thread census — must not come back as a plain calm verdict.
func TestAdviseReboot_staleProducerThreadAxis_namedNotCalm(t *testing.T) {
	// What the stale deployed fak.exe wrote on 2026-07-09: handles measured (and
	// under the reboot line at that moment), threads absent from the record even
	// though the same WindowsTerminal held 3,206 of them.
	stale := Sample{
		TopHandles: []ProcHandles{{PID: 4242, Name: "WindowsTerminal.exe", Handles: 21953}},
	}
	got := AdviseReboot(stale, DefaultRebootThresholds())
	if got.Advised {
		t.Fatalf("a census this reading does not contain cannot page, got %+v", got)
	}
	if got.Measured() {
		t.Fatalf("the thread axis had no census; the verdict must not claim it was measured: %+v", got)
	}
	if len(got.Unmeasured) != 1 || got.Unmeasured[0] != "thread_high_water" {
		t.Fatalf("want the thread axis named as unmeasured and the measured handle axis left out, got %v", got.Unmeasured)
	}
}

// TestAdviseReboot_staleAndCalmDoNotRenderAlike is the defect stated in the only
// terms a consumer actually sees: the serialized record. Before the guard both of
// these marshalled to the identical `{"advised":false}`, so no reader downstream
// of the JSON — the --watch trail, stallpage, operatorbrief, an operator — could
// tell a host that was measured and found calm from one whose thread axis the
// producing binary never had. Written against the wire form on purpose: it fails
// on any build that folds the two together, whatever the in-memory API looks like.
func TestAdviseReboot_staleAndCalmDoNotRenderAlike(t *testing.T) {
	stale, err := json.Marshal(AdviseReboot(Sample{
		TopHandles: []ProcHandles{{PID: 4242, Name: "WindowsTerminal.exe", Handles: 21953}},
	}, DefaultRebootThresholds()))
	if err != nil {
		t.Fatal(err)
	}
	calm, err := json.Marshal(AdviseReboot(rebootSample(21953, 1097), DefaultRebootThresholds()))
	if err != nil {
		t.Fatal(err)
	}
	if string(stale) == string(calm) {
		t.Fatalf("a reading with no thread census renders exactly like a measured calm one (%s) — a stale producer's silence is unreadable as such", stale)
	}
}

// TestAdviseReboot_emptyCensus_bothAxesUnmeasured covers the fully silent reading
// — no monitor output at all, or a census call that failed outright. Neither axis
// was decidable, so both must say so rather than one calm `advised:false`.
func TestAdviseReboot_emptyCensus_bothAxesUnmeasured(t *testing.T) {
	got := AdviseReboot(Sample{}, DefaultRebootThresholds())
	if got.Advised || got.Measured() {
		t.Fatalf("an empty census decides nothing; want not-advised and not-measured, got %+v", got)
	}
	if len(got.Unmeasured) != 2 || got.Unmeasured[0] != "handle_high_water" || got.Unmeasured[1] != "thread_high_water" {
		t.Fatalf("want both axes named, handle first, got %v", got.Unmeasured)
	}
}

// TestAdviseReboot_disabledAxis_notUnmeasured separates the two silences that look
// alike from outside: an operator who set a line to 0 turned that axis off on
// purpose and needs no alarm, whereas a missing census on an ENABLED axis is the
// skew this guard exists to catch.
func TestAdviseReboot_disabledAxis_notUnmeasured(t *testing.T) {
	// Threads disabled, handle census present: nothing is unmeasured.
	off := AdviseReboot(rebootSample(15000, 900), RebootThresholds{HandleHighWater: 30000})
	if !off.Measured() || len(off.Unmeasured) != 0 {
		t.Fatalf("a line of 0 is off on purpose, not unmeasured, got %+v", off)
	}
	// Both lines 0: every axis is off, so an empty sample raises nothing either.
	none := AdviseReboot(Sample{}, RebootThresholds{})
	if !none.Measured() || len(none.Unmeasured) != 0 {
		t.Fatalf("both axes disabled leaves nothing to measure, got %+v", none)
	}
}

// TestAdviseReboot_pageStillNamesMissingAxis guards the half-blind page: crossing
// on handles must not let the missing thread census ride along unmentioned, or an
// operator reads the page as "handles are the whole story".
func TestAdviseReboot_pageStillNamesMissingAxis(t *testing.T) {
	got := AdviseReboot(Sample{
		TopHandles: []ProcHandles{{PID: 4242, Name: "WindowsTerminal.exe", Handles: 33054}},
	}, DefaultRebootThresholds())
	if !got.Advised || got.Axis != "handle_high_water" {
		t.Fatalf("33054 handles crosses the 30k line; want a handle-axis page, got %+v", got)
	}
	if got.Measured() || len(got.Unmeasured) != 1 || got.Unmeasured[0] != "thread_high_water" {
		t.Fatalf("a page from a half-blind reading must still name the axis it could not read, got %v", got.Unmeasured)
	}
}

// TestAdviseReboot_fullCensus_recordUnchanged keeps the guard free for every
// healthy reading: a sample that covered both axes must be measured, and its JSON
// must not grow a key. Byte-for-byte, because the record travels in the --watch
// JSONL trail and through stallpage/operatorbrief.
func TestAdviseReboot_fullCensus_recordUnchanged(t *testing.T) {
	for _, s := range []Sample{rebootSample(15000, 900), rebootSample(31000, 2400)} {
		got := AdviseReboot(s, DefaultRebootThresholds())
		if !got.Measured() || len(got.Unmeasured) != 0 {
			t.Fatalf("both censuses present: nothing is unmeasured, got %+v", got)
		}
		blob, err := json.Marshal(got)
		if err != nil {
			t.Fatal(err)
		}
		var rec map[string]any
		if err := json.Unmarshal(blob, &rec); err != nil {
			t.Fatal(err)
		}
		if _, ok := rec["unmeasured"]; ok {
			t.Fatalf("a fully measured record must not carry the key at all, got %s", blob)
		}
	}
}

// TestAdviseReboot_crossersStayOneLevel holds the structural invariant the
// Crossers doc already promises: the coverage note lives on the headline only, so
// the crosser set does not turn into a nested tree of repeated advice.
func TestAdviseReboot_crossersStayOneLevel(t *testing.T) {
	got := AdviseReboot(Sample{
		TopHandles: []ProcHandles{
			{PID: 4242, Name: "WindowsTerminal.exe", Handles: 33054},
			{PID: 9001, Name: "svchost.exe", Handles: 31200},
		},
	}, DefaultRebootThresholds())
	if len(got.Crossers) != 2 {
		t.Fatalf("want both handle crossers, got %+v", got.Crossers)
	}
	if len(got.Unmeasured) != 1 {
		t.Fatalf("the headline carries the coverage note, got %v", got.Unmeasured)
	}
	for i, c := range got.Crossers {
		if len(c.Unmeasured) != 0 {
			t.Fatalf("crosser %d must leave the coverage note unset, got %v", i, c.Unmeasured)
		}
	}
}

// TestUnmeasured_isNotArmed ties the two silences this package now distinguishes
// to the same rule: arming.go answers "was a reading taken at all", Unmeasured
// answers "did the reading that WAS taken cover this axis". A caller must be able
// to reach a not-measured state from either, so neither can be mistaken for calm.
func TestUnmeasured_isNotArmed(t *testing.T) {
	now := time.Now()
	missingRecord := ClassifyArming(LedgerRead{}, now, time.Minute, true)
	if missingRecord.Armed() {
		t.Fatalf("no reading at all is not armed, got %+v", missingRecord)
	}
	// A reading exists and is fresh, but the axis inside it is blank — armed at the
	// record level, still undecidable at the axis level.
	armed := ClassifyArming(LedgerRead{Found: true, Parsed: true, Timestamp: now}, now, time.Minute, true)
	if !armed.Armed() {
		t.Fatalf("a fresh parsed reading is armed, got %+v", armed)
	}
	if AdviseReboot(Sample{}, DefaultRebootThresholds()).Measured() {
		t.Fatal("an armed record whose census is empty must still report the axis as unmeasured")
	}
}
