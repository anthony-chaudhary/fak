package dispatchdoa

import (
	"os"
	"path/filepath"
	"regexp"
	"testing"
)

// corpusRel is this repo's own retained dispatch-run evidence — the one good thing the
// #5868 outage left behind. It is not committed (a multi-GB rolling trace tree), so the
// test SKIPS where it is absent (CI) and runs as a full both-directions regression on a
// maintainer checkout that still holds it.
const corpusRel = "../../.dispatch-runs"

var (
	corpusStampRE    = regexp.MustCompile(`resolve-\d+-(\d{8})-\d{6}\.log$`)
	corpusFlagRE     = regexp.MustCompile(`flag provided but not defined`)
	corpusLaunchRE   = regexp.MustCompile(`kernel-adjudicated:`)
	corpusOutageFrom = "20260728"
	corpusOutageTo   = "20260803"
	corpusCleanFrom  = "20260804"
)

// TestClassifyAgainstRealOutageCorpus grades EVERY retained resolve-*.log both
// directions against ground truth derived independently of the classifier.
//
// Ground truth is the forensic signature the #5868 count was taken on: a record is an
// outage record iff its log is stub-sized AND carries Go's flag-parse rejection. The
// classifier never sees that rule — it keys off the absence of the guard's launch
// banner — so agreement across the whole corpus is real evidence, not a tautology.
//
// The two windows are used deliberately and separately: 2026-07-28..08-03 is the
// outage, and 2026-08-04 onward is the clean fleet. A trailing-7-day window would
// straddle the outage and is never used here.
func TestClassifyAgainstRealOutageCorpus(t *testing.T) {
	logs, err := filepath.Glob(filepath.Join(corpusRel, "resolve-*.log"))
	if err != nil || len(logs) < 100 {
		t.Skipf("no local .dispatch-runs corpus (%d logs) — this regression needs the retained outage evidence", len(logs))
	}

	var graded, outageRuns, outageDOA, cleanRuns, cleanDOA int
	var falsePos, falseNeg []string
	var launchedButDOA []string
	for _, p := range logs {
		m := corpusStampRE.FindStringSubmatch(filepath.ToSlash(p))
		if m == nil {
			continue
		}
		st, serr := os.Stat(p)
		if serr != nil {
			continue
		}
		head := readHead(t, p)
		graded++
		v := Classify(head, st.Size())
		truth := st.Size() <= StubMaxBytes && corpusFlagRE.MatchString(head)

		switch day := m[1]; {
		case day >= corpusOutageFrom && day <= corpusOutageTo:
			outageRuns++
			if v.DOA {
				outageDOA++
			}
		case day >= corpusCleanFrom:
			cleanRuns++
			if v.DOA {
				cleanDOA++
			}
		}
		if v.DOA && !truth {
			falsePos = append(falsePos, p)
		}
		if truth && (!v.DOA || v.Cause != CauseFlagParse) {
			falseNeg = append(falseNeg, p)
		}
		// The discriminator itself: nothing that reached the guard's launch banner may
		// ever be graded DOA. This is the false-positive class that would matter.
		if v.DOA && corpusLaunchRE.MatchString(head) {
			launchedButDOA = append(launchedButDOA, p)
		}
	}

	t.Logf("real corpus: %d graded records | outage window %s..%s: %d DOA of %d (%.1f%%) | clean window %s+: %d DOA of %d",
		graded, corpusOutageFrom, corpusOutageTo, outageDOA, outageRuns,
		100*float64(outageDOA)/float64(max1(outageRuns)), corpusCleanFrom, cleanDOA, cleanRuns)

	if len(falsePos) > 0 {
		t.Errorf("%d false positive(s) on real records, first: %s", len(falsePos), falsePos[0])
	}
	if len(falseNeg) > 0 {
		t.Errorf("%d false negative(s) on real outage records, first: %s", len(falseNeg), falseNeg[0])
	}
	if len(launchedButDOA) > 0 {
		t.Errorf("%d record(s) that REACHED LAUNCH were graded DOA — the discriminator is broken; first: %s",
			len(launchedButDOA), launchedButDOA[0])
	}
	// FIRES on the outage: the window it was built for must be dominated by DOA.
	if outageRuns == 0 {
		t.Skip("corpus no longer retains the 2026-07-28..08-03 outage window")
	}
	if rate := float64(outageDOA) / float64(outageRuns); rate < 0.5 {
		t.Errorf("outage window graded %d DOA of %d (%.1f%%) — the detector does not see the outage it exists for",
			outageDOA, outageRuns, rate*100)
	}
	// SILENT on the healthy fleet: a detector that cries on a working fleet is worse
	// than none. The clean window is the post-outage 08-04+ slice, never a trailing 7d.
	if cleanDOA != 0 {
		t.Errorf("clean %s+ window graded %d DOA of %d — the detector fires on a healthy fleet", corpusCleanFrom, cleanDOA, cleanRuns)
	}
}

func readHead(t *testing.T, path string) string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	buf := make([]byte, HeadBytes)
	n, _ := f.Read(buf)
	if n <= 0 {
		return ""
	}
	return string(buf[:n])
}

func max1(n int) int {
	if n < 1 {
		return 1
	}
	return n
}
