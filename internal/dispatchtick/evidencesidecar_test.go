package dispatchtick

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func TestASidecarPathReplacesTheLogExtensionAndNothingElse(t *testing.T) {
	for _, tc := range []struct{ log, want string }{
		{"/runs/w-12.log", "/runs/w-12.zone"},
		{`C:\runs\w-12.log`, `C:\runs\w-12.zone`},
		// A dotted DIRECTORY must not eat the stem. If it did, every slot in the wave would
		// collapse onto one sidecar and the last worker to exit would overwrite the rest.
		{"/runs/wave-1.2/w-12.log", "/runs/wave-1.2/w-12.zone"},
		{"/runs/wave-1.2/w-12", "/runs/wave-1.2/w-12.zone"},
		// Nothing to key on: a bare suffix in the runs directory is worse than no sidecar,
		// because every unlogged slot would share it.
		{"", ""},
		{"   ", ""},
	} {
		if got := SidecarPath(tc.log, ZoneSidecarSuffix); got != tc.want {
			t.Errorf("SidecarPath(%q) = %q, want %q", tc.log, got, tc.want)
		}
	}
}

func TestEverySidecarSuffixIsDistinct(t *testing.T) {
	// Two facts sharing a suffix is a silent overwrite: the second writer wins and the sweep
	// reads one fact where it believes it read two.
	seen := map[string]string{}
	for name, suffix := range map[string]string{
		"wave":      WaveSidecarSuffix,
		"account":   AccountSidecarSuffix,
		"basesha":   BaseSHASidecarSuffix,
		"model":     ModelSidecarSuffix,
		"zone":      ZoneSidecarSuffix,
		"workclass": WorkClassSidecarSuffix,
	} {
		if !strings.HasPrefix(suffix, ".") {
			t.Errorf("%s suffix %q does not start with a dot — SidecarPath would fuse it to the stem", name, suffix)
		}
		if prev, dup := seen[suffix]; dup {
			t.Errorf("%s and %s both write %q", name, prev, suffix)
		}
		seen[suffix] = name
	}
}

func TestOnlyTheThreeKnownRungsParseBackOutOfAZoneSidecar(t *testing.T) {
	for _, z := range modelroute.Zones() {
		got, ok := ZoneFromSidecar(string(z) + "\n")
		if !ok || got != z {
			t.Errorf("round trip of %q = %q/%v", z, got, ok)
		}
	}
	// The refusals. "devi" is a crash mid-write; "local" is the ENGINE-route family name
	// rather than a rung, and reading it as device is precisely how a vendor call gets
	// counted as on-box.
	for _, raw := range []string{"", "   ", "devi", "DEVICE", "local", "on-device", "self-hosted", "vendor extra"} {
		if got, ok := ZoneFromSidecar(raw); ok || got != "" {
			t.Errorf("ZoneFromSidecar(%q) = %q/%v, want unattributed", raw, got, ok)
		}
	}
}

func TestNoSidecarCanMintTheSecurityReleaseClass(t *testing.T) {
	// The floor on ClassSecurityRelease is what stops a cheap model serving push/delete/
	// release work. A file on disk is the softest input in this path, so the class no
	// dispatch label can name must also be one no file can name.
	for _, raw := range []string{
		string(modelroute.ClassSecurityRelease),
		" " + string(modelroute.ClassSecurityRelease) + "\n",
		"security-release",
	} {
		if got, ok := WorkClassFromSidecar(raw); ok || got != "" {
			t.Errorf("WorkClassFromSidecar(%q) = %q/%v — a hand-written file minted the class whose floor guards destructive work", raw, got, ok)
		}
	}
}

func TestEveryClassASlotCanBeLaunchedUnderSurvivesItsOwnSidecar(t *testing.T) {
	// Writer/reader agreement. The spawn writes string(class); if the reader's allowlist
	// ever drifts from bucketWorkClass, every slot would write a sidecar the sweep then
	// refuses, and the fleet would report zero gradeable work with no error anywhere.
	for bucket, class := range bucketWorkClass {
		got, ok := WorkClassFromSidecar(string(class))
		if !ok || got != class {
			t.Errorf("bucket %q writes %q, which reads back as %q/%v", bucket, class, got, ok)
		}
	}
	// And the launch path's own answer round-trips end to end.
	class, why := WorkClassForIssue([]string{"tier/T0-optimal", "tier/T0-required"})
	if why != ClassFromTierLabel {
		t.Fatalf("why = %q", why)
	}
	if got, ok := WorkClassFromSidecar(string(class)); !ok || got != class {
		t.Errorf("launch class %q reads back as %q/%v", class, got, ok)
	}
}

func TestATruncatedOrEditedClassSidecarGradesNothing(t *testing.T) {
	for _, raw := range []string{"", "  ", "rout", "Routine", "tier/T2-optimal", "T0", "ultra", "normal", "ultra-hard extra"} {
		if got, ok := WorkClassFromSidecar(raw); ok || got != "" {
			t.Errorf("WorkClassFromSidecar(%q) = %q/%v, want ungraded", raw, got, ok)
		}
	}
	// Surrounding whitespace is the ordinary case (a trailing newline), not corruption.
	if got, ok := WorkClassFromSidecar(" routine \r\n"); !ok || got != modelroute.ClassRoutine {
		t.Errorf("WorkClassFromSidecar with surrounding whitespace = %q/%v", got, ok)
	}
}

func TestTheSidecarResolverReadsEachSlotsOwnLogRatherThanTheFirstOne(t *testing.T) {
	// The bug this catches is a resolver that reuses one scraped class for the sweep: a
	// single tier/T0 slot would then grade every model in the tick at the frontier tier.
	byLog := map[string]modelroute.WorkClass{
		"/runs/a.log": modelroute.ClassUltraHard,
		"/runs/b.log": modelroute.ClassRoutine,
	}
	resolve := ClassResolverFromSidecars(byLog)
	for log, want := range map[string]modelroute.WorkClass{
		"/runs/a.log": modelroute.ClassUltraHard,
		"/runs/b.log": modelroute.ClassRoutine,
		"/runs/c.log": "", // no sidecar was scraped for this slot
		"":            "",
	} {
		if got := resolve(WitnessRecord{Log: log}); got != want {
			t.Errorf("resolve(%q) = %q, want %q", log, got, want)
		}
	}
	// A sweep that scraped nothing grades nothing rather than panicking on a nil map.
	if got := ClassResolverFromSidecars(nil)(WitnessRecord{Log: "/runs/a.log"}); got != "" {
		t.Errorf("nil map resolved %q", got)
	}
}

func TestSidecarScrapedClassesFlowIntoTheProducerAndTheRestAreCounted(t *testing.T) {
	// End to end through the shipped producer: what the sweep scraped becomes evidence, and
	// what it could not scrape is dropped AND counted, never filed at the unknown-class T0
	// floor.
	records := []WitnessRecord{
		{Issue: 1, Log: "/runs/a.log", Model: "qwen3.6-4b", Zone: string(modelroute.ZoneDevice), Claim: ClaimWitnessed, TestClaim: ClaimTestGreen},
		{Issue: 2, Log: "/runs/b.log", Model: "qwen3.6-4b", Zone: string(modelroute.ZoneDevice), Claim: ClaimWitnessed, TestClaim: ClaimTestGreen},
		{Issue: 3, Log: "/runs/c.log", Model: "qwen3.6-4b", Zone: string(modelroute.ZoneDevice), Claim: ClaimWitnessed, TestClaim: ClaimTestGreen},
	}
	byLog := map[string]modelroute.WorkClass{
		"/runs/a.log": modelroute.ClassRoutine,
		"/runs/b.log": modelroute.ClassRoutine,
	}
	out, stats := TurnOutcomesFromWitness(records, WitnessEvidenceOptions{Class: ClassResolverFromSidecars(byLog)})
	if len(out) != 2 || stats.Produced != 2 || stats.Unclassified != 1 {
		t.Fatalf("out=%d stats=%+v", len(out), stats)
	}
	for _, o := range out {
		if o.Class != modelroute.ClassRoutine {
			t.Errorf("outcome graded %q, want routine", o.Class)
		}
	}
	// The zone fold reads the same records and agrees about what it could attribute.
	if s := FoldZoneShare(records); s.Attributed != 3 || s.SelfHosted() != 3 {
		t.Errorf("zone share = %+v", s)
	}
}
