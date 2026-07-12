package gateway

import "testing"

// TestScaleToZeroWarmResume is the committed witness for #2853 (Track D of #2834): a
// scale-to-zero going-idle -> wake cycle priced two ways. The Hermes cold-hibernate arm
// wakes with a cold KV cache and reads fraction 0; the fak warm-restore arm wakes through
// the going-idle descriptor and serves the resident prefix from cache. The lift is the
// warm-on-wake continuity a serverless hibernate structurally cannot get.
func TestScaleToZeroWarmResume(t *testing.T) {
	r := MeasureScaleToZeroWarmResume(DefaultScaleToZeroScenario)

	// The Hermes arm is cold on wake by construction: nothing served from cache, the whole
	// prompt cold-created, fraction 0, never warm.
	if r.HermesColdHibernate.CacheReadTokens != 0 {
		t.Fatalf("cold hibernate must read 0 from cache, got %d", r.HermesColdHibernate.CacheReadTokens)
	}
	if r.HermesColdHibernate.CacheReadFraction != 0 {
		t.Fatalf("cold hibernate fraction must be 0, got %g", r.HermesColdHibernate.CacheReadFraction)
	}
	if r.HermesColdHibernate.CacheCreationTokens != DefaultScaleToZeroScenario.ResumeTurnPromptTokens {
		t.Fatalf("cold hibernate must cold-create the whole prompt %d, got %d",
			DefaultScaleToZeroScenario.ResumeTurnPromptTokens, r.HermesColdHibernate.CacheCreationTokens)
	}
	if r.HermesColdHibernate.Warm {
		t.Fatal("cold hibernate must never be witnessed warm")
	}

	// The fak arm serves the resident prefix from the restored cache: read == resident,
	// cold-creation is only the short new turn, fraction clears the warm floor.
	if r.FakWarmRestore.CacheReadTokens != DefaultScaleToZeroScenario.ResidentPrefixTokens {
		t.Fatalf("warm restore must read the resident prefix %d from cache, got %d",
			DefaultScaleToZeroScenario.ResidentPrefixTokens, r.FakWarmRestore.CacheReadTokens)
	}
	wantCreate := DefaultScaleToZeroScenario.ResumeTurnPromptTokens - DefaultScaleToZeroScenario.ResidentPrefixTokens
	if r.FakWarmRestore.CacheCreationTokens != wantCreate {
		t.Fatalf("warm restore must cold-create only the new turn %d, got %d", wantCreate, r.FakWarmRestore.CacheCreationTokens)
	}
	if !r.FakWarmRestore.Warm {
		t.Fatalf("warm restore must be witnessed warm, fraction %g < floor %g",
			r.FakWarmRestore.CacheReadFraction, r.WarmResumeFloor)
	}

	// The witnessed comparison: the warm arm reads materially more of its prompt from cache
	// than the cold baseline — the warm-on-wake lift the issue asks be witnessed.
	if r.CacheReadFractionLift <= 0 {
		t.Fatalf("warm restore must lift the cache-read fraction over the cold baseline, got %g", r.CacheReadFractionLift)
	}
	if r.CacheReadFractionLift != r.FakWarmRestore.CacheReadFraction-r.HermesColdHibernate.CacheReadFraction {
		t.Fatal("lift must equal fak fraction minus hermes fraction")
	}
	if r.WarmResumeFloor != WarmResumeFloor {
		t.Fatalf("witness must carry the shared warm floor %g, got %g", WarmResumeFloor, r.WarmResumeFloor)
	}
}

// TestGoingIdlePersistNormalizes checks the going-idle KV-persist hook: it stamps the
// schema, trims, defaults the TTL tier to 5m, and floors a negative resident length so a
// malformed capture can never mint a negative-token descriptor.
func TestGoingIdlePersistNormalizes(t *testing.T) {
	d := GoingIdlePersist("  sess-1  ", "  prefix:sess-1  ", -5, CacheTTL(""))
	if d.Schema != scaleToZeroWarmResumeSchema {
		t.Fatalf("schema not stamped: %q", d.Schema)
	}
	if d.Trace != "sess-1" || d.PrefixDigest != "prefix:sess-1" {
		t.Fatalf("trace/digest not trimmed: %q %q", d.Trace, d.PrefixDigest)
	}
	if d.ResidentPrefixTokens != 0 {
		t.Fatalf("negative resident length must floor to 0, got %d", d.ResidentPrefixTokens)
	}
	if d.TTLTier != CacheTTL5m {
		t.Fatalf("empty TTL must default to 5m, got %q", d.TTLTier)
	}
	if d1 := GoingIdlePersist("s", "p", 10, CacheTTL1h); d1.TTLTier != CacheTTL1h {
		t.Fatalf("1h tier must be preserved, got %q", d1.TTLTier)
	}
}

// TestWakeWarmRestoreEmptyDigestReadsCold guards the honesty floor: a going-idle that
// captured NO prefix (empty digest) cannot be restored, so its wake reads cold (fraction 0)
// and is never mislabeled warm — the same discipline that keeps a cold re-send honest.
func TestWakeWarmRestoreEmptyDigestReadsCold(t *testing.T) {
	empty := GoingIdlePersist("s", "", 8000, CacheTTL5m)
	arm := WakeWarmRestore(empty, 8192, -1)
	if arm.CacheReadTokens != 0 || arm.CacheReadFraction != 0 || arm.Warm {
		t.Fatalf("empty-digest wake must read cold, got read=%d frac=%g warm=%v",
			arm.CacheReadTokens, arm.CacheReadFraction, arm.Warm)
	}
	if arm.CacheCreationTokens != 8192 {
		t.Fatalf("empty-digest wake must cold-create the whole prompt, got %d", arm.CacheCreationTokens)
	}
}

// TestWakeWarmRestoreObservedBill exercises the promotion/dogfood path: feeding the OBSERVED
// provider cache_read for a live wake turn prices the fraction from real bytes rather than
// the model. A below-floor observed read is honestly reported as not-warm; a clamp guards an
// upstream cache_read that overshoots the prompt.
func TestWakeWarmRestoreObservedBill(t *testing.T) {
	d := GoingIdlePersist("s", "prefix:s", 8000, CacheTTL5m)

	// Observed warm: provider served 7,000 of an 8,192-token turn from the restored prefix.
	warm := WakeWarmRestore(d, 8192, 7000)
	if warm.CacheReadTokens != 7000 || !warm.Warm {
		t.Fatalf("observed 7000/8192 must be warm, got read=%d frac=%g warm=%v",
			warm.CacheReadTokens, warm.CacheReadFraction, warm.Warm)
	}

	// Observed cold-ish: provider served only 1,000 — below the floor, reported not-warm.
	cool := WakeWarmRestore(d, 8192, 1000)
	if cool.Warm {
		t.Fatalf("observed 1000/8192 is below the warm floor and must not be warm, frac=%g", cool.CacheReadFraction)
	}

	// Clamp: an upstream cache_read exceeding the prompt is capped at the prompt (fraction 1).
	over := WakeWarmRestore(d, 8192, 99999)
	if over.CacheReadTokens != 8192 || over.CacheReadFraction != 1 {
		t.Fatalf("over-prompt cache_read must clamp to the prompt, got read=%d frac=%g",
			over.CacheReadTokens, over.CacheReadFraction)
	}
}
