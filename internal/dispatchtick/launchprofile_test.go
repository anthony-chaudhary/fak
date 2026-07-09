package dispatchtick

import "testing"

// TestLaunchProfileForIssue pins the per-issue label -> launch profile mapping across
// the four canonical buckets, the self-sufficient ultra label, and the conservative
// degrade for untagged/ambiguous issues (which must keep the seat default, not uplift).
func TestLaunchProfileForIssue(t *testing.T) {
	cases := []struct {
		name       string
		labels     []string
		wantBucket LaunchBucket
		wantProf   LaunchProfile
		wantOK     bool
	}{
		{
			name:       "routine T2 -> fable+xhigh",
			labels:     []string{"tier/T2-required", "tier/T2-optimal"},
			wantBucket: BucketRoutine,
			wantProf:   ProfileFableXHigh,
			wantOK:     true,
		},
		{
			name:       "normal T1 -> opus+xhigh",
			labels:     []string{"tier/T1-required", "tier/T1-optimal"},
			wantBucket: BucketNormal,
			wantProf:   ProfileOpusXHigh,
			wantOK:     true,
		},
		{
			name:       "hard T0 -> opus+ultracode",
			labels:     []string{"tier/T0-required", "tier/T0-optimal"},
			wantBucket: BucketHard,
			wantProf:   ProfileOpusUltracode,
			wantOK:     true,
		},
		{
			name:       "optimal more demanding than required drives the bucket",
			labels:     []string{"tier/T1-required", "tier/T0-optimal"},
			wantBucket: BucketHard,
			wantProf:   ProfileOpusUltracode,
			wantOK:     true,
		},
		{
			name:       "ultra label with T0 tier -> fable+ultracode",
			labels:     []string{"tier/T0-required", "tier/T0-optimal", "tier/ultra"},
			wantBucket: BucketUltra,
			wantProf:   ProfileFableUltracode,
			wantOK:     true,
		},
		{
			name:       "ultra label alone is self-sufficient",
			labels:     []string{"tier/ultra"},
			wantBucket: BucketUltra,
			wantProf:   ProfileFableUltracode,
			wantOK:     true,
		},
		{
			name:       "ultra label is case-insensitive",
			labels:     []string{"Tier/ULTRA"},
			wantBucket: BucketUltra,
			wantProf:   ProfileFableUltracode,
			wantOK:     true,
		},
		{
			name:   "untagged issue keeps the seat default",
			labels: []string{"priority/P1", "area/dispatch"},
			wantOK: false,
		},
		{
			name:   "contradictory tags degrade (no uplift)",
			labels: []string{"tier/T0-required", "tier/T1-optimal"},
			wantOK: false,
		},
		{
			name:   "required-only (optimal missing) degrades",
			labels: []string{"tier/T2-required"},
			wantOK: false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			prof, bucket, ok := LaunchProfileForIssue(c.labels, DefaultTierLaunchTable())
			if ok != c.wantOK {
				t.Fatalf("hasProfile=%v want %v (labels=%v)", ok, c.wantOK, c.labels)
			}
			if !ok {
				return
			}
			if bucket != c.wantBucket {
				t.Fatalf("bucket=%q want %q", bucket, c.wantBucket)
			}
			if prof != c.wantProf {
				t.Fatalf("profile=%+v want %+v", prof, c.wantProf)
			}
		})
	}
}

// TestLaunchProfileTableFallback confirms a nil or partial override table fills each
// undefined bucket from the built-in default, so a defined bucket always resolves.
func TestLaunchProfileTableFallback(t *testing.T) {
	// Nil table -> default for every bucket.
	prof, bucket, ok := LaunchProfileForIssue([]string{"tier/T2-required", "tier/T2-optimal"}, nil)
	if !ok || bucket != BucketRoutine || prof != ProfileFableXHigh {
		t.Fatalf("nil table: got prof=%+v bucket=%q ok=%v", prof, bucket, ok)
	}

	// Partial override redefines only routine; a hard issue still resolves to the
	// default hard profile.
	override := TierLaunchTable{
		BucketRoutine: {Model: WorkerModelFable, Effort: "high"},
	}
	rp, rb, rok := LaunchProfileForIssue([]string{"tier/T2-required", "tier/T2-optimal"}, override)
	if !rok || rb != BucketRoutine || rp.Effort != "high" {
		t.Fatalf("override routine: got prof=%+v bucket=%q ok=%v", rp, rb, rok)
	}
	hp, hb, hok := LaunchProfileForIssue([]string{"tier/T0-required", "tier/T0-optimal"}, override)
	if !hok || hb != BucketHard || hp != ProfileOpusUltracode {
		t.Fatalf("override hard fallback: got prof=%+v bucket=%q ok=%v", hp, hb, hok)
	}
}

// TestCanonicalProfilesUseRealModelIDs guards the crash-loop invariant: worker model
// ids are always the versioned form, never a bare alias.
func TestCanonicalProfilesUseRealModelIDs(t *testing.T) {
	if WorkerModelFable == "fable" || WorkerModelOpus == "opus" {
		t.Fatalf("worker model ids must be versioned, got opus=%q fable=%q", WorkerModelOpus, WorkerModelFable)
	}
	for _, p := range []LaunchProfile{ProfileOpusXHigh, ProfileOpusUltracode, ProfileFableXHigh, ProfileFableUltracode} {
		if p.Ultracode && p.Effort != "" {
			t.Fatalf("profile %+v sets both ultracode and effort; they are mutually exclusive on emit", p)
		}
		if !p.Ultracode && p.Effort == "" {
			t.Fatalf("non-ultracode profile %+v must set an effort", p)
		}
	}
}
