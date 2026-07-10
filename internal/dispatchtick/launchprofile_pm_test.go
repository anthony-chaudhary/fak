package dispatchtick

import "testing"

// TestLaunchProfilePMLabel pins the project-management launch label (PMLabel): a bare
// PM issue routes to the cheap fable coordination bucket, but the label YIELDS to a
// valid tier tag so a genuinely hard PM planning issue still escalates to opus. This
// is the launch-side expression of the modelroute pm-fable preset's
// work_kind=project_management, and the "PM -> fable by default" behavior.
func TestLaunchProfilePMLabel(t *testing.T) {
	cases := []struct {
		name       string
		labels     []string
		wantBucket LaunchBucket
		wantProf   LaunchProfile
		wantOK     bool
	}{
		{
			name:       "pm label alone -> fable+xhigh (the default PM route)",
			labels:     []string{"tier/pm"},
			wantBucket: BucketPM,
			wantProf:   ProfileFableXHigh,
			wantOK:     true,
		},
		{
			name:       "pm label is case-insensitive",
			labels:     []string{"Tier/PM"},
			wantBucket: BucketPM,
			wantProf:   ProfileFableXHigh,
			wantOK:     true,
		},
		{
			name:       "pm label alongside unrelated labels still routes to fable",
			labels:     []string{"area/dispatch", "tier/pm", "priority/P2"},
			wantBucket: BucketPM,
			wantProf:   ProfileFableXHigh,
			wantOK:     true,
		},
		{
			// The escalation path: an explicit, valid hard tier WINS over the PM default,
			// so a hard planning issue tagged tier/T0 runs on opus, not fable.
			name:       "pm label yields to a valid hard tier -> opus+ultracode",
			labels:     []string{"tier/pm", "tier/T0-required", "tier/T0-optimal"},
			wantBucket: BucketHard,
			wantProf:   ProfileOpusUltracode,
			wantOK:     true,
		},
		{
			name:       "pm label with a valid routine tier follows the tier bucket",
			labels:     []string{"tier/pm", "tier/T2-required", "tier/T2-optimal"},
			wantBucket: BucketRoutine,
			wantProf:   ProfileFableXHigh,
			wantOK:     true,
		},
		{
			// Ultra is the most explicit signal and is checked before PM.
			name:       "ultra label beats a co-tagged pm label",
			labels:     []string{"tier/pm", "tier/ultra"},
			wantBucket: BucketUltra,
			wantProf:   ProfileFableUltracode,
			wantOK:     true,
		},
		{
			// A broken/ambiguous tier tag is not trusted (HasTier=false); the PM label
			// then routes the issue to fable rather than the seat default.
			name:       "pm label with an ambiguous tier tag falls to the PM route",
			labels:     []string{"tier/pm", "tier/T2-required"},
			wantBucket: BucketPM,
			wantProf:   ProfileFableXHigh,
			wantOK:     true,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			prof, bucket, ok := LaunchProfileForIssue(c.labels, DefaultTierLaunchTable())
			if ok != c.wantOK {
				t.Fatalf("hasProfile=%v want %v (labels=%v)", ok, c.wantOK, c.labels)
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

// TestPMLabelInertToTierGrammar guards that PMLabel is NOT parsed as a T<N> tier by the
// tiertag grammar, so it can only ever select BucketPM (never a tier bucket) and never
// spuriously trips HasTier.
func TestPMLabelInertToTierGrammar(t *testing.T) {
	it, _ := IssueTierFromLabels([]string{PMLabel})
	if it.HasTier {
		t.Fatalf("PMLabel %q must not resolve a tier, got HasTier=true (%+v)", PMLabel, it)
	}
}

// TestPMBucketFallsBackToDefault confirms a nil or PM-less override table still resolves
// a PM issue to the built-in fable coordination profile, so the bucket always resolves.
func TestPMBucketFallsBackToDefault(t *testing.T) {
	// Nil table -> default PM profile.
	prof, bucket, ok := LaunchProfileForIssue([]string{"tier/pm"}, nil)
	if !ok || bucket != BucketPM || prof != ProfileFableXHigh {
		t.Fatalf("nil table PM: got prof=%+v bucket=%q ok=%v", prof, bucket, ok)
	}
	// An override table that redefines only routine still fills BucketPM from the default.
	override := TierLaunchTable{BucketRoutine: {Model: WorkerModelFable, Effort: "high"}}
	pp, pb, pok := LaunchProfileForIssue([]string{"tier/pm"}, override)
	if !pok || pb != BucketPM || pp != ProfileFableXHigh {
		t.Fatalf("override PM fallback: got prof=%+v bucket=%q ok=%v", pp, pb, pok)
	}
}
