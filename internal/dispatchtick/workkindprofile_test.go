package dispatchtick

import "testing"

// TestIsCoordinationWorkKind pins the coordination vocabulary. "engineering" — the
// claude DefaultWorkKind — must NEVER be coordination, or every ordinary
// implementation tick would silently downgrade to the cheap model.
func TestIsCoordinationWorkKind(t *testing.T) {
	coordination := []string{
		"project_management",
		"project-management",
		"Project Management",
		"  PROJECT_MANAGEMENT  ",
		"gardening",
		"Gardening",
	}
	for _, wk := range coordination {
		if !IsCoordinationWorkKind(wk) {
			t.Errorf("IsCoordinationWorkKind(%q) = false, want true", wk)
		}
	}
	notCoordination := []string{
		"engineering", // the claude DefaultWorkKind — the load-bearing negative
		"Engineering",
		"",
		"release",
		"pm", // the launch LABEL suffix is not a work kind
	}
	for _, wk := range notCoordination {
		if IsCoordinationWorkKind(wk) {
			t.Errorf("IsCoordinationWorkKind(%q) = true, want false", wk)
		}
	}
}

// TestLaunchProfileForDispatch pins the "PM runs on fable by default" rung: an
// unlabelled issue on a coordination tick routes to the cheap PM bucket, while
// per-issue labels still win and an engineering tick keeps the seat default.
func TestLaunchProfileForDispatch(t *testing.T) {
	cases := []struct {
		name       string
		labels     []string
		workKind   string
		wantBucket LaunchBucket
		wantProf   LaunchProfile
		wantOK     bool
	}{
		{
			// THE DEFAULT: no per-issue label at all, but the tick is a PM loop.
			name:       "unlabelled issue on a project_management tick -> fable",
			labels:     nil,
			workKind:   WorkKindProjectManagement,
			wantBucket: BucketPM,
			wantProf:   ProfileFableXHigh,
			wantOK:     true,
		},
		{
			name:       "unlabelled issue on a gardening tick -> fable",
			labels:     []string{"area/dispatch", "priority/P2"},
			workKind:   WorkKindGardening,
			wantBucket: BucketPM,
			wantProf:   ProfileFableXHigh,
			wantOK:     true,
		},
		{
			// THE SAFETY INVARIANT: an ordinary implementation tick is untouched.
			name:     "unlabelled issue on an engineering tick keeps the seat default",
			labels:   []string{"area/dispatch"},
			workKind: "engineering",
			wantOK:   false,
		},
		{
			name:     "unlabelled issue with an empty work kind keeps the seat default",
			labels:   nil,
			workKind: "",
			wantOK:   false,
		},
		{
			// Labels WIN over the work kind: a hard planning issue still escalates.
			name:       "valid hard tier beats a project_management tick -> opus+ultracode",
			labels:     []string{"tier/T0-required", "tier/T0-optimal"},
			workKind:   WorkKindProjectManagement,
			wantBucket: BucketHard,
			wantProf:   ProfileOpusUltracode,
			wantOK:     true,
		},
		{
			name:       "ultra label beats a project_management tick",
			labels:     []string{"tier/ultra"},
			workKind:   WorkKindProjectManagement,
			wantBucket: BucketUltra,
			wantProf:   ProfileFableUltracode,
			wantOK:     true,
		},
		{
			// A normal T1 issue on a PM tick follows its explicit tier, not the tick.
			name:       "valid normal tier beats a gardening tick -> opus+xhigh",
			labels:     []string{"tier/T1-required", "tier/T1-optimal"},
			workKind:   WorkKindGardening,
			wantBucket: BucketNormal,
			wantProf:   ProfileOpusXHigh,
			wantOK:     true,
		},
		{
			// The label path still works on an engineering tick.
			name:       "tier/pm label routes to fable even on an engineering tick",
			labels:     []string{"tier/pm"},
			workKind:   "engineering",
			wantBucket: BucketPM,
			wantProf:   ProfileFableXHigh,
			wantOK:     true,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			prof, bucket, ok := LaunchProfileForDispatch(c.labels, c.workKind, DefaultTierLaunchTable())
			if ok != c.wantOK {
				t.Fatalf("hasProfile=%v want %v (labels=%v workKind=%q)", ok, c.wantOK, c.labels, c.workKind)
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

// TestLaunchProfileForDispatchTableFallback confirms the work-kind rung honors a nil or
// PM-less override table by filling BucketPM from the built-in default.
func TestLaunchProfileForDispatchTableFallback(t *testing.T) {
	prof, bucket, ok := LaunchProfileForDispatch(nil, WorkKindProjectManagement, nil)
	if !ok || bucket != BucketPM || prof != ProfileFableXHigh {
		t.Fatalf("nil table: got prof=%+v bucket=%q ok=%v", prof, bucket, ok)
	}
	override := TierLaunchTable{BucketRoutine: {Model: WorkerModelFable, Effort: "high"}}
	pp, pb, pok := LaunchProfileForDispatch(nil, WorkKindGardening, override)
	if !pok || pb != BucketPM || pp != ProfileFableXHigh {
		t.Fatalf("override fallback: got prof=%+v bucket=%q ok=%v", pp, pb, pok)
	}
}
