package superloop

import "testing"

// TestFrontDoorForClassifiesEachKind pins the four-way classification the drive's
// execution rung branches on: a container descends, a "/skill" is agent-only, an empty
// Enter is nothing to run, and a concrete command line is runnable.
func TestFrontDoorForClassifiesEachKind(t *testing.T) {
	cases := []struct {
		name    string
		dec     DriveDecision
		want    FrontDoorKind
		wantCmd string
	}{
		{
			name: "container descends, never executes a subtree",
			dec:  DriveDecision{Container: true, Member: Member{Kind: KindSuperloop, Ref: "drain-issues", Enter: "ignored"}},
			want: FrontDescend,
		},
		{
			name: "skill front door is agent-only",
			dec:  DriveDecision{Member: Member{Kind: KindScorecard, Ref: "slop", Enter: "/slop-score"}},
			want: FrontSkill, wantCmd: "/slop-score",
		},
		{
			name: "runnable shell command",
			dec:  DriveDecision{Member: Member{Kind: KindLoop, Ref: "throughput", Enter: "go run ./cmd/fak dispatch auto --goal throughput"}},
			want: FrontRunnable, wantCmd: "go run ./cmd/fak dispatch auto --goal throughput",
		},
		{
			name: "compound command line is still runnable (a shell runs it)",
			dec:  DriveDecision{Member: Member{Kind: KindUtilization, Ref: "account-limits", Enter: "go run ./cmd/fak accounts next && go run ./cmd/fak dispatch auto --goal throughput"}},
			want: FrontRunnable, wantCmd: "go run ./cmd/fak accounts next && go run ./cmd/fak dispatch auto --goal throughput",
		},
		{
			name: "python scorecard is runnable",
			dec:  DriveDecision{Member: Member{Kind: KindScorecard, Ref: "learning", Enter: "python tools/learning_scorecard.py --json"}},
			want: FrontRunnable, wantCmd: "python tools/learning_scorecard.py --json",
		},
		{
			name: "no Enter declared is nothing to run",
			dec:  DriveDecision{Member: Member{Kind: KindLoop, Ref: "cadence"}},
			want: FrontNone,
		},
		{
			name: "whitespace-only Enter is nothing to run",
			dec:  DriveDecision{Member: Member{Kind: KindLoop, Ref: "dojo", Enter: "   "}},
			want: FrontNone,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := FrontDoorFor(tc.dec)
			if got.Kind != tc.want {
				t.Errorf("kind = %q, want %q (note: %s)", got.Kind, tc.want, got.Note)
			}
			if got.Command != tc.wantCmd {
				t.Errorf("command = %q, want %q", got.Command, tc.wantCmd)
			}
			if got.Runnable() != (tc.want == FrontRunnable) {
				t.Errorf("Runnable() = %v for kind %q", got.Runnable(), got.Kind)
			}
			if got.Note == "" {
				t.Error("every classification must carry a one-line note")
			}
		})
	}
}

// TestFrontDoorForEveryRegistryMember is a no-drift witness: every member of every
// registered intent classifies into exactly one kind, and a container member (the walk
// surfaces sub-super-loops/gardens as descend pointers) is never mistaken for runnable.
// It also proves the registry's declared Enter hints are well-formed for the exec rung:
// a "/"-prefixed Enter is a skill, a non-empty non-slash Enter is runnable.
func TestFrontDoorForEveryRegistryMember(t *testing.T) {
	for _, s := range Registry() {
		for _, m := range s.Members {
			// A garden member is always a container; a super-loop member is descended by
			// the shell (measured), but as a raw member its Enter is empty. Model the
			// worst case (container) and the leaf case (not) and require both stay honest.
			container := m.Kind == KindGarden || m.Kind == KindSuperloop
			fd := FrontDoorFor(DriveDecision{Member: m, Container: container})
			if container {
				if fd.Kind != FrontDescend {
					t.Errorf("%s/%s (%s) as container: kind %q, want descend", s.Name, m.Ref, m.Kind, fd.Kind)
				}
				continue
			}
			enter := m.Enter
			switch {
			case enter == "":
				if fd.Kind != FrontNone {
					t.Errorf("%s/%s: empty Enter classified %q, want none", s.Name, m.Ref, fd.Kind)
				}
			case enter[0] == '/':
				if fd.Kind != FrontSkill {
					t.Errorf("%s/%s: skill %q classified %q, want skill", s.Name, m.Ref, enter, fd.Kind)
				}
			default:
				if fd.Kind != FrontRunnable {
					t.Errorf("%s/%s: command %q classified %q, want runnable", s.Name, m.Ref, enter, fd.Kind)
				}
			}
		}
	}
}
