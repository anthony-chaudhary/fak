package adjudicator

import "testing"

// NeverAdmits is the args-independent "this tool NAME can never be Allowed" query the
// inbound tool-def compactor (promptmmu) asks before it may drop a tool DEFINITION. The
// contract: true ONLY when no argument value could make the name Allow, and fail-safe
// false against an unconfigured floor (so an empty Policy never triggers a mass over-drop).
func TestNeverAdmits(t *testing.T) {
	floor := Policy{
		Posture:     PostureFailClosed,
		Allow:       map[string]bool{"Bash": true, "Edit": true},
		AllowPrefix: []string{"read_", "get_"},
	}

	cases := []struct {
		name string
		tool string
		want bool
		why  string
	}{
		{"explicitly allowed", "Bash", false, "named in Allow → reachable, never drop"},
		{"prefix allowed", "read_file", false, "matches an AllowPrefix → reachable, never drop"},
		{"prefix allowed 2", "get_thing", false, "matches an AllowPrefix → reachable"},
		{"floor-denied", "WebFetch", true, "absent from Allow, matches no prefix → DEFAULT_DENY for every arg, droppable"},
		{"floor-denied 2", "rm_rf", true, "not affirmatively allowed → never reachable"},
		{"empty name", "", true, "the empty name is not allowed and not read-shaped → never admitted"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := floor.NeverAdmits(c.tool); got != c.want {
				t.Fatalf("NeverAdmits(%q) = %v, want %v (%s)", c.tool, got, c.want, c.why)
			}
		})
	}
}

// The fail-safe: a Policy with no affirmative-allow surface at all (empty Allow AND empty
// AllowPrefix) denies every tool — but as a DROP signal that almost always means "no floor
// installed," so NeverAdmits must report FALSE for everything rather than mark all tools
// droppable (which would prune every tool-def against a zero floor).
func TestNeverAdmitsRefusesToPruneAgainstUnconfiguredFloor(t *testing.T) {
	for _, p := range []Policy{
		{},                           // zero value
		{Posture: PostureFailClosed}, // fail-closed but no allow surface
		{Allow: map[string]bool{}},   // explicitly empty allow
	} {
		for _, tool := range []string{"Bash", "WebFetch", "anything", ""} {
			if p.NeverAdmits(tool) {
				t.Fatalf("NeverAdmits(%q) on an unconfigured floor = true; must be false (fail-safe, no mass over-drop). policy=%+v", tool, p)
			}
		}
	}
}

// Under PostureAdmitAndLog a low-risk read-shaped DEFAULT_DENY is downgraded to Allow, so
// such a name CAN still be invoked — it must NOT be marked droppable even though it is
// absent from Allow. A write-shaped name under the same posture is still hard-denied and
// remains droppable. This is the posture half of the "behavior-preserving by construction"
// guarantee: NeverAdmits never drops a tool the running floor could Allow.
func TestNeverAdmitsRespectsAdmitAndLogPosture(t *testing.T) {
	floor := Policy{
		Posture:     PostureAdmitAndLog,
		AllowPrefix: []string{"keep_"}, // a real allow surface so the unconfigured-floor guard is off
	}
	// A read-shaped name is admitted-and-logged → reachable → NOT droppable.
	if floor.NeverAdmits("read_secrets") {
		t.Fatalf("a read-shaped name under PostureAdmitAndLog is reachable (admit-and-log); NeverAdmits must be false, not drop it")
	}
	// A write-shaped name is still hard default-denied under this posture → droppable.
	if !floor.NeverAdmits("write_file") {
		t.Fatalf("a write-shaped name under PostureAdmitAndLog is still DEFAULT_DENY for every arg; NeverAdmits should be true")
	}
}
