package repoguard

import (
	"reflect"
	"testing"
)

// TestDefaultSeverity pins the permissive default posture: no rung hard-denies,
// the routine/false-positive-prone rungs are silent-record, the wasted-turn rungs
// warn, and any UNKNOWN reason fails safe to deny.
func TestDefaultSeverity(t *testing.T) {
	cases := map[string]Severity{
		guardReason:                 SeverityRecord, // OUT_OF_TREE_WRITE
		ReasonLiveMonitorOutputRead: SeverityRecord,
		ReasonInteractiveHang:       SeverityWarn,
		ReasonForegroundSleep:       SeverityWarn,
		"SOME_FUTURE_REASON":        SeverityDeny, // unknown → fail safe
	}
	for reason, want := range cases {
		if got := DefaultSeverity(reason); got != want {
			t.Errorf("DefaultSeverity(%q) = %v, want %v", reason, got, want)
		}
	}
}

// TestResolveSeverity_Precedence exercises the resolver's four-step precedence.
func TestResolveSeverity_Precedence(t *testing.T) {
	deny := map[string]Severity{guardReason: SeverityDeny}
	tests := []struct {
		name   string
		reason string
		over   map[string]Severity
		mode   string
		want   Severity
	}{
		{"default enforce, no override", guardReason, nil, "enforce", SeverityRecord},
		{"empty mode == enforce", guardReason, nil, "", SeverityRecord},
		{"per-reason deny escalates", guardReason, deny, "enforce", SeverityDeny},
		{"global off wins over deny override", guardReason, deny, "off", SeverityOff},
		{"global warn CAPS a deny override", guardReason, deny, "warn", SeverityWarn},
		{"global warn leaves record below the cap", guardReason, nil, "warn", SeverityRecord},
		{"unknown reason denies by default", "MYSTERY", nil, "enforce", SeverityDeny},
		{"global warn caps an unknown-reason deny", "MYSTERY", nil, "warn", SeverityWarn},
	}
	for _, tc := range tests {
		if got := ResolveSeverity(tc.reason, tc.over, tc.mode); got != tc.want {
			t.Errorf("%s: ResolveSeverity(%q,%v,%q) = %v, want %v",
				tc.name, tc.reason, tc.over, tc.mode, got, tc.want)
		}
	}
}

func TestParseSeverity(t *testing.T) {
	ok := map[string]Severity{
		"off": SeverityOff, "OFF": SeverityOff,
		"record": SeverityRecord, " Record ": SeverityRecord,
		"warn": SeverityWarn, "advisory": SeverityWarn,
		"deny": SeverityDeny, "enforce": SeverityDeny, "block": SeverityDeny,
	}
	for in, want := range ok {
		got, valid := ParseSeverity(in)
		if !valid || got != want {
			t.Errorf("ParseSeverity(%q) = (%v,%v), want (%v,true)", in, got, valid, want)
		}
	}
	for _, bad := range []string{"", "loud", "kill", "warnx"} {
		if _, valid := ParseSeverity(bad); valid {
			t.Errorf("ParseSeverity(%q) should be invalid", bad)
		}
	}
}

func TestParseSeverityOverrides(t *testing.T) {
	tests := []struct {
		name string
		spec string
		want map[string]Severity
	}{
		{"empty", "", nil},
		{"blank", "   ", nil},
		{"single", "OUT_OF_TREE_WRITE=deny", map[string]Severity{"OUT_OF_TREE_WRITE": SeverityDeny}},
		{"multi", "OUT_OF_TREE_WRITE=deny,INTERACTIVE_HANG=off",
			map[string]Severity{"OUT_OF_TREE_WRITE": SeverityDeny, "INTERACTIVE_HANG": SeverityOff}},
		{"lowercases reason to upper, level tolerant",
			"out_of_tree_write=Record", map[string]Severity{"OUT_OF_TREE_WRITE": SeverityRecord}},
		{"skips malformed pairs but keeps good ones",
			"OUT_OF_TREE_WRITE=deny,garbage,NOEQ,X=notalevel,INTERACTIVE_HANG=warn",
			map[string]Severity{"OUT_OF_TREE_WRITE": SeverityDeny, "INTERACTIVE_HANG": SeverityWarn}},
		{"all-malformed yields nil", "garbage,=deny,X=nope", nil},
		{"whitespace around pairs", " OUT_OF_TREE_WRITE = deny , INTERACTIVE_HANG = off ",
			map[string]Severity{"OUT_OF_TREE_WRITE": SeverityDeny, "INTERACTIVE_HANG": SeverityOff}},
	}
	for _, tc := range tests {
		got := ParseSeverityOverrides(tc.spec)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: ParseSeverityOverrides(%q) = %v, want %v", tc.name, tc.spec, got, tc.want)
		}
	}
}

func TestSeverity_StringAndLabel(t *testing.T) {
	strs := map[Severity]string{
		SeverityOff: "off", SeverityRecord: "record", SeverityWarn: "warn", SeverityDeny: "deny",
	}
	for s, want := range strs {
		if got := s.String(); got != want {
			t.Errorf("Severity(%d).String() = %q, want %q", s, got, want)
		}
	}
	labels := map[Severity]string{
		SeverityOff: "", SeverityRecord: "record", SeverityWarn: "advisory", SeverityDeny: "deny",
	}
	for s, want := range labels {
		if got := s.DecisionLabel(); got != want {
			t.Errorf("Severity(%d).DecisionLabel() = %q, want %q", s, got, want)
		}
	}
}
