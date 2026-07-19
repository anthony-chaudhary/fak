package supervisoragent

import (
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// fakeVerbs is the recording AdmissionVerbs double: it logs every deterministic
// verb invocation (name + args) and hands back the witnessed artifact that verb
// would leave (a lease row / an admit receipt / a packet head with its assigned
// id). Setting a refuse error models the underlying admission REFUSING — the
// refusal must propagate; there is no force path.
type fakeVerbs struct {
	calls []string

	arbitrateErr error
	admitErr     error
	emitErr      error
}

func (f *fakeVerbs) Arbitrate(lane string, tree []string) (Lease, error) {
	f.calls = append(f.calls, fmt.Sprintf("arbitrate(%s,%v)", lane, tree))
	if f.arbitrateErr != nil {
		return Lease{}, f.arbitrateErr
	}
	granted := tree
	if granted == nil {
		granted = []string{"internal/" + lane + "/**"}
	}
	return Lease{Lane: lane, Kind: "cluster", Tree: granted}, nil
}

func (f *fakeVerbs) Admit(issue, lane, supersedes string) (AdmitReceipt, error) {
	f.calls = append(f.calls, fmt.Sprintf("admit(%s,%s,%s)", issue, lane, supersedes))
	if f.admitErr != nil {
		return AdmitReceipt{}, f.admitErr
	}
	return AdmitReceipt{RunID: "RID-new", Issue: issue, Lane: lane, Supersedes: supersedes}, nil
}

func (f *fakeVerbs) EmitEscalation(head Escalation) (Escalation, error) {
	f.calls = append(f.calls, fmt.Sprintf("emit(%s/%s/%s)", head.RunID, head.Issue, head.ReasonCode))
	if f.emitErr != nil {
		return Escalation{}, f.emitErr
	}
	head.ID = "PKT-1"
	return head, nil
}

// TestSupervisorActionClosedVocabulary pins the closed verb set: exactly six
// action kinds, each spelled by exactly one union case type. A seventh verb (or
// a respelled token) fails this pin until the vocabulary is DELIBERATELY
// widened — the whole point of a closed union.
func TestSupervisorActionClosedVocabulary(t *testing.T) {
	got := map[ActionKind]SupervisorAction{
		SpawnAction{}.Kind():      SpawnAction{},
		ReplaceAction{}.Kind():    ReplaceAction{},
		RedispatchAction{}.Kind(): RedispatchAction{},
		WidenAction{}.Kind():      WidenAction{},
		EscalateAction{}.Kind():   EscalateAction{},
		HoldAction{}.Kind():       HoldAction{},
	}
	want := []ActionKind{ActionSpawn, ActionReplace, ActionRedispatch, ActionWiden, ActionEscalate, ActionHold}
	if len(got) != len(want) {
		t.Fatalf("action kinds collide: %d distinct kinds from 6 case types", len(got))
	}
	for _, k := range want {
		if _, ok := got[k]; !ok {
			t.Errorf("closed vocabulary is missing kind %q", k)
		}
	}
	wantTokens := []string{"escalate", "hold", "replace", "replan", "spawn", "widen"}
	var gotTokens []string
	for k := range got {
		gotTokens = append(gotTokens, string(k))
	}
	sort.Strings(gotTokens)
	if !reflect.DeepEqual(gotTokens, wantTokens) {
		t.Errorf("vocabulary tokens drifted: got %v, want %v", gotTokens, wantTokens)
	}
}

// TestActionArgsClosedFieldSet is the golden pin on each case's typed args: an
// action carries ONLY the args its deterministic verb needs — no free-text, no
// transcript, and no "extra context" field can be added silently.
func TestActionArgsClosedFieldSet(t *testing.T) {
	cases := map[string]struct {
		typ  reflect.Type
		want []string
	}{
		"SpawnAction":      {reflect.TypeOf(SpawnAction{}), []string{"Issue:string", "Lane:string"}},
		"ReplaceAction":    {reflect.TypeOf(ReplaceAction{}), []string{"Issue:string", "Lane:string", "RunID:string"}},
		"RedispatchAction": {reflect.TypeOf(RedispatchAction{}), []string{"Issue:string", "Lane:string"}},
		"WidenAction":      {reflect.TypeOf(WidenAction{}), []string{"Lane:string", "Tree[]:string"}},
		"EscalateAction":   {reflect.TypeOf(EscalateAction{}), []string{"Class:string", "Issue:string", "ReasonCode:string", "RunID:string", "Severity:string"}},
		"HoldAction":       {reflect.TypeOf(HoldAction{}), nil},
	}
	for name, c := range cases {
		var got []string
		walkFields(c.typ, "", &got)
		sort.Strings(got)
		want := append([]string(nil), c.want...)
		sort.Strings(want)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s arg fields drifted from the closed contract.\n got: %v\nwant: %v", name, got, want)
		}
	}
}

// TestActionNoPayloadFieldName extends the package's payload fence to the action
// side: no action arg, receipt, or effect field may name a transcript / message /
// free-text payload. An action that carried prose would let the agent smuggle a
// recognizer's output into the admission path.
func TestActionNoPayloadFieldName(t *testing.T) {
	forbidden := []string{
		"transcript", "message", "msg", "body", "payload", "prose",
		"freetext", "rawlog", "log", "content", "chat", "narrative",
		"summary", "comment", "text",
	}
	types := []reflect.Type{
		reflect.TypeOf(SpawnAction{}),
		reflect.TypeOf(ReplaceAction{}),
		reflect.TypeOf(RedispatchAction{}),
		reflect.TypeOf(WidenAction{}),
		reflect.TypeOf(EscalateAction{}),
		reflect.TypeOf(HoldAction{}),
		reflect.TypeOf(AdmitReceipt{}),
		reflect.TypeOf(ActionEffect{}),
	}
	for _, typ := range types {
		var names []string
		collectFieldNames(typ, &names)
		for _, n := range names {
			low := strings.ToLower(n)
			for _, bad := range forbidden {
				if strings.Contains(low, bad) {
					t.Errorf("%s field %q contains forbidden payload token %q", typ.Name(), n, bad)
				}
			}
		}
	}
}

// TestAdmissionVerbsClosedMethodSet pins the lowering surface itself: the ONLY
// deterministic verbs an action can reach are the three existing admission
// calls. No shell, no raw spawn, no side door — adding a method here is adding
// an authority, and must break this pin first.
func TestAdmissionVerbsClosedMethodSet(t *testing.T) {
	typ := reflect.TypeOf((*AdmissionVerbs)(nil)).Elem()
	var got []string
	for i := 0; i < typ.NumMethod(); i++ {
		got = append(got, typ.Method(i).Name)
	}
	sort.Strings(got)
	want := []string{"Admit", "Arbitrate", "EmitEscalation"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("AdmissionVerbs method set drifted: got %v, want %v — every method is an authority", got, want)
	}
}

// TestEachActionLowersToWitnessedArtifact drives every verb in the vocabulary
// through Lower against the recording double and asserts (a) the exact
// deterministic admission call it lowered to, and (b) the witnessed artifact
// the effect carries — a lease row for spawn/widen, an admit receipt for
// replace/replan, a packet head for escalate, and NOTHING (zero calls, zero
// artifacts) for hold.
func TestEachActionLowersToWitnessedArtifact(t *testing.T) {
	cases := []struct {
		name      string
		action    SupervisorAction
		wantCalls []string
		wantVerb  LoweredVerb
		check     func(t *testing.T, eff ActionEffect)
	}{
		{
			name:      "spawn lowers to dos_arbitrate admission and leaves the granted lease row",
			action:    SpawnAction{Issue: "4479", Lane: "supervisoragent"},
			wantCalls: []string{"arbitrate(supervisoragent,[])"},
			wantVerb:  VerbArbitrate,
			check: func(t *testing.T, eff ActionEffect) {
				if eff.Lease == nil || eff.Lease.Lane != "supervisoragent" {
					t.Errorf("spawn effect lease = %+v, want granted row for lane supervisoragent", eff.Lease)
				}
			},
		},
		{
			name:      "widen lowers to dos_arbitrate RE-arbitration (no force path) and leaves the re-granted lease row",
			action:    WidenAction{Lane: "supervisoragent", Tree: []string{"internal/supervisoragent/**", "docs/notes/**"}},
			wantCalls: []string{"arbitrate(supervisoragent,[internal/supervisoragent/** docs/notes/**])"},
			wantVerb:  VerbArbitrate,
			check: func(t *testing.T, eff ActionEffect) {
				if eff.Lease == nil || len(eff.Lease.Tree) != 2 {
					t.Errorf("widen effect lease = %+v, want re-granted row with widened tree", eff.Lease)
				}
			},
		},
		{
			name:      "replace lowers to the dispatch admit path superseding the dead run",
			action:    ReplaceAction{RunID: "RID-dead", Issue: "4479", Lane: "supervisoragent"},
			wantCalls: []string{"admit(4479,supervisoragent,RID-dead)"},
			wantVerb:  VerbAdmit,
			check: func(t *testing.T, eff ActionEffect) {
				if eff.Admit == nil || eff.Admit.Supersedes != "RID-dead" {
					t.Errorf("replace effect receipt = %+v, want admit receipt superseding RID-dead", eff.Admit)
				}
			},
		},
		{
			name:      "replan lowers to the dispatch admit path with no superseded run",
			action:    RedispatchAction{Issue: "4479", Lane: "supervisoragent"},
			wantCalls: []string{"admit(4479,supervisoragent,)"},
			wantVerb:  VerbAdmit,
			check: func(t *testing.T, eff ActionEffect) {
				if eff.Admit == nil || eff.Admit.Supersedes != "" {
					t.Errorf("replan effect receipt = %+v, want fresh admit receipt", eff.Admit)
				}
			},
		},
		{
			name:      "escalate lowers to the escalation packet emit and leaves the assigned packet head",
			action:    EscalateAction{RunID: "RID-1", Issue: "4479", Class: "stall", Severity: "operator", ReasonCode: "REQUIRE_WITNESS"},
			wantCalls: []string{"emit(RID-1/4479/REQUIRE_WITNESS)"},
			wantVerb:  VerbEmitEscalation,
			check: func(t *testing.T, eff ActionEffect) {
				if eff.Packet == nil || eff.Packet.ID == "" {
					t.Errorf("escalate effect packet = %+v, want emitted head with assigned id", eff.Packet)
				}
			},
		},
		{
			name:      "hold lowers to nothing: zero admission calls, zero artifacts",
			action:    HoldAction{},
			wantCalls: nil,
			wantVerb:  VerbNone,
			check:     func(t *testing.T, eff ActionEffect) {},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			v := &fakeVerbs{}
			eff, err := Lower(c.action, v)
			if err != nil {
				t.Fatalf("Lower(%s) error: %v", c.action.Kind(), err)
			}
			if !reflect.DeepEqual(v.calls, c.wantCalls) {
				t.Errorf("admission calls = %v, want %v", v.calls, c.wantCalls)
			}
			if eff.Action != c.action.Kind() {
				t.Errorf("effect action = %q, want %q", eff.Action, c.action.Kind())
			}
			if eff.Verb != c.wantVerb {
				t.Errorf("effect verb = %q, want %q", eff.Verb, c.wantVerb)
			}
			// Exactly one artifact for an artifact-leaving verb, none for hold.
			artifacts := 0
			if eff.Lease != nil {
				artifacts++
			}
			if eff.Admit != nil {
				artifacts++
			}
			if eff.Packet != nil {
				artifacts++
			}
			wantArtifacts := 1
			if c.wantVerb == VerbNone {
				wantArtifacts = 0
			}
			if artifacts != wantArtifacts {
				t.Errorf("effect carries %d artifacts, want %d", artifacts, wantArtifacts)
			}
			c.check(t, eff)
		})
	}
}

// rogueAction is the out-of-vocabulary probe: it satisfies the union's method
// set (which only a same-package type can, the interface being sealed), but is
// not one of the six case types the lowering recognizes.
type rogueAction struct{}

func (rogueAction) Kind() ActionKind    { return ActionKind("exfiltrate") }
func (rogueAction) isSupervisorAction() {}

// TestOutOfVocabularyActionRejected is the negative witness: an action outside
// the closed vocabulary is REJECTED, and no admission verb runs for it — not
// even partially. A nil action is likewise rejected.
func TestOutOfVocabularyActionRejected(t *testing.T) {
	v := &fakeVerbs{}
	if _, err := Lower(rogueAction{}, v); !errors.Is(err, ErrOutOfVocabulary) {
		t.Errorf("Lower(rogue) error = %v, want ErrOutOfVocabulary", err)
	}
	if _, err := Lower(nil, v); !errors.Is(err, ErrOutOfVocabulary) {
		t.Errorf("Lower(nil) error = %v, want ErrOutOfVocabulary", err)
	}
	if len(v.calls) != 0 {
		t.Errorf("a rejected action still reached admission verbs: %v", v.calls)
	}
}

// TestVerbRefusalPropagates proves an admission refusal is FINAL at this layer:
// the error propagates, the effect carries no artifact, and nothing retries or
// widens around it — widen is a re-arbitration, never a bypass.
func TestVerbRefusalPropagates(t *testing.T) {
	refuse := errors.New("LANE_HELD")
	cases := []struct {
		name   string
		verbs  *fakeVerbs
		action SupervisorAction
	}{
		{"spawn refused by arbiter", &fakeVerbs{arbitrateErr: refuse}, SpawnAction{Issue: "4479", Lane: "supervisoragent"}},
		{"widen refused by arbiter", &fakeVerbs{arbitrateErr: refuse}, WidenAction{Lane: "supervisoragent"}},
		{"replace refused by admit path", &fakeVerbs{admitErr: refuse}, ReplaceAction{RunID: "RID-dead", Issue: "4479", Lane: "supervisoragent"}},
		{"replan refused by admit path", &fakeVerbs{admitErr: refuse}, RedispatchAction{Issue: "4479", Lane: "supervisoragent"}},
		{"escalate refused by emit", &fakeVerbs{emitErr: refuse}, EscalateAction{RunID: "RID-1", Issue: "4479"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			eff, err := Lower(c.action, c.verbs)
			if !errors.Is(err, refuse) {
				t.Fatalf("Lower error = %v, want the admission refusal to propagate", err)
			}
			if eff.Lease != nil || eff.Admit != nil || eff.Packet != nil {
				t.Errorf("a refused action still carries an artifact: %+v", eff)
			}
		})
	}
}
