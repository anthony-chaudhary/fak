package deploykit

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/selfupdate"
)

var wantStageTokens = []string{
	"resolve", "acquire", "verify", "preflight", "drain", "stage",
	"swap", "activate", "probe", "commit", "rollback", "receipt",
}

func TestParseStageRoundTrip(t *testing.T) {
	stages := Stages()
	if len(stages) != len(wantStageTokens) {
		t.Fatalf("Stages() has %d stages, want %d", len(stages), len(wantStageTokens))
	}
	for i, tok := range wantStageTokens {
		st, err := ParseStage(tok)
		if err != nil {
			t.Fatalf("ParseStage(%q) error: %v", tok, err)
		}
		if st != stages[i] || st.String() != tok || !st.Valid() {
			t.Fatalf("ParseStage(%q) = %q (canonical %q, valid=%v)", tok, st, stages[i], st.Valid())
		}
		b, err := json.Marshal(st)
		if err != nil {
			t.Fatalf("marshal %q: %v", st, err)
		}
		var back Stage
		if err := json.Unmarshal(b, &back); err != nil || back != st {
			t.Fatalf("JSON round-trip %q -> %s -> %q (err %v)", st, b, back, err)
		}
	}
	stages[0] = "mutated"
	if Stages()[0] != StageResolve {
		t.Fatal("Stages() exposes the package's canonical order to mutation")
	}
}

func TestParseStageRejectsUnknown(t *testing.T) {
	for _, tok := range []string{"deploy", "", "SWAP", "Swap", " swap", "swap ", "rollback\n"} {
		st, err := ParseStage(tok)
		if err == nil {
			t.Fatalf("ParseStage(%q) = %q, want error", tok, st)
		}
		if !errors.Is(err, ErrUnknownStage) {
			t.Fatalf("ParseStage(%q) error %v does not wrap ErrUnknownStage", tok, err)
		}
		if st.Valid() {
			t.Fatalf("ParseStage(%q) returned valid stage %q alongside an error", tok, st)
		}
	}
	var zero Stage
	if zero.Valid() {
		t.Fatal("the zero Stage must not be valid")
	}
	var st Stage
	if err := json.Unmarshal([]byte(`"deploy"`), &st); !errors.Is(err, ErrUnknownStage) {
		t.Fatalf("decoding an unknown stage: err %v, want ErrUnknownStage", err)
	}
	if _, err := json.Marshal(Stage("SWAP")); !errors.Is(err, ErrUnknownStage) {
		t.Fatalf("encoding an unknown stage: err %v, want ErrUnknownStage", err)
	}
	var m StageMS
	if err := m.Set("deploy", 1); !errors.Is(err, ErrUnknownStage) {
		t.Fatalf("StageMS.Set(unknown): err %v, want ErrUnknownStage", err)
	}
	if m != (StageMS{}) {
		t.Fatalf("StageMS.Set(unknown) changed the timings: %+v", m)
	}
}

func TestStageMSCoversEveryStage(t *testing.T) {
	var m StageMS
	for i, st := range Stages() {
		if err := m.Set(st, int64(i+1)); err != nil {
			t.Fatalf("Set(%q): %v", st, err)
		}
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"resolve":1,"acquire":2,"verify":3,"preflight":4,"drain":5,"stage":6,"swap":7,"activate":8,"probe":9,"commit":10,"rollback":11,"receipt":12}`
	if string(b) != want {
		t.Fatalf("StageMS JSON\n got %s\nwant %s", b, want)
	}
}

func fullReceipt() Receipt {
	old, next := "1111111", "2222222"
	r := NewReceipt("router", "corr-0001")
	r.Status = "rolled_back"
	r.Stage = StageRollback
	r.OldRevision = &old
	r.NewRevision = &next
	r.Targets = []ReceiptTarget{{
		Role:                    "primary",
		Path:                    "C:/fak/bin/fak-server.exe",
		CompatibilityGroup:      "server",
		DesiredArtifactDigest:   "sha256:aa",
		InstalledArtifactDigest: "sha256:bb",
		Acquisition:             "build",
		Activation:              "service-restart",
		Rollback:                "restored",
	}}
	r.Attempted = 1
	r.Changed = 1
	r.RollbackStatus = "succeeded"
	r.RollbackErrors = []string{"probe readyz: 503"}
	r.RestartRequired = true
	r.NextCommand = "retry-cmd"
	r.Detail = "probe failed; restored prior slot"
	r.TotalMS = 4321
	for i, st := range Stages() {
		_ = r.StageMS.Set(st, int64(100+i))
	}
	return r
}

const fullReceiptJSON = `{"schema":"fak.deploy.receipt/v1","schema_version":1,"correlation_id":"corr-0001","deployable":"router","status":"rolled_back","stage":"rollback","old_revision":"1111111","new_revision":"2222222","targets":[{"role":"primary","path":"C:/fak/bin/fak-server.exe","compatibility_group":"server","desired_artifact_digest":"sha256:aa","installed_artifact_digest":"sha256:bb","acquisition":"build","activation":"service-restart","rollback":"restored"}],"attempted":1,"changed":1,"rollback_status":"succeeded","rollback_errors":["probe readyz: 503"],"restart_required":true,"next_command":"retry-cmd","detail":"probe failed; restored prior slot","total_ms":4321,"stage_ms":{"resolve":100,"acquire":101,"verify":102,"preflight":103,"drain":104,"stage":105,"swap":106,"activate":107,"probe":108,"commit":109,"rollback":110,"receipt":111}}`

func TestReceiptRoundTripStable(t *testing.T) {
	r := fullReceipt()
	requireNoZeroFields(t, reflect.ValueOf(r), "Receipt")

	first, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	if string(first) != fullReceiptJSON {
		t.Fatalf("receipt wire shape changed (a field was dropped, renamed, or reordered)\n got %s\nwant %s", first, fullReceiptJSON)
	}
	var back Receipt
	if err := json.Unmarshal(first, &back); err != nil {
		t.Fatal(err)
	}
	if ReceiptSchema != "fak.deploy.receipt/v1" || back.Schema != ReceiptSchema || back.SchemaVersion != 1 {
		t.Fatalf("schema = %q v%d", back.Schema, back.SchemaVersion)
	}
	if !reflect.DeepEqual(back, r) {
		t.Fatalf("decoded receipt differs\n got %+v\nwant %+v", back, r)
	}
	second, err := json.Marshal(back)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatalf("round-trip not byte-stable\nfirst  %s\nsecond %s", first, second)
	}
}

// requireNoZeroFields fails when any exported field of the fixture is zero, so a
// field added to Receipt without being covered by the golden bytes fails the
// round-trip witness instead of silently passing as an omitted/zero value.
func requireNoZeroFields(t *testing.T, v reflect.Value, path string) {
	t.Helper()
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			t.Fatalf("%s is nil in the round-trip fixture", path)
		}
		requireNoZeroFields(t, v.Elem(), path)
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			f := v.Type().Field(i)
			if !f.IsExported() {
				continue
			}
			requireNoZeroFields(t, v.Field(i), path+"."+f.Name)
		}
	case reflect.Slice:
		if v.Len() == 0 {
			t.Fatalf("%s is empty in the round-trip fixture", path)
		}
		for i := 0; i < v.Len(); i++ {
			requireNoZeroFields(t, v.Index(i), path)
		}
	default:
		if v.IsZero() {
			t.Fatalf("%s is zero in the round-trip fixture", path)
		}
	}
}

func TestReceiptRoundTripDetectsLoss(t *testing.T) {
	// The negative half: a wire image missing one field, or with two fields
	// swapped, must not pass as the golden receipt.
	dropped := strings.Replace(fullReceiptJSON, `"restart_required":true,`, "", 1)
	var r Receipt
	if err := json.Unmarshal([]byte(dropped), &r); err != nil {
		t.Fatal(err)
	}
	again, _ := json.Marshal(r)
	if string(again) == fullReceiptJSON {
		t.Fatal("a receipt that lost restart_required re-encoded as the golden bytes")
	}
	swapped := strings.Replace(fullReceiptJSON, `"attempted":1,"changed":1`, `"changed":1,"attempted":1`, 1)
	if swapped == fullReceiptJSON {
		t.Fatal("fixture edit did not apply")
	}
	if err := json.Unmarshal([]byte(swapped), &r); err != nil {
		t.Fatal(err)
	}
	again, _ = json.Marshal(r)
	if string(again) == swapped {
		t.Fatal("encoding preserved a reordered wire image; field order must be canonical")
	}
}

func TestNewReceiptDefaults(t *testing.T) {
	r := NewReceipt("fak", "")
	if r.Schema != ReceiptSchema || r.SchemaVersion != ReceiptSchemaVersion || r.Deployable != "fak" {
		t.Fatalf("NewReceipt stamp = %+v", r)
	}
	if r.CorrelationID == "" {
		t.Fatal("empty correlation id was not replaced")
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"targets":[]`, `"rollback_errors":[]`, `"old_revision":null`, `"rollback_status":"not_attempted"`} {
		if !strings.Contains(string(b), want) {
			t.Fatalf("empty receipt %s lacks %s", b, want)
		}
	}
	var top map[string]json.RawMessage
	if err := json.Unmarshal(b, &top); err != nil {
		t.Fatal(err)
	}
	if _, ok := top["stage"]; ok {
		t.Fatalf("empty receipt %s encodes an unset stage", b)
	}
}

var allOutcomes = []Outcome{
	OutcomeInstalled, OutcomeTargetCurrent, OutcomeCheckOnly, OutcomeBusy,
	OutcomeGateFailed, OutcomePrepareFailed, OutcomePinSkew, OutcomeRolledBack,
	OutcomeRollbackFailed, OutcomeHandoffRefused, OutcomeHotCopyDivergent,
	OutcomeRestartRequired,
}

func TestClassifyOutcomeMatchesSelfUpdate(t *testing.T) {
	if len(allOutcomes) != len(knownOutcomes) {
		t.Fatalf("test lists %d outcomes, package knows %d", len(allOutcomes), len(knownOutcomes))
	}
	next := NextCommands{Inspect: "inspect", Retry: "retry", Check: "check"}
	role := map[string]string{
		"fak version":             "inspect",
		"fak self-update":         "retry",
		"fak self-update --check": "check",
	}
	for _, tc := range []struct{ oldRev, newRev, detail string }{
		{"", "", ""},
		{"aaa", "bbb", "boom"},
		{"aaa", "aaa", ""},
	} {
		for _, o := range allOutcomes {
			st, rb, restart, cmd, errs := ClassifyOutcome(o, tc.oldRev, tc.newRev, tc.detail, next)
			wst, wrb, wrestart, wcmd, werrs := selfupdate.ClassifyOutcome(selfupdate.Outcome(o), tc.oldRev, tc.newRev, tc.detail)
			if st != wst || rb != wrb || restart != wrestart || !reflect.DeepEqual(errs, werrs) {
				t.Fatalf("%s %+v: deploykit (%q %q %v %v) != selfupdate (%q %q %v %v)", o, tc, st, rb, restart, errs, wst, wrb, wrestart, werrs)
			}
			wantRole, ok := role[wcmd]
			if !ok {
				t.Fatalf("%s: selfupdate next command %q has no deploykit role; extend ClassifyOutcome", o, wcmd)
			}
			if cmd != wantRole {
				t.Fatalf("%s %+v: next command %q, want %q (selfupdate said %q)", o, tc, cmd, wantRole, wcmd)
			}
		}
	}
}

func TestClassifyOutcomeUnknownFailsClosed(t *testing.T) {
	next := NextCommands{Inspect: "inspect", Retry: "retry", Check: "check"}
	for _, o := range []Outcome{"", "deployed", "self-fresh", "INSTALLED"} {
		st, rb, restart, cmd, errs := ClassifyOutcome(o, "a", "b", "detail", next)
		if st != StatusUnclassified || rb != "not_attempted" || restart || cmd != "check" || errs == nil || len(errs) != 0 {
			t.Fatalf("ClassifyOutcome(%q) = %q %q %v %q %v; want unclassified posture", o, st, rb, restart, cmd, errs)
		}
	}
}

func TestLocalExecutorRunsRealCommands(t *testing.T) {
	x := Local()
	out, ok := x.Run(context.Background(), "", "go", "env", "GOVERSION")
	if !ok || !strings.HasPrefix(strings.TrimSpace(out), "go") {
		t.Fatalf("Local().Run(go env GOVERSION) = %q ok=%v", out, ok)
	}
	if out, ok := x.Run(context.Background(), "", "go", "definitely-not-a-go-subcommand"); ok {
		t.Fatalf("a failing command reported ok; output %q", out)
	}
	if _, ok := x.Run(context.Background(), "", "fak-deploykit-no-such-binary-4308"); ok {
		t.Fatal("a missing binary reported ok")
	}
}

func TestHooksReceiveExecutorAndDescriptor(t *testing.T) {
	d := Deployable{
		Name:     "fixture",
		Source:   Source{Kind: "build"},
		Targets:  []Target{{Path: "bin/fixture", Role: "primary"}},
		Probes:   []Probe{{Name: "readyz", Endpoint: "http://127.0.0.1:0/readyz"}},
		Rollback: RollbackPolicy{KeepSlots: 1},
	}
	refuse := errors.New("preflight refused")
	var seen string
	h := Hooks{Preflight: func(ctx context.Context, x Executor, got Deployable) error {
		out, _ := x.Run(ctx, "", "go", "env", "GOOS")
		seen = got.Name + ":" + strings.TrimSpace(out)
		return refuse
	}}
	if err := h.Preflight(context.Background(), Local(), d); !errors.Is(err, refuse) {
		t.Fatalf("Preflight returned %v", err)
	}
	if !strings.HasPrefix(seen, "fixture:") || seen == "fixture:" {
		t.Fatalf("hook saw %q", seen)
	}
	if h.Probe != nil {
		t.Fatal("unset hook must stay nil")
	}
}
