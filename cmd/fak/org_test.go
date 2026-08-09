package main

import (
	"bytes"
	"crypto/ed25519"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// cmd/fak/org_test.go — the acceptance bar for `fak org status` (#5322).
//
// internal/policy already proves the lattice. What is testable ONLY here is the
// operator-facing contract, and the load-bearing case is the boring one: an
// UN-ENROLLED box must SAY it has no central authority. "Enrolled and the org asked
// for nothing" and "nobody governs this box" produce an identical floor, and a report
// that leaves the operator to infer which one they are looking at is the failure this
// verb exists to prevent.

// runOrgCapture drives one invocation and returns both streams separately, so a test can
// assert WHICH stream a message landed on: a refusal belongs on stderr, and the --json
// arm must be alone on stdout or no consumer can parse it.
func runOrgCapture(t *testing.T, argv ...string) (rc int, stdout, stderr string) {
	t.Helper()
	var out, errb bytes.Buffer
	rc = runOrg(&out, &errb, argv)
	return rc, out.String(), errb.String()
}

// isolateGuardOverlays points the operator allow/deny overlays at paths that do not
// exist, so a status test reads an empty operator stage instead of whatever the
// developer's own .fak/guard/ happens to hold.
func isolateGuardOverlays(t *testing.T) {
	t.Helper()
	dir := t.TempDir()
	t.Setenv(guardAllowOverlayEnv, filepath.Join(dir, "allow.json"))
	t.Setenv(guardDenyOverlayEnv, filepath.Join(dir, "deny.json"))
}

// orgTestFloor writes a minimal valid manifest admitting exactly the named tools and
// returns its path — the "compiled-in" stage for a test.
func orgTestFloor(t *testing.T, allow ...string) string {
	t.Helper()
	b, err := json.Marshal(policy.Manifest{Version: policy.Version, Allow: allow})
	if err != nil {
		t.Fatalf("marshal floor manifest: %v", err)
	}
	path := filepath.Join(t.TempDir(), "floor.json")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatalf("write floor manifest: %v", err)
	}
	return path
}

func orgTestKey(t *testing.T, seed byte) (ed25519.PublicKey, ed25519.PrivateKey) {
	t.Helper()
	raw := make([]byte, ed25519.SeedSize)
	for i := range raw {
		raw[i] = seed + byte(i)
	}
	priv := ed25519.NewKeyFromSeed(raw)
	return priv.Public().(ed25519.PublicKey), priv
}

// TestOrgStatusUnenrolledSaysNoCentralAuthority is the DoD's un-enrolled row, and it
// checks the WORDS rather than the exit code on purpose: the verb succeeding while
// printing an ambiguous screen is the outcome that leaves an operator guessing.
func TestOrgStatusUnenrolledSaysNoCentralAuthority(t *testing.T) {
	isolateGuardOverlays(t)
	store := filepath.Join(t.TempDir(), "absent-enrollment.json")

	rc, stdout, stderr := runOrgCapture(t, "status", "--enrollment", store,
		"--cache", filepath.Join(t.TempDir(), "absent-cache.json"))
	if rc != 0 {
		t.Fatalf("org status rc = %d, want 0 (stderr: %s)", rc, stderr)
	}
	for _, want := range []string{"INERT", "no central authority", "fak enroll --org"} {
		if !strings.Contains(stdout, want) {
			t.Errorf("un-enrolled status missing %q:\n%s", want, stdout)
		}
	}
	// Provenance is still rendered: an un-enrolled box has plenty to say about which
	// knobs its own operator set, and hiding the table until enrollment would make the
	// verb useless to exactly the people evaluating whether to adopt central policy.
	if !strings.Contains(stdout, "per-knob provenance") {
		t.Errorf("un-enrolled status dropped the provenance table:\n%s", stdout)
	}
}

// TestOrgStatusJSONReportsEveryKeyOnAnUnenrolledBox pins the #5299 house rule on the
// arm most likely to break it. A consumer must be able to read central_authority=false
// rather than infer "no org" from a key that is missing — an absent key is also what a
// truncated write looks like.
func TestOrgStatusJSONReportsEveryKeyOnAnUnenrolledBox(t *testing.T) {
	isolateGuardOverlays(t)
	rc, stdout, stderr := runOrgCapture(t, "status", "--json",
		"--enrollment", filepath.Join(t.TempDir(), "absent.json"),
		"--cache", filepath.Join(t.TempDir(), "absent-cache.json"))
	if rc != 0 {
		t.Fatalf("org status --json rc = %d, want 0 (stderr: %s)", rc, stderr)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(stdout), &got); err != nil {
		t.Fatalf("org status --json emitted unparseable stdout: %v\n%s", err, stdout)
	}
	for _, key := range []string{
		"posture", "central_authority", "enrolled", "org_url", "issuer", "device_id",
		"version", "age_seconds", "max_staleness_sec", "reason", "error", "floor_source",
		"operator_overlay", "central_widen", "central_tighten", "central_refused",
		"operator_clamped", "grants_journaled", "knobs",
	} {
		if _, ok := got[key]; !ok {
			t.Errorf("--json is missing the always-present key %q:\n%s", key, stdout)
		}
	}
	if got["central_authority"] != false || got["enrolled"] != false {
		t.Errorf("un-enrolled box reported authority: %v / %v", got["central_authority"], got["enrolled"])
	}
	if got["posture"] != policy.OrgPostureInert {
		t.Errorf("posture = %v, want %q", got["posture"], policy.OrgPostureInert)
	}
	knobs, ok := got["knobs"].([]any)
	if !ok || len(knobs) == 0 {
		t.Fatalf("knobs is empty; the provenance table is the point of the verb:\n%s", stdout)
	}
}

// TestOrgStatusRendersCentralProvenanceOnAnEnrolledBox is the DoD's enrolled row, run
// end to end over a real signed envelope served by a real (loopback) endpoint: enroll,
// pull, verify, compose, render. Anything less would prove the renderer works on a
// hand-built fold rather than on the plane the four sibling children actually built.
func TestOrgStatusRendersCentralProvenanceOnAnEnrolledBox(t *testing.T) {
	isolateGuardOverlays(t)
	pub, priv := orgTestKey(t, 11)
	now := time.Now()

	// The org grants one extra tool over the compiled floor — the IT-enable-more path.
	body, err := json.Marshal(policy.Manifest{
		Version: policy.Version,
		Allow:   []string{"search_web", "deploy_stage"},
	})
	if err != nil {
		t.Fatalf("marshal central body: %v", err)
	}
	envelope, err := policy.SignEnvelope(policy.OrgEnvelope{
		Issuer:    "acme-corp",
		Version:   9,
		NotBefore: now.Add(-time.Hour).Unix(),
		Expires:   now.Add(time.Hour).Unix(),
		Body:      body,
	}, priv)
	if err != nil {
		t.Fatalf("sign envelope: %v", err)
	}
	raw, err := envelope.Marshal()
	if err != nil {
		t.Fatalf("marshal envelope: %v", err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(raw)
	}))
	defer srv.Close()

	store := filepath.Join(t.TempDir(), "enrollment.json")
	if _, err := policy.EnrollOrg(store, policy.OrgEnrollRequest{
		OrgURL:   srv.URL,
		Issuer:   "acme-corp",
		RootKey:  pub,
		DeviceID: "node-a",
		Now:      now,
	}); err != nil {
		t.Fatalf("enroll: %v", err)
	}

	rc, stdout, stderr := runOrgCapture(t, "status", "--json", "--pull",
		"--url", srv.URL,
		"--policy", orgTestFloor(t, "search_web"),
		"--enrollment", store,
		"--cache", filepath.Join(t.TempDir(), "lastgood.json"))
	if rc != 0 {
		t.Fatalf("org status rc = %d, want 0 (stderr: %s)", rc, stderr)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(stdout), &got); err != nil {
		t.Fatalf("unparseable stdout: %v\n%s", err, stdout)
	}
	if got["posture"] != policy.OrgPostureFresh {
		t.Fatalf("posture = %v, want %q (reason %v, error %v)",
			got["posture"], policy.OrgPostureFresh, got["reason"], got["error"])
	}
	if got["central_authority"] != true {
		t.Errorf("a verified envelope did not register as central authority:\n%s", stdout)
	}
	if got["issuer"] != "acme-corp" {
		t.Errorf("issuer = %v, want acme-corp", got["issuer"])
	}
	if v, _ := got["version"].(float64); v != 9 {
		t.Errorf("version = %v, want 9 (the envelope version an auditor traces the grant to)", got["version"])
	}
	widen, _ := got["central_widen"].([]any)
	if len(widen) != 1 {
		t.Fatalf("central_widen = %v, want the single deploy_stage grant", got["central_widen"])
	}
	if first, _ := widen[0].(map[string]any); first["knob"] != "Allow" || first["new"] != "deploy_stage" {
		t.Errorf("central_widen[0] = %v, want Allow/deploy_stage", widen[0])
	}
	// The whole point: the Allow knob must be attributed to `central`, not to the floor.
	var allowChannel any
	for _, k := range got["knobs"].([]any) {
		if km, _ := k.(map[string]any); km["knob"] == "Allow" {
			allowChannel = km["channel"]
		}
	}
	if allowChannel != policy.ChannelCentral {
		t.Errorf("Allow provenance = %v, want %q", allowChannel, policy.ChannelCentral)
	}

	// The same box, rendered for a human. The two arms are built from one resolved
	// view, but only asserting the JSON would let the text arm rot into saying
	// something different about the same box — which is the arm an operator reads.
	rc, text, stderr := runOrgCapture(t, "status", "--pull",
		"--url", srv.URL,
		"--policy", orgTestFloor(t, "search_web"),
		"--enrollment", store,
		"--cache", filepath.Join(t.TempDir(), "lastgood.json"))
	if rc != 0 {
		t.Fatalf("text arm rc = %d, want 0 (stderr: %s)", rc, stderr)
	}
	for _, want := range []string{
		"FRESH", "acme-corp", "node-a", "version:    9",
		"central widened", "deploy_stage", policy.ChannelCentral,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("enrolled text arm missing %q:\n%s", want, text)
		}
	}
	if strings.Contains(text, "no central authority") {
		t.Errorf("an enrolled box under a verified manifest claimed no central authority:\n%s", text)
	}
}

// TestOrgStatusPullWithoutEndpointIsUsage keeps the network arm honest: --pull with
// nowhere to pull from is a usage error, not a silent offline read that would report a
// posture the operator did not ask for.
func TestOrgStatusPullWithoutEndpointIsUsage(t *testing.T) {
	isolateGuardOverlays(t)
	t.Setenv(policy.OrgPolicyURLEnv, "")
	rc, stdout, stderr := runOrgCapture(t, "status", "--pull",
		"--enrollment", filepath.Join(t.TempDir(), "absent.json"))
	if rc != 2 {
		t.Fatalf("rc = %d, want 2 (stdout: %s / stderr: %s)", rc, stdout, stderr)
	}
	if !strings.Contains(stderr, policy.OrgPolicyURLEnv) {
		t.Errorf("the refusal does not name the env var that would fix it:\n%s", stderr)
	}
}

// TestOrgStatusDamagedEnrollmentRefuses is the fail-closed rule `fak enroll` already
// follows, restated here because this verb reads the same store: a box whose anchor
// will not parse must NOT render as an un-enrolled box. They lead to opposite actions.
func TestOrgStatusDamagedEnrollmentRefuses(t *testing.T) {
	isolateGuardOverlays(t)
	store := filepath.Join(t.TempDir(), "damaged.json")
	if err := os.WriteFile(store, []byte("{not json"), 0o600); err != nil {
		t.Fatalf("write damaged store: %v", err)
	}
	rc, stdout, stderr := runOrgCapture(t, "status", "--enrollment", store)
	if rc != 1 {
		t.Fatalf("rc = %d, want 1 (a damaged anchor is a refusal)", rc)
	}
	if strings.Contains(stdout, "no central authority") {
		t.Errorf("a damaged anchor rendered as an un-enrolled box:\n%s", stdout)
	}
	if !strings.Contains(stderr, "fak enroll --revoke") {
		t.Errorf("the refusal does not name its cure:\n%s", stderr)
	}
}

func TestOrgUnknownSubcommandIsUsage(t *testing.T) {
	rc, _, stderr := runOrgCapture(t, "explain")
	if rc != 2 {
		t.Fatalf("rc = %d, want 2", rc)
	}
	if !strings.Contains(stderr, "usage: fak org status") {
		t.Errorf("no usage on an unknown subcommand:\n%s", stderr)
	}
	if rc, _, stderr := runOrgCapture(t); rc != 2 || !strings.Contains(stderr, "usage:") {
		t.Errorf("bare `fak org` rc = %d, stderr %q; want 2 + usage", rc, stderr)
	}
}

// ---------------------------------------------------------------------------
// journal rows
// ---------------------------------------------------------------------------

// TestOrgCentralGrantRowsCarryIssuerAndVersion is the third DoD item. Issuer AND version
// both matter: the issuer says which org handed out the capability, the version says
// which of that org's manifests did. An auditor with only the first cannot find the
// document.
func TestOrgCentralGrantRowsCarryIssuerAndVersion(t *testing.T) {
	fold := policy.OrgFold{CentralWiden: []policy.AmendmentChange{
		{Field: "Allow", Label: "added_allow", New: "deploy_stage"},
		{Field: "AllowPrefix", Label: "added_allow_prefix", New: "report_"},
	}}
	rows := orgCentralGrantRows(fold, "acme-corp", 9)
	if len(rows) != 2 {
		t.Fatalf("got %d rows, want one per widened knob", len(rows))
	}
	for _, r := range rows {
		if r.Channel != journal.GrantChannelCentral {
			t.Errorf("row %s channel = %q, want %q", r.Knob, r.Channel, journal.GrantChannelCentral)
		}
		if r.Source != "acme-corp@v9" {
			t.Errorf("row %s source = %q, want issuer+version", r.Knob, r.Source)
		}
		if r.Actor != "acme-corp" {
			t.Errorf("row %s actor = %q, want the issuing org", r.Knob, r.Actor)
		}
	}
	if rows[0].Knob != "Allow" || rows[0].New != "deploy_stage" {
		t.Errorf("row 0 = %+v, want the Allow grant", rows[0])
	}
}

// TestOrgCentralGrantRowsReportTheDeclaredClass guards against flattening. A widening of
// a RATCHET knob means the org REMOVED a tighten, which is a louder event than a
// gated widen — recording both as GATED_WIDEN would hide the row worth finding.
func TestOrgCentralGrantRowsReportTheDeclaredClass(t *testing.T) {
	rows := orgCentralGrantRows(policy.OrgFold{CentralWiden: []policy.AmendmentChange{
		{Field: "SelfModifyGlobs", Label: "removed_self_modify_glob", Old: ".fak/guard/*.json"},
		{Field: "Allow", Label: "added_allow", New: "deploy_stage"},
	}}, "acme-corp", 3)
	if rows[0].Class != string(policy.AmendRatchet) {
		t.Errorf("SelfModifyGlobs class = %q, want %q", rows[0].Class, policy.AmendRatchet)
	}
	if rows[1].Class != string(policy.AmendGatedWiden) {
		t.Errorf("Allow class = %q, want %q", rows[1].Class, policy.AmendGatedWiden)
	}
}

// TestOrgGrantChannelMirrorsPolicyChannel pins the mirror internal/journal explicitly
// asks the CALLER side to pin: the ledger declares its own copy of the channel
// vocabulary so a policy-side rename cannot silently rewrite rows already on disk, which
// only works if somewhere the two are checked against each other. This is that somewhere.
func TestOrgGrantChannelMirrorsPolicyChannel(t *testing.T) {
	if journal.GrantChannelCentral != policy.ChannelCentral {
		t.Fatalf("journal %q != policy %q — a rename split the wire vocabulary",
			journal.GrantChannelCentral, policy.ChannelCentral)
	}
}

// TestOrgEmitCentralGrantsIsNilJournalSafe covers the ordinary CLI run: outside a
// session journal.Active() is nil, and the verb must stay byte-identical rather than
// panic on the one path every user takes.
func TestOrgEmitCentralGrantsIsNilJournalSafe(t *testing.T) {
	n := orgEmitCentralGrants(nil, policy.OrgFold{CentralWiden: []policy.AmendmentChange{
		{Field: "Allow", Label: "added_allow", New: "deploy_stage"},
	}}, "acme-corp", 1)
	if n != 1 {
		t.Fatalf("orgEmitCentralGrants = %d, want 1 row attempted even with no journal", n)
	}
}

// ---------------------------------------------------------------------------
// the clone that keeps the diff honest
// ---------------------------------------------------------------------------

// TestOrgClonePolicyContainersIsolatesOverlayMutation is the subtle one. The guard
// overlay appliers write into the maps they are handed, so if the operator proposal were
// built on the central stage's own maps, the "before" snapshot would already contain the
// overlay and every operator widening would silently read as a no-op — a clamp that
// never fires and a provenance table that credits central for the operator's work.
func TestOrgClonePolicyContainersIsolatesOverlayMutation(t *testing.T) {
	base := adjudicator.Policy{
		Allow:       map[string]bool{"search_web": true},
		AllowPrefix: []string{"read_"},
	}
	clone := orgClonePolicyContainers(base)
	rt := policy.Runtime{Adjudicator: clone}
	guardApplyAllowOverlay(&rt, guardAllowOverlay{Allow: []string{"deploy_prod"}, AllowPrefix: []string{"list_"}})

	if base.Allow["deploy_prod"] {
		t.Error("applying the overlay to the clone mutated the original Allow map")
	}
	if len(base.AllowPrefix) != 1 {
		t.Errorf("applying the overlay to the clone mutated the original AllowPrefix: %v", base.AllowPrefix)
	}
	if !rt.Adjudicator.Allow["deploy_prod"] {
		t.Error("the overlay did not reach the clone at all")
	}
}

// TestOrgSortedKnobsPutsMovedKnobsFirst keeps the table readable. Twenty-one rows in
// registry order buries the two an operator came to find.
func TestOrgSortedKnobsPutsMovedKnobsFirst(t *testing.T) {
	got := orgSortedKnobs([]policy.OrgKnobProvenance{
		{Field: "Posture", Channel: policy.ChannelCompiledIn},
		{Field: "Deny", Channel: policy.ChannelOperatorOverlay},
		{Field: "Allow", Channel: policy.ChannelCentral},
		{Field: "Profile", Channel: policy.ChannelCompiledIn},
	})
	want := []string{"Allow", "Deny", "Posture", "Profile"}
	for i, w := range want {
		if got[i].Field != w {
			t.Fatalf("order = %v..., want %v (central, then operator, then the floor)", got[i].Field, want)
		}
	}
}
