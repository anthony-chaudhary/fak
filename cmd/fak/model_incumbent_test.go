package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func incumbentTestArgv(t *testing.T, dir string) string {
	t.Helper()
	// The fixture mirrors the reviewed incumbent command shape with public-safe
	// placeholders; the real (private) argv stays operator-supplied.
	argv := []string{
		"/opt/llama-server",
		"--model", "/models/Qwen3.6-27B.q4_k_m.gguf",
		"--alias", incumbentDefaultAlias,
		"--host", "127.0.0.1",
		"--port", "8090",
	}
	raw, err := json.Marshal(argv)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "argv.json")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestModelIncumbentRenderBindsExpectedIdentity(t *testing.T) {
	dir := t.TempDir()
	argvFile := incumbentTestArgv(t, dir)
	out := filepath.Join(dir, "com.fak.qwen36-model.plist")
	expected := incumbentCommandDigest([]string{
		"/opt/llama-server",
		"--model", "/models/Qwen3.6-27B.q4_k_m.gguf",
		"--alias", incumbentDefaultAlias,
		"--host", "127.0.0.1",
		"--port", "8090",
	})
	summary, err := modelIncumbentRender(incumbentRenderInput{
		ArgvFile: argvFile,
		Out:      out,
		Label:    incumbentDefaultLabel,
		Expected: expected,
		Cwd:      dir,
	})
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	if !summary.CommandMatchesExpected || summary.CommandSHA256 != expected {
		t.Fatalf("summary identity drift: %+v", summary)
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	body := string(raw)
	for _, want := range []string{
		"\t<key>Label</key>",
		"\t<string>" + incumbentDefaultLabel + "</string>",
		"<key>ProgramArguments</key>",
		"<string>/opt/llama-server</string>",
		"<string>--alias</string>",
		"<string>" + incumbentDefaultAlias + "</string>",
		"<key>RunAtLoad</key>\n\t<true/>",
		"<key>KeepAlive</key>\n\t<true/>",
		"<key>ThrottleInterval</key>\n\t<integer>10</integer>",
		"</plist>\n",
	} {
		if !strings.Contains(body, want) {
			t.Fatalf("rendered plist missing %q:\n%s", want, body)
		}
	}
	if summary.PlistSHA256 != digestBytes(raw) {
		t.Fatalf("plist digest drift: %s vs %s", summary.PlistSHA256, digestBytes(raw))
	}
	if summary.Schema != incumbentRenderSchema || summary.KeepAlive != true {
		t.Fatalf("summary shape drift: %+v", summary)
	}
}

func TestModelIncumbentRenderRefusesIdentityDrift(t *testing.T) {
	dir := t.TempDir()
	argvFile := incumbentTestArgv(t, dir)
	// A synthetic argv can never silently pass against the preserved identity:
	// only the operator's exact incumbent command binds it.
	tampered := filepath.Join(dir, "tampered.json")
	if err := os.WriteFile(tampered, []byte(`["/opt/llama-server","--model","/models/other.gguf"]`), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := modelIncumbentRender(incumbentRenderInput{ArgvFile: tampered, Out: filepath.Join(dir, "x.plist"), Label: incumbentDefaultLabel, Expected: "sha256:" + incumbentPreservedCommandSHA256, Cwd: dir}); err == nil {
		t.Fatal("tampered argv rendered against the preserved identity")
	}
	if _, err := modelIncumbentRender(incumbentRenderInput{ArgvFile: argvFile, Out: filepath.Join(dir, "y.plist"), Label: incumbentDefaultLabel, Expected: "sha256:" + incumbentPreservedCommandSHA256, Cwd: dir}); err == nil {
		t.Fatal("fixture argv must not bind the preserved identity")
	}
	// A conscious migration must pass its own digest.
	summary, err := modelIncumbentRender(incumbentRenderInput{
		ArgvFile: tampered,
		Out:      filepath.Join(dir, "migrated.plist"),
		Label:    incumbentDefaultLabel,
		Expected: incumbentCommandDigest([]string{"/opt/llama-server", "--model", "/models/other.gguf"}),
		Cwd:      dir,
	})
	if err != nil {
		t.Fatalf("conscious migration refused: %v", err)
	}
	if !summary.CommandMatchesExpected {
		t.Fatalf("migrated summary identity drift: %+v", summary)
	}
}

func TestModelIncumbentRenderRefusesUnsafeOutputsAndLabels(t *testing.T) {
	dir := t.TempDir()
	argvFile := incumbentTestArgv(t, dir)
	cases := []struct {
		name string
		in   incumbentRenderInput
		want string
	}{
		{
			name: "output inside the repository",
			in:   incumbentRenderInput{ArgvFile: argvFile, Out: filepath.Join(dir, "sub", "p.plist"), Label: incumbentDefaultLabel, Expected: "sha256:" + incumbentPreservedCommandSHA256, Cwd: dir},
			want: "inside this repository",
		},
		{
			name: "label outside the namespace",
			in:   incumbentRenderInput{ArgvFile: argvFile, Out: filepath.Join(dir, "..", "out", "p.plist"), Label: "com.other.model", Expected: "sha256:" + incumbentPreservedCommandSHA256, Cwd: dir},
			want: "com.fak.*",
		},
		{
			name: "missing argv file",
			in:   incumbentRenderInput{Out: filepath.Join(dir, "..", "out", "p.plist"), Label: incumbentDefaultLabel, Expected: "sha256:" + incumbentPreservedCommandSHA256, Cwd: dir},
			want: "--argv-file is required",
		},
		{
			name: "bad expected digest",
			in:   incumbentRenderInput{ArgvFile: argvFile, Out: filepath.Join(dir, "..", "out", "p.plist"), Label: incumbentDefaultLabel, Expected: "nope", Cwd: dir},
			want: "SHA-256 digest",
		},
		{
			name: "relative output",
			in:   incumbentRenderInput{ArgvFile: argvFile, Out: "p.plist", Label: incumbentDefaultLabel, Expected: "sha256:" + incumbentPreservedCommandSHA256, Cwd: dir},
			want: "absolute path",
		},
	}
	// The "inside the repository" case needs a go.mod marker; create it.
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module test\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := modelIncumbentRender(tc.in)
			if err == nil {
				t.Fatalf("expected refusal containing %q", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("refusal %q does not name %q", err.Error(), tc.want)
			}
		})
	}
}

func TestModelIncumbentRenderOverwriteRequiresForce(t *testing.T) {
	dir := t.TempDir()
	argvFile := incumbentTestArgv(t, dir)
	outside := filepath.Join(t.TempDir(), "p.plist")
	in := incumbentRenderInput{
		ArgvFile: argvFile,
		Out:      outside,
		Label:    incumbentDefaultLabel,
		Expected: incumbentCommandDigest([]string{
			"/opt/llama-server",
			"--model", "/models/Qwen3.6-27B.q4_k_m.gguf",
			"--alias", incumbentDefaultAlias,
			"--host", "127.0.0.1",
			"--port", "8090",
		}),
		Cwd: t.TempDir(),
	}
	if _, err := modelIncumbentRender(in); err != nil {
		t.Fatalf("first render: %v", err)
	}
	if _, err := modelIncumbentRender(in); err == nil {
		t.Fatal("overwrite without --force admitted")
	}
	in.Force = true
	if _, err := modelIncumbentRender(in); err != nil {
		t.Fatalf("overwrite with --force refused: %v", err)
	}
}

func TestClassifyIncumbentPreflightFailsClosed(t *testing.T) {
	preserved := "sha256:" + incumbentPreservedCommandSHA256
	owned := incumbentPreflightReceipt{
		ListenerPort:  incumbentDefaultPort,
		LaunchdTarget: "gui/501/com.fak.qwen36-model",
		ExpectedOwner: incumbentPreflightExpectedOwner{Label: incumbentDefaultLabel, JobPresent: true, JobPID: 709, PlistPath: "/Library/LaunchAgents/com.fak.qwen36-model.plist"},
		Incumbent: incumbentPreflightIncumbent{
			ListenerPresent: true, ListenerPID: 709, CommandSHA256: preserved,
			OwnerLabelSHA256: "sha256:aa", OwnerResolved: true,
			HealthStatus: 200, ModelsStatus: 200, ModelAlias: incumbentDefaultAlias,
		},
	}
	verdict := func(r incumbentPreflightReceipt) (string, string) {
		return classifyIncumbentPreflight(r).Verdict, classifyIncumbentPreflight(r).Reason
	}
	if v, _ := verdict(owned); v != incumbentVerdictOwned {
		t.Fatalf("owned state classified %s", v)
	}

	jobAbsentAlternate := owned
	jobAbsentAlternate.ExpectedOwner.JobPresent = false
	jobAbsentAlternate.ExpectedOwner.JobPID = 0
	jobAbsentAlternate.ExpectedOwner.PlistPath = ""
	v, reason := verdict(jobAbsentAlternate)
	if v != incumbentVerdictJobAbsent || reason != "alternate_launchd_supervisor_owns_incumbent" {
		t.Fatalf("alternate-supervisor state classified %s/%s", v, reason)
	}

	nothing := jobAbsentAlternate
	nothing.Incumbent = incumbentPreflightIncumbent{}
	v, _ = verdict(nothing)
	if v != incumbentVerdictJobAbsent {
		t.Fatalf("empty state classified %s", v)
	}

	commandDrift := owned
	commandDrift.Incumbent.CommandSHA256 = "sha256:bbbb"
	v, _ = verdict(commandDrift)
	if v != incumbentVerdictMismatch {
		t.Fatalf("command drift classified %s", v)
	}

	unbound := owned
	unbound.ExpectedOwner.JobPID = 999
	v, _ = verdict(unbound)
	if v != incumbentVerdictMismatch {
		t.Fatalf("unbound job classified %s", v)
	}

	unhealthy := owned
	unhealthy.Incumbent.HealthStatus = 404
	v, _ = verdict(unhealthy)
	if v != incumbentVerdictUnhealthy {
		t.Fatalf("unhealthy incumbent classified %s", v)
	}

	aliasDrift := owned
	aliasDrift.Incumbent.ModelAlias = "qwen2.5-coder-14b"
	v, _ = verdict(aliasDrift)
	if v != incumbentVerdictUnhealthy {
		t.Fatalf("alias drift classified %s", v)
	}

	observationFailure := owned
	observationFailure.Reason = "observe listener: boom"
	v, reason = verdict(observationFailure)
	if v != incumbentVerdictFailed || reason != "observe listener: boom" {
		t.Fatalf("observation failure classified %s/%s", v, reason)
	}

	noPlist := owned
	noPlist.ExpectedOwner.PlistPath = ""
	v, _ = verdict(noPlist)
	if v != incumbentVerdictMismatch {
		t.Fatalf("job without plist path classified %s", v)
	}
}

func TestIncumbentVerdictExitAdmitsOnlyOwned(t *testing.T) {
	if incumbentVerdictExit(incumbentVerdictOwned) != 0 {
		t.Fatal("owned verdict must exit 0")
	}
	for _, verdict := range []string{incumbentVerdictJobAbsent, incumbentVerdictUnhealthy, incumbentVerdictMismatch, incumbentVerdictFailed} {
		if incumbentVerdictExit(verdict) != 2 {
			t.Fatalf("verdict %s must refuse with exit 2", verdict)
		}
	}
}

func TestIncumbentCommandDigestMatchesPreservedSemantics(t *testing.T) {
	// The preserved drill receipts hash the space-joined argv; pin the semantics
	// so a future refactor cannot silently change identity binding.
	argv := []string{"/bin/llama-server", "--port", "8090"}
	want := digestBytes([]byte("/bin/llama-server --port 8090"))
	if got := incumbentCommandDigest(argv); got != want {
		t.Fatalf("digest %s != %s", got, want)
	}
}
