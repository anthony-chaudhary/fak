package validate

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestValidateWSLCapabilitySurfaceCoversEveryExternalCommand(t *testing.T) {
	got := strings.Join(validateWSLRequiredCommandNames(), ",")
	want := "bash,cp,git,go,gofmt,ls,mkdir,mv,pwd,rm,tail,tar,wsl.exe,xargs"
	if got != want {
		t.Fatalf("WSL capability surface = %q, want %q", got, want)
	}
	phaseCommands := make(map[string]bool)
	for _, surface := range validateWSLCommandSurface {
		if surface.phase == "" || len(surface.commands) == 0 {
			t.Fatalf("invalid WSL command surface: %+v", surface)
		}
		for _, command := range surface.commands {
			phaseCommands[surface.phase+"/"+command] = true
		}
	}
	for _, want := range []string{
		"test_materialize/tar", "extract_tip/git", "extract_tip/xargs",
		"cleanup/rm", "overlay/cp", "overlay/pwd", "go_checks/gofmt",
	} {
		if !phaseCommands[want] {
			t.Fatalf("WSL command surface omits %s", want)
		}
	}
}

func TestValidateWSLCapabilityPreflightReportsMissingAndCachesByIdentity(t *testing.T) {
	checkRuns := stubValidateWSLCapabilityCommands(t, "Ubuntu-24.04\n/usr/local/bin:/usr/bin", "rm\ngit\n")
	first := preflightValidateWSLCapabilitiesWithin(context.Background())
	if first.Status != "missing" || !strings.HasPrefix(first.Identity, "Ubuntu-24.04@path:") || strings.Contains(first.Identity, "/usr") || strings.Join(first.Missing, ",") != "git,rm" || first.Cached {
		t.Fatalf("first verdict = %+v", first)
	}
	if !strings.Contains(first.Detail, "no fallback selected") {
		t.Fatalf("detail = %q; want explicit no-fallback verdict", first.Detail)
	}
	encoded, err := json.Marshal(first)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), "/usr/local/bin") {
		t.Fatalf("verdict leaks raw PATH: %s", encoded)
	}
	second := preflightValidateWSLCapabilitiesWithin(context.Background())
	if !second.Cached || strings.Join(second.Missing, ",") != "git,rm" {
		t.Fatalf("cached verdict = %+v", second)
	}
	if *checkRuns != 1 {
		t.Fatalf("capability check runs = %d, want 1 for stable identity", *checkRuns)
	}
}

func TestValidateWSLCapabilityScriptReportsGofmtMissingFromPATH(t *testing.T) {
	script := validateWSLCapabilityScript([]string{"wsl.exe", "go", "gofmt"})
	if !strings.Contains(script, "for cmd in 'go' 'gofmt'") || !strings.Contains(script, "command -v \"$cmd\"") {
		t.Fatalf("capability script does not check bare go and gofmt through PATH: %s", script)
	}
	if strings.Contains(script, "GOROOT") || strings.Contains(script, "go env") {
		t.Fatalf("capability script accepts a GOROOT-only gofmt that the real command cannot invoke: %s", script)
	}

	stubValidateWSLCapabilityCommands(t, "Ubuntu-24.04\n/usr/bin", "gofmt\n")
	verdict := preflightValidateWSLCapabilitiesWithin(context.Background())
	if verdict.Status != "missing" || strings.Join(verdict.Missing, ",") != "gofmt" {
		t.Fatalf("verdict = %+v; want go present and bare gofmt missing", verdict)
	}
}

func TestValidateWSLCapabilityPreflightTypesMissingHostLauncher(t *testing.T) {
	resetValidateWSLCapabilityCacheForTest()
	oldLookPath := validateWSLLookPath
	oldCommand := validateWSLCommand
	validateWSLLookPath = func(string) (string, error) { return "", os.ErrNotExist }
	validateWSLCommand = func(context.Context, ...string) ([]byte, error) {
		t.Fatal("WSL command must not run without the host launcher")
		return nil, nil
	}
	t.Cleanup(func() {
		validateWSLLookPath = oldLookPath
		validateWSLCommand = oldCommand
		resetValidateWSLCapabilityCacheForTest()
	})
	verdict := preflightValidateWSLCapabilitiesWithin(context.Background())
	if verdict.Status != "missing" || strings.Join(verdict.Missing, ",") != "wsl.exe" {
		t.Fatalf("verdict = %+v", verdict)
	}
}

func TestValidateWSLCapabilityPreflightBoundsProbe(t *testing.T) {
	resetValidateWSLCapabilityCacheForTest()
	oldLookPath := validateWSLLookPath
	oldCommand := validateWSLCommand
	validateWSLLookPath = func(string) (string, error) { return `C:\Windows\System32\wsl.exe`, nil }
	var remaining time.Duration
	validateWSLCommand = func(ctx context.Context, _ ...string) ([]byte, error) {
		deadline, ok := ctx.Deadline()
		if !ok {
			t.Fatal("WSL capability probe has no deadline")
		}
		remaining = time.Until(deadline)
		return []byte(validateWSLDistroPrefix + "Ubuntu-24.04\n" + validateWSLPathPrefix + "/usr/bin\n"), nil
	}
	t.Cleanup(func() {
		validateWSLLookPath = oldLookPath
		validateWSLCommand = oldCommand
		resetValidateWSLCapabilityCacheForTest()
	})

	verdict := preflightValidateWSLCapabilitiesWithin(context.Background())
	if verdict.Status != "ready" {
		t.Fatalf("verdict = %+v", verdict)
	}
	if remaining <= 0 || remaining > validateWSLPreflightTimeout {
		t.Fatalf("probe deadline remaining = %s, want (0, %s]", remaining, validateWSLPreflightTimeout)
	}
}

func stubValidateWSLCapabilityCommands(t *testing.T, identity, missing string) *int {
	t.Helper()
	resetValidateWSLCapabilityCacheForTest()
	oldLookPath := validateWSLLookPath
	oldCommand := validateWSLCommand
	checkRuns := 0
	validateWSLLookPath = func(string) (string, error) { return `C:\Windows\System32\wsl.exe`, nil }
	validateWSLCommand = func(_ context.Context, args ...string) ([]byte, error) {
		if len(args) != 4 || args[0] != "-e" || args[1] != "bash" || args[2] != "-lc" {
			return nil, fmt.Errorf("unexpected WSL capability command: %v", args)
		}
		if !strings.Contains(args[3], validateWSLDistroPrefix) || !strings.Contains(args[3], validateWSLPathPrefix) {
			return nil, fmt.Errorf("capability script omits identity protocol: %s", args[3])
		}
		checkRuns++
		parts := strings.SplitN(identity, "\n", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid stub identity %q", identity)
		}
		return []byte(validateWSLDistroPrefix + parts[0] + "\n" + validateWSLPathPrefix + parts[1] + "\n" + missing), nil
	}
	t.Cleanup(func() {
		validateWSLLookPath = oldLookPath
		validateWSLCommand = oldCommand
		resetValidateWSLCapabilityCacheForTest()
	})
	return &checkRuns
}

func resetValidateWSLCapabilityCacheForTest() {
	validateWSLCapabilities.Lock()
	validateWSLCapabilities.byIdentity = make(map[string]validateWSLCapabilityVerdict)
	validateWSLCapabilities.identityByLauncher = make(map[string]string)
	validateWSLCapabilities.Unlock()
}

func TestReadWSLWorktreeGitDirTranslatesWindowsPaths(t *testing.T) {
	cases := []struct {
		name    string
		gitdir  string
		wantWSL string
		wantOK  bool
	}{
		{
			name:    "windows absolute forward slash",
			gitdir:  "gitdir: C:/work/fak/.git/worktrees/worker-1\n",
			wantWSL: "/mnt/c/work/fak/.git/worktrees/worker-1",
			wantOK:  true,
		},
		{
			name:    "windows absolute backslash",
			gitdir:  "gitdir: C:\\work\\fak\\.git\\worktrees\\worker-2\n",
			wantWSL: "/mnt/c/work/fak/.git/worktrees/worker-2",
			wantOK:  true,
		},
		{
			name:    "d drive path",
			gitdir:  "gitdir: D:/data/repo/.git/worktrees/sub1",
			wantWSL: "/mnt/d/data/repo/.git/worktrees/sub1",
			wantOK:  true,
		},
		{
			name:    "not a gitdir file",
			gitdir:  "ref: refs/heads/main\n",
			wantWSL: "",
			wantOK:  false,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, ".git"), []byte(tc.gitdir), 0644); err != nil {
				t.Fatal(err)
			}
			gotWSL, ok := readWSLWorktreeGitDir(dir)
			if ok != tc.wantOK {
				t.Fatalf("ok=%v want %v", ok, tc.wantOK)
			}
			if gotWSL != tc.wantWSL {
				t.Fatalf("got=%q want %q", gotWSL, tc.wantWSL)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestValidateWSLPreflightBoundsSaturatesRetry(t *testing.T) {
	const maxDuration = time.Duration(math.MaxInt64)
	const maxFirst = maxDuration / validateWSLPreflightRetryFactor
	cases := []struct {
		name  string
		raw   string
		first time.Duration
		retry time.Duration
	}{
		{"default", "", validateWSLPreflightTimeout, 3 * validateWSLPreflightTimeout},
		{"invalid", "invalid", validateWSLPreflightTimeout, 3 * validateWSLPreflightTimeout},
		{"zero", "0s", validateWSLPreflightTimeout, 3 * validateWSLPreflightTimeout},
		{"negative", "-1s", validateWSLPreflightTimeout, 3 * validateWSLPreflightTimeout},
		{"parse overflow", "9223372036854775808ns", validateWSLPreflightTimeout, 3 * validateWSLPreflightTimeout},
		{"small positive", " 1ns ", time.Nanosecond, 3 * time.Nanosecond},
		{"below multiplication limit", (maxFirst - 1).String(), maxFirst - 1, 3 * (maxFirst - 1)},
		{"multiplication limit", maxFirst.String(), maxFirst, 3 * maxFirst},
		{"above multiplication limit", (maxFirst + 1).String(), maxFirst + 1, maxDuration},
		{"largest duration", maxDuration.String(), maxDuration, maxDuration},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(validateWSLPreflightTimeoutEnv, tc.raw)
			got := validateWSLPreflightBounds()
			if len(got) != 2 || got[0] != tc.first || got[1] != tc.retry {
				t.Fatalf("probe bounds = %v, want [%s %s]", got, tc.first, tc.retry)
			}
		})
	}
}

func stubValidateWSLProbeTimeouts(t *testing.T, timeouts int) *[]time.Duration {
	t.Helper()
	resetValidateWSLCapabilityCacheForTest()
	oldLookPath := validateWSLLookPath
	oldCommand := validateWSLCommand
	validateWSLLookPath = func(string) (string, error) { return `C:\Windows\System32\wsl.exe`, nil }
	var bounds []time.Duration
	validateWSLCommand = func(ctx context.Context, _ ...string) ([]byte, error) {
		deadline, _ := ctx.Deadline()
		bounds = append(bounds, time.Until(deadline))
		if len(bounds) <= timeouts {
			<-ctx.Done()
			return nil, ctx.Err()
		}
		return []byte(validateWSLDistroPrefix + "Ubuntu-24.04\n" + validateWSLPathPrefix + "/usr/bin\n"), nil
	}
	t.Cleanup(func() {
		validateWSLLookPath = oldLookPath
		validateWSLCommand = oldCommand
		resetValidateWSLCapabilityCacheForTest()
	})
	return &bounds
}

func TestValidateWSLCapabilityPreflightRetriesLoadedProbeWithLongerBound(t *testing.T) {
	t.Setenv(validateWSLPreflightTimeoutEnv, "20ms")
	bounds := stubValidateWSLProbeTimeouts(t, 1)

	verdict := preflightValidateWSLCapabilitiesWithin(context.Background())
	if verdict.Status != "ready" {
		t.Fatalf("verdict status = %q, want ready after one slow probe", verdict.Status)
	}
	if len(*bounds) != 2 || (*bounds)[1] <= 20*time.Millisecond {
		t.Fatalf("probe bounds = %v, want a second attempt longer than the first", *bounds)
	}
}

func TestValidateWSLCapabilityPreflightStaysTypedWhenEveryAttemptTimesOut(t *testing.T) {
	t.Setenv(validateWSLPreflightTimeoutEnv, "10ms")
	bounds := stubValidateWSLProbeTimeouts(t, 2)

	verdict := preflightValidateWSLCapabilitiesWithin(context.Background())
	if verdict.Status != "unavailable" || len(verdict.Missing) != 0 {
		t.Fatalf("verdict = status %q missing %v, want unavailable with nothing missing", verdict.Status, verdict.Missing)
	}
	if len(*bounds) != 2 {
		t.Fatalf("probe attempts = %d, want 2", len(*bounds))
	}
}

func TestValidateWSLCapabilityPreflightRecordsProbePhaseTiming(t *testing.T) {
	stubValidateWSLCapabilityCommands(t, "Ubuntu-24.04\n/usr/bin", validateWSLCheckUSPrefix+"137000\n")
	clock := time.Unix(0, 0)
	oldNow := validateNow
	validateNow = func() time.Time { clock = clock.Add(400 * time.Millisecond); return clock }
	t.Cleanup(func() { validateNow = oldNow })

	verdict := preflightValidateWSLCapabilitiesWithin(context.Background())
	if verdict.Status != "ready" || len(verdict.Missing) != 0 {
		t.Fatalf("verdict = status %q missing %v, want ready with the timing line not read as a missing command", verdict.Status, verdict.Missing)
	}
	want := validateWSLProbeAttempt{BoundMS: validateWSLPreflightTimeout.Milliseconds(), ElapsedMS: 400, LaunchMS: 263, CheckMS: 137, Outcome: validateWSLProbeOK}
	if len(verdict.Attempts) != 1 || verdict.Attempts[0] != want {
		t.Fatalf("attempts = %+v, want [%+v]", verdict.Attempts, want)
	}
}

func TestValidateWSLCapabilityPreflightTypesEachAttemptOutcome(t *testing.T) {
	t.Setenv(validateWSLPreflightTimeoutEnv, "10ms")
	stubValidateWSLProbeTimeouts(t, 2)

	verdict := preflightValidateWSLCapabilitiesWithin(context.Background())
	if len(verdict.Attempts) != 2 {
		t.Fatalf("attempts = %+v, want 2", verdict.Attempts)
	}
	for i, wantBound := range []int64{10, 30} {
		if got := verdict.Attempts[i]; got.Outcome != validateWSLProbeTimeout || got.BoundMS != wantBound {
			t.Fatalf("attempt %d = %+v, want outcome %q bound %dms", i, got, validateWSLProbeTimeout, wantBound)
		}
	}
}

func TestValidateWSLCapabilityScriptReportsCheckTimeUnderRealWSL(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("WSL is Windows-only")
	}
	if _, err := exec.LookPath("wsl.exe"); err != nil {
		t.Skip("wsl.exe unavailable")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	out, err := runValidateWSLCapabilityCommand(ctx, "-e", "bash", "-lc", validateWSLCapabilityScript([]string{"bash"}))
	if err != nil {
		t.Skipf("WSL not runnable here: %v", err)
	}
	_, rest, err := parseValidateWSLCapabilityOutput(out)
	if err != nil {
		t.Fatal(err)
	}
	var missing, timing []string
	for _, line := range strings.Split(strings.TrimSpace(string(rest)), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, validateWSLCheckUSPrefix) {
			timing = append(timing, line)
		} else if line != "" {
			missing = append(missing, line)
		}
	}
	if len(missing) != 0 || len(timing) != 1 {
		t.Fatalf("probe output missing=%v timing=%v, want no missing command and one timing line", missing, timing)
	}
}
