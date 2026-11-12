package servelaunch

import (
	"encoding/json"
	"errors"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func sampleReceipt() *Receipt {
	r := &Receipt{ServerName: "s1", SSHHost: "gpu-box-a"}
	launch := time.Date(2026, 11, 11, 10, 0, 0, 0, time.UTC)
	r.MarkLaunched(launch, "12345.bench-s1-sglang-1109", "/tmp/bench-logs/bench-s1-sglang-1109.log", "python -m sglang.launch_server --port 30000")
	r.HwScreenSession = "12346.bench-s1-hwmon-1109"
	r.HwScreenLogPath = "/tmp/bench-logs/bench-s1-hwmon-1109.log"
	r.AutoSelectedPort = 30000
	r.AutoSelectedGPUs = []int{0, 1}
	r.MarkReady(launch.Add(92 * time.Second))
	return r
}

// TestReceiptRoundTrip proves the durability contract: a receipt marshals to
// JSON and unmarshals back deep-equal on every populated field, so what a later
// tool reads is what the launch recorded.
// fak-test:runtime fast est=1ms lane=default
func TestReceiptRoundTrip(t *testing.T) {
	orig := sampleReceipt()
	orig.AddError("transient hiccup")
	orig.CrashLog = "line one\nline two"
	data, err := orig.Marshal()
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var back Receipt
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if !reflect.DeepEqual(orig, &back) {
		t.Errorf("round trip mismatch:\n got %+v\nwant %+v", &back, orig)
	}
}

// TestReceiptMarkReadyDerivesDuration proves the launch duration is derived from
// the launch timestamp, and stays zero when there was no launch.
// fak-test:runtime fast est=1ms lane=default
func TestReceiptMarkReadyDerivesDuration(t *testing.T) {
	r := sampleReceipt()
	if r.LaunchDurationSec < 91.9 || r.LaunchDurationSec > 92.1 {
		t.Errorf("duration = %v, want ~92s", r.LaunchDurationSec)
	}
	empty := &Receipt{}
	empty.MarkReady(time.Now())
	if empty.LaunchDurationSec != 0 {
		t.Errorf("duration = %v, want 0 with no launch timestamp", empty.LaunchDurationSec)
	}
}

// TestReceiptRenderBanner proves the operator banner carries exactly what a
// human needs to get back in: the session name, the log path, and the literal
// reattach command; and the hw-monitor lines when present.
// fak-test:runtime fast est=1ms lane=default
func TestReceiptRenderBanner(t *testing.T) {
	got := sampleReceipt().Render()
	for _, want := range []string{
		"s1 on gpu-box-a",
		"12345.bench-s1-sglang-1109",
		"/tmp/bench-logs/bench-s1-sglang-1109.log",
		"attach: screen -r 12345.bench-s1-sglang-1109",
		"hw mon: 12346.bench-s1-hwmon-1109",
		"launched=true ready=true stopped=false",
		"92.0s to ready",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("Render() =\n%s\nmissing %q", got, want)
		}
	}
}

// TestReceiptRenderNoSession proves a launch that never reached the host's
// screen renders an explicit no-session line instead of an empty banner, and
// omits the hw-monitor block cleanly.
// fak-test:runtime fast est=1ms lane=default
func TestReceiptRenderNoSession(t *testing.T) {
	r := &Receipt{ServerName: "s1", SSHHost: "gpu-box-a"}
	got := r.Render()
	if !strings.Contains(got, "no session") {
		t.Errorf("Render() = %q, want an explicit no-session line", got)
	}
	if strings.Contains(got, "hw mon") {
		t.Errorf("Render() = %q, must not render an hw-monitor block when absent", got)
	}
	if strings.Contains(got, "attach:") {
		t.Errorf("Render() = %q, must not print an attach line with no session", got)
	}
}

// TestReceiptSaveAndLoad proves the atomic save survives a re-save and reads
// back identically through LoadReceipt.
// fak-test:runtime fast est=1ms lane=default
func TestReceiptSaveAndLoad(t *testing.T) {
	dir := t.TempDir()
	// A nested path exercises the MkdirAll branch.
	path := filepath.Join(dir, "receipts", "s1.json")
	orig := sampleReceipt()
	data, err := orig.Save(path)
	if err != nil {
		t.Fatalf("Save: %v", err)
	}
	if len(data) == 0 {
		t.Fatalf("Save returned no bytes")
	}
	back, err := LoadReceipt(path)
	if err != nil {
		t.Fatalf("LoadReceipt: %v", err)
	}
	if !reflect.DeepEqual(orig, back) {
		t.Errorf("saved receipt mismatch:\n got %+v\nwant %+v", back, orig)
	}
	// Overwrite: a second Save over the same path must succeed.
	orig.AddError("second attempt")
	if _, err := orig.Save(path); err != nil {
		t.Fatalf("second Save: %v", err)
	}
	back2, err := LoadReceipt(path)
	if err != nil {
		t.Fatalf("LoadReceipt (2): %v", err)
	}
	if !reflect.DeepEqual(orig, back2) {
		t.Errorf("re-saved receipt mismatch:\n got %+v\nwant %+v", back2, orig)
	}
	// No leftover temp files in the dir.
	entries, _ := filepath.Glob(filepath.Join(dir, "receipts", "*.tmp"))
	if len(entries) != 0 {
		t.Errorf("temp files left behind: %v", entries)
	}
}

// TestReceiptSaveRefusesEmptyPath proves the fail-closed save.
// fak-test:runtime fast est=1ms lane=default
func TestReceiptSaveRefusesEmptyPath(t *testing.T) {
	if _, err := (&Receipt{}).Save("  "); !errors.Is(err, ErrReceiptInvalid) {
		t.Errorf("err = %v, want errors.Is(err, ErrReceiptInvalid)", err)
	}
}

// TestReceiptCollector proves the collector fills the receipt from injected
// probes with no transport: the screen tri-state, the resolved session name,
// and the fetched crash log all land, and a nil probe is a silent no-op rather
// than a panic.
// fak-test:runtime fast est=1ms lane=default
func TestReceiptCollector(t *testing.T) {
	available := true
	c := NewCollector("s1", "gpu-box-a", LaunchProbes{
		ScreenAvailable: func() (bool, error) { return available, nil },
		ResolveSession:  func(name string) (string, error) { return "999." + name, nil },
		CrashLogTail:    func(string, int) (string, error) { return "traceback...", nil },
	})
	c.RecordScreenProbe()
	if c.Receipt.ScreenAvailable == nil || *c.Receipt.ScreenAvailable != true {
		t.Errorf("ScreenAvailable = %v, want true", c.Receipt.ScreenAvailable)
	}
	at := time.Date(2026, 11, 11, 10, 0, 0, 0, time.UTC)
	c.RecordLaunch(at, "bench-s1", "/tmp/s.log", "run")
	if c.Receipt.ScreenSession != "999.bench-s1" {
		t.Errorf("ScreenSession = %q, want the resolved full name", c.Receipt.ScreenSession)
	}
	if !c.Receipt.Launched || c.Receipt.ScreenLogPath != "/tmp/s.log" {
		t.Errorf("receipt = %+v, want launched with log path", c.Receipt)
	}
	c.RecordCrash("/tmp/s.log", 0)
	if c.Receipt.CrashLog != "traceback..." {
		t.Errorf("CrashLog = %q, want the fetched tail", c.Receipt.CrashLog)
	}

	// Nil probes: none of these may panic, and nothing false is learned.
	nilC := NewCollector("s2", "gpu-box-b", LaunchProbes{})
	nilC.RecordScreenProbe()
	nilC.RecordLaunch(at, "bench-s2", "/tmp/s2.log", "run")
	nilC.RecordCrash("/tmp/s2.log", 0)
	if nilC.Receipt.ScreenAvailable != nil {
		t.Errorf("ScreenAvailable = %v, want nil (never probed)", nilC.Receipt.ScreenAvailable)
	}
	if nilC.Receipt.ScreenSession != "bench-s2" {
		t.Errorf("ScreenSession = %q, want the short name kept when no resolver", nilC.Receipt.ScreenSession)
	}
	if nilC.Receipt.CrashLog != "" {
		t.Errorf("CrashLog = %q, want empty with no probe", nilC.Receipt.CrashLog)
	}
}

// TestReceiptCollectorProbeErrors proves a failing probe records an error line
// and does NOT corrupt the receipt: screen availability stays unknown and the
// session keeps its short name, so a lookup failure never loses the field the
// operator needs.
// fak-test:runtime fast est=1ms lane=default
func TestReceiptCollectorProbeErrors(t *testing.T) {
	c := NewCollector("s1", "gpu-box-a", LaunchProbes{
		ScreenAvailable: func() (bool, error) { return false, errors.New("ssh down") },
		ResolveSession:  func(string) (string, error) { return "", errors.New("screen -ls failed") },
	})
	c.RecordScreenProbe()
	if c.Receipt.ScreenAvailable != nil {
		t.Errorf("ScreenAvailable = %v, want nil after a probe error", c.Receipt.ScreenAvailable)
	}
	c.RecordLaunch(time.Now(), "bench-s1", "/tmp/s.log", "run")
	if c.Receipt.ScreenSession != "bench-s1" {
		t.Errorf("ScreenSession = %q, want the short name preserved on resolve error", c.Receipt.ScreenSession)
	}
	if len(c.Receipt.Errors) != 2 {
		t.Errorf("Errors = %v, want 2 recorded probe failures", c.Receipt.Errors)
	}
}
