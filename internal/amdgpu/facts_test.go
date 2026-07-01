package amdgpu

import (
	"strings"
	"testing"
	"time"
)

const goodAMD = `{"name":"AMD Radeon RX 7600","driver_version":"32.0.31019.2002","adapter_ram":4293918720,"vram_used_bytes":6400000000,"compute_util_pct":0.0,"total_util_pct":97.5,"engines":[{"type":"Graphics","util_pct":88.0},{"type":"Compute","util_pct":0.0}]}`

func fakeRunner(ok bool, out, errText string) Runner {
	return func(script string, timeout time.Duration) (bool, string, string) {
		return ok, out, errText
	}
}

func TestFactsSuccess(t *testing.T) {
	f := Facts("", fakeRunner(true, goodAMD, ""))
	if f["available"] != true || f["name"] != "AMD Radeon RX 7600" {
		t.Fatalf("facts=%+v", f)
	}
	want := round1(6400000000.0 / (1024 * 1024))
	if f["vram_used_mib"] != want {
		t.Fatalf("vram=%v, want %v", f["vram_used_mib"], want)
	}
	if !strings.Contains(f["note"].(string), "WMI-WORD-capped") {
		t.Fatalf("note=%q", f["note"])
	}
}

func TestBusiestEngine(t *testing.T) {
	f := Facts("", fakeRunner(true, goodAMD, ""))
	if f["busiest_engine"] != "Graphics" || f["busiest_util_pct"] != 88.0 {
		t.Fatalf("busiest fields=%+v", f)
	}
	if !strings.Contains(f["note"].(string), "compute_util_pct") || !strings.Contains(f["note"].(string), "3d") {
		t.Fatalf("note=%q", f["note"])
	}
}

func TestNameFilter(t *testing.T) {
	if Facts("7600", fakeRunner(true, goodAMD, ""))["available"] != true {
		t.Fatal("name filter should match")
	}
	f := Facts("a100", fakeRunner(true, goodAMD, ""))
	if f["available"] != false || f["saw"] != "AMD Radeon RX 7600" {
		t.Fatalf("filter reject=%+v", f)
	}
}

func TestUnavailableAndBadJSON(t *testing.T) {
	f := Facts("", fakeRunner(false, "", "no PowerShell"))
	if f["available"] != false || !strings.Contains(f["error"].(string), "PowerShell") {
		t.Fatalf("unavailable=%+v", f)
	}
	f = Facts("", fakeRunner(true, "not json", ""))
	if f["available"] != false || !strings.Contains(f["error"].(string), "parse failed") {
		t.Fatalf("bad json=%+v", f)
	}
}
