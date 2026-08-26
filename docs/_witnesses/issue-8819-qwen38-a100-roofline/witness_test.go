package issue8819witness

import (
	"bufio"
	"encoding/json"
	"os"
	"testing"
)

type profile struct {
	Engine, Backend string
	Phases          []struct {
		Phase    string
		E2EMS    float64 `json:"e2e_ms"`
		Families []struct {
			Family string
			GPUMS  float64 `json:"gpu_ms"`
			Bytes  int64   `json:"bytes"`
		} `json:"families"`
	} `json:"phases"`
}
type receipt struct {
	Schema         string `json:"schema"`
	ArtifactSHA256 string `json:"artifact_sha256"`
	Execution      struct {
		Engine        string `json:"engine"`
		ForwardPath   string `json:"forward_path"`
		FallbackCount int    `json:"fallback_count"`
	} `json:"execution"`
	Controls struct {
		CacheState  string `json:"cache_state"`
		Repetitions int    `json:"repetitions"`
	} `json:"controls"`
}

type event struct {
	Operation   string `json:"operation"`
	DurationNS  int64  `json:"duration_ns"`
	DeviceBytes int64  `json:"device_bytes"`
	Tokens      int    `json:"tokens"`
	Backend     string `json:"backend"`
}
type row struct {
	Arm         string `json:"arm"`
	QualityPass bool   `json:"quality_pass"`
}

func readJSON(t *testing.T, path string, out any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, out); err != nil {
		t.Fatal(err)
	}
}

func TestAcceptedProfileSelectsOneBoundedLever(t *testing.T) {
	var p profile
	readJSON(t, "profile.json", &p)
	var r receipt
	readJSON(t, "receipt.json", &r)
	if r.Execution.Engine != "fak-native" {
		t.Fatalf("not fak-native CUDA: %+v %+v", p, r.Execution)
	}
	if r.ArtifactSHA256 != "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169" {
		t.Fatal("wrong artifact")
	}
	if r.Controls.Repetitions != 5 || r.Controls.CacheState == "" {
		t.Fatal("quality/operating envelope missing")
	}
	if len(p.Phases) != 6 {
		t.Fatalf("ranking lost: %+v", p.Phases)
	}
}

func TestRawProfileExoneratesTransfer(t *testing.T) {
	f, err := os.Open("prefix-profile.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	n := 0
	var total int64
	s := bufio.NewScanner(f)
	for s.Scan() {
		var e event
		if err = json.Unmarshal(s.Bytes(), &e); err != nil {
			t.Fatal(err)
		}
		n++
		total += e.DurationNS
		if e.Operation != "device_clone" || e.Backend != "cuda" || e.DeviceBytes != 161218560 || e.Tokens != 22 {
			t.Fatalf("unexpected event: %+v", e)
		}
	}
	if err = s.Err(); err != nil {
		t.Fatal(err)
	}
	if n != 4 {
		t.Fatalf("events=%d want 4", n)
	}
	medianMS := float64(total) / float64(n) / 1e6
	if medianMS < 20 || medianMS > 40 {
		t.Fatalf("clone mean %.2fms outside captured envelope", medianMS)
	}
	var rows []row
	readJSON(t, "rows.json", &rows)
	if len(rows) != 10 {
		t.Fatalf("rows=%d want 10", len(rows))
	}
	for _, x := range rows {
		if x.QualityPass {
			t.Fatal("receipt must remain HOLD while quality fails")
		}
	}
}
