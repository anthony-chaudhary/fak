package main

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessinit"
)

func TestHarnessCrossDogfoodCLI(t *testing.T) {
	prior := harnessCrossDogfoodRun
	t.Cleanup(func() { harnessCrossDogfoodRun = prior })
	harnessCrossDogfoodRun = func(context.Context, string) (harnessinit.CrossDogfoodMatrix, error) {
		return harnessinit.CrossDogfoodMatrix{
			Schema: harnessinit.CrossDogfoodSchema, Platform: "test/arch", Network: "disabled:GOPROXY=off",
			Hosts: 3, SubsystemsPerHost: 5, DriftRefusals: 3,
		}, nil
	}
	var stdout, stderr bytes.Buffer
	if code := runHarness(&stdout, &stderr, []string{"cross-dogfood", "--selfcheck", "--json"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var got harnessinit.CrossDogfoodMatrix
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != harnessinit.CrossDogfoodSchema || got.Hosts != 3 || got.DriftRefusals != 3 {
		t.Fatalf("matrix=%+v", got)
	}
}

func TestHarnessCrossDogfoodCLIRequiresSelfcheck(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runHarnessCrossDogfood(&stdout, &stderr, []string{"--json"}); code != 2 || !strings.Contains(stderr.String(), "--selfcheck") {
		t.Fatalf("exit=%d stderr=%q", code, stderr.String())
	}
}
