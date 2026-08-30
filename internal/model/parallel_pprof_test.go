package model

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"runtime/pprof"
	"strings"
	"testing"
)

const parWorkerPprofHelperEnv = "FAK_PAR_WORKER_PPROF_HELPER"

func TestParWorkersScrubInheritedPprofLabels(t *testing.T) {
	cmd := exec.Command(os.Args[0], "-test.run=^TestParWorkerPprofLabelHelper$")
	cmd.Env = append(os.Environ(), parWorkerPprofHelperEnv+"=1")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("profile helper: %v\n%s", err, out)
	}
}

func TestParWorkerPprofLabelHelper(t *testing.T) {
	if os.Getenv(parWorkerPprofHelperEnv) != "1" {
		t.Skip("helper subprocess")
	}
	if numWorkers <= 1 {
		t.Skipf("numWorkers=%d has no persistent pool workers", numWorkers)
	}

	for _, value := range []string{"hostile-first-10241", "hostile-second-10241"} {
		pprof.Do(context.Background(), pprof.Labels(
			"tenant", value,
			"request", value,
			"secret", value,
		), func(context.Context) {
			parFor(numWorkers*parChunkGranularity, numWorkers, func(_, _ int) {})
		})
	}

	var profile bytes.Buffer
	if err := pprof.Lookup("goroutine").WriteTo(&profile, 1); err != nil {
		t.Fatal(err)
	}
	got := profile.String()
	if !strings.Contains(got, "parWorkerLoop") {
		t.Fatalf("profile did not capture persistent parallel workers:\n%s", got)
	}
	for _, hostile := range []string{"hostile-first-10241", "hostile-second-10241", "tenant", "request", "secret"} {
		if strings.Contains(got, hostile) {
			t.Fatalf("persistent worker profile retained inherited label %q:\n%s", hostile, got)
		}
	}
	if !strings.Contains(got, `"fak.component":"model.parallel"`) {
		t.Fatalf("persistent worker profile omitted closed component label:\n%s", got)
	}
}
