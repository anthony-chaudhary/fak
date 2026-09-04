package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/extensionfault"
)

// runToolprocExtension invokes or checks an isolated extension subprocess behind
// extensionfault's bounded startup/call deadlines and per-extension circuit breaker.
func runToolprocExtension(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("toolproc extension", flag.ContinueOnError)
	fs.SetOutput(stderr)

	name := fs.String("name", "ext", "extension identifier")
	cmdStr := fs.String("cmd", "", "command to launch the extension subprocess")
	callTimeout := fs.Duration("call-timeout", 2*time.Second, "per-call deadline")
	startupTimeout := fs.Duration("startup-timeout", 2*time.Second, "startup readiness deadline")
	restarts := fs.Int("max-restarts", 1, "maximum restart attempts before quarantine")
	payload := fs.String("call", "", "payload string to send to the extension")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON status")

	if !parseFlags(fs, argv) {
		return 2
	}
	if strings.TrimSpace(*cmdStr) == "" {
		fmt.Fprintln(stderr, "fak toolproc extension: --cmd is required")
		return 2
	}

	command := strings.Fields(*cmdStr)
	if len(command) == 0 {
		fmt.Fprintln(stderr, "fak toolproc extension: invalid --cmd")
		return 2
	}

	spec := extensionfault.Spec{
		Name:           *name,
		Command:        command,
		StartupTimeout: *startupTimeout,
		CallTimeout:    *callTimeout,
		MaxRestarts:    *restarts,
	}

	sup, err := extensionfault.New(spec)
	if err != nil {
		fmt.Fprintf(stderr, "fak toolproc extension: %v\n", err)
		return 1
	}
	defer sup.Close()

	ctx, cancel := context.WithTimeout(context.Background(), *startupTimeout+*callTimeout+time.Second)
	defer cancel()

	res, err := sup.Call(ctx, *name, *payload)
	st, _ := sup.Status(*name)

	if *asJSON {
		type report struct {
			Name        string `json:"name"`
			Result      string `json:"result,omitempty"`
			Error       string `json:"error,omitempty"`
			Running     bool   `json:"running"`
			Quarantined bool   `json:"quarantined"`
			Failures    int    `json:"failures"`
			Restarts    int    `json:"restarts"`
			PID         int    `json:"pid"`
		}
		rep := report{
			Name:        st.Name,
			Result:      res,
			Running:     st.Running,
			Quarantined: st.Quarantined,
			Failures:    st.Failures,
			Restarts:    st.Restarts,
			PID:         st.PID,
		}
		if err != nil {
			rep.Error = err.Error()
		}
		return encodeJSONOrFail(stdout, stderr, rep, "fak toolproc extension")
	}

	if err != nil {
		fmt.Fprintf(stderr, "fak toolproc extension: %v (quarantined=%v failures=%d)\n", err, st.Quarantined, st.Failures)
		return 1
	}

	fmt.Fprintln(stdout, res)
	return 0
}
