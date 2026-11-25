package main

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// A local model server re-prefills the whole conversation every turn, so a
// legitimate turn can run for minutes; the hosted-provider defaults would kill it.
// The client's 1024-token completion cap cuts a whole-file Write mid-call, so a
// local endpoint gets the local server's own default budget instead.
const (
	agentLocalPlannerTimeoutS = 1800
	agentLocalStreamStallS    = 600
	agentLocalMaxTokens       = agent.Inkling2LoopDefaultMaxTokens
)

// ensureAgentLocalEndpointDefaults raises the planner and stream-stall
// timeouts and the completion budget for a local endpoint, only where the
// operator left them unset.
func ensureAgentLocalEndpointDefaults(baseURL string, stderr io.Writer) {
	if !agent.IsLocalEndpoint(baseURL) {
		return
	}
	var defaulted []string
	for _, knob := range []struct {
		name string
		val  int
	}{
		{"FAK_PLANNER_TIMEOUT_S", agentLocalPlannerTimeoutS},
		{"FAK_STREAM_STALL_TIMEOUT_S", agentLocalStreamStallS},
		{"FAK_PROVIDER_DEFAULT_MAX_TOKENS", agentLocalMaxTokens},
	} {
		if strings.TrimSpace(os.Getenv(knob.name)) != "" {
			continue
		}
		_ = os.Setenv(knob.name, strconv.Itoa(knob.val))
		defaulted = append(defaulted, fmt.Sprintf("%s=%d", knob.name, knob.val))
	}
	if len(defaulted) > 0 && stderr != nil {
		fmt.Fprintf(stderr, "fak agent: local endpoint %s; defaulted %s (set any to override)\n", baseURL, strings.Join(defaulted, " "))
	}
}
