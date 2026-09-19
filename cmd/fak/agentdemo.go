package main

// `fak agentdemo` — the OFFLINE demo moved off `fak agent` (lower priority). It runs
// exactly what a bare `fak agent` used to run: the deterministic offline mock planner
// driving the LIVE turn-count A/B (dual arm), no network, no key, no real model. The
// native agent itself (`fak agent`) is the REAL agent: it runs a real model and fails
// loud when none is configured, so the demo lives here, as an explicit command.
//
//   fak agentdemo            the offline mock A/B demo (same flags as fak agent)
//   fak agentdemo --task X   forward any fak agent flag (e.g. --task, --effort)

func cmdAgentDemo(argv []string) {
	runAgent(append([]string{"--offline"}, argv...))
}
