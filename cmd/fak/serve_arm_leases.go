package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/researcharm"
)

const (
	// defaultArmMaxConcurrency is the shipped per-arm in-flight ceiling for
	// research arms. It is 128, not the legacy 16, because on a proxy/bench node
	// the arm ceiling — not the engine — was the binding constraint: a recorded
	// sweep saw HTTP 429 at C32 while the native scheduler the gateway fronts
	// recorded NO shedding, so the front door refused load the engine would have
	// served. 128 (C128) is the interim rung of the powers-of-two concurrency
	// witness ladder, so the default no longer spuriously caps a legitimate
	// single-arm sweep.
	//
	// This remains a TUNABLE, not a law: `--arm-max-concurrency` (0 = unlimited)
	// overrides it per node, `fak arms limit --arm <id>` overrides it per arm at
	// runtime, and a shared/exclusive lease may carry its own concurrency. It is
	// still a hand constant, not a measured value: the correct value is the
	// S(C) knee of the served stack, and the tracked work to replace the
	// constant with that measurement is the autotune ladder row
	// `autotune.concurrency.arm-admission`.
	defaultArmMaxConcurrency = 128

	// defaultArmAdmitWait is how long a proxied request may queue at a saturated
	// arm before the 429 refusal (0 = legacy immediate refusal).
	defaultArmAdmitWait = 30 * time.Second
)

// resolveArmMaxConcurrency returns the effective per-arm ceiling from the parsed
// serve flags: the explicit `--arm-max-concurrency` value, otherwise the
// built-in default. 0 means unlimited. The value lives on the config surface
// (flag), never the environment — behavioral settings are not env vars
// (internal/envconfiglint, CONFIG_NOT_ENV).
func resolveArmMaxConcurrency(sf *serveFlags) int {
	if sf != nil && sf.armMaxConcurrency != nil {
		return *sf.armMaxConcurrency
	}
	return defaultArmMaxConcurrency
}

// resolveArmAdmitWait returns the effective admission-queue budget from the
// parsed serve flags (0 = legacy immediate refusal).
func resolveArmAdmitWait(sf *serveFlags) time.Duration {
	if sf != nil && sf.armAdmitWait != nil {
		return *sf.armAdmitWait
	}
	return defaultArmAdmitWait
}

// configureServeArmLeases runs during gateway construction, before a listener
// can bind. Native and proxy modes use the same configured control authority.
func (rt *serveRuntime) configureServeArmLeases(sf *serveFlags, srv *gateway.Server) error {
	if rt.armCoordinator != nil {
		return fmt.Errorf("research-arm coordinator already configured")
	}
	maxConcurrency := resolveArmMaxConcurrency(sf)
	var coordinator *researcharm.Coordinator
	if sf.armLeaseStore != nil && strings.TrimSpace(*sf.armLeaseStore) != "" {
		var err error
		coordinator, err = researcharm.NewDurableCoordinator(maxConcurrency, *sf.armLeaseStore)
		if err != nil {
			return fmt.Errorf("open research-arm lease store: %w", err)
		}
	} else {
		coordinator = researcharm.NewCoordinator(maxConcurrency)
	}
	rt.armCoordinator = coordinator
	// Seed the coordinator's own wait/queue policy too: the gateway passes the wait per
	// call, but the coordinator's Retry-After hint and the queue BOUND need the values,
	// so a shed carries a sane hint and an unbounded backlog can never form.
	wait := resolveArmAdmitWait(sf)
	coordinator.SetAdmitWait(wait)
	if depth, note := researcharm.ParseArmQueueDepth(os.Getenv); note != "" {
		fmt.Fprintf(os.Stderr, "fak serve: research-arm admission queue: %s\n", note)
	} else if depth > 0 {
		coordinator.SetQueueDepth(depth)
	}
	srv.SetResearchArmCoordinator(coordinator)
	srv.SetArmAdmitWait(wait)
	return nil
}

// closeServeArmLeases is called after the serving lifecycle has drained. Startup
// failures close it before any listener was created; repeated cleanup is safe.
func (rt *serveRuntime) closeServeArmLeases() error {
	if rt.armCoordinator == nil {
		return nil
	}
	return rt.armCoordinator.Close()
}
