package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/researcharm"
)

const (
	defaultArmMaxConcurrency = 16
	defaultArmAdmitWait      = 30 * time.Second
)

// armMaxConcurrencyFromEnv reads FAK_ARM_MAX_CONCURRENCY: the per-arm concurrency
// ceiling for research arms (default 16; 0 = unlimited). Unparseable or negative
// values fall back to the default rather than misconfiguring admission.
func armMaxConcurrencyFromEnv() int {
	raw := strings.TrimSpace(os.Getenv("FAK_ARM_MAX_CONCURRENCY"))
	if raw == "" {
		return defaultArmMaxConcurrency
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 {
		return defaultArmMaxConcurrency
	}
	return n
}

// armAdmitWaitFromEnv reads FAK_ARM_ADMIT_WAIT: how long a proxied request may
// queue at a saturated arm before the 429 refusal (default 30s; 0 = legacy
// immediate refusal). Unparseable or negative values fall back to the default.
func armAdmitWaitFromEnv() time.Duration {
	raw := strings.TrimSpace(os.Getenv("FAK_ARM_ADMIT_WAIT"))
	if raw == "" {
		return defaultArmAdmitWait
	}
	d, err := time.ParseDuration(raw)
	if err != nil || d < 0 {
		return defaultArmAdmitWait
	}
	return d
}

// configureServeArmLeases runs during gateway construction, before a listener
// can bind. Native and proxy modes use the same configured control authority.
func (rt *serveRuntime) configureServeArmLeases(sf *serveFlags, srv *gateway.Server) error {
	if rt.armCoordinator != nil {
		return fmt.Errorf("research-arm coordinator already configured")
	}
	maxConcurrency := armMaxConcurrencyFromEnv()
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
	srv.SetResearchArmCoordinator(coordinator)
	srv.SetArmAdmitWait(armAdmitWaitFromEnv())
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
