package main

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/researcharm"
)

// configureServeArmLeases runs during gateway construction, before a listener
// can bind. Native and proxy modes use the same configured control authority.
func (rt *serveRuntime) configureServeArmLeases(sf *serveFlags, srv *gateway.Server) error {
	if rt.armCoordinator != nil {
		return fmt.Errorf("research-arm coordinator already configured")
	}
	var coordinator *researcharm.Coordinator
	if sf.armLeaseStore != nil && strings.TrimSpace(*sf.armLeaseStore) != "" {
		var err error
		coordinator, err = researcharm.NewDurableCoordinator(16, *sf.armLeaseStore)
		if err != nil {
			return fmt.Errorf("open research-arm lease store: %w", err)
		}
	} else {
		coordinator = researcharm.NewCoordinator(16)
	}
	rt.armCoordinator = coordinator
	srv.SetResearchArmCoordinator(coordinator)
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
