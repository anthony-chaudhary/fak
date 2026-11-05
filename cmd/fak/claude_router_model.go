package main

import (
	"time"

	"github.com/anthony-chaudhary/fak/pkg/routerlaunch"
)

// claudeRouterPreferredModel prefers the router's default-backend model over
// the first advertised id when nothing pins the model explicitly. It degrades
// to "" (caller keeps its own default) on any lookup failure.
func claudeRouterPreferredModel(origin string) string {
	return routerlaunch.PreferredDefaultBackendModel(origin, 2*time.Second)
}

// Package var so tests stub the preference without dialing.
var claudeRouterPreferredModelFn = claudeRouterPreferredModel
