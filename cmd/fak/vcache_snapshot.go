package main

import (
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
	"github.com/anthony-chaudhary/fak/internal/vcachesnapshot"
)

func writeConfiguredVCacheSnapshot(turns []vcacheobserve.Turn) (string, bool, error) {
	if len(turns) == 0 {
		return "", false, nil
	}
	return vcachesnapshot.WriteConfigured(turns)
}
