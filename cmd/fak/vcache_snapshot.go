package main

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
	"github.com/anthony-chaudhary/fak/internal/vcachesnapshot"
)

func writeConfiguredVCacheSnapshot(turns []vcacheobserve.Turn) (string, bool, error) {
	if len(turns) == 0 {
		return "", false, nil
	}
	return vcachesnapshot.WriteConfigured(turns)
}

func writeExplicitVCacheSnapshot(turns []vcacheobserve.Turn) (string, bool, error) {
	if len(turns) == 0 {
		return "", false, nil
	}
	raw := strings.TrimSpace(os.Getenv(vcachesnapshot.EnvPath))
	if raw == "" || strings.EqualFold(raw, "off") {
		return "", false, nil
	}
	return vcachesnapshot.WriteConfigured(turns)
}
