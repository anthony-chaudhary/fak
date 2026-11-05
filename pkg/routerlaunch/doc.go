// Package routerlaunch resolves the launch-time model preference for a
// supervised local fak router: which advertised model id the router's
// configured default backend actually serves, so a launcher (e.g. `fak
// claude`) can prefer it over the first advertised id when nothing pins the
// model explicitly.
//
// It is a PUBLIC package (github.com/anthony-chaudhary/fak/pkg/routerlaunch):
// pure mechanism (credential discovery, /v1/models listing, routing-table
// read) with no policy and no host assumptions beyond the user home dir.
// Credential values are never logged.
//
// Hermetic test hooks (package vars, not exported): userHomeFunc stands in
// for os.UserHomeDir on every home-relative read, and advertisedFetch stands
// in for AdvertisedModels inside PreferredDefaultBackendModel, so tests never
// dial a router or read host config.
package routerlaunch
