package main

// serve_route_reload.go — the `fak serve` half of model-routing hot reload: one watcher,
// two triggers.
//
// The background poll loop (#842) follows the manifest file and swaps the live policy on a
// validated edit. Its change gate is size + mtime-nanos (modelroute.fileSig), which is cheap
// but not total: a same-length edit inside the filesystem's timestamp granularity (NFSv3 ~1s,
// FAT 2s, some container/network mounts), or a deploy tool that restores mtime, is
// permanently invisible to it. modelroute.Reload() is the documented content-compare bypass
// for exactly that case — but in fak SIGHUP is a TERMINATING signal routed to the graceful
// drain (serve_stages.go run()), so the SIGHUP-style manual trigger had no production caller
// and an operator whose edit missed the gate had to restart the server, the very thing hot
// reload exists to avoid. armRouteHotReload publishes the watcher to the gateway so
// POST /v1/fak/route/reload drives it (#4003), the route-plane twin of /v1/fak/policy/reload.
//
// Both triggers share ONE watcher deliberately: same last-good gate, same atomic Live swap,
// one reload/reject counter pair. A second watcher would give the manual trigger its own
// last-good baseline and let the two disagree about which bytes are installed.

import (
	"context"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// armRouteHotReload builds the route-manifest watcher over the server's live routing
// policy, installs it as the gateway's forced-reload trigger, and starts its poll loop
// bound to ctx (so it stops with the server). It returns the watcher, or nil when no
// --route-manifest is installed (routing is static and there is nothing to reload).
//
// A malformed edit is rejected and the last-good policy kept — the fail-loud startup
// contract extended to reload — on BOTH triggers, because both go through this watcher.
// Reloads and rejections are logged and journalled as CONFIG_SWAP rows (#3959) so an
// operator can confirm which bytes became authoritative.
func armRouteHotReload(ctx context.Context, srv *gateway.Server, manifestPath string) *modelroute.Watcher {
	live := srv.RouteLive()
	if live == nil {
		return nil
	}
	watcher := modelroute.NewWatcher(manifestPath, live, 0, func(ev modelroute.ReloadEvent) {
		if ev.Err != nil {
			fmt.Fprintf(os.Stderr, "fak: route-manifest reload REJECTED: %v\n", ev.Err)
			// A rejected route swap kept last-good, but the refused edit is exactly what
			// an auditor asks about (#3959): record it with the digest of the malformed
			// bytes on disk. Nil journal → no-op, unjournaled serve stays byte-identical.
			journal.Active().AppendConfigSwap(journal.ConfigSwapRoute, ev.Path, configFileDigest(ev.Path), journal.ConfigSwapRejected, ev.Err.Error())
			return
		}
		if ev.Reloaded {
			fmt.Fprintf(os.Stderr, "fak: model-routing policy hot-reloaded from %s (reload #%d)\n", ev.Path, ev.Reloads)
			// The live model-routing boundary just changed: record which bytes became
			// authoritative (source path + sha256) as a durable CONFIG_SWAP row (#3959).
			journal.Active().AppendConfigSwap(journal.ConfigSwapRoute, ev.Path, configFileDigest(ev.Path), journal.ConfigSwapOK, "")
		}
	})
	// Publish the SAME watcher behind POST /v1/fak/route/reload (#4003). Without this the
	// route answers 404 on every shipped serve and Reload()'s gate bypass is test-only.
	srv.SetRouteWatcher(watcher)
	go func() { _ = watcher.Run(ctx) }()
	return watcher
}
