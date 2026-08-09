package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/fleetcap"
)

// accounts_headroom.go — the bridge that makes the config-plane ROTATION (fak accounts
// next / launch --rotate) headroom-aware. The config registry (internal/accounts) is pure
// over disk-derived identity and carries no usage/quota telemetry, so on its own it can only
// rotate stable-by-name — which happily hands back an account the live fleet layer already
// knows is capped or unservable. This derives the missing signal from the RUNTIME layer
// (internal/fleetaccounts: the same usage-throttle / offerability fold `fak fleet-accounts`
// prints) and folds it into an accounts.RotationHeadroom the pure planner consumes.
//
// The score is a BANDED, HONEST tier: the runtime layer tells us "offerable now", "capped /
// can't serve now", or "unknown", and within the two known tiers a coarse signal already on
// the row breaks what used to be an arbitrary name-order tie. The bands never overlap, so the
// pure planner's plain most-headroom-first sort orders tiers correctly AND orders within a
// tier without any planner change:
//
//	(1,2]  OFFERABLE (available, not throttled, creds present). Base accounts.OfferableBase (+1),
//	       plus a load bonus 1/(1+live_sessions) so the LEAST-loaded offerable bucket sorts first.
//	       Always > 1, so every offerable bucket still outranks unknown and walled.
//	 0     UNKNOWN (accounts.UnknownScore) — no runtime row for the bucket. Neither preferred nor
//	       penalised.
//	[-1,0) WALLED (usage-throttled or blocked). Base accounts.WalledBase (-1), plus a
//	       reset-soonness bonus in [0,1) so among walled buckets the SOONEST-to-reset (recovers
//	       first) sorts highest. Always < 0, so a walled bucket never outranks an unknown or
//	       offerable one.
//
// The band ANCHORS (accounts.WalledBase/UnknownScore/OfferableBase) and the sign→tier read
// (accounts.Classify, and accounts.HeadroomLabel for the display word) live in internal/accounts
// so producer and interpreter share one source of truth — this file only ADDS the within-tier
// bonus. This is the "finer most-room-first" ordering the header used to defer: a TIE-BREAK
// within a tier from signals the roster already carries (live_sessions, the throttle reset
// string), NOT a continuous remaining-quota number — a real quota API remains a follow-on.

// rotationHeadroom builds the config-plane rotation headroom signal from the live runtime
// roster. homeDir is the same home the rotation registry was discovered under (the --home
// flag), so a test running against an isolated TempDir home discovers no real accounts and
// gets an empty signal — which makes RotationPlan fall back to the historical stable-by-name
// order, keeping unit tests hermetic. A nil/empty result means "no signal" by design.
func rotationHeadroom(homeDir string) accounts.RotationHeadroom {
	cwd, _ := os.Getwd()
	toolsDir := filepath.Join(findRepoRoot(cwd), "tools")
	paths := fleetaccounts.ResolvePaths(toolsDir)
	home := homeDir
	if home == "" {
		home = paths.Home
	}
	pol := fleetaccounts.LoadPolicy(paths)
	reg := fleetaccounts.LoadRegistry(paths.RegistryPath)
	rows := fleetaccounts.AnnotatedRoster(home, paths.ConfigHome, pol, reg)
	return rosterHeadroom(rows, time.Now().UTC())
}

// rosterHeadroom is the ONE fold from a live runtime roster to the banded headroom signal:
// the roster tiering plus the durable usage-limit cooldown override. Every producer of the
// signal — the rotation planner (rotationHeadroom) and the `fak accounts headroom` display
// verb alike — must go through here, so what an operator READS can never disagree with what
// rotation ACTS on. It used to be two call sites and the display verb called
// headroomFromRoster bare, printing a bucket `fak accounts cooldown` reported walled by a
// live usage-limit cooldown as OFFERABLE — exactly the misread that tempts an operator into
// `fak accounts cooldown --clear` on a genuinely rate-limited seat (#5853).
//
// The cooldown override is the fresher signal: an account the launcher just watched bounce
// off its own cap is walled with certainty, where the roster's offerability read can lag or
// miss a cap that fired seconds ago. Forcing every cooled bucket into the walled tier makes
// rotation sort it last (and NextRotationDecision skip it as non-servable) until the window
// elapses. Fail-open: an unreadable store leaves the roster signal as-is.
func rosterHeadroom(rows []fleetaccounts.Account, now time.Time) accounts.RotationHeadroom {
	hr := headroomFromRoster(rows, now)
	if cd, err := accounts.LoadCooldownStore(defaultCooldownStorePath()); err == nil {
		hr = applyCooldownToHeadroom(hr, cd, now)
	}
	return hr
}

// applyCooldownToHeadroom forces every account within an active cooldown window to
// the walled tier of the rotation headroom signal. The cooldown store is keyed by
// the same "uuid:<AccountUUID>" bucket key AccountKey() produces, which is exactly
// the key headroomFromRoster scores on, so the override lines up bucket-for-bucket.
// A walled score is -1 (the tier floor), overriding any offerable/unknown roster
// read. Pure and time-injected. A nil roster map is materialized so a cooled
// account with no live row still registers as walled.
func applyCooldownToHeadroom(hr accounts.RotationHeadroom, cd *accounts.CooldownStore, now time.Time) accounts.RotationHeadroom {
	if cd == nil {
		return hr
	}
	active := cd.Active(now)
	if len(active) == 0 {
		return hr
	}
	if hr == nil {
		hr = accounts.RotationHeadroom{}
	}
	for _, e := range active {
		hr[e.Account] = accounts.WalledBase
	}
	return hr
}

// headroomFromRoster folds an annotated runtime roster into a per-bucket headroom score keyed
// by the SAME account-bucket key the config registry dedups on ("uuid:<AccountUUID>"), so the
// scores line up with the pool seats. It scores only Claude worker rows (the config-plane
// rotation is over Claude CLAUDE_CONFIG_DIR seats) that carry a resolved AccountUUID. When
// several dirs map to one bucket, login/auth noise still uses the BEST score (a duplicate dir
// can prove the shared account is usable), but an active usage/weekly throttle dominates the
// whole bucket because it is account-wide. now anchors the walled tier's reset-soonness
// tie-break (injected for deterministic tests). It is pure — no I/O — so the banded tiering is
// unit-tested directly.
func headroomFromRoster(rows []fleetaccounts.Account, now time.Time) accounts.RotationHeadroom {
	type bucketHeadroom struct {
		best      float64
		haveBest  bool
		limit     float64
		haveLimit bool
	}
	buckets := map[string]bucketHeadroom{}
	for _, r := range rows {
		if r.Product != "claude" || r.AccountUUID == nil || *r.AccountUUID == "" {
			continue
		}
		key := accounts.UUIDBucketKey(*r.AccountUUID)
		score := bucketScore(r, now)
		b := buckets[key]
		if !b.haveBest || score > b.best {
			b.best = score
			b.haveBest = true
		}
		if accountUsageLimitActive(r, now) {
			limit := score
			// Clamp a usage-capped row into the WALLED band: if its own score is unknown/offerable
			// (>= the walled/non-walled boundary), force it to the walled floor. The boundary is
			// UnknownScore (0), distinct from the floor WalledBase (-1).
			if limit >= accounts.UnknownScore {
				limit = accounts.WalledBase
			}
			if !b.haveLimit || limit > b.limit {
				b.limit = limit
				b.haveLimit = true
			}
		}
		buckets[key] = b
	}
	if len(buckets) == 0 {
		return nil
	}
	hr := accounts.RotationHeadroom{}
	for key, b := range buckets {
		if b.haveLimit {
			hr[key] = b.limit
			continue
		}
		hr[key] = b.best
	}
	return hr
}

// bucketScore maps one annotated Claude worker row to its banded headroom score (see the
// package-level table): OFFERABLE -> (1,2] least-loaded-first, WALLED -> [-1,0)
// soonest-reset-first, UNKNOWN -> 0. The tier is decided first (walled beats offerable when
// both flags are set — a throttled-but-still-marked-available row is walled, matching the old
// switch's throttle precedence), then a within-tier bonus that never leaves the tier's band.
func bucketScore(r fleetaccounts.Account, now time.Time) float64 {
	avail := r.Available != nil && *r.Available
	blocked := r.Blocked != nil && *r.Blocked
	throttled := r.Throttled != nil && *r.Throttled
	switch {
	case throttled || blocked:
		// Walled: base -1, plus a reset-soonness bonus in [0,1) so the account that recovers
		// soonest sorts highest among walled buckets while all stay strictly below 0.
		bonus := 0.0
		if r.Reset != nil {
			if s, ok := fleetaccounts.ResetSoonness(*r.Reset, now); ok {
				bonus = s
			}
		}
		return accounts.WalledBase + bonus
	case avail:
		// Offerable: base +1, plus a load bonus 1/(1+live) in (0,1] so the least-loaded bucket
		// sorts highest while all offerable buckets stay strictly above 1.
		live := liveLoad(r)
		return accounts.OfferableBase + 1.0/float64(1+live)
	default:
		return accounts.UnknownScore // unknown — no runtime availability signal
	}
}

// accountUsageLimitActive reports a clear account-wide usage cap. Unlike auth/login noise,
// this must dominate the whole account bucket: a duplicate or stale peer row marked available
// cannot safely reopen a bucket while a current weekly/session usage wall is known.
func accountUsageLimitActive(r fleetaccounts.Account, now time.Time) bool {
	kind := strings.ToLower(headroomString(r.BlockKind))
	throttled := r.Throttled != nil && *r.Throttled
	if kind != "usage" && !throttled {
		return false
	}
	reset := firstNonEmpty(headroomString(r.Weekly), headroomString(r.Reset))
	if reset == "" {
		return true
	}
	t, ok := fleetaccounts.ResetInstant(reset, now)
	if !ok {
		return true
	}
	return !t.Before(now)
}

func headroomString(p *string) string {
	if p == nil {
		return ""
	}
	return strings.TrimSpace(*p)
}

// liveLoad reads a row's live session count for the offerable load tie-break, preferring
// LiveSessions (sessions seen live now), falling back to ActiveSessions, then 0. A negative
// count is clamped to 0 so the bonus denominator stays >= 1.
func liveLoad(r fleetaccounts.Account) int {
	n := 0
	switch {
	case r.LiveSessions != nil:
		n = *r.LiveSessions
	case r.ActiveSessions != nil:
		n = *r.ActiveSessions
	}
	if n < 0 {
		n = 0
	}
	return n
}

type accountsSeatDeficit struct {
	Required     int    `json:"required"`
	FreshCeiling int    `json:"fresh_ceiling"`
	Shortfall    int    `json:"shortfall"`
	Verdict      string `json:"verdict"`
}

type accountsHeadroomPayload struct {
	Schema      string                    `json:"schema"`
	Product     string                    `json:"product"`
	Headroom    accounts.RotationHeadroom `json:"headroom"`
	SeatDeficit accountsSeatDeficit       `json:"seat_deficit"`
}

func buildAccountsSeatDeficit(rows []fleetaccounts.Account, product string, required int) accountsSeatDeficit {
	rep := fleetaccounts.BuildCapacityPreflight(rows, product, required)
	shortfall := required - rep.TrueConcurrentCeiling
	if shortfall < 0 {
		shortfall = 0
	}
	return accountsSeatDeficit{Required: required, FreshCeiling: rep.TrueConcurrentCeiling, Shortfall: shortfall, Verdict: rep.Verdict}
}

func accountsRequiredDemandFromEnv() int {
	target, _ := strconv.ParseFloat(strings.TrimSpace(os.Getenv("FAK_FLEET_TARGET_IPH")), 64)
	session, _ := strconv.ParseFloat(strings.TrimSpace(os.Getenv("FAK_FLEET_SESSION_MIN")), 64)
	if session <= 0 || math.IsNaN(session) || math.IsInf(session, 0) {
		session = fleetForecastDefaultSessionMinutes
	}
	return fleetcap.RequiredWorkers(target, session)
}

func runAccountsHeadroom(stdout, stderr io.Writer, args []string) int {
	product := "claude"
	required := -1
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--product":
			if i+1 >= len(args) {
				fmt.Fprintln(stderr, "fak accounts headroom: --product requires a value")
				return 2
			}
			i++
			product = args[i]
		case "--required":
			if i+1 >= len(args) {
				fmt.Fprintln(stderr, "fak accounts headroom: --required requires a value")
				return 2
			}
			i++
			n, err := strconv.Atoi(args[i])
			if err != nil || n < 0 {
				fmt.Fprintln(stderr, "fak accounts headroom: --required must be non-negative")
				return 2
			}
			required = n
		default:
			fmt.Fprintf(stderr, "fak accounts headroom: unknown argument %q\n", args[i])
			return 2
		}
	}
	if required < 0 {
		required = accountsRequiredDemandFromEnv()
	}
	cwd, _ := os.Getwd()
	paths := fleetaccounts.ResolvePaths(filepath.Join(findRepoRoot(cwd), "tools"))
	rows := fleetaccounts.AnnotatedRoster(paths.Home, paths.ConfigHome, fleetaccounts.LoadPolicy(paths), fleetaccounts.LoadRegistry(paths.RegistryPath))
	payload := accountsHeadroomPayload{Schema: "fak.accounts-headroom.v1", Product: product, Headroom: rosterHeadroom(rows, time.Now().UTC()), SeatDeficit: buildAccountsSeatDeficit(rows, product, required)}
	if err := json.NewEncoder(stdout).Encode(payload); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	return 0
}
