// Package slackquiet is the ONE global brake for fak's outbound Slack writes: when
// FAK_SLACK_QUIET is engaged, every public Slack surface goes silent without any
// surface being edited, and every queued outbox row is HELD rather than destroyed.
//
// Why a package and not a check at each call site: fak's public Slack writes already
// converge on one transport (internal/slackwire), so the brake belongs beside it — the
// same placement argument that produced slackwire itself (#2261). A per-surface switch
// would be N switches to flip and N to forget, which is the status quo the operator
// asked to end.
//
// Polarity: TRUTHY means quiet. This is deliberately NOT the fleet's default-on
// off-vocabulary (internal/gardenbundle's offValues, cmd/fak's fleetEnvSwitchOn), and the
// reason is that this switch's ON state is SILENCE. Those guards are named for a feature
// that is on by default, so an off-ish word disables it and an unset variable keeps it on.
// Applied here that grammar would make an UNSET FAK_SLACK_QUIET mean "muted" — the fleet
// would go silent on a fresh clone, which is the opposite of the fail-loud default every
// other surface has. So the vocabulary is the opt-in one: 1/on/true/yes/enable/enabled
// engages the brake, and unset, blank, or anything unrecognized leaves notifications
// flowing. A typo therefore fails toward the pre-existing behavior rather than silently
// muting the fleet, and the value an operator will actually type for a variable named
// QUIET is the one that works.
//
// Resolution goes through internal/slackenv, the ONE resolver for Slack-surface
// configuration, so the switch is settable from the process environment OR the
// gitignored .env.slack.local the operator already uses for the token and channel —
// one file quiets a whole workspace. That also keeps the read off the envconfiglint
// ratchet: slackenv's os.Getenv takes a variable, not a string literal, so the
// literal-only scanner does not see it (the same reason every other Slack surface
// resolves through slackenv rather than reading os.Getenv directly).
//
// This is a foundation-composite (tier 2), not a primitive: it composes slackenv(1)
// with the shared off-vocabulary. It is deliberately NOT imported by slackwire(1) —
// a primitive may import only abi, and the transport should not know policy anyway.
// slackwire takes the brake as an injected Option and the composition layer (scoreboard,
// chatrelay, the cmd/fak slack verbs, the outbox) passes Engaged in.
//
// Boundary: pure stdlib + internal/slackenv; off the hot path. It holds no secret and
// no channel id — it only knows HOW to read the switch.
package slackquiet

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/slackenv"
)

// EnvVar is the switch every public Slack surface consults. It is exported so a
// diagnostic ("fak slack health") can name the variable it read.
const EnvVar = "FAK_SLACK_QUIET"

// onValues is the opt-in vocabulary: any of these as FAK_SLACK_QUIET engages the brake.
// It is the mirror of internal/gardenbundle's offValues and cmd/fak's fleetEnvSwitchOn —
// same words, opposite polarity, because this switch's ON state is silence (see the
// package doc). "0"/"off"/"false"/"no"/"disable"/"disabled" are deliberately NOT here:
// they are the fleet's off-words, and for a variable named QUIET they read as "not
// quiet", which is exactly what they mean.
var onValues = map[string]bool{
	"1": true, "on": true, "true": true, "yes": true,
	"enable": true, "enabled": true,
}

// Engaged reports whether outbound Slack writes are currently braked. An unset or
// blank switch is NOT quiet (notifications flow); any on-ish word is quiet; any other
// value is not quiet, so a typo fails toward the pre-existing behavior rather than
// silently muting the fleet.
//
// It is a func() bool so it can be handed straight to slackwire.WithQuiet.
func Engaged() bool {
	return EngagedValue(slackenv.Lookup(EnvVar).Value)
}

// EngagedValue applies the opt-in vocabulary to an already-resolved value. It is the
// pure half of Engaged, exported so a caller holding a value (a test, or a surface
// that resolved the switch itself) shares one grammar instead of re-spelling it.
func EngagedValue(raw string) bool {
	return onValues[strings.ToLower(strings.TrimSpace(raw))]
}

// Value is the raw switch value as resolved, for diagnostics. A surface that reports
// "muted" should echo the word the operator actually typed (1, on, yes, …) rather than
// a hardcoded one, so the message cannot disagree with the state it is describing.
func Value() string {
	return slackenv.Lookup(EnvVar).Value
}

// Source names where the switch was read from, for diagnostics: "env", "file", or
// "unset". It mirrors slackenv.Source so a health report can say whether the operator
// set the switch in the environment or in .env.slack.local.
func Source() string {
	return string(slackenv.Lookup(EnvVar).Source)
}
