package policy

import (
	"fmt"
	"strings"
)

// AutoRepairEnv opts the reversibility rung into in-flight repair of a sanctioned
// compiled sidestep (adjudicator.Policy.AutoRepairSidestep).
//
// This is an ENV knob rather than a manifest field on purpose. The substitution it
// enables is a property of the SESSION an operator is supervising -- "repair my bare
// pushes while I drive this loop" -- not of the policy a repo ships to everyone who
// clones it. Putting it in the manifest would make one contributor's supervision
// preference a checked-in default for every other contributor, which is the wrong
// blast radius for a knob that changes what a held call does.
const AutoRepairEnv = "FAK_GUARD_AUTOREPAIR"

// autoRepairSidestepMode is the one recognized ON value. Spelled as a mode rather
// than a boolean so later sanctioned repair classes can be added as sibling tokens
// without redefining what "1" or "true" meant to an already-deployed operator.
const autoRepairSidestepMode = "sidestep"

// autoRepairSidestepFromEnv parses AutoRepairEnv's value into the knob.
//
// Unset, empty, "off", "0", "false", and "none" all disable -- the default is the
// preview-confirm hold. Only "sidestep" enables.
//
// An UNRECOGNIZED value is a LOUD error, not a silent off. The failure it prevents is
// the operator who types FAK_GUARD_AUTOREPAIR=sidesteps and then believes repair is
// active: they would spend the session reading every hold as a bug in the safe-subset
// gate rather than as a typo in their own env. Refusing to start names the mistake at
// the point it was made. The safe direction of the silent failure is not a defence --
// a config that reads enabled and behaves disabled is a lie about the guard's posture,
// and this package already fails loud on an unknown egress.block_lists name for the
// same reason.
func autoRepairSidestepFromEnv(v string) (bool, error) {
	switch tok := strings.ToLower(strings.TrimSpace(v)); tok {
	case "", "off", "0", "false", "none":
		return false, nil
	case autoRepairSidestepMode:
		return true, nil
	default:
		return false, fmt.Errorf("%s: unknown mode %q; valid: %s (enable) or off (default)",
			AutoRepairEnv, v, autoRepairSidestepMode)
	}
}
