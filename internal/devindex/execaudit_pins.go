package devindex

// The committed exception set for the executable audit (#5648).
//
// A pin admits ONE executable package that fails an axis, for ONE named reason. It is
// deliberately not an ignore list: foldExecPins reds the audit the moment a pin stops
// doing work — when its package leaves the domain, when the package starts passing on
// its own, when the declared expiry passes, or when the reason is blank. That is the
// only property that keeps an exception honest, because an exception nobody is forced
// to revisit is just undocumented debt with a nicer name.
//
// Empty is the correct default. The audit over the current tree reports its failures
// by name rather than hiding them behind a bulk baseline: which of fak's executables
// are unwired is a real, currently-true finding, and the committed witness
// (docs/exec-audit.witness.json) is where that denominator is preserved. Add an entry
// here only when a specific package has a specific reason to be exempt, with a date.
var ExecAuditPins = []ExecPin{}
