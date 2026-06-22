package boundarylint

// Status is whether a tell is enforced by DefaultRules or only documented as a roadmap.
type Status string

const (
	StatusEnforced Status = "enforced"
	StatusProposed Status = "proposed"
)

// CatalogEntry documents one boundary tell.
type CatalogEntry struct {
	Code   string
	Title  string
	Status Status
	Note   string // why it's a tell, and (for proposed) the false-positive hazard to solve first
}

// Catalog is the full family of boundary tells — the "policy by default" view. Enforced
// entries are checked in CI (here, or in pathlint/urllint for the first two); proposed
// entries are the prioritized backlog, each annotated with the FP hazard that must be
// handled before it can graduate to enforced without crying wolf.
var Catalog = []CatalogEntry{
	{
		Code:   "UNEXPANDED_USER_PATH",
		Title:  "path flag opened without ~ expansion",
		Status: StatusEnforced,
		Note:   "enforced by internal/pathlint. A -gguf/-hf/-tok/-dir flag given as ~/x is opened as a literal '~' dir.",
	},
	{
		Code:   "UNVERIFIED_EXTERNAL_URL",
		Title:  "hardcoded download URL outside the audited builder",
		Status: StatusEnforced,
		Note:   "enforced by internal/urllint. A pasted model-download URL literal escapes the reachability-tested chokepoint.",
	},
	{
		Code:   "MISSING_HTTP_TIMEOUT",
		Title:  "outbound HTTP with no timeout",
		Status: StatusEnforced,
		Note:   "http.Get/Post/Head/PostForm and http.DefaultClient cannot carry a timeout; a dead peer hangs forever.",
	},
	{
		Code:   "UNCHECKED_HTTP_STATUS",
		Title:  "response body used without checking StatusCode",
		Status: StatusProposed,
		Note:   "FP hazard: track the resp var per-function and allow code that returns/branches on StatusCode in any form before reading Body.",
	},
	{
		Code:   "IGNORED_BOUNDARY_ERROR",
		Title:  "error from a boundary call discarded with _",
		Status: StatusProposed,
		Note:   "scope to a boundary-call allowlist (os.Open/ReadFile, exec *.Run/Output, http *.Do) so ordinary _-drops (fmt.Fprintln) aren't flagged.",
	},
	{
		Code:   "UNCLOSED_RESPONSE_BODY",
		Title:  "http.Response.Body never closed",
		Status: StatusProposed,
		Note:   "the bodyclose analyzer already does this well; wire it in rather than re-implementing the escape analysis.",
	},
	{
		Code:   "ASSUMED_EXECUTABLE",
		Title:  "exec.Command on a literal binary not preflighted",
		Status: StatusProposed,
		Note:   "needs an allowlist: git/go are fair to assume; curl/node/npm/npx/python/playwright are environment claims that deserve a LookPath preflight with a clear error.",
	},
	{
		Code:   "MANUAL_PATH_SEPARATOR",
		Title:  "filesystem path built by string-joining with '/'",
		Status: StatusProposed,
		Note:   "the class the original ~ bug came from. FP hazard: distinguish filesystem paths from URLs/format strings — only flag concat that flows into os.Open/Stat/Create.",
	},
	{
		Code:   "UNVALIDATED_ENV",
		Title:  "os.Getenv consumed without a default or validation",
		Status: StatusProposed,
		Note:   "60+ call sites; only valuable with a severity/allowlist model, else it's noise.",
	},
	{
		Code:   "NONDETERMINISTIC_DEFAULT",
		Title:  "time.Now()/rand seeding a value that should be reproducible",
		Status: StatusProposed,
		Note:   "reproducibility tell rather than a network/OS one; narrow to seeds/IDs that feed persisted or compared output.",
	},
}
