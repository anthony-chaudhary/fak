package main

// recover_config.go — the CONFIG class of the `fak recover` catalog.
//
// recover.go's original vocabulary is the guard/DOS refusal set: OFF_TRUNK,
// COLLISION_RISK, MERGE_IN_PROGRESS — failures of a WORKING TREE. That is one
// half of what stops a run. The other half is configuration: a capability floor
// that will not parse, an env var the operator believes holds a key and does
// not, a `--reset-on-budget` with nothing to reset. Those bailed with a bare
// sentence and no token, so they had no recovery to look up and no way for a
// fleet log to be counted by cause.
//
// These plans complete the map. Each token here is emitted by a real bail site
// (bail.go's vocabulary is the emitted set; the ratchet test binds the two), and
// each plan answers the same three questions the tree-class plans do: what
// actually went wrong, what to READ to see it yourself, and what CLEARS it.
//
// # Placeholders and what "executable" means here
//
// A tree-class recovery is executable because its commands are complete —
// `git fetch origin main` needs nothing from the failure. A config recovery is
// about a specific path, env var, or address, so its commands carry
// placeholders (<path>, <env>, <addr>) that a static catalog cannot fill. The
// bail site knows those values, so it prints them pre-bound:
//
//	next:   fak recover POLICY_LOAD_FAILED --set path=guard-policy.json
//
// recover.go substitutes --set bindings into every step and note before running
// or printing anything, and refuses --execute while a placeholder is still
// unbound rather than shelling out a literal `<path>`.
//
// # Why so few steps are Safe
//
// Safe means `--execute` runs it unattended, so the bar is: read-only, and
// correct without knowing what the operator MEANT. `fak policy --check` clears
// it — it prints the precise validation error and changes nothing. Most config
// bails do not: fak cannot invent a missing credential, cannot pick a context
// budget, and cannot know whether the fix for a bad `--reset-on-budget` is to
// add a budget or drop the flag. Guessing there would turn a loud refusal into
// a silent mis-configuration, which is the failure this whole path exists to
// stop. Those plans stay manual and spend their words on the checks instead.

// configRecoveryPlans returns the config-class recovery plans, keyed by the same
// reason tokens bail.go emits. Kept separate from recoveryPlans' tree class so
// the two vocabularies stay independently readable; recoveryPlans merges them.
func configRecoveryPlans() map[string]recoveryPlan {
	return map[string]recoveryPlan{
		bailPolicyLoadFailed: {
			Reason:     bailPolicyLoadFailed,
			Summary:    "the capability floor at --policy did not load; fak refuses to serve on a floor it could not read",
			Executable: true,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "policy", "--check", "<path>"}, Summary: "validate the manifest and print the precise rejection (read-only)", Safe: true},
				{Argv: []string{"fak", "policy", "--dump"}, Summary: "print the default manifest to diff your file against", Safe: true},
			},
			Notes: []string{
				"a floor that fails to load is never downgraded to a permissive default — that is the refusal working, not a bug",
				"read the bail's `file` line first: it says whether <path> could not be READ (a wrong or relative path — fak resolves it against ITS working directory, which under an MCP launcher is the editor's) or was read and did not PARSE",
				"every deny in the manifest must cite a closed-vocabulary reason; a free-text reason is the most common rejection",
				"to serve without a custom floor at all, omit --policy rather than passing a file you have not validated",
			},
		},
		bailPolicyCheckNoFile: {
			Reason:     bailPolicyCheckNoFile,
			Summary:    "--policy-check validates a manifest and none was given",
			Executable: true,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "policy", "--dump"}, Summary: "print the default manifest as a starting file", Safe: true},
			},
			Notes: []string{"pass the file too: fak serve --policy FILE --policy-check"},
		},
		bailKeyEnvUnset: {
			Reason:     bailKeyEnvUnset,
			Summary:    "a key-bearing env var was named but is unset or empty; fak refuses to run the auth path with no secret behind it",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "doctor", "mcp"}, Summary: "check how this process's environment is actually populated"},
			},
			Notes: []string{
				"set <env> in the environment that LAUNCHES fak — a var exported in your shell does not reach a service, a launchd/systemd unit, or an MCP server started by an editor",
				"or omit the flag: naming the var is what arms the requirement, so dropping it serves unauthenticated ON PURPOSE rather than by accident",
				"fak never prints the value; a check that echoes the secret to confirm it is set defeats the point",
			},
		},
		bailKeyPrincipalUnresolved: {
			Reason:     bailKeyPrincipalUnresolved,
			Summary:    "a --key-principal binding did not resolve, so the tenant keyset is only half armed",
			Executable: false,
			Notes: []string{
				"each spec is PRINCIPAL=ENV_VAR and every named var must hold a distinct non-empty key",
				"this fails closed for a reason: with no keyset matched, the gateway attributes the turn from the caller-supplied X-Fak-Principal header, so a forgotten binding lets one tenant assert another",
				"two principals sharing one key is also refused — issue each tenant its own",
			},
		},
		bailBudgetFlagIncoherent: {
			Reason:     bailBudgetFlagIncoherent,
			Summary:    "the session-budget flags contradict each other",
			Executable: false,
			Notes: []string{
				"--reset-on-budget resets a trace when its context budget is spent, so it needs --context-budget-tokens N with N > 0",
				"either add the budget or drop --reset-on-budget; fak will not pick one, because the two mean different things about how you want long sessions to end",
				"a negative --context-budget-tokens or --engine-cache-idle-timeout is always a typo — both are magnitudes, and 0 is the disable value",
			},
		},
		bailAddrRequired: {
			Reason:     bailAddrRequired,
			Summary:    "fak serve was given no transport: --addr was cleared and --stdio was not passed",
			Executable: false,
			Notes: []string{
				"pass --addr HOST:PORT for the HTTP surface, or --stdio to serve MCP over stdin/stdout",
				"--addr has a default (127.0.0.1:8080), so an empty one means it was explicitly set to the empty string — check for an --addr= with nothing after it, or a shell variable that expanded to nothing",
			},
		},
		bailBackendUnavailable: {
			Reason:     bailBackendUnavailable,
			Summary:    "the named --backend is not registered in this binary",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "doctor", "serve"}, Summary: "report the host's decode tier and serve readiness"},
			},
			Notes: []string{
				"a device backend needs BOTH a matching build tag at compile time and a reachable device at runtime — the bail line lists the backends this binary actually registered",
				"the name is not silently downgraded to CPU: a typo that quietly served on the wrong device would misreport every throughput number taken from it",
				"omit --backend entirely to serve on the CPU path deliberately",
			},
		},
		bailRouteManifestInvalid: {
			Reason:     bailRouteManifestInvalid,
			Summary:    "the model-routing manifest or account roster did not load",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"git", "diff", "--", "<path>"}, Summary: "review recent edits to the manifest"},
			},
			Notes: []string{
				"a malformed manifest fails loud rather than default-routing every call to the kernel default — a mis-routed model is a security boundary, not a preference",
				"a roster carries env var NAMES only, never secrets; a literal key in the file is a rejection, not a shortcut",
			},
		},
		bailWeightsRequired: {
			Reason:     bailWeightsRequired,
			Summary:    "the requested artifact is derived from the model's own header, and no weights were given",
			Executable: false,
			Steps: []recoveryStep{
				{Argv: []string{"fak", "ls"}, Summary: "list the locally cached models --gguf can name"},
			},
			Notes: []string{
				"pass --gguf FILE, an hf:// URI, or a registry alias; the alias resolves through the model registry before anything opens it",
				"--plan-json reads the GGUF header only — it loads no tensor and binds no listener, so pointing it at a large remote model is cheap",
			},
		},
		bailUnauthenticatedBind: {
			Reason:     bailUnauthenticatedBind,
			Summary:    "the requested --addr is reachable from off this host and no inbound token door is configured",
			Executable: false,
			Notes: []string{
				"bind loopback instead: --addr 127.0.0.1:8080 (the default)",
				"or require a bearer: --require-key-env FAK_API_KEY, with that variable actually set in the launching environment",
				"or bind per-tenant keys: --key-principal acme=ACME_KEY (repeatable, one distinct key per principal)",
				"--stdio is exempt because it binds no socket at all; if the surface is for one local client, prefer it",
				"if this host really is meant to serve an unauthenticated interface, --unsafe-allow-unauthenticated-bind proceeds and says so loudly on every boot — that is the intended escape, not a workaround to hide",
			},
		},
		bailNotAWorkspace: {
			Reason:     bailNotAWorkspace,
			Summary:    "the working directory is not inside a fak workspace (no dos.toml upward), so the corpus, devindex, and session-state planes will bind the wrong tree",
			Executable: true,
			Steps: []recoveryStep{
				{Argv: []string{"git", "rev-parse", "--show-toplevel"}, Summary: "print the enclosing repo root to compare against your fak checkout", Safe: true},
			},
			Notes: []string{
				"launch fak from a fak checkout, or point the launcher at one",
				"when fak runs as an MCP server the cwd is the EDITOR's, not yours — set the server entry's \"cwd\" to your fak workspace root",
				"this one is a warning, not a refusal: fak still serves, which is exactly why a silently mis-bound tree is worth checking before you trust what it reports",
			},
		},
	}
}
