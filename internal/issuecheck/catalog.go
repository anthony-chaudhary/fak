package issuecheck

import "strings"

// CatalogVersion binds a review to the exact checker vocabulary it used.
const CatalogVersion = "fak.issuecheck.catalog.v1"

// Check is one stable thought-check option. The agent chooses exactly five
// entries for a specific issue; this package validates the closed selection but
// deliberately does not pretend lexical scoring can judge semantic relevance.
type Check struct {
	ID       string `json:"id"`
	Name     string `json:"name"`
	Question string `json:"question"`
	When     string `json:"when"`
}

var catalog = [...]Check{
	{ID: "TC-01", Name: "Penny wise, pound foolish", Question: "Are small savings creating larger lifecycle costs?", When: "Cost, scope, test, or quality reductions."},
	{ID: "TC-02", Name: "Measure twice, cut once", Question: "What validates the assumptions before an irreversible move?", When: "Migrations, deletions, releases, or destructive changes."},
	{ID: "TC-03", Name: "The cure worse than the disease", Question: "Could the remedy cause more harm than the original problem?", When: "Fixes with broad side effects or failure modes."},
	{ID: "TC-04", Name: "Right ladder, wrong wall", Question: "What user or business outcome does this actually improve?", When: "Features and optimizations whose objective may be unclear."},
	{ID: "TC-05", Name: "Treating the symptom, feeding the cause", Question: "What root cause remains after this change?", When: "Recurring bugs, incidents, and workarounds."},
	{ID: "TC-06", Name: "The map is not the territory", Question: "Which model assumptions have been checked against operating reality?", When: "Designs based on abstractions, simulations, or stale descriptions."},
	{ID: "TC-07", Name: "Looking under the streetlight", Question: "What relevant evidence is missing because it is harder to obtain?", When: "Decisions driven by convenient measurements."},
	{ID: "TC-08", Name: "When a measure becomes a target", Question: "How could the metric improve while reality gets worse?", When: "KPIs, benchmarks, quotas, and evaluations."},
	{ID: "TC-09", Name: "What gets rewarded gets repeated", Question: "What behavior will these incentives actually encourage?", When: "Workflow, policy, permissions, and reward changes."},
	{ID: "TC-10", Name: "Chesterton's fence", Question: "Why does the existing behavior or constraint exist?", When: "Removal of legacy behavior, safeguards, or compatibility paths."},
	{ID: "TC-11", Name: "Throwing good money after bad", Question: "Ignoring prior investment, is this still the best next move?", When: "Continuation of a troubled project or approach."},
	{ID: "TC-12", Name: "Opportunity cost has no invoice", Question: "Which better alternative does this work delay or displace?", When: "Large, lengthy, or capacity-consuming proposals."},
	{ID: "TC-13", Name: "The second-domino check", Question: "What happens after the immediate intended result?", When: "Cross-system, behavioral, or policy changes."},
	{ID: "TC-14", Name: "More horsepower, same bottleneck", Question: "Is this the limiting constraint, and what evidence proves it?", When: "Performance, capacity, and throughput work."},
	{ID: "TC-15", Name: "A chain is only as strong as its weakest link", Question: "Which dependency determines end-to-end reliability?", When: "Multi-component reliability and delivery paths."},
	{ID: "TC-16", Name: "All eggs in one basket", Question: "What single failure could disable the whole system?", When: "Centralized services, stores, credentials, or dependencies."},
	{ID: "TC-17", Name: "Margin before miracle", Question: "What buffer exists for ordinary variance, failure, and bad luck?", When: "Tight estimates and optimistic operating assumptions."},
	{ID: "TC-18", Name: "One-way door or two-way door?", Question: "How reversible is this decision, and what is the rollback path?", When: "Architectural, data, rollout, and compatibility decisions."},
	{ID: "TC-19", Name: "Base rate before backstory", Question: "What normally happens in comparable cases?", When: "Forecasts and exceptional-outcome claims."},
	{ID: "TC-20", Name: "Survivors tell an incomplete story", Question: "Which failures or abandoned attempts are absent from the evidence?", When: "Success stories used to justify a decision."},
	{ID: "TC-21", Name: "Correlation needs a mechanism", Question: "What mechanism connects the proposed cause and effect?", When: "Analytics and causal assertions."},
	{ID: "TC-22", Name: "Absence of evidence is not evidence of absence", Question: "Was the test capable of detecting the suspected problem?", When: "Negative tests and could-not-reproduce outcomes."},
	{ID: "TC-23", Name: "Precision is not accuracy", Question: "What uncertainty is hidden by the precise-looking number?", When: "Detailed estimates, metrics, and forecasts."},
	{ID: "TC-24", Name: "Scale changes the game", Question: "Which assumptions fail at ten or one hundred times the scale?", When: "Growth, batching, fan-out, and multi-tenancy work."},
	{ID: "TC-25", Name: "The temporary bridge becomes the permanent road", Question: "Who removes the workaround, when, and what enforces that deadline?", When: "Compatibility layers, flags, manual steps, and temporary fixes."},
	{ID: "TC-26", Name: "Motion is not progress", Question: "Which observable outcome proves this activity mattered?", When: "Activity-heavy work with an unclear result."},
	{ID: "TC-27", Name: "If everything is urgent, nothing is prioritized", Question: "What does this outrank, and why?", When: "Priority, escalation, and interruption requests."},
	{ID: "TC-28", Name: "The last mile is half the journey", Question: "Are rollout, migration, documentation, adoption, and support covered?", When: "Features nearing implementation or release."},
	{ID: "TC-29", Name: "Robbing tomorrow to pay today", Question: "What future obligation is being created, and who owns it?", When: "Technical debt and rushed delivery."},
	{ID: "TC-30", Name: "The exception becomes the system", Question: "What happens if this workaround becomes common practice?", When: "Manual overrides, exemptions, and special cases."},
}

// Catalog returns a copy so callers cannot mutate the process-wide vocabulary.
func Catalog() []Check {
	out := make([]Check, len(catalog))
	copy(out, catalog[:])
	return out
}

// Lookup returns the catalog entry with the exact stable ID.
func Lookup(id string) (Check, bool) {
	id = strings.TrimSpace(id)
	for _, check := range catalog {
		if check.ID == id {
			return check, true
		}
	}
	return Check{}, false
}
