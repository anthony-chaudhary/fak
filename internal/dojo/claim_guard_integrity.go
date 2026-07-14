package dojo

// guardIntegrityClaim is an intentional zero-tolerance floor. It lives behind
// the additive registry seam so one KPI leaf does not churn the central map.
var guardIntegrityClaim = RegisterClaim("guard-integrity", "bad_stop_leak_rate", floor(0.0, true,
	"no stop the guard classifies as would-be-bad is allowed: bounded stand-down and fail-open rows are leaks; blocked continue rows are catches"))
