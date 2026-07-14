package dojo

// guardStopClaim defends the Stop guard's catch rate as a one-way floor: the
// RSI loop may not recalibrate the guarantee down to an observed miss rate.
var guardStopClaim = RegisterClaim("guard-stop", "bad_stop_block_rate", floor(1.0, false,
	"every stop labeled would-be-bad is blocked; continue rows are catches and bounded stand-down or fail-open rows are misses"))
