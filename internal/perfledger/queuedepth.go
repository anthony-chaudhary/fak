// Adapted from ai-dynamo/aiperf
// from the original): the _rp_queue_depth token-pressure model. Past a
// per-request token count, tokenizing a finished record takes longer than the
// workers take to finish the next one, so records back up in the processor's
// pull queue in proportion to the token count, up to a cap.
//
// The arithmetic lives in rpQueueDepth, which the memory model already uses;
// this file is the exported, input-hardened face of it.
package perfledger

const (
	// QueueTokenPressureThreshold is the per-request token count (input plus
	// output) above which the queue starts to back up.
	QueueTokenPressureThreshold = 10000
	// QueueTokenPressureCap is the largest multiple of the base queue depth
	// the model predicts, reached at ten times the threshold.
	QueueTokenPressureCap = 10.0
)

// QueueTokenPressure is the multiple of the base queue depth the model
// predicts for requests averaging avgISL input and avgOSL output tokens: 1 at
// or below the threshold, then tokens/threshold, capped. Negative counts are
// read as zero.
func QueueTokenPressure(avgISL, avgOSL int) float64 {
	tokens := float64(max(avgISL, 0)) + float64(max(avgOSL, 0))
	if tokens <= QueueTokenPressureThreshold {
		return 1
	}
	return min(tokens/QueueTokenPressureThreshold, QueueTokenPressureCap)
}

// QueueDepth is the modeled pull-queue depth of one record processor serving
// concPerRP concurrent requests: the concurrency scaled by the token pressure
// and truncated to whole records. Negative inputs are read as zero, and a
// concurrency past 2^52 is held there.
func QueueDepth(concPerRP, avgISL, avgOSL int) int {
	// Every count at or past the cap point gives the same answer, so clamping
	// there keeps the sum of two huge counts from overflowing.
	const capTokens = QueueTokenPressureThreshold * int(QueueTokenPressureCap)
	isl := min(max(avgISL, 0), capTokens)
	osl := min(max(avgOSL, 0), capTokens)
	// Past 2^52 the scaled depth is no longer an exact integer in float64,
	// and soon after does not fit an int at all.
	conc := min(max(concPerRP, 0), 1<<52)
	return rpQueueDepth(conc, isl, osl)
}
