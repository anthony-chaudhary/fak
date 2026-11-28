package gateway

// ADAPT/INSPIRE of ai-dynamo/aiperf (Apache-2.0),
// src/aiperf/workers/worker.py:810 @ab610add4e8876fca223197596c728e17365f381:
// dual-channel hot path. The dispatch side is receive-only; results return on a
// separate send-only channel, so a slow consumer of results can never stall
// dispatch. Component and host oracle only: not wired into a serve path.

// ChannelSplitJob is one unit of dispatched work.
type ChannelSplitJob struct {
	ID      int
	Payload string
}

// ChannelSplitResult is one completed unit returned to the producer side.
type ChannelSplitResult struct {
	ID  int
	Out string
}

// ChannelSplit owns both ends; callers receive only the half they need.
type ChannelSplit struct {
	dispatch chan ChannelSplitJob
	returns  chan ChannelSplitResult
}

// NewChannelSplit allocates the two independent channels.
func NewChannelSplit(dispatchDepth, returnDepth int) *ChannelSplit {
	return &ChannelSplit{
		dispatch: make(chan ChannelSplitJob, dispatchDepth),
		returns:  make(chan ChannelSplitResult, returnDepth),
	}
}

// Submit enqueues a job on the dispatch channel (producer side).
func (c *ChannelSplit) Submit(j ChannelSplitJob) { c.dispatch <- j }

// Dispatch is the worker's receive-only view of the dispatch channel.
func (c *ChannelSplit) Dispatch() <-chan ChannelSplitJob { return c.dispatch }

// Returns is the worker's send-only view of the return channel.
func (c *ChannelSplit) Returns() chan<- ChannelSplitResult { return c.returns }

// Results is the producer's receive-only view of the return channel.
func (c *ChannelSplit) Results() <-chan ChannelSplitResult { return c.returns }

// Close ends dispatch; workers drain it and stop.
func (c *ChannelSplit) Close() { close(c.dispatch) }
