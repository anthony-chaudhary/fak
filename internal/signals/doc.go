// Package signals defines plain-English BEHAVIORAL signals over an agent's turns —
// the behavioral complement to the structural anti-pattern detectors.
//
// # Shape vs behavior
//
// A structural detector answers "does this trajectory have shape X?" — a regex over
// tool sequences, a fixed threshold on step count. A behavioral signal answers a
// question you can only phrase in English: "did the agent give up before it had
// evidence?", "did it apologize instead of fixing the bug?", "did it silently widen the
// scope it was asked for?". You cannot grep for those. So a [Signal] carries three
// things: a natural-language Prompt (the behavioral question), an output Schema (the
// structured verdict a judge must return, so the answer is machine-joinable, not prose),
// and a SampleRate (behavioral judging costs a model call — you sample, you don't judge
// every turn).
//
// # Deterministic sampling — no RNG
//
// [Signal.Sampled] decides membership by hashing (signal name, item id), NOT by a random
// draw. Same signal + same item => same decision, forever: a sampled set is reproducible
// across runs and across machines, and a test can assert exactly which items a rate
// admits. Over many ids the admitted fraction converges to SampleRate.
//
// # The judge boundary
//
// [Run] samples the items, then asks an injected [Evaluator] to answer each sampled
// item's Prompt and VALIDATES the returned verdict against the signal's Schema before
// accepting it — a judge that answers off-schema is a caught error, not silent drift.
// The model call lives behind the Evaluator interface so the sampling, schema, and
// config logic are unit-testable with a fake judge and no network. [RenderPrompt] builds
// the exact judge instruction (behavioral question + "emit JSON matching this schema" +
// the item) that a production Evaluator sends.
//
// Borrowed from agent-lens Signals (behavioral eval: nl prompt + schema + sampleRate).
package signals
