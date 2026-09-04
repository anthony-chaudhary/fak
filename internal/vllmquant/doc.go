// Package vllmquant decides which vLLM quantization kernel may serve a
// quantized artifact, and refuses to invent an answer when the evidence does
// not contain one.
//
// vLLM serves a checkpoint's quantization method through one of several
// kernels, and which of them is admissible is a joint property of three
// independent facts: what the artifact declares (method, weight width, group
// size, symmetry, activation scheme), what the server build advertises (its
// version and its compiled kernel set), and what the device can execute (its
// compute capability). A GPTQ 4-bit checkpoint is servable by the gptq kernel
// on a Pascal card and by gptq_marlin only on Ampere and later; the same
// checkpoint quantized asymmetrically drops gptq_marlin entirely. Collapsing
// those three axes into one "supported" flag is what makes a launch fail at
// load time instead of at admission time.
//
// The rule this leaf exists to hold: it never picks a winner among kernels a
// server advertises. When more than one kernel can serve the artifact, the
// choice belongs to whoever declared a preference — the server, via
// KernelOrderIsPreference, or the runtime itself, via RuntimeSelects. With
// neither, the selection delegates: it reports the candidates and the launch
// arguments that are licensed without naming a kernel, rather than ranking them
// by a fak-owned opinion. Marlin is not "better" here; it is admissible or it
// is not.
//
// Every input gets a typed outcome — supported, delegate, unsupported, abstain,
// or refuse — and a stable reason code. The last three are deliberately
// distinct: unsupported is a fact about the build ("nothing you compiled can
// serve this, on this device"), abstain is a missing witness ("nobody declared
// the group size"), and refuse is a contradiction in the declaration itself
// ("fp8 at 4 bits", "a group size of 0"). Only the first is fixed by changing
// the build; only the last must be fixed at the producer. Nothing falls back
// silently: an unparseable version or an unrecognized kernel abstains rather
// than being guessed into the nearest known shape. A supported result licenses
// the exact launch arguments it returns and nothing else; it is not a
// throughput, quality, or measured-hardware claim — and not even a claim about
// which kernel ultimately runs, since vLLM resolves that at load time.
//
// When no kernel is admissible, the verdict carries the whole family in
// Excluded with the single reason each kernel was dropped, so "unsupported" is
// answerable: it names what the build would have needed.
//
// The version and compute-capability floors in the kernel table are this
// contract's own conservative admission thresholds under SchemaVersion, not a
// claim about the exact upstream release in which a kernel first appeared. They
// are data: a newer schema may lower them on evidence.
//
// It defines no fak-owned artifact format and requires no conversion — the
// descriptor is the producer's own metadata, read as-is, under the field names
// a Hugging Face quantization_config and a vLLM build actually carry (bits,
// group_size, sym, methods, a dotted compute_capability like "8.0").
//
// Invariant: vLLM quantization kernel selection is fail-closed and deterministic.
// Guard: undeclared configurations, unknown quantization methods, and contradictory
// descriptor parameters refuse admission rather than guessing fallback runtime kernels.
//
// Tier: foundation (1) - see internal/architest. Stdlib-only.
package vllmquant
