/*
 * libfakmobile.h — the stable, hand-authored declaration of fak's mobile FFI
 * surface. Both platform samples (android/, ios/) include THIS header; it is
 * the C contract, independent of which arm64 archive is linked.
 *
 * `go build -buildmode=c-archive -o libfakmobile_<goos>_<goarch>.a .` also emits
 * a generated header (libfakmobile_<goos>_<goarch>.h) carrying cgo's boilerplate
 * typedefs; the two functions it declares are exactly the two below. We keep
 * this minimal copy checked in so the samples read and compile against a fixed
 * surface without first running the cross-compile.
 *
 * Contract:
 *   - FakAdjudicate takes a proposed tool call as a JSON C string
 *       {"tool":"send_sms","args":{...}}
 *     and returns a JSON Decision C string
 *       {"tool":"send_sms","allow":false,"verdict":"DENY",
 *        "reason":"POLICY_BLOCK","by":"mobile/floor"}
 *     Dispatch the Android Intent / Apple App Intent IFF "allow" is true.
 *   - The returned char* is malloc'd and OWNED by the caller: release it with
 *     FakFree exactly once. FakFree(NULL) is a no-op.
 *   - Both calls are thread-safe and side-effect-free (the reference floor holds
 *     an immutable policy); the same input always yields the same Decision.
 */
#ifndef FAK_LIBFAKMOBILE_H
#define FAK_LIBFAKMOBILE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Adjudicate one proposed tool call. Returns a malloc'd JSON Decision string
 * the caller must release with FakFree. Never returns NULL for well-formed
 * input; malformed input yields a DENY(MALFORMED) Decision, not a null. */
extern char *FakAdjudicate(char *toolCallJSON);

/* Release a string returned by FakAdjudicate. FakFree(NULL) is a no-op. */
extern void FakFree(char *p);

#ifdef __cplusplus
}
#endif

#endif /* FAK_LIBFAKMOBILE_H */
