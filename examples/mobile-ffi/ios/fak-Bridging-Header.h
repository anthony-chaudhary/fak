/*
 * fak-Bridging-Header.h — the Swift/C bridge for the iOS sample. Set this as the
 * target's "Objective-C Bridging Header" (Build Settings ->
 * SWIFT_OBJC_BRIDGING_HEADER) so AgentGate.swift can call FakAdjudicate / FakFree
 * directly. It re-exports the module's C contract.
 */
#import "libfakmobile.h"
