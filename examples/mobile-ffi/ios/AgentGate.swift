import Foundation

// AgentGate routes an on-device agent's PROPOSED tool call through fak's
// adjudicator floor (native C, via the bridging header -> libfakmobile) BEFORE
// the call is allowed to become an Apple App Intent. Apple's intent resolver
// trusts its own dispatch; this sample makes each proposed call pass fak's
// default-deny floor first — least-privilege-per-tool as a gate.
//
// Wiring: add libfakmobile_ios_arm64.a and libfakmobile.h to the target and set
// the bridging header to fak-Bridging-Header.h (see README.md). FakAdjudicate /
// FakFree then resolve as C functions.
enum AgentGate {

    struct Decision: Decodable {
        let allow: Bool
        let verdict: String
        let reason: String?
    }

    /// The one FFI seam: JSON tool call in, JSON Decision out. The C buffer is
    /// malloc'd by fak and released here with FakFree so the boundary is leak-free.
    static func adjudicate(_ toolCallJSON: String) -> Decision {
        let cResult: UnsafeMutablePointer<CChar>? = toolCallJSON.withCString { cIn in
            FakAdjudicate(UnsafeMutablePointer(mutating: cIn))
        }
        guard let cResult else {
            return Decision(allow: false, verdict: "DENY", reason: "MALFORMED")
        }
        defer { FakFree(cResult) }
        let json = String(cString: cResult)
        guard let data = json.data(using: .utf8),
              let d = try? JSONDecoder().decode(Decision.self, from: data)
        else {
            return Decision(allow: false, verdict: "DENY", reason: "MALFORMED")
        }
        return d
    }

    /// The enforcement point: perform the App Intent action ONLY on an allow. A
    /// denied dangerous call (send_sms) returns false and the intent never fires;
    /// a benign one (get_battery_level) proceeds.
    @discardableResult
    static func performIfAllowed(_ toolCallJSON: String, dispatch: () -> Void) -> Bool {
        let d = adjudicate(toolCallJSON)
        guard d.allow else { return false }   // denied at the floor — do not dispatch
        dispatch()
        return true
    }
}

// Illustrative call site (an on-device agent proposing two tool calls):
//
//   // dangerous: denied at the floor, the App Intent never fires
//   AgentGate.performIfAllowed(#"{"tool":"send_sms","args":{"to":"+1900"}}"#) {
//       SendMessageIntent().donateAndPerform()
//   }                                             // -> false (POLICY_BLOCK)
//
//   // benign: continues to dispatch
//   AgentGate.performIfAllowed(#"{"tool":"get_battery_level","args":{}}"#) {
//       ShowBatteryIntent().donateAndPerform()
//   }                                             // -> true (ALLOW)
