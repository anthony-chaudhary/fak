package dev.fak.sample

import android.content.Context
import android.content.Intent
import org.json.JSONObject

/**
 * AgentGate routes an on-device agent's PROPOSED tool call through fak's
 * adjudicator floor (native, via JNI -> libfakmobile) BEFORE the call is allowed
 * to become an android.content.Intent. This is the concrete fill for the gap
 * Android names itself: a single coarse grant auto-authorizes "related
 * sub-tools" ("excessive agency"). fak makes each proposed call pass the
 * default-deny floor first — least-privilege-per-tool as a gate, not developer
 * discipline.
 *
 * Wiring: place libfakmobile_android_arm64.a + fak_gate.c in the module's
 * native build (see README.md) so the shared lib "fakgate" exports
 * nativeAdjudicate.
 */
class AgentGate {

    /** The one native seam: JSON tool call in, JSON Decision out. */
    private external fun nativeAdjudicate(toolCallJson: String): String

    /** Parsed form of the FFI Decision (see libfakmobile.h). */
    data class Decision(val allow: Boolean, val verdict: String, val reason: String)

    fun adjudicate(toolCallJson: String): Decision {
        val o = JSONObject(nativeAdjudicate(toolCallJson))
        return Decision(
            allow = o.optBoolean("allow", false),
            verdict = o.optString("verdict", "DENY"),
            reason = o.optString("reason", ""),
        )
    }

    /**
     * dispatchIfAllowed is the enforcement point: the proposed call is
     * adjudicated, and the Intent is started ONLY on an allow. A denied
     * dangerous call (e.g. send_sms) returns false and never reaches
     * startActivity; a benign one (e.g. get_battery_level) proceeds.
     */
    fun dispatchIfAllowed(ctx: Context, toolCallJson: String, intent: Intent): Boolean {
        val d = adjudicate(toolCallJson)
        if (!d.allow) {
            // Denied at the floor — surface the reason; do NOT dispatch.
            return false
        }
        ctx.startActivity(intent)
        return true
    }

    companion object {
        init {
            System.loadLibrary("fakgate")
        }
    }
}

/*
 * Illustrative call site (an on-device agent proposing two tool calls):
 *
 *   val gate = AgentGate()
 *
 *   // dangerous: denied at the floor, no SMS Intent is ever built/dispatched
 *   gate.dispatchIfAllowed(ctx,
 *       """{"tool":"send_sms","args":{"to":"+1900PREMIUM"}}""",
 *       Intent(Intent.ACTION_SENDTO))            // -> false (POLICY_BLOCK)
 *
 *   // benign: continues to dispatch
 *   gate.dispatchIfAllowed(ctx,
 *       """{"tool":"get_battery_level","args":{}}""",
 *       Intent("dev.fak.sample.SHOW_BATTERY"))   // -> true (ALLOW)
 */
