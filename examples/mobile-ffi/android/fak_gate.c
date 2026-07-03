/*
 * fak_gate.c — the JNI bridge for the Android NDK sample. It exposes the fak
 * adjudicator floor (linked in as libfakmobile_android_arm64.a) to Kotlin/Java
 * as one native method: AgentGate.nativeAdjudicate(String) -> String.
 *
 * The Kotlin side (AgentGate.kt) calls this BEFORE it builds an android.content
 * .Intent, so a denied tool call never reaches startActivity/startService — the
 * enforcement layer Android leaves empty (one coarse grant auto-authorizing
 * "related sub-tools") is filled by fak's default-deny floor at the FFI seam.
 *
 * Build: see README.md — cross-compile the archive with the NDK clang, then
 * compile this file into the app's native library against libfakmobile.h.
 */
#include <jni.h>
#include <stdlib.h>
#include "libfakmobile.h"

/*
 * Java_dev_fak_sample_AgentGate_nativeAdjudicate mirrors the Kotlin package
 * dev.fak.sample and class AgentGate. It converts the Java string to UTF-8,
 * routes it through FakAdjudicate, and returns the JSON Decision string —
 * releasing the malloc'd C buffer with FakFree so the boundary does not leak.
 */
JNIEXPORT jstring JNICALL
Java_dev_fak_sample_AgentGate_nativeAdjudicate(JNIEnv *env, jobject thiz,
                                               jstring toolCallJson) {
  (void)thiz;
  const char *in = (*env)->GetStringUTFChars(env, toolCallJson, NULL);
  if (in == NULL) {
    /* OOM already pending on the JVM; return a fail-closed DENY. */
    return (*env)->NewStringUTF(
        env, "{\"allow\":false,\"verdict\":\"DENY\",\"reason\":\"MALFORMED\"}");
  }

  char *decision = FakAdjudicate((char *)in);
  (*env)->ReleaseStringUTFChars(env, toolCallJson, in);

  jstring out = (*env)->NewStringUTF(env, decision);
  FakFree(decision); /* the boundary owns the buffer; free exactly once */
  return out;
}
