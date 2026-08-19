// Generated from schema/protocol.schema.json. Do not edit.
export const PROTOCOL = "fak.harness-sidecar/v1";
export interface Identity { name: string; version: string; digest: string }
export interface Limits { max_frame: number; max_inflight: number; cancel_grace: number }
export interface Handshake { protocol: typeof PROTOCOL; identity: Identity; capabilities: string[]; limits: Limits }
export interface Request { id: string; method: string; payload?: unknown; deadline_unix_nano?: number }
export interface Response { id: string; payload?: unknown; error?: string; done: boolean }
export type Frame = {kind:"handshake";handshake:Handshake}|{kind:"request";request:Request}|{kind:"response";response:Response}|{kind:"cancel";cancel_id:string};
export function encodeFrame(frame: Frame, maxFrame: number): Uint8Array {
  const body = new TextEncoder().encode(JSON.stringify(frame));
  if (!body.length || body.length > maxFrame) throw new Error("frame exceeds negotiated bound");
  const out = new Uint8Array(4 + body.length); new DataView(out.buffer).setUint32(0, body.length); out.set(body, 4); return out;
}
