import { selfcheckTool } from './user-harness.ts';

const result = await selfcheckTool.execute?.({ request: 'refund status' }, {} as never);
if (result?.reply !== 'readonly support ready: refund status' || result.mutationAllowed !== false) {
  throw new Error(`selfcheck failed: ${JSON.stringify(result)}`);
}
console.log(`BASELINE_SELFCHECK ok reply=${JSON.stringify(result.reply)} mutation_allowed=${result.mutationAllowed}`);
