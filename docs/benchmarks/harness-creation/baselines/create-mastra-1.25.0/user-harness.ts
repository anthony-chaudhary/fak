import { createTool } from '@mastra/core/tools';
import { z } from 'zod';

export const config = {
  for: 'a support operator who must never mutate customer records',
  greeting: 'readonly support ready',
};

export const selfcheckTool = createTool({
  id: 'offline-selfcheck',
  description: 'Return the user-owned offline harness greeting.',
  inputSchema: z.object({ request: z.string() }),
  outputSchema: z.object({ reply: z.string(), mutationAllowed: z.boolean() }),
  execute: async ({ request }) => ({
    reply: `${config.greeting}: ${request}`,
    mutationAllowed: false,
  }),
});
