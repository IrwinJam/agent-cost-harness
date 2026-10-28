// A small tool-calling agent that talks to the harness instead of the provider.
//
//   python -m harness serve --simulate --ab          (or --base-url ... for a real provider)
//   cd examples/typescript && npm install && npx tsx agent.ts
//
// Nothing here knows about the harness except the base URL. Each conversation gets its own key in
// the path, so the harness can trace it and, with --ab, put it in one half of the test.

import OpenAI from "openai";

const HARNESS = process.env.HARNESS_URL ?? "http://127.0.0.1:8787";
const MODEL = process.env.MODEL ?? "example-model";

const tools: OpenAI.Chat.ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "get_order",
      description: "Look up an order by its id.",
      parameters: { type: "object", properties: { order_id: { type: "string" } }, required: ["order_id"] },
    },
  },
];

function getOrder(orderId: string): string {
  return JSON.stringify({ order_id: orderId, status: "shipped", items: ["lamp"], eta: "2 days" });
}

async function runConversation(key: string, question: string): Promise<string> {
  // Any API key works here. The real one stays in the harness.
  const client = new OpenAI({ baseURL: `${HARNESS}/s/${key}/v1`, apiKey: "unused" });
  const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
    { role: "system", content: "You help customers with their orders. Use the tools when you need order details." },
    { role: "user", content: question },
  ];

  for (let step = 0; step < 5; step++) {
    const reply = await client.chat.completions.create({ model: MODEL, messages, tools });
    const message = reply.choices[0].message;
    messages.push(message);
    if (!message.tool_calls?.length) return message.content ?? "";
    for (const call of message.tool_calls) {
      const args = JSON.parse(call.function.arguments || "{}");
      messages.push({ role: "tool", tool_call_id: call.id, content: getOrder(args.order_id) });
    }
  }
  return "(gave up after 5 steps)";
}

const questions = ["Where is order 1001?", "Has order 2002 shipped yet?", "When will order 3003 arrive?"];
for (const [i, question] of questions.entries()) {
  const answer = await runConversation(`ts-${Date.now()}-${i}`, question);
  console.log(`> ${question}\n${answer}\n`);
}
console.log("See where the money went: python -m harness profile --db runs/serve.sqlite --list");
