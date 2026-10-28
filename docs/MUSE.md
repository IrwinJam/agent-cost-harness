# Using it with Muse

Everything here was measured on Meta's Model API with muse-spark-1.3, on a public customer-service agent
benchmark. The full numbers are in [RESULTS.md](RESULTS.md).

## Setup

```bash
export HARNESS_API_KEY=your-meta-model-api-key   # Windows PowerShell: $env:HARNESS_API_KEY="..."
python -m harness serve --muse --policies cache --max-per-day 2
```

Point your agent's OpenAI base URL at `http://127.0.0.1:8787/s/<conversation-key>/v1`, with a new key
per conversation. `--muse` sets the base URL to `https://api.meta.ai/v1`, looks up Muse prices, and
checks your requests against Muse's rules before they're sent:

- `tool_choice` can only be `"auto"`.
- Reasoning can't be turned off. Effort `"none"` is a 400, so use `"minimal"` or `"low"`.
- `previous_response_id` makes usage count only the newest turn, which would make every cost number
  wrong, so the harness refuses it. Send the whole conversation.

The prices in `harness/prices.yaml` come from Meta's published list. They haven't been checked against a
real bill yet.

## What we found in Muse's API

These are two behaviors that cost Muse developers money without them knowing. Both showed up in the
profiler and were confirmed in separate runs.

### 1. Without a cache key, identical prompts keep missing the cache

Muse caches prompt starts automatically. But in our baseline, where every conversation started with the
same tools and system prompt, **16% of input was still billed at full price**. The profiler marked these
as "prefix unchanged but not served from cache", which points at routing: the call landed on a server
that didn't have the prompt cached.

Sending one `prompt_cache_key` for every conversation with the same start fixed it:

| | No cache key | Shared cache key |
|---|---:|---:|
| Input from cache (240-run benchmark) | 83.6% | 94.4% |
| Calls that mostly missed the cache | 37 of 1,554 | 1 of 1,548 |
| Input from cache (64 live conversations) | 71.2% | 97.1% |
| Agent cost | baseline | **-21%** (benchmark), **-47%** (short live conversations) |

`cache_layout` does this by default. If you don't want the harness, you can get most of this by sending
the same `prompt_cache_key` on every request that shares a system prompt and tools.

### 2. Each reasoning effort level has its own cache

When a conversation switched from `low` to `medium` effort, only the shared tools and system prompt came
from the cache. The rest of the conversation was billed again at full price. Switching back to `low`
hit the cache right away. So changing effort mid-conversation is expensive on Muse.

The fix is to pick one effort per conversation. On this benchmark, **`low` for the whole conversation
cut another 20%** on top of the cache key, and tasks passed went from 92% to 94%. That's
`--policies cache,reason --set reason.escalate=false`. Muse at `medium` only reasoned about 200 to 300
tokens per call here, and `low` did the job just as well.

## Why this is good for Meta too

A tool that lowers what developers pay Meta can look like a bad deal for Meta. It isn't:

- **Cache hits are the cheap path to serve.** A cached prompt start skips most of the compute for that
  part of the request. Moving traffic from misses to hits lowers what each call costs Meta to run, not
  just what it bills.
- **Less reasoning means fewer GPU-seconds.** Cutting reasoning tokens by 60% with no drop in success
  frees up capacity.
- **Cheaper agents get built and kept.** Cost is one of the main reasons agent projects stall. An agent
  that costs a third less is more likely to ship and keep running on Muse.
- **The cache-key finding is a default Meta could change.** If the API grouped requests by prompt start
  on its own, every Muse developer would get the first saving without doing anything.

## Ideas for Meta's API team

- Route by prompt start automatically, or document `prompt_cache_key` front and center. Our numbers
  suggest a lot of developers are paying for cache misses they don't know about.
- Say in the docs whether the cache is shared across reasoning effort levels. Developers who switch
  effort mid-conversation are paying for it.
- Publish guidance on when `low` effort is enough for agent work. On this benchmark it was.
