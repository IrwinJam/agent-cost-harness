# How much does it save?

This guide answers that question with the numbers from a real test, then helps you guess whether your
project would see the same. The full test write-up is in [RESULTS.md](RESULTS.md).

## The number

On a customer-service agent benchmark, `cache_layout` cut what the agent spent by **21.5%**. The 95%
confidence interval runs from 17% to 25%. In plain terms, for every $100 the agent was spending, it
saved somewhere between $17 and $25, most likely about $21.

The benchmark also pays for a model that plays the customer and grades some answers. The harness
doesn't touch those calls. Counting them, the total bill dropped about 15%.

Adding `reasoning` with `escalate=false` (low reasoning effort for the whole conversation) cut another
20 to 21% on the same 40 tasks, in two separate runs, one of them side by side on the same day. Success
didn't drop in either. Together that's roughly 37% off the agent's cost. See sections 6 and 7 of
[RESULTS.md](RESULTS.md).

## The test behind it

- 40 tasks, each run 3 times per setup, so 240 runs in all. $17.10 total.
- Same model and same tasks for both setups. The only difference was whether the calls
  went through `cache_layout`.
- The two setups took turns going first on each task, so neither always got a warm cache.
- The test plan was written and fingerprinted before the run. The pass/fail rule couldn't be changed
  after seeing the results.

**How the confidence interval is worked out.** Each task gets an average cost under both setups. The
harness then picks 40 tasks at random, with repeats allowed, and works out the cost change for that
pick. It does that 10,000 times. The middle 95% of those results is the interval. Picking whole tasks
keeps each baseline run matched with its own `cache_layout` run, so an expensive task can't make one
side look worse just by luck.

**Why it's caching and not luck.** Both setups made 12.9 calls per run on average, and
wrote about the same number of tokens. So the agent didn't save money by doing less. What changed was
how much input came from the cache:

| | Baseline | cache_layout |
|---|---:|---:|
| Input from cache | 83.6% | 94.4% |
| Calls that mostly missed the cache | 37 of 1,554 | 1 of 1,548 |
| Uncached input per run | $0.0192 | $0.0065 |
| Agent cost per run | $0.0502 | $0.0394 |

The saving shows up in that uncached input line. Cached input and output went up a little, and the
net is the 21.5%.

## Did it make the agent worse?

Maybe a little, and the test couldn't rule it out. Success went from 94% to 92%. The plan said the
harness had to show success dropped by 3 points or less, and it couldn't.

Part of that is on the plan. With 40 tasks, the uncertainty on success is about 6 points either way. So
even if both setups had scored exactly the same, the test still would have failed the 3-point rule. To
check a 3-point margin you'd need roughly 150 tasks, or you'd have to accept a margin of about 6
points.

The rest is one task the agent kept getting wrong with `cache_layout` (0 of 3, vs 3 of 3 on baseline).
On this benchmark `cache_layout` doesn't change any words the model sees. It sorts the tool list and
adds a cache key. So that task is either chance or the new tool order nudging the model. If you use
it, check your own success rate, not just your bill.

## Guessing your own savings

The savings only come from input tokens that start hitting the cache. You can get a rough number from
your own traces:

```
saved per call = input tokens x (miss rate before - miss rate after) x (full price - cached price)
```

For the test above: 93,345 input tokens per run, misses going from 16.4% to 5.6%, and $1.10 per
million saved on each token moved into the cache. That comes to about $0.011 per run, which matches
the measured drop.

Run `python -m harness profile` on a few real conversations first. If your input is already over 95%
cached and the cache breaks list is empty, there isn't much left to save.

## Who gets the most out of it

**Agents that make lots of calls per conversation.** Every call resends the prompt start. An agent loop
that calls a tool 15 times pays for that start 15 times. Here it averaged 13 calls, and the tool list
plus the system prompt alone were about 27% of the cost.

**Big, fixed prompt starts.** A long policy document or 20+ tool definitions. The bigger
that block is next to what changes each turn, the more caching it is worth.

**Many conversations at once with the same start.** This is what paid off in the test. The prompt start
never changed, but the provider kept sending calls to servers that didn't have it cached. One shared
cache key for every conversation with the same start fixed that. Support bots and batch jobs that
run many copies of one agent fit here.

**Prompt starts that change for no reason.** A timestamp in the system prompt or tools listed in a
different order each time. Either one breaks the cache on every call.
`cache_layout` moves those lines out of the way. The test above didn't have this problem, so this part
has only been checked on the simulator, where it cut cost by about 80%. Real numbers will be lower.

**Long tool outputs that get resent.** File reads, search results, logs or web pages that stay in the
conversation for 10 more turns. `output_handles` and `masking` are made for this. They haven't been
tested on real runs yet, because the benchmark's tool outputs were small.

## Who won't see much

- One-off calls. There's no second call to reuse the cache.
- Short prompts. Most providers only cache past a minimum size, often around 1,000 tokens.
- Output-heavy work. In the test, output was about half of the agent's cost after the change, and
  caching does nothing for output. Reasoning models make this worse, since reasoning tokens count as
  output.
- Agents already sending their own cache key, with a prompt start that never changes. That's what
  `cache_layout` would do for you.
- Providers with no discount for cached input.

## Caveats

These numbers come from one model on one benchmark. The prices come from the provider's
published list and haven't been checked against a real bill. Treat 21% as what's possible on a setup
like this one, then measure your own with `profile` and `eval`.
