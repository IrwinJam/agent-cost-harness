# Results so far

These are early numbers from real runs, not proof. The biggest run so far is 5 tasks with 1 trial each,
which is far too small to say anything for sure about savings. They're here to show the harness working
on a real model, what it found, and what a full test would cost.

**Setup.** A hosted reasoning model behind an OpenAI-compatible API, reasoning effort set to medium. The
tasks come from a public customer-service agent benchmark: a retail domain with 40 test tasks, a
simulated customer played by the same model, and automatic grading (plus a model acting as judge on 11
of the 40 tasks). The benchmark's own agent runs unchanged, with its calls routed through the harness.
Costs use the provider's published prices, which I haven't checked line by line against the billing
page yet.

Total spent on everything below: about **$0.90**.

## 1. The basics work

A handful of direct calls ($0.04 total):

- Both the Responses and Chat Completions formats work.
- **Prefix caching is real and large.** Sending the same 8.4k-token prompt start twice, the second call
  got 8,305 of 8,430 input tokens from the cache (98.5%).
- Tool calls work with `tool_choice: "auto"`, and a short `user` message in the middle of a
  conversation (what `cache_layout` adds when a line changes) is accepted.
- **Reasoning tokens add up fast.** Even at the lowest effort, a one-word answer used up to 318
  reasoning tokens.

## 2. One full task

One benchmark task, start to finish, through the harness. It passed.

| Part | Calls | Cost |
|---|---:|---:|
| Agent | 17 | $0.0725 |
| Simulated customer | 8 | $0.0232 |
| Judge | 1 | $0.0096 |
| **Total** | | **$0.105** (45 seconds) |

What the profiler showed for the agent:

- 81% of its input came from the cache with no policies at all. The provider caches automatically.
- Output was the biggest cost: 36% of agent spend, 22% from reasoning alone.
- Three calls weren't cached even though the prompt start hadn't changed. Those cost about 15% of the
  agent's spend.
- Tool outputs were small. The context never went past about 12k tokens.

## 3. Pilot: baseline vs cache_layout

5 tasks, 1 trial each, with the two setups taking turns going first ($0.75).

| | Baseline | cache_layout |
|---|---:|---:|
| Tasks passed | 4 of 5 | 3 of 5 |
| Agent cost per trial | $0.050 | $0.043 (**-15%**) |
| Total per trial, incl. customer and judge | $0.080 | $0.070 |
| Input from cache | 90% | 93% |
| Reasoning tokens | 17,957 | 14,761 |

- `cache_layout` shared one cache key across all 5 tasks, so every task after the first started with
  the 5.1k-token prompt start already cached. The benchmark's system prompt never changes, so it
  correctly moved nothing out of it.
- **Most of the -15% probably isn't caching.** Cache hit rates only differ by a few points. More of the
  gap comes from some conversations just going differently, like one task finishing in 12 calls
  instead of 16.
- **Pass rates swing a lot at this size.** One trial per task can't separate a real effect from noise.
  The task `cache_layout` failed and baseline passed could be either.

## 4. What a full test would cost

Based on about $0.08 per trial including the simulated customer and judge, and about 52 seconds per
trial run one at a time:

| Test | Trials | Estimated cost | Estimated time |
|---|---:|---:|---:|
| Baseline vs `cache_layout`, 40 tasks, 3 trials each | 240 | about $19 | about 3.5 hours |
| All four setups, 40 tasks, 3 trials each | 480 | about $38 | about 7 hours |

Plan for some extra on top of that, since reasoning varies a lot from task to task. Both would stay well
under a $30 or $120 spending cap.

The full test hasn't been run yet. On this benchmark, tool outputs are small, so `output_handles` and
`masking` would barely do anything. Those two need a benchmark with long tool outputs to show what
they can do.
