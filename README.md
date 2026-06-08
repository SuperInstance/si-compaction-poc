# si-compaction-poc

**Proof of concept: conservation-law-optimal context compaction — Heddle's auto-compact formalized as γ/η budget split.**

## The Thesis

When an AI agent's context window fills up, the system must decide what to keep and what to discard. This is called **auto-compaction** (or auto-compact) in systems like Heddle. The system applies heuristic compression when context exceeds a threshold.

**This is not just a practical engineering problem — it is a fundamental information-theoretic operation governed by conservation laws.**

### The Conservation Law

Every token in a session must be either:

- **γ (gamma)** — kept in the compacted context (information preserved)
- **η (eta)** — discarded from the context (information lost)

The conservation invariant:

```
γ + η = total_budget
```

No token disappears. Every token is either kept or discarded. This is exactly like energy conservation in physics: energy is neither created nor destroyed, only transformed.

### Why This Matters

Understanding auto-compaction as conservation-law entropy management gives us:

1. **Formal optimality criteria** — We can prove a compaction strategy is optimal rather than guessing.
2. **Rate-distortion theory** — The mathematical framework for lossy compression applies directly.
3. **Measurable trade-offs** — Every compaction decision has a quantifiable distortion cost.
4. **Composable strategies** — Different compaction approaches can be compared on the same scale.

## Rate-Distortion Theory for Sessions

The compaction problem maps directly to **rate-distortion theory** from information theory:

| Session Concept | Rate-Distortion Concept |
|---|---|
| Full session (N tokens) | Source X |
| Compacted context (γ tokens) | Rate R |
| Information lost (η tokens) | Distortion D |
| Budget constraint | Channel capacity C |

The optimal compaction minimizes the Lagrangian:

```
minimize: D + λ·R
```

Where:
- **D** = distortion (total information value of discarded messages)
- **R** = rate (tokens kept = γ usage)
- **λ** = Lagrange multiplier controlling the rate-distortion trade-off

A small λ means "keep more, worry less about space." A large λ means "aggressively compress, space is precious."

## The γ/η Framing

### Why γ/η Is the Right Framing

1. **Conservation is explicit** — Every compaction operation maintains γ + η = total. You can audit this.
2. **Budget is first-class** — The γ budget is a resource to be managed, not an afterthought.
3. **Information loss is tracked** — η isn't just "deleted tokens" — it's the entropy budget, the cost of compaction.
4. **It connects to thermodynamics** — Just as entropy in a closed system can only increase, η in a session can only accumulate. You can never recover information once discarded.

### The Budget Model

```rust
use si_compaction_poc::ConservationBudget;

// 60/40 split: 60% kept, 40% available for discard
let budget = ConservationBudget::new(10000);

// Custom split
let budget = ConservationBudget::with_ratio(10000, 0.8);

// Spend from γ
budget.spend(500)?;

// Replenish after compaction (frees γ, accumulates η)
budget.replenish(200);
```

## Compaction Strategies

This crate implements four compaction strategies, each making different trade-offs:

### 1. Greedy

Keep the highest information-value messages until the budget is exhausted.

**Pros:** Simple, fast, predictable.
**Cons:** No global optimization. May discard a cluster of medium-value messages in favor of one high-value one.

```rust
use si_compaction_poc::{compact_greedy, ConservationBudget, Message, Role};

let messages = vec![/* ... */];
let budget = ConservationBudget::new(1000);
let result = compact_greedy(&messages, &budget);
```

### 2. Rate-Distortion Optimal

Minimize D + λ·R using dynamic programming over the message sequence.

**Pros:** Mathematically optimal for the given λ. Provable guarantees.
**Cons:** O(n × γ) time complexity. Can be slow for very large sessions.

```rust
use si_compaction_poc::compact_rate_distortion;

let result = compact_rate_distortion(&messages, &budget, 1.0);
//                                              lambda ↑ controls rate-distortion trade-off
```

### 3. Cluster-Based

Group similar messages into clusters, keep one representative per cluster.

**Pros:** Good for sessions with repetitive tool output or similar queries.
**Cons:** Loses temporal ordering. Cluster quality depends on distance metric.

```rust
use si_compaction_poc::compact_cluster;

let result = compact_cluster(&messages, &budget, 5);
//                                               ↑ number of clusters
```

### 4. Layered (Heddle-style)

Layered priority: keep all system → best user → compress assistant → drop tool.

**Pros:** Closest to what real agent systems actually do. Respects role hierarchy.
**Cons:** May not be globally optimal. Tool messages (which can contain important results) are deprioritized.

```rust
use si_compaction_poc::compact_layered;

let result = compact_layered(&messages, &budget);
```

## Information-Theoretic Measures

The crate provides several measures for analyzing compaction quality:

### Shannon Entropy

Measures the "spread" of messages across roles. Higher entropy = more balanced session.

```rust
use si_compaction_poc::shannon_entropy;

let h = shannon_entropy(&messages);
// H ≈ 2.0 bits for a session with equal roles
// H ≈ 0.0 bits for a session with only one role
```

### KL Divergence

Measures how different two token distributions are. Useful for comparing before/after compaction.

```rust
use si_compaction_poc::kl_divergence;

let divergence = kl_divergence(&original_dist, &compacted_dist);
```

### Mutual Information

Measures how much knowing one message tells you about the next. High MI = structured session.

```rust
use si_compaction_poc::mutual_information;

let mi = mutual_information(&messages);
// High for alternating user/assistant conversations
// Low for random message orderings
```

### Distortion

The fraction of information value lost during compaction.

```rust
use si_compaction_poc::distortion;

let d = distortion(&original, &compacted);
// 0.0 = no information lost
// 1.0 = everything lost
```

### Rate

The compression ratio: compacted size / original size.

```rust
use si_compaction_poc::rate;

let r = rate(&compacted, &original);
// 0.5 = kept half the tokens
// 1.0 = no compression
```

## Experimental Comparison

The crate includes an experiment runner that compares all four strategies:

```rust
use si_compaction_poc::{compare, comparison_table, ConservationBudget, synthetic_session};

let messages = synthetic_session(10);
let budget = ConservationBudget::new(500);
let results = compare(&messages, &budget);
let table = comparison_table(&results);
println!("{}", table);
```

Example output:

```
| Strategy        | γ Used     | η Saved    | Distortion  | Rate   | H(before) | H(after)  | Runtime   |
|-----------------|------------|------------|-------------|--------|-----------|-----------|-----------|
| Greedy          |        280 |        170 |      0.2341 | 62.17% |     1.846 |     1.500 |        0ms |
| RateDistortion  |        240 |        210 |      0.1892 | 53.33% |     1.846 |     1.585 |        1ms |
| Cluster         |        190 |        260 |      0.4123 | 42.22% |     1.846 |     1.292 |        0ms |
| Layered         |        300 |        150 |      0.2678 | 66.67% |     1.846 |     1.756 |        0ms |
```

### Key Observations

1. **Rate-distortion achieves lowest distortion** for a given rate — by construction.
2. **Layered is fast and role-respecting** but may not be globally optimal.
3. **Greedy is a strong baseline** — often close to optimal in practice.
4. **Cluster trades temporal coherence** for representation efficiency.

## How to Integrate with Heddle

Heddle's auto-compact can be upgraded to conservation-law-optimal by:

### Step 1: Track the Budget

Replace ad-hoc token counting with `ConservationBudget`:

```rust
let budget = ConservationBudget::new(model.context_window);
// On every message:
budget.spend(message.token_count)?;
// On compaction:
budget.replenish(compacted_tokens);
```

### Step 2: Choose a Strategy

For production use, we recommend **layered** (closest to current behavior) or **greedy** (simple, fast):

```rust
let result = compact_layered(&session.messages, &budget);
session.messages = result.kept;
```

### Step 3: Measure Quality

After compaction, measure distortion to ensure quality:

```rust
let d = distortion(&original, &compacted);
if d > 0.5 {
    log::warn!("High distortion ({:.2}) during compaction — may lose important context", d);
}
```

### Step 4: Adapt γ/η Ratio

Monitor distortion over time. If sessions consistently have high distortion after compaction:

- Increase γ ratio (keep more context)
- Switch to a more optimal strategy (rate-distortion)
- Consider a larger context window

## API Reference

### `ConservationBudget`

```rust
pub struct ConservationBudget {
    pub gamma: usize,   // γ: tokens allocated for keeping
    pub eta: usize,     // η: tokens that have been discarded
    pub total: usize,   // total token budget
}
```

| Method | Description |
|---|---|
| `new(total)` | Create with 60/40 γ/η split |
| `with_ratio(total, ratio)` | Create with custom γ ratio |
| `can_spend(tokens)` | Check if tokens can be spent from γ |
| `spend(tokens)` | Spend tokens from γ |
| `replenish(tokens)` | Recover γ tokens after compaction |
| `utilization()` | Current γ utilization (0.0–1.0) |
| `budget_remaining()` | Remaining γ tokens |
| `is_exhausted()` | Whether γ is fully used |

### `Message`

```rust
pub struct Message {
    pub role: Role,
    pub content: String,
    pub token_count: usize,
    pub timestamp: f64,
}

pub enum Role { System, User, Assistant, Tool }
```

| Method | Description |
|---|---|
| `new(role, content, tokens, ts)` | Create a message |
| `auto(role, content, ts)` | Auto-estimate token count |
| `information_value(ctx)` | Estimate semantic importance |
| `word_set()` | Set of unique words |

### `semantic_distance(a, b)`

Jaccard + role + recency distance between two messages. Returns 0.0 (identical) to 1.0 (completely different).

### Compaction Functions

| Function | Description |
|---|---|
| `compact_greedy(msgs, budget)` | Keep highest-value messages |
| `compact_rate_distortion(msgs, budget, λ)` | Lagrangian-optimal selection |
| `compact_cluster(msgs, budget, k)` | Cluster-based selection |
| `compact_layered(msgs, budget)` | Role-priority selection |

All return `CompactionResult`:

```rust
pub struct CompactionResult {
    pub kept: Vec<Message>,
    pub discarded: Vec<Message>,
    pub gamma_used: usize,
    pub eta_saved: usize,
    pub distortion: f64,
}
```

### Entropy Functions

| Function | Signature | Description |
|---|---|---|
| `shannon_entropy` | `(&[Message]) -> f64` | Role distribution entropy |
| `kl_divergence` | `(&[f64], &[f64]) -> f64` | KL(p ∥ q) |
| `mutual_information` | `(&[Message]) -> f64` | Consecutive-message MI |
| `information_density` | `(&[Message], &[Message]) -> f64` | Density ratio before/after |
| `distortion` | `(&[Message], &[Message]) -> f64` | Information value lost |
| `rate` | `(&[Message], &[Message]) -> f64` | Compression ratio |

### Experiment Functions

| Function | Description |
|---|---|
| `Experiment::new(name, budget, strategy)` | Create an experiment |
| `experiment.run(messages)` | Run and measure |
| `compare(messages, budget)` | Run all strategies |
| `comparison_table(results)` | Format as Markdown table |
| `synthetic_session(turns)` | Generate test session |
| `tool_heavy_session(tools)` | Generate tool-heavy session |

## Running the Tests

```bash
# All 46 tests
cargo test

# Just budget tests
cargo test budget

# Just compaction tests
cargo test compaction

# Just entropy tests
cargo test entropy

# Just experiment tests
cargo test experiment
```

## The Deeper Connection: Sessions as Thermodynamic Systems

This isn't just an analogy — the math is structurally identical:

| Thermodynamics | Session Compaction |
|---|---|
| Total energy E | Total tokens N |
| Useful work W | Kept tokens γ |
| Waste heat Q | Discarded tokens η |
| First law: E = W + Q | Conservation: N = γ + η |
| Entropy S | Information loss η |
| Carnot efficiency η = 1 - T_cold/T_hot | Compaction efficiency = 1 - distortion |
| Second law: S only increases | η only accumulates |

The **second law of thermodynamics** maps directly: in any compaction step, you can never recover the full information of the discarded messages. η (entropy) only accumulates. Each compaction loses a little more.

The **Carnot limit** maps too: there's a theoretical maximum efficiency for compaction, determined by the rate-distortion function. You can't do better than optimal, and any practical strategy approaches this limit.

## Future Directions

1. **Learned information value** — Replace heuristics with a learned model of message importance.
2. **Adaptive λ** — Dynamically adjust the rate-distortion trade-off based on session characteristics.
3. **Hierarchical compaction** — Apply different strategies to different sections of the session.
4. **Streaming compaction** — Compact incrementally as messages arrive, rather than in bulk.
5. **Cross-session learning** — Learn which types of messages are most valuable across sessions.

## License

MIT

## Author

SuperInstance — formalizing the beautiful connection between session management and physics.
