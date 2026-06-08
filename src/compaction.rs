//! Compaction strategies: selecting which messages to keep (γ) and which to discard (η).
//!
//! Every strategy obeys the conservation invariant:
//! `sum(token_count of kept) + sum(token_count of discarded) == sum(token_count of all)`.
//!
//! Strategies:
//! - **Greedy**: keep highest-value messages first.
//! - **Rate-distortion**: optimal Lagrangian (D + λ·R) selection.
//! - **Cluster**: group similar messages, keep one per cluster.
//! - **Layered**: keep all system, then best user, then compress assistant/tool.

use crate::budget::ConservationBudget;
use crate::message::{Message, semantic_distance};

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub kept: Vec<Message>,
    pub discarded: Vec<Message>,
    pub gamma_used: usize,
    pub eta_saved: usize,
    pub distortion: f64,
}

impl CompactionResult {
    /// Verify the conservation invariant.
    pub fn verify_conservation(&self, original_total: usize) -> bool {
        let kept_tokens: usize = self.kept.iter().map(|m| m.token_count).sum();
        let disc_tokens: usize = self.discarded.iter().map(|m| m.token_count).sum();
        kept_tokens + disc_tokens == original_total
    }
}

/// Greedy compaction: keep the highest information-value messages until budget exhausted.
///
/// Simple but effective baseline. Sorts by `information_value` descending and keeps
/// until γ budget is filled.
pub fn compact_greedy(messages: &[Message], budget: &ConservationBudget) -> CompactionResult {
    let gamma_limit = budget.gamma;

    // Compute information values
    let mut indexed: Vec<(usize, f64)> = messages
        .iter()
        .enumerate()
        .map(|(i, m)| (i, m.information_value(messages)))
        .collect();

    // Sort by value descending
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut kept_indices = std::collections::HashSet::new();
    let mut gamma_used = 0usize;

    for (idx, _val) in &indexed {
        let tokens = messages[*idx].token_count;
        if gamma_used + tokens <= gamma_limit {
            kept_indices.insert(*idx);
            gamma_used += tokens;
        }
    }

    let (kept, discarded) = partition(messages, &kept_indices);
    let eta_saved: usize = discarded.iter().map(|m| m.token_count).sum();

    CompactionResult {
        distortion: compute_distortion(&kept, messages),
        kept,
        discarded,
        gamma_used,
        eta_saved,
    }
}

/// Rate-distortion optimal compaction using Lagrangian formulation.
///
/// Minimizes D + λ·R where:
/// - D = distortion (information lost by discarding messages)
/// - R = rate (tokens kept, i.e., γ usage)
/// - λ = Lagrange multiplier controlling rate–distortion trade-off
///
/// Uses dynamic programming on the message sequence. Each message is either kept
/// (costs its token_count in rate, zero distortion) or discarded (zero rate, costs
/// its information_value in distortion).
pub fn compact_rate_distortion(
    messages: &[Message],
    budget: &ConservationBudget,
    lambda: f64,
) -> CompactionResult {
    if messages.is_empty() {
        return CompactionResult {
            kept: vec![],
            discarded: vec![],
            gamma_used: 0,
            eta_saved: 0,
            distortion: 0.0,
        };
    }

    let gamma_limit = budget.gamma;
    let n = messages.len();

    // Precompute information values
    let info_values: Vec<f64> = messages
        .iter()
        .map(|m| m.information_value(messages))
        .collect();

    // DP: dp[tokens_used] = minimum (D + λ·R) achievable with exactly tokens_used tokens kept
    // We track which messages were kept via backtracking
    let max_tokens = gamma_limit + 1;

    // dp[t] = min cost to use exactly t tokens after processing messages so far
    let mut dp = vec![f64::INFINITY; max_tokens];
    let mut choice: Vec<Vec<bool>> = vec![vec![false; max_tokens]; n];
    dp[0] = 0.0;

    for i in 0..n {
        let tokens = messages[i].token_count;
        let d = info_values[i]; // distortion if we discard this message
        let r = tokens as f64; // rate cost if we keep it
        let cost_keep = lambda * r; // keeping: zero distortion, lambda * rate
        let cost_discard = d; // discarding: distortion, zero rate

        // Must use a fresh array for this step
        let prev_dp = dp.clone();
        let mut new_dp = vec![f64::INFINITY; max_tokens];

        for t in 0..max_tokens {
            // Option 1: discard message i (keep same token usage)
            if prev_dp[t].is_finite() {
                let discard_cost = prev_dp[t] + cost_discard;
                new_dp[t] = new_dp[t].min(discard_cost);
            }
            // Option 2: keep message i (add its tokens)
            if t >= tokens && prev_dp[t - tokens].is_finite() {
                let keep_cost = prev_dp[t - tokens] + cost_keep;
                if keep_cost < new_dp[t] {
                    new_dp[t] = keep_cost;
                    choice[i][t] = true;
                }
            }
        }
        dp = new_dp;
    }

    // Find best token count
    let best_t = (0..max_tokens)
        .min_by(|&a, &b| dp[a].partial_cmp(&dp[b]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    // Backtrack to find which messages were kept
    let mut kept_indices = std::collections::HashSet::new();
    let mut remaining = best_t;
    for i in (0..n).rev() {
        if choice[i][remaining] {
            kept_indices.insert(i);
            remaining -= messages[i].token_count;
        }
    }

    let (kept, discarded) = partition(messages, &kept_indices);
    let gamma_used: usize = kept.iter().map(|m| m.token_count).sum();
    let eta_saved: usize = discarded.iter().map(|m| m.token_count).sum();

    CompactionResult {
        distortion: compute_distortion(&kept, messages),
        kept,
        discarded,
        gamma_used,
        eta_saved,
    }
}

/// Cluster-based compaction: group similar messages, keep one representative per cluster.
///
/// Uses a simple greedy agglomerative approach. Good for sessions with lots of
/// repetitive tool output or similar user queries.
pub fn compact_cluster(
    messages: &[Message],
    budget: &ConservationBudget,
    n_clusters: usize,
) -> CompactionResult {
    if messages.is_empty() || n_clusters == 0 {
        return CompactionResult {
            kept: vec![],
            discarded: vec![].into_iter().chain(messages.iter().cloned()).collect(),
            gamma_used: 0,
            eta_saved: messages.iter().map(|m| m.token_count).sum(),
            distortion: 1.0,
        };
    }

    let n_clusters = n_clusters.min(messages.len());

    // Simple greedy clustering: pick first message as cluster center, then pick the
    // message farthest from all existing centers.
    let mut centers: Vec<usize> = vec![0];
    while centers.len() < n_clusters {
        let mut best_idx = 0;
        let mut best_min_dist = f64::NEG_INFINITY;
        for i in 0..messages.len() {
            if centers.contains(&i) {
                continue;
            }
            let min_dist = centers
                .iter()
                .map(|&c| semantic_distance(&messages[i], &messages[c]))
                .fold(f64::INFINITY, f64::min);
            if min_dist > best_min_dist {
                best_min_dist = min_dist;
                best_idx = i;
            }
        }
        centers.push(best_idx);
    }

    // Assign each message to nearest center
    let mut assignments: Vec<usize> = vec![0; messages.len()];
    for i in 0..messages.len() {
        let (best_center, _) = centers
            .iter()
            .map(|&c| (c, semantic_distance(&messages[i], &messages[c])))
            .fold((centers[0], f64::INFINITY), |best, (c, d)| {
                if d < best.1 { (c, d) } else { best }
            });
        assignments[i] = best_center;
    }

    // Keep cluster representatives, sorted by information value within budget
    let mut representatives: Vec<usize> = centers;
    // Sort by information value
    representatives.sort_by(|&a, &b| {
        let va = messages[a].information_value(messages);
        let vb = messages[b].information_value(messages);
        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept_indices = std::collections::HashSet::new();
    let mut gamma_used = 0usize;
    for &idx in &representatives {
        let tokens = messages[idx].token_count;
        if gamma_used + tokens <= budget.gamma {
            kept_indices.insert(idx);
            gamma_used += tokens;
        }
    }

    let (kept, discarded) = partition(messages, &kept_indices);
    let eta_saved: usize = discarded.iter().map(|m| m.token_count).sum();

    CompactionResult {
        distortion: compute_distortion(&kept, messages),
        kept,
        discarded,
        gamma_used,
        eta_saved,
    }
}

/// Layered compaction: keep all system → best user → compress assistant/tool.
///
/// This is closest to what real agent systems (like Heddle) actually do:
/// 1. System messages are always kept (they're instructions).
/// 2. User messages are kept by recency/value.
/// 3. Assistant and tool messages are compressed first.
pub fn compact_layered(messages: &[Message], budget: &ConservationBudget) -> CompactionResult {
    let mut kept_indices = std::collections::HashSet::new();
    let mut gamma_used = 0usize;

    // Layer 1: Always keep system messages
    for (i, m) in messages.iter().enumerate() {
        if m.role == Role::System {
            if gamma_used + m.token_count <= budget.gamma {
                kept_indices.insert(i);
                gamma_used += m.token_count;
            }
        }
    }

    // Layer 2: User messages by information value
    let mut user_msgs: Vec<(usize, f64)> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == Role::User && !kept_indices.contains(&(*m as *const _ as usize)))
        .map(|(i, m)| (i, m.information_value(messages)))
        .collect();
    user_msgs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (idx, _) in &user_msgs {
        let tokens = messages[*idx].token_count;
        if gamma_used + tokens <= budget.gamma {
            kept_indices.insert(*idx);
            gamma_used += tokens;
        }
    }

    // Layer 3: Assistant messages by value
    let mut ast_msgs: Vec<(usize, f64)> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == Role::Assistant)
        .map(|(i, m)| (i, m.information_value(messages)))
        .collect();
    ast_msgs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (idx, _) in &ast_msgs {
        let tokens = messages[*idx].token_count;
        if gamma_used + tokens <= budget.gamma {
            kept_indices.insert(*idx);
            gamma_used += tokens;
        }
    }

    // Layer 4: Tool messages if space remains
    let mut tool_msgs: Vec<(usize, f64)> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == Role::Tool)
        .map(|(i, m)| (i, m.information_value(messages)))
        .collect();
    tool_msgs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (idx, _) in &tool_msgs {
        let tokens = messages[*idx].token_count;
        if gamma_used + tokens <= budget.gamma {
            kept_indices.insert(*idx);
            gamma_used += tokens;
        }
    }

    let (kept, discarded) = partition(messages, &kept_indices);
    let eta_saved: usize = discarded.iter().map(|m| m.token_count).sum();

    CompactionResult {
        distortion: compute_distortion(&kept, messages),
        kept,
        discarded,
        gamma_used,
        eta_saved,
    }
}

// Role re-export for layered
use crate::message::Role;

/// Partition messages into kept/discarded based on indices.
fn partition(
    messages: &[Message],
    kept_indices: &std::collections::HashSet<usize>,
) -> (Vec<Message>, Vec<Message>) {
    let kept: Vec<Message> = messages
        .iter()
        .enumerate()
        .filter(|(i, _)| kept_indices.contains(i))
        .map(|(_, m)| m.clone())
        .collect();
    let discarded: Vec<Message> = messages
        .iter()
        .enumerate()
        .filter(|(i, _)| !kept_indices.contains(i))
        .map(|(_, m)| m.clone())
        .collect();
    (kept, discarded)
}

/// Compute distortion as the sum of information values of discarded messages.
fn compute_distortion(kept: &[Message], original: &[Message]) -> f64 {
    let kept_contents: std::collections::HashSet<String> =
        kept.iter().map(|m| m.content.clone()).collect();
    original
        .iter()
        .filter(|m| !kept_contents.contains(&m.content))
        .map(|m| m.information_value(original))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session() -> Vec<Message> {
        vec![
            Message::new(Role::System, "You are a helpful assistant.", 10, 0.0),
            Message::new(Role::User, "What is Rust?", 8, 10.0),
            Message::new(Role::Assistant, "Rust is a systems programming language...", 30, 20.0),
            Message::new(Role::User, "How does ownership work?", 10, 30.0),
            Message::new(Role::Assistant, "Ownership is Rust's memory management...", 40, 40.0),
            Message::new(Role::Tool, "file_read: output...", 20, 45.0),
            Message::new(Role::User, "Show me an example", 8, 50.0),
            Message::new(Role::Assistant, "Here is an example of ownership...", 35, 60.0),
        ]
    }

    #[test]
    fn greedy_basic() {
        let msgs = make_session();
        let budget = ConservationBudget::new(100);
        let result = compact_greedy(&msgs, &budget);
        assert!(result.verify_conservation(msgs.iter().map(|m| m.token_count).sum()));
        assert!(result.gamma_used <= budget.gamma);
    }

    #[test]
    fn rate_distortion_basic() {
        let msgs = make_session();
        let budget = ConservationBudget::new(100);
        let result = compact_rate_distortion(&msgs, &budget, 1.0);
        assert!(result.verify_conservation(msgs.iter().map(|m| m.token_count).sum()));
        assert!(result.gamma_used <= budget.gamma);
    }

    #[test]
    fn cluster_basic() {
        let msgs = make_session();
        let budget = ConservationBudget::new(100);
        let result = compact_cluster(&msgs, &budget, 3);
        assert!(result.verify_conservation(msgs.iter().map(|m| m.token_count).sum()));
    }

    #[test]
    fn layered_keeps_system_first() {
        let msgs = make_session();
        let budget = ConservationBudget::new(100);
        let result = compact_layered(&msgs, &budget);
        assert!(result.verify_conservation(msgs.iter().map(|m| m.token_count).sum()));
        // System message should be kept
        assert!(result.kept.iter().any(|m| m.role == Role::System));
    }

    #[test]
    fn empty_messages() {
        let msgs: Vec<Message> = vec![];
        let budget = ConservationBudget::new(100);
        for (name, result) in [
            ("greedy", compact_greedy(&msgs, &budget)),
            ("rd", compact_rate_distortion(&msgs, &budget, 1.0)),
            ("cluster", compact_cluster(&msgs, &budget, 3)),
            ("layered", compact_layered(&msgs, &budget)),
        ] {
            assert!(result.kept.is_empty(), "{name} kept messages on empty input");
            assert!(result.discarded.is_empty(), "{name} discarded on empty input");
        }
    }

    #[test]
    fn single_message() {
        let msgs = vec![Message::new(Role::User, "hello", 5, 0.0)];
        let budget = ConservationBudget::new(100);
        let result = compact_greedy(&msgs, &budget);
        assert_eq!(result.kept.len(), 1);
        assert!(result.verify_conservation(5));
    }

    #[test]
    fn all_same_role() {
        let msgs: Vec<Message> = (0..10)
            .map(|i| Message::new(Role::Assistant, format!("message {i}"), 10, i as f64))
            .collect();
        let budget = ConservationBudget::new(60);
        let result = compact_greedy(&msgs, &budget);
        assert!(result.verify_conservation(100));
        assert!(result.kept.len() < 10);
    }

    #[test]
    fn budget_too_small() {
        let msgs = vec![Message::new(Role::User, "hello world this is a test", 100, 0.0)];
        let budget = ConservationBudget::with_ratio(100, 0.1); // gamma = 10
        let result = compact_greedy(&msgs, &budget);
        assert!(result.kept.is_empty() || result.gamma_used <= budget.gamma);
    }

    #[test]
    fn conservation_invariant_all_strategies() {
        let msgs = make_session();
        let total_tokens: usize = msgs.iter().map(|m| m.token_count).sum();
        let budget = ConservationBudget::new(80);

        for result in [
            compact_greedy(&msgs, &budget),
            compact_rate_distortion(&msgs, &budget, 1.0),
            compact_cluster(&msgs, &budget, 3),
            compact_layered(&msgs, &budget),
        ] {
            assert!(
                result.verify_conservation(total_tokens),
                "Conservation invariant violated: kept {} + discarded {} != {}",
                result.kept.iter().map(|m| m.token_count).sum::<usize>(),
                result.discarded.iter().map(|m| m.token_count).sum::<usize>(),
                total_tokens
            );
        }
    }

    #[test]
    fn long_session() {
        let mut msgs = vec![Message::new(Role::System, "Instructions", 10, 0.0)];
        for i in 0..1000 {
            let role = match i % 4 {
                0 => Role::User,
                1 => Role::Assistant,
                2 => Role::Tool,
                _ => Role::User,
            };
            msgs.push(Message::new(role, format!("message content {i}"), 10, (i + 1) as f64));
        }
        let total: usize = msgs.iter().map(|m| m.token_count).sum();
        let budget = ConservationBudget::new(2000);
        let result = compact_greedy(&msgs, &budget);
        assert!(result.verify_conservation(total));
        assert!(result.gamma_used <= budget.gamma);
    }
}
