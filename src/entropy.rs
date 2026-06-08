//! Information-theoretic measures for session analysis.
//!
//! These functions compute entropy, divergence, mutual information, and distortion
//! to quantify how much information is preserved or lost during compaction.

use crate::message::Message;

/// Compute Shannon entropy of a message distribution.
///
/// Treats each message as a symbol and computes the entropy of the frequency
/// distribution over roles. Higher entropy means more balanced role distribution.
pub fn shannon_entropy(messages: &[Message]) -> f64 {
    if messages.is_empty() {
        return 0.0;
    }

    let mut counts: [usize; 4] = [0; 4]; // system, user, assistant, tool
    for m in messages {
        let idx = match m.role {
            crate::message::Role::System => 0,
            crate::message::Role::User => 1,
            crate::message::Role::Assistant => 2,
            crate::message::Role::Tool => 3,
        };
        counts[idx] += 1;
    }

    let total = messages.len() as f64;
    let mut entropy = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Compute KL divergence between two token distributions.
///
/// `p` and `q` must be the same length. `q` values must be > 0 wherever `p` > 0.
/// Returns KL(p || q).
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .filter(|(&pi, _)| pi > 0.0)
        .map(|(&pi, &qi)| pi * (pi / qi.max(1e-10)).ln())
        .sum()
}

/// Compute mutual information between consecutive messages.
///
/// Measures how much knowing one message tells you about the next.
/// Higher values indicate more structured/predictable sessions.
pub fn mutual_information(messages: &[Message]) -> f64 {
    if messages.len() < 2 {
        return 0.0;
    }

    // Build bigram counts of role transitions
    let mut joint: [[usize; 4]; 4] = [[0; 4]; 4];
    for w in messages.windows(2) {
        let i = role_idx(w[0].role);
        let j = role_idx(w[1].role);
        joint[i][j] += 1;
    }

    let n = (messages.len() - 1) as f64;
    let mut marginal_x = [0usize; 4];
    let mut marginal_y = [0usize; 4];
    for i in 0..4 {
        for j in 0..4 {
            marginal_x[i] += joint[i][j];
            marginal_y[j] += joint[i][j];
        }
    }

    let mut mi = 0.0;
    for i in 0..4 {
        for j in 0..4 {
            if joint[i][j] > 0 && marginal_x[i] > 0 && marginal_y[j] > 0 {
                let p_xy = joint[i][j] as f64 / n;
                let p_x = marginal_x[i] as f64 / n;
                let p_y = marginal_y[j] as f64 / n;
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }
    }
    mi
}

/// Information density ratio: how much information per token after compaction.
///
/// A ratio > 1.0 means compaction increased information density (good).
pub fn information_density(before: &[Message], after: &[Message]) -> f64 {
    let before_tokens: usize = before.iter().map(|m| m.token_count).sum();
    let after_tokens: usize = after.iter().map(|m| m.token_count).sum();

    if before_tokens == 0 || after_tokens == 0 {
        return 0.0;
    }

    let before_entropy = shannon_entropy(before);
    let after_entropy = shannon_entropy(after);

    (after_entropy / after_tokens as f64) / (before_entropy / before_tokens as f64)
}

/// Compute distortion between original and compacted messages.
///
/// Distortion is the fraction of original information value that was lost.
/// 0.0 = no loss, 1.0 = everything lost.
pub fn distortion(original: &[Message], compacted: &[Message]) -> f64 {
    if original.is_empty() {
        return 0.0;
    }

    let original_value: f64 = original.iter().map(|m| m.information_value(original)).sum();
    if original_value == 0.0 {
        return 0.0;
    }

    let compacted_value: f64 = compacted.iter().map(|m| m.information_value(original)).sum();
    1.0 - (compacted_value / original_value).min(1.0)
}

/// Compute compression rate: compacted_size / original_size.
///
/// 0.0 = perfect compression (nothing kept), 1.0 = no compression.
pub fn rate(compressed: &[Message], original: &[Message]) -> f64 {
    let original_tokens: usize = original.iter().map(|m| m.token_count).sum();
    let compressed_tokens: usize = compressed.iter().map(|m| m.token_count).sum();

    if original_tokens == 0 {
        return 1.0;
    }
    compressed_tokens as f64 / original_tokens as f64
}

fn role_idx(role: crate::message::Role) -> usize {
    match role {
        crate::message::Role::System => 0,
        crate::message::Role::User => 1,
        crate::message::Role::Assistant => 2,
        crate::message::Role::Tool => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, Role};

    #[test]
    fn shannon_entropy_uniform() {
        let msgs = vec![
            Message::new(Role::System, "a", 1, 0.0),
            Message::new(Role::User, "b", 1, 1.0),
            Message::new(Role::Assistant, "c", 1, 2.0),
            Message::new(Role::Tool, "d", 1, 3.0),
        ];
        let h = shannon_entropy(&msgs);
        assert!((h - 2.0).abs() < 0.01, "uniform 4-role entropy should be ~2 bits, got {h}");
    }

    #[test]
    fn shannon_entropy_single_role() {
        let msgs = vec![
            Message::new(Role::User, "a", 1, 0.0),
            Message::new(Role::User, "b", 1, 1.0),
        ];
        let h = shannon_entropy(&msgs);
        assert!((h - 0.0).abs() < 0.01, "single role entropy should be 0, got {h}");
    }

    #[test]
    fn shannon_entropy_empty() {
        assert_eq!(shannon_entropy(&[]), 0.0);
    }

    #[test]
    fn kl_divergence_identical() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        assert!((kl_divergence(&p, &q)).abs() < 0.01);
    }

    #[test]
    fn kl_divergence_different() {
        let p = vec![1.0, 0.0, 0.0, 0.0];
        let q = vec![0.25, 0.25, 0.25, 0.25];
        assert!(kl_divergence(&p, &q) > 0.0);
    }

    #[test]
    fn mutual_information_ordered() {
        // Alternating user/assistant should have high MI
        let msgs: Vec<Message> = (0..20)
            .flat_map(|i| {
                vec![
                    Message::new(Role::User, format!("u{i}"), 1, (i * 2) as f64),
                    Message::new(Role::Assistant, format!("a{i}"), 1, (i * 2 + 1) as f64),
                ]
            })
            .collect();
        let mi = mutual_information(&msgs);
        assert!(mi > 0.0, "ordered session should have positive MI, got {mi}");
    }

    #[test]
    fn mutual_information_short() {
        let msgs = vec![Message::new(Role::User, "hi", 1, 0.0)];
        assert_eq!(mutual_information(&msgs), 0.0);
    }

    #[test]
    fn distortion_no_loss() {
        let msgs = vec![Message::new(Role::User, "hello", 5, 0.0)];
        assert!((distortion(&msgs, &msgs)).abs() < 0.01);
    }

    #[test]
    fn distortion_total_loss() {
        let original = vec![Message::new(Role::User, "important content", 10, 0.0)];
        let compacted: Vec<Message> = vec![];
        assert!((distortion(&original, &compacted) - 1.0).abs() < 0.01);
    }

    #[test]
    fn rate_no_compression() {
        let msgs = vec![Message::new(Role::User, "hello", 5, 0.0)];
        assert!((rate(&msgs, &msgs) - 1.0).abs() < 0.01);
    }

    #[test]
    fn rate_half_compression() {
        let original = vec![
            Message::new(Role::User, "hello", 5, 0.0),
            Message::new(Role::User, "world", 5, 1.0),
        ];
        let compressed = vec![Message::new(Role::User, "hello", 5, 0.0)];
        assert!((rate(&compressed, &original) - 0.5).abs() < 0.01);
    }

    #[test]
    fn information_density_improved() {
        // Remove low-value messages -> density should increase
        let before = vec![
            Message::new(Role::System, "instructions", 10, 0.0),
            Message::new(Role::Tool, "noise output noise output noise", 20, 1.0),
            Message::new(Role::User, "important query", 10, 2.0),
        ];
        let after = vec![
            Message::new(Role::System, "instructions", 10, 0.0),
            Message::new(Role::User, "important query", 10, 2.0),
        ];
        let density = information_density(&before, &after);
        // After removing low-value tool output, density of roles should be reasonable
        assert!(density > 0.0);
    }
}
