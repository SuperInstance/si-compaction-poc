//! Messages with information-value estimation.
//!
//! Each message carries an estimated "information value" — a heuristic measure of
//! how semantically important it is relative to the surrounding context. This value
//! drives compaction decisions.

use std::fmt;

/// Message role in a conversation session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    /// Weight for information-value calculation. System messages are most important.
    pub fn weight(&self) -> f64 {
        match self {
            Role::System => 4.0,
            Role::User => 3.0,
            Role::Assistant => 2.0,
            Role::Tool => 1.0,
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

/// A single message in a session.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub token_count: usize,
    /// Unix timestamp (seconds)
    pub timestamp: f64,
}

impl Message {
    /// Create a new message.
    pub fn new(role: Role, content: impl Into<String>, token_count: usize, timestamp: f64) -> Self {
        Self {
            role,
            content: content.into(),
            token_count,
            timestamp,
        }
    }

    /// Convenience: create with auto-estimated token count (rough: ~4 chars per token).
    pub fn auto(role: Role, content: impl Into<String>, timestamp: f64) -> Self {
        let s: String = content.into();
        let tokens = (s.len() as f64 / 4.0).ceil() as usize;
        Self::new(role, s, tokens.max(1), timestamp)
    }

    /// Estimate the information value of this message within a context.
    ///
    /// Components:
    /// - **Recency**: exponential decay based on timestamp distance from latest message.
    /// - **Uniqueness**: TF-IDF-like measure — how much unique vocabulary this message
    ///   contributes vs the rest of the context.
    /// - **Role weight**: system > user > assistant > tool.
    /// - **Length bonus**: longer messages carry more information (up to a point).
    pub fn information_value(&self, context: &[Message]) -> f64 {
        let role_w = self.role.weight();

        // Recency: exponential decay, half-life of 100 time units
        let max_ts = context.iter().map(|m| m.timestamp).fold(self.timestamp, f64::max);
        let recency = 0.5_f64.powf((max_ts - self.timestamp) / 100.0);

        // Uniqueness: fraction of words in this message not found in other messages
        let my_words: std::collections::HashSet<&str> =
            self.content.split_whitespace().collect();
        let other_words: std::collections::HashSet<&str> = context
            .iter()
            .filter(|m| !std::ptr::eq(*m, self))
            .flat_map(|m| m.content.split_whitespace())
            .collect();
        let unique_count = my_words.iter().filter(|w| !other_words.contains(**w)).count();
        let uniqueness = if my_words.is_empty() {
            0.5
        } else {
            unique_count as f64 / my_words.len() as f64
        };

        // Length factor: diminishing returns
        let length_factor = (1.0 + self.token_count as f64).ln();

        role_w * recency * (0.5 + 0.5 * uniqueness) * length_factor
    }

    /// Get the set of words in this message.
    pub fn word_set(&self) -> std::collections::HashSet<&str> {
        self.content.split_whitespace().collect()
    }
}

/// Compute semantic distance between two messages.
///
/// Uses word-overlap (Jaccard distance) combined with role and recency differences.
/// Returns 0.0 (identical) to 1.0 (completely different).
pub fn semantic_distance(a: &Message, b: &Message) -> f64 {
    let wa = a.word_set();
    let wb = b.word_set();

    // Jaccard distance
    let intersection = wa.intersection(&wb).count() as f64;
    let union = wa.union(&wb).count() as f64;
    let jaccard_dist = if union == 0.0 { 0.0 } else { 1.0 - intersection / union };

    // Role distance
    let role_dist = if a.role == b.role { 0.0 } else { 0.3 };

    // Recency distance (normalized)
    let ts_diff = (a.timestamp - b.timestamp).abs();
    let recency_dist = 1.0 - (-ts_diff / 1000.0).exp();

    // Weighted combination
    0.6 * jaccard_dist + 0.2 * role_dist + 0.2 * recency_dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_weights() {
        assert!(Role::System.weight() > Role::User.weight());
        assert!(Role::User.weight() > Role::Assistant.weight());
        assert!(Role::Assistant.weight() > Role::Tool.weight());
    }

    #[test]
    fn message_auto_estimates_tokens() {
        let m = Message::auto(Role::User, "Hello world", 0.0);
        assert!(m.token_count >= 1);
    }

    #[test]
    fn information_value_system_highest() {
        let msgs = vec![
            Message::new(Role::System, "instructions", 10, 100.0),
            Message::new(Role::User, "hello", 5, 100.0),
            Message::new(Role::Assistant, "hi there", 5, 100.0),
            Message::new(Role::Tool, "output", 5, 100.0),
        ];
        let sys_val = msgs[0].information_value(&msgs);
        let usr_val = msgs[1].information_value(&msgs);
        let ast_val = msgs[2].information_value(&msgs);
        let tool_val = msgs[3].information_value(&msgs);
        assert!(sys_val > usr_val);
        assert!(usr_val > ast_val);
        assert!(ast_val > tool_val);
    }

    #[test]
    fn information_value_recency_decay() {
        let msgs = vec![
            Message::new(Role::User, "old message", 10, 0.0),
            Message::new(Role::User, "recent message", 10, 200.0),
        ];
        let old_val = msgs[0].information_value(&msgs);
        let new_val = msgs[1].information_value(&msgs);
        assert!(new_val > old_val);
    }

    #[test]
    fn information_value_uniqueness() {
        let msgs = vec![
            Message::new(Role::User, "alpha beta gamma", 10, 100.0),
            Message::new(Role::User, "alpha beta gamma", 10, 100.0),
            Message::new(Role::User, "delta epsilon zeta", 10, 100.0),
        ];
        let dup_val = msgs[0].information_value(&msgs);
        let unique_val = msgs[2].information_value(&msgs);
        assert!(unique_val > dup_val);
    }

    #[test]
    fn semantic_distance_identical() {
        let a = Message::new(Role::User, "hello world", 5, 100.0);
        let b = Message::new(Role::User, "hello world", 5, 100.0);
        assert!((semantic_distance(&a, &b)).abs() < 0.01);
    }

    #[test]
    fn semantic_distance_completely_different() {
        let a = Message::new(Role::System, "alpha beta", 5, 0.0);
        let b = Message::new(Role::Tool, "gamma delta", 5, 10000.0);
        let d = semantic_distance(&a, &b);
        assert!(d > 0.5);
    }
}
