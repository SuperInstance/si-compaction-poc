//! Experimental validation: compare compaction strategies and measure performance.
//!
//! Runs all four strategies on the same input and produces a comparison table
//! showing distortion, rate, entropy preservation, and runtime.

use std::time::Instant;

use crate::budget::ConservationBudget;
use crate::compaction::{
    compact_cluster, compact_greedy, compact_layered, compact_rate_distortion,
};
use crate::entropy::{distortion, rate, shannon_entropy};
use crate::message::{Message, Role};

/// Compaction strategy identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Greedy,
    RateDistortion,
    Cluster,
    Layered,
}

impl std::fmt::Display for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Strategy::Greedy => write!(f, "Greedy"),
            Strategy::RateDistortion => write!(f, "RateDistortion"),
            Strategy::Cluster => write!(f, "Cluster"),
            Strategy::Layered => write!(f, "Layered"),
        }
    }
}

/// Result of running a single experiment.
#[derive(Debug, Clone)]
pub struct ExperimentResult {
    pub strategy: Strategy,
    pub gamma_used: usize,
    pub eta_saved: usize,
    pub distortion: f64,
    pub rate: f64,
    pub entropy_before: f64,
    pub entropy_after: f64,
    pub runtime_ms: u128,
}

impl ExperimentResult {
    /// Formatted summary row for this result.
    pub fn summary(&self) -> String {
        format!(
            "| {:<15} | {:>10} | {:>10} | {:>10.4} | {:>6.2}% | {:>8.3} | {:>8.3} | {:>8}ms |",
            self.strategy.to_string(),
            self.gamma_used,
            self.eta_saved,
            self.distortion,
            self.rate * 100.0,
            self.entropy_before,
            self.entropy_after,
            self.runtime_ms,
        )
    }
}

/// A configured experiment.
#[derive(Debug, Clone)]
pub struct Experiment {
    pub name: String,
    pub budget: ConservationBudget,
    pub strategy: Strategy,
}

impl Experiment {
    pub fn new(name: impl Into<String>, budget: ConservationBudget, strategy: Strategy) -> Self {
        Self {
            name: name.into(),
            budget,
            strategy,
        }
    }

    /// Run this experiment on a set of messages.
    pub fn run(&self, messages: &[Message]) -> ExperimentResult {
        let entropy_before = shannon_entropy(messages);

        let start = Instant::now();
        let result = match self.strategy {
            Strategy::Greedy => compact_greedy(messages, &self.budget),
            Strategy::RateDistortion => compact_rate_distortion(messages, &self.budget, 1.0),
            Strategy::Cluster => compact_cluster(messages, &self.budget, 3),
            Strategy::Layered => compact_layered(messages, &self.budget),
        };
        let runtime_ms = start.elapsed().as_millis();

        let entropy_after = shannon_entropy(&result.kept);
        let r = rate(&result.kept, messages);
        let d = distortion(messages, &result.kept);

        ExperimentResult {
            strategy: self.strategy,
            gamma_used: result.gamma_used,
            eta_saved: result.eta_saved,
            distortion: d,
            rate: r,
            entropy_before,
            entropy_after,
            runtime_ms,
        }
    }
}

/// Run all four strategies on the same messages with the same budget and compare.
pub fn compare(messages: &[Message], budget: &ConservationBudget) -> Vec<ExperimentResult> {
    let strategies = [
        Strategy::Greedy,
        Strategy::RateDistortion,
        Strategy::Cluster,
        Strategy::Layered,
    ];

    strategies
        .iter()
        .map(|&s| {
            let exp = Experiment::new(format!("{s}"), budget.clone(), s);
            exp.run(messages)
        })
        .collect()
}

/// Print a comparison table header + rows.
pub fn comparison_table(results: &[ExperimentResult]) -> String {
    let mut out = String::new();
    out.push_str("# Compaction Strategy Comparison\n\n");
    out.push_str("| Strategy        | γ Used     | η Saved    | Distortion  | Rate   | H(before) | H(after)  | Runtime   |\n");
    out.push_str("|-----------------|------------|------------|-------------|--------|-----------|-----------|-----------|\n");
    for r in results {
        out.push_str(&r.summary());
        out.push('\n');
    }
    out
}

/// Generate a synthetic session for testing.
pub fn synthetic_session(n_turns: usize) -> Vec<Message> {
    let mut messages = vec![Message::new(
        Role::System,
        "You are a helpful coding assistant. Follow instructions carefully.",
        20,
        0.0,
    )];

    for i in 0..n_turns {
        let t = ((i + 1) * 10) as f64;
        messages.push(Message::new(
            Role::User,
            format!("Can you explain concept {i} in detail?"),
            15,
            t,
        ));
        messages.push(Message::new(
            Role::Assistant,
            format!("Concept {i} is about... Let me explain with examples and code. [detailed explanation #{i}]"),
            40,
            t + 1.0,
        ));
        if i % 3 == 0 {
            messages.push(Message::new(
                Role::Tool,
                format!("tool_output_{i}: execution result with some data"),
                20,
                t + 2.0,
            ));
        }
    }

    messages
}

/// Generate a session heavy on tool output (worst case for compression).
pub fn tool_heavy_session(n_tools: usize) -> Vec<Message> {
    let mut messages = vec![Message::new(Role::System, "Run these tools.", 10, 0.0)];
    for i in 0..n_tools {
        let t = ((i + 1) * 5) as f64;
        messages.push(Message::new(
            Role::User,
            format!("run command {i}"),
            5,
            t,
        ));
        messages.push(Message::new(
            Role::Tool,
            format!("command_{i} output: line1\nline2\nline3\nline4\nline5"),
            25,
            t + 1.0,
        ));
    }
    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_all_strategies() {
        let msgs = synthetic_session(10);
        let budget = ConservationBudget::new(300);
        let results = compare(&msgs, &budget);
        assert_eq!(results.len(), 4);
        // All should use <= budget
        for r in &results {
            assert!(r.gamma_used <= budget.gamma);
        }
    }

    #[test]
    fn rate_distortion_runs_and_conserves() {
        let msgs = synthetic_session(3);
        let budget = ConservationBudget::with_ratio(500, 0.7);
        let total_tokens: usize = msgs.iter().map(|m| m.token_count).sum();

        let result = compact_rate_distortion(&msgs, &budget, 1.0);
        assert!(
            result.verify_conservation(total_tokens),
            "Conservation invariant must hold"
        );
        assert!(result.gamma_used <= budget.gamma);

        // With very small lambda (favor keeping), should keep more
        let result_keep = compact_rate_distortion(&msgs, &budget, 0.01);
        assert!(
            result_keep.verify_conservation(total_tokens),
            "Conservation invariant must hold"
        );
    }

    #[test]
    fn experiment_run() {
        let msgs = synthetic_session(3);
        let budget = ConservationBudget::new(100);
        let exp = Experiment::new("test", budget, Strategy::Greedy);
        let result = exp.run(&msgs);
        assert_eq!(result.strategy, Strategy::Greedy);
        assert!(result.runtime_ms < 1000);
    }

    #[test]
    fn comparison_table_formats() {
        let msgs = synthetic_session(5);
        let budget = ConservationBudget::new(200);
        let results = compare(&msgs, &budget);
        let table = comparison_table(&results);
        assert!(table.contains("Greedy"));
        assert!(table.contains("RateDistortion"));
        assert!(table.contains("Layered"));
    }

    #[test]
    fn synthetic_session_sizes() {
        let msgs = synthetic_session(5);
        assert!(msgs.len() >= 11); // 1 system + 5*(user+assistant) + ~2 tool calls
        let tool_msgs = tool_heavy_session(5);
        assert!(tool_msgs.len() >= 11);
    }

    #[test]
    fn conservation_holds_across_experiments() {
        let msgs = synthetic_session(8);
        let total: usize = msgs.iter().map(|m| m.token_count).sum();
        let budget = ConservationBudget::new(300);

        for s in [Strategy::Greedy, Strategy::RateDistortion, Strategy::Cluster, Strategy::Layered]
        {
            let exp = Experiment::new(format!("{s}"), budget.clone(), s);
            let result = exp.run(&msgs);
            assert!(
                result.gamma_used + result.eta_saved == total,
                "Conservation violated for {s}: {} + {} != {}",
                result.gamma_used,
                result.eta_saved,
                total
            );
        }
    }
}
