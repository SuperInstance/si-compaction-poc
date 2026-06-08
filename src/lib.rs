//! si-compaction-poc: Conservation-law-optimal context compaction
//!
//! This crate proves that session auto-compaction is equivalent to entropy management
//! under a conservation law: γ (kept) + η (discarded) = total_budget.
//!
//! The optimal compaction minimizes information loss (η) subject to the budget
//! constraint (γ), which is exactly rate-distortion theory from information theory.

pub mod budget;
pub mod compaction;
pub mod entropy;
pub mod experiment;
pub mod message;

pub use budget::ConservationBudget;
pub use compaction::CompactionResult;
pub use entropy::{
    distortion, information_density, kl_divergence, mutual_information, rate, shannon_entropy,
};
pub use experiment::{Experiment, ExperimentResult, Strategy, compare, comparison_table};
pub use message::{Message, Role, semantic_distance};
