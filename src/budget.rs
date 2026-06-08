//! Conservation budget: γ (kept) + η (discarded) = total
//!
//! The conservation law says every token in a session must be either kept (γ) or
//! discarded (η). This module formalizes that invariant.

use std::fmt;

/// Conservation budget tracking γ (information kept) and η (information discarded).
///
/// Invariant: `gamma + eta <= total` at all times.
/// Initially, `gamma = total * gamma_ratio` and `eta = 0`.
/// As tokens are spent from gamma, they move to eta upon compaction.
#[derive(Debug, Clone)]
pub struct ConservationBudget {
    /// Tokens allocated for keeping (γ budget)
    pub gamma: usize,
    /// Tokens that have been discarded (η accumulated)
    pub eta: usize,
    /// Total token budget
    pub total: usize,
    /// Tokens currently spent from gamma
    spent: usize,
}

impl ConservationBudget {
    /// Create a new budget with a 60/40 γ/η split.
    pub fn new(total: usize) -> Self {
        Self::with_ratio(total, 0.6)
    }

    /// Create a new budget with a custom γ ratio (0.0–1.0).
    ///
    /// `gamma_ratio` determines what fraction of `total` is allocated to γ.
    /// The remainder is available for η (discard budget).
    pub fn with_ratio(total: usize, gamma_ratio: f64) -> Self {
        let gamma_ratio = gamma_ratio.clamp(0.0, 1.0);
        let gamma = (total as f64 * gamma_ratio).round() as usize;
        Self {
            gamma,
            eta: 0,
            total,
            spent: 0,
        }
    }

    /// Check if `tokens` can be spent from the γ budget.
    pub fn can_spend(&self, tokens: usize) -> bool {
        self.spent + tokens <= self.gamma
    }

    /// Spend tokens from the γ budget.
    ///
    /// Returns an error if spending would exceed the γ allocation.
    pub fn spend(&mut self, tokens: usize) -> Result<(), BudgetError> {
        if self.can_spend(tokens) {
            self.spent += tokens;
            Ok(())
        } else {
            Err(BudgetError::InsufficientGamma {
                requested: tokens,
                available: self.gamma - self.spent,
            })
        }
    }

    /// Replenish γ budget by recovering tokens from compaction.
    ///
    /// This models the effect of compaction: discarded tokens (η) free up γ space.
    pub fn replenish(&mut self, tokens: usize) {
        self.spent = self.spent.saturating_sub(tokens);
        self.eta = self.eta.saturating_add(tokens);
    }

    /// How much of the total budget is currently used by γ.
    pub fn utilization(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.spent as f64 / self.total as f64
    }

    /// Remaining tokens in the γ budget.
    pub fn budget_remaining(&self) -> usize {
        self.gamma.saturating_sub(self.spent)
    }

    /// Whether the γ budget is fully exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.spent >= self.gamma
    }
}

impl fmt::Display for ConservationBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "γ={} η={} total={}",
            self.gamma, self.eta, self.total
        )
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum BudgetError {
    #[error("insufficient γ budget: requested {requested}, available {available}")]
    InsufficientGamma { requested: usize, available: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_default_60_40_split() {
        let b = ConservationBudget::new(1000);
        assert_eq!(b.gamma, 600);
        assert_eq!(b.eta, 0);
        assert_eq!(b.total, 1000);
    }

    #[test]
    fn with_ratio_custom_split() {
        let b = ConservationBudget::with_ratio(1000, 0.8);
        assert_eq!(b.gamma, 800);
        assert_eq!(b.eta, 0);
    }

    #[test]
    fn with_ratio_clamps_to_valid_range() {
        let b = ConservationBudget::with_ratio(1000, 1.5);
        assert_eq!(b.gamma, 1000);
        let b = ConservationBudget::with_ratio(1000, -0.1);
        assert_eq!(b.gamma, 0);
    }

    #[test]
    fn can_spend_within_budget() {
        let b = ConservationBudget::new(1000);
        assert!(b.can_spend(600));
        assert!(!b.can_spend(601));
    }

    #[test]
    fn spend_success() {
        let mut b = ConservationBudget::new(1000);
        assert!(b.spend(300).is_ok());
        assert_eq!(b.budget_remaining(), 300);
    }

    #[test]
    fn spend_exceeds_budget() {
        let mut b = ConservationBudget::new(1000);
        assert!(b.spend(601).is_err());
    }

    #[test]
    fn replenish_recovers_tokens() {
        let mut b = ConservationBudget::new(1000);
        b.spend(500).unwrap();
        b.replenish(200);
        assert_eq!(b.spent, 300);
        assert_eq!(b.eta, 200);
    }

    #[test]
    fn utilization_tracks_ratio() {
        let mut b = ConservationBudget::new(1000);
        assert!((b.utilization() - 0.0).abs() < 1e-9);
        b.spend(300).unwrap();
        assert!((b.utilization() - 0.3).abs() < 1e-9);
    }

    #[test]
    fn is_exhausted() {
        let mut b = ConservationBudget::new(1000);
        assert!(!b.is_exhausted());
        b.spend(600).unwrap();
        assert!(b.is_exhausted());
    }

    #[test]
    fn display_format() {
        let b = ConservationBudget::new(1000);
        assert_eq!(format!("{b}"), "γ=600 η=0 total=1000");
    }

    #[test]
    fn budget_remaining() {
        let mut b = ConservationBudget::new(1000);
        assert_eq!(b.budget_remaining(), 600);
        b.spend(200).unwrap();
        assert_eq!(b.budget_remaining(), 400);
    }
}
