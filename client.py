"""PPO Clipped Surrogate Loss Evaluator.
100% Python Standard Library.
"""

import math

class PPOClippedLossEvaluator:
    """Evaluates PPO clipped surrogate objective with KL penalty."""
    @staticmethod
    def evaluate_clipped_loss(logprob_new: float, logprob_old: float, advantage: float, epsilon: float = 0.2, kl_penalty_beta: float = 0.02) -> dict:
        ratio = math.exp(logprob_new - logprob_old)
        surr1 = ratio * advantage
        clipped_ratio = max(1.0 - epsilon, min(1.0 + epsilon, ratio))
        surr2 = clipped_ratio * advantage
        policy_loss = -min(surr1, surr2)

        approx_kl = (ratio - 1.0) - (logprob_new - logprob_old)
        total_loss = policy_loss + kl_penalty_beta * approx_kl

        return {
            "ratio": round(ratio, 4),
            "is_clipped": ratio != clipped_ratio,
            "policy_loss": round(policy_loss, 4),
            "approx_kl": round(approx_kl, 4),
            "total_loss": round(total_loss, 4)
        }
