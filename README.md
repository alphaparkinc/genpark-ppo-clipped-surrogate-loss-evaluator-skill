# genpark-ppo-clipped-surrogate-loss-evaluator-skill

Proximal Policy Optimization (PPO) clipped surrogate loss calculator enforcing trust-region policy updates and KL penalty terms.

## Architecture

```mermaid
flowchart TD
    LogProbs["Logprob New & Old"] --> Ratio["Importance Ratio r(theta)"]
    Advantage["Advantage Estimate A_t"] --> Surr1["Unclipped Objective r * A"]
    Ratio --> Clip["Clip to 1 - eps .. 1 + eps"]
    Clip --> Surr2["Clipped Objective clip(r) * A"]
    Surr1 & Surr2 --> Min["Min(Surr1, Surr2)"]
    Min --> Final["PPO Policy Loss + KL Penalty"]
```

## Features
- **Clipping Detection**: Flags when updates exceed trust region threshold.
- **Pure Python**: 100% standard library.
