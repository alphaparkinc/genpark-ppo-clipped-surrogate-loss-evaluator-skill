from client import PPOClippedLossEvaluator

res = PPOClippedLossEvaluator.evaluate_clipped_loss(-0.4, -0.7, 1.2)
print("PPO Loss Evaluation:", res)
