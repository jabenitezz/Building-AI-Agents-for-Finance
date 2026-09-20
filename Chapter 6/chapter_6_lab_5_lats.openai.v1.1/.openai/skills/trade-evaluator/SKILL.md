---
name: trade-evaluator
description: Score a partial tech-sector trim trajectory. Use when asked to
  evaluate how close a trajectory is to the target reduction at acceptable cost.
---

Score the trajectory from 1 to 10 using these criteria:

- 6-10: progress on target exposure exceeds elapsed time fraction, slippage
  below 8 bps per child order, residual risk broadly diversified.
- 3-5: on-pace but slippage between 8 and 15 bps, or concentrated residual.
- 1-2: behind schedule, or slippage above 15 bps on any child order.

Return a single integer score and a one-sentence justification.
