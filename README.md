# Flappy Bird RL (Text)

This repository contains a small RL project on a text-based Flappy Bird environment.
The environment is a Gym-style wrapper that exposes two actions:
`0 = Idle` and `1 = Flap`. Observations are compact (distance to the next gap).

Two agents are implemented:
- **Agent 1**: Monte Carlo Control (on-policy, epsilon-soft)
- **Agent 2**: True Online SARSA(lambda) with linear function approximation

Key code lives in `env/` (environment + training scripts). The notebook contains
experiments and plots. For full methodology, figures, and analysis, please refer
to:
- `Flappy_bird.pdf`
- `Flappy_bird_english.pdf`
