import argparse
import random

import numpy as np

from text_flappy_bird_env_simple import TextFlappyBirdEnvSimple


def _bounds(space):
    # Convert a Discrete space into numeric bounds [start, end].
    start = int(getattr(space, "start", 0))
    n = int(space.n)
    end = start + n - 1
    return start, end


def _normalize(value, vmin, vmax):
    # Normalize to [-1, 1] with clipping.
    if vmax == vmin:
        return 0.0
    value = min(max(value, vmin), vmax)
    return 2.0 * (value - vmin) / (vmax - vmin) - 1.0


def _state_features(state, dx_bounds, dy_bounds):
    # Compact polynomial features for (dx, dy).
    dx, dy = float(state[0]), float(state[1])
    dx_n = _normalize(dx, dx_bounds[0], dx_bounds[1])
    dy_n = _normalize(dy, dy_bounds[0], dy_bounds[1])
    return np.array(
        [1.0, dx_n, dy_n, dx_n * dy_n, dx_n * dx_n, dy_n * dy_n],
        dtype=np.float64,
    )


def _action_features(state, action, dx_bounds, dy_bounds, n_actions):
    # Action-dependent features: concat(state_features) per action.
    base = _state_features(state, dx_bounds, dy_bounds)
    d = base.shape[0]
    x = np.zeros(d * n_actions, dtype=np.float64)
    start = action * d
    x[start : start + d] = base
    return x


def q_value(w, state, action, dx_bounds, dy_bounds, n_actions):
    # Linear approximation: q(s,a) = w^T x(s,a).
    x = _action_features(state, action, dx_bounds, dy_bounds, n_actions)
    return float(np.dot(w, x))


def q_values(w, state, dx_bounds, dy_bounds, n_actions):
    return np.array(
        [q_value(w, state, a, dx_bounds, dy_bounds, n_actions) for a in range(n_actions)],
        dtype=np.float64,
    )


def plot_value_heatmap(env, w, title="State Value Heatmap"):
    # Visualize V(s)=max_a q(s,a) over the (dx, dy) grid.
    # Interpretation: brighter regions indicate states the greedy policy expects
    # to yield higher long-term return; sharp transitions often align with
    # decision boundaries (e.g., when to flap vs. stay idle).
    n_actions = env.action_space.n
    dx_bounds = _bounds(env.observation_space.spaces[0])
    dy_bounds = _bounds(env.observation_space.spaces[1])
    dx_vals = list(range(dx_bounds[0], dx_bounds[1] + 1))
    dy_vals = list(range(dy_bounds[0], dy_bounds[1] + 1))

    grid = np.zeros((len(dy_vals), len(dx_vals)), dtype=np.float64)
    for i, dy in enumerate(dy_vals):
        for j, dx in enumerate(dx_vals):
            state = (dx, dy)
            qs = q_values(w, state, dx_bounds, dy_bounds, n_actions)
            grid[i, j] = float(np.max(qs))

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 5))
        plt.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[dx_bounds[0], dx_bounds[1], dy_bounds[0], dy_bounds[1]],
            cmap="viridis",
        )
        plt.colorbar(label="V(s) = max_a Q(s,a)")
        plt.title(title)
        plt.xlabel("dx")
        plt.ylabel("dy")
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"Plot skipped (matplotlib not available): {exc}")


def plot_policy_heatmap(env, w, title="Greedy Policy Heatmap"):
    # Visualize the greedy action per state (0=Idle, 1=Flap).
    n_actions = env.action_space.n
    dx_bounds = _bounds(env.observation_space.spaces[0])
    dy_bounds = _bounds(env.observation_space.spaces[1])
    dx_vals = list(range(dx_bounds[0], dx_bounds[1] + 1))
    dy_vals = list(range(dy_bounds[0], dy_bounds[1] + 1))

    grid = np.zeros((len(dy_vals), len(dx_vals)), dtype=np.int32)
    for i, dy in enumerate(dy_vals):
        for j, dx in enumerate(dx_vals):
            state = (dx, dy)
            qs = q_values(w, state, dx_bounds, dy_bounds, n_actions)
            action = int(np.argmax(qs))
            grid[i, j] = action

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 5))
        plt.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[dx_bounds[0], dx_bounds[1], dy_bounds[0], dy_bounds[1]],
            cmap="coolwarm",
            vmin=0,
            vmax=1,
        )
        plt.colorbar(label="Action (0=Idle, 1=Flap)")
        plt.title(title)
        plt.xlabel("dx")
        plt.ylabel("dy")
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"Plot skipped (matplotlib not available): {exc}")


def epsilon_greedy(w, state, dx_bounds, dy_bounds, n_actions, epsilon):
    # Epsilon-greedy action selection from the linear value estimates.
    if np.random.rand() < epsilon:
        return int(np.random.randint(n_actions))
    qs = q_values(w, state, dx_bounds, dy_bounds, n_actions)
    max_q = np.max(qs)
    best_actions = np.flatnonzero(qs == max_q)
    return int(np.random.choice(best_actions))


def greedy_action(w, state, dx_bounds, dy_bounds, n_actions):
    # Greedy action selection (ties broken uniformly).
    qs = q_values(w, state, dx_bounds, dy_bounds, n_actions)
    max_q = np.max(qs)
    best_actions = np.flatnonzero(qs == max_q)
    return int(np.random.choice(best_actions))


def evaluate_greedy(env, w, episodes=10, max_steps=10000):
    n_actions = env.action_space.n
    dx_bounds = _bounds(env.observation_space.spaces[0])
    dy_bounds = _bounds(env.observation_space.spaces[1])
    total = 0.0
    # Greedy evaluation without exploration.
    for _ in range(episodes):
        obs, _ = env.reset()
        state = (int(obs[0]), int(obs[1]))
        done = False
        steps = 0
        ep_return = 0.0
        while not done and steps < max_steps:
            action = greedy_action(w, state, dx_bounds, dy_bounds, n_actions)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
            steps += 1
            if done:
                break
            state = (int(obs[0]), int(obs[1]))
        total += ep_return
    return total / max(1, episodes)


def evaluate_on_config(
    w,
    height=15,
    width=20,
    pipe_gap=4,
    episodes=10,
    max_steps=10000,
):
    """
    Evaluate a trained SARSA(lambda) agent on a specific environment configuration.
    """
    env = TextFlappyBirdEnvSimple(height=height, width=width, pipe_gap=pipe_gap)
    return evaluate_greedy(env, w, episodes=episodes, max_steps=max_steps)


def run_greedy_agent(
    env,
    w,
    episodes=1,
    render=True,
    max_steps=10000,
    plot_states=True,
    title="Visited States Heatmap",
):
    n_actions = env.action_space.n
    dx_bounds = _bounds(env.observation_space.spaces[0])
    dy_bounds = _bounds(env.observation_space.spaces[1])
    dx_vals = list(range(dx_bounds[0], dx_bounds[1] + 1))
    dy_vals = list(range(dy_bounds[0], dy_bounds[1] + 1))
    visit_grid = np.zeros((len(dy_vals), len(dx_vals)), dtype=np.int64)

    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        state = (int(obs[0]), int(obs[1]))
        done = False
        steps = 0
        ep_return = 0.0
        while not done and steps < max_steps:
            dx, dy = state
            j = dx - dx_bounds[0]
            i = dy - dy_bounds[0]
            if 0 <= i < visit_grid.shape[0] and 0 <= j < visit_grid.shape[1]:
                visit_grid[i, j] += 1
            action = greedy_action(w, state, dx_bounds, dy_bounds, n_actions)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
            steps += 1
            if render:
                print(env.render(), end="")
            if done:
                break
            state = (int(obs[0]), int(obs[1]))
        returns.append(ep_return)

    if plot_states:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(7, 5))
            plt.imshow(
                visit_grid,
                origin="lower",
                aspect="auto",
                extent=[dx_bounds[0], dx_bounds[1], dy_bounds[0], dy_bounds[1]],
                cmap="magma",
            )
            plt.colorbar(label="Visit count")
            plt.title(title)
            plt.xlabel("dx")
            plt.ylabel("dy")
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            print(f"Plot skipped (matplotlib not available): {exc}")
    return returns


def train(
    env=None,
    episodes=3000,
    alpha=0.1,
    gamma=0.99,
    lambda_=0.9,
    epsilon=0.2,
    epsilon_min=0.05,
    epsilon_decay=0.999,
    seed=0,
    log_every=100,
    eval_every=200,
    eval_episodes=20,
    max_steps=10000,
    plot_eval=True,
    keep_best=True,
):
    """
    True Online SARSA(lambda) with linear function approximation.
    Returns weights w plus training curves and best policy snapshot.
    """
    random.seed(seed)
    np.random.seed(seed)

    if env is None:
        env = TextFlappyBirdEnvSimple()

    n_actions = env.action_space.n
    dx_bounds = _bounds(env.observation_space.spaces[0])
    dy_bounds = _bounds(env.observation_space.spaces[1])
    base_dim = 6
    w = np.zeros(base_dim * n_actions, dtype=np.float64)
    best_w = w.copy()
    best_eval = -float("inf")

    episode_lengths = []
    eval_steps = []
    eval_returns = []

    for episode_idx in range(1, episodes + 1):
        # 1) Initialize episode and traces.
        obs, _ = env.reset()
        state = (int(obs[0]), int(obs[1]))
        action = epsilon_greedy(w, state, dx_bounds, dy_bounds, n_actions, epsilon)
        x = _action_features(state, action, dx_bounds, dy_bounds, n_actions)
        z = np.zeros_like(w)
        Qold = 0.0

        done = False
        steps = 0

        # 2) Step through the episode and apply true-online updates.
        while not done and steps < max_steps:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            if done:
                Q_next = 0.0
                x_next = np.zeros_like(w)

            else:
                next_state = (int(next_obs[0]), int(next_obs[1]))
                next_action = epsilon_greedy(w, next_state, dx_bounds, dy_bounds, n_actions, epsilon)
                x_next = _action_features(next_state, next_action, dx_bounds, dy_bounds, n_actions)
                Q_next = float(np.dot(w, x_next))

            Q = float(np.dot(w, x))
            delta = reward + gamma * Q_next - Q

            # True-online trace update.
            z = gamma * lambda_ * z + (1.0 - alpha * gamma * lambda_ * np.dot(z, x)) * x
            # Weight update (true-online SARSA(lambda)).
            w = w + alpha * (delta + Q - Qold) * z - alpha * (Q - Qold) * x

            Qold = Q_next
            if done:
                break
            state = next_state
            action = next_action
            x = x_next

        episode_lengths.append(steps)
        # 3) Decay epsilon to reduce exploration over time.
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if log_every and episode_idx % log_every == 0:
            recent = episode_lengths[-log_every:]
            avg_len = sum(recent) / max(1, len(recent))
            print(f"Episode {episode_idx:5d} | avg length {avg_len:6.2f} | epsilon {epsilon:.4f}")

        if eval_every and episode_idx % eval_every == 0:
            avg_return = evaluate_greedy(env, w, episodes=eval_episodes, max_steps=max_steps)
            eval_steps.append(episode_idx)
            eval_returns.append(avg_return)
            print(f"Eval @ {episode_idx:5d} | greedy avg return {avg_return:6.2f}")
            if keep_best and avg_return >= best_eval:
                best_eval = avg_return
                best_w = w.copy()

    if plot_eval and eval_steps:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 4))
            plt.plot(eval_steps, eval_returns, marker="o")
            plt.title("Greedy Average Return (Eval)")
            plt.xlabel("Episode")
            plt.ylabel("Avg Return")
            # Interpretation: an upward trend suggests improving policy quality,
            # while oscillations indicate ongoing exploration or instability in
            # the value estimates.
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            print(f"Plot skipped (matplotlib not available): {exc}")

    return {
        "w": w,
        "best_w": best_w if best_w is not None else w.copy(),
        "best_eval": best_eval,
        "episode_lengths": episode_lengths,
        "eval_steps": eval_steps,
        "eval_returns": eval_returns,
    }
