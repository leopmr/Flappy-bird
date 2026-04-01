import argparse
import random

import numpy as np

from text_flappy_bird_env_simple import TextFlappyBirdEnvSimple


def _get_q_row(Q, state, n_actions):
    # Lazy-init Q for unseen states.
    if state not in Q:
        Q[state] = np.zeros(n_actions, dtype=np.float64)
    return Q[state]


def _get_q_row_readonly(Q, state, n_actions):
    # Read-only access: do not create new entries during evaluation.
    return Q.get(state, np.zeros(n_actions, dtype=np.float64))


def _bounds(space):
    # Convert a Discrete space into numeric bounds [start, end].
    start = int(getattr(space, "start", 0))
    n = int(space.n)
    end = start + n - 1
    return start, end


def plot_value_heatmap(env, Q, title="State Value Heatmap"):
    # Visualize V(s)=max_a Q(s,a) over the (dx, dy) grid.
    dx_bounds = _bounds(env.observation_space.spaces[0])
    dy_bounds = _bounds(env.observation_space.spaces[1])
    dx_vals = list(range(dx_bounds[0], dx_bounds[1] + 1))
    dy_vals = list(range(dy_bounds[0], dy_bounds[1] + 1))

    grid = np.zeros((len(dy_vals), len(dx_vals)), dtype=np.float64)
    for i, dy in enumerate(dy_vals):
        for j, dx in enumerate(dx_vals):
            state = (dx, dy)
            q_row = Q.get(state)
            if q_row is None:
                v = 0.0
            else:
                v = float(np.max(q_row))
            grid[i, j] = v

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


def plot_policy_heatmap(env, Q, title="Greedy Policy Heatmap"):
    # Visualize the greedy action per state (0=Idle, 1=Flap).
    dx_bounds = _bounds(env.observation_space.spaces[0])
    dy_bounds = _bounds(env.observation_space.spaces[1])
    dx_vals = list(range(dx_bounds[0], dx_bounds[1] + 1))
    dy_vals = list(range(dy_bounds[0], dy_bounds[1] + 1))

    grid = np.full((len(dy_vals), len(dx_vals)), -1, dtype=np.int32)
    for i, dy in enumerate(dy_vals):
        for j, dx in enumerate(dx_vals):
            state = (dx, dy)
            q_row = Q.get(state)
            if q_row is None:
                action = -1
            else:
                action = int(np.argmax(q_row))
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
            vmin=-1,
            vmax=1,
        )
        plt.colorbar(label="Action (-1=Unseen, 0=Idle, 1=Flap)")
        plt.title(title)
        plt.xlabel("dx")
        plt.ylabel("dy")
        plt.tight_layout()
        plt.show()
    except Exception as exc:
        print(f"Plot skipped (matplotlib not available): {exc}")


def sample_action(Q, state, n_actions, epsilon):
    # Epsilon-greedy sampling from the current Q.
    q_row = _get_q_row(Q, state, n_actions)
    if np.random.rand() < epsilon:
        return int(np.random.randint(n_actions))
    return int(np.argmax(q_row))


def run_episode(env, Q, n_actions, epsilon, render=False, max_steps=10000):
    obs, _ = env.reset()
    state = (int(obs[0]), int(obs[1]))
    action = sample_action(Q, state, n_actions, epsilon)

    episode = []
    done = False
    steps = 0

    # Generate an episode following the epsilon-greedy policy.
    while not done:
        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        done = terminated or truncated
        steps += 1
        if steps >= max_steps:
            break
        if render:
            print(env.render(), end="")
        if done:
            break
        state = (int(next_obs[0]), int(next_obs[1]))
        action = sample_action(Q, state, n_actions, epsilon)

    return episode


def greedy_action(Q, state, n_actions):
    q_row = _get_q_row_readonly(Q, state, n_actions)
    return int(np.argmax(q_row))


def evaluate_greedy(env, Q, episodes=10, max_steps=10000):
    n_actions = env.action_space.n
    total = 0.0
    # Greedy evaluation without exploration.
    for _ in range(episodes):
        obs, _ = env.reset()
        state = (int(obs[0]), int(obs[1]))
        done = False
        steps = 0
        ep_return = 0.0
        while not done and steps < max_steps:
            action = greedy_action(Q, state, n_actions)
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
    Q,
    height=15,
    width=20,
    pipe_gap=4,
    episodes=10,
    max_steps=10000,
):
    """
    Evaluate a trained MC agent on a specific environment configuration.
    """
    env = TextFlappyBirdEnvSimple(height=height, width=width, pipe_gap=pipe_gap)
    return evaluate_greedy(env, Q, episodes=episodes, max_steps=max_steps)


def run_greedy_agent(
    env,
    Q,
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
            action = greedy_action(Q, state, n_actions)
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
    gamma=0.99,
    epsilon=0.2,
    epsilon_min=0.05,
    epsilon_decay=0.999,
    seed=0,
    log_every=100,
    eval_every=200,
    eval_episodes=20,
    max_steps=10000,
    plot_eval=True,
):
    """
    First-Visit MC Control (on-policy, epsilon-soft).
    Returns learned Q plus training curves.
    """
    random.seed(seed)
    np.random.seed(seed)

    if env is None:
        env = TextFlappyBirdEnvSimple()

    n_actions = env.action_space.n

    Q = {}
    returns_sum = {}
    returns_count = {}

    episode_lengths = []
    eval_steps = []
    eval_returns = []

    for episode_idx in range(1, episodes + 1):
        # 1) Sample an episode using the current epsilon-greedy policy.
        ep = run_episode(env, Q, n_actions, epsilon, render=False, max_steps=max_steps)

        # 2) Compute Monte-Carlo returns for the episode.
        returns = []
        G = 0.0
        for _, _, reward in reversed(ep):
            G = gamma * G + reward
            returns.append(G)
        returns.reverse()

        # 3) First-visit update of Q(s,a) using average returns.
        visited = set()
        for (state, action, _), G in zip(ep, returns):
            key = (state[0], state[1], action)
            if key in visited:
                continue
            visited.add(key)
            returns_sum[key] = returns_sum.get(key, 0.0) + G
            returns_count[key] = returns_count.get(key, 0.0) + 1.0
            q_row = _get_q_row(Q, state, n_actions)
            q_row[action] = returns_sum[key] / returns_count[key]
        # 4) Decay epsilon to reduce exploration over time.
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        episode_lengths.append(len(ep))

        if log_every and episode_idx % log_every == 0:
            recent = episode_lengths[-log_every:]
            avg_len = sum(recent) / max(1, len(recent))
            print(f"Episode {episode_idx:5d} | avg length {avg_len:6.2f} | epsilon {epsilon:.4f}")

        if eval_every and episode_idx % eval_every == 0:
            avg_return = evaluate_greedy(env, Q, episodes=eval_episodes, max_steps=max_steps)
            eval_steps.append(episode_idx)
            eval_returns.append(avg_return)
            print(f"Eval @ {episode_idx:5d} | greedy avg return {avg_return:6.2f}")

    if plot_eval and eval_steps:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 4))
            plt.plot(eval_steps, eval_returns, marker="o")
            plt.title("Greedy Average Return (Eval)")
            plt.xlabel("Episode")
            plt.ylabel("Avg Return")
            plt.tight_layout()
            plt.show()
        except Exception as exc:
            print(f"Plot skipped (matplotlib not available): {exc}")

    return {
        "Q": Q,
        "episode_lengths": episode_lengths,
        "eval_steps": eval_steps,
        "eval_returns": eval_returns,
    }
