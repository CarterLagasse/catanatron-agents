import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from catanatron.jax_rl import checkpoint  # noqa: E402
from catanatron.jax_rl.env import CatanEnv  # noqa: E402
from catanatron.jax_rl.jax_player import JaxPlayer  # noqa: E402
from catanatron.jax_rl.network import GraphPolicy  # noqa: E402
from catanatron.jax_rl.obs import encode_current  # noqa: E402
from catanatron.jax_rl.ppo import PPOConfig, Trainer, Transition, train  # noqa: E402
from catanatron.jax_rl.psro import (  # noqa: E402
    PSROConfig,
    exploitability,
    solve_meta,
)
from catanatron.jax_rl.psro import train as train_psro  # noqa: E402
from catanatron.game import Game  # noqa: E402
from catanatron.models.player import Color, RandomPlayer  # noqa: E402

TINY = dict(
    map_type="MINI",
    num_envs=16,
    rollout_ticks=8,
    pool_segments=2,
    hidden=8,
    rounds=1,
    num_minibatches=2,
)


def _gae_inputs(seats, rewards, dones, values, learner=None):
    t = len(seats)
    learner = np.ones(t, bool) if learner is None else np.array(learner)
    return Transition(
        state=None,
        seat=jnp.array(seats, jnp.int8)[:, None],
        is_learner=jnp.array(learner)[:, None],
        action=jnp.zeros((t, 1), jnp.int16),
        log_prob=jnp.zeros((t, 1)),
        value=jnp.array(values, jnp.float32)[:, None],
        rewards=jnp.array(rewards, jnp.float32)[:, None, :],
        done=jnp.array(dones)[:, None],
    )


@pytest.fixture(scope="module")
def trainer():
    config = PPOConfig(total_timesteps=16 * 8 * 2, **TINY)
    return Trainer(config)


def _gae(trainer, gamma, lam, transitions, last_values):
    trainer_config = trainer.config
    object.__setattr__(trainer_config, "gamma", gamma)
    object.__setattr__(trainer_config, "gae_lambda", lam)
    return trainer.compute_gae(transitions, jnp.array(last_values, jnp.float32)[None])


def test_gae_chains_follow_each_seat(trainer):
    # Seat 0 decides at t=0 and t=2, seat 1 at t=1. Rewards on seat 1's tick must
    # accrue to seat 0's decision at t=0, and each seat bootstraps from its own next
    # decision rather than the next tick.
    transitions = _gae_inputs(
        seats=[0, 1, 0],
        rewards=[[0.0, 0.0], [0.5, -0.5], [0.0, 0.0]],
        dones=[False, False, False],
        values=[0.1, 0.2, 0.3],
    )
    advantages, returns = _gae(trainer, 1.0, 0.0, transitions, [0.7, -0.7])
    advantages = np.asarray(advantages)[:, 0]
    assert advantages[0] == pytest.approx(0.5 + 0.3 - 0.1)
    assert advantages[1] == pytest.approx(-0.5 + (-0.7) - 0.2)
    assert advantages[2] == pytest.approx(0.0 + 0.7 - 0.3)
    assert np.asarray(returns)[0, 0] == pytest.approx(0.8)


def test_gae_game_end_cuts_chains(trainer):
    # The game ends at t=1 on seat 1's move: seat 0 lost at the decision before it.
    # The new game's value at t=2 and the final bootstrap must not leak backward.
    transitions = _gae_inputs(
        seats=[0, 1, 0],
        rewards=[[0.0, 0.0], [-1.0, 1.0], [0.0, 0.0]],
        dones=[False, True, False],
        values=[0.25, 0.5, 0.9],
    )
    advantages = np.asarray(_gae(trainer, 1.0, 1.0, transitions, [0.3, -0.3])[0])[:, 0]
    assert advantages[0] == pytest.approx(-1.0 - 0.25)
    assert advantages[1] == pytest.approx(1.0 - 0.5)
    assert advantages[2] == pytest.approx(0.3 - 0.9)


def test_gae_skips_opponent_ticks(trainer):
    # Against a frozen opponent only seat 0 learns; seat 1's ticks must pass through.
    transitions = _gae_inputs(
        seats=[0, 1, 1, 0],
        rewards=[[0.1, -0.1], [0.2, -0.2], [0.3, -0.3], [0.0, 0.0]],
        dones=[False, False, False, False],
        values=[1.0, 99.0, 99.0, 2.0],
        learner=[True, False, False, True],
    )
    advantages = np.asarray(_gae(trainer, 1.0, 0.0, transitions, [0.0, 0.0])[0])[:, 0]
    assert advantages[0] == pytest.approx(0.1 + 0.2 + 0.3 + 2.0 - 1.0)


def test_network_output_shape_and_mask_compatibility():
    env = CatanEnv("BASE")
    policy = GraphPolicy(env, hidden=8, rounds=1)
    params = policy.init(jax.random.PRNGKey(0))
    state = jax.tree_util.tree_map(lambda x: x[0], env.batch_reset(jax.random.PRNGKey(0), 1))
    logits, value = policy.apply(params, encode_current(env, state))
    assert logits.shape == (env.num_actions,)
    assert value.shape == ()


def test_network_action_heads_line_up_with_the_action_layout():
    """Perturbing one node must move that node's settlement and city logits most, and
    likewise for edges and tiles, or the heads would be wired to the wrong actions."""
    env = CatanEnv("BASE")
    topo = env.topo
    policy = GraphPolicy(env, hidden=8, rounds=0)
    params = policy.init(jax.random.PRNGKey(0))
    state = jax.tree_util.tree_map(lambda x: x[0], env.batch_reset(jax.random.PRNGKey(0), 1))
    obs = encode_current(env, state)
    base, _ = policy.apply(params, obs)

    changed = obs._replace(nodes=obs.nodes.at[7].add(1.0))
    diff = np.abs(np.asarray(policy.apply(params, changed)[0] - base))
    assert set(np.flatnonzero(diff)) == {topo.SETTLEMENT + 7, topo.CITY + 7}

    changed = obs._replace(edges=obs.edges.at[11].add(1.0))
    diff = np.abs(np.asarray(policy.apply(params, changed)[0] - base))
    assert set(np.flatnonzero(diff)) == {topo.ROAD + 11}

    changed = obs._replace(tiles=obs.tiles.at[4].add(1.0))
    diff = np.abs(np.asarray(policy.apply(params, changed)[0] - base))
    assert set(np.flatnonzero(diff)) == {topo.ROBBER + 8, topo.ROBBER + 9}


def test_ppo_smoke_and_checkpoint_round_trip(tmp_path):
    config = PPOConfig(
        total_timesteps=16 * 8 * 2,
        eval_every=2,
        eval_envs=4,
        eval_ticks=8,
        snapshot_every=1,
        phase_iterations=1,
        **TINY,
    )
    params = train(config, tmp_path, log=lambda _: None)
    assert (tmp_path / "latest.npz").exists()
    assert (tmp_path / "metrics.jsonl").read_text().count("\n") == 2

    loaded, meta = checkpoint.load_params(tmp_path / "latest.npz")
    assert meta == {"hidden": 8, "rounds": 1, "map_type": "MINI"}
    flat_a = jax.tree_util.tree_leaves(params)
    flat_b = jax.tree_util.tree_leaves(loaded)
    assert len(flat_a) == len(flat_b)
    assert jax.tree_util.tree_structure(params) == jax.tree_util.tree_structure(loaded)
    for a, b in zip(flat_a, flat_b):
        assert np.array_equal(np.asarray(a), b)


def _resume_config(timesteps):
    return PPOConfig(
        total_timesteps=timesteps,
        eval_every=1,
        eval_envs=4,
        eval_ticks=8,
        snapshot_every=1,
        phase_iterations=1,
        **TINY,
    )


def test_ppo_resume_continues_where_it_stopped(tmp_path):
    full = 16 * 8 * 4
    # A learning-rate schedule only matches if both runs plan the same total length,
    # so the first run is cut short by raising out of its logger.
    class Stop(Exception):
        pass

    lines = []

    def stop_after_two(line):
        lines.append(line)
        if len(lines) == 3:  # the header, then iterations 1 and 2
            raise Stop

    with pytest.raises(Stop):
        train(_resume_config(full), tmp_path, log=stop_after_two)

    train(_resume_config(full), tmp_path, log=lambda _: None, resume=True)
    rows = (tmp_path / "metrics.jsonl").read_text().splitlines()
    import json

    iterations = [json.loads(r)["iteration"] for r in rows]
    assert iterations == [1, 2, 3, 4]
    assert json.loads(rows[-1])["steps"] == full


def test_ppo_resume_replaces_rows_after_the_saved_state(tmp_path):
    import json

    config = PPOConfig(
        total_timesteps=16 * 8 * 3,
        eval_every=2,
        eval_envs=4,
        eval_ticks=8,
        snapshot_every=1,
        phase_iterations=1,
        **TINY,
    )
    train(config, tmp_path, log=lambda _: None)
    # The last save was at iteration 3; pretend the run died during iteration 3
    # by rolling the saved state back to the fallback at iteration 2.
    (tmp_path / "train_state.pkl").unlink()
    metrics = tmp_path / "metrics.jsonl"
    rows = metrics.read_text().splitlines()
    last = json.loads(rows[-1])
    last.pop("eval_heuristic_win")
    metrics.write_text("".join(r + chr(10) for r in rows[:-1] + [json.dumps(last)]))

    train(config, tmp_path, log=lambda _: None, resume=True)
    iterations = [json.loads(r)["iteration"] for r in metrics.read_text().splitlines()]
    assert iterations == [1, 2, 3]


def test_ppo_resume_without_train_state_uses_checkpoint(tmp_path):
    full = 16 * 8 * 3
    train(_resume_config(16 * 8 * 2), tmp_path, log=lambda _: None)
    (tmp_path / "train_state.pkl").unlink()
    train(_resume_config(full), tmp_path, log=lambda _: None, resume=True)
    assert (tmp_path / "metrics.jsonl").read_text().count(chr(10)) == 3


def test_ppo_learns_a_simple_game(tmp_path):
    """On MINI to 4 VP, a short run must clearly beat the random bot."""
    pytest.importorskip("optax")
    config = PPOConfig(
        map_type="MINI",
        vps_to_win=4,
        num_envs=64,
        rollout_ticks=32,
        pool_fraction=0.0,
        scripted_fraction=0.5,
        heuristic_share=0.0,
        hidden=16,
        rounds=2,
        num_minibatches=4,
        total_timesteps=64 * 32 * 60,
        learning_rate=1e-3,
        final_learning_rate=3e-4,
        eval_every=60,
        eval_envs=128,
        eval_ticks=200,
    )
    rows = []
    train(config, tmp_path, log=rows.append)
    assert "EVAL random" in rows[-1]
    win_rate = float(rows[-1].split("EVAL random ")[1].split(" ")[0])
    assert win_rate > 0.7, rows[-1]


def test_meta_solver_on_known_games():
    rock_paper_scissors = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], float)
    strategy = solve_meta(rock_paper_scissors)
    assert np.allclose(strategy, 1 / 3, atol=0.01)
    assert exploitability(rock_paper_scissors, strategy) < 0.01

    # A strictly dominant member takes all the weight.
    dominated = np.array([[0, 1, 1], [-1, 0, 0.5], [-1, -0.5, 0]])
    assert solve_meta(dominated)[0] == pytest.approx(1.0)


def test_psro_smoke_and_mixture_plays_engine_games(tmp_path):
    config = PSROConfig(
        map_type="MINI",
        iterations=2,
        first_response_timesteps=16 * 8 * 2,
        response_timesteps=16 * 8,
        num_envs=16,
        rollout_ticks=8,
        pool_segments=2,
        hidden=8,
        rounds=1,
        phase_iterations=1,
        eval_envs=8,
        eval_ticks=16,
    )
    population = train_psro(config, tmp_path, log=lambda _: None)
    assert len(population) == len(config.scripted_population) + 2
    # The payoff matrix stays antisymmetric.
    assert np.allclose(population.payoffs, -population.payoffs.T)

    player = JaxPlayer(Color.BLUE, str(tmp_path))
    from catanatron.models.map import build_map

    game = Game([player, RandomPlayer(Color.RED)], catan_map=build_map("MINI"))
    game.play()
    assert game.state.num_turns > 0


def test_psro_includes_init_agent_and_resumes(tmp_path):
    config = PSROConfig(
        map_type="MINI",
        iterations=1,
        first_response_timesteps=16 * 8,
        response_timesteps=16 * 8,
        num_envs=16,
        rollout_ticks=8,
        pool_segments=2,
        hidden=8,
        rounds=1,
        phase_iterations=1,
        eval_envs=8,
        eval_ticks=16,
    )
    env = CatanEnv("MINI")
    init = GraphPolicy(env, hidden=8, rounds=1).init(jax.random.PRNGKey(1))
    first = train_psro(config, tmp_path, log=lambda _: None, init_params=init)
    assert first.names == ["random", "weighted", "heuristic", "init", "br_00"]

    more = PSROConfig(**{**config.__dict__, "iterations": 2})
    second = train_psro(more, tmp_path, log=lambda _: None, resume=True)
    assert second.names == first.names + ["br_01"]
    assert np.allclose(second.payoffs[:5, :5], first.payoffs)
    assert np.allclose(second.payoffs, -second.payoffs.T)

    # A resumed run with no time left adds nothing.
    third = PSROConfig(**{**config.__dict__, "iterations": 5})
    stopped = train_psro(third, tmp_path, log=lambda _: None, resume=True, max_hours=0.0)
    assert stopped.names == second.names


def test_cli_parses_jax_player(tmp_path):
    from catanatron.cli.cli_players import parse_cli_string

    env = CatanEnv("BASE")
    policy = GraphPolicy(env, hidden=8, rounds=1)
    path = tmp_path / "agent.npz"
    checkpoint.save_params(path, policy.init(jax.random.PRNGKey(0)), 8, 1, "BASE")
    # Relative-style params, since a Windows drive colon would split the player spec.
    import os

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        players = parse_cli_string("JAX:agent.npz:1,R")
    finally:
        os.chdir(cwd)
    assert type(players[0]).__name__ == "JaxPlayer"
    assert players[0].greedy
    game = Game(players)
    game.play()
    assert game.state.num_turns > 0
