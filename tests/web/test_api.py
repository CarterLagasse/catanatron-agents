import pytest
import json
from catanatron.web import api, create_app
from catanatron.web.models import db, GameState, get_game_state


@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    # Setup an in-memory SQLite database for testing
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "SECRET_KEY": "test",
        }
    )

    with app.app_context():
        db.create_all()

    yield app

    # Teardown: drop all tables after each test (optional, if tests are isolated)
    # with app.app_context():
    #     db.drop_all()


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


def test_post_game_endpoint(client):
    """Test creating a new game."""
    response = client.post("/api/games", json={"players": ["RANDOM", "RANDOM"]})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "game_id" in data
    # Further check: Ensure the game was actually created in the db
    with client.application.app_context():
        assert (
            db.session.query(GameState).filter_by(uuid=data["game_id"]).first()
            is not None
        )


@pytest.fixture
def jax_agent(tmp_path, monkeypatch):
    """A small untrained agent standing in for the trained ones in runs/."""
    jax = pytest.importorskip("jax")
    from catanatron.jax_rl import checkpoint
    from catanatron.jax_rl.env import CatanEnv
    from catanatron.jax_rl.network import GraphPolicy

    params = GraphPolicy(CatanEnv("BASE"), 8, 1).init(jax.random.PRNGKey(0))
    path = tmp_path / "agent.npz"
    checkpoint.save_params(path, params, 8, 1, "BASE")
    monkeypatch.setattr(
        api, "JAX_AGENTS", {"test": {"label": "Test agent", "path": str(path)}}
    )
    return "JAX:test"


def test_jax_agents_lists_only_agents_on_disk(client, jax_agent, monkeypatch):
    agents = client.get("/api/jax-agents").get_json()["agents"]
    assert agents == [{"key": jax_agent, "label": "Test agent"}]

    monkeypatch.setattr(
        api, "JAX_AGENTS", {"gone": {"label": "Gone", "path": "does/not/exist.npz"}}
    )
    assert client.get("/api/jax-agents").get_json()["agents"] == []


def test_human_can_play_a_jax_agent_across_requests(client, jax_agent):
    """Each request reloads the game from the database, so the agent must survive
    being pickled and keep playing, whichever color it gets."""
    response = client.post("/api/games", json={"players": ["HUMAN", jax_agent]})
    assert response.status_code == 200
    game_id = response.get_json()["game_id"]

    # An empty body makes whoever is next play one move.
    for _ in range(40):
        assert client.post(f"/api/games/{game_id}/actions").status_code == 200

    with client.application.app_context():
        game = get_game_state(game_id)
    assert len(game.state.action_records) == 40
    assert any(player.__class__.__name__ == "JaxPlayer" for player in game.state.players)


@pytest.mark.parametrize(
    "extra",
    [
        {"players": ["HUMAN", "{agent}", "RANDOM"]},
        {"players": ["HUMAN", "{agent}"], "map_template": "MINI"},
        {"players": ["HUMAN", "JAX:unknown"]},
    ],
)
def test_jax_agents_reject_setups_they_cannot_play(client, jax_agent, extra):
    body = {**extra, "players": [p.format(agent=jax_agent) for p in extra["players"]]}
    assert client.post("/api/games", json=body).status_code == 400


def test_post_game_endpoint_accepts_custom_config(client):
    response = client.post(
        "/api/games",
        json={
            "players": ["WEIGHTED_RANDOM", "CATANATRON"],
            "map_template": "MINI",
            "vps_to_win": 15,
            "discard_limit": 12,
            "friendly_robber": True,
        },
    )
    assert response.status_code == 200
    data = json.loads(response.data)

    with client.application.app_context():
        game = get_game_state(data["game_id"])
        assert game.friendly_robber is True

    state_response = client.get(f"/api/games/{data['game_id']}/states/latest")
    assert state_response.status_code == 200
    state_data = json.loads(state_response.data)
    land_tiles = [
        tile for tile in state_data["tiles"] if tile["tile"]["type"] != "WATER"
    ]
    assert len(land_tiles) == 7


def test_post_game_endpoint_rejects_invalid_config(client):
    response = client.post(
        "/api/games",
        json={
            "players": ["RANDOM"],
            "map_template": "INVALID",
            "vps_to_win": 25,
            "discard_limit": 2,
        },
    )
    assert response.status_code == 400


def test_get_game_endpoint(client):
    """Test retrieving a specific game state."""
    # First, create a game to retrieve
    post_response = client.post("/api/games", json={"players": ["RANDOM", "RANDOM"]})
    game_id = json.loads(post_response.data)["game_id"]

    # Retrieve the initial state (state_index 0)
    response = client.get(f"/api/games/{game_id}/states/0")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "nodes" in data
    assert "edges" in data
    assert data["is_initial_build_phase"] is True
    assert data["winning_color"] is None


def test_get_latest_game_endpoint(client):
    """Test retrieving the latest game state."""
    post_response = client.post("/api/games", json={"players": ["RANDOM", "RANDOM"]})
    game_id = json.loads(post_response.data)["game_id"]

    response = client.get(f"/api/games/{game_id}/states/latest")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "nodes" in data
    assert "edges" in data
    assert data["is_initial_build_phase"] is True
    assert data["winning_color"] is None


def test_get_game_not_found(client):
    """Test retrieving a non-existent game."""
    response = client.get("/api/games/nonexistentgameid/states/0")
    assert response.status_code == 404


def test_post_action_bot_turn(client):
    """Test posting an action when it's a bot's turn."""
    # Create a game with at least one bot (RANDOM is a bot)
    post_response = client.post("/api/games", json={"players": ["RANDOM", "HUMAN"]})
    assert post_response.status_code == 200
    game_id = json.loads(post_response.data)["game_id"]

    data_before_res = client.get(f"/api/games/{game_id}/states/latest")
    data_before = json.loads(data_before_res.data)

    after_action_res = client.post(f"/api/games/{game_id}/actions", json={})
    assert after_action_res.status_code == 200
    data_after = json.loads(after_action_res.data)

    # Check if game state progressed, e.g., turn changed or actions list grew
    assert len(data_after["action_records"]) > len(data_before["action_records"])


def test_repeated_bot_actions_advance_latest_state(client):
    """Latest state should keep advancing across persisted bot turns."""
    post_response = client.post("/api/games", json={"players": ["RANDOM", "RANDOM"]})
    assert post_response.status_code == 200
    game_id = json.loads(post_response.data)["game_id"]

    latest_before = json.loads(client.get(f"/api/games/{game_id}/states/latest").data)
    first_tick = json.loads(client.post(f"/api/games/{game_id}/actions", json={}).data)
    second_tick = json.loads(client.post(f"/api/games/{game_id}/actions", json={}).data)
    latest_after = json.loads(client.get(f"/api/games/{game_id}/states/latest").data)

    assert first_tick["state_index"] == latest_before["state_index"] + 1
    assert second_tick["state_index"] == first_tick["state_index"] + 1
    assert len(second_tick["action_records"]) == len(first_tick["action_records"]) + 1
    assert latest_after["state_index"] == second_tick["state_index"]
    assert latest_after["action_records"] == second_tick["action_records"]


def test_mcts_analysis_endpoint(client):
    """Test the MCTS analysis endpoint."""
    post_response = client.post("/api/games", json={"players": ["RANDOM", "RANDOM"]})
    game_id = json.loads(post_response.data)["game_id"]

    # Request MCTS analysis for the latest state
    response = client.get(f"/api/games/{game_id}/states/latest/mcts-analysis")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert "probabilities" in data
    # Further checks on probabilities structure if known
    assert len(data["probabilities"]) == 2  # For two players


def test_mcts_analysis_game_not_found(client):
    """Test MCTS analysis for a non-existent game."""
    response = client.get("/api/games/nonexistent/states/nonexistent/mcts-analysis")
    assert response.status_code == 400


# Stress test endpoint is simple, just check if it runs
def test_stress_test_endpoint(client):
    response = client.get("/api/stress-test")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["winning_color"] is None
