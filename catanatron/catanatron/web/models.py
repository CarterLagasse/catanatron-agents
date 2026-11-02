import os
import pickle
import re
import zlib
from contextlib import contextmanager

from flask import abort
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import (
    Column,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    create_engine,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

from catanatron.game import Game
from catanatron.state_functions import get_state_index

# Using approach from: https://stackoverflow.com/questions/41004540/using-sqlalchemy-models-in-and-out-of-flask/41014157
metadata = MetaData()
Base = declarative_base(metadata=metadata)


def with_installed_driver(database_url: str) -> str:
    """Names the Postgres driver we install, since SQLAlchemy's default depends on its
    version. Other databases pass through."""
    return re.sub(r"^postgres(ql)?://", "postgresql+psycopg2://", database_url)


class GameState(Base):
    __tablename__ = "game_states"
    __table_args__ = (Index("ix_game_states_uuid_state_index", "uuid", "state_index"),)

    id = Column(Integer, primary_key=True)
    uuid = Column(String(64), nullable=False)
    state_index = Column(Integer, nullable=False)
    state = Column(String, nullable=False)  # Unused: only older rows hold JSON here
    pickle_data = Column(LargeBinary, nullable=False)

    # TODO: unique uuid and state_index
    @staticmethod
    def from_game(game: Game):
        pickle_data = zlib.compress(pickle.dumps(game, pickle.HIGHEST_PROTOCOL))
        return GameState(
            uuid=game.id,
            state_index=get_state_index(game.state),
            state="",
            pickle_data=pickle_data,
        )


db = SQLAlchemy(metadata=metadata)


@contextmanager
def database_session():
    """Can use like:
    with database_session() as session:
        game_states = session.query(GameState).all()
    """
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://catanatron:victorypoint@127.0.0.1:5432/catanatron_db",
    )
    engine = create_engine(with_installed_driver(database_url))
    session = Session(engine)
    try:
        yield session
    finally:
        session.expunge_all()
        session.close()


def upsert_game_state(game: Game, session_param=None):
    game_state = GameState.from_game(game)
    session = session_param or db.session
    session.add(game_state)
    session.commit()
    return game_state


def get_game_state(game_id, state_index=None) -> Game | None:
    """
    Returns the game from database.
    """
    if state_index is None:
        result = (
            db.session.query(GameState)
            .filter_by(uuid=game_id)
            .order_by(GameState.state_index.desc())
            .first()
        )
        if result is None:
            abort(404)
    else:
        result = (
            db.session.query(GameState)
            .filter_by(uuid=game_id, state_index=state_index)
            .first()
        )
        if result is None:
            abort(404)
    db.session.commit()
    return _load_game(result.pickle_data)


def _load_game(data) -> Game:
    # Rows saved before compression are plain pickles, which start with this byte.
    if data[:1] != b"\x80":
        data = zlib.decompress(data)
    return pickle.loads(data)


def delete_oldest_games(keep: int):
    """Keeps the most recently played games, so a size-limited database never fills."""
    stale = (
        db.session.query(GameState.uuid)
        .group_by(GameState.uuid)
        .order_by(func.max(GameState.id).desc())
        .offset(keep)
        .all()
    )
    if stale:
        db.session.query(GameState).filter(
            GameState.uuid.in_([row.uuid for row in stale])
        ).delete(synchronize_session=False)
        db.session.commit()
