from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.engine import Connection
from sqlalchemy import inspect, text
from models.base import Base


def create_engine_and_session_factory(database_url: str,) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, session_factory


def sqlite_add_missing_columns(sync_conn: Connection) -> None:
    if sync_conn.dialect.name != "sqlite":
        return
    inspector = inspect(sync_conn)
    for table in Base.metadata.sorted_tables:
        if table.name not in inspector.get_table_names():
            continue
        existing = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing:
                continue
            coltype = column.type.compile(dialect=sync_conn.dialect)
            sync_conn.execute(
                text(f"ALTER TABLE {table.name} ADD COLUMN {column.name} {coltype}")
            )
