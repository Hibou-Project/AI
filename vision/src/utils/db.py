from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, AsyncSession, create_async_engine
from sqlalchemy import inspect, text, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine import Connection
from src.settings import SETTINGS
from src.models.base import Base


class Database:
    """
    Singleton Database manager for async SQLAlchemy.
    Handles engine, session factory, and DB initialization.
    """
    _instance = None
    _engine: AsyncEngine = None
    _session_factory: async_sessionmaker[AsyncSession] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_engine(cls) -> AsyncEngine:
        """Get or create the singleton engine."""
        if cls._engine is None:
            cls._engine = create_async_engine(
                f"sqlite+aiosqlite:///{SETTINGS.DB_FILE}",
                echo=False,
            )
        return cls._engine

    @classmethod
    def get_session_factory(cls) -> async_sessionmaker[AsyncSession]:
        """Get or create the singleton async session factory."""
        if cls._session_factory is None:
            cls._session_factory = async_sessionmaker(
                cls.get_engine(),
                expire_on_commit=False
            )
        return cls._session_factory

    @classmethod
    async def init_db(cls):
        """Initialize the database tables and add missing columns for SQLite."""
        engine = cls.get_engine()
        async with engine.begin() as connection:
            # Create tables from metadata
            await connection.run_sync(Base.metadata.create_all)
            # Add any missing columns for SQLite
            await connection.run_sync(cls._sqlite_add_missing_columns)


    @staticmethod
    def _sqlite_add_missing_columns(sync_conn: Connection) -> None:
        """Add missing columns for SQLite tables if necessary."""
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

    @classmethod
    async def get_or_create(cls, session: AsyncSession, model, **kwargs):
        result = await session.execute(select(model).filter_by(**kwargs))
        instance = result.scalars().first()

        if instance:
            return instance

        instance = model(**kwargs)
        session.add(instance)

        try:
            await session.flush()  # get ID
        except IntegrityError:
            await session.rollback()
            result = await session.execute(select(model).filter_by(**kwargs))
            instance = result.scalars().first()

        return instance