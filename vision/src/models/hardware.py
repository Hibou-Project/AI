from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, UniqueConstraint
from src.models.base import Base


class Hardware(Base):
    __tablename__ = "hardware"
    __table_args__ = (
        UniqueConstraint("name", name="uq_hardware_name"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    type: Mapped[str] = mapped_column(String(50))  # GPU / CPU / TPU