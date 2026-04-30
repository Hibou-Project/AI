from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, UniqueConstraint
from models.base import Base


class Hardware(Base):
    __tablename__ = "hardware"
    __table_args__ = (
        UniqueConstraint("name", "driver", name="uq_hardware_name_driver"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), index=True)
    type: Mapped[str] = mapped_column(String(50))  # GPU / CPU / TPU
    memory_gb: Mapped[int] = mapped_column(nullable=True)
    driver: Mapped[str] = mapped_column(String(255), nullable=True)
    cores: Mapped[int] = mapped_column(nullable=True)