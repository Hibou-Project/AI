from sqlalchemy.orm import Mapped, mapped_column
from src.models.base import Base
from sqlalchemy import String


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    size: Mapped[str] = mapped_column(String(255)) # nano, small, medium
    yolo_version: Mapped[int] = mapped_column() # 11, 26
