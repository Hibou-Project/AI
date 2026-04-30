from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, UniqueConstraint
from models.base import Base


class Format(Base):
    __tablename__ = "formats"
    __table_args__ = (
        UniqueConstraint("name", "precision", name="uq_format_precision"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), index=True)         # PyTorch / ONNX / TensorRT
    precision: Mapped[str] = mapped_column(String(50), index=True)    # FP32 / FP16 / INT8