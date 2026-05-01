from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import Integer, Float, DateTime, ForeignKey, UniqueConstraint
from datetime import datetime, timezone
from models.base import Base


class Benchmark(Base):
    __tablename__ = "benchmarks"
    __table_args__ = (
        # Ensure a model, on specific hardware, in a specific format, is only benchmarked once per run
        UniqueConstraint(
            "model_id", "hardware_id", "format_id", "batch_size", name="uq_model_hardware_format_batch"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # Foreign keys
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    hardware_id: Mapped[int] = mapped_column(ForeignKey("hardware.id"), nullable=False)
    format_id: Mapped[int] = mapped_column(ForeignKey("formats.id"), nullable=False)

    # Benchmark metrics
    latency_ms: Mapped[float] = mapped_column(nullable=False)        # ms per image
    throughput_fps: Mapped[float] = mapped_column(nullable=True)     # images per second, optional
    map50_95: Mapped[float] = mapped_column(nullable=True)           # Accuracy metric
    model_size_mb: Mapped[float] = mapped_column(nullable=True)      # File size
    batch_size: Mapped[int] = mapped_column(default=1)               # Batch size used in benchmark

    # Timestamp
    run_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    model = relationship("Model", backref="benchmarks")
    hardware = relationship("Hardware", backref="benchmarks")
    format = relationship("Format", backref="benchmarks")