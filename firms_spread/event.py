"""Общая временнáя шкала события.

Все очаги живут на одной сетке: от первой съёмки до конца горизонта.
Очаги загораются по мере обнаружения, а не все разом.

Шаг сетки укрупняется автоматически: восемь суток наблюдений по часу —
это под две сотни кадров, что и грузится долго, и смотрится утомительно.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Больше этого числа кадров шкала становится неповоротливой.
MAX_FRAMES = 96


@dataclass
class EventTimeline:
    """Сетка кадров от начала наблюдений до конца прогноза."""

    start: pd.Timestamp
    end: pd.Timestamp
    step_hours: int = 1

    def __post_init__(self):
        self.stamps = list(
            pd.date_range(self.start, self.end, freq=f"{self.step_hours}h", tz="UTC")
        )
        self._lookup = {stamp: index for index, stamp in enumerate(self.stamps)}

    def __len__(self) -> int:
        return len(self.stamps)

    def index_of(self, stamp) -> int | None:
        """Номер кадра для отметки времени, прижатой к сетке."""
        moment = pd.Timestamp(stamp)
        if moment.tzinfo is None:
            moment = moment.tz_localize("UTC")

        direct = self._lookup.get(moment.floor("h"))
        if direct is not None:
            return direct

        # Сетка крупнее часа — прижимаем к ближайшему предыдущему узлу.
        offset = (moment - self.start).total_seconds() / 3600.0
        if offset < 0:
            return None
        index = int(offset // self.step_hours)
        return index if 0 <= index < len(self.stamps) else None

    def labels(self) -> list[dict]:
        """Подписи кадров для интерфейса."""
        return [
            {"label": f"{stamp:%d.%m %H:%M}", "hours": index * self.step_hours}
            for index, stamp in enumerate(self.stamps)
        ]


def build_event_timeline(clusters, horizon_hours: int, step_hours: int | None = None):
    """Сетка от самой ранней съёмки до последней плюс горизонт.

    Если шаг не задан, он подбирается так, чтобы кадров осталось разумное
    количество: длинное окно наблюдений иначе даёт неподъёмную анимацию.
    """
    earliest = min(c.points["acquired_at"].min() for c in clusters).floor("h")
    latest = max(c.latest for c in clusters).floor("h")
    end = latest + pd.Timedelta(hours=horizon_hours)

    if step_hours is None:
        span_hours = int((end - earliest).total_seconds() // 3600) + 1
        step_hours = max(1, -(-span_hours // MAX_FRAMES))

    return EventTimeline(start=earliest, end=end, step_hours=step_hours)
