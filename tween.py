from typing import Callable


def linear_in(t: float) -> float:
    return t

def linear_out(t: float) -> float:
    return 1 - t

def high(t: float) -> float:
    return 1

def low(t: float) -> float:
    return 0

class Segment:
    def __init__(self, duration: float, f: Callable[[float], float]):
        self.duration = duration
        self.tween = f

class LinearIn(Segment):
    def __init__(self, duration: float):
        super().__init__(
            duration=duration,
            f=linear_in)

class LinearOut(Segment):
    def __init__(self, duration: float):
        super().__init__(
            duration=duration,
            f=linear_out)

class High(Segment):
    def __init__(self, duration: float):
        super().__init__(
            duration=duration,
            f=high)

class Low(Segment):
    def __init__(self, duration: float):
        super().__init__(
            duration=duration,
            f=low)


class Tween:
    def __init__(self, *segments):
        self.segments = segments

    def _find_segment(self, t: float):
        if t < 0:
            return self.segments[0], 0

        start = 0
        for segment in self.segments:
            stop = start + segment.duration
            if t >= start and t < stop:
                return segment, (t - start) / segment.duration

            start = stop

        return self.segments[-1], 1

    def __call__(self, t: float):
        segment, normalized_t = self._find_segment(t)
        return segment.tween(normalized_t)

    def __len__(self) -> float:
        return sum(segment.duration for segment in self.segments)