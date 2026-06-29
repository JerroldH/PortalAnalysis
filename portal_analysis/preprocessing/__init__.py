__all__ = [
    "HandPoseExtractor",
    "DistanceCalculator",
    "HandMovementAnglesProcessor",
    "HandTremorProcessor",
    "TapTrimmer",
]


def __getattr__(name):
    if name == "HandPoseExtractor":
        from .hand_pose import HandPoseExtractor

        return HandPoseExtractor
    if name == "DistanceCalculator":
        from .distances import DistanceCalculator

        return DistanceCalculator
    if name == "HandMovementAnglesProcessor":
        from .hand_movement_angles import HandMovementAnglesProcessor

        return HandMovementAnglesProcessor
    if name == "HandTremorProcessor":
        from .hand_tremor import HandTremorProcessor

        return HandTremorProcessor
    if name == "TapTrimmer":
        from .tap_trimmer import TapTrimmer

        return TapTrimmer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
