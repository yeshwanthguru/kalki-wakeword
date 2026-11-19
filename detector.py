"""Compatibility shim: the detector now lives in the installable package `kalki_wakeword`.

    from detector import WakeWordDetector      # still works from a checkout
    from kalki_wakeword import WakeWordDetector  # preferred (pip install .)
"""
from kalki_wakeword.detector import DEFAULT_MODEL, HOP, SR, WIN, Detection, WakeWordDetector  # noqa: F401
