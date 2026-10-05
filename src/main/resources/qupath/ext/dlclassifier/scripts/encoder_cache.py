"""
Pre-download the pretrained encoder weights, and report what has gone stale.

Every pretrained encoder is fetched from the HuggingFace Hub on first use and
cached outside the Appose environment, so a rebuilt environment does not carry
the weights with it and the first training run needs the network.

Inputs:
    action: str - "prewarm" (default) or "status"

Outputs:
    success: bool
    message: str
    stale: int - repositories whose weights have been superseded
"""

import logging

logger = logging.getLogger("dlclassifier.appose.encoder_cache")

action = globals().get("action", "prewarm")

try:
    from dlclassifier_server.services.encoder_cache import (
        OFFERED_ENCODERS,
        STALE,
        prewarm,
        status,
    )

    rows = status()
    stale = [r for r in rows if r.state == STALE]

    if action == "status":
        task.outputs["message"] = (
            "%d repositories cached, %d with newer weights published."
            % (
                len(rows),
                len(stale),
            )
        )
    else:
        done = prewarm()
        missing = [e for e in OFFERED_ENCODERS if e not in done]
        if missing:
            task.outputs["message"] = (
                "%d of %d encoders are ready to use offline. Could not fetch: %s."
                % (len(done), len(OFFERED_ENCODERS), ", ".join(missing))
            )
        else:
            task.outputs["message"] = (
                "All %d encoders are ready to use offline. %d repositories cached, %d with newer weights published."
                % (len(done), len(rows), len(stale))
            )

    task.outputs["success"] = True
    task.outputs["stale"] = len(stale)

except Exception as e:
    logger.warning("Encoder cache %s failed: %s", action, e)
    task.outputs["success"] = False
    task.outputs["message"] = "Error: %s" % str(e)
    task.outputs["stale"] = 0
