# Design

This service infers gender and age from a few seconds of call audio so a logistics voice agent can greet someone without a stored profile. The pipeline is decode → quality gate → wav2vec2 heads → JSON. ffmpeg converts compressed or telephony audio to 16 kHz mono over pipes, so samples never hit disk. Duration, RMS, clipping, and spectral flatness stop guesses on silence and mark warehouse-like noise as degraded rather than returning a silent miss.

I used audEERING’s 6-layer wav2vec2 age/gender checkpoint because it is public, jointly trained for both heads, and lighter than the 24-layer variant—important for a ~500 ms budget on a 5 s chunk. Age is a scalar mapped to product brackets; gender uses adult classes and becomes unknown when the child head dominates or confidence is low. Whisper tiny is only a bonus language hint and is allowed to fail closed.

With more time I would fine-tune on noisy handsets, add VAD so we score speech frames only, calibrate on Common Voice, and replace the NC-licensed net before a commercial ship.

A thousand concurrent calls should not share one process lock. Run CPU or GPU replicas behind a load balancer, cap queue depth, pin each WebSocket to a replica, and autoscale on latency. HTTP is stateless; the audio buffer dies with the request.
