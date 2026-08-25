# Audio bundled here

`sfx/enroll_success.wav`, `sfx/match_found.wav`, and `music/ambient_pad.wav`
are procedurally generated (plain sine-wave tones with fade envelopes and,
for `ambient_pad.wav`, a couple of detuned layers + slow amplitude vibrato)
— not sourced or licensed audio. They're deliberately simple placeholders so
`audio.py` has something real to load and play out of the box, not a
finished sound design. Swap any of them for better ones anytime; `audio.py`
just loads whatever's at these paths by filename.

Generated with plain numpy + Python's stdlib `wave` module (see the
generation script used during development — not shipped here since it's a
one-time tool, not a runtime dependency). All three fade to exact silence at
both ends, so there's no click at note boundaries or at the ambient loop
point.
