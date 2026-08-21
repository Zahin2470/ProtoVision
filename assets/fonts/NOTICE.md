# Fonts bundled here

`Poppins-Regular.ttf`, `Poppins-Medium.ttf`, `Poppins-Bold.ttf`, and
`Poppins-Light.ttf` are bundled directly so the project works out of the box
— no manual asset copying needed.

**License:** Poppins is licensed under the SIL Open Font License 1.1, which
explicitly permits redistribution (including bundled with software). For the
canonical license text, see the Poppins project on Google Fonts or
https://scripts.sil.org/OFL — not reproduced here to avoid shipping a
possibly-stale copy.

**Missing weights:** only Light/Regular/Medium/Bold are included (that's
what's available as static Poppins weights via Google Fonts). If your
SignSense assets include additional weights (e.g. SemiBold, ExtraBold) and
you want ProtoVision to use them too, drop the `.ttf` files in here and
register them in `protovision/ui.py`'s `FONT_FILES` dict.
