# Interactive Registration

`ManualRegistration` provides a three-panel overlay (fixed | moving | overlay)
for visual fine-tuning of the alignment.  It works in both Jupyter notebooks
and plain Python scripts — the correct UI is selected automatically.

---

## Basic usage

```python
from multimodal_registration import ManualRegistration

mr = ManualRegistration(dct, pct, step=1.0, alpha=0.5)
mr.show()

# After closing the UI, commit the result:
mr.apply()
```

`apply()` writes the accumulated shifts and flips back to the `dct` object.
Nothing is modified until you call it explicitly.

---

## Jupyter notebooks

!!! tip "Recommended backend"
    Run `%matplotlib widget` before importing matplotlib for live figure updates.
    Requires `ipympl`: `pip install "multimodal-registration[jupyter]"`.

Controls are rendered as ipywidgets beside the figure:

| Widget | Effect |
|--------|--------|
| **Plane** dropdown | Switch between xz / xy / yz cross-sections |
| **Step** dropdown | Set shift increment (supports sub-pixel values) |
| **Overlay α** slider | Adjust DCT layer opacity |
| **Δ axis 0/1/2** text boxes | Enter cumulative shift per axis directly |
| **Flip axis 0/1/2** toggles | Toggle mirror along each axis |
| **Apply to DCT** button | Write result to the DCT object |

---

## Script / terminal

The figure window captures keyboard events:

| Key | Action |
|-----|--------|
| `←` `→` `↑` `↓` | Shift in-plane (column / row direction) |
| `w` / `s` | Shift out of plane (depth) |
| `0` / `1` / `2` | Toggle flip along axis 0 / 1 / 2 |
| `Tab` | Cycle plane: xz → xy → yz |
| `+` / `-` | Increase / decrease step size |
| `a` | Apply transforms and close |
| `q` | Close without applying |

The current state (plane, step, cumulative shifts, flips) is displayed at the
bottom of the figure.

---

## Step sizes

Available step sizes cycle through:
`0.01 → 0.05 → 0.1 → 0.5 → 1.0 → 2.0 → 5.0 → 10.0` voxels.

Sub-pixel shifts are applied to the **preview** using `scipy.ndimage.shift`
(order-1, bilinear) and passed directly to `dct.shift()` on apply.

---

## Choosing fields

By default the overlay uses the binary masks (`dct.Mask` and `pct.mask`).
Any other attribute can be used:

```python
mr = ManualRegistration(dct, pct,
                         dct_field="GIDvol",
                         pct_field="vol")
```

---

## Important notes

- **The depth shift (w/s)** is tracked but not visible in the 2-D preview.
  It is correctly passed to `dct.shift()` on apply.
- **Flips accumulate**: toggling the same axis twice returns to the original
  state.  On apply, the actual `dct.flip()` is called once per active flip.
- **Nothing is committed** to the DCT object until `mr.apply()` is called.
