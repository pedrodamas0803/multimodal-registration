"""
Interactive manual registration tool.

Displays middle cross-sections of the DCT and PCT volumes in a three-panel
figure (fixed | moving | overlay) and lets you adjust shifts and flips
interactively before committing the result to the DCT object.

**Jupyter notebooks** (recommended backend: ``%matplotlib widget``)
    Controls rendered as ipywidgets sliders / buttons beside the figure.

**Plain Python scripts**
    Controls via keyboard shortcuts on the matplotlib figure window.

Keyboard shortcuts (script mode)
---------------------------------
Arrow keys      shift in-plane (row / column)
w / s           shift out of plane (depth)
0 / 1 / 2      toggle flip along axis 0 / 1 / 2
Tab             cycle through xz → xy → yz planes
+ / -           increase / decrease step size
a               apply transforms to DCT object and close
q               close without applying
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import shift as _sp_shift

from .dct import DCT
from .pct import ReferencePCT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STEPS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# For each plane: (row_axis, col_axis, depth_axis)
_PLANE_AXES: dict[str, tuple[int, int, int]] = {
    "xz": (0, 2, 1),
    "xy": (0, 1, 2),
    "yz": (1, 2, 0),
}
_PLANES = ["xz", "xy", "yz"]


def _is_jupyter() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and hasattr(ip, "kernel")
    except ImportError:
        return False


def _shift_2d(arr: np.ndarray, dr: float, dc: float) -> np.ndarray:
    """Sub-pixel 2-D shift (row, col)."""
    if dr == 0.0 and dc == 0.0:
        return arr
    return _sp_shift(arr.astype(float), [dr, dc], order=1)


def _next_step(current: float, direction: int) -> float:
    try:
        idx = _STEPS.index(current)
    except ValueError:
        idx = _STEPS.index(1.0)
    return _STEPS[max(0, min(len(_STEPS) - 1, idx + direction))]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ManualRegistration:
    """Interactive cross-section overlay for manual registration.

    Parameters
    ----------
    dct:
        DCT object (already upscaled and padded to PCT frame).
    pct:
        PCT reference object.
    dct_field:
        Attribute on *dct* used as the moving image (default ``'Mask'``).
    pct_field:
        Attribute on *pct* used as the fixed image (default ``'mask'``).
    step:
        Initial shift step in voxels. Supports sub-pixel values.
    alpha:
        Initial opacity of the DCT overlay layer (0–1).
    """

    def __init__(
        self,
        dct: DCT,
        pct: ReferencePCT,
        dct_field: str = "Mask",
        pct_field: str = "mask",
        step: float = 1.0,
        alpha: float = 0.5,
    ):
        self.dct = dct
        self.pct = pct
        self._dct_vol = np.asarray(getattr(dct, dct_field), dtype=float)
        self._pct_vol = np.asarray(getattr(pct, pct_field), dtype=float)

        # Mutable state
        self.shifts: list[float] = [0.0, 0.0, 0.0]   # [ax0, ax1, ax2]
        self.flips: list[bool] = [False, False, False]
        self._plane_idx: int = 0
        self.step: float = step
        self.alpha: float = alpha

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @property
    def plane(self) -> str:
        return _PLANES[self._plane_idx]

    def _status_str(self) -> str:
        return (
            f"plane={self.plane}  step={self.step}  "
            f"shifts={[round(s, 3) for s in self.shifts]}  "
            f"flips={self.flips}"
        )

    def _get_slices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return *(pct_slice, dct_slice)* for the current plane and state."""
        dct = self._dct_vol
        pct = self._pct_vol

        # Apply flips to the DCT preview copy
        for axis, flipped in enumerate(self.flips):
            if flipped:
                dct = np.flip(dct, axis=axis)

        row_ax, col_ax, _ = _PLANE_AXES[self.plane]
        shape = dct.shape

        if self.plane == "xz":
            mid = shape[1] // 2
            pct_sl = pct[:, mid, :]
            dct_sl = dct[:, mid, :]
        elif self.plane == "xy":
            mid = shape[2] // 2
            pct_sl = pct[:, :, mid]
            dct_sl = dct[:, :, mid]
        else:  # yz
            mid = shape[0] // 2
            pct_sl = pct[mid, :, :]
            dct_sl = dct[mid, :, :]

        dct_sl = _shift_2d(dct_sl, self.shifts[row_ax], self.shifts[col_ax])

        return np.asarray(pct_sl, dtype=float), dct_sl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self) -> None:
        """Apply accumulated flips and shifts to the DCT object in-place."""
        for axis, flipped in enumerate(self.flips):
            if flipped:
                self.dct.flip(axis=axis)
        if any(s != 0.0 for s in self.shifts):
            self.dct.shift(tuple(self.shifts))  # type: ignore[arg-type]
        self.shifts = [0.0, 0.0, 0.0]
        self.flips = [False, False, False]

    def show(self) -> None:
        """Launch the interactive UI."""
        if _is_jupyter():
            self._show_jupyter()
        else:
            self._show_script()

    # ------------------------------------------------------------------
    # Jupyter UI
    # ------------------------------------------------------------------

    def _show_jupyter(self) -> None:
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as e:
            raise ImportError(
                "ipywidgets is required for Jupyter mode: pip install ipywidgets"
            ) from e

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
        fig.tight_layout()

        pct_sl, dct_sl = self._get_slices()
        im_pct    = axes[0].imshow(pct_sl, cmap="gray", origin="lower")
        im_dct    = axes[1].imshow(dct_sl, cmap="hot",  origin="lower")
        im_ov_bg  = axes[2].imshow(pct_sl, cmap="gray", origin="lower")
        im_ov_fg  = axes[2].imshow(dct_sl, cmap="hot",  origin="lower", alpha=self.alpha)
        for ax, title in zip(axes, ["PCT (fixed)", "DCT (moving)", "Overlay"]):
            ax.set_title(title)
            ax.axis("off")
        plt.show()

        # --- widgets -------------------------------------------------------
        style = {"description_width": "initial"}

        w_plane = widgets.Dropdown(options=_PLANES, value=self.plane,
                                   description="Plane:", style=style)
        w_step  = widgets.Dropdown(options=_STEPS, value=self.step,
                                   description="Step (vox):", style=style)
        w_alpha = widgets.FloatSlider(value=self.alpha, min=0.0, max=1.0, step=0.05,
                                      description="Overlay α:", style=style)

        w_d0 = widgets.FloatText(value=0.0, description="Δ axis 0:", step=self.step, style=style)
        w_d1 = widgets.FloatText(value=0.0, description="Δ axis 1:", step=self.step, style=style)
        w_d2 = widgets.FloatText(value=0.0, description="Δ axis 2:", step=self.step, style=style)

        w_flip0 = widgets.ToggleButton(value=False, description="Flip axis 0", button_style="warning")
        w_flip1 = widgets.ToggleButton(value=False, description="Flip axis 1", button_style="warning")
        w_flip2 = widgets.ToggleButton(value=False, description="Flip axis 2", button_style="warning")

        w_apply  = widgets.Button(description="Apply to DCT", button_style="success")
        w_status = widgets.Label(value=self._status_str())

        def _update(*_):
            self.shifts = [w_d0.value, w_d1.value, w_d2.value]
            self.flips  = [w_flip0.value, w_flip1.value, w_flip2.value]
            self._plane_idx = _PLANES.index(w_plane.value)
            self.step  = w_step.value
            self.alpha = w_alpha.value

            pct_sl, dct_sl = self._get_slices()
            for im, data in [(im_pct, pct_sl), (im_dct, dct_sl),
                             (im_ov_bg, pct_sl), (im_ov_fg, dct_sl)]:
                im.set_data(data)
                im.autoscale()
            im_ov_fg.set_alpha(self.alpha)
            fig.canvas.draw_idle()
            w_status.value = self._status_str()

        def _on_apply(_):
            self.apply()
            w_status.value = "Applied — shifts and flips written to DCT object."

        for w in [w_d0, w_d1, w_d2, w_flip0, w_flip1, w_flip2,
                  w_plane, w_step, w_alpha]:
            w.observe(_update, names="value")
        w_apply.on_click(_on_apply)

        display(widgets.VBox([
            widgets.HBox([w_plane, w_step, w_alpha]),
            widgets.HBox([w_d0, w_d1, w_d2]),
            widgets.HBox([w_flip0, w_flip1, w_flip2]),
            widgets.HBox([w_apply, w_status]),
        ]))

    # ------------------------------------------------------------------
    # Script UI (matplotlib keyboard events)
    # ------------------------------------------------------------------

    def _show_script(self) -> None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
        fig.subplots_adjust(top=0.82, bottom=0.12)

        pct_sl, dct_sl = self._get_slices()
        im_pct   = axes[0].imshow(pct_sl, cmap="gray", origin="lower")
        im_dct   = axes[1].imshow(dct_sl, cmap="hot",  origin="lower")
        im_ov_bg = axes[2].imshow(pct_sl, cmap="gray", origin="lower")
        im_ov_fg = axes[2].imshow(dct_sl, cmap="hot",  origin="lower", alpha=self.alpha)
        for ax, title in zip(axes, ["PCT (fixed)", "DCT (moving)", "Overlay"]):
            ax.set_title(title)
            ax.axis("off")

        help_lines = (
            "← → ↑ ↓ : shift in-plane   |   w / s : shift out-of-plane   |   "
            "0 / 1 / 2 : flip axis\n"
            "Tab : cycle plane   |   + / - : step size   |   "
            "a : apply & close   |   q : quit"
        )
        fig.text(0.5, 0.97, help_lines, ha="center", va="top",
                 fontsize=8, family="monospace", transform=fig.transFigure)

        status_txt = fig.text(0.5, 0.02, self._status_str(),
                              ha="center", va="bottom", fontsize=9, family="monospace")

        def _redraw():
            pct_sl, dct_sl = self._get_slices()
            for im, data in [(im_pct, pct_sl), (im_dct, dct_sl),
                             (im_ov_bg, pct_sl), (im_ov_fg, dct_sl)]:
                im.set_data(data)
                im.autoscale()
            im_ov_fg.set_alpha(self.alpha)
            status_txt.set_text(self._status_str())
            fig.canvas.draw_idle()

        def _on_key(event):
            if event.key is None:
                return

            row_ax, col_ax, dep_ax = _PLANE_AXES[self.plane]
            s = self.step

            if   event.key == "up":       self.shifts[row_ax] += s
            elif event.key == "down":     self.shifts[row_ax] -= s
            elif event.key == "right":    self.shifts[col_ax] += s
            elif event.key == "left":     self.shifts[col_ax] -= s
            elif event.key in ("w","W"):  self.shifts[dep_ax] += s
            elif event.key in ("s","S"):  self.shifts[dep_ax] -= s
            elif event.key == "0":        self.flips[0] = not self.flips[0]
            elif event.key == "1":        self.flips[1] = not self.flips[1]
            elif event.key == "2":        self.flips[2] = not self.flips[2]
            elif event.key == "tab":
                self._plane_idx = (self._plane_idx + 1) % len(_PLANES)
            elif event.key in ("+", "="):
                self.step = _next_step(self.step, +1)
            elif event.key == "-":
                self.step = _next_step(self.step, -1)
            elif event.key in ("a", "A"):
                self.apply()
                plt.close(fig)
                return
            elif event.key == "q":
                plt.close(fig)
                return
            else:
                return

            _redraw()

        fig.canvas.mpl_connect("key_press_event", _on_key)
        plt.show()
