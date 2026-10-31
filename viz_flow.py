"""
Side-by-side heatmaps of predicted vs. actual frames, for any spatiotemporal dataset.

Works with anything that yields (x, y) windows shaped (T, C, H, W), for any number of
channels and any grid size: TaxiBJ, Moving MNIST, your own arrays. The model maps
x (B, t_in, C, H, W) -> y_hat (B, t_out, C, H, W).

Usage:
    from viz_flow import browse, browse_notebook, plot_prediction, plot_horizon, windows

    browse_notebook(model)                                  # TaxiBJ test set (default)
    browse_notebook(model, dataset=my_test_ds)              # any (x, y) dataset
    browse_notebook(model, dataset=frames, t_in=10, t_out=10)   # raw (N, C, H, W) array
    plot_prediction(model, dataset=mnist_test, idx=3, channels=["Digits"], cmap="gray")
    plot_horizon(model, idx=100, channel=0)

Options every function accepts:
    dataset   Dataset/sequence of (x, y); or a raw frame array (N, C, H, W) or (N, H, W),
              which is cut into sliding windows using t_in / t_out.
              None -> TaxiBJ test set from data.py, matching the model's horizon.
    scale     How to undo the data's scaling for display:
                None      -> read (lo, hi) from the dataset if it has it
                             (FlowWindowDataset does); otherwise show values as-is
                (lo, hi)  -> value * (hi - lo) + lo
                callable  -> applied to every tensor
                False     -> never unscale
    channels  Names for the channels, e.g. ["Inflow", "Outflow"]. Default: TaxiBJ names
              for the default dataset, "Channel 0", "Channel 1", ... otherwise.
    cmap      Colormap for actual / predicted (default "viridis").
    unit      Colorbar label (default "flow" for TaxiBJ, "value" otherwise).

browse() needs an interactive matplotlib backend (run it as a script).
In a notebook use browse_notebook(), which only needs `pip install ipywidgets`.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons, TextBox
from torch.utils.data import Dataset, ConcatDataset, Subset

TAXIBJ_CHANNELS = ["Inflow", "Outflow"]


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------
class windows(Dataset):
    """Sliding (x, y) windows over a raw frame array (N, C, H, W) or (N, H, W).
    Values are shown as given (no rescaling)."""

    def __init__(self, frames, t_in, t_out):
        f = torch.as_tensor(np.asarray(frames), dtype=torch.float32)
        if f.ndim == 3:
            f = f.unsqueeze(1)                       # (N, H, W) -> (N, 1, H, W)
        assert f.ndim == 4, f"expected (N, C, H, W) or (N, H, W), got {tuple(f.shape)}"
        self.f, self.t_in, self.t_out = f, t_in, t_out

    def __len__(self):
        return max(len(self.f) - self.t_in - self.t_out + 1, 0)

    def __getitem__(self, i):
        j = i + self.t_in
        return self.f[i:j], self.f[j:j + self.t_out]


def _horizon(model):
    """(t_in, t_out) from the model. Checks t_in/t_out (baselines, run_viz) and
    T_in/T_out (DistVolNet); falls back to 4 -> 4."""
    t_in = getattr(model, "t_in", getattr(model, "T_in", 4))
    t_out = getattr(model, "t_out", getattr(model, "T_out", 4))
    return t_in, t_out


def _scale_from(dataset):
    """Find (lo, hi) on a FlowWindowDataset, or inside a ConcatDataset / Subset of them."""
    if hasattr(dataset, "lo") and hasattr(dataset, "span"):
        return dataset.lo, dataset.lo + dataset.span
    if isinstance(dataset, ConcatDataset) and dataset.datasets:
        return _scale_from(dataset.datasets[0])
    if isinstance(dataset, Subset):
        return _scale_from(dataset.dataset)
    return None


def _as_4d(t):
    """Accept (T, C, H, W) or (T, H, W); return (T, C, H, W)."""
    t = torch.as_tensor(t)
    return t.unsqueeze(1) if t.ndim == 3 else t


class _View:
    """Everything the plotting functions need, resolved once."""

    def __init__(self, model, dataset=None, scale=None, channels=None, cmap="viridis",
                 unit=None, t_in=None, t_out=None):
        mi, mo = _horizon(model)
        t_in, t_out = t_in or mi, t_out or mo
        taxibj = dataset is None

        if taxibj:
            from data import taxibj_splits             # only needed for the default
            _, dataset = taxibj_splits(t_in, t_out)
        elif isinstance(dataset, (np.ndarray, torch.Tensor)):
            dataset = windows(dataset, t_in, t_out)
        self.dataset = dataset
        self.n = len(dataset)
        if self.n == 0:
            raise ValueError("dataset is empty (too few frames for t_in + t_out?)")

        # Scaling
        if scale is False:
            self.unscale = lambda t: t
        elif callable(scale):
            self.unscale = scale
        else:
            lohi = scale if scale is not None else _scale_from(dataset)
            if lohi is None:
                self.unscale = lambda t: t
            else:
                lo, hi = lohi
                self.unscale = lambda t: t * (hi - lo) + lo

        x0, y0 = dataset[0][:2]
        y0 = _as_4d(y0)
        self.T, self.C, self.H, self.W = y0.shape
        if channels is None:
            channels = (TAXIBJ_CHANNELS if self.C == 2 and taxibj
                        else [f"Channel {c}" for c in range(self.C)])
        assert len(channels) == self.C, f"{len(channels)} names for {self.C} channels"
        self.names = list(channels)
        self.cmap = cmap
        self.unit = unit or ("flow" if taxibj else "value")
        self.model = model
        self._cache = {}

    @torch.no_grad()
    def predict(self, idx, device=None):
        """(x, y_true, y_pred) for one window, as numpy (T, C, H, W), unscaled."""
        model = self.model
        if device is None:
            p = next(model.parameters(), None)
            device = p.device if p is not None else "cpu"
        model.eval()
        x, y = self.dataset[idx][:2]
        x, y = _as_4d(x).float(), _as_4d(y).float()
        out = model(x.unsqueeze(0).to(device))
        if isinstance(out, (tuple, list)):              # e.g. return_parts=True models
            out = out[0]
        y_hat = out.squeeze(0).cpu().reshape(y.shape)
        f = lambda t: np.asarray(self.unscale(t), dtype=np.float32)
        return f(x), f(y), f(y_hat)

    def get(self, i):
        if i not in self._cache:
            self._cache[i] = self.predict(i)            # (x, y, y_hat)
        return self._cache[i]


def _fmt(v):
    return f"{v:.3g}"


def _fill_grid(v, axes, y, y_hat, s, images=None, vlims=None):
    """Draw / update the C x 3 grid (actual | predicted | |error|) for step s.
    With images=None it creates new images and returns them.
    vlims: optional per-channel (vmin, vmax) for actual / predicted."""
    created = images is None
    images = images or []
    for c, name in enumerate(v.names):
        true, pred = y[s, c], y_hat[s, c]
        err = np.abs(pred - true)
        vmin, vmax = (vlims[c] if vlims is not None else
                      (min(true.min(), pred.min()), max(true.max(), pred.max())))
        emax = max(err.max(), 1e-12)
        if created:
            row = [axes[c, 0].imshow(true, cmap=v.cmap, vmin=vmin, vmax=vmax),
                   axes[c, 1].imshow(pred, cmap=v.cmap, vmin=vmin, vmax=vmax),
                   axes[c, 2].imshow(err, cmap="Reds", vmin=0, vmax=emax)]
            images.append(row)
        else:
            row = images[c]
            for im, img in zip(row[:2], (true, pred)):
                im.set_data(img); im.set_clim(vmin, vmax)
            row[2].set_data(err); row[2].set_clim(0, emax)
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        axes[c, 0].set_title(f"Actual {name}")
        axes[c, 1].set_title(f"Predicted {name}")
        axes[c, 2].set_title(f"|Error|  (RMSE {_fmt(rmse)}, MAE {_fmt(err.mean())})")
    return images


def _colorbars(v, fig, axes, images):
    for c in range(v.C):
        fig.colorbar(images[c][0], ax=list(axes[c, :2]), shrink=0.85, label=v.unit)
        fig.colorbar(images[c][2], ax=axes[c, 2], shrink=0.85, label="abs error")


# ---------------------------------------------------------------------------
# Interactive browser
# ---------------------------------------------------------------------------
def browse(model, dataset=None, start=0, step=0, **opts):
    """Interactive C x 3 viewer (rows = channels; columns = actual | predicted | |error|)
    over every window in the dataset.

    Controls:
        Prev / Next buttons, or Left / Right arrow keys  -> previous / next window
        Random button                                    -> jump to a random window
        Slider                                           -> scrub through all windows
        "Go to" box                                      -> type an index, press Enter
        t+1 ... t+T selector, or Up / Down arrow keys    -> which future step to show
    """
    v = _View(model, dataset, **opts)
    n, T, C = v.n, v.T, v.C

    # plots on top, controls in a fixed-height strip along the bottom
    fig_h = 2.0 + 4.0 * C
    ctrl = 2.0 / fig_h                                  # fraction used by controls
    fig = plt.figure(figsize=(13, fig_h))
    gs = fig.add_gridspec(C, 3, left=0.03, right=0.95, top=1 - 0.6 / fig_h,
                          bottom=ctrl + 0.2 / fig_h, wspace=0.15, hspace=0.25)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(C)])
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    _, y0, yh0 = v.get(min(max(start, 0), n - 1))
    images = _fill_grid(v, axes, y0, yh0, min(step, T - 1))
    _colorbars(v, fig, axes, images)

    row = lambda frac: frac * ctrl                      # position inside control strip
    slider = Slider(fig.add_axes([0.12, row(0.62), 0.62, row(0.14)]), "Window", 0, n - 1,
                    valinit=min(max(start, 0), n - 1), valstep=1, valfmt="%d")
    b_prev = Button(fig.add_axes([0.12, row(0.16), 0.09, row(0.26)]), "< Prev")
    b_next = Button(fig.add_axes([0.22, row(0.16), 0.09, row(0.26)]), "Next >")
    b_rand = Button(fig.add_axes([0.32, row(0.16), 0.09, row(0.26)]), "Random")
    goto = TextBox(fig.add_axes([0.52, row(0.16), 0.08, row(0.26)]), "Go to ", initial="")
    radio = RadioButtons(fig.add_axes([0.82, row(0.07), 0.1, row(0.8)]),
                         [f"t+{t + 1}" for t in range(T)], active=min(step, T - 1))
    fig.text(0.62, row(0.29), f"of {n - 1}", va="center")
    state = {"step": min(step, T - 1)}
    rng = np.random.default_rng()

    def render(_=None):
        i, s = int(slider.val), state["step"]
        _, y, y_hat = v.get(i)
        _fill_grid(v, axes, y, y_hat, s, images)
        fig.suptitle(f"Window {i} / {n - 1}, prediction step t+{s + 1}")
        fig.canvas.draw_idle()

    def go(i):
        slider.set_val(int(i) % n)                      # triggers render via on_changed

    def on_step(label):
        state["step"] = int(label[2:]) - 1
        render()

    def on_goto(text):
        try:
            go(int(text))
        except ValueError:
            pass
        goto.set_val("")

    def on_key(event):
        if goto.capturekeystrokes:                      # typing in the Go-to box
            return
        if event.key == "right":
            go(slider.val + 1)
        elif event.key == "left":
            go(slider.val - 1)
        elif event.key in ("up", "down"):
            s = (state["step"] + (1 if event.key == "down" else -1)) % T
            radio.set_active(s)                         # triggers on_step

    slider.on_changed(render)
    b_prev.on_clicked(lambda _: go(slider.val - 1))
    b_next.on_clicked(lambda _: go(slider.val + 1))
    b_rand.on_clicked(lambda _: go(rng.integers(n)))
    goto.on_submit(on_goto)
    radio.on_clicked(on_step)
    fig.canvas.mpl_connect("key_press_event", on_key)

    # widgets stop responding if they get garbage-collected, so keep references
    fig._browser_widgets = (slider, b_prev, b_next, b_rand, goto, radio)
    render()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Notebook browser (ipywidgets; works with the default inline backend)
# ---------------------------------------------------------------------------
def _grid_figure(v, y, y_hat, s, title, x=None):
    """Build the figure without touching pyplot, so it renders the same under any
    backend. Bottom: C x 3 grid (actual | predicted | |error|) for step s.
    With x given, a top strip shows the input frames (rows = channels,
    columns = t-(T_in-1) ... t), and the bottom grid gains a 4th column,
    |predicted - input t|, on the same scale as |error|. Inputs, actual and
    predicted share one color scale per channel so they can be compared directly."""
    from matplotlib.figure import Figure
    C = v.C
    grid_h = 4.0 * C
    if x is None:
        fig = Figure(figsize=(13, grid_h), constrained_layout=True)
        bottom = fig
        vlims = None
    else:
        T_in = x.shape[0]
        cell = min(2.6, 14.0 / T_in)                    # input frame size, inches
        in_h = 0.6 + cell * C
        fig = Figure(figsize=(16, in_h + grid_h + 0.4), constrained_layout=True)
        top, bottom = fig.subfigures(2, 1, height_ratios=[in_h, grid_h])
        vlims = [(min(x[:, c].min(), y[s, c].min(), y_hat[s, c].min()),
                  max(x[:, c].max(), y[s, c].max(), y_hat[s, c].max())) for c in range(C)]

        in_axes = np.atleast_2d(top.subplots(C, T_in, squeeze=False))
        for c, name in enumerate(v.names):
            for t in range(T_in):
                im = in_axes[c, t].imshow(x[t, c], cmap=v.cmap,
                                          vmin=vlims[c][0], vmax=vlims[c][1])
                k = T_in - 1 - t
                if c == 0:
                    in_axes[c, t].set_title("t" if k == 0 else f"t-{k}", fontsize=10)
            in_axes[c, 0].set_ylabel(name, fontsize=10)
            top.colorbar(im, ax=list(in_axes[c, :]), shrink=0.9, label=v.unit)
        for ax in in_axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        top.suptitle(f"Input ({T_in} frames)", fontsize=12)

    ncols = 3 if x is None else 4
    axes = np.atleast_2d(bottom.subplots(C, ncols, squeeze=False))
    images = _fill_grid(v, axes, y, y_hat, s, vlims=vlims)
    if x is None:
        _colorbars(v, bottom, axes, images)
    else:
        # 4th column: how far the prediction moved away from the last input frame.
        # Shares a color scale with |error| so the two are directly comparable.
        for c in range(C):
            diff = np.abs(y_hat[s, c] - x[-1, c])
            err = np.abs(y_hat[s, c] - y[s, c])
            emax = max(diff.max(), err.max(), 1e-12)
            images[c][2].set_clim(0, emax)
            axes[c, 3].imshow(diff, cmap="Reds", vmin=0, vmax=emax)
            axes[c, 3].set_title(f"|Predicted − input t|  (MAE {_fmt(diff.mean())})")
            bottom.colorbar(images[c][0], ax=list(axes[c, :2]), shrink=0.85, label=v.unit)
            bottom.colorbar(images[c][2], ax=list(axes[c, 2:]), shrink=0.85, label="abs difference")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    if x is not None:
        bottom.suptitle(f"Prediction step t+{s + 1}", fontsize=12)
    fig.suptitle(title, fontsize=14)
    return fig


def browse_notebook(model, dataset=None, start=0, step=0, dpi=80, show_input=True, **opts):
    """Jupyter version of browse(): Prev / Next / Random buttons, a window
    slider, a Go-to box, t+1..t+T step buttons and a "Show input" checkbox,
    built with ipywidgets. With the input shown, the input frames appear above
    the actual / predicted / |error| grid, on the same color scale.
    Needs `pip install ipywidgets`; no special matplotlib backend required."""
    import io
    import ipywidgets as w
    from IPython.display import display

    v = _View(model, dataset, **opts)
    n, T = v.n, v.T

    image = w.Image(format="png", layout=w.Layout(width="100%", max_width="1000px"))
    slider = w.IntSlider(value=min(max(start, 0), n - 1), min=0, max=n - 1,
                         description="Window", continuous_update=False,
                         layout=w.Layout(width="60%"))
    b_prev = w.Button(description="◀ Prev")
    b_next = w.Button(description="Next ▶")
    b_rand = w.Button(description="Random")
    goto = w.BoundedIntText(value=slider.value, min=0, max=n - 1,
                            description="Go to", layout=w.Layout(width="160px"))
    steps = w.ToggleButtons(options=[(f"t+{t + 1}", t) for t in range(T)],
                            value=min(step, T - 1), description="Step")
    show_in = w.Checkbox(value=show_input, description="Show input", indent=False)
    rng = np.random.default_rng()

    def render(*_):
        i, s = slider.value, steps.value
        x, y, y_hat = v.get(i)
        fig = _grid_figure(v, y, y_hat, s, f"Window {i} / {n - 1}",
                           x=x if show_in.value else None)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        image.value = buf.getvalue()
        goto.value = i

    def go(i):
        slider.value = int(i) % n                       # triggers render

    slider.observe(render, names="value")
    steps.observe(render, names="value")
    show_in.observe(render, names="value")
    b_prev.on_click(lambda _: go(slider.value - 1))
    b_next.on_click(lambda _: go(slider.value + 1))
    b_rand.on_click(lambda _: go(rng.integers(n)))
    goto.observe(lambda ch: go(ch["new"]) if ch["new"] != slider.value else None, names="value")

    render()
    display(w.VBox([image, w.HBox([b_prev, b_next, b_rand, goto, show_in]), slider, steps]))


# ---------------------------------------------------------------------------
# Static plots
# ---------------------------------------------------------------------------
def predict(model, dataset=None, idx=0, device=None, **opts):
    """Run the model on one window. Returns (x, y_true, y_pred) as numpy arrays,
    each (T, C, H, W), with the dataset's scaling undone."""
    return _View(model, dataset, **opts).predict(idx, device)


def plot_prediction(model, dataset=None, idx=0, step=0, save=None, **opts):
    """C x 3 grid: rows = channels; columns = actual | predicted | |error|.
    Actual and predicted share a color scale so they're directly comparable."""
    v = _View(model, dataset, **opts)
    _, y, y_hat = v.predict(idx)
    fig, axes = plt.subplots(v.C, 3, figsize=(13, 4.0 * v.C),
                             constrained_layout=True, squeeze=False)
    images = _fill_grid(v, axes, y, y_hat, step)
    _colorbars(v, fig, axes, images)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Window {idx}, prediction step t+{step + 1}")
    if save:
        fig.savefig(save, dpi=150)
    plt.show()
    return fig


def plot_horizon(model, dataset=None, idx=0, channel=0, save=None, **opts):
    """Rows = actual / predicted / |error|; columns = each future step.
    Shows how error grows across the prediction horizon for one channel."""
    v = _View(model, dataset, **opts)
    _, y, y_hat = v.predict(idx)
    T = y.shape[0]
    true, pred = y[:, channel], y_hat[:, channel]
    err = np.abs(pred - true)
    vmin, vmax = min(true.min(), pred.min()), max(true.max(), pred.max())

    fig, axes = plt.subplots(3, T, figsize=(3.2 * T, 9.5),
                             constrained_layout=True, squeeze=False)
    for t in range(T):
        im_v = axes[0, t].imshow(true[t], cmap=v.cmap, vmin=vmin, vmax=vmax)
        axes[1, t].imshow(pred[t], cmap=v.cmap, vmin=vmin, vmax=vmax)
        im_e = axes[2, t].imshow(err[t], cmap="Reds", vmin=0, vmax=max(err.max(), 1e-12))
        axes[0, t].set_title(f"t+{t + 1}")
        axes[2, t].set_xlabel(f"MAE {_fmt(err[t].mean())}")
    for r, label in enumerate(["Actual", "Predicted", "|Error|"]):
        axes[r, 0].set_ylabel(label, fontsize=12)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    fig.colorbar(im_v, ax=axes[:2, :], shrink=0.8, label=v.unit)
    fig.colorbar(im_e, ax=axes[2, :], shrink=0.8, label="abs error")
    fig.suptitle(f"{v.names[channel]} across the horizon (window {idx})")
    if save:
        fig.savefig(save, dpi=150)
    plt.show()
    return fig


if __name__ == "__main__":
    # Demo with a "persistence" baseline: repeat the last input frame.
    class Persistence(torch.nn.Module):
        def __init__(self, t_out=4):
            super().__init__()
            self.t_out = t_out

        def forward(self, x):
            return x[:, -1:].repeat(1, self.t_out, 1, 1, 1)

    browse(Persistence())