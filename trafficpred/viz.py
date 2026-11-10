import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
from torch.utils.data import Dataset

from trafficpred.data import Splits, get_splits
from trafficpred.data.windows import base_datasets
from trafficpred.utils import predict


class Windows(Dataset):
    def __init__(self, frames, t_in, t_out):
        f = torch.as_tensor(np.asarray(frames), dtype=torch.float32)
        if f.ndim == 3:
            f = f.unsqueeze(1)
        assert f.ndim == 4, f"expected (N, C, H, W) or (N, H, W), got {tuple(f.shape)}"
        self.f, self.t_in, self.t_out = f, t_in, t_out

    def __len__(self):
        return max(len(self.f) - self.t_in - self.t_out + 1, 0)

    def __getitem__(self, i):
        j = i + self.t_in
        return self.f[i:j], self.f[j:j + self.t_out]


def _horizon(model):
    t_in = getattr(model, "t_in", getattr(model, "T_in", 4))
    t_out = getattr(model, "t_out", getattr(model, "T_out", 4))
    return t_in, t_out


def _scale_from(dataset):
    base = base_datasets(dataset)[0]
    if hasattr(base, "lo") and hasattr(base, "span"):
        return base.lo, base.lo + base.span
    return None


def _as_4d(t):
    t = torch.as_tensor(t)
    return t.unsqueeze(1) if t.ndim == 3 else t


class _View:
    def __init__(self, model, dataset=None, scale=None, channels=None, cmap="viridis", unit=None,
                 t_in=None, t_out=None):
        model_in, model_out = _horizon(model)
        t_in, t_out = t_in or model_in, t_out or model_out
        if dataset is None:
            dataset = get_splits("taxibj", t_in, t_out)
        if isinstance(dataset, Splits):
            channels = channels or dataset.channels
            scale = scale if scale is not None else (dataset.lo, dataset.hi)
            unit = unit or ("pixel value" if dataset.dataset == "mmnist" else "flow")
            dataset = dataset.test
        elif isinstance(dataset, (np.ndarray, torch.Tensor)):
            dataset = Windows(dataset, t_in, t_out)
        self.dataset = dataset
        self.n = len(dataset)
        if self.n == 0:
            raise ValueError("dataset is empty (too few frames for t_in + t_out?)")
        self.unscale = self._unscaler(scale, dataset)

        y0 = _as_4d(dataset[0][1])
        self.T, self.C, self.H, self.W = y0.shape
        channels = channels or [f"Channel {c}" for c in range(self.C)]
        assert len(channels) == self.C, f"{len(channels)} names for {self.C} channels"
        self.names = list(channels)
        self.cmap = cmap
        self.unit = unit or "value"
        self.model = model
        self._cache = {}

    @staticmethod
    def _unscaler(scale, dataset):
        if scale is False:
            return lambda t: t
        if callable(scale):
            return scale
        lohi = scale if scale is not None else _scale_from(dataset)
        if lohi is None:
            return lambda t: t
        lo, hi = lohi
        return lambda t: t * (hi - lo) + lo

    @torch.no_grad()
    def predict(self, idx, device=None):
        if device is None:
            p = next(self.model.parameters(), None)
            device = p.device if p is not None else "cpu"
        self.model.eval()
        item = self.dataset[idx]
        x, y = _as_4d(item[0]).float(), _as_4d(item[1]).float()
        ext = item[2].unsqueeze(0).to(device) if len(item) > 2 else None
        y_hat = predict(self.model, x.unsqueeze(0).to(device), ext).squeeze(0).cpu().reshape(y.shape)
        to_numpy = lambda t: np.asarray(self.unscale(t), dtype=np.float32)
        return to_numpy(x), to_numpy(y), to_numpy(y_hat)

    def get(self, i):
        if i not in self._cache:
            self._cache[i] = self.predict(i)
        return self._cache[i]


def _fmt(v):
    return f"{v:.3g}"


def _clear_ticks(axes):
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])


def _fill_grid(v, axes, y, y_hat, s, images=None, vlims=None):
    created = images is None
    images = images or []
    for c, name in enumerate(v.names):
        true, pred = y[s, c], y_hat[s, c]
        err = np.abs(pred - true)
        vmin, vmax = (vlims[c] if vlims is not None else
                      (min(true.min(), pred.min()), max(true.max(), pred.max())))
        emax = max(err.max(), 1e-12)
        if created:
            images.append([axes[c, 0].imshow(true, cmap=v.cmap, vmin=vmin, vmax=vmax),
                           axes[c, 1].imshow(pred, cmap=v.cmap, vmin=vmin, vmax=vmax),
                           axes[c, 2].imshow(err, cmap="Reds", vmin=0, vmax=emax)])
        else:
            row = images[c]
            for im, img in zip(row[:2], (true, pred)):
                im.set_data(img)
                im.set_clim(vmin, vmax)
            row[2].set_data(err)
            row[2].set_clim(0, emax)
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        axes[c, 0].set_title(f"Actual {name}")
        axes[c, 1].set_title(f"Predicted {name}")
        axes[c, 2].set_title(f"|Error|  (RMSE {_fmt(rmse)}, MAE {_fmt(err.mean())})")
    return images


def _colorbars(v, fig, axes, images):
    for c in range(v.C):
        fig.colorbar(images[c][0], ax=list(axes[c, :2]), shrink=0.85, label=v.unit)
        fig.colorbar(images[c][2], ax=axes[c, 2], shrink=0.85, label="abs error")


def browse(model, dataset=None, start=0, step=0, **opts):
    v = _View(model, dataset, **opts)
    n, T, C = v.n, v.T, v.C
    start, step = min(max(start, 0), n - 1), min(step, T - 1)

    fig_h = 2.0 + 4.0 * C
    ctrl = 2.0 / fig_h
    fig = plt.figure(figsize=(13, fig_h))
    gs = fig.add_gridspec(C, 3, left=0.03, right=0.95, top=1 - 0.6 / fig_h,
                          bottom=ctrl + 0.2 / fig_h, wspace=0.15, hspace=0.25)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(C)])
    _clear_ticks(axes)
    _, y0, y_hat0 = v.get(start)
    images = _fill_grid(v, axes, y0, y_hat0, step)
    _colorbars(v, fig, axes, images)

    row = lambda frac: frac * ctrl
    slider = Slider(fig.add_axes([0.12, row(0.62), 0.62, row(0.14)]), "Window", 0, n - 1,
                    valinit=start, valstep=1, valfmt="%d")
    b_prev = Button(fig.add_axes([0.12, row(0.16), 0.09, row(0.26)]), "< Prev")
    b_next = Button(fig.add_axes([0.22, row(0.16), 0.09, row(0.26)]), "Next >")
    b_rand = Button(fig.add_axes([0.32, row(0.16), 0.09, row(0.26)]), "Random")
    goto = TextBox(fig.add_axes([0.52, row(0.16), 0.08, row(0.26)]), "Go to ", initial="")
    radio = RadioButtons(fig.add_axes([0.82, row(0.07), 0.1, row(0.8)]),
                         [f"t+{t + 1}" for t in range(T)], active=step)
    fig.text(0.62, row(0.29), f"of {n - 1}", va="center")
    state = {"step": step}
    rng = np.random.default_rng()

    def render(_=None):
        i, s = int(slider.val), state["step"]
        _, y, y_hat = v.get(i)
        _fill_grid(v, axes, y, y_hat, s, images)
        fig.suptitle(f"Window {i} / {n - 1}, prediction step t+{s + 1}")
        fig.canvas.draw_idle()

    def go(i):
        slider.set_val(int(i) % n)

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
        if goto.capturekeystrokes:
            return
        if event.key == "right":
            go(slider.val + 1)
        elif event.key == "left":
            go(slider.val - 1)
        elif event.key in ("up", "down"):
            radio.set_active((state["step"] + (1 if event.key == "down" else -1)) % T)

    slider.on_changed(render)
    b_prev.on_clicked(lambda _: go(slider.val - 1))
    b_next.on_clicked(lambda _: go(slider.val + 1))
    b_rand.on_clicked(lambda _: go(rng.integers(n)))
    goto.on_submit(on_goto)
    radio.on_clicked(on_step)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig._browser_widgets = (slider, b_prev, b_next, b_rand, goto, radio)
    render()
    plt.show()
    return fig


def _input_strip(v, top, x, vlims):
    T_in = x.shape[0]
    axes = np.atleast_2d(top.subplots(v.C, T_in, squeeze=False))
    for c, name in enumerate(v.names):
        for t in range(T_in):
            im = axes[c, t].imshow(x[t, c], cmap=v.cmap, vmin=vlims[c][0], vmax=vlims[c][1])
            lag = T_in - 1 - t
            if c == 0:
                axes[c, t].set_title("t" if lag == 0 else f"t-{lag}", fontsize=10)
        axes[c, 0].set_ylabel(name, fontsize=10)
        top.colorbar(im, ax=list(axes[c, :]), shrink=0.9, label=v.unit)
    _clear_ticks(axes)
    top.suptitle(f"Input ({T_in} frames)", fontsize=12)


def _grid_figure(v, y, y_hat, s, title, x=None):
    from matplotlib.figure import Figure
    C, grid_h = v.C, 4.0 * v.C
    if x is None:
        fig = Figure(figsize=(13, grid_h), constrained_layout=True)
        bottom, vlims = fig, None
    else:
        in_h = 0.6 + min(2.6, 14.0 / x.shape[0]) * C
        fig = Figure(figsize=(16, in_h + grid_h + 0.4), constrained_layout=True)
        top, bottom = fig.subfigures(2, 1, height_ratios=[in_h, grid_h])
        vlims = [(min(x[:, c].min(), y[s, c].min(), y_hat[s, c].min()),
                  max(x[:, c].max(), y[s, c].max(), y_hat[s, c].max())) for c in range(C)]
        _input_strip(v, top, x, vlims)

    axes = np.atleast_2d(bottom.subplots(C, 3 if x is None else 4, squeeze=False))
    images = _fill_grid(v, axes, y, y_hat, s, vlims=vlims)
    if x is None:
        _colorbars(v, bottom, axes, images)
    else:
        for c in range(C):
            moved = np.abs(y_hat[s, c] - x[-1, c])
            err = np.abs(y_hat[s, c] - y[s, c])
            emax = max(moved.max(), err.max(), 1e-12)
            images[c][2].set_clim(0, emax)
            axes[c, 3].imshow(moved, cmap="Reds", vmin=0, vmax=emax)
            axes[c, 3].set_title(f"|Predicted − input t|  (MAE {_fmt(moved.mean())})")
            bottom.colorbar(images[c][0], ax=list(axes[c, :2]), shrink=0.85, label=v.unit)
            bottom.colorbar(images[c][2], ax=list(axes[c, 2:]), shrink=0.85, label="abs difference")
        bottom.suptitle(f"Prediction step t+{s + 1}", fontsize=12)
    _clear_ticks(axes)
    fig.suptitle(title, fontsize=14)
    return fig


def browse_notebook(model, dataset=None, start=0, step=0, dpi=80, show_input=True, **opts):
    import ipywidgets as w
    from IPython.display import display

    v = _View(model, dataset, **opts)
    n, T = v.n, v.T
    image = w.Image(format="png", layout=w.Layout(width="100%", max_width="1000px"))
    slider = w.IntSlider(value=min(max(start, 0), n - 1), min=0, max=n - 1, description="Window",
                         continuous_update=False, layout=w.Layout(width="60%"))
    b_prev, b_next, b_rand = w.Button(description="◀ Prev"), w.Button(description="Next ▶"), \
        w.Button(description="Random")
    goto = w.BoundedIntText(value=slider.value, min=0, max=n - 1, description="Go to",
                            layout=w.Layout(width="160px"))
    steps = w.ToggleButtons(options=[(f"t+{t + 1}", t) for t in range(T)], value=min(step, T - 1),
                            description="Step")
    show_in = w.Checkbox(value=show_input, description="Show input", indent=False)
    rng = np.random.default_rng()

    def render(*_):
        i, s = slider.value, steps.value
        x, y, y_hat = v.get(i)
        fig = _grid_figure(v, y, y_hat, s, f"Window {i} / {n - 1}", x=x if show_in.value else None)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        image.value = buf.getvalue()
        goto.value = i

    def go(i):
        slider.value = int(i) % n

    for widget in (slider, steps, show_in):
        widget.observe(render, names="value")
    b_prev.on_click(lambda _: go(slider.value - 1))
    b_next.on_click(lambda _: go(slider.value + 1))
    b_rand.on_click(lambda _: go(rng.integers(n)))
    goto.observe(lambda ch: go(ch["new"]) if ch["new"] != slider.value else None, names="value")
    render()
    display(w.VBox([image, w.HBox([b_prev, b_next, b_rand, goto, show_in]), slider, steps]))


def predict(model, dataset=None, idx=0, device=None, **opts):
    return _View(model, dataset, **opts).predict(idx, device)


def plot_prediction(model, dataset=None, idx=0, step=0, save=None, **opts):
    v = _View(model, dataset, **opts)
    _, y, y_hat = v.predict(idx)
    fig, axes = plt.subplots(v.C, 3, figsize=(13, 4.0 * v.C), constrained_layout=True, squeeze=False)
    images = _fill_grid(v, axes, y, y_hat, step)
    _colorbars(v, fig, axes, images)
    _clear_ticks(axes)
    fig.suptitle(f"Window {idx}, prediction step t+{step + 1}")
    if save:
        fig.savefig(save, dpi=150)
    plt.show()
    return fig


def plot_horizon(model, dataset=None, idx=0, channel=0, save=None, **opts):
    v = _View(model, dataset, **opts)
    _, y, y_hat = v.predict(idx)
    T = y.shape[0]
    true, pred = y[:, channel], y_hat[:, channel]
    err = np.abs(pred - true)
    vmin, vmax = min(true.min(), pred.min()), max(true.max(), pred.max())

    fig, axes = plt.subplots(3, T, figsize=(3.2 * T, 9.5), constrained_layout=True, squeeze=False)
    for t in range(T):
        im_value = axes[0, t].imshow(true[t], cmap=v.cmap, vmin=vmin, vmax=vmax)
        axes[1, t].imshow(pred[t], cmap=v.cmap, vmin=vmin, vmax=vmax)
        im_error = axes[2, t].imshow(err[t], cmap="Reds", vmin=0, vmax=max(err.max(), 1e-12))
        axes[0, t].set_title(f"t+{t + 1}")
        axes[2, t].set_xlabel(f"MAE {_fmt(err[t].mean())}")
    for r, label in enumerate(["Actual", "Predicted", "|Error|"]):
        axes[r, 0].set_ylabel(label, fontsize=12)
    _clear_ticks(axes)
    fig.colorbar(im_value, ax=axes[:2, :], shrink=0.8, label=v.unit)
    fig.colorbar(im_error, ax=axes[2, :], shrink=0.8, label="abs error")
    fig.suptitle(f"{v.names[channel]} across the horizon (window {idx})")
    if save:
        fig.savefig(save, dpi=150)
    plt.show()
    return fig
