"""
Copyright (c) 2024 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

preamble = r"\usepackage{times} "
JMLR_PARAMS = {
    "text.usetex": True,
    "font.size": 10.95,
    "axes.titlesize": 10.95,
    "axes.labelsize": 10.95,
    "xtick.labelsize": 10.95,
    "ytick.labelsize": 10.95,
    "legend.fontsize": 10.95,
    "figure.titlesize": 10.95,
    "font.family": "serif",
    "text.latex.preamble": "",
}


LIGHT_GREY = [211 / 255, 211 / 255, 211 / 255]


def default_process_label(params):
    label = ""
    for param in params:
        if str(param) != "nan":
            label += rf"\texttt{{ {param}}}"
    return label


def plot_with_symmetric_intervals(
    ax,
    x,
    y,
    yerr,
    color,
    label,
    ls="-",
    title="",
    face_color="w",
    grid_color=LIGHT_GREY,
):
    plot_with_intervals(
        ax, x, y, y - yerr, y + yerr, color, label, ls, title, face_color, grid_color
    )


def plot_with_intervals(
    ax,
    x,
    y,
    ylow,
    yhigh,
    color,
    label,
    ls="-",
    title="",
    face_color="w",
    grid_color=LIGHT_GREY,
):
    ax.plot(x, y, ls=ls, lw=1.5, label=label, c=color)
    ax.fill_between(x, ylow, yhigh, alpha=0.2, color=color)
    ax.set_title(title)
    ax.set_facecolor(face_color)
    ax.grid(color=grid_color)


def handle_2D_axes_and_legend(axes, legend_ncol, legend_offset, order=None):
    if order is not None:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        axes[0, 0].legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc="lower center",
            ncol=legend_ncol,
            bbox_to_anchor=legend_offset,
            frameon=False,
        )
    else:
        axes[0, 0].legend(
            loc="lower center",
            ncol=legend_ncol,
            bbox_to_anchor=legend_offset,
            frameon=False,
        )

    # Episode and return axes labels only on border plots
    for ax in axes[:, 0]:
        ax.set_ylabel("Return")

    for ax in axes[-1, :]:
        ax.set_xlabel("Episodes")


def handle_1D_axes_and_legend(
    axes, legend_ncol, legend_offset, columnspacing=1.0, order=None
):
    if order is not None:
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc="lower center",
            ncol=legend_ncol,
            columnspacing=columnspacing,
            bbox_to_anchor=legend_offset,
            frameon=False,
        )
    else:
        axes[0].legend(
            loc="lower center",
            ncol=legend_ncol,
            columnspacing=columnspacing,
            bbox_to_anchor=legend_offset,
            frameon=False,
        )

    # Episode and return axes labels only on border plots
    axes[0].set_ylabel("Return")
    for ax in axes:
        ax.set_xlabel("Episodes")
