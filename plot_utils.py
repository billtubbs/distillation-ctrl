"""Plot utilities for distillation column simulation results."""

import matplotlib.pyplot as plt


def _var_title(name, var_info):
    """Return the subplot title for a variable: info['name'] or var_name."""
    info = (var_info or {}).get(name)
    return info["name"] if info else name


def _var_ylabel(name, var_info, delta=False):
    """Return the y-axis label for a variable: 'name [units]' or 'name'.

    Prepends a delta symbol when delta=True (deviation plots).
    """
    info = (var_info or {}).get(name)
    prefix = "\u0394" if delta else ""
    if info:
        return f"{prefix}{name} [{info['units']}]"
    return f"{prefix}{name}"


def _input_ylabel(names, var_info, delta=False):
    """Return a shared y-axis label for one or more input variables.

    When all variables share the same units the label is
    'n1, n2 [units]'; otherwise just 'n1, n2'.  The delta symbol is
    prepended only for a single-variable subplot in deviation mode.
    """
    prefix = "\u0394" if (delta and len(names) == 1) else ""
    combined = ", ".join(names)
    if var_info:
        units = [var_info.get(n, {}).get("units") for n in names]
        if units[0] and all(u == units[0] for u in units):
            return f"{prefix}{combined} [{units[0]}]"
    return f"{prefix}{combined}"


def make_input_output_tsplot(
    sim_results,
    output_names,
    input_names,
    output_refs=None,
    input_refs=None,
    deviation=True,
    output_line_labels=None,
    var_info=None,
    time_label="Time",
    figsize=None,
):
    """Plot time-series of selected outputs and inputs from a simulation result.

    Produces a stacked figure with one subplot per output variable (above)
    and a single shared subplot for one or more input variables (below).
    Each subplot carries an axes title (variable name) and a y-axis label
    (var_name [units]), both derived from var_info when available.

    Parameters
    ----------
    sim_results : pd.DataFrame
        MultiIndex-column DataFrame with time as index and level-0 column
        names "Inputs" and "Outputs", as returned by run_sim.
    output_names : list of str
        Output variable names (level-1 keys under "Outputs") to plot, one
        subplot each.
    input_names : list of str
        Input variable names (level-1 keys under "Inputs") overlaid in the
        bottom subplot.
    output_refs : dict {name: float}, optional
        Reference values for outputs.  With deviation=True (default), the
        plot shows (y - ref) with a dashed zero line.  With deviation=False,
        the plot shows y with a dashed line at the ref value.
    input_refs : dict {name: float}, optional
        Reference values for inputs, with the same deviation/absolute logic.
    deviation : bool
        If True (default), subtract refs from data and draw hline at 0.
        If False, plot absolute values and mark refs with a dashed hline.
    output_line_labels : list of str, optional
        Legend labels for the output data lines (one per output_name).  When
        provided alongside output_refs in absolute mode (deviation=False), an
        "SS" legend entry is also added for the reference line.
    var_info : dict, optional
        Variable metadata dict (e.g. var_info from var_info.py).  Each entry
        must have 'name' (human-readable) and 'units' (Pint-compatible string).
        Used to populate subplot titles and y-axis labels.  Falls back to the
        variable name when a key is absent.
    time_label : str
        x-axis label used when sim_results.index has no name.  Default "Time".
    figsize : tuple, optional
        Figure size (width, height) in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : list of matplotlib.axes.Axes
    """
    n_out = len(output_names)
    if figsize is None:
        figsize = (7, 1 + 1.5 * (n_out + 1))

    fig, axs = plt.subplots(n_out + 1, 1, figsize=figsize, sharex=True)

    t = sim_results.index
    xlabel = t.name if t.name else time_label

    # ── Output subplots ───────────────────────────────────────────────────
    for i, name in enumerate(output_names):
        y = sim_results["Outputs"][name].copy()
        ref = (output_refs or {}).get(name)

        if ref is not None:
            if deviation:
                y = y - ref
                axs[i].axhline(0.0, color="k", linewidth=0.5, linestyle="--")
            else:
                hline_kw = {"color": "k", "linewidth": 0.5, "linestyle": "--"}
                if output_line_labels:
                    hline_kw["label"] = "SS"
                axs[i].axhline(ref, **hline_kw)

        line_kw = {"color": "tab:blue"}
        if output_line_labels:
            line_kw["label"] = output_line_labels[i]
        axs[i].plot(t, y, **line_kw)

        if output_line_labels and ref is not None and not deviation:
            axs[i].legend(loc="best", fontsize=8)

        axs[i].set_title(_var_title(name, var_info), loc="left", fontsize=9)
        axs[i].set_ylabel(_var_ylabel(name, var_info, delta=deviation and ref is not None))
        axs[i].grid(True, alpha=0.3)

    # ── Input subplot ─────────────────────────────────────────────────────
    single_input = len(input_names) == 1
    has_any_ref = any((input_refs or {}).get(n) is not None for n in input_names)

    for name in input_names:
        u = sim_results["Inputs"][name].copy()
        ref = (input_refs or {}).get(name)
        if deviation and ref is not None:
            u = u - ref

        kw = {"where": "post"}
        if single_input:
            kw["color"] = "tab:orange"
        else:
            kw["label"] = name
        axs[-1].step(t, u, **kw)

    if deviation and has_any_ref:
        axs[-1].axhline(0.0, color="k", linewidth=0.5, linestyle="--")
    elif not deviation and input_refs and single_input:
        ref = input_refs.get(input_names[0])
        if ref is not None:
            axs[-1].axhline(ref, color="k", linewidth=0.5, linestyle="--")

    if not single_input:
        axs[-1].legend(loc="best")

    input_title = ", ".join(_var_title(n, var_info) for n in input_names)
    axs[-1].set_title(input_title, loc="left", fontsize=9)
    axs[-1].set_ylabel(
        _input_ylabel(input_names, var_info, delta=deviation and has_any_ref)
    )
    axs[-1].set_xlabel(xlabel)
    axs[-1].grid(True, alpha=0.3)

    return fig, axs
