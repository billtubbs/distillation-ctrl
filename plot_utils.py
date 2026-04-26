"""Plot utilities for distillation column simulation results."""

import matplotlib.pyplot as plt


def make_tsplots(
    data,
    plot_info,
    var_info=None,
    time_label="Time",
    figsize=None,
):
    """Plot one or more time-series subplots from a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with time as index.  Columns may be a plain Index or a
        MultiIndex (e.g. as returned by ``run_simulation``).  For plain
        columns use string keys in ``plot_info``; for MultiIndex columns
        use ``(group, name)`` tuple keys.
    plot_info : dict
        Ordered mapping of subplot title → {var_name: kwargs}.  Each key
        becomes the subplot title; each inner dict maps column keys to
        keyword arguments passed directly to ``ax.plot()``.

        The special string key ``"ylabel"`` may be included to set a custom
        y-axis label for that subplot, overriding the auto-generated one::

            {"D and B flows": {
                "ylabel": "Flow [kmol/min]",
                ("Outputs", "D"): {"color": "C0"},
                ("Outputs", "B"): {"color": "C1"},
            }}

        Any valid ``ax.plot()`` kwargs are accepted.  Use
        ``drawstyle="steps-post"`` to draw a step plot.
    var_info : dict, optional
        Variable metadata dict keyed by bare variable name with ``"name"``
        and ``"units"`` entries.  For tuple keys the bare name is the second
        element of the tuple.  Used to build legend labels (multi-line
        subplots) and y-axis labels.  Falls back to the bare name when absent.
    time_label : str
        x-axis label used when ``data.index`` has no name.
    figsize : tuple, optional
        Figure size ``(width, height)`` in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : list of matplotlib.axes.Axes

    Examples
    --------
    >>> # MultiIndex DataFrame from run_simulation:
    >>> plot_info = {
    ...     "Bottoms composition": {("States", "x0"): {"color": "C0"}},
    ...     "D and B flows": {
    ...         "ylabel": "Flow [kmol/min]",
    ...         ("Outputs", "D"): {"color": "C1"},
    ...         ("Outputs", "B"): {"color": "C2"},
    ...     },
    ...     "Reflux flow": {
    ...         ("Inputs", "LT"): {"color": "C3", "drawstyle": "steps-post"},
    ...     },
    ... }
    >>> fig, axs = make_tsplots(sim_results, plot_info, var_info=var_info)
    """
    n_subplots = len(plot_info)
    if figsize is None:
        figsize = (7, 1 + 1.5 * n_subplots)

    fig, axs = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    if n_subplots == 1:
        axs = [axs]

    t = data.index
    xlabel = t.name if t.name else time_label

    for ax, (title, var_dict) in zip(axs, plot_info.items()):
        # Intercept optional custom ylabel before iterating variables
        custom_ylabel = var_dict.get("ylabel")
        var_items = {k: v for k, v in var_dict.items() if k != "ylabel"}
        multi = len(var_items) > 1
        units_seen = []

        for var_name, kwargs in var_items.items():
            kw = dict(kwargs)  # copy so caller's dict is not mutated
            bare = var_name[1] if isinstance(var_name, tuple) else var_name
            info = (var_info or {}).get(bare, {})
            if multi and "label" not in kw:
                full_name = info.get("name")
                kw["label"] = f"{full_name} ({bare})" if full_name else bare
            ax.plot(t, data[var_name], **kw)
            units_seen.append(info.get("units"))

        if multi:
            ax.legend(loc="best")

        if custom_ylabel is not None:
            ax.set_ylabel(custom_ylabel)
        elif multi:
            unique_units = set(u for u in units_seen if u)
            if len(unique_units) == 1:
                ax.set_ylabel(f"[{unique_units.pop()}]")
            elif len(unique_units) > 1:
                raise ValueError(
                    f"Subplot '{title}' mixes variables with different units: "
                    f"{sorted(unique_units)}. Provide a custom 'ylabel' to override."
                )
        else:
            bare = next(iter(var_items))
            bare = bare[1] if isinstance(bare, tuple) else bare
            ax.set_ylabel(_var_ylabel(bare, var_info))

        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)

    axs[-1].set_xlabel(xlabel)

    return fig, list(axs)


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


def make_input_output_tsplots(
    sim_results,
    output_names,
    input_names,
    output_refs=None,
    input_refs=None,
    output_line_labels=None,
    var_info=None,
    time_label="Time",
    figsize=None,
    title_prefix="",
    delta_ylabel=False,
):
    """Plot time-series of selected outputs and inputs from a simulation result.

    Produces a stacked figure with one subplot per output variable (above)
    and a single shared subplot for one or more input variables (below).
    Reference values, when provided, are shown as dashed horizontal lines.
    Subplot titles and y-axis labels are derived from var_info when available.

    Parameters
    ----------
    sim_results : pd.DataFrame
        MultiIndex-column DataFrame with time as index and level-0 column
        names "Inputs" and "Outputs", as returned by run_simulation.
    output_names : list of str
        Output variable names (level-1 keys under "Outputs") to plot, one
        subplot each.
    input_names : list of str
        Input variable names (level-1 keys under "Inputs") overlaid in the
        bottom subplot.
    output_refs : dict {name: float}, optional
        Reference values for outputs, drawn as dashed horizontal lines.
    input_refs : dict {name: float}, optional
        Reference values for inputs, drawn as dashed horizontal lines.
    output_line_labels : list of str, optional
        Legend labels for output data lines (one per output_name).  When
        provided alongside output_refs, an "SS" entry is added for the
        reference line.
    var_info : dict, optional
        Variable metadata dict with 'name' and 'units' entries.  Used to
        populate subplot titles and y-axis labels.
    time_label : str
        x-axis label used when sim_results.index has no name.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    title_prefix : str
        String prepended to each output subplot title, e.g. "Deviation in ".
    delta_ylabel : bool
        If True, prepend Δ to output and input y-axis labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : list of matplotlib.axes.Axes
    """
    n_out = len(output_names)
    single_input = len(input_names) == 1

    # ── Build plot_info for make_tsplots ──────────────────────────────────
    plot_info = {}

    for i, name in enumerate(output_names):
        title = title_prefix + _var_title(name, var_info)
        kw = {"color": "tab:blue"}
        if output_line_labels:
            kw["label"] = output_line_labels[i]
        plot_info[title] = {("Outputs", name): kw}

    input_title = ", ".join(_var_title(n, var_info) for n in input_names)
    input_subplot = {}
    for name in input_names:
        kw = {"drawstyle": "steps-post"}
        if single_input:
            kw["color"] = "tab:orange"
        input_subplot[("Inputs", name)] = kw
    input_subplot["ylabel"] = _input_ylabel(input_names, var_info, delta=delta_ylabel)
    plot_info[input_title] = input_subplot

    if figsize is None:
        figsize = (7, 1 + 1.5 * (n_out + 1))

    fig, axs = make_tsplots(
        sim_results, plot_info, var_info=var_info, time_label=time_label,
        figsize=figsize,
    )

    # ── Override ylabels with delta prefix if requested ───────────────────
    if delta_ylabel:
        for ax, name in zip(axs[:-1], output_names):
            ax.set_ylabel(_var_ylabel(name, var_info, delta=True))

    # ── Overlay reference hlines ──────────────────────────────────────────
    for i, name in enumerate(output_names):
        ref = (output_refs or {}).get(name)
        if ref is not None:
            hline_kw = {"color": "k", "linewidth": 0.5, "linestyle": "--"}
            if output_line_labels:
                hline_kw["label"] = "SS"
            axs[i].axhline(ref, **hline_kw)
        if output_line_labels and ref is not None:
            axs[i].legend(loc="best")

    if single_input:
        ref = (input_refs or {}).get(input_names[0])
        if ref is not None:
            axs[-1].axhline(ref, color="k", linewidth=0.5, linestyle="--")
    else:
        axs[-1].legend(loc="best")

    return fig, axs


def make_input_output_tsplots_sub_refs(
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
    """Plot time-series of selected outputs and inputs, optionally subtracting
    reference values to show deviations.

    When ``deviation=True`` (default) reference values are subtracted from each
    variable, subplot titles are prefixed with "Deviation in ", y-axis labels
    get a Δ prefix, and a dashed zero line is drawn.  When ``deviation=False``
    this delegates directly to ``make_input_output_tsplots`` with absolute
    values and dashed reference lines.

    Parameters
    ----------
    sim_results : pd.DataFrame
        MultiIndex-column DataFrame with time as index and level-0 column
        names "Inputs" and "Outputs", as returned by run_simulation.
    output_names : list of str
        Output variable names (level-1 keys under "Outputs") to plot, one
        subplot each.
    input_names : list of str
        Input variable names (level-1 keys under "Inputs") overlaid in the
        bottom subplot.
    output_refs : dict {name: float}, optional
        Reference values for outputs.  With deviation=True the plot shows
        (y - ref) with a dashed zero line; with deviation=False shows a
        dashed line at the ref value.
    input_refs : dict {name: float}, optional
        Reference values for inputs, with the same deviation/absolute logic.
    deviation : bool
        If True (default), subtract refs from data and draw hline at 0.
        If False, plot absolute values and mark refs with a dashed hline.
    output_line_labels : list of str, optional
        Legend labels for the output data lines (one per output_name).
    var_info : dict, optional
        Variable metadata dict with 'name' and 'units' entries.
    time_label : str
        x-axis label used when sim_results.index has no name.
    figsize : tuple, optional
        Figure size (width, height) in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : list of matplotlib.axes.Axes
    """
    if not deviation:
        return make_input_output_tsplots(
            sim_results, output_names, input_names,
            output_refs=output_refs, input_refs=input_refs,
            output_line_labels=output_line_labels, var_info=var_info,
            time_label=time_label, figsize=figsize,
        )

    # Subtract refs from a copy of the data
    data = sim_results.copy()
    for name in output_names:
        ref = (output_refs or {}).get(name)
        if ref is not None:
            data[("Outputs", name)] -= ref
    for name in input_names:
        ref = (input_refs or {}).get(name)
        if ref is not None:
            data[("Inputs", name)] -= ref

    # After subtraction all refs are 0 (for hlines)
    out_refs_zero = {n: 0.0 for n in output_names if (output_refs or {}).get(n) is not None}
    in_refs_zero = {n: 0.0 for n in input_names if (input_refs or {}).get(n) is not None}

    return make_input_output_tsplots(
        data, output_names, input_names,
        output_refs=out_refs_zero, input_refs=in_refs_zero,
        output_line_labels=output_line_labels, var_info=var_info,
        time_label=time_label, figsize=figsize,
        title_prefix="Deviation in ", delta_ylabel=True,
    )
