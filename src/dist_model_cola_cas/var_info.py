"""Physical variable metadata for the Column A distillation model.

Each entry covers a scalar physical variable or parameter of the system.

Dictionary structure
--------------------
Key (var_name) : variable name as used in the codebase (matches Skogestad
                 notation).
Value : dict with fields:
    name   : short human-readable name
    symbol : LaTeX representation of the variable name
    units  : unit string in Pint-compatible slash notation;
             "dimensionless" for pure ratios, "mol/mol" for mole fractions
    desc   : longer plain-English description with extra context (optional)
"""

var_info = {
    # ------------------------------------------------------------------
    # Column flows and feed conditions  (model inputs)
    # ------------------------------------------------------------------
    "LT": {
        "name": "Reflux flow",
        "symbol": r"$L_T$",
        "units": "kmol/min",
    },
    "VB": {
        "name": "Boilup flow",
        "symbol": r"$V_B$",
        "units": "kmol/min",
    },
    "D": {
        "name": "Distillate flow",
        "symbol": r"$D$",
        "units": "kmol/min",
    },
    "B": {
        "name": "Bottoms flow",
        "symbol": r"$B$",
        "units": "kmol/min",
    },
    "F": {
        "name": "Feed flow rate",
        "symbol": r"$F$",
        "units": "kmol/min",
    },
    "zF": {
        "name": "Feed composition",
        "symbol": r"$z_F$",
        "units": "mol/mol",
        "desc": "Mole fraction of light component A in feed",
    },
    "qF": {
        "name": "Feed liquid fraction",
        "symbol": r"$q_F$",
        "units": "dimensionless",
        "desc": "q-value; 1 = saturated liquid feed",
    },
    # ------------------------------------------------------------------
    # Column parameters
    # ------------------------------------------------------------------
    "alpha": {
        "name": "Relative volatility",
        "symbol": r"$\alpha$",
        "units": "dimensionless",
    },
    "taul": {
        "name": "Liquid flow time constant",
        "symbol": r"$\tau_L$",
        "units": "min",
    },
    "F0": {
        "name": "Nominal feed rate",
        "symbol": r"$F_0$",
        "units": "kmol/min",
    },
    "qF0": {
        "name": "Nominal feed liquid fraction",
        "symbol": r"$q_{F,0}$",
        "units": "dimensionless",
    },
    "L0": {
        "name": "Nominal liquid flow above feed",
        "symbol": r"$L_0$",
        "units": "kmol/min",
    },
    "L0b": {
        "name": "Nominal liquid flow below feed",
        "symbol": r"$L_{0,b}$",
        "units": "kmol/min",
    },
    "lam": {
        "name": "K2-effect coefficient",
        "symbol": r"$\lambda$",
        "units": "dimensionless",
        "desc": "Vapour effect on liquid flow; 0 = no K2 effect",
    },
    "V0": {
        "name": "Nominal boilup flow",
        "symbol": r"$V_0$",
        "units": "kmol/min",
    },
    "V0t": {
        "name": "Nominal vapour flow above feed",
        "symbol": r"$V_{0,t}$",
        "units": "kmol/min",
    },
    # ------------------------------------------------------------------
    # LV closed-loop model: P-controller parameters
    # ------------------------------------------------------------------
    "KcD": {
        "name": "Condenser level controller gain",
        "symbol": r"$K_{c,D}$",
        "units": "1/min",
    },
    "KcB": {
        "name": "Reboiler level controller gain",
        "symbol": r"$K_{c,B}$",
        "units": "1/min",
    },
    "Ds": {
        "name": "Nominal distillate flow setpoint",
        "symbol": r"$D_s$",
        "units": "kmol/min",
    },
    "Bs": {
        "name": "Nominal bottoms flow setpoint",
        "symbol": r"$B_s$",
        "units": "kmol/min",
    },
}
