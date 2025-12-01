import streamlit as st
st.title("BESS_OPTI_TOOL")
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------
# Helper functions (very simplified!)
# ---------------------------

def simulate_bess(load_kw, pv_kw, dt_hours, batt_kwh, batt_kw, eta_rt):
    """
    Very simplified daily dispatch that:
      - charges from PV surplus
      - discharges to reduce load peaks
    Returns:
      net_load (after PV + battery),
      soc profile
    """
    n = len(load_kw)
    soc = np.zeros(n)
    net_load = load_kw.copy()

    # If PV is provided, subtract from load first (self-consumption baseline)
    if pv_kw is not None:
        net_load = load_kw - np.minimum(pv_kw, load_kw)

    capacity = batt_kwh
    max_power = batt_kw
    soc_energy = 0.0

    # Simple control: clip net_load to its median (toy example)
    target = np.median(net_load)

    for t in range(n):
        load = net_load[t]

        # Discharge to shave peaks
        if load > target and soc_energy > 0:
            possible_discharge = min(load - target, max_power) * dt_hours
            discharged = min(possible_discharge, soc_energy)
            soc_energy -= discharged
            net_load[t] -= discharged / dt_hours  # convert energy back to power

        # Charge from remaining PV (if any)
        if pv_kw is not None:
            surplus_pv = max(0, pv_kw[t] - load)
            if surplus_pv > 0 and soc_energy < capacity:
                possible_charge = min(surplus_pv, max_power) * dt_hours
                # account for efficiency on charging side
                charged = min(possible_charge * np.sqrt(eta_rt), capacity - soc_energy)
                soc_energy += charged

        soc[t] = soc_energy / capacity if capacity > 0 else 0.0

    return net_load, soc


def economics(load_kw, net_load_kw, dt_hours, energy_price, demand_price,
              batt_kwh, batt_kw, capex_per_kwh, capex_per_kw, opex_pct,
              lifetime_years, discount_rate):
    """
    Compute simple annual savings and NPV.
    """

    hours_per_step = dt_hours
    energy_baseline = np.sum(load_kw) * hours_per_step
    energy_with_bess = np.sum(net_load_kw) * hours_per_step

    # Energy cost
    cost_baseline = energy_baseline * energy_price
    cost_with_bess = energy_with_bess * energy_price

    # Demand cost (assume 1-year profile; use max kW)
    demand_baseline = np.max(load_kw) * demand_price
    demand_with_bess = np.max(net_load_kw) * demand_price

    annual_savings = (cost_baseline + demand_baseline) - (cost_with_bess + demand_with_bess)

    # Investment
    capex = batt_kwh * capex_per_kwh + batt_kw * capex_per_kw
    annual_opex = capex * opex_pct

    net_annual_cashflow = annual_savings - annual_opex

    # NPV (simple annuity model)
    r = discount_rate
    T = lifetime_years
    if r == 0:
        npv = net_annual_cashflow * T - capex
    else:
        annuity_factor = (1 - (1 + r) ** -T) / r
        npv = net_annual_cashflow * annuity_factor - capex

    # Simple payback
    payback = capex / annual_savings if annual_savings > 0 else np.inf

    return {
        "annual_savings": annual_savings,
        "npv": npv,
        "payback": payback,
        "capex": capex,
    }

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="BESS Sizing Tool (CH)", layout="wide")

st.title("🔋 BESS Sizing Optimiser for Industrial Sites (CH)")

st.sidebar.header("Inputs")

# Data upload
load_file = st.sidebar.file_uploader("Load profile CSV", type=["csv"])
pv_file = st.sidebar.file_uploader("PV profile CSV (optional)", type=["csv"])

dt = st.sidebar.selectbox("Time resolution", [0.25, 1.0], format_func=lambda x: f"{int(60*x)} min")

energy_price = st.sidebar.number_input("Energy price [CHF/kWh]", value=0.20, min_value=0.0)
demand_price = st.sidebar.number_input("Demand charge [CHF/kW/year]", value=120.0, min_value=0.0)

st.sidebar.subheader("Battery sizing range")
batt_kwh_min = st.sidebar.number_input("Min capacity [kWh]", value=100.0, min_value=0.0)
batt_kwh_max = st.sidebar.number_input("Max capacity [kWh]", value=2000.0, min_value=0.0)
batt_kwh_step = st.sidebar.number_input("Step [kWh]", value=100.0, min_value=1.0)

c_rate = st.sidebar.number_input("C-rate", value=0.5, min_value=0.1, max_value=4.0)
eta_rt = st.sidebar.slider("Round-trip efficiency", 0.7, 0.99, 0.9)

st.sidebar.subheader("Costs & finance")
capex_per_kwh = st.sidebar.number_input("Capex per kWh [CHF/kWh]", value=300.0, min_value=0.0)
capex_per_kw = st.sidebar.number_input("Capex per kW [CHF/kW]", value=200.0, min_value=0.0)
opex_pct = st.sidebar.number_input("Opex % of Capex per year", value=0.02, min_value=0.0, max_value=0.2)
lifetime_years = st.sidebar.number_input("Project lifetime [years]", value=10, min_value=1)
discount_rate = st.sidebar.number_input("Discount rate", value=0.05, min_value=0.0, max_value=0.2)

run_button = st.sidebar.button("Run optimisation")

if run_button:
    if load_file is None:
        st.error("Please upload a load profile CSV with a 'load_kw' column.")
    else:
        load_df = pd.read_csv(load_file)
        if "load_kw" not in load_df.columns:
            st.error("Load profile must contain a 'load_kw' column.")
        else:
            load_kw = load_df["load_kw"].values

            pv_kw = None
            if pv_file is not None:
                pv_df = pd.read_csv(pv_file)
                if "pv_kw" in pv_df.columns:
                    pv_kw = pv_df["pv_kw"].values
                else:
                    st.warning("PV file has no 'pv_kw' column; ignoring PV.")

            sizes = np.arange(batt_kwh_min, batt_kwh_max + batt_kwh_step, batt_kwh_step)
            results = []

            for batt_kwh in sizes:
                batt_kw = batt_kwh * c_rate

                net_load, soc = simulate_bess(
                    load_kw, pv_kw, dt_hours=dt,
                    batt_kwh=batt_kwh, batt_kw=batt_kw,
                    eta_rt=eta_rt
                )

                econ = economics(
                    load_kw, net_load, dt_hours=dt,
                    energy_price=energy_price,
                    demand_price=demand_price,
                    batt_kwh=batt_kwh,
                    batt_kw=batt_kw,
                    capex_per_kwh=capex_per_kwh,
                    capex_per_kw=capex_per_kw,
                    opex_pct=opex_pct,
                    lifetime_years=lifetime_years,
                    discount_rate=discount_rate
                )
                econ["batt_kwh"] = batt_kwh
                econ["batt_kw"] = batt_kw
                results.append(econ)

            res_df = pd.DataFrame(results)
            optimal = res_df.loc[res_df["npv"].idxmax()]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Optimal energy size [kWh]", f"{optimal['batt_kwh']:.0f}")
            col2.metric("Optimal power [kW]", f"{optimal['batt_kw']:.0f}")
            col3.metric("NPV [CHF]", f"{optimal['npv']:.0f}")
            col4.metric("Payback [years]", f"{optimal['payback']:.1f}")

            # ROI vs size
            fig_npv = px.line(res_df, x="batt_kwh", y="npv",
                              title="NPV vs Battery Size",
                              labels={"batt_kwh": "Battery size [kWh]", "npv": "NPV [CHF]"})
            st.plotly_chart(fig_npv, use_container_width=True)

            # Savings breakdown for optimal size (toy: we only stored total annual_savings)
            st.subheader("Annual savings for optimal size")
            st.write(f"Annual savings: **{optimal['annual_savings']:.0f} CHF/year**")

            # Example time-series for optimal size (just re-run for that size)
            net_load_opt, soc_opt = simulate_bess(
                load_kw, pv_kw, dt_hours=dt,
                batt_kwh=optimal["batt_kwh"],
                batt_kw=optimal["batt_kw"],
                eta_rt=eta_rt
            )
            ts_df = pd.DataFrame({
                "Original load [kW]": load_kw,
                "Net load with BESS [kW]": net_load_opt,
                "SoC": soc_opt
            }).reset_index(names="timestep")

            st.subheader("Sample dispatch (full profile)")
            fig_ts = px.line(ts_df, x="timestep", y=["Original load [kW]", "Net load with BESS [kW]"],
                             title="Load profile before/after BESS")
            st.plotly_chart(fig_ts, use_container_width=True)

            fig_soc = px.line(ts_df, x="timestep", y="SoC", title="Battery State of Charge")
            st.plotly_chart(fig_soc, use_container_width=True)


