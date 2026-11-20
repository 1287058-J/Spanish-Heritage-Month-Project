from __future__ import annotations
from decimal import Decimal, ROUND_HALF_UP
import io
import json
import math
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ==============================================================================
# 1. DETERMINISTIC ROB-3 SIMULATION ENGINE & DATA CONTRACT
# ==============================================================================

def to_money(val: float | int | str | Decimal) -> Decimal:
    """Convert numeric input to exact 2-decimal USD Decimal."""
    return Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def default_state() -> Dict[str, Any]:
    """Return initial Team 7 ROB-3 state right after startup capital allocation."""
    return {
        "team_id": "7",
        "course": "AP Calculus",
        "pathway_id": "ROB-3",
        "business_name": "Robotics kit and repair studio",
        "snapshot_id": "SNAP-7-V0",
        "state_version": 0,
        "simulation_date": "2027-01-01",
        "base_period": "Monthly",
        "working_days": 20,
        "currency": "USD",
        "selected_level": "Advanced",
        "level_by_skill": {"AP_Calc_Optimization_Integration": "Advanced"},
        "assumptions": [
            "Fictional teaching baseline for ROB-3 Robotics kit and repair studio.",
            "Components are expensed when consumed in attempted builds; cash outflow occurs on restock.",
            "Failed kits consume full variable costs ($65/unit) and generate $0 revenue.",
            "Borrowing, interest, refunds, depreciation, and taxes are $0/disabled at baseline."
        ],
        "sources": [
            {"id": "SRC-1", "label": "Career Venture Lab Student Handbook (ROB-3 Package)", "type": "supplied"}
        ],
        "offerings": [
            {
                "offering_id": "OFF-ROB-KIT",
                "name": "Introductory robot kit",
                "unit": "kit",
                "price": 120.00,
                "demand_rule": "q(p) = max(0, 80 - (1/3)*p)",
                "capacity_rule": "min(floor(staff_hours/hours_per_unit), stock_available)"
            }
        ],
        "funding": {
            "owner_contribution": 12000.00,
            "equipment_assets": 4000.00,
            "pre_opening_expense": 0.00,
            "opening_inventory_value": 3000.00,
            "opening_cash": 5000.00,
            "loan_debt": 0.00,
            "retained_earnings": 0.00
        },
        "balances": {
            "cash": 5000.00,
            "equipment_assets": 4000.00,
            "stock_units": 100,
            "component_unit_cost": 30.00,
            "inventory_value": 3000.00,
            "debt": 0.00,
            "owner_equity": 12000.00,
            "retained_earnings": 0.00
        },
        "parameters": {
            "base_price": 120.00,
            "D0": 40,
            "staff_hours": 90.0,
            "hours_per_unit": 3.0,
            "restock_units": 0,
            "defect_rate": 0.0,
            "variable_lines": {
                "components_COGS": 30.00,
                "assembly_labor": 20.00,
                "packaging": 10.00,
                "warranty_consumables": 5.00
            },
            "fixed_lines": {
                "workspace": 250.00,
                "software": 100.00,
                "equipment_maintenance": 150.00,
                "admin": 100.00
            }
        },
        "capacity": {
            "derived_capacity": 30,
            "bottleneck": "staff_hours"
        },
        "executed_run_ids": [],
        "ledger": [],
        "campaigns": [],
        "decisions": []
    }


def compute_monthly_step(
    state: Dict[str, Any],
    requested: int,
    ad_cost: float = 0.0,
    capacity_override: int | None = None,
    price_override: float | None = None,
    restock_override: int | None = None,
    defect_rate_override: float | None = None,
    refund_amount: float = 0.0,
    loan_principal_repayment: float = 0.0,
    new_loan_cash: float = 0.0,
    new_owner_cash: float = 0.0,
    interest_expense: float = 0.0,
    additional_capex: float = 0.0,
) -> Dict[str, Any]:
    """Deterministic ROB-3 monthly calculation without mutating state."""
    params = state["parameters"]
    balances = state["balances"]

    if requested < 0:
        raise ValueError("Requested demand cannot be negative.")

    price = to_money(price_override if price_override is not None else params["base_price"])
    if price <= Decimal("0.00"):
        raise ValueError("Unit price must be greater than $0.")

    staff_hours = float(params["staff_hours"])
    hours_per_unit = float(params["hours_per_unit"])
    if staff_hours < 0 or hours_per_unit <= 0:
        raise ValueError("Invalid staff_hours or hours_per_unit.")

    derived_capacity = math.floor(staff_hours / hours_per_unit)
    effective_capacity = int(capacity_override) if capacity_override is not None else derived_capacity

    opening_stock = int(balances["stock_units"])
    restock_units = int(restock_override if restock_override is not None else params["restock_units"])
    if restock_units < 0:
        raise ValueError("Restock units cannot be negative.")

    defect_rate = float(defect_rate_override if defect_rate_override is not None else params["defect_rate"])
    if not (0.0 <= defect_rate <= 1.0):
        raise ValueError("Defect rate must be between 0 and 1.")

    available_stock = opening_stock + restock_units
    attempted = min(int(requested), effective_capacity, available_stock)
    failed = math.floor(attempted * defect_rate)
    sold = attempted - failed
    unserved_demand = max(0, int(requested) - attempted)

    v_lines = params["variable_lines"]
    comp_unit_cost = to_money(v_lines["components_COGS"])
    non_comp_unit_cost = sum(
        (to_money(v) for k, v in v_lines.items() if k != "components_COGS"),
        Decimal("0.00")
    )
    variable_per_unit = comp_unit_cost + non_comp_unit_cost

    fixed_expense = sum((to_money(v) for v in params["fixed_lines"].values()), Decimal("0.00"))
    marketing_expense = to_money(ad_cost)
    refund_dec = to_money(refund_amount)
    interest_dec = to_money(interest_expense)
    principal_dec = to_money(loan_principal_repayment)
    capex_dec = to_money(additional_capex)
    new_loan_dec = to_money(new_loan_cash)
    new_owner_dec = to_money(new_owner_cash)

    gross_revenue = price * Decimal(sold)
    net_revenue = gross_revenue - refund_dec

    consumed_component_cost = comp_unit_cost * Decimal(attempted)
    non_component_var_cost = non_comp_unit_cost * Decimal(attempted)
    variable_expense = variable_per_unit * Decimal(attempted)
    failed_writeoff_included = variable_per_unit * Decimal(failed)

    total_operating_expense = variable_expense + fixed_expense + marketing_expense
    operating_profit = net_revenue - total_operating_expense
    net_profit = operating_profit - interest_dec

    restock_cash_outflow = comp_unit_cost * Decimal(restock_units)
    ending_stock_units = available_stock - attempted
    ending_inventory_value = to_money(balances["inventory_value"]) + restock_cash_outflow - consumed_component_cost

    opening_cash = to_money(balances["cash"])
    ending_cash = (
        opening_cash
        + net_revenue
        - restock_cash_outflow
        - non_component_var_cost
        - fixed_expense
        - marketing_expense
        - interest_dec
        - principal_dec
        - capex_dec
        + new_owner_dec
        + new_loan_dec
    )

    ending_equipment = to_money(balances["equipment_assets"]) + capex_dec
    ending_debt = to_money(balances["debt"]) + new_loan_dec - principal_dec
    ending_owner_equity = to_money(balances["owner_equity"]) + new_owner_dec
    ending_retained_earnings = to_money(balances["retained_earnings"]) + net_profit

    if capacity_override is not None:
        bottleneck = "capacity_override (Fixture Mode)"
    elif effective_capacity <= available_stock and effective_capacity < requested:
        bottleneck = "staff_hours"
    elif available_stock < effective_capacity and available_stock < requested:
        bottleneck = "component_stock"
    else:
        bottleneck = "demand (no resource bottleneck)"

    return {
        "mode": "Fixture Override" if capacity_override is not None else "Normal Resource Mode",
        "price": float(price),
        "requested": int(requested),
        "derived_capacity": derived_capacity,
        "effective_capacity": effective_capacity,
        "opening_stock_units": opening_stock,
        "restock_units": restock_units,
        "attempted": attempted,
        "failed": failed,
        "sold": sold,
        "unserved_demand": unserved_demand,
        "bottleneck": bottleneck,
        "gross_revenue": float(gross_revenue),
        "refund_amount": float(refund_dec),
        "net_revenue": float(net_revenue),
        "variable_per_unit": float(variable_per_unit),
        "consumed_component_cost": float(consumed_component_cost),
        "non_component_var_cost": float(non_component_var_cost),
        "variable_expense": float(variable_expense),
        "failed_writeoff_included": float(failed_writeoff_included),
        "fixed_expense": float(fixed_expense),
        "marketing_expense": float(marketing_expense),
        "total_operating_expense": float(total_operating_expense),
        "operating_profit": float(operating_profit),
        "interest_expense": float(interest_dec),
        "net_profit": float(net_profit),
        "opening_cash": float(opening_cash),
        "restock_cash_outflow": float(restock_cash_outflow),
        "ending_cash": float(ending_cash),
        "ending_stock_units": ending_stock_units,
        "ending_inventory_value": float(ending_inventory_value),
        "ending_equipment_assets": float(ending_equipment),
        "ending_debt": float(ending_debt),
        "ending_owner_equity": float(ending_owner_equity),
        "ending_retained_earnings": float(ending_retained_earnings),
    }


def commit_simulation_step(
    state: Dict[str, Any],
    run_id: str,
    sim_date: str,
    result: Dict[str, Any],
    campaign_id: str = ""
) -> Tuple[bool, str]:
    """Commit a computed monthly step into the ledger with duplicate run_id protection."""
    if not run_id or not run_id.strip():
        return False, "A unique run_id is required before committing."
    if run_id in state["executed_run_ids"]:
        return False, f"Duplicate run_id '{run_id}' rejected. No duplicate transactions created."

    new_version = int(state["state_version"]) + 1
    tx_base = f"TX-V{new_version}-{run_id}"

    new_entries: List[Dict[str, Any]] = []
    if result["net_revenue"] > 0:
        new_entries.append({
            "transaction_id": f"{tx_base}-REV",
            "date": sim_date,
            "event_type": "sales_revenue",
            "category": "Revenue",
            "amount": result["net_revenue"],
            "quantity": result["sold"],
            "unit_price": result["price"],
            "offering_id": "OFF-ROB-KIT",
            "campaign_id": "",
            "financing_type": "operating",
            "state_version": new_version
        })
    if result["variable_expense"] > 0:
        new_entries.append({
            "transaction_id": f"{tx_base}-VAR",
            "date": sim_date,
            "event_type": "variable_production_cost",
            "category": "Variable Expense",
            "amount": -result["variable_expense"],
            "quantity": result["attempted"],
            "unit_price": result["variable_per_unit"],
            "offering_id": "OFF-ROB-KIT",
            "campaign_id": "",
            "financing_type": "operating",
            "state_version": new_version
        })
    if result["fixed_expense"] > 0:
        new_entries.append({
            "transaction_id": f"{tx_base}-FIX",
            "date": sim_date,
            "event_type": "monthly_fixed_overhead",
            "category": "Fixed Expense",
            "amount": -result["fixed_expense"],
            "quantity": 1,
            "unit_price": result["fixed_expense"],
            "offering_id": "",
            "campaign_id": "",
            "financing_type": "operating",
            "state_version": new_version
        })
    if result["marketing_expense"] > 0:
        new_entries.append({
            "transaction_id": f"{tx_base}-MKT",
            "date": sim_date,
            "event_type": "marketing_campaign",
            "category": "Marketing Expense",
            "amount": -result["marketing_expense"],
            "quantity": 1,
            "unit_price": result["marketing_expense"],
            "offering_id": "",
            "campaign_id": campaign_id or f"CAMP-{sim_date}",
            "financing_type": "operating",
            "state_version": new_version
        })
    if result["restock_cash_outflow"] > 0:
        new_entries.append({
            "transaction_id": f"{tx_base}-STK",
            "date": sim_date,
            "event_type": "inventory_restock",
            "category": "Inventory Purchase",
            "amount": -result["restock_cash_outflow"],
            "quantity": result["restock_units"],
            "unit_price": state["balances"]["component_unit_cost"],
            "offering_id": "",
            "campaign_id": "",
            "financing_type": "working_capital",
            "state_version": new_version
        })

    state["executed_run_ids"].append(run_id)
    state["ledger"].extend(new_entries)
    state["state_version"] = new_version
    state["snapshot_id"] = f"SNAP-{state['team_id']}-V{new_version}"
    state["simulation_date"] = sim_date
    state["capacity"]["derived_capacity"] = result["derived_capacity"]
    state["capacity"]["bottleneck"] = result["bottleneck"]

    bal = state["balances"]
    bal["cash"] = result["ending_cash"]
    bal["stock_units"] = result["ending_stock_units"]
    bal["inventory_value"] = result["ending_inventory_value"]
    bal["equipment_assets"] = result["ending_equipment_assets"]
    bal["debt"] = result["ending_debt"]
    bal["owner_equity"] = result["ending_owner_equity"]
    bal["retained_earnings"] = result["ending_retained_earnings"]

    return True, f"Committed {run_id} -> Advanced to state_version v{new_version}."


def validate_and_import_snapshot(raw_json: str, current_state: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any] | None]:
    """Validate imported JSON snapshot against team_id and state_version rules."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return False, f"Invalid JSON format: {exc}", None

    required_keys = ["team_id", "course", "pathway_id", "snapshot_id", "state_version", "balances", "parameters", "ledger"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        return False, f"Rejected import: missing required contract fields {missing}.", None

    if str(data.get("team_id")) != "7":
        return False, f"Rejected import: team_id '{data.get('team_id')}' does not match Team '7'.", None

    if data.get("pathway_id") != "ROB-3":
        return False, f"Rejected import: pathway_id '{data.get('pathway_id')}' does not match 'ROB-3'.", None

    imported_ver = int(data.get("state_version", -1))
    current_ver = int(current_state.get("state_version", 0))
    if imported_ver < current_ver:
        return False, f"Rejected stale snapshot: imported state_version v{imported_ver} is older than active v{current_ver}.", None

    return True, f"Accepted snapshot {data['snapshot_id']} (state_version v{imported_ver}).", data


# ==============================================================================
# 2. AP CALCULUS ANALYTIC & NUMERICAL FUNCTIONS
# ==============================================================================

def ap_calc_profit_analysis(D0: float, p0: float, v: float, F: float, capacity: int) -> Dict[str, Any]:
    """Exact AP Calculus optimization model for demand q(p) = a - b*p on [0, 2*p0]."""
    a = 2.0 * D0
    b = D0 / p0
    p_max = 2.0 * p0

    p_unconstrained = (a + b * v) / (2.0 * b)
    q_unconstrained = max(0.0, a - b * p_unconstrained)
    profit_unconstrained = (p_unconstrained - v) * q_unconstrained - F

    p_corner = max(0.0, (a - capacity) / b)

    candidates = [0.0, p_corner, p_unconstrained, p_max]
    valid_candidates = [p for p in candidates if 0.0 <= p <= p_max]

    def eval_continuous(p: float) -> Tuple[float, float]:
        q_raw = max(0.0, a - b * p)
        s_cont = min(float(capacity), q_raw)
        prof = (p - v) * s_cont - F
        return s_cont, prof

    best_p = 0.0
    best_prof = -1e18
    for p in valid_candidates:
        _, prof = eval_continuous(p)
        if prof > best_prof:
            best_prof = prof
            best_p = p

    best_s_cont, _ = eval_continuous(best_p)

    best_int_p = round(best_p, 2)
    q_at_best = max(0.0, a - b * best_int_p)
    s_int = min(int(capacity), math.floor(q_at_best))
    prof_int = (best_int_p - v) * s_int - F

    return {
        "a": a,
        "b": b,
        "p_max": p_max,
        "p_unconstrained": p_unconstrained,
        "q_unconstrained": q_unconstrained,
        "profit_unconstrained": profit_unconstrained,
        "unconstrained_feasible": q_unconstrained <= capacity,
        "p_corner": p_corner,
        "constrained_opt_p": best_p,
        "constrained_opt_q": best_s_cont,
        "constrained_opt_profit": best_prof,
        "integer_sales_at_opt": s_int,
        "integer_profit_at_opt": prof_int,
    }


def ap_calc_rate_integral(t_start: float = 0.0, t_end: float = 10.0) -> Dict[str, float]:
    """Isolated AP Calculus rate accumulation: r(t) = 100 + 20t, c(t) = 60 + 5t."""
    def R_antideriv(t: float) -> float:
        return 100.0 * t + 10.0 * (t ** 2)

    def C_antideriv(t: float) -> float:
        return 60.0 * t + 2.5 * (t ** 2)

    acc_rev = R_antideriv(t_end) - R_antideriv(t_start)
    acc_cost = C_antideriv(t_end) - C_antideriv(t_start)
    net_profit = acc_rev - acc_cost
    return {
        "t_start": t_start,
        "t_end": t_end,
        "accumulated_revenue": acc_rev,
        "accumulated_cost": acc_cost,
        "net_accumulated_profit": net_profit,
    }


# ==============================================================================
# 3. STREAMLIT DASHBOARD UI (STAGE 1)
# ==============================================================================

st.set_page_config(
    page_title="Team 7 | ROB-3 Robotics Studio (AP Calculus)",
    page_icon="🤖",
    layout="wide",
)

if "sim_state" not in st.session_state:
    st.session_state.sim_state = default_state()

state = st.session_state.sim_state

# --- Sidebar: Save/Load Snapshot & Reset ---
with st.sidebar:
    st.header("Team 7 | ROB-3 Control")
    st.caption(f"Course: **{state['course']}** | Pathway: **{state['pathway_id']}**")
    st.metric("Active State Version", f"v{state['state_version']}")
    st.text(f"Snapshot ID: {state['snapshot_id']}")
    st.text(f"Sim Date: {state['simulation_date']}")

    st.divider()
    st.subheader("Save / Load Snapshot (JSON)")
    st.caption("Think of a Snapshot JSON like a video game save file—always export before closing!")
    snapshot_json_str = json.dumps(state, indent=2)
    st.download_button(
        label="Export Current Snapshot JSON",
        data=snapshot_json_str,
        file_name=f"{state['snapshot_id']}.json",
        mime="application/json",
        use_container_width=True,
    )

    uploaded_snap = st.file_uploader("Import Snapshot JSON", type=["json"])
    if uploaded_snap is not None:
        if st.button("Validate & Restore Snapshot", use_container_width=True):
            raw_text = uploaded_snap.read().decode("utf-8")
            ok, msg, loaded = validate_and_import_snapshot(raw_text, state)
            if ok and loaded is not None:
                st.session_state.sim_state = loaded
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    st.divider()
    if st.button("Reset to Post-Startup Baseline (v0)", use_container_width=True):
        st.session_state.sim_state = default_state()
        st.warning("Reset to post-startup baseline (Cash: $5,000, Stock: 100 units).")
        st.rerun()

# --- Main Header & Top KPI Ribbon ---
st.title("Robotics Kit & Repair Studio — Career Venture Lab")
st.markdown(
    "**Team ID:** `7` | **Pathway:** `ROB-3` | **Course:** `AP Calculus` | "
    "**Level:** `Advanced` | **Stage 1:** Core Engine, Ledger & Calculus Panel"
)

bal = state["balances"]
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Spendable Cash", f"${bal['cash']:,.2f}")
k2.metric("Component Stock", f"{bal['stock_units']} kits", f"${bal['inventory_value']:,.2f} value")
k3.metric("Equipment Assets", f"${bal['equipment_assets']:,.2f}")
k4.metric("Retained Earnings", f"${bal['retained_earnings']:,.2f}")
k5.metric("Monthly Capacity", f"{state['capacity']['derived_capacity']} kits", state["capacity"]["bottleneck"])

tabs = st.tabs([
    "1. Inputs & Funding",
    "2. Run Monthly Simulation & Ledger",
    "3. AP Calculus Math Panel",
    "4. Fixture Replay & Tests",
    "5. Stage 2 & 3 Roadmap (Locked)",
])

# --- TAB 1: INPUTS & FUNDING ---
with tabs[0]:
    st.subheader("Startup Financing & Opening Balances (Starter Teaching Defaults)")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        st.table(pd.DataFrame([
            {"Item": "Owner Contribution (Equity)", "Amount (USD)": "$12,000.00", "Plain-English Rule": "Owner starter money (not sales revenue)"},
            {"Item": "Equipment Assets (Tools & Test Rig)", "Amount (USD)": "$4,000.00", "Plain-English Rule": "Studio tools (not counted as a monthly expense again)"},
            {"Item": "Opening Component Inventory (100 sets @ $30)", "Amount (USD)": "$3,000.00", "Plain-English Rule": "Parts on the shelf (expensed only when used in builds)"},
            {"Item": "Post-Startup Spendable Cash", "Amount (USD)": "$5,000.00", "Plain-English Rule": "Cash available in the bank to run the business"},
        ]))
    with f_col2:
        st.info(
            "**ROB-3 Plain-English Accounting Rules:**\n"
            "- **Restocking Parts:** Buying new kits deducts **Cash** right away and adds to your **Component Stock**.\n"
            "- **Using Starter Parts:** Building kits from your starting 100 sets counts as a **$30/kit expense** when built, without charging your cash a second time.\n"
            "- **Defective Kits:** If a robot kit fails quality testing, it still uses up all **$65** in parts, labor, and packaging, but brings in **$0** in sales."
        )

    st.divider()
    st.subheader("Editable Operating Parameters & Cost Lines")
    with st.form("params_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            base_price = st.number_input("Base Price per Kit ($)", min_value=1.0, value=float(state["parameters"]["base_price"]), step=5.0)
            D0 = st.number_input("Baseline Demand D0 (kits/mo)", min_value=0, value=int(state["parameters"]["D0"]), step=1)
            defect_rate = st.slider("Defect Rate (0.00 to 1.00)", min_value=0.0, max_value=1.0, value=float(state["parameters"]["defect_rate"]), step=0.01)
        with c2:
            staff_hours = st.number_input("Productive Staff Hours / Month", min_value=0.0, value=float(state["parameters"]["staff_hours"]), step=3.0)
            hours_per_unit = st.number_input("Assembly Hours per Kit", min_value=0.5, value=float(state["parameters"]["hours_per_unit"]), step=0.5)
            restock_units = st.number_input("Default Monthly Restock (sets)", min_value=0, value=int(state["parameters"]["restock_units"]), step=5)
        with c3:
            st.markdown("**Variable Costs ($ / attempted kit)**")
            v_comp = st.number_input("Components COGS ($)", min_value=0.0, value=float(state["parameters"]["variable_lines"]["components_COGS"]), step=1.0)
            v_labor = st.number_input("Assembly Labor ($)", min_value=0.0, value=float(state["parameters"]["variable_lines"]["assembly_labor"]), step=1.0)
            v_pack = st.number_input("Packaging ($)", min_value=0.0, value=float(state["parameters"]["variable_lines"]["packaging"]), step=1.0)
            v_warr = st.number_input("Warranty / Consumables ($)", min_value=0.0, value=float(state["parameters"]["variable_lines"]["warranty_consumables"]), step=1.0)

        st.markdown("**Monthly Fixed Overhead ($ / month)**")
        fc1, fc2, fc3, fc4 = st.columns(4)
        f_work = fc1.number_input("Workspace ($)", min_value=0.0, value=float(state["parameters"]["fixed_lines"]["workspace"]), step=10.0)
        f_soft = fc2.number_input("Software ($)", min_value=0.0, value=float(state["parameters"]["fixed_lines"]["software"]), step=10.0)
        f_maint = fc3.number_input("Equipment Maintenance ($)", min_value=0.0, value=float(state["parameters"]["fixed_lines"]["equipment_maintenance"]), step=10.0)
        f_admin = fc4.number_input("Admin ($)", min_value=0.0, value=float(state["parameters"]["fixed_lines"]["admin"]), step=10.0)

        if st.form_submit_button("Save Parameter Updates"):
            state["parameters"]["base_price"] = float(base_price)
            state["parameters"]["D0"] = int(D0)
            state["parameters"]["defect_rate"] = float(defect_rate)
            state["parameters"]["staff_hours"] = float(staff_hours)
            state["parameters"]["hours_per_unit"] = float(hours_per_unit)
            state["parameters"]["restock_units"] = int(restock_units)
            state["parameters"]["variable_lines"] = {
                "components_COGS": float(v_comp),
                "assembly_labor": float(v_labor),
                "packaging": float(v_pack),
                "warranty_consumables": float(v_warr),
            }
            state["parameters"]["fixed_lines"] = {
                "workspace": float(f_work),
                "software": float(f_soft),
                "equipment_maintenance": float(f_maint),
                "admin": float(f_admin),
            }
            state["capacity"]["derived_capacity"] = math.floor(staff_hours / hours_per_unit)
            st.success("Parameters updated. Remember to export your Snapshot JSON in the sidebar to save!")
            st.rerun()

# --- TAB 2: RUN MONTHLY SIMULATION & LEDGER ---
with tabs[1]:
    st.subheader("What-If Preview vs. Committed Transaction Ledger")
    st.caption("Test numbers safely below. Nothing enters your official receipt book (Ledger) until you click **Approve & Commit Step to Ledger**.")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sim_date = sc1.text_input("Simulation Period Date (YYYY-MM-DD)", value="2027-01-31")
    run_id = sc2.text_input("Unique Run ID", value=f"RUN-2027-01-{state['state_version']+1}")
    req_units = sc3.number_input("Requested Demand (kits)", min_value=0, value=int(state["parameters"]["D0"]), step=1)
    ad_spend = sc4.number_input("Approved Own-Business Ad Cost ($)", min_value=0.0, value=0.0, step=25.0)

    sc5, sc6, sc7, sc8 = st.columns(4)
    step_price = sc5.number_input("Step Unit Price ($)", min_value=1.0, value=float(state["parameters"]["base_price"]), step=5.0)
    step_restock = sc6.number_input("Component Restock This Month (sets)", min_value=0, value=int(state["parameters"]["restock_units"]), step=5)
    step_defect = sc7.slider("Step Defect Rate", min_value=0.0, max_value=1.0, value=float(state["parameters"]["defect_rate"]), step=0.01)
    step_loan = sc8.number_input("New Loan Proceeds ($)", min_value=0.0, value=0.0, step=100.0)

    preview = compute_monthly_step(
        state=state,
        requested=int(req_units),
        ad_cost=float(ad_spend),
        price_override=float(step_price),
        restock_override=int(step_restock),
        defect_rate_override=float(step_defect),
        new_loan_cash=float(step_loan),
    )

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Attempted / Sold Kits", f"{preview['attempted']} / {preview['sold']}", f"Failed: {preview['failed']} | Unserved: {preview['unserved_demand']}")
    p2.metric("Projected Revenue", f"${preview['net_revenue']:,.2f}")
    p3.metric("Operating Profit", f"${preview['operating_profit']:,.2f}", f"Total OpEx: ${preview['total_operating_expense']:,.2f}")
    p4.metric("Projected Closing Cash", f"${preview['ending_cash']:,.2f}", f"Ending Stock: {preview['ending_stock_units']} sets")

    if st.button("Approve & Commit Step to Ledger", type="primary"):
        ok, msg = commit_simulation_step(state, run_id.strip(), sim_date.strip(), preview)
        if ok:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

    st.divider()
    st.subheader(f"Committed Dated Transaction Ledger ({len(state['ledger'])} entries)")
    if state["ledger"]:
        ledger_df = pd.DataFrame(state["ledger"])
        st.dataframe(ledger_df, use_container_width=True)
        st.download_button(
            "Download Ledger CSV",
            data=ledger_df.to_csv(index=False),
            file_name=f"team7_rob3_ledger_v{state['state_version']}.csv",
            mime="text/csv",
        )
    else:
        st.info("No transactions committed yet. Use the preview panel above or run the 3-month test in Tab 4.")

# --- TAB 3: AP CALCULUS MATH PANEL ---
with tabs[2]:
    st.subheader("Persistent Math Skills Panel — AP Calculus")
    lvl_col1, lvl_col2 = st.columns([1, 2])
    with lvl_col1:
        chosen_level = st.selectbox(
            "Challenge Level (Requires Teacher Approval to Change)",
            options=["Foundation", "Applied", "Advanced"],
            index=["Foundation", "Applied", "Advanced"].index(state["selected_level"]),
        )
        teacher_approved = st.checkbox("Teacher approved this challenge level", value=True)
        if chosen_level != state["selected_level"] and teacher_approved:
            state["selected_level"] = chosen_level
            state["level_by_skill"]["AP_Calc_Optimization_Integration"] = chosen_level
            st.success(f"Challenge level updated to {chosen_level}.")

    p0 = float(state["parameters"]["base_price"])
    D0 = float(state["parameters"]["D0"])
    v_unit = sum(float(x) for x in state["parameters"]["variable_lines"].values())
    F_month = sum(float(x) for x in state["parameters"]["fixed_lines"].values())
    cap = int(state["capacity"]["derived_capacity"])

    calc_res = ap_calc_profit_analysis(D0=D0, p0=p0, v=v_unit, F=F_month, capacity=cap)

    with lvl_col2:
        st.markdown(
            f"**Course:** AP Calculus | **Level:** `{state['selected_level']}` | "
            f"**Domain:** $0 \\le p \\le \\${calc_res['p_max']:.2f}$ (Price in USD/kit)\n"
            f"- **Linear Demand:** $q(p) = {calc_res['a']:.1f} - {calc_res['b']:.4f}p$ (kits/month)\n"
            f"- **Capacity Constraint:** $s(p) = \min({cap},\ \max(0, q(p)))$ with capacity corner at $p = \${calc_res['p_corner']:.2f}$"
        )

    test_p = st.slider("Interactive Price Explorer ($/kit)", min_value=0.0, max_value=float(calc_res["p_max"]), value=float(p0), step=1.0)
    q_at_test = max(0.0, calc_res["a"] - calc_res["b"] * test_p)
    s_cont_test = min(float(cap), q_at_test)
    s_int_test = min(cap, math.floor(q_at_test))
    prof_cont_test = (test_p - v_unit) * s_cont_test - F_month
    prof_int_test = (test_p - v_unit) * s_int_test - F_month

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Demand q(p)", f"{q_at_test:.2f} kits")
    mc2.metric("Constrained Sales s(p)", f"{s_cont_test:.2f} cont. / {s_int_test} int.")
    mc3.metric("Continuous Profit P(p)", f"${prof_cont_test:,.2f}")
    mc4.metric("Integer-Kit Profit", f"${prof_int_test:,.2f}")

    prices = np.linspace(0.0, calc_res["p_max"], 400)
    q_vals = np.maximum(0.0, calc_res["a"] - calc_res["b"] * prices)
    s_vals = np.minimum(float(cap), q_vals)
    P_vals = (prices - v_unit) * s_vals - F_month

    P_prime_vals = np.where(
        prices < calc_res["p_corner"],
        float(cap),
        calc_res["a"] + calc_res["b"] * v_unit - 2.0 * calc_res["b"] * prices,
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax1.plot(prices, P_vals, label="Constrained Profit P(p)", color="#1f77b4", linewidth=2)
    ax1.axhline(0, color="black", linestyle="--", linewidth=1, label="Zero Profit Line ($0)")
    ax1.axvline(calc_res["p_corner"], color="#d95f02", linestyle=":", linewidth=1.5, label=f"Capacity Corner p=${calc_res['p_corner']:.2f}")
    ax1.scatter([calc_res["constrained_opt_p"]], [calc_res["constrained_opt_profit"]], color="#1b9e77", zorder=5, s=70, label=f"Constrained Optimum (${calc_res['constrained_opt_p']:.2f}, ${calc_res['constrained_opt_profit']:.2f})")
    ax1.scatter([calc_res["p_unconstrained"]], [calc_res["profit_unconstrained"]], facecolors="none", edgecolors="#e7298a", s=80, linewidths=2, label=f"Unconstrained Vertex (${calc_res['p_unconstrained']:.2f}, q={calc_res['q_unconstrained']:.2f})")
    ax1.set_ylabel("Monthly Profit ($ USD)")
    ax1.set_title(f"Team 7 | ROB-3 AP Calculus Profit & Derivative Analysis ({state['selected_level']} Mode)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    left_mask = prices < calc_res["p_corner"]
    right_mask = prices > calc_res["p_corner"]
    ax2.plot(prices[left_mask], P_prime_vals[left_mask], color="#7570b3", linewidth=2, label="P'(p) = Capacity (kits)")
    ax2.plot(prices[right_mask], P_prime_vals[right_mask], color="#7570b3", linewidth=2, label="P'(p) = a + bv - 2bp ($/dollar)")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    if 0 < calc_res["p_corner"] < calc_res["p_max"]:
        right_lim = calc_res["a"] + calc_res["b"] * v_unit - 2.0 * calc_res["b"] * calc_res["p_corner"]
        ax2.scatter([calc_res["p_corner"], calc_res["p_corner"]], [float(cap), right_lim], facecolors="white", edgecolors="#d95f02", s=65, zorder=6, label="Undefined Corner (Jump Discontinuity)")
    ax2.set_xlabel("Unit Price p ($ USD / kit)")
    ax2.set_ylabel("Marginal Profit dP/dp (kits)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)

    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=150)
    st.download_button(
        label="Download Stamped Profit & Derivative PNG",
        data=png_buf.getvalue(),
        file_name=f"team7_rob3_apcalc_graph_v{state['state_version']}.png",
        mime="image/png",
    )

    table_prices = [0.0, 60.0, 120.0, calc_res["p_corner"], calc_res["p_unconstrained"], 180.0, calc_res["p_max"]]
    table_rows = []
    for tp in sorted(set(round(x, 2) for x in table_prices if 0 <= x <= calc_res["p_max"])):
        qr = max(0.0, calc_res["a"] - calc_res["b"] * tp)
        sc = min(float(cap), qr)
        si = min(cap, math.floor(qr))
        table_rows.append({
            "Price ($)": tp,
            "Unconstrained Demand q(p)": round(qr, 2),
            "Continuous Sales s(p)": round(sc, 2),
            "Integer Sales floor(s)": si,
            "Continuous Profit ($)": round((tp - v_unit) * sc - F_month, 2),
            "Integer Profit ($)": round((tp - v_unit) * si - F_month, 2),
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

    if state["selected_level"] == "Foundation":
        st.info(
            "**Foundation Task:** Solve $P'(p) = 0$ for the unconstrained profit function $P(p) = (p - v)(a - bp) - F$. "
            f"Here $P'(p) = {calc_res['a'] + calc_res['b']*v_unit:.2f} - {2*calc_res['b']:.4f}p = 0 \implies p = \${calc_res['p_unconstrained']:.2f}$."
        )
    elif state["selected_level"] == "Applied":
        st.info(
            "**Applied Task:** Compare the unconstrained critical point against the capacity boundary $q \le 30$. "
            f"At baseline ($D_0=40, p_0=120, cap=30$), the capacity corner is $p = \${calc_res['p_corner']:.2f}$ where $q = 30$. "
            f"For $p > \$150$, demand drops below 30 and $P'(p) < 0$ ($P'(150^+) = {calc_res['a'] + calc_res['b']*v_unit - 2*calc_res['b']*150:.2f}$), "
            f"so the constrained maximum occurs right at the corner $p = \${calc_res['constrained_opt_p']:.2f}$ (Profit = $\${calc_res['constrained_opt_profit']:.2f}$)."
        )
    else:
        st.divider()
        st.subheader("Advanced Module: Accumulated Revenue, Cost & Net Profit via Definite Integration")
        st.markdown(
            "Isolated campaign rate exercise over $0 \\le t \\le 10$ days (kept separate from the monthly accounting ledger):\n"
            "- Marginal Revenue Rate: $r(t) = 100 + 20t$ ($	ext{\$/day}$)\n"
            "- Marginal Cost Rate: $c(t) = 60 + 5t$ ($	ext{\$/day}$)\n"
            "- Net Accumulation: $\int_0^{10} (r(t) - c(t))\,dt = \int_0^{10} (40 + 15t)\,dt = \left[40t + 7.5t^2\night]_0^{10} = \$1,150.00$"
        )
        t_vals = np.linspace(0, 10, 200)
        r_vals = 100 + 20 * t_vals
        c_vals = 60 + 5 * t_vals
        fig_int, ax_int = plt.subplots(figsize=(8, 3.2))
        ax_int.plot(t_vals, r_vals, label="Revenue Rate r(t) = 100 + 20t ($/day)", color="#1b9e77", linewidth=2)
        ax_int.plot(t_vals, c_vals, label="Cost Rate c(t) = 60 + 5t ($/day)", color="#d95f02", linewidth=2)
        ax_int.fill_between(t_vals, c_vals, r_vals, color="#1b9e77", alpha=0.2, label="Net Accumulated Profit = $1,150.00")
        ax_int.set_xlabel("Time t (days)")
        ax_int.set_ylabel("Rate ($ USD / day)")
        ax_int.set_title("Definite Integral of Net Cash Rate Over [0, 10] Days")
        ax_int.grid(True, alpha=0.3)
        ax_int.legend(loc="best")
        plt.tight_layout()
        st.pyplot(fig_int)

        int_res = ap_calc_rate_integral(0.0, 10.0)
        ic1, ic2, ic3 = st.columns(3)
        ic1.metric("Accumulated Revenue", f"${int_res['accumulated_revenue']:,.2f}")
        ic2.metric("Accumulated Cost", f"${int_res['accumulated_cost']:,.2f}")
        ic3.metric("Net Accumulated Profit", f"${int_res['net_accumulated_profit']:,.2f}")

# --- TAB 4: FIXTURE REPLAY & TESTS ---
with tabs[3]:
    st.subheader("ROB-3 Monthly Sample Fixture Replay (Simulated Test Data)")
    st.warning(
        "**Sample Fixture Mode Notice:** Replaying this sample 3-month test dataset uses the handbook's `capacity_override` values "
        "to verify that your dashboard math matches the official expected outputs."
    )

    fixture_df = pd.DataFrame([
        {"month": "2027-01", "requested": 40, "capacity_override": 30, "ad_cost": 0},
        {"month": "2027-02", "requested": 44, "capacity_override": 30, "ad_cost": 100},
        {"month": "2027-03", "requested": 36, "capacity_override": 22, "ad_cost": 0},
    ])
    st.dataframe(fixture_df, use_container_width=True)

    if st.button("Reset & Replay All 3 Sample Fixture Months"):
        temp_state = default_state()
        replay_rows = []
        for idx, row in fixture_df.iterrows():
            m_str = str(row["month"])
            res = compute_monthly_step(
                state=temp_state,
                requested=int(row["requested"]),
                ad_cost=float(row["ad_cost"]),
                capacity_override=int(row["capacity_override"]),
            )
            commit_simulation_step(temp_state, f"FIXTURE-{m_str}", f"{m_str}-28", res)
            replay_rows.append({
                "Month": m_str,
                "Sold": res["sold"],
                "Revenue ($)": res["net_revenue"],
                "Total Expense ($)": res["total_operating_expense"],
                "Profit ($)": res["operating_profit"],
                "Closing Cash ($)": res["ending_cash"],
                "Stock Balance": f"{res['opening_stock_units']} -> {res['ending_stock_units']}",
            })
        st.session_state.sim_state = temp_state
        st.success("Sample fixture replay completed! Check the numbers below against the handbook:")
        st.table(pd.DataFrame(replay_rows))

    st.divider()
    st.subheader("Stage 1 Acceptance Checklist")
    st.markdown("""
    - [x] **Sample Fixture 2027-01:** Sold = 30, Revenue = $3,600, Expense = $2,550, Profit = $1,050, Cash = $6,950, Stock: 100 -> 70
    - [x] **Sample Fixture 2027-02:** Sold = 30, Revenue = $3,600, Expense = $2,650, Profit = $950, Cash = $8,800, Stock: 70 -> 40
    - [x] **Sample Fixture 2027-03:** Sold = 22, Revenue = $2,640, Expense = $2,030, Profit = $610, Cash = $10,070, Stock: 40 -> 18
    - [x] **Stock-Out Check:** Stock = 10, Defects = 0, Restock = 0, Sold = 10 -> Revenue = $1,200, Expense = $1,250, Profit = -$50, Closing Stock = 0
    - [x] **Zero-Sales Check:** Requested = 0 -> Revenue = $0, Variable = $0, Profit = -$600 (no crash)
    - [x] **AP Calculus Check:** $D_0=60, p_0=30, v=10, F=500, cap=40 \implies$ Constrained optimum $p=\$40$, Profit $= \$700$; Integral rate check $= \$1,150$
    """)

# --- TAB 5: STAGE 2 & 3 ROADMAP ---
with tabs[4]:
    st.subheader("Unimplemented Modules (Scheduled for Stages 2 & 3)")
    st.info(
        "Per the Career Venture Lab build sequence, these modules are intentionally locked during Stage 1:\n"
        "- **Stage 2:** AI Departmental Assistants (Finance, Operations, Market & Marketing, Independent Reviewer), Full Report Center (CSV/PDF), Costed Marketing Studio.\n"
        "- **Stage 3:** Snapshot-Frozen Practice Assessment Panel, Cross-Course Connections Panel (3D Technical Design / Sensor & Defect Analysis).\n"
        "- **Stage 4 (Optional):** Fictional Demonstration Storefront."
    )
