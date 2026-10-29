import io
import math
import re
import textwrap
from datetime import date, datetime

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="📈", layout="wide")

INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn would live if current mortality patterns continued."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US dollars per person", "Gross domestic product divided by midyear population. This is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; it is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people, net total", "Number of immigrants minus emigrants over the period. It is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicide victims per 100,000 people."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a share of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US dollars", "Total reserves including gold, valued in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a share of GDP."),
}
COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
WB_BASE = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=20000"

CONTEXTS = {
    "Business": ("Exponential growth/decay", "time", "years", "revenue", "currency units", "exp(0.08*x)"),
    "Engineering": ("Quadratic model", "time", "seconds", "position", "meters", "x**2 - 4*x + 3"),
    "Medical": ("Exponential decay", "time", "hours", "concentration", "mg/L", "100*exp(-0.25*x)"),
    "Sports science": ("Sinusoidal movement", "time", "seconds", "height", "meters", "1.2*sin(2*pi*x)"),
    "Biology": ("Logistic-like growth (illustrative)", "time", "days", "population", "organisms", "100/(1+9*exp(-0.5*x))"),
    "Bioengineering": ("Sinusoidal signal", "time", "seconds", "signal", "mV", "2*sin(2*pi*x/5)"),
    "Economic forecasting": ("Quadratic trend", "time", "years", "indicator", "index units", "0.2*x**2 + 2*x + 10"),
    "Weather": ("Sinusoidal cycle", "time", "hours", "temperature", "°C", "10 + 5*sin(2*pi*x/24)"),
    "Custom": ("Custom function", "x", "units", "f(x)", "units", "x**2"),
}
VISUALS = ["Moving object / projectile", "Walking person with prosthetic leg", "Tank / concentration", "Population / cell colony", "Thermometer / weather", "Business / economic indicator"]
COURSES = ["Honors Algebra 2", "Precalculus", "Calculus"]

def wb_fetch(country_code, indicator_code, start_year, end_year):
    """Fetch authentic annual World Bank observations; no synthetic fallback."""
    url = WB_BASE.format(country=country_code, indicator=indicator_code, start=start_year, end=end_year)
    try:
        response = requests.get(url, timeout=18, headers={"User-Agent": "HispanicHeritageHistoricalDataExplorer/1.0"})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return pd.DataFrame(columns=["year", "value"]), f"World Bank returned no observation list. URL: {url}"
        rows = []
        for item in payload[1]:
            if item and item.get("date") and item.get("value") is not None:
                try:
                    rows.append({"year": int(item["date"]), "value": float(item["value"])})
                except (TypeError, ValueError):
                    continue
        frame = pd.DataFrame(rows, columns=["year", "value"]).drop_duplicates("year").sort_values("year")
        msg = None if not frame.empty else "The API returned no non-null observations for this selection."
        return frame, msg
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), f"Request timed out. Try again later. URL: {url}"
    except (requests.RequestException, ValueError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"World Bank request failed: {exc}. URL: {url}"

@st.cache_data(ttl=3600, show_spinner=False)
def cached_wb_fetch(country_code, indicator_code, start_year, end_year):
    return wb_fetch(country_code, indicator_code, start_year, end_year)

def validate_data(frame, indicator_name):
    errors = []
    if not {"year", "value"}.issubset(frame.columns):
        return ["The table must contain year and value columns."]
    years = pd.to_numeric(frame["year"], errors="coerce")
    values = pd.to_numeric(frame["value"], errors="coerce")
    if years.isna().any() or ((years % 1) != 0).any():
        errors.append("Every year must be a whole-number integer.")
    if years.dropna().duplicated().any():
        errors.append("Years must be unique; remove duplicate years.")
    if values.isna().any():
        errors.append("Every value must be numeric and non-empty.")
    clean_values = values.dropna()
    if not np.isfinite(clean_values.to_numpy(dtype=float)).all():
        errors.append("Values must be finite numbers.")
    if indicator_name in ("Life expectancy at birth", "Crude birth rate", "Intentional homicide rate", "Total reserves including gold") and (clean_values < 0).any():
        errors.append(f"{indicator_name} cannot be negative.")
    if indicator_name == "Life expectancy at birth" and (clean_values > 125).any():
        errors.append("Life expectancy above 125 years is implausible; verify the edited value.")
    if indicator_name == "Unemployment, national estimate" and ((clean_values < 0) | (clean_values > 100)).any():
        errors.append("Unemployment rate must be between 0 and 100.")
    if indicator_name == "Consumer price inflation" and (clean_values < -100).any():
        errors.append("Inflation below -100% is not valid for this indicator.")
    if indicator_name == "Central government debt" and (clean_values < 0).any():
        errors.append("Central government debt share cannot be negative in this app's validation rules.")
    if indicator_name == "External balance on goods and services" and ((clean_values < -100) | (clean_values > 100)).any():
        errors.append("External balance as % of GDP is expected between -100 and 100 in this app.")
    return errors

def fit_polynomial(frame, degree):
    data = frame.sort_values("year").copy()
    if len(data) < degree + 1 or data["year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree + 1} distinct valid years; only {data['year'].nunique()} are available.")
    base_year = int(data["year"].min())
    t = (data["year"].to_numpy(dtype=float) - base_year).reshape(-1, 1)
    y = data["value"].to_numpy(dtype=float)
    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
    model.fit(t, y)
    predicted = model.predict(t)
    r2 = float(r2_score(y, predicted)) if len(y) >= 2 else float("nan")
    mae = float(mean_absolute_error(y, predicted))
    return model, base_year, predicted, r2, mae

def polynomial_equation(model, base_year, degree, value_name="y"):
    linear = model.named_steps["linearregression"]
    intercept = float(linear.intercept_)
    coefs = np.asarray(linear.coef_, dtype=float)
    terms = [f"{intercept:.10g}"]
    for power, coefficient in enumerate(coefs, start=1):
        terms.append(f"{coefficient:+.10g}·(year − {base_year})^{power}")
    return f"{value_name} = " + " ".join(terms)

def safe_function(expr_text, x_symbol):
    """Parse a limited, explicit mathematical vocabulary; reject arbitrary Python."""
    expr_text = expr_text.strip().replace("^", "**")
    if not expr_text or len(expr_text) > 220:
        raise ValueError("Enter a non-empty function shorter than 220 characters.")
    allowed = {"x": x_symbol, "pi": sp.pi, "E": sp.E, "sin": sp.sin, "cos": sp.cos,
               "tan": sp.tan, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
               "abs": sp.Abs, "ln": sp.log}
    try:
        expr = sp.sympify(expr_text, locals=allowed, evaluate=True)
    except Exception as exc:
        raise ValueError("Could not parse this expression. Use notation such as x**2, sin(x), exp(-x), log(x), or 1/(x-2).") from exc
    if expr.free_symbols - {x_symbol}:
        raise ValueError("Only the variable x and supported functions/constants are allowed.")
    if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("The expression contains an undefined or infinite constant.")
    return expr

def numeric_func(expr, x_symbol):
    fn = sp.lambdify(x_symbol, expr, modules=["numpy"])
    def evaluate(values):
        with np.errstate(all="ignore"):
            result = np.asarray(fn(values), dtype=float)
        if result.ndim == 0:
            result = np.full_like(np.asarray(values, dtype=float), float(result), dtype=float)
        return result
    return evaluate

def fmt_num(value, digits=6):
    try:
        if not np.isfinite(float(value)):
            return "undefined / non-finite"
        return f"{float(value):,.{digits}g}"
    except (TypeError, ValueError):
        return str(value)

def calculus_analysis(expr, x, lo, hi, x0):
    derivative = sp.diff(expr, x)
    second = sp.diff(derivative, x)
    f = numeric_func(expr, x)
    fp = numeric_func(derivative, x)
    fpp = numeric_func(second, x)
    a, b = min(float(lo), float(hi)), max(float(lo), float(hi))
    notes = []
    # Numerical roots are candidates only; classification uses sign checks.
    candidates = []
    try:
        roots = sp.solveset(sp.Eq(derivative, 0), x, domain=sp.Interval(a, b))
        if isinstance(roots, sp.FiniteSet):
            candidates.extend([float(sp.N(root)) for root in roots if abs(float(sp.im(sp.N(root)))) < 1e-8])
    except Exception:
        pass
    grid = np.linspace(a, b, 1201)
    with np.errstate(all="ignore"):
        grid_fp = fp(grid)
        grid_fpp = fpp(grid)
    # Add sign-change approximations for roots not solved symbolically.
    for vals, target in ((grid_fp, candidates), (grid_fpp, [])):
        for i in range(len(grid) - 1):
            if np.isfinite(vals[i]) and np.isfinite(vals[i+1]) and vals[i] * vals[i+1] < 0:
                target.append(float((grid[i] + grid[i+1]) / 2))
    critical = []
    for c in sorted(set(round(v, 7) for v in candidates if a < v < b)):
        eps = max(1e-5, (b-a)*1e-5)
        left, right = float(fp(c-eps)), float(fp(c+eps))
        if not np.isfinite(left) or not np.isfinite(right):
            kind = "classification uncertain (near domain issue)"
        elif left > 0 and right < 0:
            kind = "local maximum (f′ changes + to −)"
        elif left < 0 and right > 0:
            kind = "local minimum (f′ changes − to +)"
        else:
            kind = "not a local extremum by first-derivative sign test"
        critical.append((c, kind, float(f(c))))
    # Increasing/decreasing sampled sign intervals.
    segments = []
    cuts = [a] + [c for c, _, _ in critical] + [b]
    for left, right in zip(cuts[:-1], cuts[1:]):
        mid = (left + right) / 2
        slope = float(fp(mid))
        trend = "increasing" if np.isfinite(slope) and slope > 1e-8 else "decreasing" if np.isfinite(slope) and slope < -1e-8 else "approximately flat / unresolved"
        segments.append((left, right, trend))
    # Concavity-change candidates are checked on both sides.
    inflection = []
    try:
        roots2 = sp.solveset(sp.Eq(second, 0), x, domain=sp.Interval(a, b))
        cands2 = [float(sp.N(r)) for r in roots2] if isinstance(roots2, sp.FiniteSet) else []
    except Exception:
        cands2 = []
    for c in sorted(set(round(v, 7) for v in cands2 if a < v < b)):
        eps = max(1e-5, (b-a)*1e-5)
        left, right = float(fpp(c-eps)), float(fpp(c+eps))
        if np.isfinite(left) and np.isfinite(right) and left * right < 0:
            inflection.append((c, float(f(c)), "confirmed: concavity changes"))
        else:
            inflection.append((c, float(f(c)), "not confirmed: no sampled concavity change"))
    # Absolute extrema on closed interval when the function is defined throughout the candidates.
    candidates_ext = [(a, float(f(a)), "left endpoint"), (b, float(f(b)), "right endpoint")]
    candidates_ext += [(c, val, kind) for c, kind, val in critical]
    finite_ext = [(xx, yy, label) for xx, yy, label in candidates_ext if np.isfinite(yy)]
    absolute = None
    if finite_ext:
        absolute = (min(finite_ext, key=lambda z: z[1]), max(finite_ext, key=lambda z: z[1]))
    return derivative, second, f, fp, fpp, critical, segments, inflection, absolute

def make_chart(observed, model, base_year, degree, sample_increment, estimate_rows, title, units, x_min, x_max, model_expr=None, x_symbol=None, show_derivatives=False, x0=None):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if not observed.empty:
        ax.scatter(observed["year"], observed["value"], s=25, label="Observed / working data", zorder=4)
        ax.axvline(observed["year"].min(), linestyle=":", linewidth=1, label="Historical boundary")
        years = observed["year"].to_numpy(dtype=int)
        sampled = years[::sample_increment]
        if len(years) and years[-1] not in sampled:
            sampled = np.append(sampled, years[-1])
        if model is not None:
            t_sample = (sampled - base_year).reshape(-1, 1)
            ax.scatter(sampled, model.predict(t_sample), marker="x", s=35, label=f"Model markers (every {sample_increment} year(s))")
            dense = np.linspace(min(x_min, observed["year"].min()), max(x_max, observed["year"].max()), 500)
            pred = model.predict((dense - base_year).reshape(-1, 1))
            boundary = observed["year"].max()
            histmask = dense <= boundary
            if histmask.any():
                ax.plot(dense[histmask], pred[histmask], linewidth=2, label="Fitted curve")
            if (~histmask).any():
                ax.plot(dense[~histmask], pred[~histmask], linestyle="--", linewidth=2, label="Polynomial extrapolation")
    elif model_expr is not None:
        dense = np.linspace(x_min, x_max, 500)
        fn = numeric_func(model_expr, x_symbol)
        values = fn(dense)
        values[~np.isfinite(values)] = np.nan
        ax.plot(dense, values, linewidth=2, label="Function")
    if estimate_rows:
        valid = [(r["year"], r["estimate"]) for r in estimate_rows if np.isfinite(r["estimate"])]
        if valid:
            ax.scatter([v[0] for v in valid], [v[1] for v in valid], marker="*", s=100, label="Calculated estimates", zorder=5)
    ax.set_title(title)
    ax.set_xlabel("Year / independent variable")
    ax.set_ylabel(units)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig

def build_pdf(summary_lines, chart_png, data_preview, results_preview):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=28, leftMargin=28, topMargin=25, bottomMargin=25)
    styles = getSampleStyleSheet()
    story = [Paragraph("Hispanic Heritage Historical Data Explorer — Report", styles["Title"]), Spacer(1, 8)]
    for line in summary_lines:
        story.append(Paragraph(str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["BodyText"]))
        story.append(Spacer(1, 4))
    if chart_png:
        story.extend([Spacer(1, 8), RLImage(io.BytesIO(chart_png), width=650, height=345)])
    for title, frame in [("Data preview", data_preview), ("Calculated years", results_preview)]:
        story.extend([Spacer(1, 10), Paragraph(title, styles["Heading2"])])
        if frame is not None and not frame.empty:
            table_data = [list(map(str, frame.columns))] + frame.head(18).astype(str).values.tolist()
            tbl = Table(table_data, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#d9eaf7")),
                ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
                ("FONTSIZE", (0,0), (-1,-1), 7),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
            ]))
            story.append(tbl)
        else:
            story.append(Paragraph("No rows available.", styles["BodyText"]))
    doc.build(story)
    return buffer.getvalue()

st.title("📈 Hispanic Heritage Historical Data Explorer")
st.caption("Authentic World Bank history + editable scenarios + polynomial modeling + interactive function and calculus lab.")
with st.expander("Research setup and data notes", expanded=False):
    st.write("Research question (customize): How have selected economic, demographic, and social indicators changed across selected Latin American countries?")
    st.write("Brazil is included as a Latin American comparison country. Latin America is a geographic-cultural region; Brazil is Latin American even though Portuguese, not Spanish, is its principal language.")
    st.write("The World Bank API supplies the listed indicators. Missing API observations remain missing; the app does not invent data.")
    st.write("Suggested research question: How does the historical trend in [indicator] differ between [country] and [comparison country], and how reliable is a polynomial model outside the observed years?")

with st.sidebar:
    st.header("Data & model controls")
    country_label = st.selectbox("Featured country", list(COUNTRIES), index=0)
    indicator_name = st.selectbox("World Bank indicator", list(INDICATORS), index=0)
    comparison_label = st.selectbox("Comparison country", ["None"] + [c for c in COUNTRIES if c != country_label], index=1)
    degree = st.slider("Polynomial degree (3–8)", 3, 8, 3)
    sample_increment = st.slider("Model marker / sample-table increment (years)", 1, 10, 5)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    printer_mode = st.toggle("Printer-friendly report view", value=False)
    course = st.selectbox("Mathematics pathway", COURSES, index=2)
    st.divider()
    st.subheader("Function / context lab")
    context = st.selectbox("Context", list(CONTEXTS), index=0)
    visual = st.selectbox("Contextual visual simulation", VISUALS, index=0)
    st.caption("Visuals are symbolic classroom illustrations, not medical, engineering, or forecasting advice.")

country_code = COUNTRIES[country_label]
indicator_code, units, definition = INDICATORS[indicator_name]
end_year = date.today().year - 1
start_year = date.today().year - 70
cache_key = f"{country_code}|{indicator_code}|{start_year}|{end_year}"
raw, api_message = cached_wb_fetch(country_code, indicator_code, start_year, end_year)
if "source_frame" not in st.session_state or st.session_state.get("source_key") != cache_key:
    st.session_state.source_frame = raw.copy()
    st.session_state.working_frame = raw.copy()
    st.session_state.source_key = cache_key
    st.session_state.data_edited = False
    st.session_state.calc_result = None
    st.session_state.calc_signature = None
if api_message:
    st.warning(api_message)
source = st.session_state.source_frame.copy()
st.caption(f"Source: World Bank API · Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M')} · Requested years: {start_year}–{end_year} inclusive.")
st.write(f"**Indicator definition:** {definition} **Units:** {units} **API code:** `{indicator_code}`")
st.markdown(f"**Source link:** [World Bank indicator page](https://data.worldbank.org/indicator/{indicator_code}) · [API endpoint](https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?date={start_year}%3A{end_year}&format=json&per_page=20000)")
st.info(f"Actual source coverage: {int(source['year'].min()) if not source.empty else 'none'}–{int(source['year'].max()) if not source.empty else 'none'}; {len(source)} non-null observations returned. The requested window is 70 years, but API coverage may be shorter.")

tab_data, tab_model, tab_function, tab_exports = st.tabs(["1 · Data", "2 · Historical model", "3 · Function & calculus lab", "4 · Export & verification"])
with tab_data:
    st.subheader("Source data and editable working copy")
    st.write("The source table is preserved separately. Edit the working table to explore a user-modified scenario; this does not alter the World Bank source.")
    if source.empty:
        st.error("No source observations are available for this country/indicator/window. Choose another indicator/country or retry after checking the API.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Original source data (read-only)**")
        st.dataframe(source, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Editable working data**")
        edited = st.data_editor(
            st.session_state.working_frame,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "year": st.column_config.NumberColumn("year", step=1, format="%d"),
                "value": st.column_config.NumberColumn("value", format="%.8g"),
            },
            key=f"editor_{cache_key}",
        )
        if not edited.equals(st.session_state.working_frame):
            st.session_state.working_frame = edited.copy()
            st.session_state.data_edited = True
            st.session_state.calc_result = None
    b1, b2 = st.columns([1, 3])
    with b1:
        if st.button("Reset to source", use_container_width=True):
            st.session_state.working_frame = source.copy()
            st.session_state.data_edited = False
            st.session_state.calc_result = None
            st.rerun()
    working = st.session_state.working_frame.copy()
    errors = validate_data(working, indicator_name)
    if errors:
        for error in errors:
            st.error(error)
    elif st.session_state.data_edited:
        st.warning("USER-MODIFIED SCENARIO: calculations and exports use edited values, not just the source data.")
    else:
        st.success("Working data currently matches the source data.")
    st.caption("Validation is applied before modeling. For transparent classroom use, indicator-specific range checks are conservative and should be reviewed if editing unusual historical values.")
    if not working.empty and not errors:
        st.write(f"Working observations: {len(working)} · mean: {working['value'].mean():,.6g} {units} · range: {int(working['year'].min())}–{int(working['year'].max())}")

with tab_model:
    st.subheader("Polynomial model of the working data")
    working = st.session_state.working_frame.copy()
    errors = validate_data(working, indicator_name)
    model = None
    fit_error = None
    if errors:
        st.error("Correct the working table errors before fitting a model.")
    elif working.empty:
        st.warning("There is no data to fit.")
    else:
        try:
            model, base_year, fitted, r2, mae = fit_polynomial(working, degree)
            working_sorted = working.sort_values("year").reset_index(drop=True)
            working_sorted["fitted_value"] = fitted
            working_sorted["residual"] = working_sorted["value"] - fitted
            equation = polynomial_equation(model, base_year, degree)
            actual_min, actual_max = int(working["year"].min()), int(working["year"].max())
            st.markdown(f"**Model equation (t = year − {base_year}):**  \n`{equation}`")
            metrics = st.columns(5)
            metrics[0].metric("Degree", degree)
            metrics[1].metric("R² · in-sample", f"{r2:.4f}")
            metrics[2].metric("MAE · in-sample", f"{mae:.4g} {units}")
            metrics[3].metric("Observation count", len(working))
            metrics[4].metric("Mean observed", f"{working['value'].mean():.5g}")
            st.write(f"**Historical range:** {actual_min}–{actual_max}. **Base year:** {base_year}. All fit metrics above are in-sample.")
            st.dataframe(working_sorted[["year", "value", "fitted_value", "residual"]], use_container_width=True, hide_index=True)
            if len(working_sorted) >= max(10, degree + 4):
                holdout_n = max(2, min(5, len(working_sorted)//5))
                train = working_sorted.iloc[:-holdout_n]
                test = working_sorted.iloc[-holdout_n:]
                if len(train) >= degree + 1:
                    try:
                        hold_model, hold_base, _, _, _ = fit_polynomial(train[["year", "value"]], degree)
                        hold_pred = hold_model.predict((test["year"].to_numpy(dtype=float)-hold_base).reshape(-1,1))
                        st.write(f"**Chronological holdout MAE:** {mean_absolute_error(test['value'], hold_pred):.5g} {units} on the last {len(test)} observations (separate from in-sample MAE).")
                    except Exception as exc:
                        st.caption(f"Holdout comparison unavailable: {exc}")
            else:
                st.caption("Chronological holdout comparison omitted: too few observations for a useful split at this degree.")
            display_end = actual_max + horizon
            estimates = []
            if st.session_state.calc_result and st.session_state.calc_signature == (cache_key, degree, tuple(map(tuple, working[["year","value"]].to_numpy()))):
                estimates = st.session_state.calc_result
            fig = make_chart(working, model, base_year, degree, sample_increment, estimates, f"{country_label}: {indicator_name}", units, actual_min, display_end)
            st.pyplot(fig, use_container_width=True)
            if horizon > 0:
                future = np.arange(actual_max + 1, actual_max + horizon + 1)
                fut = model.predict((future-base_year).reshape(-1,1))
                st.caption("Dashed line indicates extrapolation beyond the latest observation. Polynomial forecasts can become implausible rapidly; they are not causal explanations or guaranteed forecasts.")
                if np.any(~np.isfinite(fut)) or np.any(np.abs(fut) > max(1e6, np.nanmax(np.abs(working["value"])) * 1e4)):
                    st.warning("The extrapolated curve includes extreme or non-finite values. Treat the forward segment as mathematically generated, not credible forecasting.")
            st.markdown("**Interpretation & limitations**")
            st.write(f"The model summarizes a mathematical trend in {indicator_name.lower()} for {country_label}. A high in-sample R² does not prove causation or guarantee future accuracy. Residuals show observed minus fitted values; MAE is the average absolute in-sample error.")
            st.write("Baseline uncertainty summary: use the residual table and MAE to describe typical fit error. This app does not label a simple ±MAE band as a 95% prediction interval.")
            if course == "Honors Algebra 2":
                st.write("**Algebra 2 focus:** domain is the real-valued year input for the polynomial; its range depends on degree and coefficients. End behavior is controlled by the leading term. Compare average rates of change between years to describe graphical trends.")
            elif course == "Precalculus":
                st.write("**Precalculus focus:** degree controls possible turning-point complexity; coefficients shape scale and curvature. Changes in average rates of change help describe where the graph bends or changes trend. A degree-d polynomial has at most d−1 turning points.")
            else:
                st.write("**Calculus focus:** use the Function & calculus lab to inspect derivatives, tangent/secant lines, critical points, extrema, concavity, and inflection points for a chosen function.")
            st.session_state.model_frame = working_sorted
            st.session_state.model_equation = equation
            st.session_state.model_metrics = {"r2": r2, "mae": mae, "base_year": base_year, "actual_min": actual_min, "actual_max": actual_max}
            st.session_state.model_object = model
            plt.close(fig)
        except Exception as exc:
            fit_error = str(exc)
            st.error(f"Model could not be fitted: {fit_error}")
    if comparison_label != "None":
        comp_code = COUNTRIES[comparison_label]
        comp_raw, comp_msg = cached_wb_fetch(comp_code, indicator_code, start_year, end_year)
        st.markdown(f"**Same-indicator comparison: {comparison_label}**")
        if comp_msg:
            st.warning(f"Comparison data note: {comp_msg}")
        if not comp_raw.empty:
            st.write(f"Coverage: {int(comp_raw.year.min())}–{int(comp_raw.year.max())}, {len(comp_raw)} observations. Gaps can reflect missing country-years; do not assume unmatched years are zero.")
            fig2, ax2 = plt.subplots(figsize=(9, 4))
            ax2.plot(comp_raw.year, comp_raw.value, marker="o", markersize=3, label=comparison_label)
            if not working.empty and not errors:
                ax2.plot(working.year, working.value, marker="o", markersize=3, label=country_label)
            ax2.set_title(f"{indicator_name}: country comparison")
            ax2.set_xlabel("Year")
            ax2.set_ylabel(units)
            ax2.grid(True, alpha=.25)
            ax2.legend()
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)
        else:
            st.info("No comparison observations are available for this selection.")

with tab_function:
    st.subheader("Interactive function and calculus lab")
    st.write("This lab supports symbolic polynomial, rational, exponential, logarithmic, and trigonometric expressions. It is separate from the historical-year model unless you explicitly select the fitted-data function.")
    use_fitted = st.checkbox("Use the generated polynomial from the historical-data fit", value=False, disabled=not hasattr(st.session_state, "model_equation"))
    default_suggestion = CONTEXTS[context][5]
    if context == "Economic forecasting":
        default_suggestion = "0.2*x**2 + 2*x + 10"
    custom_default = default_suggestion
    if use_fitted and hasattr(st.session_state, "model_object"):
        # Model is in centered year variable; function lab x represents actual year.
        m = st.session_state.model_object
        base = st.session_state.model_metrics["base_year"]
        linear = m.named_steps["linearregression"]
        coefs = np.asarray(linear.coef_, dtype=float)
        # Convert model prediction into a readable SymPy polynomial in x.
        x_tmp = sp.Symbol("x", real=True)
        expr_fitted = sp.Float(linear.intercept_)
        for power, coef in enumerate(coefs, start=1):
            expr_fitted += sp.Float(coef) * (x_tmp - base) ** power
        custom_default = str(sp.expand(expr_fitted))
    suggested = st.text_input("Function f(x) (use x, + - * / **, parentheses, sin, cos, tan, exp, log/ln, sqrt, abs)", value=custom_default, key=f"func_text_{context}_{use_fitted}_{indicator_name}")
    c1, c2, c3 = st.columns(3)
    with c1:
        x_name = st.text_input("Independent variable name", value=CONTEXTS[context][1])
        x_units = st.text_input("Independent variable units", value=CONTEXTS[context][2])
    with c2:
        y_name = st.text_input("Dependent variable name", value=CONTEXTS[context][3])
        y_units = st.text_input("Dependent variable units", value=CONTEXTS[context][4])
    with c3:
        context_description = st.text_input("Context description", value=f"{context}: illustrative model for learning function behavior.")
        analysis_lo = st.number_input("Analysis interval start", value=0.0, step=1.0, key="analysis_lo")
        analysis_hi = st.number_input("Analysis interval end", value=10.0, step=1.0, key="analysis_hi")
    if analysis_lo == analysis_hi:
        st.warning("Choose distinct analysis interval endpoints.")
    x = sp.Symbol("x", real=True)
    try:
        expr = safe_function(suggested, x)
        derivative = sp.diff(expr, x)
        second = sp.diff(derivative, x)
        st.markdown(f"**Function:** `f(x) = {sp.sstr(expr)}`")
        st.markdown(f"**First derivative:** `f′(x) = {sp.sstr(derivative)}`")
        st.markdown(f"**Second derivative:** `f″(x) = {sp.sstr(second)}`")
        lo, hi = min(analysis_lo, analysis_hi), max(analysis_lo, analysis_hi)
        if hi - lo > 1e5:
            st.error("Choose an analysis interval no wider than 100,000 units.")
            raise ValueError("Interval is too wide.")
        current_x = st.slider("Current input x", min_value=float(lo), max_value=float(hi), value=float(np.clip(2.0, lo, hi)), step=max((hi-lo)/200, 0.01), key=f"current_x_{context}")
        r1, r2, r3 = st.columns(3)
        with r1:
            secant_a = st.slider("Average-rate interval start a", float(lo), float(hi), float(lo), step=max((hi-lo)/200, 0.01), key=f"secant_a_{context}")
        with r2:
            secant_b = st.slider("Average-rate interval end b", float(lo), float(hi), float(hi), step=max((hi-lo)/200, 0.01), key=f"secant_b_{context}")
        with r3:
            speed = st.select_slider("Animation speed", options=["Slow", "Medium", "Fast"], value="Medium")
        play = st.toggle("▶ Play point animation", value=False)
        if st.button("Reset moving point"):
            st.session_state[f"current_x_{context}"] = float(np.clip(2.0, lo, hi))
            st.rerun()
        if play:
            # Streamlit reruns are used for simple automatic playback.
            import time
            st.session_state[f"current_x_{context}"] = lo if current_x >= hi else min(hi, current_x + max((hi-lo)/30, 0.05))
            st.caption("Playback advances on reruns. Toggle off to pause; Reset moving point returns to the starting position.")
            time.sleep({"Slow": 0.8, "Medium": 0.35, "Fast": 0.12}[speed])
            st.rerun()
        f = numeric_func(expr, x)
        fp = numeric_func(derivative, x)
        fpp = numeric_func(second, x)
        y0, slope0, curve0 = float(f(current_x)), float(fp(current_x)), float(fpp(current_x))
        st.markdown(f"**At x = {current_x:.5g}:** f(x) = {fmt_num(y0)} {y_units}; f′(x) = {fmt_num(slope0)} {y_units}/{x_units}; f″(x) = {fmt_num(curve0)} {y_units}/{x_units}².")
        if np.isfinite(slope0) and np.isfinite(y0):
            st.write(f"Tangent line at x={current_x:.5g}: y = {fmt_num(y0)} + ({fmt_num(slope0)})(x − {current_x:.5g}).")
        if secant_a != secant_b:
            fa, fb = float(f(secant_a)), float(f(secant_b))
            avg_rate = (fb-fa)/(secant_b-secant_a) if np.isfinite(fa) and np.isfinite(fb) else float("nan")
            st.write(f"Average rate of change from {secant_a:.5g} to {secant_b:.5g}: ({fmt_num(fb)} − {fmt_num(fa)}) / ({secant_b:.5g} − {secant_a:.5g}) = **{fmt_num(avg_rate)} {y_units}/{x_units}**.")
            st.write(f"Secant line: y = {fmt_num(fa)} + ({fmt_num(avg_rate)})(x − {secant_a:.5g}).")
        else:
            st.info("Average rate of change is undefined when interval endpoints are equal.")
        if course == "Calculus":
            deriv, second_deriv, ff, ffp, ffpp, critical, segments, inflection, absolute = calculus_analysis(expr, x, lo, hi, current_x)
            st.markdown("#### Calculus analysis on the selected interval")
            st.write("Critical points (numerical classifications use derivative sign checks; symbolic roots are reported when available):")
            if critical:
                st.dataframe(pd.DataFrame(critical, columns=["x", "classification", "f(x)"]), hide_index=True, use_container_width=True)
            else:
                st.write("No interior critical points were identified on this interval.")
            st.write("Increasing/decreasing intervals (sign sampled between critical candidates):")
            st.dataframe(pd.DataFrame(segments, columns=["start", "end", "behavior"]), hide_index=True, use_container_width=True)
            st.write("Concavity and inflection candidates:")
            if inflection:
                st.dataframe(pd.DataFrame(inflection, columns=["x", "f(x)", "test"]), hide_index=True, use_container_width=True)
            else:
                st.write("No verified interior inflection point identified. A zero of f″ alone is not enough.")
            if absolute:
                st.write(f"On the closed interval, among defined endpoints and detected critical points: absolute minimum candidate x={absolute[0][0]:.6g}, f(x)={fmt_num(absolute[0][1])}; absolute maximum candidate x={absolute[1][0]:.6g}, f(x)={fmt_num(absolute[1][1])}. This numerical search is not a proof when domain gaps or missed critical points exist.")
            st.write(f"**Domain/discontinuity note:** expression `{sp.sstr(expr)}` may have restrictions. The graph masks undefined/non-finite values so lines are not intentionally connected across gaps. For rational/log functions, inspect denominator zeros and logarithm arguments. Symbolic domain: `{sp.sstr(sp.calculus.util.continuous_domain(expr, x, sp.Interval(lo, hi))) if hasattr(sp.calculus.util, 'continuous_domain') else 'inspect expression restrictions'}`.")
        elif course == "Precalculus":
            st.write("**Precalculus focus:** compare average rates on subintervals to see whether change is speeding up or slowing down. Polynomial degree affects possible turns; coefficients affect scale, direction, and curvature.")
        else:
            st.write("**Honors Algebra 2 focus:** inspect domain, range over the selected interval, graph shape, end behavior, and average rate of change. The derivative readout is optional enrichment.")
        st.markdown("#### Interactive graph")
        show_f = st.checkbox("Show f(x)", value=True)
        show_fp = st.checkbox("Show f′(x)", value=(course == "Calculus"))
        show_fpp = st.checkbox("Show f″(x)", value=False)
        grid = np.linspace(lo, hi, 1000)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        if show_f:
            yy = f(grid); yy[~np.isfinite(yy)] = np.nan
            ax.plot(grid, yy, label="f(x)")
        if show_fp:
            yy = fp(grid); yy[~np.isfinite(yy)] = np.nan
            ax.plot(grid, yy, label="f′(x)")
        if show_fpp:
            yy = fpp(grid); yy[~np.isfinite(yy)] = np.nan
            ax.plot(grid, yy, label="f″(x)")
        if np.isfinite(y0):
            ax.scatter([current_x], [y0], marker="o", s=75, label="Moving point")
        if np.isfinite(y0) and np.isfinite(slope0):
            xx = np.array([max(lo, current_x-(hi-lo)*0.15), min(hi, current_x+(hi-lo)*0.15)])
            ax.plot(xx, y0+slope0*(xx-current_x), linestyle="--", label="Tangent line")
        if secant_a != secant_b:
            fa, fb = float(f(secant_a)), float(f(secant_b))
            if np.isfinite(fa) and np.isfinite(fb):
                sec_slope = (fb-fa)/(secant_b-secant_a)
                xx = np.array([lo, hi])
                ax.plot(xx, fa+sec_slope*(xx-secant_a), linestyle=":", label="Secant line")
                ax.scatter([secant_a, secant_b], [fa, fb], s=35, label="Secant endpoints")
        ax.set_xlabel(f"{x_name} ({x_units})")
        ax.set_ylabel(f"{y_name} ({y_units})")
        ax.set_title(f"{context}: {context_description}")
        ax.grid(True, alpha=.25); ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        # Simple symbolic visual that responds to the current function value.
        st.markdown("#### Contextual simulation (scaled classroom illustration)")
        st.caption("The indicator maps the current function value to a display scale only; it is not a physical simulation unless your function and units define one.")
        value_scale = float(np.clip((y0 if np.isfinite(y0) else 0), -1000, 1000))
        normalized = float(np.clip((value_scale + 1000)/2000, 0, 1))
        if visual == "Moving object / projectile":
            st.write(f"🚀 Object position indicator: {normalized*100:.1f}% across the display. Current model output: {fmt_num(y0)} {y_units}.")
            st.progress(normalized)
        elif visual == "Walking person with prosthetic leg":
            st.write(f"🚶 Prosthetic-leg walking illustration: gait indicator is scaled to {normalized*100:.1f}% from the model output. This is a symbolic classroom visual, not a medical assessment.")
            st.progress(normalized)
        elif visual == "Tank / concentration":
            st.write(f"🧪 Tank/concentration indicator: {normalized*100:.1f}% display fill. Actual concentration interpretation depends on your entered units and function.")
            st.progress(normalized)
        elif visual == "Population / cell colony":
            st.write(f"🦠 Population/cell-colony indicator: {normalized*100:.1f}% display scale.")
            st.progress(normalized)
        elif visual == "Thermometer / weather":
            st.write(f"🌡️ Thermometer indicator: scaled model reading {normalized*100:.1f}% of the display. Set y units to °C/°F if appropriate.")
            st.progress(normalized)
        else:
            st.write(f"📊 Business/economic indicator: {normalized*100:.1f}% display scale; model output is {fmt_num(y0)} {y_units}.")
            st.progress(normalized)
        # Explicit context interpretation with dimensions.
        st.markdown("#### Interpret in context")
        st.write(f"**Function value:** {y_name} is {fmt_num(y0)} {y_units} when {x_name} is {current_x:.5g} {x_units}.")
        st.write(f"**First derivative:** {fmt_num(slope0)} {y_units} per {x_units} describes the instantaneous rate of change at this point.")
        st.write(f"**Second derivative:** {fmt_num(curve0)} {y_units} per {x_units}² describes how the rate itself is changing; positive values indicate upward concavity where defined, negative values downward concavity.")
        if context == "Medical":
            st.caption("For a concentration model, the first derivative represents concentration change per time and the second derivative represents change in that rate. This is educational only, not dosing guidance.")
        if context == "Engineering" and y_name.lower() == "position":
            st.caption("If f is position versus time, f′ is velocity and f″ is acceleration, with units m/s and m/s² when the inputs are seconds and meters.")
    except Exception as exc:
        st.error(f"Function or analysis error: {exc}")

with tab_exports:
    st.subheader("Export current selection and scenario")
    working = st.session_state.working_frame.copy()
    valid = not validate_data(working, indicator_name) and not working.empty
    model_frame = getattr(st.session_state, "model_frame", pd.DataFrame())
    if st.session_state.data_edited:
        st.warning("Exports will be labeled USER-MODIFIED SCENARIO.")
    metadata = {
        "country": country_label, "indicator": indicator_name, "indicator_code": indicator_code,
        "units": units, "retrieved_date": datetime.now().strftime("%Y-%m-%d"),
        "requested_years": f"{start_year}-{end_year}", "data_edited": st.session_state.data_edited,
        "degree": degree, "comparison_country": comparison_label, "course": course,
    }
    if valid:
        csv_bytes = working.sort_values("year").to_csv(index=False).encode("utf-8")
        st.download_button("Download working data CSV", data=csv_bytes, file_name="hispanic_heritage_working_data.csv", mime="text/csv")
        xlsx_buffer = io.BytesIO()
        with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
            pd.DataFrame([metadata]).to_excel(writer, sheet_name="Summary", index=False)
            working.sort_values("year").to_excel(writer, sheet_name="Working Data", index=False)
            if not model_frame.empty:
                model_frame.to_excel(writer, sheet_name="Model Residuals", index=False)
            result_rows = st.session_state.calc_result or []
            pd.DataFrame(result_rows).to_excel(writer, sheet_name="Calculated Years", index=False)
        st.download_button("Download Excel workbook (.xlsx)", data=xlsx_buffer.getvalue(), file_name="hispanic_heritage_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        summary = [
            f"Country: {country_label}; indicator: {indicator_name} ({indicator_code}); units: {units}.",
            f"Definition: {definition}",
            f"Requested years: {start_year}–{end_year}; actual non-null coverage: {int(working.year.min())}–{int(working.year.max())}, {len(working)} observations.",
            f"Retrieved: {metadata['retrieved_date']}; edited scenario: {st.session_state.data_edited}; course: {course}; degree: {degree}.",
        ]
        if hasattr(st.session_state, "model_equation"):
            summary.append(f"Polynomial equation: {st.session_state.model_equation}")
        if hasattr(st.session_state, "model_metrics"):
            summary.append(f"In-sample R²: {st.session_state.model_metrics['r2']:.6g}; in-sample MAE: {st.session_state.model_metrics['mae']:.6g} {units}.")
        fig_for_pdf = None
        if hasattr(st.session_state, "model_object"):
            mm = st.session_state.model_object
            bm = st.session_state.model_metrics["base_year"]
            chart = make_chart(working, mm, bm, degree, sample_increment, st.session_state.calc_result or [], f"{country_label}: {indicator_name}", units, int(working.year.min()), int(working.year.max())+horizon)
            png = io.BytesIO()
            chart.savefig(png, format="png", dpi=140, bbox_inches="tight")
            fig_for_pdf = png.getvalue()
            plt.close(chart)
        pdf_bytes = build_pdf(summary, fig_for_pdf, working.sort_values("year"), pd.DataFrame(st.session_state.calc_result or []))
        st.download_button("Download PDF report", data=pdf_bytes, file_name="hispanic_heritage_report.pdf", mime="application/pdf")
    else:
        st.info("Correct validation errors and provide at least one observation to enable data exports.")
    st.markdown("#### Calculate single or batch years")
    st.write("Historical-data prediction years use the fitted polynomial. Results are retained across reruns and cleared when country, indicator, degree, or working data changes.")
    actual_signature = (cache_key, degree, tuple(map(tuple, working[["year","value"]].to_numpy())) if {"year","value"}.issubset(working.columns) else ())
    if st.session_state.get("calc_signature") != actual_signature:
        st.session_state.calc_result = None
        st.session_state.calc_signature = actual_signature
    year_single = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=2025, step=1, key="year_single")
    calc_col1, calc_col2 = st.columns(2)
    with calc_col1:
        if st.button("Calculate Value", disabled=not valid):
            try:
                m, base, _, _, _ = fit_polynomial(working, degree)
                val = float(m.predict(np.array([[float(year_single-base)]]))[0])
                within = int(working.year.min()) <= int(year_single) <= int(working.year.max())
                observed_match = working.loc[working.year.astype(int) == int(year_single), "value"]
                estimates = st.session_state.calc_result or []
                estimates = [r for r in estimates if r.get("year") != int(year_single)]
                estimates.append({"year": int(year_single), "estimate": val, "units": units,
                                  "classification": "within fitted historical range" if within else "outside fitted historical range",
                                  "status": "fitted value at observed year (not the actual observation)" if not observed_match.empty else "model estimate"})
                st.session_state.calc_result = sorted(estimates, key=lambda r: r["year"])
                st.session_state.calc_signature = actual_signature
                st.rerun()
            except Exception as exc:
                st.error(f"Could not calculate: {exc}")
    with calc_col2:
        batch_mode = st.radio("Batch input type", ["Year range", "Comma-separated custom years"], horizontal=True)
    batch_rows = []
    if batch_mode == "Year range":
        ba, bb = st.columns(2)
        with ba:
            batch_start = st.number_input("Batch start year", 1950, 2100, 2000, step=1)
        with bb:
            batch_end = st.number_input("Batch end year", 1950, 2100, 2025, step=1)
        batch_step = st.slider("Range step (1–10 years)", 1, 10, 5)
        requested_years = list(range(int(min(batch_start, batch_end)), int(max(batch_start, batch_end))+1, batch_step))
    else:
        custom_year_text = st.text_input("Custom years", value="1950, 2000, 2025, 2050")
        try:
            requested_years = []
            for token in custom_year_text.split(","):
                token = token.strip()
                if not token:
                    continue
                if not re.fullmatch(r"[+-]?\d+", token):
                    raise ValueError(f"'{token}' is not an integer year.")
                year_value = int(token)
                if not 1950 <= year_value <= 2100:
                    raise ValueError(f"Year {year_value} is outside the allowed 1950–2100 range.")
                requested_years.append(year_value)
            requested_years = sorted(set(requested_years))
        except ValueError as exc:
            requested_years = []
            st.error(f"Invalid custom years: {exc}")
    if st.button("Calculate batch years", disabled=not valid):
        try:
            if not requested_years:
                raise ValueError("Enter at least one valid year.")
            m, base, _, _, _ = fit_polynomial(working, degree)
            for yr in requested_years:
                val = float(m.predict(np.array([[float(yr-base)]]))[0])
                within = int(working.year.min()) <= yr <= int(working.year.max())
                observed_match = working.loc[working.year.astype(int) == yr, "value"]
                batch_rows.append({"year": yr, "estimate": val, "units": units,
                                   "classification": "within fitted historical range" if within else "outside fitted historical range",
                                   "status": "fitted value at observed year (not the actual observation)" if not observed_match.empty else "model estimate"})
            existing = {r["year"]: r for r in (st.session_state.calc_result or [])}
            existing.update({r["year"]: r for r in batch_rows})
            st.session_state.calc_result = sorted(existing.values(), key=lambda r: r["year"])
            st.session_state.calc_signature = actual_signature
            st.rerun()
        except Exception as exc:
            st.error(f"Batch calculation failed: {exc}")
    if st.session_state.calc_result:
        st.markdown("**Saved calculated-year results**")
        result_df = pd.DataFrame(st.session_state.calc_result)
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        if st.button("Clear calculated-year results"):
            st.session_state.calc_result = None
            st.rerun()
    else:
        st.caption("No calculated-year results yet. Calculate a single year or batch to populate the table and export sheets.")

st.divider()
st.caption("Educational research tool. Historical associations and polynomial fits do not establish causation. Always cite the World Bank indicator definition and report missing coverage.")
