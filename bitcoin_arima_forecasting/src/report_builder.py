"""Build the academic report (DOCX) from the saved results: python -m src.report_builder

Tables and quoted numbers are read from results/.  The interpretive prose was
written for one data snapshot, so the builder refuses to run on any other.
"""
import io
import json

import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from . import config
from .preprocessing import clean_data

REPORT_SHA256 = "247e3fa4cf7763584543337cc004507351c8f6bbb1a6201fe023f785d99829d4"
REPORT_PATH = config.ROOT / "report" / "bitcoin_arima_assessment_report.docx"


class Report:
    def __init__(self):
        self.doc = Document()
        self.figure_n = self.table_n = self.equation_n = 0
        section = self.doc.sections[0]
        section.page_width, section.page_height = Inches(8.27), Inches(11.69)
        section.left_margin = section.right_margin = Inches(1)
        section.top_margin = section.bottom_margin = Inches(0.9)
        normal = self.doc.styles["Normal"]
        normal.font.name, normal.font.size = "Calibri", Pt(11)
        normal.paragraph_format.space_after = Pt(6)
        normal.paragraph_format.line_spacing = 1.12
        for name in ("Heading 1", "Heading 2"):
            self.doc.styles[name].font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self._field(footer, "PAGE", "1")

    @staticmethod
    def _field(paragraph, instruction, placeholder):
        for kind, text in (("begin", None), (None, instruction), ("separate", None),
                           ("text", placeholder), ("end", None)):
            run = paragraph.add_run()
            if kind == "text":
                run.text = text
                continue
            element = OxmlElement("w:instrText" if kind is None else "w:fldChar")
            if kind is None:
                element.text = text
            else:
                element.set(qn("w:fldCharType"), kind)
            run._r.append(element)

    def h(self, text, level=1):
        self.doc.add_heading(text, level)

    def p(self, *paragraphs):
        for text in paragraphs:
            self.doc.add_paragraph(text)

    def bullets(self, items, style="List Bullet"):
        for item in items:
            self.doc.add_paragraph(item, style=style)

    def equation(self, latex):
        self.equation_n += 1
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, f"${latex}$     ({self.equation_n})", fontsize=12)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        buffer.seek(0)
        height, width = plt.imread(buffer).shape[:2]
        buffer.seek(0)
        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.add_run().add_picture(buffer, width=Inches(min(width / 300, 6.0)))

    def figure(self, filename, caption, width=6.1):
        self.figure_n += 1
        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.keep_with_next = True
        paragraph.add_run().add_picture(str(config.OUTPUT_DIR / filename), width=Inches(width))
        self._caption(f"Figure {self.figure_n}. {caption}")

    def _caption(self, text):
        paragraph = self.doc.add_paragraph(text, style="Caption")
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def table(self, frame, caption, size=8.5):
        self.table_n += 1
        self._caption(f"Table {self.table_n}. {caption}")
        self.doc.paragraphs[-1].paragraph_format.keep_with_next = True
        table = self.doc.add_table(rows=1, cols=len(frame.columns))
        table.style, table.alignment = "Light Grid Accent 1", WD_TABLE_ALIGNMENT.CENTER
        rows = [list(frame.columns)] + frame.astype(str).values.tolist()
        for index, values in enumerate(rows):
            cells = table.rows[0].cells if index == 0 else table.add_row().cells
            for cell, value in zip(cells, values):
                cell.text = str(value)
                paragraph = cell.paragraphs[0]
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.runs[0].font.size = Pt(size)
        header = OxmlElement("w:tblHeader")
        table.rows[0]._tr.get_or_add_trPr().append(header)
        self.doc.add_paragraph()


def usd(x):
    return f"${x:,.2f}"


def pv(x):
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def build() -> None:
    s = json.loads((config.RESULTS_DIR / "analysis_summary.json").read_text())
    if s["metadata"]["csv_sha256"] != REPORT_SHA256:
        raise SystemExit("The data snapshot differs from the one this report text describes. "
                         "Tables and figures are current in results/ and outputs/, but the "
                         "narrative in src/report_builder.py must be reviewed first.")
    read = lambda name: pd.read_csv(config.RESULTS_DIR / name)
    check = json.loads((config.RESULTS_DIR / "source_cross_check.json").read_text())
    price, _ = clean_data(pd.read_csv(config.DATA_CSV))
    ev, cand, grid = read("model_evaluation.csv"), read("candidate_comparison.csv"), read("arima_grid_search.csv")
    stat, fc, detail = read("stationarity_tests.csv"), read("seven_day_forecast.csv"), read("test_predictions_one_step.csv")
    meta, q, e, sp, f = s["metadata"], s["quality"], s["eda"], s["split"], s["forecast"]
    sens, dtr, dfu = s["sensitivity"], s["diagnostics_train"], s["diagnostics_full"]
    name, other = s["selected"]["name"], sens["model"]
    assert name == "ARIMA(0, 1, 0)" and other == "ARIMA(6, 1, 4)"

    def m(evaluation, model, column):
        return ev[(ev.evaluation == evaluation) & (ev.model == model)][column].iloc[0]

    def c(model, column):
        return cand[cand.model == model][column].iloc[0]

    def metric_table(frame):
        out = frame[["evaluation", "model", "n", "MAE", "RMSE", "MAPE_%", "sMAPE_%", "Directional_accuracy_%"]].copy()
        for col in ("MAE", "RMSE"):
            out[col] = out[col].map("{:,.0f}".format)
        for col in ("MAPE_%", "sMAPE_%"):
            out[col] = out[col].map("{:.2f}".format)
        out["Directional_accuracy_%"] = out["Directional_accuracy_%"].map(
            lambda v: "n/a" if pd.isna(v) else f"{v:.1f}")
        out.columns = ["Evaluation", "Model", "n", "MAE", "RMSE", "MAPE %", "sMAPE %", "Dir. acc. %"]
        return out

    r = Report()
    d = r.doc
    # 1. Title page
    for text, size, bold, gap in (
            ("Forecasting the Seven-Day Price Movement of Bitcoin with ARIMA", 22, True, 130),
            ("A reproducible time-series analysis of Blockchain.com daily market-price data", 13, False, 20),
            ("Student name: [enter name]\nStudent ID: [enter ID]\nModule: [enter module code and title]\n"
             "Institution: [enter institution]\nLecturer: [enter name]\nSubmission date: [enter date]", 12, False, 90),
            (f"Data window: {q['start_date']} to {q['end_date']}\nData retrieved: {meta['retrieved_utc']}", 11, False, 60)):
        paragraph = d.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_before = Pt(gap)
        run = paragraph.add_run(text)
        run.font.size, run.bold = Pt(size), bold
    d.add_page_break()

    r.h("Contents")
    for item in ["Abstract", "Introduction", "Problem Statement", "Research Objectives",
                 "Dataset Description and Data Source", "Methodology", "Data Preprocessing",
                 "Exploratory Data Analysis", "Stationarity Testing", "ACF and PACF Analysis",
                 "ARIMA Model Development", "Training and Testing", "Model Evaluation",
                 "Residual Diagnostics", "Seven-Day Forecast Results", "Discussion", "Limitations",
                 "Conclusion", "References", "Appendix: Test-Period Predictions"]:
        d.add_paragraph(item).paragraph_format.space_after = Pt(1)
    d.add_page_break()

    r.h("Abstract")
    r.p(f"This study forecasts the Bitcoin market price for seven days using {q['clean_rows']} daily observations "
        f"from Blockchain.com covering {q['start_date']} to {q['end_date']}. Over this window the price moved from "
        f"{usd(e['first_price'])} to {usd(e['last_price'])} ({e['period_change_pct']:.1f}%), with annualised return "
        f"volatility of {e['annualised_volatility_pct']:.1f}%. Unit-root tests support one difference of the log "
        f"price (d = 1). A grid of {s['grid_models']} ARIMA specifications was searched on the first "
        f"{sp['train_n']} observations, and a shortlist was compared by rolling-origin validation inside the training "
        f"sample. After residual and root-admissibility screening, the selected model was {name}, a random walk "
        f"without drift. On {sp['test_n']} unseen observations it gave a one-step MAPE of "
        f"{m('1-step walk-forward', name, 'MAPE_%'):.2f}% and RMSE of {usd(m('1-step walk-forward', name, 'RMSE'))}. "
        "These figures are identical to those of the naive forecast because the two methods are mathematically the "
        f"same. The forecast for {fc.forecast_date.iloc[0]} to {fc.forecast_date.iloc[-1]} is therefore flat at "
        f"{usd(f['origin_price'])} (classified Neutral), with a 95% interval on the seventh day of "
        f"{usd(f['lower_day7'])} to {usd(f['upper_day7'])}. The analysis finds no reliable evidence that past prices "
        "alone predict the direction of Bitcoin over this horizon. It is not investment advice.")

    r.h("Introduction")
    r.p("Bitcoin is a decentralised digital asset that trades continuously and is known for large price swings. "
        "Short-horizon forecasting is attractive but difficult: if prices rapidly incorporate available information, "
        "past prices should contain little exploitable information about future changes (Fama, 1970). Studies of "
        "exchange rates have shown that a random walk can be a demanding benchmark for structural and time-series "
        "models (Meese and Rogoff, 1983). Financial returns also commonly show heavy tails and volatility clustering "
        "(Cont, 2001).",
        "The autoregressive integrated moving average (ARIMA) family, developed in the Box-Jenkins tradition "
        "(Box et al., 2015), is a transparent linear framework for modelling serial dependence. This report applies "
        "it end to end and judges it against simple benchmarks, so that apparent accuracy is not mistaken for "
        "predictive skill.")

    r.h("Problem Statement")
    r.p("The assessment task is to predict the price movement of Bitcoin for the next seven days from the past "
        "year of historical prices, using ARIMA in Python. The analytical question is whether a linear model of past "
        "daily prices indicates a direction and size of movement that can be distinguished from random fluctuation.")

    r.h("Research Objectives")
    r.bullets(["Obtain and validate one year of daily Bitcoin market-price data from a documented source.",
               "Describe the trend, return distribution and volatility of the series.",
               "Determine the differencing order from statistical evidence.",
               "Identify, compare and diagnose candidate ARIMA models without using test data for selection.",
               "Evaluate out-of-sample accuracy against naive, drift and moving-average benchmarks.",
               "Produce a seven-day forecast with 95% prediction intervals and interpret its uncertainty."], "List Number")

    r.h("Dataset Description and Data Source")
    r.p(f"The data come from the Blockchain.com chart shown at {meta['chart_page']}. They were retrieved from the "
        f"official chart-data API behind that chart (not scraped from the web page) on {meta['retrieved_utc']}, using "
        f"the request {meta['request_url']}. The API names the metric '{meta['metric_name']}' and defines it as: "
        f"'{meta['metric_description']}' Values are in {meta['unit']} with a period of one {meta['period']}. The API "
        "response does not state which exchanges or weights are used. The unmodified response and the CSV file "
        f"bitcoin_historical_data.csv are preserved; the CSV SHA-256 checksum is {meta['csv_sha256']}.",
        "As an independent plausibility check (not used in modelling), the series was compared with Coinbase "
        f"BTC-USD daily closes. A Blockchain.com value stamped day D differs from the Coinbase close of day D-1 by "
        f"{check['previous_day_close']['mean_abs_diff_pct']:.3f}% on average (maximum "
        f"{check['previous_day_close']['max_abs_diff_pct']:.2f}%, {check['previous_day_close']['days']} days), "
        f"against {check['same_date']['mean_abs_diff_pct']:.2f}% when the same dates are matched. This indicates, "
        "although the API does not document it, that each stamp behaves like a price at 00:00 UTC. Forecast dates "
        "below follow the same convention.")
    ends = pd.concat([price.head(), price.tail()]).reset_index()
    ends.columns = ["Date", "Market price"]
    ends["Date"], ends["Market price"] = ends.Date.dt.strftime("%Y-%m-%d"), ends["Market price"].map(usd)
    ends.insert(0, "Position", ["First five"] * 5 + ["Last five"] * 5)
    r.table(ends, "First and last five observations")
    r.table(pd.DataFrame({"Property": ["Observations", "Date range", "Missing values", "Missing calendar dates",
                                       "Duplicate dates", "Non-positive values", "Data types"],
                          "Result": [q["clean_rows"], f"{q['start_date']} to {q['end_date']}", q["non_numeric_values"],
                                     q["missing_calendar_dates"], q["duplicate_dates"], q["non_positive_values"],
                                     "date: datetime; price: float64; raw timestamp: integer"]}), "Data validation summary")

    r.h("Methodology")
    r.p("Let P(t) denote the price on day t and y(t) its natural logarithm. The log transform makes changes "
        "proportional, moderates scale-dependent variance and guarantees positive forecasts. With B the backshift "
        "operator, an ARIMA(p, d, q) model is")
    r.equation(r"\phi(B)\,(1-B)^d\,y_t = c + \theta(B)\,\varepsilon_t")
    r.equation(r"\phi(B)=1-\phi_1B-\cdots-\phi_pB^p,\quad \theta(B)=1+\theta_1B+\cdots+\theta_qB^q")
    r.p("where the errors are assumed to be white noise (Gaussian for likelihood and interval calculations). "
        "Parameters were estimated by exact maximum likelihood in statsmodels (Seabold and Perktold, 2010). A "
        "constant in a once-differenced model represents drift. Because forecasts are made on the log scale, the "
        "exponentiated point forecast is the conditional median of price. Under log-normality the conditional mean "
        "is higher:")
    r.equation(r"\hat P^{\,med}_{T+h}=e^{\hat y_{T+h}},\quad \hat P^{\,mean}_{T+h}=e^{\hat y_{T+h}+\hat\sigma_h^2/2}")
    r.equation(r"95\%\ \mathrm{interval}:\ \exp(\hat y_{T+h}\pm 1.96\,\hat\sigma_h)")
    r.p("The intervals reflect future-error variance under the fitted model. They do not include uncertainty about "
        "which model is correct or, in the standard state-space calculation, parameter estimation uncertainty.")

    r.h("Data Preprocessing")
    r.p("Unix timestamps were converted to UTC dates, observations sorted chronologically, and prices checked to be "
        "numeric and positive. Duplicate dates would be resolved by keeping the latest timestamp. The series was "
        "re-indexed to a complete daily calendar. The declared rule for any gap was forward filling, which uses only "
        "past information and therefore avoids look-ahead leakage. No duplicates, gaps or invalid values were found, "
        "so no observation was altered or filled. Outliers were retained because extreme moves are genuine features "
        "of the market and removing them would understate risk.")

    r.h("Exploratory Data Analysis")
    desc = read("descriptive_statistics.csv").rename(columns={"Unnamed: 0": "Statistic"})
    for col in desc.columns[1:]:
        desc[col] = desc[col].map("{:,.2f}".format)
    r.table(desc, "Descriptive statistics")
    r.figure("historical_price.png", "Daily Bitcoin market price over the past year")
    r.p(f"The series is not level around a constant mean. The maximum was {usd(e['max_price'])} on {e['max_date']} "
        f"and the minimum {usd(e['min_price'])} on {e['min_date']}; the largest peak-to-trough decline was "
        f"{abs(e['max_drawdown_pct']):.1f}%. Visible phases include a sharp fall from late January to early February "
        "2026, a recovery to May, a decline in early June, a rapid rise in late August, and a pronounced fall in "
        "November. These are descriptions of the price path; this study does not analyse their causes.")
    r.figure("daily_returns.png", "Daily percentage returns")
    r.p(f"Mean daily return was {e['mean_return_pct']:.3f}% with standard deviation {e['sd_return_pct']:.2f}%. There "
        f"were {e['up_days']} rising and {e['down_days']} falling days. The largest fall was "
        f"{e['largest_falls'][0][1]:.2f}% ({e['largest_falls'][0][0]}) and the largest gain "
        f"{e['largest_gains'][0][1]:.2f}% ({e['largest_gains'][0][0]}), on consecutive days. Excess kurtosis of "
        f"{e['excess_kurtosis_return']:.2f} shows far heavier tails than a normal distribution, and large moves "
        "appear in clusters.")
    r.figure("daily_changes.png", "Daily absolute price changes")
    r.p("Absolute changes tell the same story in dollars: quiet stretches are interrupted by bursts of several "
        "thousand dollars per day. Because a given percentage move is worth more dollars at a higher price, the log "
        "scale is preferred for modelling.")
    r.figure("rolling_statistics.png", "30-day rolling mean and rolling standard deviation")
    r.p(f"The rolling mean drifts substantially and lags turning points. The rolling standard deviation ranged from "
        f"{usd(e['rolling_sd_min'])} ({e['rolling_sd_min_date']}) to {usd(e['rolling_sd_max'])} "
        f"({e['rolling_sd_max_date']}). It measures dispersion of price levels, so it rises during trends as well as "
        "during erratic trading. Both statistics change over time, which is evidence against stationarity.")
    r.figure("price_distribution.png", "Distribution of daily prices and daily returns")
    r.p("The price histogram has several clusters that correspond to different market phases; it describes time "
        "spent at each level and should not be read as a stable probability distribution. Returns are centred near "
        "zero, sharply peaked and heavy-tailed.")
    r.figure("monthly_average_price.png", "Monthly average price (asterisks mark partial months)")
    monthly = pd.DataFrame(e["monthly"], columns=["Month", "Mean price", "Days"])
    monthly["Mean price"] = monthly["Mean price"].map(usd)
    r.table(monthly, "Monthly average price")
    r.p("Monthly means were highest in January 2026 and lowest in June 2026. The first and last months are partial "
        "and are not directly comparable with full months.")

    r.h("Stationarity Testing")
    r.p("ARIMA assumes that the differenced series is stationary: its mean and autocovariance do not depend on time. "
        "Otherwise estimated relationships are unstable and can be spurious. Two complementary tests were used. "
        "The augmented Dickey-Fuller (ADF) test (Dickey and Fuller, 1979; Said and Dickey, 1984) estimates")
    r.equation(r"\Delta y_t=\alpha+\beta t+\gamma y_{t-1}+\sum_{i=1}^{k}\delta_i\Delta y_{t-i}+\varepsilon_t")
    r.p("ADF null hypothesis H0: gamma = 0 (unit root, non-stationary). Alternative H1: gamma < 0 (stationary around "
        "the included deterministic terms). Lag length k was chosen by AIC. The KPSS test (Kwiatkowski et al., 1992) "
        "reverses the hypotheses: H0 is level (or trend) stationarity and H1 is a unit root. The significance level "
        "was 5%. KPSS p-values are interpolated from a table and truncated to the range 0.01 to 0.10.")
    table = stat.copy()
    table["statistic"] = table.statistic.map("{:.3f}".format)
    table["p_value"] = [pv(v) if t == "ADF" else ("≥0.100" if v >= 0.1 else "≤0.010" if v <= 0.01 else f"{v:.3f}")
                        for v, t in zip(table.p_value, table.test)]
    table["critical"] = [f"{a:.3f} / {b:.3f} / {x:.3f}" for a, b, x in zip(table.crit_1pct, table.crit_5pct, table.crit_10pct)]
    table = table[["sample", "series", "test", "deterministic", "statistic", "p_value", "critical", "conclusion_5pct"]]
    table.columns = ["Sample", "Series", "Test", "Terms", "Statistic", "p", "Critical 1/5/10%", "5% result"]
    r.table(table, "ADF and KPSS results", 7)
    r.figure("differencing.png", "Log price before and after first differencing")
    pick = lambda sample, series, test: stat[(stat["sample"] == sample) & (stat.series == series)
                                             & (stat.test == test) & (stat.deterministic == "constant")].iloc[0]
    a0, a1 = pick("Training", "Log price", "ADF"), pick("Training", "Differenced log price", "ADF")
    r.p(f"For the training log price, ADF does not reject a unit root (statistic {a0.statistic:.3f}, p = "
        f"{a0.p_value:.3f}) and KPSS rejects stationarity. After one difference ADF rejects the unit root (statistic "
        f"{a1.statistic:.3f}, p < 0.001) and KPSS does not reject stationarity. The full sample agrees. The smallest "
        "order that passes is therefore d = 1; further differencing was not applied, as it would risk "
        "over-differencing. The order was determined on training data only. Mean stationarity does not imply "
        "constant variance, which is examined in the residual diagnostics.")

    r.h("ACF and PACF Analysis")
    r.p("The autocorrelation function (ACF) measures correlation between a series and its lag k. The partial "
        "autocorrelation function (PACF) measures that correlation after removing intermediate lags. In textbook "
        "cases a pure AR(p) process has a PACF that cuts off after lag p, and a pure MA(q) process has an ACF that "
        "cuts off after lag q.")
    r.figure("acf_pacf.png", "ACF and PACF of the differenced training log price")
    cor = s["correlation"]
    lags = lambda key: ", ".join(f"lag {i} ({v:.3f})" for i, v in cor[key])
    r.p(f"With n = {cor['n']}, the approximate 95% bound is ±{cor['bound']:.3f}. ACF values outside it: "
        f"{lags('significant_acf')}. PACF values outside it: {lags('significant_pacf')}. Lags 1 to 6 are all "
        "insignificant, which points to little short-term linear dependence and is consistent with p = q = 0. The "
        "lag-7 spike could reflect a weekly effect, so the grid was extended to p, q ≤ 7.",
        "These plots have clear limits. With 30 lags, about one or two spikes would be expected outside 95% bounds "
        "by chance. The bounds assume independent, constant-variance noise, which is doubtful here. Mixed ARMA "
        "processes produce no clean cut-off. The plots therefore guided the search range but did not choose the model.")

    r.h("ARIMA Model Development")
    r.p(f"All combinations of p = 0 to {config.MAX_P}, d = 1 and q = 0 to {config.MAX_Q}, with and without drift, "
        f"were estimated on the training sample ({s['grid_models']} models; {s['grid_converged']} converged). Fit was "
        "compared by the Akaike (1974) and Schwarz (1978) criteria, where k is the number of parameters, n the "
        "sample size and L the maximised likelihood:")
    r.equation(r"AIC=2k-2\ln\hat L,\qquad BIC=k\ln n-2\ln\hat L")
    top = pd.concat([grid.nsmallest(5, "aic").assign(ranking="AIC"), grid.nsmallest(5, "bic").assign(ranking="BIC")])
    top = pd.DataFrame({"Ranking": top.ranking, "Model": top.model, "AIC": top.aic.map("{:.2f}".format),
                        "BIC": top.bic.map("{:.2f}".format), "Min. root": top.min_root_modulus.map(
                            lambda v: "none" if v == float("inf") else f"{v:.5f}"),
                        "Converged": top.converged})
    r.table(top, "Five best grid models by AIC and by BIC (training sample)")
    r.p(f"The criteria disagree. AIC prefers large models such as ARIMA(6, 1, 5), while BIC, which penalises "
        f"complexity more heavily, prefers {name}. A stepwise pmdarima auto_arima search also returned "
        f"{s['auto_arima']}; it follows the automatic-search idea of Hyndman and Khandakar (2008) and was treated as "
        "one candidate rather than an authority.", )
    r.h("Selection rule", 2)
    r.bullets(["Shortlist the five best AIC models, the five best BIC models, the auto_arima result and the random walk.",
               f"Reject models that did not converge or fail the Ljung-Box test at 5% (lags {', '.join(map(str, config.LB_LAGS))}, "
               "where the lag exceeds p + q).",
               f"Reject models with any AR or MA root of modulus below {config.MIN_ROOT_MODULUS}.",
               "Among the rest, take the lowest seven-day rolling-origin RMSE over the last "
               f"{sp['validation_n']} training observations (from {sp['validation_start']}); within "
               f"{config.PARSIMONY_TOLERANCE:.0%} of the best, take the fewest parameters."], "List Number")
    comp = pd.DataFrame({"Model": cand.model, "Params": cand.n_parameters, "LB min p": cand.min_ljung_box_p.map(pv),
                         "Converged": cand.converged, "Roots admissible": cand.admissible_roots,
                         "Val. RMSE 1-day": cand.val_RMSE_1step.map("{:,.0f}".format),
                         "Val. RMSE 7-day": cand.val_RMSE_7day.map("{:,.0f}".format)})
    r.table(comp, "Shortlisted models: diagnostics and validation accuracy (Params includes the error variance)", 8)
    r.h("A disclosed revision to the rule", 2)
    r.p(f"The first run of this analysis did not include the root screen and selected {other}, whose seven-day "
        f"validation RMSE ({c(other, 'val_RMSE_7day'):,.0f}) was "
        f"{(1 - c(other, 'val_RMSE_7day') / c(name, 'val_RMSE_7day')) * 100:.1f}% below that of {name} "
        f"({c(name, 'val_RMSE_7day'):,.0f}). Inspection showed that its smallest MA root modulus was "
        f"{c(other, 'min_root_modulus'):.6f}, essentially on the unit circle. This indicates a non-invertible, "
        "probably over-parameterised fit: moving-average standard errors were extremely large in the training fit, "
        f"and on the full sample its residuals failed Ljung-Box at lag 20 (p = "
        f"{sens['diagnostics_full']['ljung_box']['20']['p_value']:.3f}). The threshold of 1.001 follows Hyndman and "
        "Khandakar (2008). The screen itself uses only training-sample estimates, but it was added after the first "
        "results, including test results, had been seen. The selection is therefore not fully blind. To show the "
        f"effect of this decision, {other} is retained throughout as a sensitivity analysis.")
    r.p(f"The final specification is {name}:")
    r.equation(r"y_t=y_{t-1}+\varepsilon_t,\qquad \hat y_{T+h}=y_T,\qquad \sigma_h^2=h\,\sigma^2")
    sig = dfu["parameters"]["sigma2"]
    r.p(f"It has one estimated parameter, the error variance (full-sample estimate {sig[0]:.6f}, standard error "
        f"{sig[1]:.6f}, equivalent to a daily log-price standard deviation of {sig[0] ** 0.5 * 100:.2f}%). It is "
        "appropriate here because it has the lowest BIC, agrees with auto_arima, passes the pre-set diagnostic "
        "screen and has the lowest validation RMSE among admissible candidates. 'Appropriate' means that the data "
        "do not justify a more complex linear mean model; it does not mean the model describes all features of the series.")

    r.h("Training and Testing")
    r.p(f"The split was chronological: training {sp['train_start']} to {sp['train_end']} (n = {sp['train_n']}, 80%) "
        f"and testing {sp['test_start']} to {sp['test_end']} (n = {sp['test_n']}, 20%). Random shuffling would let "
        "a model learn from observations later than those it predicts and would destroy the lag structure, a form "
        "of data leakage that overstates accuracy. Three test designs were used (Tashman, 2000):")
    r.bullets(["One-step walk-forward: refit on all data up to each day and predict the next day.",
               f"Seven-day rolling origin: refit at {sp['test_n'] - 6} successive origins and predict one to seven days "
               f"ahead ({(sp['test_n'] - 6) * 7} overlapping forecasts), matching the assignment horizon.",
               f"Static multi-step: one fit on the training set forecasting all {sp['test_n']} test days."])
    r.p("The model order stayed fixed during testing; only parameters were re-estimated. The optimiser reported a "
        f"convergence warning at {s['walk_forward_nonconverged']} one-step origins. For the random walk the point "
        "forecast is the last observation regardless of the variance estimate, so this does not affect point accuracy.")

    r.h("Model Evaluation")
    r.p("For actual values A(t), forecasts F(t) and n forecasts:")
    r.equation(r"MAE=\frac{1}{n}\sum|A_t-F_t|,\qquad RMSE=\sqrt{\frac{1}{n}\sum(A_t-F_t)^2}")
    r.equation(r"MAPE=\frac{100}{n}\sum\frac{|A_t-F_t|}{|A_t|},\qquad sMAPE=\frac{100}{n}\sum\frac{2|A_t-F_t|}{|A_t|+|F_t|}")
    r.equation(r"DA=\frac{100}{m}\sum 1[\,\mathrm{sign}(F_t-P_o)=\mathrm{sign}(A_t-P_o)\,]")
    r.p("P(o) is the price at the forecast origin and m counts forecasts that state a direction. MAE is the typical "
        "dollar error. RMSE penalises large errors more. MAPE and sMAPE express errors relative to price. MAPE is "
        "stable here because prices are far from zero (Hyndman and Koehler, 2006 discuss its weaknesses), but a low "
        "MAPE on price levels mostly reflects the fact that tomorrow's price is close to today's. It is not evidence "
        "of skill. Directional accuracy is undefined for a forecast of no change.")
    main = ev[ev.role.isin(["selected", "baseline"])]
    r.table(metric_table(main), "Test accuracy of the selected model and baselines", 8)
    r.figure("test_predictions.png", "Test-period forecasts and 95% prediction intervals")
    moves = s["test_moves"]
    r.p(f"The selected model and the naive baseline are identical, so their errors match exactly. The drift baseline "
        f"had slightly lower RMSE (one-step {m('1-step walk-forward', 'Drift', 'RMSE'):,.0f} against "
        f"{m('1-step walk-forward', name, 'RMSE'):,.0f}; seven-day {m('7-day rolling origin', 'Drift', 'RMSE'):,.0f} "
        f"against {m('7-day rolling origin', name, 'RMSE'):,.0f}). The test period ended in a strong decline, and "
        f"{moves['down']} of {moves['up'] + moves['down']} one-day moves were downward; a persistently negative drift "
        "benefits from such a period, so its directional score is not evidence of general forecasting ability. The "
        "seven-day moving average was clearly worst on error size.",
        "In the one-step plot the forecast is the actual series shifted by one day. The visual closeness must not be "
        f"mistaken for prediction. The static forecast (MAPE {m('Static multi-step', name, 'MAPE_%'):.2f}%) shows how "
        "quickly information decays: it remains at the last training price while the market rises and then falls.",
        f"Empirical coverage of the nominal 95% interval was {s['coverage']['one_step']:.1f}% one day ahead and "
        f"{s['coverage']['weekly_by_horizon']['7']:.1f}% seven days ahead. The forecasts overlap and the sample is "
        "short, but the seven-day intervals appear somewhat too narrow. Static coverage of "
        f"{s['coverage']['static']:.0f}% comes from a single forecast path and is not informative.")
    first = detail.head(10)
    pred = pd.DataFrame({"Date": first.date, "Actual": first.actual.map(usd), "Predicted": first.predicted.map(usd),
                         "Absolute error": first.absolute_error.map(usd), "Percentage error": first.percentage_error.map("{:.2f}%".format)})
    r.table(pred, "Actual and predicted prices, first ten test days (all days in the Appendix)")
    r.h("Comparison with other candidates (post-hoc)", 2)
    post = ev[ev.role.str.startswith("comparison") & (ev.model != name)]
    r.table(metric_table(post), "Test accuracy of the other shortlisted models (not used for selection)", 7.5)
    dm = s["diebold_mariano"]
    smallest = min(v["p_value"] for k, v in dm.items() if v["p_value"] is not None)
    r.p("Some more complex models, notably ARIMA(4, 1, 4) and ARIMA(4, 1, 5), had lower test errors than the random "
        "walk. Both failed the pre-set screen, and choosing them now would be selection on the test set. "
        "Diebold-Mariano tests on one-step squared errors with the small-sample correction (Diebold and Mariano, "
        "1995; Harvey et al., 1997) found no model significantly different from the naive forecast (smallest p = "
        f"{smallest:.3f}). For {other}, p = {dm[other]['p_value']:.3f}.")

    r.h("Residual Diagnostics")
    r.p("An adequate mean model should leave residuals resembling white noise. The Ljung and Box (1978) statistic, "
        "with H0 of no autocorrelation up to lag h, is")
    r.equation(r"Q=n(n+2)\sum_{k=1}^{h}\frac{\hat\rho_k^2}{n-k}\ \sim\ \chi^2_{h-p-q}")
    r.figure("model_diagnostics.png", f"Residual diagnostics, {name}, training sample")
    r.figure("model_diagnostics_full_sample.png", f"Residual diagnostics, {name}, full sample")
    rows = []
    for label, x in ((f"{name} training", dtr), (f"{name} full", dfu),
                     (f"{other} training", sens["diagnostics_train"]), (f"{other} full", sens["diagnostics_full"])):
        lb = x["ljung_box"]
        rows.append([label, pv(lb["10"]["p_value"]) if "10" in lb else "n/a", pv(lb["20"]["p_value"]),
                     pv(x["mean_t_p"]), f"{x['excess_kurtosis']:.2f}", pv(x["jarque_bera"][1]), pv(x["arch_lm"][1])])
    r.table(pd.DataFrame(rows, columns=["Model and sample", "LB(10) p", "LB(20) p", "Mean=0 p", "Excess kurt.",
                                        "Jarque-Bera p", "ARCH LM p"]),
            "Residual tests (LB(10) is undefined when p + q = 10)", 8)
    r.p(f"For {name}, Ljung-Box does not reject at 5% on either sample, and the mean residual is not distinguishable "
        f"from zero. The full-sample result is borderline (lag 10 p = {dfu['ljung_box']['10']['p_value']:.3f}), and "
        "the ACF shows an isolated lag-7 spike, so weak dependence cannot be excluded. The residuals are not "
        "independent, identically distributed Gaussian noise: Jarque-Bera strongly rejects normality, and both the "
        "ARCH test (Engle, 1982) and the autocorrelation of squared residuals show volatility clustering. The model "
        "is therefore acceptable for the linear mean, but inadequate as a description of risk. Constant-variance "
        "Gaussian intervals will be too narrow in turbulent periods and too wide in calm ones.")

    r.h("Seven-Day Forecast Results")
    r.p(f"{name} was refitted to all {q['clean_rows']} observations. The forecast origin is {f['origin_date']} "
        f"at {usd(f['origin_price'])}.")
    out = pd.DataFrame({"Date": fc.forecast_date, "Forecast (USD)": fc.predicted_price_usd.map(usd),
                        "Lower 95%": fc.lower_95_usd.map(usd), "Upper 95%": fc.upper_95_usd.map(usd),
                        "Daily change": fc.expected_daily_change_pct.map("{:.2f}%".format), "Direction": fc.direction})
    r.table(out, "Seven-day forecast")
    r.figure("seven_day_forecast.png", "Historical price, seven-day forecast and 95% prediction interval")
    r.p(f"Classification: {f['classification']}. A daily change smaller than {config.DAILY_NEUTRAL_PCT:.2f}% in "
        f"absolute value is labelled Neutral, and a seven-day change smaller than {config.WEEKLY_NEUTRAL_PCT:.0f}% "
        "is classed as Neutral overall (larger rises are Bullish, larger falls Bearish). These thresholds are "
        "reporting conventions chosen for this study, not statistical tests.",
        f"The expected change is {f['change_pct']:.2f}% and there are no daily fluctuations in the point forecast, "
        "because a driftless random walk assigns the same median to every future day. Actual prices will fluctuate. "
        f"The interval widens from {usd(f['width_day1'])} on day one to {usd(f['width_day7'])} on day seven "
        f"({f['width_day7_pct']:.1f}% of the forecast). The log-normal mean on day seven is "
        f"{usd(f['mean_adjusted_day7'])}; its excess over the median arises from the variance correction and is not "
        "a directional signal.")
    st = pd.DataFrame(sens["table"])
    r.table(pd.DataFrame({"Date": st.forecast_date, "Forecast (USD)": st.predicted_price_usd.map(usd),
                          "Lower 95%": st.lower_95_usd.map(usd), "Upper 95%": st.upper_95_usd.map(usd),
                          "Daily change": st.expected_daily_change_pct.map("{:.2f}%".format), "Direction": st.direction}),
            f"Sensitivity analysis: forecast from {other} (not the selected model)")
    sf = sens["forecast"]
    r.p(f"The sensitivity model gives alternating daily moves and a seven-day change of {sf['change_pct']:.2f}% "
        f"(to {usd(sf['end_price'])}), which is also {sf['classification']} under the same threshold. Its departure "
        "from the flat forecast is small relative to the interval width, so the overall conclusion does not depend on "
        "the root screen, although the day-to-day directions do.")

    r.h("Discussion")
    r.p("The expected direction is neutral. This is a result rather than a failure of the procedure: lags 1 to 6 "
        "carry little linear information, the more complex candidates were either inadmissible or no better in "
        "validation, and no alternative showed a statistically significant test improvement over the naive forecast.",
        "The forecast is not a statistically meaningful directional prediction. The model does not predict that the "
        f"price will stay at {usd(f['origin_price'])}; it states that, given only past prices, upward and downward "
        "moves in log price are treated as symmetric around today's value. A seven-day outcome anywhere between "
        f"about {usd(f['lower_day7'])} and {usd(f['upper_day7'])} would be consistent with the fitted model, and "
        "outcomes outside that range remain possible, particularly given heavy tails and the recent rise in rolling "
        "dispersion.",
        "The main strengths of the approach are its transparency, chronological validation, explicit benchmarks and "
        "honest intervals relative to a point forecast. Its main weakness is that it offers no skill beyond the naive "
        "method. These forecasts are statistical outputs for academic assessment and are not investment advice.")

    r.h("Limitations")
    r.bullets(["One year of data covers few market regimes; results may not generalise.",
               "ARIMA is linear and univariate. It excludes trading volume, other markets, news and on-chain variables.",
               "Residual variance is not constant and tails are heavy, so Gaussian intervals are approximate.",
               "The root screen was added after initial results were seen, so model selection was not fully blind.",
               f"The validation ({sp['validation_n']} days) and test ({sp['test_n']} days) samples are short, giving "
               "low statistical power; seven-day forecasts also overlap.",
               "Differences between candidate models are small and could reverse with a different window.",
               "The source does not fully document how its market price is aggregated or whether values are revised.",
               "The lag-7 correlation was not modelled with a seasonal term. SARIMA and GARCH-type models are natural "
               "extensions and should be assessed on new data."])

    r.h("Conclusion")
    r.p(f"Using genuine Blockchain.com data for {q['start_date']} to {q['end_date']}, the evidence supports one "
        f"difference of log price and selects {name} under the stated screen. The seven-day forecast is "
        f"{f['classification']}, with a median of {usd(f['origin_price'])} and a seventh-day 95% interval of "
        f"{usd(f['lower_day7'])} to {usd(f['upper_day7'])}. Out-of-sample accuracy equals that of the naive forecast. "
        "ARIMA therefore provides a disciplined baseline and a measure of uncertainty, but no demonstrated ability to "
        "predict the direction of Bitcoin over the coming week.",
        "Reproducibility: run run_pipeline.py on the saved snapshot; package versions are listed in requirements.txt "
        f"(Python {s['environment']['python']}, statsmodels {s['environment']['statsmodels']}, pmdarima "
        f"{s['environment']['pmdarima']}).")

    r.h("References")
    for ref in [
        "Akaike, H. (1974) 'A new look at the statistical model identification', IEEE Transactions on Automatic Control, 19(6), pp. 716-723. doi:10.1109/TAC.1974.1100705.",
        "Blockchain.com (2026) Market Price (USD) chart. Available at: https://www.blockchain.com/ko/charts/market-price (data retrieved 25 November 2026).",
        "Blockchain.com (2026) Charts and statistics API. Available at: https://www.blockchain.com/explorer/api/charts_api.",
        "Box, G.E.P., Jenkins, G.M., Reinsel, G.C. and Ljung, G.M. (2015) Time Series Analysis: Forecasting and Control. 5th edn. Hoboken: Wiley.",
        "Coinbase (2026) Exchange API: product candles, BTC-USD. Available at: https://api.exchange.coinbase.com/products/BTC-USD/candles (used for verification only).",
        "Cont, R. (2001) 'Empirical properties of asset returns: stylized facts and statistical issues', Quantitative Finance, 1(2), pp. 223-236. doi:10.1080/713665670.",
        "Dickey, D.A. and Fuller, W.A. (1979) 'Distribution of the estimators for autoregressive time series with a unit root', Journal of the American Statistical Association, 74(366), pp. 427-431. doi:10.1080/01621459.1979.10482531.",
        "Diebold, F.X. and Mariano, R.S. (1995) 'Comparing predictive accuracy', Journal of Business & Economic Statistics, 13(3), pp. 253-263. doi:10.1080/07350015.1995.10524599.",
        "Engle, R.F. (1982) 'Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation', Econometrica, 50(4), pp. 987-1007. doi:10.2307/1912773.",
        "Fama, E.F. (1970) 'Efficient capital markets: a review of theory and empirical work', The Journal of Finance, 25(2), pp. 383-417. doi:10.2307/2325486.",
        "Harvey, D., Leybourne, S. and Newbold, P. (1997) 'Testing the equality of prediction mean squared errors', International Journal of Forecasting, 13(2), pp. 281-291. doi:10.1016/S0169-2070(96)00719-4.",
        "Hyndman, R.J. and Khandakar, Y. (2008) 'Automatic time series forecasting: the forecast package for R', Journal of Statistical Software, 27(3). doi:10.18637/jss.v027.i03.",
        "Hyndman, R.J. and Koehler, A.B. (2006) 'Another look at measures of forecast accuracy', International Journal of Forecasting, 22(4), pp. 679-688. doi:10.1016/j.ijforecast.2006.03.001.",
        "Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. and Shin, Y. (1992) 'Testing the null hypothesis of stationarity against the alternative of a unit root', Journal of Econometrics, 54(1-3), pp. 159-178. doi:10.1016/0304-4076(92)90104-Y.",
        "Ljung, G.M. and Box, G.E.P. (1978) 'On a measure of lack of fit in time series models', Biometrika, 65(2), pp. 297-303. doi:10.1093/biomet/65.2.297.",
        "Meese, R.A. and Rogoff, K. (1983) 'Empirical exchange rate models of the seventies: do they fit out of sample?', Journal of International Economics, 14(1-2), pp. 3-24. doi:10.1016/0022-1996(83)90017-X.",
        "pmdarima developers (2026) pmdarima documentation. Available at: https://alkaline-ml.com/pmdarima/.",
        "Said, S.E. and Dickey, D.A. (1984) 'Testing for unit roots in autoregressive-moving average models of unknown order', Biometrika, 71(3), pp. 599-607. doi:10.1093/biomet/71.3.599.",
        "Schwarz, G. (1978) 'Estimating the dimension of a model', The Annals of Statistics, 6(2), pp. 461-464. doi:10.1214/aos/1176344136.",
        "Seabold, S. and Perktold, J. (2010) 'Statsmodels: econometric and statistical modeling with Python', Proceedings of the 9th Python in Science Conference, pp. 92-96. doi:10.25080/Majora-92bf1922-011.",
        "statsmodels developers (2026) ARIMA documentation. Available at: https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html.",
        "Tashman, L.J. (2000) 'Out-of-sample tests of forecasting accuracy: an analysis and review', International Journal of Forecasting, 16(4), pp. 437-450. doi:10.1016/S0169-2070(00)00065-0."]:
        paragraph = d.add_paragraph(ref)
        paragraph.paragraph_format.left_indent = Inches(0.4)
        paragraph.paragraph_format.first_line_indent = Inches(-0.4)

    d.add_page_break()
    r.h("Appendix: Test-Period Predictions")
    full = pd.DataFrame({"Date": detail.date, "Actual": detail.actual.map(usd), "Predicted": detail.predicted.map(usd),
                         "Absolute error": detail.absolute_error.map(usd), "Percentage error": detail.percentage_error.map("{:.2f}%".format)})
    r.table(full, f"One-step walk-forward predictions, {name}", 8)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    d.save(REPORT_PATH)
    print(f"Saved {REPORT_PATH} ({r.figure_n} figures, {r.table_n} tables, {r.equation_n} equations)")


if __name__ == "__main__":
    config.ensure_directories()
    build()
