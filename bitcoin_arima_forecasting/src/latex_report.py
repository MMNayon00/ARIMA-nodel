"""Build the LaTeX version of the report: python -m src.latex_report

Reuses the text, tables and figures declared in report_builder.py, so both
versions always carry the same numbers. Output is a self-contained folder
(report/latex/) that can be compiled locally or uploaded to Overleaf.
"""
import re
import shutil
import zipfile
from unittest.mock import MagicMock

from . import config, report_builder

STUDENT_NAME = "AHAMED MD SAJIB"
STUDENT_ID = "2026250206"

LATEX_DIR = config.ROOT / "report" / "latex"

SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}",
           "≤": r"$\le$", "≥": r"$\ge$", "±": r"$\pm$"}


def esc(value) -> str:
    text = "".join(SPECIAL.get(ch, ch) for ch in str(value))
    text = re.sub(r"(^|[\s(])'", r"\1`", text)  # opening single quotes
    return text


def prose(text: str) -> str:
    """Escape body text, then make URLs, checksums and symbols typeset properly."""
    parts = re.split(r"(https?://[^\s]+?(?=[.,;)]?(?:\s|$))|\b[0-9a-f]{64}\b)", text)
    out = []
    for i, part in enumerate(parts):
        if i % 2:
            out.append((r"\url{%s}" if part.startswith("http") else r"{\ttfamily\seqsplit{%s}}") % part)
        else:
            part = esc(part).replace("gamma = 0", r"$\gamma = 0$").replace("gamma \\textless{} 0", r"$\gamma < 0$")
            out.append(part.replace("H0", "$H_0$").replace("H1", "$H_1$"))
    return "".join(out)


class LatexReport:
    """Same interface as report_builder.Report, but collects LaTeX."""

    def __init__(self):
        self.out, self.refs = [], []
        self.figure_n = self.table_n = self.equation_n = 0
        self.mode = "front"
        self.doc = MagicMock()
        self.doc.add_paragraph.side_effect = self._raw_paragraph

    def _raw_paragraph(self, text="", *args, **kwargs):
        if self.mode == "references":
            self.refs.append(text)
        return MagicMock()

    def h(self, text, level=1):
        if text == "Contents":
            return
        if self.mode == "references":
            self._flush_references()
        if text == "Abstract":
            self.mode = "body"
            self.out.append("\\section*{Abstract}\n\\addcontentsline{toc}{section}{Abstract}")
        elif text == "References":
            self.mode = "references"
            self.out.append("\\clearpage\n\\section*{References}\n\\addcontentsline{toc}{section}{References}")
        elif text.startswith("Appendix"):
            self.mode = "body"
            self.out.append("\\clearpage\n\\appendix\n\\section{Test-Period Predictions}")
        else:
            if text == "Introduction":
                self.out.append("\\clearpage")
            command = "section" if level == 1 else "subsection"
            self.out.append(f"\\{command}{{{esc(text)}}}")

    def _flush_references(self):
        body = "\n\n".join(prose(ref) for ref in self.refs)
        self.out.append("\\begin{hangparas}{2.5em}{1}\n\\raggedright\n" + body + "\n\\end{hangparas}")
        self.refs, self.mode = [], "body"

    def p(self, *paragraphs):
        self.out.extend(prose(text) for text in paragraphs)

    def bullets(self, items, style="List Bullet"):
        env = "enumerate" if "Number" in style else "itemize"
        body = "\n".join(f"  \\item {prose(item)}" for item in items)
        self.out.append(f"\\begin{{{env}}}\n{body}\n\\end{{{env}}}")

    def equation(self, latex):
        self.equation_n += 1
        env = "equation"
        body = latex
        if len(latex) > 80 and r",\qquad " in latex or r",\quad " in latex:
            body = re.sub(r",\\q?quad ", r",\\\\ ", latex)
            body = "\\begin{gathered}\n" + body + "\n\\end{gathered}"
        self.out.append(f"\\begin{{{env}}}\n{body}\n\\end{{{env}}}")

    def figure(self, filename, caption, width=6.1):
        self.figure_n += 1
        shutil.copy(config.OUTPUT_DIR / filename, LATEX_DIR / "figures" / filename)
        self.out.append("\\begin{figure}[H]\n\\centering\n"
                        "\\includegraphics[width=\\linewidth,height=0.78\\textheight,keepaspectratio]"
                        f"{{figures/{filename}}}\n\\caption{{{prose(caption)}}}\n\\end{{figure}}")
        if filename == "model_diagnostics_full_sample.png":
            other = "ARIMA(6, 1, 4)"
            self.figure("sensitivity_diagnostics.png",
                        f"Residual diagnostics, sensitivity model {other}, training sample")
            self.figure("sensitivity_diagnostics_full_sample.png",
                        f"Residual diagnostics, sensitivity model {other}, full sample")

    def table(self, frame, caption, size=8.5):
        self.table_n += 1
        numeric = [bool(re.fullmatch(r"[\$<≤≥-]?[\d,.\-$%]+", str(v))) for v in frame.iloc[0]]
        align = "".join("r" if n else "l" for n in numeric)
        header = " & ".join(f"\\textbf{{{esc(c)}}}" for c in frame.columns) + r" \\"
        rows = "\n".join(" & ".join(esc(v) for v in row) + r" \\"
                         for row in frame.astype(str).values.tolist())
        font = r"\small" if len(frame.columns) <= 5 else r"\footnotesize"
        title = f"\\caption{{{prose(caption)}}}"
        if len(frame) > 30:  # long tables break across pages
            self.out.append(
                f"{{{font}\n\\begin{{longtable}}{{{align}}}\n{title}\\\\\n\\toprule\n{header}\n\\midrule\n"
                f"\\endfirsthead\n\\toprule\n{header}\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n"
                f"{rows}\n\\end{{longtable}}\n}}")
        else:
            self.out.append(
                f"\\begin{{table}}[H]\n\\centering\n{title}\n{font}\n\\setlength{{\\tabcolsep}}{{4pt}}\n"
                f"\\begin{{adjustbox}}{{max width=\\linewidth}}\n\\begin{{tabular}}{{{align}}}\n\\toprule\n"
                f"{header}\n\\midrule\n{rows}\n\\bottomrule\n\\end{{tabular}}\n\\end{{adjustbox}}\n\\end{{table}}")


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern,microtype}
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb,graphicx,float,booktabs,longtable,adjustbox,seqsplit,hanging}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{xcolor,titlesec,fancyhdr,parskip,listings}
\usepackage{xurl,CJKutf8}
\usepackage[hidelinks]{hyperref}
\definecolor{navy}{HTML}{1F3A5F}
\titleformat{\section}{\Large\bfseries\color{navy}}{\thesection}{0.8em}{}
\titleformat{\subsection}{\large\bfseries\color{navy}}{\thesubsection}{0.8em}{}
\setlength{\headheight}{14pt}
\pagestyle{fancy}
\fancyhf{}
\lhead{\small Bitcoin Price Forecasting with ARIMA}
\rhead{\small %(name)s (%(sid)s)}
\cfoot{\thepage}
\lstset{language=Python,basicstyle=\ttfamily\scriptsize,breaklines=true,columns=fullflexible,
  keepspaces=true,showstringspaces=false,frame=single,keywordstyle=\color{navy}\bfseries,
  commentstyle=\color{gray}}
\hypersetup{pdftitle={Forecasting the Seven-Day Price Movement of Bitcoin with ARIMA},pdfauthor={%(name)s}}
"""

TITLE_PAGE = r"""\begin{titlepage}
\centering
\vspace*{0.5cm}
{\color{navy}\Huge\bfseries ASSIGNMENT\par}
\vspace{0.8cm}
{\color{navy}\rule{\linewidth}{1.2pt}}\\[0.6cm]
{\large\bfseries Topic\par}
\vspace{0.3cm}
{\color{navy}\LARGE\bfseries Forecasting the Seven-Day Price Movement\\[0.2cm] of Bitcoin with ARIMA\par}
\vspace{0.4cm}
{\color{navy}\rule{\linewidth}{1.2pt}}
\vfill
{\large\bfseries\color{navy} Submitted To\par}
\vspace{0.4cm}
{\Large \begin{CJK}{UTF8}{mj}정명애\end{CJK} (Professor)\par}
\vfill
{\large\bfseries\color{navy} Submitted By\par}
\vspace{0.4cm}
{\large\renewcommand{\arraystretch}{1.5}
\begin{tabular}{rl}
\textbf{Student name:} & %(name)s \\
\textbf{Student ID:} & %(sid)s \\
\textbf{University:} & Eulji University \\
\textbf{Department:} & Global Big Data \& Artificial Intelligence \\
\textbf{Submission date:} & 27 November 2026 \\
\end{tabular}\par}
\vfill
{\color{navy}\LARGE\bfseries EULJI UNIVERSITY\par}
\end{titlepage}
\pagenumbering{roman}
\tableofcontents
\clearpage
\listoffigures
\listoftables
\clearpage
\pagenumbering{arabic}
"""

SOURCE_FILES = ["run_pipeline.py", "src/config.py", "src/data_collection.py", "src/preprocessing.py",
                "src/exploratory_analysis.py", "src/stationarity.py", "src/arima_model.py",
                "src/forecasting.py", "src/cross_check.py"]


def source_appendix() -> str:
    (LATEX_DIR / "code").mkdir(exist_ok=True)
    out = ["\\clearpage\n\\section{Python Source Code}",
           "The analysis is implemented in Python. The listings below are the files that produce "
           "the data, figures and results in this report."]
    for name in SOURCE_FILES:
        text = (config.ROOT / name).read_text(encoding="utf-8")
        # listings with pdfLaTeX accepts ASCII only
        for char, plain in {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "±": "+/-", "’": "'"}.items():
            text = text.replace(char, plain)
        text = text.encode("ascii", "replace").decode()
        flat = name.replace("/", "_")
        (LATEX_DIR / "code" / flat).write_text(text)
        out.append(f"\\subsection{{\\texttt{{{esc(name)}}}}}\n\\lstinputlisting{{code/{flat}}}")
    return "\n\n".join(out)


def build() -> None:
    (LATEX_DIR / "figures").mkdir(parents=True, exist_ok=True)
    report = LatexReport()
    original_report, original_path = report_builder.Report, report_builder.REPORT_PATH
    report_builder.Report = lambda: report
    report_builder.REPORT_PATH = MagicMock()  # leave the Word report untouched
    try:
        report_builder.build()
    finally:
        report_builder.Report, report_builder.REPORT_PATH = original_report, original_path

    import json
    summary = json.loads((config.RESULTS_DIR / "analysis_summary.json").read_text())
    values = {"name": STUDENT_NAME, "sid": STUDENT_ID, "start": summary["quality"]["start_date"], "end": summary["quality"]["end_date"],
              "retrieved": summary["metadata"]["retrieved_utc"]}
    tex = "\n\n".join([PREAMBLE % values + "\\begin{document}\n" + TITLE_PAGE % values,
                       *report.out, source_appendix(), "\\end{document}\n"])
    (LATEX_DIR / "main.tex").write_text(tex, encoding="utf-8")

    archive = config.ROOT / "report" / "latex_overleaf.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(LATEX_DIR.rglob("*")):
            if path.suffix in {".tex", ".png", ".py"}:
                bundle.write(path, path.relative_to(LATEX_DIR))
    print(f"Saved {LATEX_DIR / 'main.tex'} and {archive} "
          f"({report.figure_n} figures, {report.table_n} tables, {report.equation_n} equations)")


if __name__ == "__main__":
    build()
