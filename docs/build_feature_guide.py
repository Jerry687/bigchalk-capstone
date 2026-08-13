"""Build the Big Chalk Mix Engine feature reference PDF."""
import os
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, PageBreak,
                                KeepTogether, ListFlowable, ListItem)
from reportlab.platypus.tableofcontents import TableOfContents

FONTS = "/usr/share/fonts/truetype/dejavu"
pdfmetrics.registerFont(TTFont("DJ", f"{FONTS}/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DJ-B", f"{FONTS}/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DJ-I", f"{FONTS}/DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DJM", f"{FONTS}/DejaVuSansMono.ttf"))
pdfmetrics.registerFont(TTFont("DJM-B", f"{FONTS}/DejaVuSansMono-Bold.ttf"))
pdfmetrics.registerFontFamily("DJ", normal="DJ", bold="DJ-B", italic="DJ-I")

NAVY = colors.HexColor("#1E4FA3")
INK = colors.HexColor("#101828")
INK2 = colors.HexColor("#475467")
INK3 = colors.HexColor("#667085")
MUTED = colors.HexColor("#98A2B3")
GREEN = colors.HexColor("#067647")
RED = colors.HexColor("#B42318")
WARN = colors.HexColor("#B54708")
LINE = colors.HexColor("#E4E7EC")
BG = colors.HexColor("#F8FAFC")
BGN = colors.HexColor("#EBF1FB")

ss = getSampleStyleSheet()


def st(name, **kw):
    base = dict(fontName="DJ", fontSize=9.5, leading=14, textColor=INK,
                spaceAfter=6)
    base.update(kw)
    return ParagraphStyle(name, **base)


S = {
    "title": st("title", fontName="DJ-B", fontSize=26, leading=31,
                textColor=NAVY, spaceAfter=8),
    "subtitle": st("subtitle", fontSize=12.5, leading=18, textColor=INK2,
                   spaceAfter=4),
    "meta": st("meta", fontSize=9, textColor=INK3, leading=14),
    "h1": st("h1", fontName="DJ-B", fontSize=17, leading=22, textColor=NAVY,
             spaceBefore=2, spaceAfter=8),
    "h2": st("h2", fontName="DJ-B", fontSize=12.5, leading=17, textColor=INK,
             spaceBefore=13, spaceAfter=5),
    "h3": st("h3", fontName="DJ-B", fontSize=10.5, leading=15, textColor=NAVY,
             spaceBefore=10, spaceAfter=3),
    "body": st("body", alignment=TA_JUSTIFY),
    "small": st("small", fontSize=8.7, leading=12.5, textColor=INK2),
    "bullet": st("bullet", alignment=TA_JUSTIFY, spaceAfter=3),
    "code": st("code", fontName="DJM", fontSize=8.5, leading=12.5,
               textColor=INK, backColor=BG, borderPadding=7,
               borderColor=LINE, borderWidth=0.6, spaceBefore=4, spaceAfter=8),
    "quote": st("quote", fontName="DJ-I", fontSize=9.3, leading=13.5,
                textColor=INK2, leftIndent=12, borderPadding=(0, 0, 0, 8),
                spaceBefore=4, spaceAfter=8),
    "th": st("th", fontName="DJ-B", fontSize=8.3, leading=11,
             textColor=colors.white, spaceAfter=0),
    "td": st("td", fontSize=8.3, leading=11.5, textColor=INK, spaceAfter=0),
    "tdm": st("tdm", fontName="DJM", fontSize=8, leading=11.5, textColor=INK,
              spaceAfter=0),
    "toc1": ParagraphStyle("toc1", fontName="DJ-B", fontSize=10, leading=19,
                           textColor=INK),
    "toc2": ParagraphStyle("toc2", fontName="DJ", fontSize=9, leading=15,
                           textColor=INK2, leftIndent=16),
}

story = []


def h1(t, num=None):
    txt = f"{num}&nbsp;&nbsp;{t}" if num else t
    p = Paragraph(txt, S["h1"])
    p._toc = (0, t if not num else f"{num}  {t}")
    story.append(p)
    story.append(hr(NAVY, 1.1))
    story.append(Spacer(1, 7))


def h2(t, num=None):
    txt = f"{num}&nbsp;&nbsp;{t}" if num else t
    p = Paragraph(txt, S["h2"])
    p._toc = (1, t if not num else f"{num}  {t}")
    story.append(p)


def h3(t):
    story.append(Paragraph(t, S["h3"]))


def p(t, style="body"):
    story.append(Paragraph(t, S[style]))


def code(t):
    story.append(Paragraph(t.replace("\n", "<br/>").replace(" ", "&nbsp;"),
                           S["code"]))


def quote(t, who=""):
    story.append(Paragraph(f"&ldquo;{t}&rdquo;" + (f"  &mdash; {who}" if who else ""),
                           S["quote"]))


def bullets(items, style="bullet"):
    story.append(ListFlowable(
        [ListItem(Paragraph(i, S[style]), leftIndent=15) for i in items],
        bulletType="bullet", start="\u2022", leftIndent=13,
        bulletFontName="DJ", bulletFontSize=8.5, bulletOffsetY=-0.5,
        spaceAfter=6))


def hr(color=LINE, w=0.6):
    t = Table([[""]], colWidths=[6.9 * inch], rowHeights=[0.5])
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, -1), w, color)]))
    return t


def table(rows, widths, header=True, mono_cols=(), size=8.3):
    data = []
    for r_i, row in enumerate(rows):
        out = []
        for c_i, cell in enumerate(row):
            if r_i == 0 and header:
                out.append(Paragraph(str(cell), S["th"]))
            else:
                sty = S["tdm"] if c_i in mono_cols else S["td"]
                out.append(Paragraph(str(cell), sty))
        data.append(out)
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1 if header else 0)
    style = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LINE),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
    ]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
                  ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                   [colors.white, colors.HexColor("#FBFCFD")])]
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 9))


def callout(title, body, color=NAVY, bg=BGN):
    inner = [Paragraph(f"<b>{title}</b>", st("ct", fontName="DJ-B", fontSize=9.3,
                                             textColor=color, spaceAfter=3)),
             Paragraph(body, st("cb", fontSize=9, leading=13, textColor=INK2,
                                spaceAfter=0, alignment=TA_JUSTIFY))]
    t = Table([[inner]], colWidths=[6.9 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, color),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
    ]))
    story.append(KeepTogether(t))
    story.append(Spacer(1, 9))


# ══════════════════════════════════════════════════════════════════ COVER
story.append(Spacer(1, 1.5 * inch))
story.append(Paragraph("Big Chalk Mix Engine", S["title"]))
story.append(Paragraph("Automated Regression Engine &amp; Dashboard "
                       "&mdash; Complete Feature Reference", S["subtitle"]))
story.append(Spacer(1, 10))
story.append(hr(NAVY, 1.4))
story.append(Spacer(1, 14))
story.append(Paragraph(
    "Every capability currently built into the system: what it does, how to "
    "use it, the method behind it, and why it was designed that way. Written "
    "to serve both as an operating manual and as the methodology record for "
    "the capstone.", st("lede", fontSize=11, leading=16.5, textColor=INK2,
                        alignment=TA_JUSTIFY)))
story.append(Spacer(1, 26))
table([
    ["Project", "Northwestern MSDS Capstone, Summer 2026"],
    ["Sponsor", "Big Chalk Analytics &mdash; Alex Hathcock, Arko"],
    ["Faculty", "Abid Ali"],
    ["Team", "Feifan Liu, Boqi (Jerry) Niu, Jiahao (Edison) Li"],
    ["Objective", "Automated regression models predicting Volume Sales for "
                  "every Brand &times; Retailer-Channel combination, with a "
                  "dashboard for inspection and manual tuning"],
    ["Dataset", "156 weeks (Jan 2023 &ndash; Dec 2025), 10 brands, 9 channels, "
                "113 columns"],
    ["Status", "76 models built, 6 slices deliberately not modeled"],
    ["Document date", "August 10, 2026"],
], [1.35, 5.55], header=False)
story.append(Spacer(1, 20))
callout("How to read this document",
        "Sections 1&ndash;2 describe the modeling engine and its methods. "
        "Section 3 walks the dashboard screen by screen. Sections 4&ndash;6 "
        "are reference tables for configuration fields, export files and "
        "command-line tools. Sections 7&ndash;9 report current results, the "
        "reasoning behind the significant design decisions, and what is "
        "verified. Section 10 states the known limits honestly.")
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════ TOC
story.append(Paragraph("Contents", S["h1"]))
story.append(hr(NAVY, 1.1))
story.append(Spacer(1, 10))
toc = TableOfContents()
toc.levelStyles = [S["toc1"], S["toc2"]]
story.append(toc)
story.append(PageBreak())

# ═══════════════════════════════════════════════════════ 1. OVERVIEW
h1("System overview", "1")

h2("What the system does", "1.1")
p("The engine fits one regression model per <b>Brand &times; Channel</b> slice, "
  "explaining weekly Volume Sales from marketing and market drivers: "
  "distribution, price, trade promotion, media, competitive activity, "
  "seasonality and macroeconomics. It runs the full set of slices "
  "unattended, grades each result, and decomposes every model into weekly "
  "driver contributions that reconcile exactly to the fitted line.")
p("Around the engine sits a six-screen dashboard for inspecting, tuning and "
  "exporting those models. The design goal stated by the sponsor is a system "
  "an analyst can point at a data file and get a defensible model set from, "
  "without hand-building each model &mdash; while still being able to "
  "override any decision the automation made.")

h2("Architecture", "1.2")
table([
    ["Component", "File", "Responsibility"],
    ["Modeling engine", "code/capstone_pipeline.py",
     "All statistics: loading, windowing, transforms, variable selection, "
     "constrained fitting, contributions, due-tos, curve optimization. No UI "
     "code. Importable and testable on its own."],
    ["Dashboard", "code/dashboard.py",
     "Plotly Dash application: six screens, the results cache, Excel export, "
     "config editing and template round-trip."],
    ["Batch runner", "code/run_all.py",
     "Command-line run over every slice; writes per-slice CSVs plus summary, "
     "skipped and failure files."],
    ["Single-slice runner", "code/run_brand1_channel1.py",
     "Focused run of one slice for debugging."],
    ["Equivalence check", "code/reference_alex_curvefit.py",
     "Proves our constrained solver matches the sponsor's own curve_fit "
     "approach numerically."],
    ["Scaling test", "code/scaling_test.py",
     "Runtime scaling measurements."],
    ["Variable configs", "configs/*.csv",
     "Per-product and per-product-per-channel modeling configuration. The "
     "CSV, not the code, is the source of truth."],
], [1.15, 1.75, 4.0])

h2("Running it", "1.3")
code("# dashboard\ncd code &amp;&amp; python dashboard.py        # http://127.0.0.1:8050\n\n"
     "# full batch from the command line\ncd code &amp;&amp; python run_all.py\n"
     "cd code &amp;&amp; python run_all.py 1 3      # brands 1-3 only (resumable chunks)")
p("Requirements are pinned in <font face='DJM' size='8.5'>requirements.txt</font>. "
  "The dashboard accepts any data file path in the left rail; the "
  "command-line scripts expect <font face='DJM' size='8.5'>Anonymized Data "
  "for Project.xlsx</font> in the repository root.", "small")

# ═══════════════════════════════════════════════════════ 2. ENGINE
story.append(PageBreak())
h1("The modeling engine", "2")

h2("Slice definition and data loading", "2.1")
p("A <b>slice</b> is one Brand (an Excel sheet) &times; one Channel (the "
  "Geography column). Weeks are parsed from strings of the form "
  "<font face='DJM' size='8.5'>WE 01/08/2023</font> into real dates and "
  "sorted. Every model is fit on exactly one slice; there is no pooling "
  "across brands or channels.")
p("Columns that are algebraic decompositions of the target &mdash; anything "
  "beginning <font face='DJM' size='8.5'>Volume Sales</font> or "
  "<font face='DJM' size='8.5'>Dollar Sales</font>, such as "
  "&lsquo;Volume Sales Any Promo&rsquo; &mdash; are excluded as predictors "
  "automatically. They are components of the thing being predicted, so "
  "including them would produce a model with a perfect fit and no meaning. "
  "The exclusion follows the target: change the target and the "
  "decompositions of the <i>new</i> target are the ones removed.")

h2("The fixed calendar window", "2.2")
callout("This fixed a real bug found by the sponsor",
        "The window used to be &lsquo;the last N <b>rows</b> this slice "
        "happens to have&rsquo;. A product delisted in mid-2023 therefore got "
        "its own private 2023 window, still produced a model, and leaked "
        "18-month-old due-tos into the export as a disjoint chunk of history. "
        "Alex found it by pivoting the exported weekly due-tos in Excel.",
        WARN, colors.HexColor("#FEF6EE"))
quote("The window should be predefined, so the last 104 weeks should be the "
      "same 104 weeks for everything. Instead of having the window being "
      "dependent on the data, the data is dependent on the window.",
      "Alex, 2026-07-27")
p("The window is now an <b>absolute date range</b>. It ends at the latest "
  "week present anywhere in the data file (the <i>anchor</i>) and runs back "
  "52, 104 or 156 weeks. It is identical for every slice, so models are "
  "comparable and no slice can be modeled on stale history.")
table([
    ["Window", "Calendar range on the current dataset"],
    ["52 weeks", "Jan 05, 2025 &ndash; Dec 28, 2025"],
    ["104 weeks (default)", "Jan 07, 2024 &ndash; Dec 28, 2025"],
    ["156 weeks", "Jan 08, 2023 &ndash; Dec 28, 2025"],
], [1.6, 5.3], mono_cols=(1,))
p("The anchor is computed once per data file and cached by file path, "
  "modification time and size, so repeated runs do not re-read the workbook.",
  "small")

h3("Slices that are not modeled")
p("A slice needs a minimum amount of data <i>inside</i> the window to be "
  "worth modeling. Below either threshold it is not modeled at all &mdash; it "
  "raises a typed exception carrying the counts, and is reported as skipped "
  "with the reason. It never appears as a model, and it never contributes "
  "a due-to.")
table([
    ["Threshold", "Default", "Meaning"],
    ["min_weeks", "52", "Weeks of any data inside the window"],
    ["min_nonzero_weeks", "26", "Weeks with non-zero target (actual selling weeks)"],
], [1.6, 0.8, 4.5], mono_cols=(0,))
p("Skipped slices surface in three places: a <b>Not modeled</b> table on the "
  "Model Runs screen, the <font face='DJM' size='8.5'>errors</font> sheet of "
  "the Excel export, and "
  "<font face='DJM' size='8.5'>all_models_skipped.csv</font> from the "
  "command-line runner. Nothing disappears silently &mdash; the point is that "
  "a data problem should be visible, not invisible.")

h2("Two-tier variable configuration", "2.3")
p("Predictors are not hard-coded. A CSV per product maps each raw column to a "
  "family, an expected sign, optional bounds, transforms and a role. An "
  "optional second file overrides it for a single channel.")
table([
    ["Tier", "File", "Applies to"],
    ["Product default",
     "configs/varconfig_&lt;dataset&gt;_&lt;brand&gt;.csv",
     "Every channel of that brand"],
    ["Product &times; Channel override",
     "configs/varconfig_&lt;dataset&gt;_&lt;brand&gt;__ch_&lt;channel&gt;.csv",
     "That one channel &mdash; <b>overrides wins</b>"],
], [1.5, 3.2, 2.2], mono_cols=(1,))
p("One resolver function is shared by the dashboard, the batch runner and the "
  "single-slice runner, so all three agree on which config a given slice "
  "uses. Every config write goes through a single process lock and a "
  "transactional temp-file-plus-rollback commit, so a failed edit cannot "
  "leave a half-written config on disk.")
p("<b>ACV Weighted Distribution is force-included by default.</b> This lives "
  "in the generated config's <font face='DJM' size='8.5'>role</font> column "
  "rather than in code, so it is visible and the analyst can override it back "
  "to <font face='DJM' size='8.5'>auto</font>. Justification: a volume model "
  "with no distribution term cannot track shelf-presence change; on "
  "Brand&nbsp;1&nbsp;&times;&nbsp;Channel&nbsp;1 forcing it moved holdout "
  "MAPE from 73% to 32%.")

# --- transforms
story.append(PageBreak())
h2("Transforms", "2.4")
p("Applied in this order, matching the convention used by Meta's Robyn and "
  "Google's Meridian:")
code("raw execution  &rarr;  adstock (carry-over)  &rarr;  scale (units)  &rarr;  Hill (saturation)")

h3("Adstock &mdash; carry-over")
code("a(t) = (1 &minus; d) &middot; x(t) + d &middot; a(t&minus;1)")
p("The (1&minus;d) share of this week's execution lands this week; the "
  "remainder carries forward. The formulation is <b>normalized</b>: the "
  "adstocked total approximately equals the raw total. The naive "
  "<font face='DJM' size='8.5'>a(t) = x(t) + d&middot;a(t&minus;1)</font> "
  "inflates total impressions, which in turn inflates the media contribution "
  "&mdash; a quiet way to overstate media's value.")
quote("If you see an ad and you wait to do grocery shopping until next week, "
      "it's going to have a decayed amount.", "Alex")
p("Decay is per variable. Suggested starting values: search-type media "
  "&asymp;&nbsp;0.2 (acted on immediately), TV/CTV &asymp;&nbsp;0.7 (long "
  "memory), 0.5 default. A diagnostic reports the adstocked-versus-raw total "
  "for each media variable so the normalization can be checked.")

h3("Scale &mdash; readable units")
p("Divides a variable before modeling so its coefficient reads &lsquo;per "
  "1,000 impressions&rsquo; rather than &lsquo;per impression&rsquo;. This is "
  "a <b>units change only</b>: the coefficient shrinks by exactly the factor "
  "the data does, so every contribution, due-to and fitted value is "
  "unchanged (verified to 1.2&times;10<super>-10</super>).")
p("Custom coefficient bounds are written in <b>original</b> units and "
  "transformed with the scale internally. Without that, setting a scale of "
  "1,000 would silently tighten every custom bound by a factor of 1,000. "
  "Sign constraints need no such treatment, since dividing by a positive "
  "number cannot flip a sign.")

h3("Hill &mdash; saturation")
callout("Identifying the curve",
        "The sponsor described this curve from memory: &lsquo;some Bill and "
        "this other data scientist created this for the efficacy of drugs "
        "&hellip; a curve that has a midpoint and a slope, so it makes it "
        "sigmoidal&rsquo;. That is the <b>Hill equation</b> &mdash; A. V. "
        "Hill, 1910, fitting oxygen binding to haemoglobin, and since then "
        "the standard dose&ndash;response curve in pharmacology. It is also "
        "the saturation function in both Robyn and Meridian. Two parameters, "
        "a midpoint and a slope, exactly as described.")
code("H(x) = x<super>s</super> / (x<super>s</super> + k<super>s</super>),      k = &gamma; &middot; ref")
table([
    ["Parameter", "Config column", "Meaning"],
    ["&gamma; &mdash; midpoint", "sat_midpoint",
     "Half-saturation point as a <i>fraction</i> of the reference scale. At "
     "x&nbsp;=&nbsp;&gamma;&middot;ref the response is exactly half its "
     "maximum. Stored as a fraction so the parameter transfers across brands "
     "and channels whose spend differs by orders of magnitude. Valid range "
     "(0,&nbsp;1]."],
    ["s &mdash; slope", "sat_slope",
     "Shape. s&nbsp;&gt;&nbsp;1 gives the S-curve: slow start, steep through "
     "the midpoint, then flattening. s&nbsp;&lt;&nbsp;1 diminishes straight "
     "from the origin, which weekly aggregate data often prefers."],
    ["ref", "(computed)",
     "The scale the midpoint is a fraction of &mdash; the peak adstocked, "
     "scaled execution. <b>Computed on training weeks only</b>, otherwise the "
     "transform itself would see the holdout."],
], [1.25, 1.15, 4.5], mono_cols=(1,))
p("Because H returns a value in [0,&nbsp;1), the coefficient on a saturated "
  "variable is the <b>maximum weekly volume that variable can deliver</b> "
  "&mdash; a directly useful number. Contributions remain additive, so the "
  "due-to decomposition is unaffected (verified exact to "
  "1.0&times;10<super>-10</super>).")
p("Saturation is <b>off by default</b>: both parameters blank means the "
  "variable stays linear. Half a specification is treated as no saturation "
  "rather than guessed at.")

# --- selection
story.append(PageBreak())
h2("Automated variable selection", "2.5")
p("Three stages, run on standardized data so the selection is scale-invariant:")

h3("Stage 1 &mdash; VIF pruning")
p("Iteratively drops the predictor with the highest variance inflation factor "
  "until all are below the threshold. VIFs are computed at once from the "
  "diagonal of the inverse correlation matrix &mdash; mathematically "
  "identical to running the k auxiliary regressions, roughly 100&times; "
  "faster on a wide matrix. Force-included variables are protected and never "
  "dropped.")
p("<b>Why VIF rather than pairwise correlation or PCA:</b> VIF catches a "
  "variable that is redundant with a <i>combination</i> of others, which a "
  "correlation matrix misses; and unlike PCA it keeps the raw, explainable "
  "variables the client needs to see.", "small")

h3("Stage 2 &mdash; forward stepwise with family caps")
p("Adds the variable with the lowest p-value while additions remain "
  "significant at <font face='DJM' size='8.5'>p_enter</font>. Forced "
  "variables are seeded in first.")
p("<b>Why stepwise rather than Lasso or tree importance:</b> every addition "
  "is a single testable decision that can be explained to a client. A Lasso "
  "path is harder to defend line by line, and tree importances do not give "
  "the signed, bounded coefficients the deliverable requires.", "small")
p("<b>Per-family caps</b> limit how many variables of each family may enter. "
  "The cap is enforced <i>during</i> selection, not by trimming afterwards, "
  "so a capped family spends its budget on its most significant members while "
  "the other families still fill normally. Forced variables are always kept "
  "and count toward the budget: a cap of 2 with 2 forced Trade variables "
  "admits no further Trade.")
quote("Is there a way to have variable hyperparameter tuning? We want to be "
      "more restrictive, we want to be less restrictive &hellip; more "
      "restrictive by a group. I want only two to three trade, but I want all "
      "media.", "Alex")

h3("Stage 3 &mdash; dead-coefficient pruning")
p("After the constrained fit, any variable whose standardized impact "
  "(|coefficient| &times; standard deviation) is under 1&times;10<super>-4</super> "
  "of the largest is removed and the model refit. These are variables the "
  "bounded solver has pushed to effectively zero; leaving them in would "
  "inflate the predictor count and clutter the coefficient table without "
  "changing the fit.")

h3("Strictness presets")
p("Exposed on the Variables screen as a single control. Balanced reproduces "
  "the engine's historical defaults exactly, so results are unchanged until "
  "someone opts in.")
table([
    ["Preset", "p_enter", "VIF threshold", "Avg predictors per model (76 slices)"],
    ["Punitive", "0.01", "5", "4.4"],
    ["Balanced (default)", "0.05", "10", "6.5"],
    ["Lax", "0.15", "20", "10.0"],
], [1.5, 0.9, 1.3, 3.2], mono_cols=(1, 2, 3))

h2("Constrained final fit", "2.6")
p("The final model is bounded least squares "
  "(<font face='DJM' size='8.5'>scipy.optimize.lsq_linear</font>, trust-region "
  "reflective) in original units, so coefficients are interpretable and "
  "contributions decompose additively.")
table([
    ["Guardrail", "Behaviour"],
    ["Expected sign", "A positive-signed variable can never enter with a "
                      "negative coefficient, and vice versa. The sign is a "
                      "hard constraint, not a preference."],
    ["Custom bounds", "Explicit [lo, hi] limits per coefficient; these "
                      "override the sign default. lo must be strictly &lt; hi."],
    ["Intercept", "Always unconstrained, and reported as its own due-to line."],
], [1.35, 5.55])
p("<b>Why bounded least squares rather than Ridge:</b> Ridge cannot honour "
  "sign or box constraints, and the sponsor requires them &mdash; a model "
  "that says price increases raise volume is not deliverable regardless of "
  "its fit statistics. Bounded least squares enforces any [lo, hi] per "
  "coefficient while remaining a transparent linear model.")
p("An unconstrained OLS is fit alongside purely to supply t-statistics as an "
  "inference reference, and VIFs are recomputed on the final in-model set.",
  "small")

h3("Sign-conflict diagnostic")
p("When the expected sign is positive but the unconstrained t-statistic is "
  "below &minus;1.96 (or the mirror case), the variable is flagged as a sign "
  "conflict. The constraint still holds &mdash; but the analyst is told that "
  "the data disagrees with the prior, which is usually a signal about the "
  "data rather than about the prior.")

# --- validation
story.append(PageBreak())
h2("Validation design", "2.7")
p("This is the part of the engine that determines whether any reported "
  "accuracy number can be trusted, so it is documented in full.")

h3("One structure, two fits")
p("Within the fixed window, the last up to 13 weeks are <b>always</b> "
  "reserved as a validation tail (floored to keep at least 5 training weeks; "
  "set to 0 to disable).")
bullets([
    "<b>Validation model</b> &mdash; selects the variable structure on the "
    "window <i>before</i> the tail, so selection never sees the tail, and is "
    "scored on the tail. That score is the reported <b>holdout MAPE</b>.",
    "<b>Reported model</b> &mdash; refits that <i>same</i> structure on the "
    "full window including the tail. These are the coefficients and due-tos "
    "displayed.",
])
p("So the holdout validates the same structure and the same modeling process "
  "out of sample, and the reported metric is a genuine out-of-sample number "
  "&mdash; never the all-data in-sample fit relabelled. Both fits live inside "
  "the same fixed calendar window: the tail is carved out of it rather than "
  "the window sliding backwards, so the validation model never reaches for "
  "data outside the period being modeled.")
p("<b>Why a time-based holdout rather than random K-fold:</b> weekly series "
  "are autocorrelated, and random folds leak the future into training. A "
  "model validated by random folds on time-series data will look far better "
  "than it is.", "small")

h3("Three-way split for curve optimization")
p("When the media curve optimizer runs, a two-way split is not enough. "
  "Searching a large space over-fits whatever it is scored on &mdash; that is "
  "arithmetic, not bad luck. So the window is split three ways:")
code("[ &mdash;&mdash;&mdash;&mdash; train &mdash;&mdash;&mdash;&mdash; | inner CV folds | OUTER HOLDOUT ]\n"
     "                                                    13 wks")
p("Candidates are scored by <b>rolling-origin cross-validation</b> on the "
  "inner folds &mdash; train on everything before a fold, score the fold, "
  "average across three expanding-window folds. The outer holdout is never "
  "touched during the search, so the holdout MAPE reported afterwards remains "
  "an honest number, directly comparable with a model that was never "
  "optimized.")
callout("Why this matters more than it sounds",
        "Had the optimizer been scored on the reported holdout, every "
        "optimized model would have looked better and not one of them would "
        "have been. The three-way split is what makes it possible to state "
        "honestly, in section 7, that the optimizer's gains often do not "
        "transfer.")

h2("Fit metrics and grading", "2.8")
table([
    ["Metric", "Definition"],
    ["R&sup2; / adjusted R&sup2;", "Share of weekly variance explained; "
     "adjusted R&sup2; penalizes added predictors. Shown together so a high "
     "R&sup2; from over-fitting is visible."],
    ["In-sample MAPE", "Mean absolute percentage error on the training "
     "window, over non-zero-actual weeks only (zero-sales weeks would divide "
     "by zero in partial-distribution channels)."],
    ["Holdout MAPE", "Out-of-sample error on the reserved validation tail. "
     "This is the number that grades the model."],
    ["Durbin&ndash;Watson", "Residual autocorrelation check; approximately 2 "
     "means no autocorrelation."],
    ["t-statistic", "From the unconstrained OLS as an inference reference; "
     "|t|&nbsp;&ge;&nbsp;2 is roughly significant at 5%."],
    ["VIF", "Collinearity on the final in-model variables. Above 10 is "
     "flagged as a non-blocking warning; forced variables can be high by "
     "design."],
], [1.35, 5.55])
table([
    ["Grade", "Holdout MAPE", "Colour"],
    ["Good Model", "&le; 10%", "Green"],
    ["Moderate Model", "&gt; 10% and &le; 40%", "Amber"],
    ["Bad Model", "&gt; 40%", "Red"],
    ["No holdout", "No holdout period available to score", "Grey"],
], [1.5, 2.4, 3.0])
p("These cutoffs are a team convention reusing the review-flag threshold and "
  "should be confirmed with the client.", "small")

# --- contributions
story.append(PageBreak())
h2("Contributions and due-tos", "2.9")
callout("These are two different numbers",
        "The tool previously showed only the first under the second's name. "
        "Arko raised it and Alex confirmed: &lsquo;what you have here is "
        "contribution &mdash; what is the total per period. The due-to is "
        "just the change &hellip; that's really what all the people that look "
        "at this actually care about: how is the change of macroeconomic "
        "variables impacting my sales year to year.&rsquo;")
table([
    ["Term", "Type", "Definition"],
    ["Contribution", "Level",
     "How much volume a driver accounts for <i>in</i> a period: the sum of "
     "coefficient &times; value over that period's weeks."],
    ["Due-to", "Change",
     "How much of the movement <i>between</i> two periods a driver explains: "
     "its contribution in period B minus its contribution in period A."],
], [1.15, 0.75, 5.0])
p("The decomposition is exact by construction. Intercept plus the sum over "
  "drivers of coefficient &times; value reproduces the fitted value every "
  "week, with the intercept reported as its own line. Verified to "
  "5.8&times;10<super>-11</super>; due-tos sum to the change in modeled "
  "volume to 6&times;10<super>-9</super>.")

h3("Unequal-length periods")
p("Periods of different lengths are compared as <b>per-week averages</b> "
  "automatically. Comparing totals would make a 52-week period beat a 4-week "
  "one for reasons that have nothing to do with the drivers. The caption "
  "states which basis is in use.")

h3("Averaging rule for media")
p("Average weekly contribution for media &mdash; any variable with an adstock "
  "decay &mdash; is taken <b>only over weeks with real execution</b> "
  "(non-zero raw spend), never diluted by the zero weeks of a flight that "
  "started mid-window. Non-media drivers average over all weeks. This rule "
  "was requested explicitly by the sponsor.")

h3("Model years")
p("Consecutive 52-week blocks are labelled counting back from the latest "
  "week, so the most recent block is the current model year. Labels carry "
  "their date range, for example &lsquo;Year 2 (Jan 2025-Dec 2025)&rsquo;. "
  "Current and Previous always map to the latest and prior model year "
  "regardless of window length.")

h2("Comparable coefficients", "2.10")
quote("You can't really compare the coefficient for ACV weighted distribution "
      "versus the five-year inflation expectations or gas price, because the "
      "support that you're using to build the model is not comparable.",
      "Arko, 2026-07-27")
p("Raw coefficients cannot be ranked across variables, because the inputs sit "
  "on different supports: a savings rate moves between 2 and 8, impressions "
  "between zero and millions. The engine therefore reports two derived "
  "quantities that put every driver in the <b>same unit &mdash; volume</b>:")
table([
    ["Quantity", "Formula", "Reads as"],
    ["Impact / SD", "&beta; &times; SD(x)",
     "Volume moved by a one-standard-deviation change in this variable"],
    ["Impact / range", "&beta; &times; (max &minus; min)",
     "Volume moved across the variable's full observed range"],
], [1.3, 1.5, 4.1], mono_cols=(1,))
p("Both are invariant to the scale setting, so the comparison is stable "
  "however the analyst sets units. Worked example, "
  "Brand&nbsp;1&nbsp;&times;&nbsp;Channel&nbsp;1:")
table([
    ["Variable", "Raw coefficient", "Impact / SD"],
    ["Seasonality Index", "61,473", "13,193"],
    ["ACV Weighted Distribution", "3,107", "13,151"],
], [3.0, 1.95, 1.95], mono_cols=(1, 2))
p("The raw coefficients look twenty times apart. In volume terms the two "
  "drivers are effectively tied. <b>Impact / SD is the column that can "
  "legitimately be ranked.</b>")

callout("Why not min-max normalization",
        "Arko asked whether the data could be normalized before modeling. For "
        "an unregularized constrained least-squares fit, min-max "
        "normalization is a linear re-parameterization &mdash; as he himself "
        "noted, &lsquo;it does nothing to your model&rsquo;. Selection "
        "already standardizes internally, so it was scale-invariant already. "
        "Normalizing and then inverting every reported number would introduce "
        "exactly the risk he warned about &mdash; &lsquo;when you report the "
        "results you have to go back to the original scale&rsquo; &mdash; "
        "while changing no result. So the actual problem, comparability, is "
        "solved directly instead.")

# --- optimizer
story.append(PageBreak())
h2("Media curve optimization", "2.11")
quote("Does it always have to be 0.5? Can we create an optimization engine "
      "that will pick the best ad stock based on correlation with the data, "
      "predictive power in a regression? We can do the same thing with "
      "saturation. If we want a fully automated system, that would be dope as "
      "hell.", "Alex")
p("Searches adstock decay and both Hill parameters for each media variable in "
  "a slice's model.")
table([
    ["Aspect", "Design"],
    ["Search space",
     "decay &isin; {0, 0.2, 0.4, 0.6, 0.8} &times; midpoint &isin; {0.2, 0.35, "
     "0.5, 0.65, 0.8} &times; slope &isin; {0.5, 1.0, 1.5, 2.5}, plus the "
     "option of no saturation &mdash; so the search can always decline to "
     "saturate."],
    ["Strategy",
     "Coordinate descent: one variable at a time, others held fixed, up to "
     "two passes, stopping early on convergence. A full joint search is "
     "grid<super>k</super> in the number of media variables; coordinate "
     "descent is k&times;grid per pass and finds the same optimum on a "
     "near-separable problem."],
    ["Scoring",
     "Mean MAPE across three rolling-origin cross-validation folds, all of "
     "which end before the outer holdout."],
    ["Adoption rule",
     "A candidate must not be worse in <b>any</b> fold (strict dominance) "
     "<i>and</i> must improve the mean by at least 0.5 percentage points. "
     "Without the floor the search would trade a 0.01% gain for an "
     "implausible curve shape."],
    ["Structure",
     "Held fixed during the search, so the score reflects curve shapes only "
     "&mdash; and so the search stays fast."],
    ["Output",
     "Parameters only. The optimizer writes candidates into the config table; "
     "it does not save and does not run. Nothing here becomes a second, "
     "divergent modeling path."],
], [1.15, 5.75])
p("Runtime is approximately 1.7 seconds per slice on average, up to about 9 "
  "seconds for a slice with three media variables &mdash; roughly 4 minutes "
  "for all 76 slices.", "small")
p("Section 7.3 reports what the optimizer actually achieves, including where "
  "it does not help.")

# ═══════════════════════════════════════════════════════ 3. DASHBOARD
story.append(PageBreak())
h1("The dashboard, screen by screen", "3")

h2("Global elements", "3.1")
h3("Left rail")
bullets([
    "<b>Workspace navigation</b> &mdash; six screens: Diagnostics, Variables, "
    "Contributions, Model Runs, Export, Definitions.",
    "<b>Dataset panel</b> &mdash; type any data file path and press "
    "<i>Load data</i>. Shows the loaded file and a status line. Product and "
    "channel lists populate from the file.",
])

h3("Top bar &mdash; results filter only")
p("The top bar carries <b>Product</b> and <b>Channel</b> and nothing else "
  "that can be edited. These are filters over already-computed results: "
  "changing them re-reads the cache and does not re-run anything.")
p("Target and window are <i>modeling</i> decisions, so they live in exactly "
  "one place &mdash; the Variables screen &mdash; and are echoed here "
  "read-only with the actual calendar range, for example "
  "&lsquo;Volume Sales &middot; 104 wks &middot; Jan 07, 2024 &ndash; Dec 28, "
  "2025&rsquo;.")
quote("Remove everywhere that you can change the target. You should only be "
      "able to change the target in one spot &hellip; The only thing you "
      "should be filtering on are results, which would be your product and "
      "channel.", "Alex")
p("A run-status indicator on the right reports whether the current view came "
  "from a fresh run or from the cache.", "small")

h3("The results cache &mdash; run once, filter many")
quote("Instead of having to rerun every time, you just create the data that "
      "filters into this for everything, and then you just filter down. It's "
      "going to take a little bit longer for the first one, but then when "
      "we're evaluating, it's instantaneous.", "Alex")
p("A batch run computes every slice once and caches the full results. "
  "Diagnostics, Contributions and Export then read and filter that cache. "
  "Measured: a full batch is about 19 seconds for 76 models; a filter change "
  "is about 140 milliseconds.")
p("The cache is keyed by <b>data file, target, window, strictness and family "
  "caps</b>. Change any modeling knob and the screens ask for a fresh batch "
  "rather than showing a blend of two settings. This also removes a hazard "
  "the sponsor named directly &mdash; ending up with one model on 156 weeks "
  "while everything else on screen is on 104. Re-tuning a single slice on "
  "Variables and pressing Run refreshes just that entry.")
p("If a slice is not in the cache, the screen says why &mdash; not yet run, "
  "skipped for insufficient data with the reason, or failed &mdash; rather "
  "than showing the previous slice's numbers, which would be actively "
  "misleading.")

story.append(PageBreak())
h2("Diagnostics", "3.2")
p("Everything needed to judge a single model.")
table([
    ["Element", "What it shows"],
    ["Metric strip", "R&sup2;, adjusted R&sup2;, in-sample MAPE, holdout MAPE, "
     "Durbin&ndash;Watson, predictor count with the forced count. Holdout "
     "MAPE above 40% carries a REVIEW flag."],
    ["Actual vs fitted", "Weekly actual and fitted lines over the window, with "
     "the reserved holdout period shaded and the out-of-sample forecast drawn "
     "separately. The caption states the slice, target and exact window dates."],
    ["Residuals vs fitted", "Scatter for heteroscedasticity and structure."],
    ["Residual distribution", "Histogram for skew and outliers."],
    ["Coefficient table", "One row per in-model variable plus the intercept: "
     "family chip, expected sign, t-statistic with a visual bar, VIF, "
     "coefficient (with its scale divisor when set), <b>Impact / SD</b>, and "
     "average weekly contribution. The caption reports how many of how many "
     "candidates were selected, sign-conflict count and high-VIF count."],
    ["Media response curves", "Fitted response curve per media variable with "
     "the weeks actually executed plotted on it, and the half-saturation "
     "point marked."],
], [1.35, 5.55])

h3("Reading the response curve chart")
bullets([
    "<b>A flattening curve</b> is a diminishing return. The dotted vertical "
    "is the half-saturation point: past it, the next unit of execution "
    "returns less than the last.",
    "<b>A straight line</b> means no saturation was fitted; the variable is "
    "linear in execution.",
    "<b>Where the dots sit</b> is the current operating point. Clustered on "
    "the steep part means headroom; out on the flat means extra spend is "
    "buying little.",
])

h2("Variables", "3.3")
p("The only screen where modeling decisions are made.")

h3("Modeling settings")
p("Target (any raw column may be the dependent variable; Volume Sales, Dollar "
  "Sales and Unit Sales are offered first) and window (52 / 104 / 156 weeks), "
  "with the resolved calendar range displayed alongside.")

h3("Variable selection controls")
p("Strictness preset, and an editable <b>max per family</b> table covering "
  "Distribution, Trade, Media, Competitive, Macro, Price, Category Price, "
  "Seasonality and Trend. Blank means no limit; blank, zero and unparseable "
  "entries are all treated as no limit so a half-filled table cannot "
  "accidentally throttle a family. A live note reports the effective "
  "settings.")

h3("Configuration scope")
p("Edit the Product default (all channels) or a Channel-specific override. A "
  "status line states which file is being edited and whether an override "
  "exists. <b>A Variables-tab Run is WYSIWYG</b>: it runs exactly the config "
  "on screen, which can differ from what the batch would resolve.")

h3("The variable table")
p("One row per candidate variable, with all of the columns documented in "
  "section 4, plus three read-only context columns showing the current "
  "coefficient, Impact / SD and contribution for the last run &mdash; gated "
  "to the exact Product &times; Channel &times; Target of that run, so "
  "switching slices blanks them rather than showing another slice's numbers. "
  "Rows with role <i>force</i> are tinted blue and <i>exclude</i> rows pink.")

h3("Actions")
table([
    ["Control", "Effect"],
    ["Due-to period", "Which period the context column shows (Y1 / Y2 / Total)."],
    ["Hide excluded", "View filter only. The underlying data stays complete, "
                      "so excluded rows are never dropped on save."],
    ["Upload template", "Apply a filled template (.xlsx/.csv). Rows with "
                        "channel=ALL write the Product default; a specific "
                        "channel writes that override. A variable in the data "
                        "but not listed for a group becomes excluded."],
    ["Download template", "This product's config (default plus every existing "
                          "override) as an editable file."],
    ["Reset to default", "In a channel scope, delete that channel's override "
                         "so it inherits the Product default again."],
    ["Regenerate defaults", "Rebuild the config from the data's naming "
                            "heuristics."],
    ["Save config", "Validate and write. Bounds with lo &ge; hi are rejected "
                    "with the offending variables named."],
    ["Optimize media curves", "Run the curve search and write candidates into "
                              "the table. Reports the cross-validated gain. "
                              "Does not save and does not run."],
    ["Run model", "Persist the on-screen edits, fit this slice, refresh its "
                  "cache entry and jump to Diagnostics."],
], [1.5, 5.4])
p("Template uploads are <b>atomic</b>: every group is validated first and "
  "nothing is written unless all pass, so a later failure cannot leave a "
  "partial edit. Unknown variable names are rejected rather than silently "
  "excluding everything.", "small")

story.append(PageBreak())
h2("Contributions", "3.4")
table([
    ["Element", "What it shows"],
    ["Contribution totals by model year",
     "Grouped horizontal bars per driver per model year."],
    ["Period panel",
     "A driver chart with a mode toggle &mdash; <b>Contribution</b> (level in "
     "the selected period) or <b>Due-to</b> (change between the two selected "
     "periods) &mdash; plus a primary period and a comparison period."],
    ["Contributions by model year table",
     "Signed sums per model year with the year-over-year change and a "
     "percentage pill."],
    ["Time map upload",
     "A .csv/.xlsx with a date (or week) column and a period column defines "
     "custom periods, which then appear in the period pickers."],
], [1.7, 5.2])
p("Built-in periods: entire window, current model year, previous model year, "
  "calendar Q4 of the latest year, latest 13 weeks, latest 4 weeks, plus any "
  "uploaded mapped periods that overlap the run. The comparison list excludes "
  "the currently selected primary period, so a period can never be compared "
  "with itself. A period that no longer exists for the current run (for "
  "example &lsquo;previous year&rsquo; in a 52-week window) falls back to a "
  "valid option rather than rendering mislabelled.")
p("In due-to mode the comparison period is the baseline and the primary "
  "period is the focus, so the chart reads &lsquo;what changed getting "
  "<i>to</i> the period I selected&rsquo;. The caption names both periods, "
  "the week counts, whether the basis is totals or per-week averages, and the "
  "total modeled change.")

h2("Model Runs", "3.5")
table([
    ["Element", "What it shows"],
    ["Summary pills", "Model count, median R&sup2;, median holdout MAPE, "
     "count needing review, the window range, the count not modeled, active "
     "caps and strictness if not default, and the run duration."],
    ["Model table", "Every slice: grade (fully coloured green/amber/red), "
     "product, channel, target, weeks, selling weeks, predictors, forced "
     "count, R&sup2;, in-sample MAPE, holdout MAPE and sign conflicts. "
     "Sortable and filterable; click a row to load that slice."],
    ["Not modeled table", "Every skipped slice with weeks in window, selling "
     "weeks and the reason."],
    ["Run all combinations", "Build the full results cache."],
    ["Run model", "Re-run the currently selected slice only."],
    ["Reload summary", "Re-read the saved summary CSV from disk."],
], [1.5, 5.4])

h2("Export", "3.6")
p("One Excel workbook, either for the current slice or for every "
  "combination. Served from the cache when available, so the export cannot "
  "silently disagree with what the screens show. Sheets are documented in "
  "section 5.")

h2("Definitions", "3.7")
p("An in-app reference covering fit metrics, grading, contribution versus "
  "due-to, variable roles and configuration, media curves, selection "
  "hyperparameters, and the window and validation method. It tracks the "
  "engine's actual behaviour, so it is the fastest way for a new user to "
  "check what a number on screen means without leaving the tool.")

# ═══════════════════════════════════════════════════════ 4. CONFIG
story.append(PageBreak())
h1("Configuration reference", "4")
h2("Variable config columns", "4.1")
table([
    ["Column", "Values", "Meaning"],
    ["variable", "column name", "The raw data column. Not editable in the UI."],
    ["family", "text", "Grouping used for chips, per-family caps and "
                       "reporting. Standard families are listed in 3.3."],
    ["sign", "positive / negative / unconstrained",
     "Hard guardrail on the coefficient's direction."],
    ["adstock_decay", "0 &ndash; 1, or blank",
     "Carry-over decay. Blank means no adstock; any variable with a decay is "
     "treated as media for averaging rules."],
    ["scale", "&gt; 0, default 1",
     "Divide the variable before modeling. Units only; never changes the fit."],
    ["sat_midpoint", "(0, 1], or blank",
     "Hill half-saturation as a fraction of peak execution."],
    ["sat_slope", "&gt; 0, or blank",
     "Hill shape. Both saturation fields must be set, or neither."],
    ["coef_lower", "number or blank",
     "Hard lower bound in original units; overrides the sign default."],
    ["coef_upper", "number or blank",
     "Hard upper bound; must be strictly greater than coef_lower."],
    ["role", "auto / force / exclude",
     "<b>auto</b>: the model decides. <b>force</b>: always in, bypassing "
     "selection and VIF pruning. <b>exclude</b>: never enters."],
], [1.15, 1.5, 4.25], mono_cols=(0,))

h2("Template columns for upload and download", "4.2")
p("The template adds <font face='DJM' size='8.5'>product</font> and "
  "<font face='DJM' size='8.5'>channel</font> to the front of the config "
  "columns. Channel <font face='DJM' size='8.5'>ALL</font> means the Product "
  "default; any other value writes that channel's override.")
code("product, channel, variable, family, sign, role,\n"
     "coef_lower, coef_upper, adstock_decay, scale, sat_midpoint, sat_slope")

h2("Run-level settings", "4.3")
table([
    ["Setting", "Default", "Meaning"],
    ["target", "Volume Sales", "The dependent variable."],
    ["model_weeks", "104", "Window length in weeks."],
    ["window_end", "dataset anchor", "The shared week every window ends on."],
    ["min_weeks", "52", "Minimum weeks in window to model a slice."],
    ["min_nonzero_weeks", "26", "Minimum selling weeks in window."],
    ["holdout_weeks", "13 (auto)", "Reserved validation tail; 0 disables."],
    ["max_per_family", "{} (none)", "Per-family entry caps."],
    ["vif_threshold", "10.0", "VIF pruning threshold."],
    ["p_enter", "0.05", "Stepwise entry significance."],
    ["default_media_decay", "0.5", "Fallback adstock decay."],
    ["force_include / exclude", "[]", "Legacy overrides; roles in the config "
                                      "are the preferred mechanism."],
], [1.6, 1.2, 4.1], mono_cols=(0, 1))

# ═══════════════════════════════════════════════════════ 5. EXPORT
story.append(PageBreak())
h1("Export file reference", "5")
p("A single workbook with four sheets, plus an errors sheet when anything was "
  "not modeled. The sponsor's stated workflow is to pull these into a pivot "
  "table, so the weekly sheet carries its period columns rather than "
  "requiring a lookup.")

h2("fit_statistics", "5.1")
code("product, channel, target, window_weeks, window_start, window_end,\n"
     "weeks_modeled, selling_weeks, R2, adj_R2, MAPE_in_pct,\n"
     "MAPE_holdout_pct, grade, n_selected, n_forced")
p("The grade cell is filled green, amber or red with white bold text.", "small")

h2("fit_structure", "5.2")
code("product, channel, variable, family, sign, role, coef_lower,\n"
     "coef_upper, adstock_decay, scale, sat_midpoint, sat_slope,\n"
     "current_coefficient, in_last_model")
p("This is the upload template format plus two read-only context columns "
  "requested by the sponsor &mdash; <font face='DJM' size='8.5'>current_coefficient</font> "
  "and <font face='DJM' size='8.5'>in_last_model</font>. Both are ignored on "
  "re-upload, so an exported model can be tweaked in Excel and re-uploaded "
  "without breaking the round trip.")

h2("weekly_due_tos", "5.3")
code("product, channel, date, model_year, period, quarter, year,\n"
     "driver, due_to")
p("Long format, one row per week per driver, including the intercept line so "
  "the values reconcile exactly to the fitted series. The "
  "<font face='DJM' size='8.5'>period</font> column carries labels from an "
  "uploaded time map. The date range is exactly the modeling window &mdash; "
  "no pre-window history can appear here.")

h2("due_to_change", "5.4")
code("product, channel, period_from, period_to, weeks_from, weeks_to,\n"
     "basis, driver, due_to_change, share_of_total_change_pct")
p("Period-over-period due-tos, built for the model-year pair and for "
  "consecutive uploaded time-map periods. The "
  "<font face='DJM' size='8.5'>basis</font> column states whether the "
  "comparison used totals or per-week averages.")

h2("errors", "5.5")
p("Present only when something was not modeled: product, channel and the "
  "reason, including insufficient-data-in-window messages with their counts.")

# ═══════════════════════════════════════════════════════ 6. CLI
h1("Command-line tools", "6")
table([
    ["Script", "Purpose", "Outputs"],
    ["run_all.py", "Batch every slice with the shared fixed window. Accepts a "
     "brand range for resumable chunks.",
     "outputs/all/&lt;slice&gt;_coefficients.csv, "
     "&lt;slice&gt;_contrib_by_year.csv, all_models_summary.csv, "
     "all_models_skipped.csv, all_models_failures.csv"],
    ["run_brand1_channel1.py", "Single-slice run for debugging.", "Console"],
    ["reference_alex_curvefit.py",
     "Verifies our bounded least squares matches the sponsor's curve_fit "
     "implementation.", "PASS/FAIL with the maximum relative difference"],
    ["scaling_test.py", "Runtime scaling measurements.", "Console / CSV"],
], [1.45, 2.5, 2.95], mono_cols=(0,))
p("BLAS thread counts are pinned before NumPy loads in the batch runners: "
  "many small least-squares solves thrash badly with many threads, because "
  "each solve is tiny and spawn overhead dominates.", "small")

# ═══════════════════════════════════════════════════════ 7. RESULTS
h1("Current results", "7")
h2("Model set", "7.1")
table([
    ["Measure", "Value"],
    ["Slices modeled", "76"],
    ["Slices not modeled", "6"],
    ["Failures", "0"],
    ["Modeling window", "104 weeks, Jan 07 2024 &ndash; Dec 28 2025 (identical "
                        "for every slice)"],
    ["Median R&sup2;", "0.881"],
    ["Median holdout MAPE", "11.5%"],
    ["Grades", "35 Good, 35 Moderate, 6 Bad"],
    ["Full batch runtime", "~19 seconds"],
], [1.8, 5.1])

h2("Slices deliberately not modeled", "7.2")
table([
    ["Slice", "Weeks in window", "Selling weeks", "Reason"],
    ["Brand 1 &times; Channel 8", "0", "0", "Data ends June 2023, entirely "
                                            "before the window"],
    ["Brand 2 &times; Channel 8", "0", "0", "Data ends June 2023, entirely "
                                            "before the window"],
    ["Brand 1 &times; Channel 6", "104", "24", "Almost no selling weeks &mdash; "
                                               "this was the 254% MAPE model "
                                               "the sponsor flagged"],
    ["Brand 2 &times; Channel 5", "104", "11", "Almost no selling weeks"],
    ["Brand 5 &times; Channel 6", "&mdash;", "0", "Not sold in this channel"],
    ["Brand 7 &times; Channel 6", "&mdash;", "0", "Not sold in this channel"],
], [1.75, 1.05, 0.95, 3.15])
p("All four data-driven skips were previously producing models. Two of them "
  "were the source of the disjoint history in the exports.", "small")

h2("What the curve optimizer achieves", "7.3")
p("Measured across 25 slices, of which 14 have media in the model. Full "
  "per-slice results in "
  "<font face='DJM' size='8.5'>outputs/curve_optimizer_experiment.csv</font>.")
table([
    ["Measure", "Result"],
    ["Cross-validated MAPE (what the optimizer optimizes)",
     "Improved <b>4 of 4</b> times it fired &mdash; mean &minus;2.07 points"],
    ["Reported holdout (never touched by the search)",
     "Improved <b>2 of 4</b> &mdash; mean <b>+0.85 points</b>, median +0.24"],
    ["Slices left unchanged", "10 of the 14 with media"],
], [3.1, 3.8])
table([
    ["Slice", "Media vars", "Curves changed", "CV before &rarr; after",
     "Holdout before &rarr; after"],
    ["Brand 1 &times; Ch 1", "2", "1", "12.37 &rarr; 11.57", "19.64 &rarr; 25.51"],
    ["Brand 1 &times; Ch 2", "1", "1", "19.60 &rarr; 18.22", "7.22 &rarr; 7.13"],
    ["Brand 3 &times; Ch 2", "1", "1", "7.88 &rarr; 4.91", "6.07 &rarr; 6.65"],
    ["Brand 4 &times; Ch 2", "3", "2", "7.06 &rarr; 3.90", "9.84 &rarr; 6.90"],
], [1.35, 0.85, 1.05, 1.85, 1.8], mono_cols=(1, 2, 3, 4))
callout("The honest reading",
        "The optimizer reliably improves the thing it optimizes. Whether that "
        "transfers to genuinely unseen data is close to a coin flip on this "
        "dataset. It does behave conservatively &mdash; it declines to change "
        "anything on 10 of the 14 slices with media &mdash; and when it fires "
        "on a slice with real media weight it can be worth it "
        "(Brand&nbsp;4&nbsp;&times;&nbsp;Channel&nbsp;2 gained 2.94 points of "
        "genuine out-of-sample accuracy). But Brand&nbsp;1&nbsp;&times;&nbsp;"
        "Channel&nbsp;1 lost 5.87. <b>It therefore ships off by default</b>, "
        "as an opt-in tool that reports its cross-validated gain and leaves "
        "the analyst to check the holdout.",
        WARN, colors.HexColor("#FEF6EE"))
p("Why the transfer is weak: media is a small share of these models &mdash; "
  "the dominant drivers are distribution, seasonality, trade and competitive "
  "pressure &mdash; and 104 weeks with roughly ten predictors is thin for "
  "identifying two extra non-linear parameters per media variable. Robyn and "
  "Meridian fit these with Bayesian priors and hierarchical pooling across "
  "geographies for exactly this reason: the priors do the work the data "
  "cannot. Partial pooling of curve parameters across a brand's channels is "
  "the natural next step.")

# ═══════════════════════════════════════════════════════ 8. DECISIONS
story.append(PageBreak())
h1("Design decisions and rationale", "8")
table([
    ["Decision", "Alternative rejected", "Reasoning"],
    ["Fixed calendar window anchored to the dataset",
     "Last N rows per slice",
     "The alternative modeled delisted slices on stale history and leaked "
     "disjoint due-tos into exports. Data should depend on the window, not "
     "the reverse."],
    ["Skip thin slices with a stated reason",
     "Model everything, flag afterwards",
     "A model built on 24 selling weeks is not a weak model, it is not a "
     "model. But it must be visible, so it is listed with counts and a "
     "reason rather than dropped."],
    ["Bounded least squares",
     "Ridge regression",
     "Ridge cannot honour sign or box constraints, and a model asserting that "
     "price rises lift volume is not deliverable whatever its fit."],
    ["Forward stepwise",
     "Lasso, tree importance",
     "Each addition is one testable, explainable decision. Lasso paths are "
     "hard to defend line by line; trees do not give signed bounded "
     "coefficients."],
    ["VIF pruning",
     "Pairwise correlation, PCA",
     "VIF catches redundancy against a <i>combination</i> of variables; "
     "unlike PCA it preserves raw explainable variables."],
    ["Time-based holdout",
     "Random K-fold",
     "Weekly series are autocorrelated; random folds leak the future into "
     "training and flatter the model."],
    ["Report comparable coefficients",
     "Min-max normalize the inputs",
     "For unregularized constrained least squares, normalization is a linear "
     "re-parameterization that changes nothing, while the inverse transform "
     "adds real risk. Reporting &beta;&times;SD solves the actual problem."],
    ["Three-way split for curve search",
     "Score on the reported holdout",
     "Scoring on the reported holdout would make every optimized model look "
     "better and none of them be better."],
    ["Rolling-origin CV with strict dominance",
     "Single inner validation window",
     "A single window over-fitted one quarter: it improved that window while "
     "making the untouched holdout worse."],
    ["Curve optimization off by default",
     "Optimize automatically in the batch",
     "Measured transfer to unseen data is unreliable. An engine that reports "
     "when its own optimization did not generalize is worth more than one "
     "that always claims a win."],
    ["Cache keyed by every modeling knob",
     "One cache per dataset",
     "Prevents comparing a slice modeled under one setting against slices "
     "modeled under another."],
    ["Modeling controls in one place",
     "Controls on every screen",
     "Changing a target on a results screen implies a re-model. Results "
     "screens filter; they do not model."],
], [1.5, 1.4, 4.0])

# ═══════════════════════════════════════════════════════ 9. VERIFICATION
story.append(PageBreak())
h1("Verification status", "9")
p("Every claim below is checked by an automated test run against the real "
  "dataset.")
table([
    ["Check", "Result"],
    ["Sponsor equivalence &mdash; our bounded least squares versus Alex's "
     "curve_fit implementation", "PASS, max relative coefficient difference "
     "7.0&times;10<super>-8</super>"],
    ["Decomposition exact &mdash; intercept + &Sigma;(&beta;&middot;x) equals "
     "the fitted value", "PASS, 5.8&times;10<super>-11</super>"],
    ["Decomposition still exact with saturation applied",
     "PASS, 1.0&times;10<super>-10</super>"],
    ["Due-tos sum to the change in modeled volume",
     "PASS, 6.1&times;10<super>-9</super>"],
    ["Scale is a units change only &mdash; fitted values unchanged",
     "PASS, 1.2&times;10<super>-10</super>"],
    ["Impact / SD invariant to the scale setting", "PASS"],
    ["One calendar window across all 76 models", "PASS"],
    ["Thin slices skipped with reasons; none appear as models", "PASS"],
    ["No pre-window rows in the export", "PASS, range is exactly the window"],
    ["Filter change reads cache without refitting", "PASS, ~140 ms"],
    ["Skipped slice shows a reason, not stale numbers", "PASS"],
    ["Family caps respected across the batch; forced variables survive", "PASS"],
    ["Strictness monotonic in predictor count", "PASS, 4.4 / 6.5 / 10.0"],
    ["Defaults reproduce the previous 76-model baseline exactly", "PASS"],
    ["Hill function: H(k)=0.5, monotone, bounded, H(0)=0, scale-invariant",
     "PASS"],
    ["Optimizer folds all end before the outer holdout", "PASS"],
    ["Optimizer strict dominance &mdash; no fold worse than baseline", "PASS"],
    ["Half-specified or out-of-range curve parameters cleared safely", "PASS"],
    ["Config template round-trip preserves every column", "PASS"],
    ["Dashboard callback wiring &mdash; no missing or duplicate component ids",
     "PASS, 78 components, 32 callbacks"],
], [4.3, 2.6])

# ═══════════════════════════════════════════════════════ 10. LIMITS
h1("Known limits and what is not built", "10")
h2("Limits of the current system", "10.1")
bullets([
    "<b>Curve optimization does not reliably transfer.</b> Quantified in 7.3. "
    "Off by default and documented rather than hidden.",
    "<b>No pooling across slices.</b> Each Brand &times; Channel is modeled "
    "independently. A channel with thin media history cannot borrow strength "
    "from its siblings, which is precisely what limits the curve work.",
    "<b>Grade thresholds are a team convention</b> (10% / 40% holdout MAPE) "
    "and should be confirmed with the client.",
    "<b>Six slices are not modeled</b> by design. Four of those reflect real "
    "data gaps that are worth investigating at source.",
    "<b>Model years are 52-week blocks counted back from the anchor</b>, not "
    "fiscal years. An uploaded time map is the way to impose a fiscal "
    "calendar.",
    "<b>The results cache is in memory.</b> Restarting the dashboard requires "
    "re-running the batch, which takes about 19 seconds.",
])
h2("Deliberately not built", "10.2")
bullets([
    "<b>Literal min-max normalization of inputs.</b> Rationale in 2.10; the "
    "comparability problem is solved by reporting &beta;&times;SD instead.",
    "<b>Bayesian priors or hierarchical pooling for media curves.</b> This is "
    "the identified next step for making saturation and decay estimation "
    "reliable on data of this length.",
    "<b>Automatic curve optimization inside the batch.</b> Available "
    "per-slice on demand; running it unattended across all slices would "
    "silently adopt curves that measurably do not generalize.",
])

story.append(Spacer(1, 16))
story.append(hr(NAVY, 1.1))
story.append(Spacer(1, 8))
p("Companion documents in the repository: "
  "<font face='DJM' size='8.5'>docs/Media_Curves_Math.md</font> (full "
  "derivation and the optimizer experiment), "
  "<font face='DJM' size='8.5'>docs/Action_Items_Alex_Meeting_2026-07-27.md</font> "
  "(sponsor request tracking), "
  "<font face='DJM' size='8.5'>docs/Project_Brief.md</font>, "
  "<font face='DJM' size='8.5'>docs/Findings.md</font>.", "small")


# ══════════════════════════════════════════════════════════════ RENDER
class Doc(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if hasattr(flowable, "_toc"):
            lvl, text = flowable._toc
            self.notify("TOCEntry", (lvl, text, self.page))


def decorate(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(0.5)
        canvas.line(0.85 * inch, LETTER[1] - 0.62 * inch,
                    LETTER[0] - 0.85 * inch, LETTER[1] - 0.62 * inch)
        canvas.setFont("DJ", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.85 * inch, LETTER[1] - 0.55 * inch,
                          "BIG CHALK MIX ENGINE  ·  FEATURE REFERENCE")
        canvas.drawRightString(LETTER[0] - 0.85 * inch, LETTER[1] - 0.55 * inch,
                               "August 2026")
        canvas.setFont("DJ", 8)
        canvas.setFillColor(INK3)
        canvas.drawCentredString(LETTER[0] / 2, 0.5 * inch, str(doc.page))
    canvas.restoreState()


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "BigChalk_Mix_Engine_Feature_Reference.pdf")
doc = Doc(OUT, pagesize=LETTER, leftMargin=0.85 * inch, rightMargin=0.85 * inch,
          topMargin=0.85 * inch, bottomMargin=0.8 * inch,
          title="Big Chalk Mix Engine - Complete Feature Reference",
          author="NU MSDS Capstone Team")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="n")
doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=decorate)])
doc.multiBuild(story)
print("WROTE", OUT)
