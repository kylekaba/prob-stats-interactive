"""
Manim Community scenes for the Probability & Statistics interactive site.
Render with, e.g.:
    manim -qm scenes.py CLTScene
"""
from manim import *
import numpy as np

# ---- shared palette (matches the book/site) ----
ACCENT = "#1f5fa8"
ACCENT_D = "#16406f"
GREEN = "#2a8f5a"
RED = "#b03030"
GOLD = "#c9a227"
INK = "#1c2530"

config.background_color = "#ffffff"
Text.set_default(color=INK, font="Helvetica")

rng = np.random.default_rng(7)


def title_card(title, subtitle=None):
    t = Text(title, color=ACCENT_D, weight=BOLD).scale(0.85)
    grp = VGroup(t)
    if subtitle:
        s = Text(subtitle, color=GREEN).scale(0.45)
        s.next_to(t, DOWN, buff=0.25)
        grp.add(s)
    grp.to_edge(UP, buff=0.5)
    return grp


# ======================================================================
class CLTScene(Scene):
    """The sample mean of a skewed source becomes normal."""
    def construct(self):
        head = title_card("The Central Limit Theorem",
                          "the average of many draws becomes a bell curve")
        self.play(FadeIn(head, shift=DOWN*0.2))

        # source: heavily right-skewed population (U^3)
        ax_src = Axes(x_range=[0, 1, 0.25], y_range=[0, 4, 1], x_length=5.2, y_length=2.4,
                      axis_config={"color": INK, "include_tip": False,
                                   "font_size": 22}).to_edge(LEFT, buff=0.7).shift(DOWN*0.4)
        src_lbl = Text("source: a lopsided distribution", color=ACCENT).scale(0.4)
        src_lbl.next_to(ax_src, UP, buff=0.15)
        src_curve = ax_src.plot(lambda x: 3*(1-x)**2, color=ACCENT, x_range=[0.001, 1])
        src_area = ax_src.get_area(src_curve, x_range=[0, 1], color=ACCENT, opacity=0.18)
        self.play(Create(ax_src), FadeIn(src_lbl))
        self.play(Create(src_curve), FadeIn(src_area))
        self.wait(0.3)

        # right: histogram of sample means built up as n increases
        ax_m = Axes(x_range=[0, 1, 0.25], y_range=[0, 6, 2], x_length=5.2, y_length=2.4,
                    axis_config={"color": INK, "include_tip": False,
                                 "font_size": 22}).to_edge(RIGHT, buff=0.7).shift(DOWN*0.4)
        m_lbl = Text("distribution of the sample mean", color=GREEN).scale(0.4)
        m_lbl.next_to(ax_m, UP, buff=0.15)
        self.play(Create(ax_m), FadeIn(m_lbl))

        nbins = 24
        edges = np.linspace(0, 1, nbins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        binw = edges[1] - edges[0]

        # population mean/var of the lopsided density f(x)=3(1-x)^2 drawn on the left
        mu, var = 0.25, 3/80

        def make_hist(n, reps=4000):
            # sample from f(x)=3(1-x)^2 via inverse-CDF  x = 1-(1-u)^(1/3)
            draws = 1.0 - (1.0 - rng.random((reps, n))) ** (1/3)
            means = draws.mean(axis=1)
            counts, _ = np.histogram(means, bins=edges, density=True)
            peak = max(counts.max(), 1e-9)
            scale = 5.0 / peak          # tallest bar reaches y=5 (axis goes to 6)
            bars = VGroup()
            for c, h in zip(centers, counts):
                hh = h * scale
                if hh <= 0.002:
                    continue
                base = ax_m.c2p(c, 0); top = ax_m.c2p(c, hh)
                w = ax_m.c2p(binw, 0)[0] - ax_m.c2p(0, 0)[0]
                bar = Rectangle(width=w*0.92, height=max(top[1]-base[1], 0.001),
                                fill_color=GREEN, fill_opacity=0.55, stroke_width=0.5,
                                stroke_color=WHITE)
                bar.move_to([base[0], (base[1]+top[1])/2, 0])
                bars.add(bar)
            return bars, scale

        def normal_overlay(n, scale):
            se = np.sqrt(var / n)
            return ax_m.plot(
                lambda x, s=scale, sd=se: s*np.exp(-0.5*((x-mu)/sd)**2)/(sd*np.sqrt(2*np.pi)),
                color=RED, x_range=[max(0.001, mu-4*se), min(1, mu+4*se)])

        n_tracker = Text("n = 1", color=ACCENT_D, weight=BOLD).scale(0.55)
        n_tracker.next_to(ax_m, DOWN, buff=0.3)
        bars, scale = make_hist(1)
        self.play(FadeIn(bars), FadeIn(n_tracker))
        self.wait(0.4)

        normal = None
        for n in [2, 5, 15, 40]:
            new_bars, scale = make_hist(n)
            new_n = Text(f"n = {n}", color=ACCENT_D, weight=BOLD).scale(0.55).move_to(n_tracker)
            self.play(Transform(bars, new_bars), Transform(n_tracker, new_n), run_time=0.9)
            if n >= 15:
                target = normal_overlay(n, scale)
                if normal is None:
                    self.play(Create(target), run_time=0.6); normal = target
                else:
                    self.play(Transform(normal, target), run_time=0.6)
                self.wait(0.2)
        self.wait(0.3)
        caption = Text("Skewed source  →  the average is normal.", color=ACCENT_D).scale(0.5)
        caption.to_edge(DOWN, buff=0.25)
        self.play(FadeIn(caption, shift=UP*0.2))
        self.wait(1.2)


# ======================================================================
class LLNScene(Scene):
    """Running average of coin flips converges to 1/2."""
    def construct(self):
        head = title_card("The Law of Large Numbers",
                          "running averages settle toward the true mean")
        self.play(FadeIn(head, shift=DOWN*0.2))

        ax = Axes(x_range=[0, 300, 50], y_range=[0, 1, 0.25], x_length=10, y_length=4.5,
                  axis_config={"color": INK, "include_tip": False, "font_size": 22},
                  ).shift(DOWN*0.3)
        xlab = Text("number of flips", color=INK).scale(0.42).next_to(ax, DOWN, buff=0.2)
        ylab = Text("proportion of heads", color=INK).scale(0.42).rotate(PI/2).next_to(ax, LEFT, buff=0.1)
        mean_line = DashedLine(ax.c2p(0, 0.5), ax.c2p(300, 0.5), color=RED, stroke_width=3)
        mean_lbl = Text("true mean = 0.5", color=RED).scale(0.4).next_to(ax.c2p(300, 0.5), RIGHT, buff=0.1)
        self.play(Create(ax), FadeIn(xlab), FadeIn(ylab))
        self.play(Create(mean_line), FadeIn(mean_lbl))

        colors = [ACCENT, GREEN, GOLD, "#7a4fb0"]
        for ci, col in enumerate(colors):
            flips = (rng.random(300) < 0.5).astype(float)
            run = np.cumsum(flips) / np.arange(1, 301)
            pts = [ax.c2p(i+1, run[i]) for i in range(300)]
            path = VMobject(color=col, stroke_width=2.2)
            path.set_points_as_corners(pts)
            self.play(Create(path), run_time=1.6, rate_func=linear)
        self.wait(0.4)
        caption = Text("Many wandering paths, one inevitable limit.", color=ACCENT_D).scale(0.5)
        caption.to_edge(DOWN, buff=0.2)
        self.play(FadeIn(caption, shift=UP*0.2))
        self.wait(1.2)


# ======================================================================
class PercentilesScene(Scene):
    """Build a boxplot: quartiles, median, IQR, whiskers."""
    def construct(self):
        head = title_card("Percentiles, Quartiles & the IQR",
                          "how a boxplot summarizes a dataset")
        self.play(FadeIn(head, shift=DOWN*0.2))

        data = np.sort(np.concatenate([rng.normal(40, 8, 14), rng.normal(62, 6, 10), [92]]))
        lo, hi = 15, 100
        nl = NumberLine(x_range=[lo, hi, 10], length=11, color=INK,
                        include_numbers=True, font_size=20).shift(UP*0.4)
        self.play(Create(nl))

        dots = VGroup(*[Dot(nl.n2p(v), radius=0.05, color=ACCENT) for v in data])
        self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.05, run_time=1.2))
        sort_lbl = Text("data, sorted", color=ACCENT).scale(0.42).next_to(nl, UP, buff=0.3)
        self.play(FadeIn(sort_lbl))
        self.wait(0.3)

        def q(p):
            n = len(data); h = p*(n+1)
            if h <= 1: return data[0]
            if h >= n: return data[-1]
            lo_i = int(np.floor(h)) - 1; frac = h - np.floor(h)
            return data[lo_i] + frac*(data[lo_i+1]-data[lo_i])
        q1, med, q3 = q(0.25), q(0.5), q(0.75)
        iqr = q3 - q1
        whisk_lo = data[data >= q1 - 1.5*iqr].min()
        whisk_hi = data[data <= q3 + 1.5*iqr].max()

        ybox = -1.2
        box = Rectangle(width=nl.n2p(q3)[0]-nl.n2p(q1)[0], height=1.0,
                        fill_color=ACCENT, fill_opacity=0.13, stroke_color=ACCENT, stroke_width=2.5)
        box.move_to([(nl.n2p(q1)[0]+nl.n2p(q3)[0])/2, ybox, 0])
        med_line = Line([nl.n2p(med)[0], ybox-0.5, 0], [nl.n2p(med)[0], ybox+0.5, 0],
                        color=RED, stroke_width=4)
        wl = Line([nl.n2p(whisk_lo)[0], ybox, 0], [nl.n2p(q1)[0], ybox, 0], color=ACCENT, stroke_width=2)
        wr = Line([nl.n2p(q3)[0], ybox, 0], [nl.n2p(whisk_hi)[0], ybox, 0], color=ACCENT, stroke_width=2)
        capl = Line([nl.n2p(whisk_lo)[0], ybox-0.3, 0], [nl.n2p(whisk_lo)[0], ybox+0.3, 0], color=ACCENT, stroke_width=2)
        capr = Line([nl.n2p(whisk_hi)[0], ybox-0.3, 0], [nl.n2p(whisk_hi)[0], ybox+0.3, 0], color=ACCENT, stroke_width=2)

        def tick(val, label, color, dy=0.0):
            ln = DashedLine(nl.n2p(val), [nl.n2p(val)[0], ybox+0.6, 0], color=color, stroke_width=2)
            tx = Text(label, color=color).scale(0.36).next_to(ln, UP, buff=0.05).shift(UP*dy)
            return VGroup(ln, tx)

        t1 = tick(q1, f"Q1 = {q1:.0f}", ACCENT_D, dy=0.0)
        t2 = tick(med, f"median = {med:.0f}", RED, dy=0.6)
        t3 = tick(q3, f"Q3 = {q3:.0f}", ACCENT_D, dy=0.0)
        self.play(FadeIn(t1), FadeIn(t2), FadeIn(t3))
        self.play(Create(box), Create(med_line))
        self.play(Create(wl), Create(wr), Create(capl), Create(capr))

        brace = BraceBetweenPoints([nl.n2p(q1)[0], ybox-0.7, 0], [nl.n2p(q3)[0], ybox-0.7, 0],
                                   direction=DOWN, color=GOLD)
        iqr_lbl = Text(f"IQR = {iqr:.0f}  (middle 50%)", color=GOLD).scale(0.42)
        iqr_lbl.next_to(brace, DOWN, buff=0.1)
        self.play(GrowFromCenter(brace), FadeIn(iqr_lbl))
        self.wait(1.4)


# ======================================================================
class JointScene(Scene):
    """A bivariate cloud tilts with correlation; marginals stay put."""
    def construct(self):
        head = title_card("Joint Distributions & Correlation",
                          "correlation tilts the cloud, not the marginals")
        self.play(FadeIn(head, shift=DOWN*0.2))

        ax = Axes(x_range=[-4, 4, 2], y_range=[-4, 4, 2], x_length=5.2, y_length=5.2,
                  axis_config={"color": INK, "include_tip": False, "font_size": 22}).shift(DOWN*0.3+LEFT*0.3)
        xl = Text("X", color=INK).scale(0.5).next_to(ax.x_axis, RIGHT, buff=0.1)
        yl = Text("Y", color=INK).scale(0.5).next_to(ax.y_axis, UP, buff=0.1)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl))

        z1 = rng.normal(0, 1, 220)
        z2 = rng.normal(0, 1, 220)

        def cloud_for(rho):
            x = z1
            y = rho*z1 + np.sqrt(1-rho**2)*z2
            return VGroup(*[Dot(ax.c2p(xi, yi), radius=0.045, color=GREEN, fill_opacity=0.6)
                            for xi, yi in zip(x, y)])

        rho_tracker = ValueTracker(0.0)
        cloud = always_redraw(lambda: cloud_for(rho_tracker.get_value()))
        rho_text = always_redraw(lambda: Text(f"correlation  ρ = {rho_tracker.get_value():+.2f}",
                                              color=ACCENT_D, weight=BOLD).scale(0.5)
                                 .to_edge(RIGHT, buff=1.0).shift(UP*1.0))
        self.add(cloud, rho_text)
        self.wait(0.5)
        self.play(rho_tracker.animate.set_value(0.9), run_time=2.2)
        self.wait(0.3)
        note1 = Text("ρ → +1:  Y rises with X", color=GREEN).scale(0.42).to_edge(RIGHT, buff=0.8)
        self.play(FadeIn(note1))
        self.wait(0.4)
        self.play(rho_tracker.animate.set_value(-0.9), run_time=2.6)
        note2 = Text("ρ → −1:  Y falls as X rises", color=RED).scale(0.42).next_to(note1, DOWN, buff=0.3)
        self.play(FadeIn(note2))
        self.wait(0.4)
        self.play(rho_tracker.animate.set_value(0.0), run_time=1.8)
        note3 = Text("ρ = 0:  independent", color=ACCENT).scale(0.42).next_to(note2, DOWN, buff=0.3)
        self.play(FadeIn(note3))
        self.wait(1.3)


# ======================================================================
class LeastSquaresScene(Scene):
    """A line rotates to minimize the sum of squared residuals."""
    def construct(self):
        head = title_card("The Method of Least Squares",
                          "the line that minimizes squared residuals")
        self.play(FadeIn(head, shift=DOWN*0.2))

        ax = Axes(x_range=[0, 10, 2], y_range=[0, 14, 2], x_length=9, y_length=4.6,
                  axis_config={"color": INK, "include_tip": False, "font_size": 22}).shift(DOWN*0.3)
        self.play(Create(ax))

        xs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9.0])
        ys = 1.3*xs + 2 + rng.normal(0, 1.2, len(xs))
        dots = VGroup(*[Dot(ax.c2p(x, y), radius=0.07, color=ACCENT) for x, y in zip(xs, ys)])
        self.play(LaggedStartMap(FadeIn, dots, lag_ratio=0.1, run_time=1.0))

        # least squares solution
        b = (len(xs)*np.sum(xs*ys)-np.sum(xs)*np.sum(ys))/(len(xs)*np.sum(xs**2)-np.sum(xs)**2)
        a = np.mean(ys) - b*np.mean(xs)

        a_t = ValueTracker(8.0)      # start with a poor line
        b_t = ValueTracker(-0.2)

        def line_obj():
            aa, bb = a_t.get_value(), b_t.get_value()
            return Line(ax.c2p(0, aa), ax.c2p(10, aa+bb*10), color=GREEN, stroke_width=4)

        def residuals():
            aa, bb = a_t.get_value(), b_t.get_value()
            g = VGroup()
            for x, y in zip(xs, ys):
                yhat = aa + bb*x
                g.add(Line(ax.c2p(x, y), ax.c2p(x, yhat), color=RED, stroke_width=2.5))
            return g

        def sse_val():
            aa, bb = a_t.get_value(), b_t.get_value()
            return float(np.sum((ys-(aa+bb*xs))**2))

        line = always_redraw(line_obj)
        res = always_redraw(residuals)
        sse_text = always_redraw(lambda: Text(f"sum of squared residuals = {sse_val():.0f}",
                                              color=ACCENT_D, weight=BOLD).scale(0.5)
                                 .to_edge(DOWN, buff=0.25))
        self.play(Create(line), FadeIn(res), FadeIn(sse_text))
        self.wait(0.5)
        self.play(a_t.animate.set_value(a), b_t.animate.set_value(b),
                  run_time=3.0, rate_func=smooth)
        best = Text("the least-squares line", color=GREEN, weight=BOLD).scale(0.5)
        best.next_to(ax, UP, buff=0.05).shift(RIGHT*2)
        self.play(Flash(ax.c2p(np.mean(xs), np.mean(ys)), color=GOLD), FadeIn(best))
        self.wait(1.3)
