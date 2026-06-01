#!/usr/bin/env python
"""Generate all figures for the Probability & Statistics coursebook."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
import os

rcParams.update({
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "font.family": "serif",
})
ACC = "#1f5fa8"      # primary blue
ACC2 = "#b03030"     # red
ACC3 = "#2a8f5a"     # green
GREY = "#666666"

OUT = "/Users/kylekabasares/Downloads/Math 32/build/figures"
os.makedirs(OUT, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)

# ----------------------------------------------------------------------
# Synthetic "Old Faithful" eruption durations (seconds) -- illustrative,
# constructed to match the summary stats quoted in the notes
# (min 96, max 306, median 240, mean ~209, two modes near 120 and 270).
rng = np.random.default_rng(7)
g1 = rng.normal(118, 14, 97)
g2 = rng.normal(272, 24, 175)
faithful = np.concatenate([g1, g2])
faithful = np.clip(faithful, 96, 306)
# nudge to hit endpoints
faithful[np.argmin(faithful)] = 96
faithful[np.argmax(faithful)] = 306

# ----------------------------------------------------------------------
# 1. Histograms with different bin widths
fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
for ax, bw, t in zip(axes, [2, 30, 90], ["b = 2 (too small)", "b = 30 (good)", "b = 90 (too large)"]):
    bins = np.arange(96, 306 + bw, bw)
    ax.hist(faithful, bins=bins, density=True, color=ACC, edgecolor="white", linewidth=0.4)
    ax.set_title(t, fontsize=10)
    ax.set_xlabel("duration (s)")
axes[0].set_ylabel("density")
fig.suptitle("Histograms of the Old Faithful durations for three bin widths", fontsize=11)
save(fig, "hist_binwidth")

# 2. KDE of faithful
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(figsize=(5.2, 3.4))
xs = np.linspace(60, 330, 400)
kde = gaussian_kde(faithful, bw_method=0.30)
ax.hist(faithful, bins=np.arange(96, 312, 12), density=True, color="#cdddf0", edgecolor="white")
ax.plot(xs, kde(xs), color=ACC2, lw=2)
ax.set_xlabel("duration (s)"); ax.set_ylabel("density")
ax.set_title("Kernel density estimate of the Old Faithful data")
save(fig, "kde_faithful")

# 3. Kernel shapes
fig, ax = plt.subplots(figsize=(5.6, 3.4))
u = np.linspace(-1.4, 1.4, 400)
def epan(u): return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0)
def triw(u): return np.where(np.abs(u) <= 1, 35/32 * (1 - u**2)**3, 0)
def tri(u): return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0)
def rect(u): return np.where(np.abs(u) <= 1, 0.5, 0)
ax.plot(u, epan(u), color=ACC, lw=2, label="Epanechnikov")
ax.plot(u, triw(u), color=ACC2, lw=2, label="triweight")
ax.plot(u, tri(u), color=ACC3, lw=2, label="triangular")
ax.plot(u, rect(u), color=GREY, lw=2, label="rectangular")
ax.plot(u, stats.norm.pdf(u), color="#7a4fb0", lw=2, ls="--", label="normal")
ax.legend(fontsize=8.5, frameon=False); ax.set_xlabel("u"); ax.set_ylabel("K(u)")
ax.set_title("Several well-known kernels")
save(fig, "kernels")

# 4. KDE bandwidth comparison
fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharey=False)
for ax, bm, t in zip(axes, [0.06, 0.30, 1.4], ["h small (undersmoothed)", "h moderate", "h large (oversmoothed)"]):
    kde = gaussian_kde(faithful, bw_method=bm)
    ax.plot(xs, kde(xs), color=ACC, lw=1.8)
    ax.fill_between(xs, kde(xs), color="#cdddf0", alpha=0.6)
    ax.set_title(t, fontsize=10); ax.set_xlabel("duration (s)")
axes[0].set_ylabel("density")
fig.suptitle("Effect of the bandwidth h on the kernel density estimate", fontsize=11)
save(fig, "kde_bandwidth")

# 5. ECDF toy dataset {1,3,4,7,9}
def ecdf_plot(ax, data, color=ACC):
    d = np.sort(data); n = len(d)
    F = np.arange(1, n+1)/n
    # horizontal segments
    edges = np.concatenate([[d[0]-1], d, [d[-1]+1]])
    levels = np.concatenate([[0.0], F])
    for i in range(len(levels)):
        ax.plot([edges[i], edges[i+1]], [levels[i], levels[i]], color=color, lw=1.8)
    # closed dots at jump points
    ax.plot(d, F, "o", color=color, ms=5, zorder=4)
    ax.plot(d, levels[:-1], "o", mfc="white", mec=color, ms=5, zorder=4)
fig, ax = plt.subplots(figsize=(5.0, 3.2))
ecdf_plot(ax, [4,3,9,1,7])
ax.set_xlabel("x"); ax.set_ylabel(r"$F_n(x)$")
ax.set_title("Empirical distribution function of {1, 3, 4, 7, 9}")
ax.set_ylim(-0.05, 1.05)
save(fig, "ecdf_toy")

# 6. ECDF faithful
fig, ax = plt.subplots(figsize=(5.0, 3.2))
d = np.sort(faithful); n = len(d)
ax.step(d, np.arange(1, n+1)/n, where="post", color=ACC, lw=1.4)
ax.set_xlabel("duration (s)"); ax.set_ylabel(r"$F_n(x)$")
ax.set_title("Empirical distribution function of the Old Faithful data")
save(fig, "ecdf_faithful")

# 7. Annotated boxplot + faithful boxplot
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
# annotated schematic
ax = axes[0]
data = np.concatenate([rng.normal(50, 10, 200), [88, 92, 12]])
bp = ax.boxplot(data, vert=True, widths=0.5, patch_artist=True,
                boxprops=dict(facecolor="#cdddf0", color=ACC),
                medianprops=dict(color=ACC2, lw=2),
                whiskerprops=dict(color=ACC), capprops=dict(color=ACC),
                flierprops=dict(marker="o", markerfacecolor=ACC2, markersize=4, markeredgecolor=ACC2))
q1, med, q3 = np.percentile(data, [25, 50, 75])
iqr = q3 - q1
ax.annotate("median", xy=(1.27, med), fontsize=8.5, va="center")
ax.annotate("lower quartile", xy=(1.27, q1), fontsize=8.5, va="center")
ax.annotate("upper quartile", xy=(1.27, q3), fontsize=8.5, va="center")
ax.annotate("outliers", xy=(1.05, 90), fontsize=8.5, va="center", color=ACC2)
ax.annotate("whisker\n(1.5·IQR)", xy=(0.62, q3 + 1.0*iqr), fontsize=8, va="center")
ax.set_xticks([]); ax.set_title("Anatomy of a boxplot", fontsize=10)
ax = axes[1]
ax.boxplot(faithful, vert=True, widths=0.5, patch_artist=True,
           boxprops=dict(facecolor="#cdddf0", color=ACC),
           medianprops=dict(color=ACC2, lw=2),
           whiskerprops=dict(color=ACC), capprops=dict(color=ACC))
ax.set_xticks([]); ax.set_ylabel("duration (s)")
ax.set_title("Boxplot of the Old Faithful data", fontsize=10)
save(fig, "boxplot")

# ----------------------------------------------------------------------
# Black cherry tree data (from Worksheet 10): diameter (m), height (m), volume (m^3)
diam = np.array([0.21,0.22,0.22,0.27,0.27,0.27,0.28,0.28,0.28,0.29,0.29,0.29,
0.30,0.30,0.33,0.33,0.34,0.35,0.35,0.36,0.36,0.37,0.41,0.41,0.44,0.44,0.45,
0.46,0.46,0.52,0.30])
# careful: the table has 31 rows; rebuild precisely
diam = np.array([0.21,0.22,0.22,0.27,0.27,0.27,0.28,0.28,0.28,0.29,0.29,0.29,0.30,
0.30,0.33,0.33,0.34,0.35,0.35,0.36,0.36,0.37,0.41,0.41,0.44,0.44,0.45,0.46,0.46,0.52,0.30])
height = np.array([21.3,19.8,19.2,21.9,24.7,25.3,20.1,22.9,24.4,22.9,23.2,23.2,21.0,
22.9,22.6,25.9,26.2,21.6,19.5,23.8,24.4,22.6,21.9,23.5,24.7,25.0,24.4,24.4,24.4,26.5,21.0])
volume = np.array([0.29,0.29,0.29,0.46,0.53,0.56,0.44,0.52,0.64,0.56,0.59,0.61,0.60,
0.54,0.63,0.96,0.78,0.73,0.71,0.98,0.98,1.03,1.08,1.21,1.57,1.58,1.65,1.46,1.44,2.18,0.60])
# trim to 31
diam=diam[:31]; height=height[:31]; volume=volume[:31]
xch = diam**2 * height
ych = volume
beta_ch = np.sum(xch*ych)/np.sum(xch**2)

# 8/16. cherry scatter + regression through origin
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.scatter(xch, ych, color=ACC, s=26, zorder=3, edgecolor="white", linewidth=0.4)
xx = np.linspace(0, xch.max()*1.02, 50)
ax.plot(xx, beta_ch*xx, color=ACC2, lw=1.8, label=fr"$y={beta_ch:.3f}\,x$")
ax.set_xlabel(r"$x=d^2h$"); ax.set_ylabel("volume  y")
ax.set_title("Black cherry tree data with no-intercept least-squares line")
ax.legend(frameon=False)
save(fig, "cherry_regression")

# 17. residuals cherry (good - random scatter)
res_ch = ych - beta_ch*xch
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.axhline(0, color=GREY, lw=1)
ax.scatter(xch, res_ch, color=ACC, s=26, edgecolor="white", linewidth=0.4)
ax.set_xlabel(r"$x=d^2h$"); ax.set_ylabel("residual  r")
ax.set_title("Residuals show no pattern: the linear model fits well")
save(fig, "residuals_good")

# ----------------------------------------------------------------------
# Synthetic timber data (illustrative): density vs Janka hardness
# regression line from notes: hardness = -1160.5 + 57.51*density
dens = np.linspace(24.7, 69.1, 36)
true = -1160.5 + 57.51*dens
noise = rng.normal(0, 1, 36) * (40 + 6.0*(dens-24.7))   # heteroscedastic fan-out
hard = np.clip(true + noise + 30*np.sin((dens-24)/12), 200, None)

# 18. residuals parabolic / fan-out (linear fit to timber)
b1, b0 = np.polyfit(dens, hard, 1)
res_t = hard - (b0 + b1*dens)
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
ax = axes[0]
ax.axhline(0, color=GREY, lw=1)
ax.scatter(dens, res_t, color=ACC, s=24, edgecolor="white", linewidth=0.4)
xx = np.linspace(dens.min(), dens.max(), 100)
pp = np.polyfit(dens, res_t, 2)
ax.plot(xx, np.polyval(pp, xx), color=ACC2, lw=1.4, ls="--")
ax.set_title("Linear model: residuals bend (parabolic)", fontsize=10)
ax.set_xlabel("density"); ax.set_ylabel("residual")
# quadratic fit residuals -> fan out
cc = np.polyfit(dens, hard, 2)
res_q = hard - np.polyval(cc, dens)
ax = axes[1]
ax.axhline(0, color=GREY, lw=1)
ax.scatter(dens, res_q, color=ACC, s=24, edgecolor="white", linewidth=0.4)
ax.set_title("Quadratic model: residuals fan out", fontsize=10)
ax.set_xlabel("density"); ax.set_ylabel("residual")
save(fig, "residuals_timber")

# 8b. timber scatter with quadratic
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.scatter(dens, hard, color=ACC, s=26, edgecolor="white", linewidth=0.4, zorder=3)
ax.plot(xx, np.polyval(cc, xx), color=ACC2, lw=1.8, label="quadratic fit")
ax.set_xlabel("wood density"); ax.set_ylabel("Janka hardness")
ax.set_title("Janka hardness versus density (illustrative timber data)")
ax.legend(frameon=False)
save(fig, "timber_scatter")

# ----------------------------------------------------------------------
# Mortality data (full 61 points from the Least Squares lecture table)
rate = [1.247,1.466,1.299,1.359,1.392,1.307,1.254,1.318,1.260,1.096,1.402,1.309,
1.259,1.175,1.486,1.456,1.236,1.369,1.257,1.527,1.627,1.486,1.485,1.519,1.581,
1.625,1.668,1.800,1.609,1.558,1.807,1.637,1.755,1.491,1.555,1.428,1.723,1.379,
1.742,1.574,1.569,1.591,1.772,1.828,1.704,1.702,1.427,1.724,1.696,1.711,1.444,
1.591,1.987,1.495,1.587,1.713,1.557,1.640,1.709,1.625,1.378]
calcium = [105,5,78,84,73,78,96,122,21,138,37,59,133,107,5,90,101,68,50,60,53,122,
81,21,14,13,17,14,18,10,15,10,12,20,39,39,44,94,8,9,91,16,15,8,26,44,27,6,6,13,14,
49,8,14,75,71,13,57,71,20,71]
rate = np.array(rate); calcium = np.array(calcium)
bm, am = np.polyfit(calcium, rate, 1)
fig, ax = plt.subplots(figsize=(5.4, 3.6))
ax.scatter(calcium, rate, color=ACC, s=24, edgecolor="white", linewidth=0.4, zorder=3)
xx = np.linspace(0, 140, 50)
ax.plot(xx, am + bm*xx, color=ACC2, lw=1.8)
ax.set_xlabel("calcium concentration (ppm)"); ax.set_ylabel("mortality rate (% deaths)")
ax.set_title("Mortality rate versus calcium in drinking water (61 towns)")
save(fig, "mortality_scatter")

# ----------------------------------------------------------------------
# Toy regression {(1,2),(3,1.8),(5,1)}
fig, ax = plt.subplots(figsize=(4.6, 3.4))
tx = np.array([1,3,5]); ty = np.array([2,1.8,1])
ax.scatter(tx, ty, color=ACC, s=45, zorder=3)
xx = np.linspace(0, 6, 20)
ax.plot(xx, 2.35 - 0.25*xx, color=ACC2, lw=1.8, label=r"$y=2.35-0.25x$")
for xi, yi in zip(tx, ty):
    ax.plot([xi, xi], [yi, 2.35-0.25*xi], color=GREY, ls=":", lw=1)
ax.set_xlim(0,6); ax.set_ylim(0,3); ax.legend(frameon=False)
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title("Least-squares line for a three-point dataset")
save(fig, "toy_regression")

# ----------------------------------------------------------------------
# t distributions
fig, ax = plt.subplots(figsize=(5.6, 3.4))
xx = np.linspace(-4, 4, 400)
for df, c in zip([1,5,10], [ACC3, ACC, ACC2]):
    ax.plot(xx, stats.t.pdf(xx, df), color=c, lw=1.8, label=f"t, df={df}")
ax.plot(xx, stats.norm.pdf(xx), color="black", lw=1.4, ls="--", label="N(0,1)")
ax.legend(frameon=False, fontsize=9); ax.set_xlabel("t"); ax.set_ylabel("density")
ax.set_title("The t density approaches N(0,1) as df grows")
save(fig, "tdist")

# t vs normal thick tails
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot(xx, stats.norm.pdf(xx), color="black", lw=1.6, label="N(0,1)")
ax.plot(xx, stats.t.pdf(xx, 5), color=ACC2, lw=1.6, label="t, df=5")
ax.fill_between(xx, stats.t.pdf(xx,5), stats.norm.pdf(xx),
                where=np.abs(xx)>1.6, color="#f0c9c9", alpha=0.7)
ax.legend(frameon=False); ax.set_xlabel("t"); ax.set_ylabel("density")
ax.set_title("The t density has thicker tails than the normal")
save(fig, "tdist_tails")

# normal CI shading
fig, ax = plt.subplots(figsize=(5.6, 3.2))
xx = np.linspace(-4,4,400)
ax.plot(xx, stats.norm.pdf(xx), color=ACC, lw=1.8)
mask = (xx>=-1.96)&(xx<=1.96)
ax.fill_between(xx[mask], stats.norm.pdf(xx[mask]), color="#cdddf0")
ax.axvline(-1.96, color=ACC2, lw=1); ax.axvline(1.96, color=ACC2, lw=1)
ax.text(0, 0.15, r"$1-\alpha$", ha="center", fontsize=13)
ax.text(-1.96, -0.03, r"$-z_{\alpha/2}$", ha="center", color=ACC2)
ax.text(1.96, -0.03, r"$z_{\alpha/2}$", ha="center", color=ACC2)
ax.text(2.7, 0.02, r"$\alpha/2$", color=GREY); ax.text(-3.1, 0.02, r"$\alpha/2$", color=GREY)
ax.set_xlabel("z"); ax.set_ylabel("density"); ax.set_ylim(-0.05, 0.45)
ax.set_title(r"$P(-z_{\alpha/2}<Z<z_{\alpha/2})=1-\alpha$")
save(fig, "normal_ci")

# CI coverage illustration
fig, ax = plt.subplots(figsize=(5.2, 3.8))
theta = 0.0
ax.axvline(theta, color=ACC2, lw=1.5, label=r"true $\theta$")
np.random.seed(3)
n = 25
centers = np.random.normal(0, 1, n)
half = 1.96
for i,(c) in enumerate(centers):
    lo, hi = c-half, c+half
    covers = lo <= theta <= hi
    ax.plot([lo, hi], [i, i], color=(ACC if covers else ACC2), lw=2,
            alpha=0.9)
    ax.plot(c, i, "o", ms=2.5, color=(ACC if covers else ACC2))
ax.set_yticks([]); ax.set_xlabel("value")
ax.set_title("Each interval either traps $\\theta$ or it does not;\nabout 95% do")
ax.legend(frameon=False, loc="upper right", fontsize=9)
save(fig, "ci_coverage")

# ----------------------------------------------------------------------
# Likelihood for U(0,theta), data 0.98,1.57,0.31
fig, ax = plt.subplots(figsize=(5.2, 3.2))
th = np.linspace(0.01, 4, 500)
m = 1.57
L = np.where(th >= m, 1/th**3, 0)
ax.plot(th, L, color=ACC, lw=2)
ax.axvline(m, color=ACC2, ls="--", lw=1)
ax.text(m+0.05, L.max()*0.8, r"max at $\hat\theta=\max x_i=1.57$", color=ACC2, fontsize=9)
ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$L(\theta)$")
ax.set_title(r"Likelihood for a $U(0,\theta)$ sample")
save(fig, "likelihood_uniform")

# Likelihood and loglik for smokers p^93 (1-p)^322
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
p = np.linspace(0.001, 0.999, 500)
logL = 93*np.log(p) + 322*np.log(1-p)
L = np.exp(logL - logL.max())   # scaled
ax = axes[0]
ax.plot(p, L, color=ACC, lw=2)
ax.axvline(93/415, color=ACC2, ls="--", lw=1)
ax.set_xlim(0,0.6); ax.set_xlabel("p"); ax.set_ylabel(r"$L(p)$ (scaled)")
ax.set_title(r"Likelihood $L(p)\propto p^{93}(1-p)^{322}$", fontsize=10)
ax.text(93/415+0.01, 0.5, r"$\hat p=\frac{93}{415}=0.224$", color=ACC2, fontsize=9)
ax = axes[1]
ax.plot(p, logL, color=ACC3, lw=2)
ax.axvline(93/415, color=ACC2, ls="--", lw=1)
ax.set_xlim(0,0.6); ax.set_xlabel("p"); ax.set_ylabel(r"$\ell(p)$")
ax.set_title("Loglikelihood (same maximizer)", fontsize=10)
save(fig, "likelihood_smokers")

# ----------------------------------------------------------------------
# Sum of squares paraboloid
from mpl_toolkits.mplot3d import Axes3D  # noqa
fig = plt.figure(figsize=(5.4, 4.2))
ax = fig.add_subplot(111, projection="3d")
A = np.linspace(-3, 3, 60); B = np.linspace(-3, 3, 60)
AA, BB = np.meshgrid(A, B)
SS = 2*AA**2 + 1.2*BB**2 + 1.0*AA*BB + 4
ax.plot_surface(AA, BB, SS, cmap="viridis", alpha=0.9, linewidth=0, antialiased=True)
ax.set_xlabel(r"$\alpha$"); ax.set_ylabel(r"$\beta$"); ax.set_zlabel(r"$S(\alpha,\beta)$")
ax.set_title(r"The sum of squares $S(\alpha,\beta)$ is a bowl")
ax.view_init(elev=22, azim=-60)
save(fig, "ss_paraboloid")

# R^2 illustration
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
np.random.seed(11)
xr = np.linspace(1, 9, 9)
yr = 0.7*xr + 1 + np.random.normal(0, 0.8, 9)
ybar = yr.mean()
slope, inter = np.polyfit(xr, yr, 1)
yhat = inter + slope*xr
ax = axes[0]
ax.scatter(xr, yr, color=ACC, zorder=3)
ax.axhline(ybar, color=GREY, lw=1.5)
for xi, yi in zip(xr, yr):
    s = abs(yi - ybar)
    ax.add_patch(plt.Rectangle((xi, min(yi,ybar)), s, s, color="#e7b6b6", alpha=0.6, zorder=1))
ax.set_title(r"Deviations from the mean $\bar y$  ($SS_{tot}$)", fontsize=10)
ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
ax = axes[1]
ax.scatter(xr, yr, color=ACC, zorder=3)
ax.plot(xr, yhat, color=ACC2, lw=1.6)
for xi, yi, yh in zip(xr, yr, yhat):
    s = abs(yi - yh)
    ax.add_patch(plt.Rectangle((xi, min(yi,yh)), s, s, color="#aac4e6", alpha=0.7, zorder=1))
ax.set_title(r"Deviations from the line  ($SS_{res}$)", fontsize=10)
ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
fig.suptitle(r"$R^2 = 1 - SS_{res}/SS_{tot}$: smaller blue squares $\Rightarrow$ better fit", fontsize=11)
save(fig, "r_squared")

# ----------------------------------------------------------------------
# Chi-square densities
fig, ax = plt.subplots(figsize=(5.6, 3.4))
xx = np.linspace(0, 16, 400)
for df, c in zip([1,2,3,5,8], [ACC2,"#d2691e",ACC3,ACC,"#7a4fb0"]):
    ax.plot(xx, stats.chi2.pdf(xx, df), color=c, lw=1.7, label=f"df={df}")
ax.set_ylim(0, 0.5); ax.legend(frameon=False, fontsize=9)
ax.set_xlabel("x"); ax.set_ylabel("density")
ax.set_title(r"$\chi^2$ densities for several degrees of freedom")
save(fig, "chisq")

# Mendel observed vs expected bar chart
fig, ax = plt.subplots(figsize=(5.6, 3.4))
cats = ["round\nyellow", "wrinkled\nyellow", "round\ngreen", "wrinkled\ngreen"]
obs = [315, 101, 108, 32]
exp = [556*9/16, 556*3/16, 556*3/16, 556*1/16]
x = np.arange(4); w = 0.38
ax.bar(x-w/2, obs, w, color=ACC, label="observed")
ax.bar(x+w/2, exp, w, color="#cdddf0", edgecolor=ACC, label="expected (9:3:3:1)")
ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8.5)
ax.legend(frameon=False); ax.set_ylabel("count")
ax.set_title("Mendel's peas: observed counts vs. the 9:3:3:1 prediction")
save(fig, "mendel")

print("ALL FIGURES DONE")
