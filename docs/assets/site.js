/* site.js — inject the shared nav + footer, configure MathJax.
   Each page sets <body data-page="..."> to highlight the active link. */
(function () {
  "use strict";
  const PAGES = [
    ["index", "index.html", "Home"],
    ["distributions", "distributions.html", "Distributions"],
    ["clt", "clt.html", "CLT"],
    ["lln", "lln.html", "Law of Large Numbers"],
    ["eda", "eda.html", "Percentiles & EDA"],
    ["regression", "regression.html", "Regression"],
    ["joint", "joint.html", "Joint Distributions"],
    ["animations", "animations.html", "Animations"],
  ];
  const current = document.body.getAttribute("data-page") || "index";

  const nav = document.createElement("nav");
  nav.className = "nav";
  nav.innerHTML =
    '<div class="nav-inner">' +
    '<a class="brand" href="index.html">Prob&amp;Stat <span>Interactive</span></a>' +
    '<div class="nav-links">' +
    PAGES.filter(p => p[0] !== "index").map(p =>
      '<a href="' + p[1] + '"' + (p[0] === current ? ' class="active"' : '') + '>' + p[2] + '</a>'
    ).join("") +
    '<a href="pdf/Probability_and_Statistics_Coursebook.pdf">📘 PDF</a>' +
    '</div></div>';
  document.body.insertBefore(nav, document.body.firstChild);

  const footer = document.createElement("footer");
  footer.className = "footer";
  footer.innerHTML =
    '<div class="footer-inner">' +
    '<div>Interactive companion to the <em>Probability &amp; Statistics</em> coursebook.</div>' +
    '<div>Built with Plotly.js &amp; Manim · runs entirely in your browser</div>' +
    '</div>';
  document.body.appendChild(footer);
})();
