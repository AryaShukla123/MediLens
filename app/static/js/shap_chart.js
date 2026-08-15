(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", () => {
    const dataEl = document.getElementById("shap-data");
    const canvas = document.getElementById("shap-chart");

    if (!dataEl || !canvas || typeof Chart === "undefined") {
      return;
    }

    let shapData;
    try {
      shapData = JSON.parse(dataEl.textContent);
    } catch (err) {
      console.error("Could not parse SHAP data:", err);
      return;
    }

    const features = shapData.top_features || [];
    if (features.length === 0) {
      return;
    }

    // Order smallest-impact-first so the bars read top-to-bottom as
    // most-important-first once Chart.js's horizontal bar draws them.
    const ordered = [...features].reverse();

    const labels = ordered.map((f) => f.label || f.feature);
    const values = ordered.map((f) => f.shap_value);
    const inputValues = ordered.map((f) => f.input_value);

    const RISK_UP = "#C0392B";
    const RISK_DOWN = "#2A8C82";

    const barColors = values.map((v) => (v > 0 ? RISK_UP : RISK_DOWN));

    new Chart(canvas.getContext("2d"), {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            data: values,
            backgroundColor: barColors,
            borderRadius: 4,
            barThickness: 22,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#13273F",
            padding: 10,
            titleFont: { family: "'Plus Jakarta Sans', sans-serif", weight: "600" },
            bodyFont: { family: "'Segoe UI', sans-serif" },
            callbacks: {
              label: function (context) {
                const idx = context.dataIndex;
                const impact = values[idx] > 0 ? "increases risk" : "decreases risk";
                const inputVal = inputValues[idx];
                const lines = [`Impact: ${impact} (${values[idx].toFixed(3)})`];
                if (inputVal !== null && inputVal !== undefined) {
                  lines.push(`Your value: ${inputVal}`);
                }
                return lines;
              },
            },
          },
        },
        scales: {
          x: {
            grid: { color: "#DFE3E6" },
            ticks: { color: "#5B6B7C", font: { family: "'Segoe UI', sans-serif" } },
            title: {
              display: true,
              text: "SHAP value (impact on prediction)",
              color: "#5B6B7C",
              font: { size: 12 },
            },
          },
          y: {
            grid: { display: false },
            ticks: { color: "#1C2B3A", font: { family: "'Segoe UI', sans-serif", size: 13 } },
          },
        },
      },
    });
  });
})();
