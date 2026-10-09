/**
 * gradcam_viewer.js
 *
 * Interaction for the Grad-CAM viewer on the retinopathy result page.
 * The page layers a transparent heat-map PNG on top of the retina photo;
 * this script controls how visible that layer is, so the user can compare
 * the heat map against the untouched image.
 *
 * Expected markup (see predict_eye.html):
 *   [data-gradcam-viewer]
 *     .gradcam-heat               the heat-map <img> layered over the photo
 *     [data-gradcam-slider]       <input type="range" min=0 max=100>
 *     [data-gradcam-value]        <output> showing the current percentage
 *     [data-gradcam-toggle]       button that shows / hides the heat map
 *
 * Moving the slider while the heat map is hidden turns it back on.
 */

(function () {
  "use strict";

  function initViewer(root) {
    const heat = root.querySelector(".gradcam-heat");
    const slider = root.querySelector("[data-gradcam-slider]");
    const output = root.querySelector("[data-gradcam-value]");
    const toggle = root.querySelector("[data-gradcam-toggle]");
    if (!heat || !slider || !toggle) return;

    let visible = true;

    function render() {
      const percent = Number(slider.value);
      heat.style.opacity = visible ? percent / 100 : 0;
      if (output) output.textContent = percent + "%";
      toggle.textContent = visible ? "Hide heatmap" : "Show heatmap";
      toggle.setAttribute("aria-pressed", String(visible));
    }

    slider.addEventListener("input", () => {
      visible = true;
      render();
    });

    toggle.addEventListener("click", () => {
      visible = !visible;
      // Showing again with the slider at 0 would look like nothing happened.
      if (visible && Number(slider.value) === 0) slider.value = 70;
      render();
    });

    render();
  }

  document.querySelectorAll("[data-gradcam-viewer]").forEach(initViewer);
})();
