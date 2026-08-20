/*
 * Client-side renderer for the Validate tab.
 *
 * Gradio's Plot component is display-only in the version used by this app.
 * This dependency-free SVG renderer therefore keeps threshold interactions
 * local to the browser: dragging or clicking the ROC operating point updates
 * the confusion matrix and all threshold-dependent metrics immediately.
 */

(() => {
    const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
    })[char]);

    const clamp = (value, lower = 0, upper = 1) => Math.max(lower, Math.min(upper, value));

    const percent = (value) => (
        Number.isFinite(value) ? (value * 100).toFixed(1) + "%" : "—"
    );

    const number = (value, digits = 3) => (
        Number.isFinite(value) ? Number(value).toFixed(digits) : "—"
    );

    const metricAtThreshold = (data, threshold) => {
        const operating = data.operating;
        const thresholds = operating.thresholds;
        let lower = 0;
        let upper = thresholds.length - 1;
        let selected = 0;

        // Each cached point represents predictions where score >= point's
        // threshold. Select the smallest cached threshold at or above the
        // requested value, which gives the matching score interval. In
        // particular, a score exactly equal to a slider value is positive.
        while (lower <= upper) {
            const middle = Math.floor((lower + upper) / 2);
            if (thresholds[middle] >= threshold) {
                selected = middle;
                upper = middle - 1;
            } else {
                lower = middle + 1;
            }
        }

        const tp = operating.tp[selected];
        const fp = operating.fp[selected];
        const tn = data.summary.negative_count - fp;
        const fn = data.summary.positive_count - tp;
        const divide = (numerator, denominator) => (denominator ? numerator / denominator : null);
        return {
            threshold,
            tp,
            fp,
            tn,
            fn,
            sensitivity: divide(tp, tp + fn),
            specificity: divide(tn, tn + fp),
            ppv: divide(tp, tp + fp),
            npv: divide(tn, tn + fn),
            f1: divide(2 * tp, 2 * tp + fp + fn),
        };
    };

    const svgPath = (points, xKey, yKey, x, y) => points
        .map((point, index) => (index ? "L" : "M") + x(point[xKey]).toFixed(2) + " " + y(point[yKey]).toFixed(2))
        .join(" ");

    const renderDashboard = (root, data) => {
        const viewWidth = 620;
        const viewHeight = 390;
        const left = 58;
        const right = 24;
        const top = 34;
        const bottom = 50;
        const plotWidth = viewWidth - left - right;
        const plotHeight = viewHeight - top - bottom;
        const x = (value) => left + clamp(value) * plotWidth;
        const y = (value) => top + (1 - clamp(value)) * plotHeight;
        const rocPath = svgPath(data.roc, "fpr", "tpr", x, y);
        const prPath = svgPath(data.pr, "recall", "precision", x, y);
        const calibrationPath = svgPath(
            data.calibration,
            "mean_predicted",
            "fraction_positive",
            x,
            y,
        );
        // Keep the plots deliberately quiet: visible axes and ticks make the
        // scale clear without a full grid competing with the curves.
        const tickLabel = (tick) => (tick === 0 || tick === 1 ? String(tick) : tick.toFixed(2).replace(/^0/, ""));
        const axisElements = [
            '<line class="ff-v-axis-spine" x1="' + left + '" y1="' + (top + plotHeight) + '" x2="' + (left + plotWidth) + '" y2="' + (top + plotHeight) + '"></line>',
            '<line class="ff-v-axis-spine" x1="' + left + '" y1="' + top + '" x2="' + left + '" y2="' + (top + plotHeight) + '"></line>',
            [0, 0.25, 0.5, 0.75, 1].map((tick) => (
                '<line class="ff-v-axis-tick" x1="' + x(tick) + '" y1="' + (top + plotHeight) + '" x2="' + x(tick) + '" y2="' + (top + plotHeight + 5) + '"></line>' +
                '<line class="ff-v-axis-tick" x1="' + (left - 5) + '" y1="' + y(tick) + '" x2="' + left + '" y2="' + y(tick) + '"></line>' +
                '<text class="ff-v-axis-text" x="' + x(tick) + '" y="' + (top + plotHeight + 24) + '" text-anchor="middle">' + tickLabel(tick) + '</text>' +
                '<text class="ff-v-axis-text" x="' + (left - 10) + '" y="' + (y(tick) + 4) + '" text-anchor="end">' + tickLabel(tick) + '</text>'
            )).join(""),
        ].join("");
        const calibrationDots = data.calibration.map((point) => (
            '<circle class="ff-v-calibration-point" cx="' + x(point.mean_predicted) + '" cy="' + y(point.fraction_positive) +
            '" r="' + Math.max(4, Math.min(10, 3 + Math.sqrt(point.count))) + '">' +
            '<title>n = ' + point.count + ', predicted = ' + number(point.mean_predicted) +
            ', observed = ' + number(point.fraction_positive) + '</title></circle>'
        )).join("");
        const metricCards = [
            ["sensitivity", "Sens"],
            ["specificity", "Spec"],
            ["ppv", "PPV"],
            ["npv", "NPV"],
            ["f1", "F1"],
        ].map((entry) => (
            '<div class="ff-v-metric-card">' +
            '<div class="ff-v-metric-label">' + entry[1] + '</div>' +
            '<div class="ff-v-metric-value" data-metric="' + entry[0] + '">—</div>' +
            '</div>'
        )).join("");

        const summary = data.summary;
        const excludedText = summary.excluded_rows
            ? " · " + summary.excluded_rows.toLocaleString() + " excluded"
            : "";

        root.innerHTML = [
            '<section class="ff-validation-results">',
            '<div class="ff-v-result-header">',
            '<div>',
            '<p class="ff-section-title">Performance Summary</p>',
            '<h2>Validation results</h2>',
            '<p class="ff-v-subtitle">Score: <strong>', esc(data.score_column),
            '</strong> · Ground truth: <strong>', esc(data.label_column), '</strong></p>',
            '</div>',
            '<div class="ff-v-cohort-summary">',
            '<strong>', summary.included_rows.toLocaleString(), '</strong> evaluable rows',
            ' · ', percent(summary.prevalence), ' positive', excludedText,
            '</div>',
            '</div>',
            '<div class="ff-v-metrics">', metricCards, '</div>',
            '<div class="ff-v-chart-grid">',
            '<article class="ff-v-chart-card ff-v-roc-card">',
            '<div class="ff-v-chart-heading">',
            '<div><h3>ROC curve</h3><p>False-positive rate vs. sensitivity</p></div>',
            '<span class="ff-v-auc-badge">ROC AUC ', number(data.auc.roc), '</span>',
            '</div>',
            '<div class="ff-v-roc-plot">',
            '<svg class="ff-v-chart ff-v-roc-svg" viewBox="0 0 ', viewWidth, ' ', viewHeight,
            '" role="img" aria-label="Interactive ROC curve. Drag the marker to select a threshold.">',
            axisElements,
            '<line class="ff-v-reference-line" x1="', x(0), '" y1="', y(0), '" x2="', x(1), '" y2="', y(1), '"></line>',
            '<path class="ff-v-roc-line" d="', rocPath, '"></path>',
            '<circle class="ff-v-roc-marker" data-roc-marker r="7"></circle>',
            '<text class="ff-v-axis-title" x="', left + plotWidth / 2, '" y="', viewHeight - 8,
            '" text-anchor="middle">False-positive rate</text>',
            '<text class="ff-v-axis-title" x="16" y="', top + plotHeight / 2,
            '" text-anchor="middle" transform="rotate(-90 16 ', top + plotHeight / 2, ')">Sensitivity</text>',
            '</svg>',
            '<div class="ff-v-inset" aria-label="Binary classification table">',
            '<table><thead><tr><th></th><th colspan="2">Predicted</th></tr>',
            '<tr><th>Truth</th><th>Real</th><th>Contam.</th></tr></thead>',
            '<tbody><tr><th>Real</th><td data-count="tn">—</td><td data-count="fp">—</td></tr>',
            '<tr><th>Contam.</th><td data-count="fn">—</td><td data-count="tp">—</td></tr></tbody></table>',
            '</div>',
            '</div>',
            '<div class="ff-v-threshold-control">',
            '<div><label for="ff-v-threshold-range">Decision threshold <output data-threshold>—</output></label>',
            '<span>Positive when ', esc(data.threshold_rule), '</span></div>',
            '<input id="ff-v-threshold-range" data-threshold-range type="range" min="0" max="1" step="0.001">',
            '</div>',
            '<p class="ff-v-drag-note">Drag or click the blue point on the ROC curve to change the threshold.</p>',
            '</article>',
            '<article class="ff-v-chart-card">',
            '<div class="ff-v-chart-heading">',
            '<div><h3>Precision-recall curve</h3><p>Recall vs. positive predictive value</p></div>',
            '<span class="ff-v-auc-badge">PR AUC (AP) ', number(data.auc.pr), '</span>',
            '</div>',
            '<svg class="ff-v-chart" viewBox="0 0 ', viewWidth, ' ', viewHeight,
            '" role="img" aria-label="Precision-recall curve">',
            axisElements,
            '<line class="ff-v-reference-line" x1="', x(0), '" y1="', y(summary.prevalence),
            '" x2="', x(1), '" y2="', y(summary.prevalence), '"></line>',
            '<path class="ff-v-pr-line" d="', prPath, '"></path>',
            '<circle class="ff-v-pr-marker" data-pr-marker r="7"></circle>',
            '<text class="ff-v-axis-title" x="', left + plotWidth / 2, '" y="', viewHeight - 8,
            '" text-anchor="middle">Recall (sensitivity)</text>',
            '<text class="ff-v-axis-title" x="16" y="', top + plotHeight / 2,
            '" text-anchor="middle" transform="rotate(-90 16 ', top + plotHeight / 2, ')">Precision (PPV)</text>',
            '</svg>',
            '<p class="ff-v-drag-note">The point mirrors the selected ROC operating threshold.</p>',
            '</article>',
            '</div>',
            '<article class="ff-v-chart-card ff-v-calibration-card">',
            '<div class="ff-v-chart-heading">',
            '<div><h3>Calibration</h3><p>Observed contamination rate by predicted output</p></div>',
            '<span class="ff-v-legend"><i></i> Model <b></b> Perfect calibration</span>',
            '</div>',
            '<svg class="ff-v-chart" viewBox="0 0 ', viewWidth, ' ', viewHeight,
            '" role="img" aria-label="Calibration plot">',
            axisElements,
            '<line class="ff-v-reference-line" x1="', x(0), '" y1="', y(0), '" x2="', x(1), '" y2="', y(1), '"></line>',
            data.calibration.length ? '<path class="ff-v-calibration-line" d="' + calibrationPath + '"></path>' : "",
            calibrationDots,
            '<text class="ff-v-axis-title" x="', left + plotWidth / 2, '" y="', viewHeight - 8,
            '" text-anchor="middle">Mean predicted output</text>',
            '<text class="ff-v-axis-title" x="16" y="', top + plotHeight / 2,
            '" text-anchor="middle" transform="rotate(-90 16 ', top + plotHeight / 2, ')">Observed fraction contaminated</text>',
            '</svg>',
            '</article>',
            '</section>',
        ].join("");

        const rocMarker = root.querySelector("[data-roc-marker]");
        const prMarker = root.querySelector("[data-pr-marker]");
        const thresholdSlider = root.querySelector("[data-threshold-range]");
        const rocSvg = root.querySelector(".ff-v-roc-svg");
        let threshold = clamp(Number(data.threshold));
        // The upload/settings callbacks generate the initial default report.
        // Remember that threshold so a no-op click does not create another
        // report, and only notify Gradio after a committed user interaction.
        let lastReportedThreshold = threshold;

        const updateThreshold = (nextThreshold) => {
            threshold = clamp(Number(nextThreshold));
            const metrics = metricAtThreshold(data, threshold);
            root.querySelectorAll("[data-metric]").forEach((node) => {
                node.textContent = percent(metrics[node.dataset.metric]);
            });
            root.querySelectorAll("[data-count]").forEach((node) => {
                node.textContent = metrics[node.dataset.count].toLocaleString();
            });
            root.querySelectorAll("[data-threshold]").forEach((node) => {
                node.textContent = number(threshold);
            });
            thresholdSlider.value = threshold.toFixed(3);

            rocMarker.setAttribute("cx", x(1 - metrics.specificity).toFixed(2));
            rocMarker.setAttribute("cy", y(metrics.sensitivity).toFixed(2));
            if (metrics.ppv === null) {
                prMarker.setAttribute("opacity", "0");
            } else {
                prMarker.setAttribute("opacity", "1");
                prMarker.setAttribute("cx", x(metrics.sensitivity).toFixed(2));
                prMarker.setAttribute("cy", y(metrics.ppv).toFixed(2));
            }
        };

        const reportCommittedThreshold = () => {
            if (threshold === lastReportedThreshold || typeof trigger !== "function") return;
            lastReportedThreshold = threshold;
            // ``trigger`` is provided by gr.HTML's js_on_load environment.
            // The Python click handler rebuilds just the downloadable report
            // with this exact operating threshold.
            trigger("click", { threshold: threshold });
        };

        const nearestThreshold = (event) => {
            const rect = rocSvg.getBoundingClientRect();
            const rawX = ((event.clientX - rect.left) / rect.width) * viewWidth;
            const rawY = ((event.clientY - rect.top) / rect.height) * viewHeight;
            const wantedFpr = clamp((rawX - left) / plotWidth);
            const wantedTpr = clamp((top + plotHeight - rawY) / plotHeight);
            let closestThreshold = threshold;
            let closestDistance = Infinity;
            data.roc.forEach((point) => {
                if (!Number.isFinite(point.threshold)) return;
                const dx = point.fpr - wantedFpr;
                const dy = point.tpr - wantedTpr;
                const distance = dx * dx + dy * dy;
                if (distance < closestDistance) {
                    closestDistance = distance;
                    closestThreshold = point.threshold;
                }
            });
            return closestThreshold;
        };

        thresholdSlider.addEventListener("input", () => updateThreshold(thresholdSlider.value));
        thresholdSlider.addEventListener("change", reportCommittedThreshold);

        let dragging = false;
        rocSvg.addEventListener("pointerdown", (event) => {
            dragging = true;
            if (rocSvg.setPointerCapture) rocSvg.setPointerCapture(event.pointerId);
            updateThreshold(nearestThreshold(event));
            event.preventDefault();
        });
        rocSvg.addEventListener("pointermove", (event) => {
            if (dragging) updateThreshold(nearestThreshold(event));
        });
        const stopDragging = (event, commit = true) => {
            dragging = false;
            if (rocSvg.releasePointerCapture && rocSvg.hasPointerCapture(event.pointerId)) {
                rocSvg.releasePointerCapture(event.pointerId);
            }
            if (commit) reportCommittedThreshold();
        };
        rocSvg.addEventListener("pointerup", stopDragging);
        rocSvg.addEventListener("pointercancel", (event) => stopDragging(event, false));

        updateThreshold(threshold);
    };

    let latestPayload = null;
    const renderWhenPayloadChanges = (element) => {
        const root = element.querySelector(".ff-validation-dashboard");
        if (!root) return;
        const payload = root.dataset.payload;
        // Gradio's DOM diff can clear the child SVG while leaving an identical
        // data-payload attribute in place. The rendered marker lets us restore
        // the chart in that case without rebuilding it on our own mutations.
        if (!payload || (payload === latestPayload && root.dataset.rendered === "true")) return;
        latestPayload = payload;
        try {
            renderDashboard(root, JSON.parse(payload));
            root.dataset.rendered = "true";
        } catch (error) {
            root.innerHTML = '<div class="ff-v-client-error">Could not render the validation charts.</div>';
            // Keep the error visible in browser developer tools for support.
            console.error("FluidFlagger validation dashboard error:", error);
        }
    };

    const observer = new MutationObserver(() => {
        window.requestAnimationFrame(() => renderWhenPayloadChanges(element));
    });
    observer.observe(element, {
        attributes: true,
        attributeFilter: ["data-payload"],
        childList: true,
        subtree: true,
    });
    renderWhenPayloadChanges(element);
})();
