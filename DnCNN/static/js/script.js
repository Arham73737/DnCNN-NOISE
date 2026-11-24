// DOM Elements
const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const dropZoneText = document.getElementById("drop-zone-text");
const previewImg = document.getElementById("preview-img");
const submitBtn = document.getElementById("submit-btn");
const uploadForm = document.getElementById("upload-form");
const errorMsg = document.getElementById("error-msg");

const loadingSection = document.getElementById("loading");
const resultsSection = document.getElementById("results");
const originalImg = document.getElementById("original-img");

// Recommendation Elements
const recommendationBox = document.getElementById("recommendation");
const recommendationName = document.getElementById("recommendation-name");
const verdictLine = document.getElementById("verdict-line");
const recommendationReasons = document.getElementById("recommendation-reasons");
const toggleReasonsBtn = document.getElementById("toggle-reasons-btn");

const hlBest = document.getElementById("hl-best");
const hlFastest = document.getElementById("hl-fastest");
const hlNoise = document.getElementById("hl-noise");
const hlSharpness = document.getElementById("hl-sharpness");

// View Toggle Elements
const btnViewCoverflow = document.getElementById("btn-view-coverflow");
const btnViewGrid = document.getElementById("btn-view-grid");
const coverflowView = document.getElementById("coverflow-view");
const gridView = document.getElementById("grid-view");

// CoverFlow Controls
const coverflowTrack = document.getElementById("coverflow-track");
const coverflowDots = document.getElementById("coverflow-dots");
const cfPrevBtn = document.getElementById("cf-prev");
const cfNextBtn = document.getElementById("cf-next");

let selectedFile = null;
let currentCoverflowIndex = 0;
let totalCoverflowCards = 0;
let originalImageBase64 = null;

// Filter Category Subtitles
const FILTER_DESCRIPTIONS = {
  "Mean": "Spatial Box Averaging",
  "Gaussian": "Weighted 2D Gaussian Kernel",
  "Median": "Rank-Order Impulse Filter",
  "Wiener": "Adaptive Variance Estimation",
  "Bilateral": "Range & Spatial Edge-Preserver",
  "NLM": "Non-Local Patch Matching",
  "DnCNN": "Deep Residual Neural Network"
};

// -------------------------------------------------------------
// File Drag & Drop Handling
// -------------------------------------------------------------
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    handleFile(e.dataTransfer.files[0]);
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    handleFile(fileInput.files[0]);
  }
});

function handleFile(file) {
  const allowed = ["image/png", "image/jpeg", "image/bmp"];
  if (!allowed.includes(file.type)) {
    showError("Unsupported file type. Please upload a PNG, JPG, or BMP image.");
    return;
  }

  hideError();
  selectedFile = file;
  submitBtn.disabled = false;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.classList.remove("hidden");
    dropZoneText.classList.add("hidden");
    dropZone.querySelector(".drop-icon")?.classList.add("hidden");
    dropZone.querySelector(".drop-hint")?.classList.add("hidden");
  };
  reader.readAsDataURL(file);
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.remove("hidden");
}

function hideError() {
  errorMsg.classList.add("hidden");
}

// -------------------------------------------------------------
// Form Submission & API Request
// -------------------------------------------------------------
uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedFile) return;

  hideError();
  resultsSection.classList.add("hidden");
  loadingSection.classList.remove("hidden");
  submitBtn.disabled = true;

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/api/denoise", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let errMsg = `Server error (${response.status})`;
      try {
        const errJson = await response.json();
        if (errJson && errJson.error) errMsg = errJson.error;
      } catch (_) {
        if (response.status === 502) {
          errMsg = "Server ran out of memory or timed out (502 Bad Gateway). Please try a smaller image.";
        }
      }
      throw new Error(errMsg);
    }

    const data = await response.json();

    renderResults(data);
  } catch (err) {
    showError(err.message || "An error occurred while denoising.");
  } finally {
    loadingSection.classList.add("hidden");
    submitBtn.disabled = false;
  }
});

// -------------------------------------------------------------
// Render Results (3D CoverFlow + Grid)
// -------------------------------------------------------------
function renderResults(data) {
  originalImageBase64 = data.original_image_base64;
  originalImg.src = `data:image/png;base64,${originalImageBase64}`;

  const bestMethod = data.recommendation ? data.recommendation.method : null;
  const methods = data.methods;
  const methodNames = Object.keys(methods);
  totalCoverflowCards = methodNames.length;

  // Clear previous results
  coverflowTrack.innerHTML = "";
  coverflowDots.innerHTML = "";
  gridView.innerHTML = "";

  let recommendedCardIndex = 0;

  // Build Cards
  methodNames.forEach((name, index) => {
    const item = methods[name];
    const isBest = (name === bestMethod);
    if (isBest) recommendedCardIndex = index;

    // Clean formatting for high clarity & readability
    const formattedTime = item.metrics.time_sec < 0.01 
      ? `${Number(item.metrics.time_sec).toFixed(4)}s` 
      : `${Number(item.metrics.time_sec).toFixed(3)}s`;
    const formattedNoise = Number(item.metrics.noise_estimate).toFixed(2);
    const formattedSharpness = Math.round(item.metrics.sharpness).toLocaleString();
    const formattedBrisque = item.metrics.brisque !== null ? Number(item.metrics.brisque).toFixed(1) : "N/A";

    // --- 1. Create CoverFlow 3D Card ---
    const cfCard = document.createElement("div");
    cfCard.className = `coverflow-card ${isBest ? "is-recommended" : ""}`;
    cfCard.dataset.index = index;

    cfCard.innerHTML = `
      <div class="card-media">
        <img class="filtered-img" src="data:image/png;base64,${item.image_base64}" alt="${name}" />
        <img class="original-peek" src="data:image/png;base64,${originalImageBase64}" alt="Original" />
        <button type="button" class="peek-badge" title="Hold to see original noisy image">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
            <circle cx="12" cy="12" r="3"></circle>
          </svg>
          <span>Compare Raw</span>
        </button>
      </div>

      <div class="card-content">
        <div>
          <div class="card-header-row">
            <div class="card-title-group">
              <h3>${name}</h3>
              <div class="card-subtitle">${FILTER_DESCRIPTIONS[name] || "Filtering Algorithm"}</div>
            </div>
            ${isBest ? `<span class="card-rec-badge">⭐ Recommended</span>` : ""}
          </div>

          <div class="card-metrics-grid">
            <div class="card-metric-cell" title="Execution latency (lower is faster)">
              <span class="lbl">⏱ Latency</span>
              <span class="val">${formattedTime}</span>
            </div>
            <div class="card-metric-cell" title="Residual noise estimate (lower is less noise)">
              <span class="lbl">🛡 Residual Noise</span>
              <span class="val">${formattedNoise}</span>
            </div>
            <div class="card-metric-cell" title="Edge gradient variance (higher preserves more structure)">
              <span class="lbl">📐 Sharpness</span>
              <span class="val">${formattedSharpness}</span>
            </div>
            <div class="card-metric-cell" title="Blind perceptual distortion (lower score indicates more natural appearance)">
              <span class="lbl">👁 BRISQUE Quality</span>
              <span class="val">${formattedBrisque}</span>
            </div>
          </div>
        </div>

        <div class="card-actions">
          <a class="card-download-btn" href="/download/${item.download_name}" download>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            <span>Download Output</span>
          </a>
        </div>
      </div>
    `;

    // Click to focus card
    cfCard.addEventListener("click", (e) => {
      if (e.target.closest(".card-download-btn") || e.target.closest(".peek-badge")) return;
      if (currentCoverflowIndex !== index) {
        goToCard(index);
      }
    });

    // Hold-to-compare functionality
    setupPeekButton(cfCard);

    coverflowTrack.appendChild(cfCard);

    // --- 2. Create Dot Indicator ---
    const dot = document.createElement("button");
    dot.type = "button";
    dot.className = `coverflow-dot ${isBest ? "is-recommended" : ""}`;
    dot.setAttribute("aria-label", `Go to ${name}`);
    dot.addEventListener("click", () => goToCard(index));
    coverflowDots.appendChild(dot);

    // --- 3. Create Grid View Card ---
    const gridCard = document.createElement("div");
    gridCard.className = `grid-card ${isBest ? "best" : ""}`;
    gridCard.innerHTML = `
      <img src="data:image/png;base64,${item.image_base64}" alt="${name}" />
      <div class="grid-card-body">
        <div>
          <h3>
            <span>${name}</span>
            ${isBest ? `<span class="grid-rec-badge">⭐ Best Choice</span>` : ""}
          </h3>
          <div class="grid-metric-row"><span>⏱ Latency</span><span>${formattedTime}</span></div>
          <div class="grid-metric-row"><span>🛡 Residual Noise</span><span>${formattedNoise}</span></div>
          <div class="grid-metric-row"><span>📐 Sharpness</span><span>${formattedSharpness}</span></div>
          <div class="grid-metric-row"><span>👁 BRISQUE Quality</span><span>${formattedBrisque}</span></div>
        </div>
        <a class="grid-download-btn" href="/download/${item.download_name}">Download Output</a>
      </div>
    `;
    gridView.appendChild(gridCard);
  });

  // Render Recommendation Banner & Verdict Line
  if (data.recommendation && data.recommendation.method) {
    recommendationName.textContent = `${data.recommendation.method} Filter`;

    // Set the prominent line answering: Which filter gives better output?
    verdictLine.textContent = data.recommendation.verdict_line || 
      `${data.recommendation.method} filter gives the better output by offering the optimal balance between noise suppression and edge preservation.`;

    // Highlights chips
    const hl = data.recommendation.highlights || {};
    hlBest.textContent = data.recommendation.method;
    hlFastest.textContent = hl.fastest ? `${hl.fastest.name} (${hl.fastest.value})` : "-";
    hlNoise.textContent = hl.least_noise ? `${hl.least_noise.name} (${hl.least_noise.value})` : "-";
    hlSharpness.textContent = hl.sharpness ? `${hl.sharpness.name} (${hl.sharpness.value})` : "-";

    // Reasons list
    recommendationReasons.innerHTML = "";
    (data.recommendation.reasons || []).forEach((reason) => {
      const li = document.createElement("li");
      li.textContent = reason;
      recommendationReasons.appendChild(li);
    });

    recommendationBox.classList.remove("hidden");
  } else {
    recommendationBox.classList.add("hidden");
  }

  // Set initial 3D card position to the Recommended Filter!
  currentCoverflowIndex = recommendedCardIndex;
  updateCoverflow();

  resultsSection.classList.remove("hidden");
  resultsSection.scrollIntoView({ behavior: "smooth" });
}

// -------------------------------------------------------------
// 3D CoverFlow Physics & Positioning Engine
// -------------------------------------------------------------
function updateCoverflow() {
  const cards = document.querySelectorAll(".coverflow-card");
  const dots = document.querySelectorAll(".coverflow-dot");
  if (!cards.length) return;

  cards.forEach((card, i) => {
    const offset = i - currentCoverflowIndex;
    const absOffset = Math.abs(offset);

    if (offset === 0) {
      // Center Stage (Enlarged, Forward, Sharp, Full Opacity)
      card.classList.add("active");
      card.style.transform = `translateX(0px) translateZ(120px) rotateY(0deg) scale(1.08)`;
      card.style.opacity = "1";
      card.style.zIndex = "25";
      card.style.filter = "none";
      card.style.pointerEvents = "auto";
    } else if (offset < 0) {
      // Left Cards (Angled inward, stepped back in 3D space)
      card.classList.remove("active");
      const tx = offset * 180 - 40;
      const tz = Math.max(-320, 80 - absOffset * 85);
      const rotY = Math.min(40, 26 + absOffset * 3.5);
      const scale = Math.max(0.70, 1 - absOffset * 0.08);

      card.style.transform = `translateX(${tx}px) translateZ(${tz}px) rotateY(${rotY}deg) scale(${scale})`;
      card.style.opacity = Math.max(0.25, 1 - absOffset * 0.22);
      card.style.zIndex = `${18 - absOffset}`;
      card.style.filter = `brightness(0.72) blur(${Math.min(3, (absOffset - 1) * 1.5)}px)`;
      card.style.pointerEvents = "auto";
    } else {
      // Right Cards (Angled inward, stepped back in 3D space)
      card.classList.remove("active");
      const tx = offset * 180 + 40;
      const tz = Math.max(-320, 80 - absOffset * 85);
      const rotY = Math.max(-40, -26 - absOffset * 3.5);
      const scale = Math.max(0.70, 1 - absOffset * 0.08);

      card.style.transform = `translateX(${tx}px) translateZ(${tz}px) rotateY(${rotY}deg) scale(${scale})`;
      card.style.opacity = Math.max(0.25, 1 - absOffset * 0.22);
      card.style.zIndex = `${18 - absOffset}`;
      card.style.filter = `brightness(0.72) blur(${Math.min(3, (absOffset - 1) * 1.5)}px)`;
      card.style.pointerEvents = "auto";
    }
  });

  dots.forEach((dot, i) => {
    dot.classList.toggle("active", i === currentCoverflowIndex);
  });
}

function goToCard(index) {
  if (index < 0) index = 0;
  if (index >= totalCoverflowCards) index = totalCoverflowCards - 1;
  currentCoverflowIndex = index;
  updateCoverflow();
}

// Prev / Next Navigation Handlers
cfPrevBtn.addEventListener("click", () => {
  if (currentCoverflowIndex > 0) {
    goToCard(currentCoverflowIndex - 1);
  } else {
    goToCard(totalCoverflowCards - 1); // wrap around
  }
});

cfNextBtn.addEventListener("click", () => {
  if (currentCoverflowIndex < totalCoverflowCards - 1) {
    goToCard(currentCoverflowIndex + 1);
  } else {
    goToCard(0); // wrap around
  }
});

// Keyboard Navigation (ArrowLeft / ArrowRight)
window.addEventListener("keydown", (e) => {
  if (resultsSection.classList.contains("hidden")) return;
  if (coverflowView.classList.contains("hidden")) return;

  if (e.key === "ArrowLeft") {
    cfPrevBtn.click();
  } else if (e.key === "ArrowRight") {
    cfNextBtn.click();
  }
});

// Hold to Compare Original Feature
function setupPeekButton(cardElement) {
  const peekBtn = cardElement.querySelector(".peek-badge");
  const media = cardElement.querySelector(".card-media");
  if (!peekBtn || !media) return;

  const startPeek = (e) => {
    e.stopPropagation();
    media.classList.add("peeking");
  };

  const endPeek = (e) => {
    e.stopPropagation();
    media.classList.remove("peeking");
  };

  peekBtn.addEventListener("mousedown", startPeek);
  peekBtn.addEventListener("touchstart", startPeek, { passive: true });
  window.addEventListener("mouseup", endPeek);
  window.addEventListener("touchend", endPeek);
}

// -------------------------------------------------------------
// View Mode Switcher (3D CoverFlow <-> Grid)
// -------------------------------------------------------------
btnViewCoverflow.addEventListener("click", () => {
  btnViewCoverflow.classList.add("active");
  btnViewGrid.classList.remove("active");
  coverflowView.classList.remove("hidden");
  gridView.classList.add("hidden");
  updateCoverflow();
});

btnViewGrid.addEventListener("click", () => {
  btnViewGrid.classList.add("active");
  btnViewCoverflow.classList.remove("active");
  coverflowView.classList.add("hidden");
  gridView.classList.remove("hidden");
});

// Toggle Reasons Dropdown
toggleReasonsBtn?.addEventListener("click", () => {
  const isOpen = !recommendationReasons.classList.contains("hidden");
  if (isOpen) {
    recommendationReasons.classList.add("hidden");
    toggleReasonsBtn.classList.remove("open");
    toggleReasonsBtn.querySelector("span").textContent = "Show detailed scoring metrics";
  } else {
    recommendationReasons.classList.remove("hidden");
    toggleReasonsBtn.classList.add("open");
    toggleReasonsBtn.querySelector("span").textContent = "Hide detailed scoring metrics";
  }
});
