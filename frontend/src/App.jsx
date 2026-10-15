import { useEffect, useState } from "react";

import Login from "./components/Login";

import {
  checkBackendHealth,
  deletePrediction,
  downloadPredictionReport,
  getPredictionHistory,
  getStoredImageUrl,
  predictTeaLeaf,
} from "./services/api";

import "./App.css";


function App() {
  const [isLoggedIn, setIsLoggedIn] = useState(() => {
    return (
      sessionStorage.getItem(
        "teaLeafGradingLoggedIn"
      ) === "true"
    );
  });


  function handleLogin() {
    sessionStorage.setItem(
      "teaLeafGradingLoggedIn",
      "true"
    );

    setIsLoggedIn(true);
  }


  function handleLogout() {
    sessionStorage.removeItem(
      "teaLeafGradingLoggedIn"
    );

    setIsLoggedIn(false);
  }


  if (!isLoggedIn) {
    return (
      <Login
        onLogin={handleLogin}
      />
    );
  }


  return (
    <TeaLeafDashboard
      onLogout={handleLogout}
    />
  );
}


function TeaLeafDashboard({ onLogout }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [prediction, setPrediction] = useState(null);

  const [history, setHistory] = useState([]);

  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);

  const [error, setError] = useState("");
  const [backendOnline, setBackendOnline] = useState(false);


  // ======================================================
  // BACKEND HEALTH
  // ======================================================

  async function loadBackendHealth() {
    try {
      await checkBackendHealth();
      setBackendOnline(true);
    } catch {
      setBackendOnline(false);
    }
  }


  // ======================================================
  // LOAD HISTORY
  // ======================================================

  async function loadHistory() {
    try {
      setHistoryLoading(true);

      const data = await getPredictionHistory(20);

      setHistory(data.predictions || []);
    } catch (err) {
      console.error(err);
    } finally {
      setHistoryLoading(false);
    }
  }


  useEffect(() => {
    loadBackendHealth();
    loadHistory();
  }, []);


  // ======================================================
  // IMAGE SELECT
  // ======================================================

  function handleFileChange(event) {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError("Maximum allowed image size is 10 MB.");
      return;
    }

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    const newPreviewUrl = URL.createObjectURL(file);

    setSelectedFile(file);
    setPreviewUrl(newPreviewUrl);
    setPrediction(null);
    setError("");
  }


  // ======================================================
  // PREDICT
  // ======================================================

  async function handlePredict() {
    if (!selectedFile) {
      setError("Please select a tea leaf image first.");
      return;
    }

    try {
      setLoading(true);
      setError("");

      const data = await predictTeaLeaf(selectedFile);

      setPrediction(data);

      await loadHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }


  // ======================================================
  // DOWNLOAD PDF
  // ======================================================

  function handleDownloadReport(predictionId) {
    if (!predictionId) {
      setError("Prediction ID is not available.");
      return;
    }

    downloadPredictionReport(predictionId);
  }


  // ======================================================
  // RESET
  // ======================================================

  function handleReset() {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setSelectedFile(null);
    setPreviewUrl("");
    setPrediction(null);
    setError("");

    const input = document.getElementById(
      "tea-leaf-input"
    );

    if (input) {
      input.value = "";
    }
  }


  // ======================================================
  // DELETE
  // ======================================================

  async function handleDelete(predictionId) {
    const confirmed = window.confirm(
      "Delete this prediction and its stored image?"
    );

    if (!confirmed) {
      return;
    }

    try {
      await deletePrediction(predictionId);

      setHistory((currentHistory) =>
        currentHistory.filter(
          (item) =>
            item.prediction_id !== predictionId
        )
      );

      if (
        prediction?.prediction_id === predictionId
      ) {
        setPrediction(null);
      }
    } catch (err) {
      setError(err.message);
    }
  }


  // ======================================================
  // HELPERS
  // ======================================================

  function formatGrade(grade) {
    if (!grade) {
      return "";
    }

    return grade.replaceAll("_", " ");
  }


  function getGradeClass(grade) {
    if (grade === "Best") {
      return "grade-best";
    }

    if (grade === "Below_Best") {
      return "grade-below-best";
    }

    if (grade === "Poor") {
      return "grade-poor";
    }

    return "";
  }


  const result = prediction?.result;


  return (
    <div className="app">
      <header className="topbar">
        <div>
          <p className="system-label">
            ML Powered Tea Factory Management System
          </p>

          <h1>
            Smart Tea Leaf Grading
          </h1>

          <p className="subtitle">
            OpenCV Feature Extraction + XGBoost Classification
          </p>
        </div>


        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "12px",
            flexWrap: "wrap",
          }}
        >
          <div
            className={`backend-status ${
              backendOnline
                ? "online"
                : "offline"
            }`}
          >
            <span className="status-dot" />

            {backendOnline
              ? "Online"
              : "Offline"}
          </div>


          <button
            type="button"
            onClick={onLogout}
            style={{
              border: "1px solid rgba(255,255,255,0.35)",
              borderRadius: "10px",
              padding: "10px 16px",
              background: "rgba(255,255,255,0.12)",
              color: "#ffffff",
              fontWeight: "700",
              cursor: "pointer",
            }}
          >
            Sign Out
          </button>
        </div>
      </header>


      <main className="main-content">

        {/* ================================================
            UPLOAD
        ================================================= */}

        <section className="card upload-card">
          <div className="section-heading">
            <div>
              <p className="section-number">
                01
              </p>

              <h2>
                Upload Tea Leaf
              </h2>
            </div>

            <p>
              Upload one tea leaf image for automated
              quality grading.
            </p>
          </div>


          <div className="upload-layout">
            <div className="upload-box">
              <input
                id="tea-leaf-input"
                type="file"
                accept=".jpg,.jpeg,.png,.webp,image/*"
                onChange={handleFileChange}
              />

              <label
                htmlFor="tea-leaf-input"
                className="upload-label"
              >
                <span className="upload-icon">
                  +
                </span>

                <strong>
                  Select Tea Leaf Image
                </strong>

                <span>
                  JPG, PNG or WEBP · Maximum 10 MB
                </span>
              </label>
            </div>


            <div className="preview-box">
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="Selected tea leaf preview"
                />
              ) : (
                <div className="preview-placeholder">
                  <span>
                    Leaf Preview
                  </span>
                </div>
              )}
            </div>
          </div>


          {selectedFile && (
            <div className="selected-file">
              <div>
                <strong>
                  {selectedFile.name}
                </strong>

                <span>
                  {(
                    selectedFile.size /
                    1024 /
                    1024
                  ).toFixed(2)}{" "}
                  MB
                </span>
              </div>

              <button
                type="button"
                className="secondary-button"
                onClick={handleReset}
              >
                Clear
              </button>
            </div>
          )}


          {error && (
            <div className="error-message">
              {error}
            </div>
          )}


          <button
            type="button"
            className="primary-button"
            onClick={handlePredict}
            disabled={
              !selectedFile ||
              loading ||
              !backendOnline
            }
          >
            {loading
              ? "Analyzing Tea Leaf..."
              : "Grade Tea Leaf"}
          </button>
        </section>


        {/* ================================================
            RESULT
        ================================================= */}

        {result && (
          <section className="card result-card">
            <div className="section-heading">
              <div>
                <p className="section-number">
                  02
                </p>

                <h2>
                  Grading Result
                </h2>
              </div>

              <p>
                Prediction generated by the selected
                XGBoost classification model.
              </p>
            </div>


            <div className="result-summary">
              <div
                className={`grade-result ${getGradeClass(
                  result.predicted_grade
                )}`}
              >
                <span>
                  Predicted Quality
                </span>

                <strong>
                  {formatGrade(
                    result.predicted_grade
                  )}
                </strong>
              </div>


              <div className="confidence-result">
                <span>
                  Confidence
                </span>

                <strong>
                  {result.confidence_percent}%
                </strong>

                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{
                      width: `${result.confidence_percent}%`,
                    }}
                  />
                </div>
              </div>
            </div>


            <div className="result-grid">
              <div className="result-panel">
                <h3>
                  Class Probabilities
                </h3>

                {Object.entries(
                  result.probabilities || {}
                ).map(
                  ([className, probability]) => (
                    <div
                      className="probability-row"
                      key={className}
                    >
                      <div>
                        <span>
                          {formatGrade(
                            className
                          )}
                        </span>

                        <strong>
                          {(
                            probability * 100
                          ).toFixed(2)}
                          %
                        </strong>
                      </div>

                      <div className="probability-bar">
                        <div
                          className="probability-fill"
                          style={{
                            width: `${
                              probability * 100
                            }%`,
                          }}
                        />
                      </div>
                    </div>
                  )
                )}
              </div>


              <div className="result-panel">
                <h3>
                  Leaf Measurements
                </h3>

                <div className="measurement-grid">
                  <div>
                    <span>
                      Length
                    </span>

                    <strong>
                      {
                        result.leaf_measurements
                          ?.length_px
                      }{" "}
                      px
                    </strong>
                  </div>

                  <div>
                    <span>
                      Width
                    </span>

                    <strong>
                      {
                        result.leaf_measurements
                          ?.width_px
                      }{" "}
                      px
                    </strong>
                  </div>

                  <div>
                    <span>
                      Area
                    </span>

                    <strong>
                      {
                        result.leaf_measurements
                          ?.area_px2
                      }{" "}
                      px²
                    </strong>
                  </div>

                  <div>
                    <span>
                      Perimeter
                    </span>

                    <strong>
                      {
                        result.leaf_measurements
                          ?.perimeter_px
                      }{" "}
                      px
                    </strong>
                  </div>

                  <div>
                    <span>
                      Aspect Ratio
                    </span>

                    <strong>
                      {
                        result.leaf_measurements
                          ?.aspect_ratio
                      }
                    </strong>
                  </div>

                  <div>
                    <span>
                      Segmentation
                    </span>

                    <strong>
                      {
                        result.segmentation_method
                      }
                    </strong>
                  </div>
                </div>
              </div>
            </div>


            <div className="database-status">
              <span>
                ✓
              </span>

              <div>
                <strong>
                  Saved
                </strong>

                <p>
                  Prediction ID:{" "}
                  {prediction.prediction_id}
                </p>
              </div>
            </div>


            <button
              type="button"
              className="report-button"
              onClick={() =>
                handleDownloadReport(
                  prediction.prediction_id
                )
              }
            >
              Download PDF Report
            </button>
          </section>
        )}


        {/* ================================================
            HISTORY
        ================================================= */}

        <section className="card history-card">
          <div className="section-heading">
            <div>
              <p className="section-number">
                03
              </p>

              <h2>
                Prediction History
              </h2>
            </div>

            <button
              type="button"
              className="secondary-button"
              onClick={loadHistory}
              disabled={historyLoading}
            >
              {historyLoading
                ? "Refreshing..."
                : "Refresh"}
            </button>
          </div>


          {historyLoading &&
          history.length === 0 ? (
            <div className="empty-state">
              Loading prediction history...
            </div>
          ) : history.length === 0 ? (
            <div className="empty-state">
              No predictions have been stored yet.
            </div>
          ) : (
            <div className="history-list">
              {history.map((item) => (
                <article
                  className="history-item"
                  key={item.prediction_id}
                >
                  <div className="history-image">
                    <img
                      src={getStoredImageUrl(
                        item.image_file_id
                      )}
                      alt={item.image_name}
                    />
                  </div>


                  <div className="history-info">
                    <span className="history-name">
                      {item.image_name}
                    </span>

                    <strong
                      className={getGradeClass(
                        item.predicted_grade
                      )}
                    >
                      {formatGrade(
                        item.predicted_grade
                      )}
                    </strong>

                    <span>
                      Confidence:{" "}
                      {item.confidence_percent}%
                    </span>

                    <span>
                      {item.created_at
                        ? new Date(
                            item.created_at
                          ).toLocaleString()
                        : ""}
                    </span>
                  </div>


                  <div className="history-measurements">
                    <span>
                      Length

                      <strong>
                        {
                          item.leaf_measurements
                            ?.length_px
                        }{" "}
                        px
                      </strong>
                    </span>

                    <span>
                      Width

                      <strong>
                        {
                          item.leaf_measurements
                            ?.width_px
                        }{" "}
                        px
                      </strong>
                    </span>

                    <span>
                      Area

                      <strong>
                        {
                          item.leaf_measurements
                            ?.area_px2
                        }{" "}
                        px²
                      </strong>
                    </span>
                  </div>


                  <div className="history-actions">
                    <button
                      type="button"
                      className="history-report-button"
                      onClick={() =>
                        handleDownloadReport(
                          item.prediction_id
                        )
                      }
                    >
                      Download PDF
                    </button>

                    <button
                      type="button"
                      className="delete-button"
                      onClick={() =>
                        handleDelete(
                          item.prediction_id
                        )
                      }
                    >
                      Delete
                    </button>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      </main>


      <footer>
        Smart Tea Leaf Grading and Quality Control
      </footer>
    </div>
  );
}


export default App;