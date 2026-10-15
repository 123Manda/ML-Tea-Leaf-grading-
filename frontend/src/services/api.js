const API_BASE_URL = "http://127.0.0.1:8000";


export async function predictTeaLeaf(file) {
  const formData = new FormData();

  formData.append("file", file);

  const response = await fetch(
    `${API_BASE_URL}/predict`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);

    throw new Error(
      errorData?.detail ||
        "Tea leaf prediction failed."
    );
  }

  return response.json();
}


export async function getPredictionHistory(limit = 20) {
  const response = await fetch(
    `${API_BASE_URL}/predictions?limit=${limit}`
  );

  if (!response.ok) {
    throw new Error(
      "Failed to load prediction history."
    );
  }

  return response.json();
}


export async function getPrediction(predictionId) {
  const response = await fetch(
    `${API_BASE_URL}/predictions/${predictionId}`
  );

  if (!response.ok) {
    throw new Error(
      "Failed to load prediction details."
    );
  }

  return response.json();
}


export async function deletePrediction(predictionId) {
  const response = await fetch(
    `${API_BASE_URL}/predictions/${predictionId}`,
    {
      method: "DELETE",
    }
  );

  if (!response.ok) {
    throw new Error(
      "Failed to delete prediction."
    );
  }

  return response.json();
}


export function getStoredImageUrl(imageFileId) {
  return `${API_BASE_URL}/images/${imageFileId}`;
}


export function getPredictionReportUrl(predictionId) {
  return `${API_BASE_URL}/predictions/${predictionId}/report`;
}


export function downloadPredictionReport(predictionId) {
  const reportUrl =
    getPredictionReportUrl(predictionId);

  window.open(
    reportUrl,
    "_blank"
  );
}


export async function checkBackendHealth() {
  const response = await fetch(
    `${API_BASE_URL}/health`
  );

  if (!response.ok) {
    throw new Error(
      "Backend health check failed."
    );
  }

  return response.json();
}


export { API_BASE_URL };