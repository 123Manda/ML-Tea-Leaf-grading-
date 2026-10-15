# Smart Tea Leaf Grading and Quality Control System

## ML-Powered Tea Factory Management System

This project implements the **Smart Tea Leaf Grading and Quality Control** component of the ML-Powered Tea Factory Management System.

The system uses classical machine learning and computer vision techniques to classify tea leaf images into three quality categories:

- Best
- Below Best
- Poor

The implementation combines **OpenCV-based image processing**, handcrafted image features, and an **XGBoost classifier** to perform automated tea leaf quality grading.

The system also includes a **FastAPI backend**, **MongoDB Atlas database**, **React + Vite frontend**, prediction history, image storage, and downloadable PDF reports.

---

## Research Component

**Component:** Smart Tea Leaf Grading and Quality Control

**Main System:** ML-Powered Tea Factory Management System

The classification categories used in this research are aligned with green-leaf quality terminology used in the Sri Lankan tea industry:

### Best
Tender or young tea material without visible damage.

### Below Best
Tender or young tea material containing visible damage or quality deterioration.

### Poor
Coarse or mature leaf material and mature stem-related material.

---

## Research Objective

The main objective of this component is to develop an automated machine-learning-based system capable of classifying tea leaf images according to quality using measurable visual characteristics.

The system extracts:

- Color characteristics
- Shape characteristics
- Size characteristics
- Texture characteristics

These features are then used to train and compare multiple machine-learning algorithms.

---

## Dataset

The final cleaned dataset contains:

| Class | Images |
|---|---:|
| Best | 560 |
| Below Best | 550 |
| Poor | 546 |
| **Total** | **1656** |

The dataset was divided into training, validation, and testing sets.

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Best | 392 | 84 | 84 |
| Below Best | 385 | 82 | 83 |
| Poor | 382 | 81 | 83 |
| **Total** | **1159** | **247** | **250** |

The dataset preparation process included:

- Removal of augmentation-generated files
- Removal of exact duplicate images
- Dataset cleaning
- Stratified train/validation/test splitting
- Perceptual-hash-based leakage checking
- Roboflow dataset organization and versioning

A near-duplicate leakage check was performed between the training, validation, and test sets.

---

## Roboflow Usage

Roboflow was used for:

- Dataset organization
- Class verification
- Dataset split management
- Dataset versioning

Roboflow was **not used to train the final classification model**.

The final machine-learning models were trained locally using features extracted through the OpenCV-based feature extraction pipeline.

---

## Image Processing Pipeline

The system processes each uploaded image through the following workflow:

```text
Tea Leaf Image
      ↓
Image Preprocessing
      ↓
Leaf Segmentation
      ↓
Feature Extraction
      ↓
Machine Learning Classification
      ↓
Best / Below Best / Poor
      ↓
Confidence + Measurements
      ↓
MongoDB Storage
      ↓
PDF Report
```

---

## Leaf Segmentation

The segmentation pipeline uses multiple methods to obtain the tea leaf region.

The methods include:

1. HSV-based segmentation
2. LAB-based fallback segmentation
3. GrabCut fallback segmentation

The segmentation method used for each prediction is also recorded in the prediction result.

---

## Feature Extraction

A total of **40 numerical image features** are extracted.

### Color Features

Examples include:

- Mean RGB values
- RGB standard deviation
- Mean HSV values
- HSV standard deviation
- LAB color values
- Green ratio
- Yellow/brown ratio
- Dark pixel ratio
- Bright pixel ratio

### Shape and Size Features

Examples include:

- Leaf area
- Leaf perimeter
- Bounding-box width
- Bounding-box height
- Aspect ratio
- Extent
- Solidity
- Circularity
- Equivalent diameter
- Ellipse axis ratio

### Texture Features

Examples include:

- GLCM contrast
- GLCM homogeneity
- GLCM energy
- GLCM correlation
- Grayscale mean
- Grayscale standard deviation
- Entropy
- Laplacian variance

---

## Machine Learning Models

Three classical machine-learning approaches were evaluated.

### Support Vector Machine

Validation Accuracy:

```text
87.04%
```

Macro F1 Score:

```text
0.8715
```

### Random Forest

Validation Accuracy:

```text
88.66%
```

Macro F1 Score:

```text
0.8861
```

### XGBoost

Validation Accuracy:

```text
91.09%
```

Macro F1 Score:

```text
0.9110
```

Based on validation performance, **XGBoost was selected as the final classification model**.

---

## Final Test Results

The selected XGBoost model was evaluated once using the previously unseen test dataset.

### Overall Test Performance

| Metric | Result |
|---|---:|
| Accuracy | 89.20% |
| Macro Precision | 89.68% |
| Macro Recall | 89.19% |
| Macro F1 Score | 89.04% |

### Class-Level F1 Scores

| Class | F1 Score |
|---|---:|
| Best | 87.01% |
| Below Best | 84.77% |
| Poor | 95.35% |

The validation-to-test accuracy difference was approximately **1.89 percentage points**.

---

## Explainable AI

SHAP analysis was performed to identify the features that had the highest influence on XGBoost predictions.

Important features included:

- Yellow/brown ratio
- Hue variation
- Mean hue
- Laplacian variance
- LAB color features
- Green ratio

SHAP values are used as post-hoc explainability information for the trained XGBoost model.

---

## Ablation Study

An ablation study was performed using training and validation data.

| Feature Set | Validation Accuracy |
|---|---:|
| All Features | 91.09% |
| Color + Shape | 89.88% |
| Color + Texture | 89.07% |
| Color Only | 88.26% |
| Shape + Texture | 77.73% |
| Shape Only | 65.99% |
| Texture Only | 63.56% |

The results indicate that color information provides the strongest individual contribution, while combining all feature groups provides the best overall performance.

---

## Technologies Used

### Machine Learning

- Python
- Scikit-learn
- XGBoost
- SHAP
- Joblib

### Computer Vision

- OpenCV
- Scikit-image
- Pillow
- NumPy

### Backend

- FastAPI
- Uvicorn
- Python Multipart
- ReportLab

### Database

- MongoDB Atlas
- PyMongo
- GridFS

### Frontend

- React
- Vite
- JavaScript
- CSS

### Dataset Management

- Roboflow

### Development Tools

- Visual Studio Code
- Git
- GitHub
- PowerShell

---

## Project Structure

```text
Tea_Leaf_Grading_ML/
│
├── backend/
│   ├── main.py
│   ├── database.py
│   ├── storage.py
│   ├── report_generator.py
│   └── ...
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── docs/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   ├── App.jsx
│   │   └── App.css
│   ├── package.json
│   └── ...
│
├── models/
│   ├── svm_baseline.pkl
│   ├── random_forest_baseline.pkl
│   ├── xgboost_baseline.pkl
│   ├── xgboost_feature_columns.json
│   └── xgboost_label_encoder.pkl
│
├── results/
│   ├── charts/
│   └── ...
│
├── src/
│   ├── audit_dataset.py
│   ├── build_clean_dataset.py
│   ├── split_dataset.py
│   ├── check_split_leakage.py
│   ├── extract_features.py
│   ├── train_svm.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── evaluate_xgboost_test.py
│   ├── explain_xgboost_shap.py
│   ├── run_ablation_study.py
│   ├── generate_research_charts.py
│   └── predict_single_image.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone the Repository

```powershell
git clone <repository-url>
cd Tea_Leaf_Grading_ML
```

---

## 2. Create Python Virtual Environment

Python 3.12 is recommended.

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Verify:

```powershell
python --version
```

---

## 3. Install Python Dependencies

```powershell
python -m pip install -r requirements.txt
```

If required, the main packages can also be installed using:

```powershell
python -m pip install numpy pandas pillow opencv-python scikit-image scikit-learn scipy joblib xgboost shap matplotlib fastapi uvicorn python-multipart pymongo dnspython python-dotenv reportlab
```

---

# MongoDB Configuration

Create a `.env` file in the project root.

Example:

```env
MONGODB_URI="your_mongodb_atlas_connection_string"
MONGODB_DB_NAME="tea_leaf_grading_db"
```

> Do not commit the `.env` file to GitHub.

The `.gitignore` file excludes `.env`.

---

# Start the Backend

Activate the virtual environment first:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start FastAPI:

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

Backend:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# Start the Frontend

Open another terminal:

```powershell
cd frontend
```

Install dependencies if required:

```powershell
npm install
```

Run the React application:

```powershell
npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

# Main API Endpoints

### Health Check

```http
GET /health
```

### Predict Tea Leaf Grade

```http
POST /predict
```

Accepts an uploaded tea leaf image and performs:

- Segmentation
- Feature extraction
- XGBoost prediction
- Confidence calculation
- Leaf measurement extraction
- MongoDB storage

### Prediction History

```http
GET /predictions
```

### Get Individual Prediction

```http
GET /predictions/{prediction_id}
```

### Retrieve Stored Image

```http
GET /images/{image_file_id}
```

### Delete Prediction

```http
DELETE /predictions/{prediction_id}
```

### Download PDF Report

```http
GET /predictions/{prediction_id}/report
```

---

# Frontend Features

The React frontend provides:

- Login interface
- Tea leaf image selection
- Image preview
- Automatic grading
- Best / Below Best / Poor prediction
- Prediction confidence
- Class probability visualization
- Leaf measurements
- Segmentation method
- MongoDB prediction history
- Stored image preview
- Delete prediction functionality
- PDF report download
- Backend connection status
- Sign-out functionality

---

# PDF Report

A downloadable PDF report is generated for stored predictions.

The report contains:

- Uploaded tea leaf image
- Prediction ID
- Image name
- Predicted quality grade
- Prediction confidence
- Class probabilities
- Leaf measurements
- Segmentation method
- Model information
- Prediction date and time

The report is generated using **ReportLab**.

---

# Leaf Measurements

The system provides:

- Length
- Width
- Area
- Perimeter
- Aspect ratio

These measurements are currently expressed in **image-space pixels**.

They must not be interpreted as centimetres or millimetres unless a calibrated physical reference or camera calibration process is introduced.

---

# Database Storage

MongoDB Atlas stores prediction metadata including:

- Prediction ID
- Image File ID
- Original image name
- Predicted grade
- Confidence
- Class probabilities
- Segmentation method
- Leaf measurements
- Prediction timestamp

Original tea leaf images are stored using **MongoDB GridFS**.

---

# Security Note

The current login interface is intended for project demonstration purposes.

For a production environment:

- Authentication should be implemented in the backend
- Passwords should never be stored directly in frontend source code
- Passwords should be securely hashed
- Authentication tokens or secure sessions should be used
- Secrets should remain in environment variables

---

# Research Limitations

The current system classifies individual tea leaf images using visual features.

Some tea quality properties such as tenderness, maturity, physical condition, and plucking quality may not always be completely observable from a single two-dimensional image.

The current dataset and labels should therefore be interpreted within the scope of this research implementation.

Future work can include:

- Larger datasets collected directly from multiple tea estates and factories
- Expert-verified leaf grading
- Controlled image capture environments
- Physical size calibration
- Deep-learning-based segmentation
- Mobile application support
- Real-time camera prediction
- Continuous model monitoring and retraining

---

# Conclusion

This research demonstrates an end-to-end machine-learning-based tea leaf grading pipeline integrating computer vision, classical machine learning, explainable AI, backend services, database storage, and a web-based user interface.

Among the evaluated models, **XGBoost achieved the best validation performance and obtained 89.20% accuracy on the independent test dataset**.

The completed system provides automated quality classification together with prediction confidence, image-based measurements, prediction history, and downloadable reporting to support tea leaf quality-control activities.

---

## Author

Smart Tea Leaf Grading and Quality Control  
ML-Powered Tea Factory Management System