# Wine Classification API with Docker

A machine learning API for wine classification using the Wine dataset from scikit-learn, containerized with Docker and automated with GitHub Actions.

## Features

-   **Machine Learning Model**: Random Forest classifier trained on Wine dataset
-   **RESTful API**: Flask-based API with health check and prediction endpoints
-   **Docker Support**: Fully containerized application
-   **Automated Testing**: GitHub Actions for CI/CD

## Quick Start

### Prerequisites

-   Docker installed on your system
-   Git (for cloning the repository)

### Running with Docker

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Anibalrojo/wine-classifier-docker
    cd wine-classifier-docker
    ```

2.  **Build the Docker image:**
    ```bash
    docker build -t wine-classifier .
    ```

3.  **Run the container:**
    ```bash
    docker run -p 5000:5000 wine-classifier
    ```

4.  **Test the API:**
    ```bash
    # Health check
    curl http://localhost:5000/

    # Make a prediction
    curl -X POST -H "Content-Type: application/json" -d @request.json http://localhost:5000/predict
    ```

## API Endpoints

### Health Check
```http
GET /
```

**Response Example:**
```json
{
  "message": "Classification API (wine) is up",
  "expected_features": ["alcohol", "malic_acid", "..."]
  "predict_endpoint": "/predict"
}
```

### Wine Classification
```http
POST /predict
```

**Request Body Example (from `request.json`):**
```json
{
  "instances": [
    {
      "alcohol": 13.2, "malic_acid": 1.78, "ash": 2.14, "alcalinity_of_ash": 11.2,
      "magnesium": 100.0, "total_phenols": 2.65, "flavanoids": 2.76,
      "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.28, "color_intensity": 4.38,
      "hue": 1.05, "od280/od315_of_diluted_wines": 3.4, "proline": 1050.0
    }
  ]
}
```

**Response Example:**
```json
{
  "predictions": [0],
  "classes": ["class_0"],
  "probas": [[0.98, 0.01, 0.01]],
  "class_names": ["class_0", "class_1", "class_2"]
}
```

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── docker-build.yml    # GitHub Actions CI/CD
├── Dockerfile                  # Docker configuration
├── app.py                     # Flask API application
├── train_model.py             # Model training script
├── requirements.txt           # Python dependencies
├── request.json              # Sample request for testing
├── .gitignore                # Git ignore rules
├── LICENSE                   # MIT License
└── README.md                 # This file
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with Python, Flask, scikit-learn, and Docker**
