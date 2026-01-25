-- Movie Review Sentiment Pipeline DB Schema (PostgreSQL)

-- 1. Movies Table (Master list) - TDB which movies (hand selecting some of my favorite like Chris Nolan)
CREATE TABLE movies (
    movie_id INTEGER PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    director VARCHAR(255),
    release_date DATE,
    tmdb_url VARCHAR(500),
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
);

-- 2. Reviews Table (Raw review data)
CREATE TABLE REVIEWS(
    review_id VARCHAR(100) PRIMARY_KEY, -- tmdb review id
    movie_id INTEGER REFERENCES movies(movie_id),
    review_text TEXT NOT NULL,
    rating FLOAT, -- tmdb rating (1-10)
    review_date TIMESTAMP, --when review was posted on tmdb
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    UNIQUE (review_id) -- prevent duplicates
);

-- 3. PREDICTIONS TABLE (Inference results)
CREATE TABLE predictions (
    prediction_id SERIAL PRIMARY KEY,
    review_id VARCHAR(100) REFERENCES reviews(review_id),
    model_version VARCHAR(50) NOT NULL,  -- e.g., "logreg_v1.2.0"
    predicted_sentiment VARCHAR(20) NOT NULL,  -- 'positive' or 'negative'
    prediction_confidence FLOAT NOT NULL,  -- 0.0 to 1.0
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(review_id, model_version)  -- One prediction per review per model
);

-- 4. MODEL_REGISTRY TABLE (Track deployments)
CREATE TABLE model_registry (
    model_id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) UNIQUE NOT NULL,
    model_type VARCHAR(50),  -- 'logistic_regression', 'bert', etc.
    training_date TIMESTAMP NOT NULL,
    training_samples INTEGER,
    validation_accuracy FLOAT,
    validation_auc FLOAT,
    is_production BOOLEAN DEFAULT FALSE,  -- Currently deployed?
    mlflow_run_id VARCHAR(100),  -- Link to MLFlow experiment
    artifact_path VARCHAR(500),  -- Azure Blob path to model file
    notes TEXT
);

-- 5. MONITORING_METRICS TABLE (Track performance over time)
CREATE TABLE monitoring_metrics (
    metric_id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) REFERENCES model_registry(model_version),
    metric_date DATE NOT NULL,
    total_predictions INTEGER,
    avg_confidence FLOAT,
    positive_ratio FLOAT,  -- % of positive predictions
    low_confidence_count INTEGER,  -- Predictions with confidence < 0.6
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);