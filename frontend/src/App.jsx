import { useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL;

function App() {
  const [message, setMessage] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handlePredict = async () => {
    const trimmedMessage = message.trim();

    if (!trimmedMessage) {
      setError("Enter a message before starting the analysis.");
      setResult(null);
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: trimmedMessage,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "The analysis could not be completed."
        );
      }

      setResult(data);
    } catch (requestError) {
      setError(
        requestError.message ||
          "Unable to connect to the classification service."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setMessage("");
    setResult(null);
    setError("");
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      handlePredict();
    }
  };

  const confidence = result ? result.confidence * 100 : 0;
  const hamProbability = result ? result.ham_probability * 100 : 0;
  const spamProbability = result ? result.spam_probability * 100 : 0;

  const isSpam = result?.predicted_label === "spam";

  return (
    <main className="app">
      <div className="background-glow background-glow-one"></div>
      <div className="background-glow background-glow-two"></div>

      <section className="dashboard">

        {/* Header */}

        <header className="topbar">
          <div className="brand">
            <div className="brand-mark">
              <svg
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  d="M12 3L19 6V11.5C19 16.2 16.1 19.8 12 21C7.9 19.8 5 16.2 5 11.5V6L12 3Z"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinejoin="round"
                />
                <path
                  d="M9 12L11 14L15 10"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>

            <div>
              <p className="brand-name">SENTINEL AI</p>
              <p className="brand-subtitle">
                Message Security Engine
              </p>
            </div>
          </div>

          <div className="system-status">
            <span className="status-dot"></span>
            <span>API ONLINE</span>
          </div>
        </header>


        {/* Hero */}

        <section className="hero">
          <div className="hero-badge">
            <span className="badge-pulse"></span>
            AI-POWERED THREAT ANALYSIS
          </div>

          <h1>
            Detect suspicious
            <span> messages instantly.</span>
          </h1>

          <p>
            Analyze SMS and email content using machine learning
            to identify potentially unwanted messages.
          </p>
        </section>


        {/* Main Analyzer */}

        <section className="analyzer-card">

          <div className="card-header">
            <div>
              <p className="section-label">MESSAGE ANALYZER</p>
              <h2>Analyze a message</h2>
            </div>

            <div className="model-badge">
              <span className="model-dot"></span>
              ML MODEL READY
            </div>
          </div>


          <div className="input-wrapper">

            <div className="input-topline">
              <label htmlFor="message">
                Message content
              </label>

              <span className="character-count">
                {message.length.toLocaleString()} / 5,000
              </span>
            </div>


            <textarea
              id="message"
              value={message}
              onChange={(event) => {
                setMessage(event.target.value);
                setError("");
              }}
              onKeyDown={handleKeyDown}
              placeholder="Paste an SMS or email message here..."
              maxLength={5000}
              disabled={loading}
            />


            <div className="input-footer">
              <span>
                Your message is analyzed by the local classification API.
              </span>

              <span className="shortcut">
                CTRL + ENTER
              </span>
            </div>

          </div>


          {error && (
            <div className="error-message">
              <div className="error-icon">!</div>

              <div>
                <strong>Analysis error</strong>
                <p>{error}</p>
              </div>
            </div>
          )}


          <div className="action-row">

            <button
              className="analyze-button"
              onClick={handlePredict}
              disabled={loading || !message.trim()}
            >
              {loading ? (
                <>
                  <span className="spinner"></span>
                  Analyzing message
                </>
              ) : (
                <>
                  Analyze message
                  <span className="button-arrow">→</span>
                </>
              )}
            </button>


            {(message || result) && (
              <button
                className="clear-button"
                onClick={handleClear}
                disabled={loading}
              >
                Clear
              </button>
            )}

          </div>

        </section>


        {/* Result */}

        {result && (
          <section className="result-section">

            <div className="result-header">
              <div>
                <p className="section-label">ANALYSIS COMPLETE</p>
                <h2>Security assessment</h2>
              </div>

              <span className="result-time">
                REAL-TIME ANALYSIS
              </span>
            </div>


            <div
              className={`verdict-card ${
                isSpam ? "verdict-spam" : "verdict-ham"
              }`}
            >
              <div className="verdict-main">

                <div className="verdict-icon">
                  {isSpam ? (
                    <svg
                      viewBox="0 0 24 24"
                      aria-hidden="true"
                    >
                      <path
                        d="M12 3L21 20H3L12 3Z"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.8"
                        strokeLinejoin="round"
                      />
                      <path
                        d="M12 9V13"
                        stroke="currentColor"
                        strokeWidth="1.8"
                        strokeLinecap="round"
                      />
                      <circle
                        cx="12"
                        cy="16.5"
                        r="0.8"
                        fill="currentColor"
                      />
                    </svg>
                  ) : (
                    <svg
                      viewBox="0 0 24 24"
                      aria-hidden="true"
                    >
                      <circle
                        cx="12"
                        cy="12"
                        r="9"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.8"
                      />
                      <path
                        d="M8 12L10.8 14.8L16 9.5"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.8"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  )}
                </div>


                <div className="verdict-text">
                  <span>CLASSIFICATION</span>

                  <h3>
                    {isSpam
                      ? "SPAM DETECTED"
                      : "MESSAGE APPEARS SAFE"}
                  </h3>

                  <p>
                    {isSpam
                      ? "The model identified patterns commonly associated with spam messages."
                      : "The model did not identify strong spam patterns in this message."}
                  </p>
                </div>

              </div>


              <div className="confidence-display">
                <span>CONFIDENCE</span>
                <strong>{confidence.toFixed(2)}%</strong>
              </div>

            </div>


            {/* Probability Analysis */}

            <div className="probability-card">

              <div className="probability-header">
                <div>
                  <p className="section-label">MODEL OUTPUT</p>
                  <h3>Classification probability</h3>
                </div>

                <span className="probability-note">
                  TF-IDF + Logistic Regression
                </span>
              </div>


              <div className="probability-row">

                <div className="probability-title">
                  <div className="probability-name">
                    <span className="legend-dot ham-dot"></span>
                    HAM
                  </div>

                  <strong>
                    {hamProbability.toFixed(2)}%
                  </strong>
                </div>

                <div className="probability-track">
                  <div
                    className="probability-value ham-value"
                    style={{
                      width: `${hamProbability}%`,
                    }}
                  ></div>
                </div>

              </div>


              <div className="probability-row">

                <div className="probability-title">
                  <div className="probability-name">
                    <span className="legend-dot spam-dot"></span>
                    SPAM
                  </div>

                  <strong>
                    {spamProbability.toFixed(2)}%
                  </strong>
                </div>

                <div className="probability-track">
                  <div
                    className="probability-value spam-value"
                    style={{
                      width: `${spamProbability}%`,
                    }}
                  ></div>
                </div>

              </div>

            </div>


            {/* Model Information */}

            <div className="info-grid">

              <div className="info-card">
                <span className="info-label">
                  MODEL
                </span>

                <strong>
                  Logistic Regression
                </strong>

                <p>
                  Text classification algorithm
                </p>
              </div>


              <div className="info-card">
                <span className="info-label">
                  FEATURES
                </span>

                <strong>
                  TF-IDF
                </strong>

                <p>
                  Unigram + bigram representation
                </p>
              </div>


              <div className="info-card">
                <span className="info-label">
                  RESPONSE
                </span>

                <strong>
                  Instant
                </strong>

                <p>
                  Real-time API inference
                </p>
              </div>

            </div>


            <div className="disclaimer">
              <span className="disclaimer-icon">i</span>

              <p>
                This is a machine-learning prediction, not a
                guarantee that a message is safe or malicious.
                Avoid opening suspicious links or sharing
                sensitive information based solely on this result.
              </p>
            </div>

          </section>
        )}


        {/* Empty State */}

        {!result && !error && (
          <section className="empty-state">

            <div className="empty-icon">
              <svg
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  d="M7 4H17C18.1 4 19 4.9 19 6V18C19 19.1 18.1 20 17 20H7C5.9 20 5 19.1 5 18V6C5 4.9 5.9 4 7 4Z"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                />
                <path
                  d="M8.5 8H15.5M8.5 11.5H15.5M8.5 15H12.5"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                />
              </svg>
            </div>

            <div>
              <h3>Ready for analysis</h3>
              <p>
                Enter a message above to begin threat classification.
              </p>
            </div>

          </section>
        )}


        {/* Footer */}

        <footer className="footer">
          <div className="footer-line"></div>

          <div className="footer-content">
            <span>SENTINEL AI</span>

            <span>
              Secure message intelligence
            </span>

            <span>
              v1.0.0
            </span>
          </div>
        </footer>

      </section>
    </main>
  );
}

export default App;