import { useState } from "react";


const LOGIN_EMAIL = "mandarapabasara2001@gmail.com";
const LOGIN_PASSWORD = "2001";


function Login({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");


  function handleSubmit(event) {
    event.preventDefault();

    setError("");

    if (!email.trim() || !password.trim()) {
      setError("Please enter your email and password.");
      return;
    }

    if (
      email.trim() === LOGIN_EMAIL &&
      password === LOGIN_PASSWORD
    ) {
      onLogin();
      return;
    }

    setError("Invalid email or password.");
  }


  return (
    <div style={styles.page}>
      <div style={styles.backgroundCircleOne} />
      <div style={styles.backgroundCircleTwo} />


      <div style={styles.loginCard}>
        <div style={styles.logoCircle}>
          <span style={styles.logoLeaf}>🌿</span>
        </div>


        <div style={styles.headingSection}>
          <p style={styles.systemName}>
            ML POWERED TEA FACTORY MANAGEMENT SYSTEM
          </p>

          <h1 style={styles.title}>
            Smart Tea Leaf Grading
          </h1>

          <p style={styles.description}>
            Sign in to access the tea leaf grading
            and quality control system.
          </p>
        </div>


        <form
          onSubmit={handleSubmit}
          style={styles.form}
        >
          <div style={styles.inputGroup}>
            <label
              htmlFor="email"
              style={styles.label}
            >
              Email Address
            </label>

            <input
              id="email"
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(event) => {
                setEmail(event.target.value);
                setError("");
              }}
              style={styles.input}
              autoComplete="email"
            />
          </div>


          <div style={styles.inputGroup}>
            <label
              htmlFor="password"
              style={styles.label}
            >
              Password
            </label>

            <div style={styles.passwordWrapper}>
              <input
                id="password"
                type={
                  showPassword
                    ? "text"
                    : "password"
                }
                placeholder="Enter your password"
                value={password}
                onChange={(event) => {
                  setPassword(event.target.value);
                  setError("");
                }}
                style={styles.passwordInput}
                autoComplete="current-password"
              />

              <button
                type="button"
                onClick={() =>
                  setShowPassword(
                    (current) => !current
                  )
                }
                style={styles.showButton}
              >
                {showPassword
                  ? "Hide"
                  : "Show"}
              </button>
            </div>
          </div>


          {error && (
            <div style={styles.errorBox}>
              {error}
            </div>
          )}


          <button
            type="submit"
            style={styles.loginButton}
          >
            Sign In
          </button>
        </form>


        <div style={styles.footerLine}>
          <span style={styles.statusDot} />

          Tea Leaf Grading & Quality Control
        </div>
      </div>
    </div>
  );
}


const styles = {
  page: {
    minHeight: "100vh",
    width: "100%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
    overflow: "hidden",
    padding: "24px",
    background:
      "linear-gradient(135deg, #0f2f1c 0%, #1d5b31 50%, #347c48 100%)",
    fontFamily:
      'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  },

  backgroundCircleOne: {
    position: "absolute",
    width: "420px",
    height: "420px",
    borderRadius: "50%",
    background:
      "rgba(255, 255, 255, 0.05)",
    top: "-140px",
    right: "-100px",
  },

  backgroundCircleTwo: {
    position: "absolute",
    width: "320px",
    height: "320px",
    borderRadius: "50%",
    background:
      "rgba(255, 255, 255, 0.04)",
    bottom: "-120px",
    left: "-80px",
  },

  loginCard: {
    width: "100%",
    maxWidth: "460px",
    position: "relative",
    zIndex: 1,
    background: "#ffffff",
    borderRadius: "24px",
    padding: "38px",
    boxShadow:
      "0 28px 70px rgba(0, 0, 0, 0.25)",
  },

  logoCircle: {
    width: "72px",
    height: "72px",
    margin: "0 auto 22px",
    borderRadius: "50%",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background:
      "linear-gradient(135deg, #1f6b38, #4a9b5d)",
    boxShadow:
      "0 12px 24px rgba(31, 107, 56, 0.25)",
  },

  logoLeaf: {
    fontSize: "34px",
  },

  headingSection: {
    textAlign: "center",
    marginBottom: "30px",
  },

  systemName: {
    margin: "0 0 10px",
    color: "#38804a",
    fontSize: "11px",
    lineHeight: "1.5",
    fontWeight: "800",
    letterSpacing: "0.1em",
  },

  title: {
    margin: "0",
    color: "#173d24",
    fontSize: "30px",
    lineHeight: "1.2",
  },

  description: {
    margin: "12px auto 0",
    maxWidth: "340px",
    color: "#718078",
    fontSize: "14px",
    lineHeight: "1.6",
  },

  form: {
    display: "flex",
    flexDirection: "column",
    gap: "20px",
  },

  inputGroup: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },

  label: {
    color: "#31453a",
    fontSize: "14px",
    fontWeight: "700",
  },

  input: {
    width: "100%",
    height: "50px",
    padding: "0 15px",
    border: "1px solid #cedbd1",
    borderRadius: "12px",
    outline: "none",
    background: "#f9fbf9",
    color: "#1f2a22",
    fontSize: "15px",
  },

  passwordWrapper: {
    display: "flex",
    alignItems: "center",
    width: "100%",
    border: "1px solid #cedbd1",
    borderRadius: "12px",
    background: "#f9fbf9",
    overflow: "hidden",
  },

  passwordInput: {
    flex: 1,
    height: "50px",
    minWidth: 0,
    border: "none",
    outline: "none",
    padding: "0 15px",
    background: "transparent",
    color: "#1f2a22",
    fontSize: "15px",
  },

  showButton: {
    height: "50px",
    padding: "0 16px",
    border: "none",
    background: "transparent",
    color: "#317444",
    fontWeight: "700",
    cursor: "pointer",
  },

  errorBox: {
    padding: "12px 14px",
    borderRadius: "10px",
    border: "1px solid #efc6c6",
    background: "#fff2f2",
    color: "#a43f3f",
    fontSize: "14px",
    fontWeight: "600",
  },

  loginButton: {
    width: "100%",
    height: "52px",
    marginTop: "2px",
    border: "none",
    borderRadius: "12px",
    background:
      "linear-gradient(135deg, #1d6534 0%, #3b914e 100%)",
    color: "#ffffff",
    fontSize: "15px",
    fontWeight: "800",
    cursor: "pointer",
    boxShadow:
      "0 10px 22px rgba(31, 107, 56, 0.22)",
  },

  footerLine: {
    marginTop: "28px",
    paddingTop: "20px",
    borderTop: "1px solid #e5ebe6",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    gap: "8px",
    color: "#728077",
    fontSize: "12px",
    fontWeight: "600",
  },

  statusDot: {
    width: "8px",
    height: "8px",
    borderRadius: "50%",
    background: "#48a35e",
  },
};


export default Login;