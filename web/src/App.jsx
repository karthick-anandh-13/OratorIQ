import React, { useEffect, useMemo, useRef, useState } from 'react';

const API_BASE =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8001';

/* =========================================================
   DATA HELPERS
========================================================= */

function num(value, fallback = 0) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  const parsed = Number(value);

  return Number.isFinite(parsed) ? parsed : fallback;
}

function getConfidence(value) {
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : 0;
  }

  if (typeof value === 'string') {
    return num(value);
  }

  if (value && typeof value === 'object') {
    if (value.overall_confidence !== undefined) {
      return num(value.overall_confidence);
    }

    if (value.confidence !== undefined) {
      return num(value.confidence);
    }

    if (value.overall_confidence_percent !== undefined) {
      return num(value.overall_confidence_percent) / 100;
    }
  }

  return 0;
}

function pct(value) {
  return `${(getConfidence(value) * 100).toFixed(1)}%`;
}

function seconds(value) {
  return `${num(value).toFixed(2)}s`;
}

function text(value) {
  if (value === null || value === undefined) {
    return '—';
  }

  if (typeof value === 'object') {
    return '—';
  }

  return String(value)
    .replaceAll('_', ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function rawText(value) {
  if (
    value === null ||
    value === undefined ||
    typeof value === 'object'
  ) {
    return '';
  }

  return String(value);
}

function severityName(value) {
  return String(value || '').toLowerCase();
}

/* =========================================================
   ICONS
========================================================= */

function ArrowIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
    >
      <path d="M5 12h13" />
      <path d="m13 6 6 6-6 6" />
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
    >
      <path d="M12 16V4" />
      <path d="m7 9 5-5 5 5" />
      <path d="M4 20h16" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="currentColor"
    >
      <path d="M8 5.5v13L19 12 8 5.5Z" />
    </svg>
  );
}

/* =========================================================
   APP
========================================================= */

function App() {
  const [dashboard, setDashboard] = useState(null);
  const [status, setStatus] = useState('Connecting');
  const [error, setError] = useState('');

  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);

  const inputRef = useRef(null);
  const audioRef = useRef(null);

  /* =======================================================
     API CONNECTION
  ======================================================= */

  useEffect(() => {
    let mounted = true;

    async function loadDashboard() {
      try {
        const response = await fetch(
          `${API_BASE}/api/dashboard`,
        );

        if (!response.ok) {
          throw new Error(
            `API returned ${response.status}`,
          );
        }

        const data = await response.json();

        if (mounted) {
          setDashboard(data);
          setStatus('Connected');
        }
      } catch {
        if (mounted) {
          setStatus('Offline');
        }
      }
    }

    loadDashboard();

    return () => {
      mounted = false;
    };
  }, []);

  /* =======================================================
     FILE
  ======================================================= */

  function handleFile(event) {
    const file = event.target.files?.[0];

    if (!file) return;

    setSelectedFile(file);
    setResult(null);
    setError('');
  }

  /* =======================================================
     ANALYZE
  ======================================================= */

  async function analyzeSpeech() {
    if (!selectedFile) {
      setError('Choose a recording before starting analysis.');
      return;
    }

    try {
      setAnalyzing(true);
      setError('');
      setResult(null);

      const formData = new FormData();

      formData.append('audio', selectedFile);

      const response = await fetch(
        `${API_BASE}/api/analyze/auto`,
        {
          method: 'POST',
          body: formData,
        },
      );

      let data;

      try {
        data = await response.json();
      } catch {
        throw new Error(
          'The analysis server returned an invalid response.',
        );
      }

      if (!response.ok) {
        throw new Error(
          rawText(data?.detail) ||
            `Analysis failed with status ${response.status}`,
        );
      }

      setResult(data);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setAnalyzing(false);
    }
  }

  /* =======================================================
     RESET
  ======================================================= */

  function clearAnalysis() {
    setSelectedFile(null);
    setResult(null);
    setError('');

    if (inputRef.current) {
      inputRef.current.value = '';
    }

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.removeAttribute('src');
      audioRef.current.load();
    }
  }

  /* =======================================================
     RESULT NORMALIZATION
  ======================================================= */

  const isStandalone = result?.analysis_mode === 'standalone';

  const regions = Array.isArray(
    result?.timeline?.regions,
  )
    ? result.timeline.regions
    : Array.isArray(result?.flaws_detected)
      ? result.flaws_detected
      : [];

  const windows = Array.isArray(
    result?.timeline?.windows,
  )
    ? result.timeline.windows
    : [];

  const primaryRegion =
    regions.length > 0 ? regions[0] : null;

  const detected =
    Boolean(
      result?.analysis?.detected ??
        regions.length > 0,
    );

  const overallConfidence = getConfidence(
    result?.timeline?.overall_confidence ??
      result?.analysis?.confidence ??
      result?.confidence ??
      0,
  );

  const explanation =
    primaryRegion?.explanation &&
    typeof primaryRegion.explanation === 'object'
      ? primaryRegion.explanation
      : result?.explanation &&
          typeof result.explanation === 'object'
        ? result.explanation
        : null;

  const evidence =
    explanation?.evidence_values &&
    typeof explanation.evidence_values === 'object'
      ? explanation.evidence_values
      : {};

  const detectedWindows = useMemo(
    () =>
      windows.filter(
        (window) =>
          window?.status === 'detected' ||
          window?.predicted_label === 'flawed',
      ),
    [windows],
  );

  /* =======================================================
     FLUENCY / STUTTER RESULT
     Added without changing the existing delivery pipeline.
  ======================================================= */

  const fluency = result?.fluency &&
    typeof result.fluency === 'object'
    ? result.fluency
    : null;

  const fluencyEvents =
    fluency?.events &&
    typeof fluency.events === 'object'
      ? Object.entries(fluency.events)
      : [];

  const detectedFluencyEvents = fluencyEvents.filter(
    ([, event]) => Boolean(event?.detected),
  );

  const fluencyProbability = getConfidence(
    fluency?.stutter_probability ??
      fluency?.stutter_probability_percent
        ? fluency?.stutter_probability_percent !== undefined
          ? num(fluency.stutter_probability_percent) / 100
          : fluency.stutter_probability
        : 0,
  );

  const fluencyWarning = rawText(fluency?.warning);

  /* RAG coaching is optional: the existing analysis UI works without it. */
  const coaching =
    result?.coaching && typeof result.coaching === 'object'
      ? result.coaching
      : null;

  const coachingObservations = Array.isArray(coaching?.observations)
    ? coaching.observations
    : [];

  const coachingRecommendations = Array.isArray(coaching?.recommendations)
    ? coaching.recommendations
    : [];

  const coachingExercises = Array.isArray(coaching?.exercises)
    ? coaching.exercises
    : [];

  const coachingSources = Array.isArray(coaching?.sources)
    ? coaching.sources
    : [];


  /* =======================================================
     RENDER
  ======================================================= */

  return (
    <div className="site">

      <style>{`

        /* ===================================================
           DESIGN SYSTEM
        =================================================== */

        :root {
          --paper: #f6f5ef;
          --paper-2: #efeee8;
          --white: #ffffff;

          --ink: #111111;
          --ink-soft: #555552;
          --ink-faint: #8b8a84;

          --line: #d7d5ce;
          --line-dark: #b9b7af;

          --blue: #4267ff;
          --violet: #8b5cf6;
          --coral: #ff6b5f;
          --lime: #b9dc45;
          --yellow: #f3cf54;

          --display:
            "Hortes",
            "Arial Narrow",
            "Helvetica Neue",
            Arial,
            sans-serif;

          --body:
            Inter,
            "Helvetica Neue",
            Helvetica,
            Arial,
            sans-serif;
        }

        * {
          box-sizing: border-box;
        }

        html {
          background: var(--paper);
        }

        body {
          margin: 0;

          background: var(--paper);

          color: var(--ink);

          font-family: var(--body);

          -webkit-font-smoothing: antialiased;
        }

        button,
        input {
          font: inherit;
        }

        button {
          border: 0;
        }

        /* ===================================================
           GLOBAL
        =================================================== */

        .site {
          min-height: 100vh;

          background:
            radial-gradient(
              circle at 8% 12%,
              rgba(66, 103, 255, 0.16),
              transparent 24%
            ),
            radial-gradient(
              circle at 91% 7%,
              rgba(139, 92, 246, 0.13),
              transparent 22%
            ),
            radial-gradient(
              circle at 75% 76%,
              rgba(185, 220, 69, 0.11),
              transparent 22%
            ),
            radial-gradient(
              circle at 20% 88%,
              rgba(255, 107, 95, 0.10),
              transparent 20%
            ),
            var(--paper);

          position: relative;

          overflow-x: hidden;
        }

        .site::before {
          content: "";

          position: fixed;

          inset: 0;

          pointer-events: none;

          opacity: 0.32;

          background-image:
            linear-gradient(
              rgba(0,0,0,0.025) 1px,
              transparent 1px
            ),
            linear-gradient(
              90deg,
              rgba(0,0,0,0.025) 1px,
              transparent 1px
            );

          background-size: 48px 48px;

          mask-image:
            linear-gradient(
              to bottom,
              black,
              transparent 75%
            );
        }

        .container {
          width: min(
            1400px,
            calc(100% - 64px)
          );

          margin: 0 auto;

          position: relative;

          z-index: 1;
        }

        /* ===================================================
           NAVIGATION
        =================================================== */

        .nav {
          height: 74px;

          border-bottom:
            1px solid rgba(17,17,17,0.12);

          display: flex;

          align-items: center;

          background:
            rgba(246,245,239,0.78);

          backdrop-filter:
            blur(18px);

          position: sticky;

          top: 0;

          z-index: 50;
        }

        .nav-inner {
          width: min(
            1400px,
            calc(100% - 64px)
          );

          margin: auto;

          display: flex;

          align-items: center;

          justify-content: space-between;
        }

        .logo {
          display: flex;

          align-items: center;

          gap: 10px;

          font-size: 14px;

          font-weight: 800;

          letter-spacing: -0.03em;
        }

        .logo-symbol {
          width: 25px;

          height: 25px;

          border-radius: 50%;

          background:
            conic-gradient(
              from 210deg,
              var(--blue),
              var(--violet),
              var(--coral),
              var(--yellow),
              var(--blue)
            );

          position: relative;
        }

        .logo-symbol::after {
          content: "";

          position: absolute;

          width: 9px;
          height: 9px;

          background: var(--paper);

          border-radius: 50%;

          left: 8px;
          top: 8px;
        }

        .nav-center {
          display: flex;

          align-items: center;

          gap: 34px;

          margin-left: 70px;

          margin-right: auto;
        }

        .nav-item {
          color: var(--ink-faint);

          font-size: 12px;

          font-weight: 600;

          letter-spacing: -0.01em;
        }

        .nav-item.active {
          color: var(--ink);
        }

        .nav-right {
          display: flex;

          align-items: center;

          gap: 9px;

          color: var(--ink-soft);

          font-size: 11px;
        }

        .online-dot {
          width: 7px;

          height: 7px;

          border-radius: 50%;

          background: var(--lime);

          box-shadow:
            0 0 0 3px rgba(185,220,69,0.16);
        }

        .online-dot.offline {
          background: var(--coral);
        }

        /* ===================================================
           HERO
        =================================================== */

        .hero {
          padding: 100px 0 72px;

          position: relative;
        }

        .hero-grid {
          display: grid;

          grid-template-columns:
            1.45fr
            0.55fr;

          gap: 70px;

          align-items: end;
        }

        .hero-kicker {
          font-size: 11px;

          font-weight: 800;

          letter-spacing: 0.15em;

          text-transform: uppercase;

          color: var(--ink-soft);

          margin-bottom: 22px;
        }

        .hero-title {
          margin: 0;

          font-family: var(--display);

          font-size:
            clamp(72px, 11vw, 158px);

          font-weight: 900;

          line-height: 0.82;

          letter-spacing: -0.055em;

          text-transform: uppercase;
        }

        .hero-title .gradient {
          display: block;

          background:
            linear-gradient(
              105deg,
              var(--blue) 5%,
              var(--violet) 34%,
              var(--coral) 67%,
              var(--yellow) 96%
            );

          -webkit-background-clip: text;

          background-clip: text;

          color: transparent;
        }

        .hero-copy {
          max-width: 430px;

          padding-bottom: 7px;
        }

        .hero-copy p {
          margin: 0;

          color: var(--ink-soft);

          font-size: 15px;

          line-height: 1.7;

          letter-spacing: -0.01em;
        }

        .hero-line {
          width: 100%;

          height: 1px;

          margin-top: 62px;

          background:
            linear-gradient(
              90deg,
              var(--blue),
              var(--violet),
              var(--coral),
              transparent
            );
        }

        /* ===================================================
           ANALYSIS AREA
        =================================================== */

        .section-label {
          font-size: 10px;

          font-weight: 800;

          letter-spacing: 0.14em;

          text-transform: uppercase;

          color: var(--ink-soft);

          margin-bottom: 15px;
        }

        .analysis-layout {
          display: grid;

          grid-template-columns:
            1.7fr
            0.7fr;

          gap: 20px;

          margin-bottom: 20px;
        }

        .upload {
          min-height: 320px;

          border:
            1px solid rgba(17,17,17,0.16);

          background:
            rgba(255,255,255,0.55);

          position: relative;

          overflow: hidden;

          padding: 28px;
        }

        .upload::before {
          content: "";

          position: absolute;

          width: 280px;
          height: 280px;

          right: -100px;
          bottom: -140px;

          border-radius: 50%;

          background:
            radial-gradient(
              circle,
              rgba(66,103,255,0.17),
              transparent 68%
            );
        }

        .upload-head {
          display: flex;

          justify-content: space-between;

          align-items: center;

          position: relative;

          z-index: 2;
        }

        .upload-title {
          font-family: var(--display);

          font-size: 28px;

          text-transform: uppercase;

          letter-spacing: -0.02em;
        }

        .formats {
          color: var(--ink-faint);

          font-size: 10px;

          letter-spacing: 0.1em;

          text-transform: uppercase;
        }

        .drop {
          min-height: 160px;

          margin-top: 28px;

          border:
            1px dashed var(--line-dark);

          display: flex;

          align-items: center;

          justify-content: center;

          text-align: center;

          cursor: pointer;

          position: relative;

          z-index: 2;

          background:
            rgba(246,245,239,0.48);

          transition:
            border-color 160ms ease,
            background 160ms ease;
        }

        .drop:hover {
          border-color: var(--blue);

          background:
            rgba(255,255,255,0.72);
        }

        .drop-icon {
          width: 42px;

          height: 42px;

          margin: 0 auto 13px;

          display: grid;

          place-items: center;

          border:
            1px solid var(--line-dark);

          border-radius: 50%;

          color: var(--ink-soft);
        }

        .drop-title {
          font-size: 14px;

          font-weight: 700;

          letter-spacing: -0.02em;
        }

        .drop-subtitle {
          color: var(--ink-faint);

          font-size: 11px;

          margin-top: 6px;
        }

        .selected {
          color: var(--blue);

          font-size: 11px;

          margin-top: 8px;

          max-width: 500px;

          white-space: nowrap;

          overflow: hidden;

          text-overflow: ellipsis;
        }

        .upload-footer {
          display: flex;

          justify-content: flex-end;

          gap: 8px;

          margin-top: 18px;

          position: relative;

          z-index: 2;
        }

        .button {
          height: 42px;

          padding: 0 17px;

          border: 1px solid var(--ink);

          background: var(--ink);

          color: white;

          font-size: 11px;

          font-weight: 700;

          cursor: pointer;

          display: inline-flex;

          align-items: center;

          gap: 8px;

          transition: 150ms ease;
        }

        .button:hover:not(:disabled) {
          transform: translateY(-1px);

          box-shadow:
            0 7px 18px rgba(0,0,0,0.13);
        }

        .button.light {
          color: var(--ink);

          background:
            transparent;

          border-color:
            var(--line-dark);
        }

        .button:disabled {
          opacity: 0.38;

          cursor: not-allowed;

          transform: none;
        }

        /* ===================================================
           INFO PANEL
        =================================================== */

        .info-panel {
          border:
            1px solid rgba(17,17,17,0.16);

          background: var(--ink);

          color: white;

          padding: 25px;

          min-height: 320px;

          position: relative;

          overflow: hidden;
        }

        .info-panel::before {
          content: "";

          position: absolute;

          width: 240px;

          height: 240px;

          right: -100px;

          top: -100px;

          border-radius: 50%;

          background:
            radial-gradient(
              circle,
              rgba(139,92,246,0.7),
              transparent 65%
            );
        }

        .info-panel::after {
          content: "";

          position: absolute;

          width: 180px;

          height: 180px;

          left: -90px;

          bottom: -100px;

          border-radius: 50%;

          background:
            radial-gradient(
              circle,
              rgba(66,103,255,0.55),
              transparent 65%
            );
        }

        .info-heading {
          font-family: var(--display);

          text-transform: uppercase;

          font-size: 24px;

          letter-spacing: -0.01em;

          position: relative;

          z-index: 2;
        }

        .info-list {
          margin-top: 34px;

          position: relative;

          z-index: 2;
        }

        .info-row {
          display: flex;

          justify-content: space-between;

          align-items: center;

          padding: 14px 0;

          border-bottom:
            1px solid rgba(255,255,255,0.13);

          font-size: 11px;
        }

        .info-row:last-child {
          border-bottom: 0;
        }

        .info-key {
          color: rgba(255,255,255,0.55);
        }

        .info-value {
          font-weight: 700;
        }

        /* ===================================================
           ERROR
        =================================================== */

        .error {
          border:
            1px solid rgba(255,107,95,0.35);

          background:
            rgba(255,107,95,0.08);

          color: #b93f36;

          padding: 13px 16px;

          margin-bottom: 20px;

          font-size: 12px;
        }

        /* ===================================================
           RESULT HEADER
        =================================================== */

        .result-header {
          margin-top: 80px;

          padding-top: 18px;

          border-top:
            1px solid var(--ink);

          display: flex;

          justify-content: space-between;

          align-items: flex-end;

          gap: 30px;
        }

        .result-kicker {
          font-size: 10px;

          font-weight: 800;

          letter-spacing: 0.14em;

          text-transform: uppercase;

          margin-bottom: 10px;
        }

        .result-title {
          margin: 0;

          font-family: var(--display);

          font-size: 54px;

          line-height: 0.9;

          text-transform: uppercase;

          letter-spacing: -0.03em;
        }

        .result-file {
          color: var(--ink-soft);

          font-size: 11px;

          max-width: 300px;

          text-align: right;
        }

        /* ===================================================
           METRICS
        =================================================== */

        .metrics {
          display: grid;

          grid-template-columns:
            1.35fr
            1fr
            1fr
            1fr;

          margin-top: 25px;

          border-top:
            1px solid var(--ink);

          border-bottom:
            1px solid var(--ink);
        }

        .metric {
          min-height: 145px;

          padding: 24px 20px;

          border-right:
            1px solid var(--line);
        }

        .metric:last-child {
          border-right: 0;
        }

        .metric-label {
          color: var(--ink-faint);

          font-size: 9px;

          font-weight: 800;

          letter-spacing: 0.12em;

          text-transform: uppercase;
        }

        .metric-value {
          margin-top: 22px;

          font-family: var(--display);

          font-size: 31px;

          text-transform: uppercase;

          letter-spacing: -0.025em;

          line-height: 0.95;
        }

        .metric-value.gradient {
          background:
            linear-gradient(
              100deg,
              var(--blue),
              var(--violet),
              var(--coral)
            );

          -webkit-background-clip: text;

          background-clip: text;

          color: transparent;
        }

        .metric-note {
          margin-top: 9px;

          color: var(--ink-faint);

          font-size: 10px;
        }

        /* ===================================================
           ANALYSIS PANELS
        =================================================== */

        .results-grid {
          display: grid;

          grid-template-columns:
            1.55fr
            0.75fr;

          gap: 20px;

          margin-top: 20px;
        }

        .panel {
          border:
            1px solid rgba(17,17,17,0.15);

          background:
            rgba(255,255,255,0.48);
        }

        .panel-head {
          min-height: 58px;

          padding: 0 20px;

          border-bottom:
            1px solid var(--line);

          display: flex;

          justify-content: space-between;

          align-items: center;
        }

        .panel-title {
          font-size: 11px;

          font-weight: 800;

          letter-spacing: 0.09em;

          text-transform: uppercase;
        }

        .panel-meta {
          color: var(--ink-faint);

          font-size: 10px;
        }

        /* ===================================================
           TIMELINE
        =================================================== */

        .timeline {
          padding: 28px 20px 24px;
        }

        .timeline-scale {
          display: flex;

          justify-content: space-between;

          color: var(--ink-faint);

          font-size: 9px;

          margin-bottom: 7px;
        }

        .timeline-track {
          display: flex;

          gap: 3px;

          height: 100px;
        }

        .window {
          flex: 1;

          min-width: 0;

          position: relative;

          background: #e9e8e2;

          border:
            1px solid #d8d6ce;

          overflow: hidden;

          cursor: default;
        }

        .window::after {
          content: "";

          position: absolute;

          left: 0;
          right: 0;

          bottom: 0;

          height: 4px;

          background: #d3d1ca;
        }

        .window.detected {
          background:
            linear-gradient(
              145deg,
              rgba(255,107,95,0.2),
              rgba(139,92,246,0.15),
              rgba(66,103,255,0.15)
            );

          border-color:
            rgba(139,92,246,0.45);
        }

        .window.detected::after {
          background:
            linear-gradient(
              90deg,
              var(--coral),
              var(--violet),
              var(--blue)
            );
        }

        .window-prob {
          position: absolute;

          left: 8px;

          top: 8px;

          font-size: 9px;

          font-weight: 700;

          color: var(--ink-soft);
        }

        .window-time {
          position: absolute;

          left: 8px;

          bottom: 9px;

          font-size: 8px;

          color: var(--ink-faint);
        }

        .region {
          margin-top: 20px;

          padding: 16px;

          border:
            1px solid var(--ink);

          background: white;

          display: flex;

          align-items: center;

          justify-content: space-between;

          gap: 20px;
        }

        .region-main {
          display: flex;

          align-items: center;

          gap: 13px;
        }

        .region-dot {
          width: 9px;

          height: 9px;

          border-radius: 50%;

          background:
            linear-gradient(
              135deg,
              var(--coral),
              var(--violet)
            );
        }

        .region-name {
          font-family: var(--display);

          font-size: 20px;

          text-transform: uppercase;

          letter-spacing: -0.015em;
        }

        .region-time {
          margin-top: 4px;

          color: var(--ink-faint);

          font-size: 10px;
        }

        .severity {
          font-size: 9px;

          font-weight: 800;

          text-transform: uppercase;

          letter-spacing: 0.08em;

          padding: 7px 9px;

          border: 1px solid var(--line-dark);
        }

        .severity.medium {
          border-color: #d8bf58;

          background: #fff9d8;
        }

        .severity.bad,
        .severity.extreme {
          border-color: #e28d84;

          background: #fff0ee;
        }

        .severity.slight {
          border-color: #91a7ff;

          background: #eef2ff;
        }

        /* ===================================================
           EVIDENCE
        =================================================== */

        .evidence {
          padding: 22px;
        }

        .evidence-intro {
          color: var(--ink-soft);

          font-size: 12px;

          line-height: 1.65;

          margin: 0 0 23px;
        }

        .evidence-row {
          display: flex;

          justify-content: space-between;

          align-items: baseline;

          padding: 12px 0;

          border-bottom:
            1px solid var(--line);
        }

        .evidence-row:last-child {
          border-bottom: 0;
        }

        .evidence-key {
          color: var(--ink-faint);

          font-size: 10px;
        }

        .evidence-value {
          color: var(--ink);

          font-size: 11px;

          font-weight: 800;

          font-variant-numeric:
            tabular-nums;
        }

        /* ===================================================
           FLUENCY / STUTTER
        =================================================== */

        .fluency-section {
          margin-top: 20px;
          border:
            1px solid rgba(17,17,17,0.15);
          background: white;
          overflow: hidden;
        }

        .fluency-head {
          min-height: 64px;
          padding: 0 20px;
          border-bottom: 1px solid var(--line);
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 20px;
        }

        .fluency-title-wrap {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .fluency-mark {
          width: 10px;
          height: 10px;
          border-radius: 50%;
          background:
            linear-gradient(
              135deg,
              var(--blue),
              var(--violet),
              var(--coral)
            );
          box-shadow:
            0 0 0 5px rgba(139,92,246,0.10);
        }

        .fluency-title {
          font-family: var(--display);
          font-size: 24px;
          text-transform: uppercase;
          letter-spacing: -0.015em;
        }

        .fluency-model {
          color: var(--ink-faint);
          font-size: 10px;
          text-transform: uppercase;
          letter-spacing: 0.09em;
        }

        .fluency-body {
          padding: 22px;
        }

        .fluency-overview {
          display: grid;
          grid-template-columns: 0.8fr 1.2fr;
          gap: 18px;
        }

        .fluency-score {
          min-height: 150px;
          padding: 22px;
          border: 1px solid var(--line);
          background:
            linear-gradient(
              135deg,
              rgba(66,103,255,0.08),
              rgba(139,92,246,0.08),
              rgba(255,107,95,0.07)
            );
        }

        .fluency-score-label {
          color: var(--ink-faint);
          font-size: 9px;
          font-weight: 800;
          letter-spacing: 0.12em;
          text-transform: uppercase;
        }

        .fluency-score-value {
          margin-top: 17px;
          font-family: var(--display);
          font-size: 48px;
          line-height: 0.9;
          letter-spacing: -0.035em;
          background:
            linear-gradient(
              100deg,
              var(--blue),
              var(--violet),
              var(--coral)
            );
          -webkit-background-clip: text;
          background-clip: text;
          color: transparent;
        }

        .fluency-score-note {
          margin-top: 10px;
          color: var(--ink-soft);
          font-size: 10px;
        }

        .fluency-events {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 10px;
        }

        .fluency-event {
          padding: 16px;
          border: 1px solid var(--line);
          background: var(--paper);
          min-height: 70px;
        }

        .fluency-event.detected {
          border-color: rgba(139,92,246,0.45);
          background:
            linear-gradient(
              135deg,
              rgba(139,92,246,0.10),
              rgba(255,107,95,0.07)
            );
        }

        .fluency-event-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
        }

        .fluency-event-name {
          font-size: 11px;
          font-weight: 800;
        }

        .fluency-event-state {
          font-size: 8px;
          font-weight: 800;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--ink-faint);
        }

        .fluency-event.detected .fluency-event-state {
          color: #6d45c5;
        }

        .fluency-event-prob {
          margin-top: 10px;
          font-family: var(--display);
          font-size: 22px;
          letter-spacing: -0.02em;
        }

        .fluency-bar {
          height: 4px;
          margin-top: 9px;
          background: #e2e0d8;
          overflow: hidden;
        }

        .fluency-bar > span {
          display: block;
          height: 100%;
          background:
            linear-gradient(
              90deg,
              var(--blue),
              var(--violet),
              var(--coral)
            );
        }

        .fluency-warning {
          margin-top: 16px;
          padding: 11px 13px;
          border: 1px solid rgba(243,207,84,0.7);
          background: #fff9d8;
          color: #6d5b16;
          font-size: 10px;
          line-height: 1.55;
        }

        .fluency-footnote {
          margin-top: 14px;
          color: var(--ink-faint);
          font-size: 10px;
          line-height: 1.55;
        }

        @media (max-width: 1000px) {
          .fluency-overview {
            grid-template-columns: 1fr;
          }
        }

        @media (max-width: 650px) {
          .fluency-events {
            grid-template-columns: 1fr;
          }

          .fluency-head {
            align-items: flex-start;
            flex-direction: column;
            padding: 16px 20px;
          }
        }

        /* ===================================================
           RAG COACHING
           Isolated styles; does not alter existing modules.
        =================================================== */

        .coaching-section {
          margin-top: 20px;
          border: 1px solid rgba(17,17,17,0.15);
          background: rgba(255,255,255,0.78);
          overflow: hidden;
        }

        .coaching-head {
          min-height: 64px;
          padding: 16px 20px;
          border-bottom: 1px solid var(--line);
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
        }

        .coaching-title-wrap {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .coaching-mark {
          width: 10px;
          height: 10px;
          flex: 0 0 10px;
          border-radius: 50%;
          background: linear-gradient(135deg, var(--lime), var(--blue), var(--violet));
          box-shadow: 0 0 0 5px rgba(66,103,255,0.08);
        }

        .coaching-title {
          font-family: var(--display);
          font-size: 24px;
          letter-spacing: -0.015em;
          text-transform: uppercase;
        }

        .coaching-badge {
          border: 1px solid var(--line-dark);
          padding: 7px 9px;
          color: var(--ink-soft);
          font-size: 9px;
          font-weight: 800;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          white-space: nowrap;
        }

        .coaching-body {
          padding: 22px;
        }

        .coaching-summary {
          margin: 0 0 20px;
          color: var(--ink-soft);
          font-size: 13px;
          line-height: 1.7;
        }

        .coaching-grid {
          display: grid;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          gap: 14px;
        }

        .coaching-card {
          min-width: 0;
          padding: 18px;
          border: 1px solid var(--line);
          background: var(--paper);
        }

        .coaching-card-title {
          margin-bottom: 12px;
          font-size: 10px;
          font-weight: 800;
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }

        .coaching-list {
          margin: 0;
          padding-left: 18px;
          color: var(--ink-soft);
          font-size: 12px;
          line-height: 1.75;
        }

        .coaching-list li + li {
          margin-top: 8px;
        }

        .coaching-empty {
          color: var(--ink-faint);
          font-size: 12px;
          line-height: 1.6;
        }

        .coaching-sources {
          display: flex;
          flex-wrap: wrap;
          gap: 7px;
          margin-top: 18px;
        }

        .coaching-source {
          border: 1px solid var(--line);
          background: white;
          padding: 6px 8px;
          color: var(--ink-soft);
          font-size: 9px;
        }

        .coaching-disclaimer {
          margin-top: 18px;
          padding-top: 13px;
          border-top: 1px solid var(--line);
          color: var(--ink-faint);
          font-size: 10px;
          line-height: 1.6;
        }

        @media (max-width: 650px) {
          .coaching-head {
            align-items: flex-start;
            flex-direction: column;
          }

          .coaching-grid {
            grid-template-columns: 1fr;
          }

          .coaching-body {
            padding: 16px;
          }
        }

        /* ===================================================
           TRANSCRIPT
        =================================================== */

        .transcript {
          margin-top: 20px;

          border:
            1px solid rgba(17,17,17,0.15);

          background: white;
        }

        .transcript-body {
          padding: 30px;

          font-family:
            Georgia,
            "Times New Roman",
            serif;

          font-size: 20px;

          line-height: 1.65;

          letter-spacing: -0.015em;
        }

        .transcript-footer {
          border-top:
            1px solid var(--line);

          padding: 12px 20px;

          color: var(--ink-faint);

          font-size: 9px;

          text-transform: uppercase;

          letter-spacing: 0.08em;
        }

        /* ===================================================
           EXPLANATION
        =================================================== */

        .notes {
          margin-top: 20px;

          display: grid;

          grid-template-columns:
            1fr 1fr;

          border:
            1px solid rgba(17,17,17,0.15);

          background: white;
        }

        .note {
          padding: 25px;

          min-height: 160px;

          border-right:
            1px solid var(--line);
        }

        .note:last-child {
          border-right: 0;
        }

        .note-label {
          font-size: 9px;

          font-weight: 800;

          text-transform: uppercase;

          letter-spacing: 0.12em;

          color: var(--ink-faint);

          margin-bottom: 15px;
        }

        .note-text {
          margin: 0;

          color: var(--ink-soft);

          font-size: 13px;

          line-height: 1.7;
        }

        .note-text strong {
          color: var(--ink);
        }

        /* ===================================================
           AUDIO
        =================================================== */

        .audio {
          width: 100%;

          margin-top: 20px;

          height: 42px;
        }

        /* ===================================================
           EMPTY
        =================================================== */

        .empty {
          margin-top: 20px;

          min-height: 310px;

          border:
            1px solid rgba(17,17,17,0.15);

          background: white;

          display: flex;

          align-items: center;

          justify-content: center;

          text-align: center;
        }

        .empty-inner {
          max-width: 480px;

          padding: 30px;
        }

        .empty-symbol {
          width: 48px;

          height: 48px;

          margin: 0 auto 20px;

          display: grid;

          place-items: center;

          border-radius: 50%;

          background:
            linear-gradient(
              135deg,
              var(--blue),
              var(--violet),
              var(--coral)
            );

          color: white;
        }

        .empty-title {
          font-family: var(--display);

          font-size: 27px;

          text-transform: uppercase;

          letter-spacing: -0.02em;

          margin: 0;
        }

        .empty-text {
          margin: 10px 0 0;

          color: var(--ink-faint);

          font-size: 12px;

          line-height: 1.65;
        }

        /* ===================================================
           LOADING
        =================================================== */

        .loading {
          animation:
            breathe 1.5s ease-in-out infinite;
        }

        @keyframes breathe {
          50% {
            opacity: 0.48;
          }
        }

        /* ===================================================
           MOBILE
        =================================================== */

        @media (max-width: 1000px) {

          .hero-grid,
          .analysis-layout,
          .results-grid {
            grid-template-columns: 1fr;
          }

          .hero-title {
            font-size:
              clamp(65px, 16vw, 120px);
          }

          .metrics {
            grid-template-columns:
              repeat(2, 1fr);
          }

          .metric:nth-child(2) {
            border-right: 0;
          }

          .metric:nth-child(-n+2) {
            border-bottom:
              1px solid var(--line);
          }

          .notes {
            grid-template-columns: 1fr;
          }

          .note {
            border-right: 0;

            border-bottom:
              1px solid var(--line);
          }
        }

        @media (max-width: 650px) {

          .container,
          .nav-inner {
            width:
              calc(100% - 30px);
          }

          .nav {
            height: 62px;
          }

          .nav-center {
            display: none;
          }

          .hero {
            padding-top: 65px;
          }

          .hero-title {
            font-size: 66px;
          }

          .hero-grid {
            gap: 35px;
          }

          .metrics {
            grid-template-columns: 1fr;
          }

          .metric {
            border-right: 0;

            border-bottom:
              1px solid var(--line);
          }

          .metric:last-child {
            border-bottom: 0;
          }

          .result-header {
            display: block;
          }

          .result-file {
            margin-top: 12px;

            text-align: left;
          }

          .result-title {
            font-size: 42px;
          }

          .region {
            align-items: flex-start;

            flex-direction: column;
          }

          .transcript-body {
            font-size: 17px;
          }
        }

      `}</style>

      {/* =====================================================
          NAV
      ===================================================== */}

      <nav className="nav">

        <div className="nav-inner">

          <div className="logo">

            <div className="logo-symbol" />

            OratorIQ

          </div>

          <div className="nav-center">

            <div className="nav-item active">
              Analysis
            </div>

            <div className="nav-item">
              Dataset
            </div>

            <div className="nav-item">
              Evaluation
            </div>

          </div>

          <div className="nav-right">

            <span
              className={`online-dot ${
                status === 'Offline'
                  ? 'offline'
                  : ''
              }`}
            />

            {status}

          </div>

        </div>

      </nav>

      {/* =====================================================
          PAGE
      ===================================================== */}

      <div className="container">

        {/* HERO */}

        <section className="hero">

          <div className="hero-grid">

            <div>

              <div className="hero-kicker">
                Speech intelligence / 01
              </div>

              <h1 className="hero-title">

                Hear

                <span className="gradient">
                  better.
                </span>

              </h1>

            </div>

            <div className="hero-copy">

              <p>
                OratorIQ compares recordings when
                a matching clean reference is available,
                or uses standalone fluency analysis
                when it is not.
              </p>

            </div>

          </div>

          <div className="hero-line" />

        </section>

        {/* ERROR */}

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        {/* =================================================
            UPLOAD
        ================================================= */}

        <section className="analysis-layout">

          <div className="upload">

            <div className="upload-head">

              <div className="upload-title">
                New analysis
              </div>

              <div className="formats">
                WAV / FLAC / MP3 / OGG
              </div>

            </div>

            <label className="drop">

              <input
                ref={inputRef}
                type="file"
                accept="audio/wav,audio/flac,audio/mpeg,audio/ogg,.wav,.flac,.mp3,.ogg"
                hidden
                onChange={handleFile}
              />

              <div>

                <div className="drop-icon">
                  <UploadIcon />
                </div>

                <div className="drop-title">

                  {selectedFile
                    ? 'Recording ready'
                    : 'Choose a recording'}

                </div>

                <div className="drop-subtitle">

                  {selectedFile
                    ? 'Will compare if a matching reference exists; otherwise runs standalone'
                    : 'Drag a file here or click to browse'}

                </div>

                {selectedFile && (
                  <div className="selected">
                    {selectedFile.name}
                  </div>
                )}

              </div>

            </label>

            <div className="upload-footer">

              <button
                className="button light"
                onClick={clearAnalysis}
                disabled={analyzing}
              >
                Clear
              </button>

              <button
                className="button"
                onClick={analyzeSpeech}
                disabled={
                  !selectedFile ||
                  analyzing
                }
              >

                {analyzing
                  ? 'Analyzing…'
                  : 'Run analysis'}

                {!analyzing && (
                  <ArrowIcon />
                )}

              </button>

            </div>

          </div>

          {/* SYSTEM */}

          <aside className="info-panel">

            <div className="info-heading">
              Engine status
            </div>

            <div className="info-list">

              <div className="info-row">

                <span className="info-key">
                  Pipeline
                </span>

                <span className="info-value">
                  Active
                </span>

              </div>

              <div className="info-row">

                <span className="info-key">
                  Temporal windows
                </span>

                <span className="info-value">
                  1.0s / 0.5s
                </span>

              </div>

              <div className="info-row">

                <span className="info-key">
                  Contrastive model
                </span>

                <span className="info-value">
                  Online
                </span>

              </div>

              <div className="info-row">

                <span className="info-key">
                  Dataset
                </span>

                <span className="info-value">
                  {dashboard?.entries ?? '—'}
                </span>

              </div>

            </div>

          </aside>

        </section>

        {/* =================================================
            RESULTS
        ================================================= */}

        {result && (

          <>

            <section className="result-header">

              <div>

                <div className="result-kicker">
                  {isStandalone
                    ? 'Standalone Analysis'
                    : 'Contrastive Analysis'}
                  {' / '}
                  {rawText(result.processing_status) || 'completed'}
                </div>

                <h2 className="result-title">
                  Speech report
                </h2>

              </div>

              <div className="result-file">
                {rawText(
                  result.audio_name,
                )}
              </div>

            </section>

            {/* METRICS */}

            <section className="metrics">

              <div className="metric">

                <div className="metric-label">
                  {isStandalone
                    ? 'Contrastive confidence'
                    : 'Confidence'}
                </div>

                <div className="metric-value gradient">
                  {isStandalone
                    ? 'Unavailable'
                    : pct(overallConfidence)}
                </div>

                <div className="metric-note">
                  {isStandalone
                    ? 'Requires a clean reference'
                    : 'Overall model confidence'}
                </div>

              </div>

              <div className="metric">

                <div className="metric-label">
                  Finding
                </div>

                <div className="metric-value">

                  {isStandalone
                    ? 'Not compared'
                    : detected
                    ? text(
                        primaryRegion?.flaw_type,
                      )
                    : 'Normal'}

                </div>

                <div className="metric-note">

                  {isStandalone
                    ? 'No reference comparison performed'
                    : primaryRegion
                    ? `${seconds(
                        primaryRegion.start,
                      )} — ${seconds(
                        primaryRegion.end,
                      )}`
                    : 'No grounded anomaly'}

                </div>

              </div>

              <div className="metric">

                <div className="metric-label">
                  Severity
                </div>

                <div className="metric-value">

                  {isStandalone
                    ? 'Unavailable without reference'
                    : primaryRegion
                    ? text(
                        primaryRegion.severity,
                      )
                    : 'Normal'}

                </div>

                <div className="metric-note">

                  {isStandalone
                    ? 'Reference-dependent metric'
                    : primaryRegion
                    ? text(
                        primaryRegion.family,
                      )
                    : 'Reference aligned'}

                </div>

              </div>

              <div className="metric">

                <div className="metric-label">
                  Recording
                </div>

                <div className="metric-value">
                  {seconds(
                    result.duration,
                  )}
                </div>

                <div className="metric-note">

                  {num(
                    result.sample_rate,
                  )}{' '}
                  Hz

                </div>

              </div>

            </section>

            {/* TIMELINE + EVIDENCE */}

            <section className="results-grid">

              <div className="panel">

                <div className="panel-head">

                  <div className="panel-title">
                    Temporal map
                  </div>

                  <div className="panel-meta">
                    {detectedWindows.length}{' '}
                    flagged windows
                  </div>

                </div>

                <div className="timeline">

                  {windows.length ? (

                    <>

                      <div className="timeline-scale">

                        <span>
                          0.00s
                        </span>

                        <span>
                          {seconds(
                            result.duration,
                          )}
                        </span>

                      </div>

                      <div className="timeline-track">

                        {windows.map(
                          (window, index) => {

                            const isDetected =
                              window?.status ===
                                'detected' ||
                              window?.predicted_label ===
                                'flawed';

                            const probability =
                              getConfidence(
                                window?.flawed_probability ??
                                  window?.probability ??
                                  window?.confidence ??
                                  0,
                              );

                            return (

                              <div
                                className={`window ${
                                  isDetected
                                    ? 'detected'
                                    : ''
                                }`}
                                key={`${window?.start}-${window?.end}-${index}`}
                                title={`${seconds(
                                  window?.start,
                                )} — ${seconds(
                                  window?.end,
                                )}`}
                              >

                                <span className="window-prob">
                                  {pct(
                                    probability,
                                  )}
                                </span>

                                <span className="window-time">
                                  {seconds(
                                    window?.start,
                                  )}
                                </span>

                              </div>

                            );

                          },
                        )}

                      </div>

                      {regions.map(
                        (region, index) => (

                          <div
                            className="region"
                            key={`${region?.start}-${region?.end}-${index}`}
                          >

                            <div className="region-main">

                              <span className="region-dot" />

                              <div>

                                <div className="region-name">

                                  {text(
                                    region?.flaw_type,
                                  )}

                                </div>

                                <div className="region-time">

                                  {seconds(
                                    region?.start,
                                  )}

                                  {' — '}

                                  {seconds(
                                    region?.end,
                                  )}

                                  {' · '}

                                  {pct(
                                    getConfidence(
                                      region?.confidence,
                                    ),
                                  )}

                                </div>

                              </div>

                            </div>

                            <div
                              className={`severity ${severityName(
                                region?.severity,
                              )}`}
                            >
                              {text(
                                region?.severity,
                              )}
                            </div>

                          </div>

                        ),
                      )}

                    </>

                  ) : (

                    <div
                      style={{
                        padding:
                          '55px 0',
                        textAlign:
                          'center',
                        color:
                          'var(--ink-faint)',
                        fontSize:
                          '11px',
                      }}
                    >
                      {isStandalone
                        ? 'V4 reports recording-level predictions only; event timestamps are unavailable.'
                        : 'No temporal data available.'}
                    </div>

                  )}

                </div>

              </div>

              {/* EVIDENCE */}

              <div className="panel">

                <div className="panel-head">

                  <div className="panel-title">
                    Signal evidence
                  </div>

                  <div className="panel-meta">
                    {isStandalone
                      ? 'Unavailable without reference'
                      : 'Reference comparison'}
                  </div>

                </div>

                <div className="evidence">

                  <p className="evidence-intro">

                    {isStandalone
                      ? rawText(result?.comparison?.reason)
                      : rawText(explanation?.summary) ||
                      'The model did not generate a textual explanation for this recording.'}

                  </p>

                  <EvidenceRow
                    label="Speaking rate"
                    value={
                      isStandalone
                        ? 'Unavailable without reference'
                        :
                      evidence.wpm_delta !==
                      undefined
                        ? `${num(
                            evidence.wpm_delta,
                          ).toFixed(
                            2,
                          )} WPM`
                        : '—'
                    }
                  />

                  <EvidenceRow
                    label="Rate ratio"
                    value={
                      isStandalone
                        ? 'Unavailable without reference'
                        :
                      evidence.wpm_ratio !==
                      undefined
                        ? `${num(
                            evidence.wpm_ratio,
                          ).toFixed(
                            3,
                          )}×`
                        : '—'
                    }
                  />

                  <EvidenceRow
                    label="Pitch delta"
                    value={
                      isStandalone
                        ? 'Unavailable without reference'
                        :
                      evidence.f0_delta !==
                      undefined
                        ? `${num(
                            evidence.f0_delta,
                          ).toFixed(
                            2,
                          )} Hz`
                        : '—'
                    }
                  />

                  <EvidenceRow
                    label="RMS delta"
                    value={
                      isStandalone
                        ? 'Unavailable without reference'
                        :
                      evidence.rms_delta !==
                      undefined
                        ? num(
                            evidence.rms_delta,
                          ).toFixed(4)
                        : '—'
                    }
                  />

                  <EvidenceRow
                    label="Silence delta"
                    value={
                      isStandalone
                        ? 'Unavailable without reference'
                        :
                      evidence.silence_delta !==
                      undefined
                        ? num(
                            evidence.silence_delta,
                          ).toFixed(4)
                        : '—'
                    }
                  />

                  <EvidenceRow
                    label="Alignment"
                    value={
                      isStandalone
                        ? 'Unavailable without reference'
                        :
                      evidence.alignment_error !==
                      undefined
                        ? num(
                            evidence.alignment_error,
                          ).toFixed(4)
                        : '—'
                    }
                  />

                </div>

              </div>

            </section>

            {/* =================================================
                FLUENCY / STUTTER
            ================================================= */}

            {fluency?.available && (
              <section className="fluency-section">

                <div className="fluency-head">

                  <div className="fluency-title-wrap">
                    <span className="fluency-mark" />

                    <div className="fluency-title">
                      Fluency analysis
                    </div>
                  </div>

                  <div className="fluency-model">
                    {rawText(fluency.model) || 'AI fluency model'}
                  </div>

                </div>

                <div className="fluency-body">

                  <div className="fluency-overview">

                    <div className="fluency-score">

                      <div className="fluency-score-label">
                        Stutter probability
                      </div>

                      <div className="fluency-score-value">
                        {pct(fluencyProbability)}
                      </div>

                      <div className="fluency-score-note">
                        {fluency?.detected
                          ? 'Possible fluency difficulty detected'
                          : 'No strong fluency signal detected'}
                      </div>

                    </div>

                    <div className="fluency-events">

                      {fluencyEvents.length ? (
                        fluencyEvents.map(([name, event]) => {
                          const probability = getConfidence(
                            event?.probability ??
                              event?.probability_percent !== undefined
                              ? event?.probability_percent !== undefined
                                ? num(event.probability_percent) / 100
                                : event?.probability
                              : 0,
                          );

                          const isDetected = Boolean(event?.detected);

                          return (
                            <div
                              className={`fluency-event ${
                                isDetected ? 'detected' : ''
                              }`}
                              key={name}
                            >

                              <div className="fluency-event-top">

                                <span className="fluency-event-name">
                                  {text(name)}
                                </span>

                                <span className="fluency-event-state">
                                  {isDetected
                                    ? 'Possible'
                                    : 'Not detected'}
                                </span>

                              </div>

                              <div className="fluency-event-prob">
                                {pct(probability)}
                              </div>

                              <div className="fluency-bar">
                                <span
                                  style={{
                                    width: `${Math.max(
                                      0,
                                      Math.min(
                                        100,
                                        probability * 100,
                                      ),
                                    )}%`,
                                  }}
                                />
                              </div>

                            </div>
                          );
                        })
                      ) : (
                        <div className="fluency-event">
                          No event-level fluency data available.
                        </div>
                      )}

                    </div>

                  </div>

                  {detectedFluencyEvents.length > 0 && (
                    <div className="fluency-footnote">
                      Possible events:{' '}
                      {detectedFluencyEvents
                        .map(([name]) => text(name))
                        .join(' · ')}
                    </div>
                  )}

                  {fluencyWarning && (
                    <div className="fluency-warning">
                      <strong>Demo / research note:</strong>{' '}
                      {fluencyWarning}
                    </div>
                  )}

                  <div className="fluency-footnote">
                    This section is an AI-assisted fluency signal, not a
                    medical diagnosis.
                  </div>

                  {isStandalone && (
                    <div className="fluency-footnote">
                      {rawText(result?.audio_validation?.message)}
                    </div>
                  )}

                </div>

              </section>
            )}

            {/* =================================================
                RAG COACHING
                Rendered only when the backend includes coaching.
                Existing delivery and fluency results are unchanged.
            ================================================= */}

            {coaching && coaching.status !== 'unavailable' && (
              <section className="coaching-section">
                <div className="coaching-head">
                  <div className="coaching-title-wrap">
                    <span className="coaching-mark" />
                    <div className="coaching-title">Personalised coaching</div>
                  </div>
                  <span className="coaching-badge">
                    {coaching.status === 'ok'
                      ? 'Knowledge-backed guidance'
                      : 'Coaching'}
                  </span>
                </div>

                <div className="coaching-body">
                  {rawText(coaching.summary) && (
                    <p className="coaching-summary">{rawText(coaching.summary)}</p>
                  )}

                  <div className="coaching-grid">
                    <div className="coaching-card">
                      <div className="coaching-card-title">Observations</div>
                      {coachingObservations.length > 0 ? (
                        <ul className="coaching-list">
                          {coachingObservations.map((item, index) => {
                            const label = typeof item === 'string'
                              ? item
                              : rawText(item?.message) || rawText(item?.finding) ||
                                rawText(item?.label) || rawText(item?.type);
                            return label ? <li key={`observation-${index}`}>{label}</li> : null;
                          })}
                        </ul>
                      ) : (
                        <div className="coaching-empty">No additional observations were returned.</div>
                      )}
                    </div>

                    <div className="coaching-card">
                      <div className="coaching-card-title">Recommendations</div>
                      {coachingRecommendations.length > 0 ? (
                        <ul className="coaching-list">
                          {coachingRecommendations.map((item, index) => {
                            const label = typeof item === 'string'
                              ? item
                              : rawText(item?.text) || rawText(item?.recommendation) ||
                                rawText(item?.message);
                            return label ? <li key={`recommendation-${index}`}>{label}</li> : null;
                          })}
                        </ul>
                      ) : (
                        <div className="coaching-empty">No recommendations were returned for this recording.</div>
                      )}
                    </div>

                    <div className="coaching-card">
                      <div className="coaching-card-title">Practice exercises</div>
                      {coachingExercises.length > 0 ? (
                        <ul className="coaching-list">
                          {coachingExercises.map((item, index) => {
                            const label = typeof item === 'string'
                              ? item
                              : rawText(item?.text) || rawText(item?.exercise) ||
                                rawText(item?.name) || rawText(item?.instructions);
                            return label ? <li key={`exercise-${index}`}>{label}</li> : null;
                          })}
                        </ul>
                      ) : (
                        <div className="coaching-empty">No practice exercise was returned.</div>
                      )}
                    </div>

                    <div className="coaching-card">
                      <div className="coaching-card-title">Evidence notes</div>
                      <div className="coaching-empty">
                        Coaching supplements the acoustic analysis. Model observations are estimates,
                        not a diagnosis, and should be interpreted alongside the evidence above.
                      </div>
                    </div>
                  </div>

                  {coachingSources.length > 0 && (
                    <div className="coaching-sources" aria-label="Coaching sources">
                      {coachingSources.map((source, index) => {
                        const label = typeof source === 'string'
                          ? source
                          : rawText(source?.title) || rawText(source?.source) ||
                            rawText(source?.file) || rawText(source?.name);
                        return label ? (
                          <span className="coaching-source" key={`source-${index}`}>
                            Source · {label}
                          </span>
                        ) : null;
                      })}
                    </div>
                  )}

                  {rawText(coaching.disclaimer) && (
                    <div className="coaching-disclaimer">{rawText(coaching.disclaimer)}</div>
                  )}
                </div>
              </section>
            )}

            {/* AUDIO */}

            {result.audio_url && (

              <audio
                ref={audioRef}
                className="audio"
                controls
                src={`${API_BASE}${result.audio_url}`}
              />

            )}

            {/* TRANSCRIPT */}

            <section className="transcript">

              <div className="panel-head">

                <div className="panel-title">
                  Transcript
                </div>

                <div className="panel-meta">
                  Source recording
                </div>

              </div>

              <div className="transcript-body">

                {rawText(
                  result.transcript,
                ) ||
                  'Transcript unavailable.'}

              </div>

              <div className="transcript-footer">

                {seconds(
                  result.duration,
                )}

                {' · '}

                {num(
                  result.sample_rate,
                )}{' '}
                Hz

                {' · '}

                {num(
                  result.channels,
                )}{' '}
                channel

              </div>

            </section>

            {/* NOTES */}

            {explanation && (

              <section className="notes">

                <div className="note">

                  <div className="note-label">
                    Interpretation
                  </div>

                  <p className="note-text">

                    {rawText(
                      explanation.summary,
                    ) ||
                      'The model identified a measurable deviation from the reference recording.'}

                  </p>

                </div>

                <div className="note">

                  <div className="note-label">
                    Recommendation
                  </div>

                  <p className="note-text">

                    {rawText(
                      explanation.recommendation,
                    ) ||
                      'No recommendation was generated.'}

                  </p>

                </div>

              </section>

            )}

          </>

        )}

        {/* =================================================
            EMPTY STATE
        ================================================= */}

        {!result &&
          !analyzing && (

            <section className="empty">

              <div className="empty-inner">

                <div className="empty-symbol">
                  <PlayIcon />
                </div>

                <h2 className="empty-title">
                  Your voice,
                  <br />
                  measured precisely.
                </h2>

                <p className="empty-text">
                  Upload a recording to inspect
                  pacing, pauses, volume, pitch,
                  and other measurable speech
                  deviations.
                </p>

              </div>

            </section>

          )}

        {/* =================================================
            LOADING
        ================================================= */}

        {analyzing && (

          <section className="empty loading">

            <div className="empty-inner">

              <div className="empty-symbol">
                <ArrowIcon />
              </div>

              <h2 className="empty-title">
                Reading the recording.
              </h2>

              <p className="empty-text">
                Running the supported speech analysis
                and preparing the report.
              </p>

            </div>

          </section>

        )}

      </div>

    </div>
  );
}

/* =========================================================
   EVIDENCE ROW
========================================================= */

function EvidenceRow({ label, value }) {
  return (
    <div className="evidence-row">

      <span className="evidence-key">
        {label}
      </span>

      <span className="evidence-value">
        {String(value)}
      </span>

    </div>
  );
}

export default App;