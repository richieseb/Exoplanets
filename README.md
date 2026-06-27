# ☕ Team Chai to Code: Automated Exoplanet Transit Detection Pipeline

An end-to-end automated astronomical data science pipeline designed to detect faint exoplanet transit signals from noisy stellar light curves. Built as a high-impact solution for Space Situational Awareness and Deep Space Exploration (ISRO Hackathon Challenge).

---

## 👥 About the Team: Chai to Code

We are a group of data science and engineering minds who literally convert tea into optimized algorithms. Combining strong mathematical foundations with deep learning, we specialize in stripping noise from complex datasets to build robust, production-ready AI pipelines. Powered by chai, we turn intricate space data into elegant code.

---

## 📌 Problem Statement & Challenges

The objective is to identify **exoplanet transits**—the minuscule periodic drops in a star's brightness caused by an orbiting planet passing in front of it. In real-world mission profiles, extracting these signals faces severe technical bottlenecks:

* **Extreme Class Imbalance:** Confirmed planets represent less than 1% of the observed stellar population in raw datasets, causing standard classifiers to default to "No Planet".
* **Low Signal-to-Noise Ratio (SNR):** Transit dips are extremely shallow (often $<1\%$ flux reduction) and easily buried under stellar noise.
* **Stellar & Instrumental Drift:** Low-frequency trends caused by spacecraft jitter or stellar rotation mask high-frequency transit events.
* **Astrophysical Confounders:** Eclipsing binaries mimic planetary transits but produce distinct geometric profiles.

---

## 🚀 Solution Architecture

The **Chai to Code** engine approaches this problem through a structured, 4-stage pipeline:
