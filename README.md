# 🎯 Multi-Armed Bandits for Recommendation Systems

## 📌 Overview

This project implements a range of **Multi-Armed Bandit (MAB)** algorithms for recommendation systems, covering both **classical** and **contextual** approaches. The goal is to simulate and evaluate how different bandit strategies perform in a recommendation setting, such as news article recommendation, where decisions are sequential and feedback is limited.

---

## 🧠 Background

Multi-Armed Bandits (MABs) are a framework for **sequential decision-making under uncertainty**, balancing:

- **Exploration** — trying new actions to gather information  
- **Exploitation** — using known information to maximize reward  

### 💡 Application in Recommendation Systems

- Each **arm** represents an item (e.g., a news article)
- At each step, one item is recommended
- Feedback (click or no click) is observed only for the chosen item

This leads to a **partial feedback problem**, making MABs highly suitable.

### 🧩 Contextual Bandits

Contextual bandits extend MABs by incorporating:

- **Context**: user features + item features  
- **Reward**: binary (1 = click, 0 = no click)  

The objective is to maximize cumulative reward, i.e., **click-through rate (CTR)**.

---

## 📂 Dataset

The dataset `dataset.txt` contains **10,000 user interaction events**. Each instance includes:

| Column | Description |
|--------|------------|
| 1 | Arm selected by a uniformly random policy (1–10) |
| 2 | Observed reward (1 = click, 0 = no click) |
| 3–102 | Context vector (100 features) |

### 🔍 Context Structure

- 10 arms × 10 features each = 100 features  
- Ordered sequentially:
  - Features 1–10 → Arm 1  
  - Features 11–20 → Arm 2  
  - ...  
  - Features 91–100 → Arm 10  

---

## ⚙️ Implemented Algorithms

### 🟢 Basic MAB Algorithms

- **ε-Greedy**
- **Upper Confidence Bound (UCB)**

### 🔵 Contextual Bandit Algorithms

- **LinUCB** (with evaluation and hyperparameter tuning)  
- **TreeBootstrap**  
- **KernelUCB**  

---

## 📊 Evaluation

The project uses **Off-Policy Evaluation (OPE)**:

- Evaluates performance using logged data  
- Only considers rounds where chosen actions match logged actions  
- Provides an **unbiased estimate** of algorithm performance  

---

