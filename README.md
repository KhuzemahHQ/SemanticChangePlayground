# ChronoSense | Diachronic Semantic Change Studio

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A scientific toolkit and interactive research studio to measure lexical semantic change of words over history using Google Books Ngrams, Stanford HistWords diachronic embeddings, SemEval benchmarks, and custom historical text corpora.

---

## 🌟 Key Features

1. **Multi-Corpora Historical Frequency Kinematics (Google Books Ngrams 1800–2019)**:
   - Live query integration across Google Books corpora (`en-2019`, `en-US-2019`, `en-GB-2019`, `en-fiction-2019`).
   - Annual & decadal relative frequency normalization, growth rates, peak year detection, and trajectory pattern classification (e.g. Modern Exponential Surge, Historical Archaisation, U-Shaped Resurgence).

2. **Diachronic Vector Spaces & Kinematics (1800s–1990s)**:
   - Full integration with Stanford **HistWords** (Hamilton, Leskovec, Jurafsky 2016) Skip-Gram with Negative Sampling (SGNS) decadal embeddings.
   - Built-in, zero-latency curated embedding archive (`data/curated_historical.npz`) spanning 20 decades for 2,500+ historical words and landmarks.
   - Orthogonal Procrustes alignment ($W = U V^T$ where $M_1^T M_2 = U \Sigma V^T$).

3. **Rigorous Linguistic Metrics**:
   - **Global Semantic Displacement**: Cosine distance $\text{Disp}(w, t_1, t_2) = 1 - \cos(v_w^{(t_1)}, v_w^{(t_2)})$.
   - **Semantic Velocity & Acceleration**: Rate of semantic movement per decade: $V(t) = \frac{\Delta d}{\Delta t}$, $A(t) = \frac{\Delta V}{\Delta t}$.
   - **Neighborhood Turnover (Jaccard Distance)**: Top-$k$ nearest neighbors replacement rate: $1 - \frac{|NN_k(t_1) \cap NN_k(t_2)|}{|NN_k(t_1) \cup NN_k(t_2)|}$.
   - **Second-Order Contextual Drift (CDCS)**: Alignment-free drift against invariant domain anchor lexicons.
   - **Shift Typology Classification**: Automated classification into *Amelioration*, *Pejoration*, *Broadening (Generalization)*, *Narrowing (Specialization)*, and *Domain / Technological Transfer*.

4. **Statistical Laws of Semantic Change**:
   - Replicates and confirms **Hamilton's Law of Conformity**: Rate of semantic change scales inversely with word frequency ($r \propto f^{-\alpha}$, Spearman $\rho = -0.2003$, $p < 10^{-5}$).

5. **SemEval-2020 Task 1 Evaluation Suite**:
   - Evaluates diachronic embeddings against official human gold-standard annotations (Schlechtweg et al., 2020) on continuous graded change (Spearman $\rho$) and binary change detection (Accuracy, Precision, Recall, F1).

6. **Custom Diachronic Corpus Lab**:
   - Train and measure semantic change directly from user-supplied historical texts (e.g. Period 1 vs. Period 2 text) using sliding-window co-occurrence, PPMI matrix computation, Truncated SVD, and Orthogonal Procrustes alignment.

7. **Interactive Web Studio & Visual Time Machine**:
   - Glassmorphic dark-mode laboratory dashboard (`http://localhost:8000`).
   - Interactive Chart.js frequency curves and cumulative drift curves.
   - 2D Canvas Semantic Time Machine with temporal scrubbing and playback animation.

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/KhuzemahHQ/SemanticChangePlayground.git
cd SemanticChangePlayground
pip install -r requirements.txt
```

### 2. Launch the Web Studio
```bash
python -m semantic_change.cli serve --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 💻 Command-Line Interface (CLI)

### Measure Semantic Change for a Single Word
```bash
python -m semantic_change.cli measure gay
python -m semantic_change.cli measure mouse
python -m semantic_change.cli measure apple
python -m semantic_change.cli measure plane
```

**Example Output:**
```text
====================================================================
  DIACHRONIC SEMANTIC ANALYSIS: 'GAY'
====================================================================

[Documented Historical Benchmark]
  Shift Era   : 1940s–1970s
  Shift Type  : Metaphorical / Socio-cultural Drift
  1800s Sense : Lighthearted, carefree, cheerful, brilliant, brightly colored
  Modern Sense: Homosexual; LGBTQ+ identity and community
  Context     : From the 12th to the early 20th century, 'gay' primarily denoted
                cheerfulness, carefree exuberance, and vividness...

[Google Books Ngrams Historical Frequency]
  Corpus      : English (2019)
  Peak Year   : 1803 (Relative Freq: 2.38e-05)
  Trajectory  : Fluctuating Dynamic (Peak 1803)

[Vector Semantic Displacement & Kinematics]
  Total Cosine Displacement : 0.8816 (1800s -> 1990s)
  Peak Velocity Decade      : 1870s (Rate: 0.5169/dec)
  Mean Neighborhood Turnover: 0.7965
  Second-Order Anchor Drift : 0.8680
  Shift Classification      : Pejoration (Confidence: 0.825)

[Decadal Cumulative Drift (Distance from Baseline)]
  Decade   | Cumulative Drift   | Velocity   | Turnover  
  ---------+--------------------+------------+-----------
  1830     | 0.0000            | 0.0000     | 0.0000
  1850     | 0.3308 ########   | 0.3356     | 0.8571
  1900     | 0.6730 ################ | 0.3739 | 0.5455
  1950     | 0.6720 ################ | 0.4147 | 0.9333
  1970     | 0.6693 ################ | 0.4061 | 0.5455
  1990     | 0.8816 ###################### | 0.3664 | 0.7692

[Semantic Neighborhood Timeline (Top Nearest Historical Neighbors)]
  1850s: brilliant (0.76), lively (0.76), graceful (0.69), cheerful (0.69)
  1900s: lively (0.45), cheery (0.44), lark (0.41), jovial (0.40)
  1950s: jovial (0.42), cheerful (0.39), lively (0.35)
  1990s: lesbian (0.68), lesbians (0.55), bisexual (0.53), gays (0.43)
```

### Compare Multiple Words
```bash
python -m semantic_change.cli compare gay apple cell plane
```

### Run SemEval-2020 Task 1 Benchmark Evaluation
```bash
python -m semantic_change.cli semeval
```

### Test Hamilton's Law of Conformity
```bash
python -m semantic_change.cli laws
```

---

## 🐍 Python API Usage

```python
from semantic_change import DiachronicAnalyzer, CustomDiachronicCorpusAnalyzer

# Initialize analyzer
analyzer = DiachronicAnalyzer()

# 1. Analyze single word
result = analyzer.analyze_word("mouse")
sm = result["semantic_measurement"]
print(f"Mouse displacement: {sm['total_displacement']}")
print(f"Shift type: {sm['classification']['primary_category']}")

# 2. Compare words across history
comp = analyzer.compare_words(["apple", "computer", "fruit"])
print("Pairwise similarities over decades:", comp["pairwise_similarities"])

# 3. Project 2D coordinates
traj = analyzer.get_trajectory_2d(["apple", "mouse", "technology"])
print("2D Path for apple:", traj["trajectories"]["apple"])

# 4. Measure shift in custom texts
corpus_analyzer = CustomDiachronicCorpusAnalyzer()
shift_results = corpus_analyzer.analyze_two_corpora(
    corpus_t1="Your 19th-century text corpus here...",
    corpus_t2="Your 21st-century text corpus here...",
    target_words=["communication", "wire", "battery"]
)
```

---

## 📊 Historical Shift Case Studies Included

| Word | Historical Era | 1800s Sense | Modern Sense | Shift Typology |
| :--- | :--- | :--- | :--- | :--- |
| **`gay`** | 1940s–1970s | Lighthearted, cheerful, vivid | Homosexual; LGBTQ+ identity | Domain Shift |
| **`mouse`** | 1970s–1990s | Small rodent pest | Computer pointing device | Metaphorical Transfer |
| **`apple`** | 1970s–2000s | Tree fruit, orchard harvest | Computing & digital ecosystem | Brand Metonymy |
| **`cell`** | 1980s–2000s | Monastic chamber, biology unit | Mobile cellular phone | Specialization |
| **`broadcast`** | 1910s–1930s | Sowing seeds across a field | Radio & TV transmission | Domain Transfer |
| **`plane`** | 1900s–1930s | Flat surface, carpenter's tool | Aeroplane, aviation aircraft | Domain Shift |
| **`awful`** | 1820s–1900s | Inspiring profound awe & majesty | Extremely bad or unpleasant | Pejoration |
| **`nice`** | 1700s–1900s | Fastidious, precise, delicate | Pleasant, agreeable, kind | Amelioration |
| **`terrific`** | 1900s–1950s | Terrifying, frightful | Magnificent, splendid | Amelioration |
| **`cloud`** | 1990s–2010s | Sky water vapor, condensation | Distributed server hosting | Technological Extension |
| **`virus`** | 1980s–2000s | Biological toxin, pathogen | Self-replicating malware | Metaphorical Extension |
| **`meat`** | Middle English–1900s | Solid food in general | Animal flesh specifically | Narrowing |

---

## 🔬 Scientific References

* Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). *Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change*. **ACL 2016**, [arXiv:1605.09096](https://arxiv.org/abs/1605.09096).
* Schlechtweg, D., et al. (2020). *SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection*. **SemEval 2020**, [ACL Anthology](https://aclanthology.org/2020.semeval-1.1/).
* Michel, J. B., et al. (2011). *Quantitative Analysis of Culture Using Millions of Digitized Books*. **Science**, 331(6014), 176-182.
* Kutuzov, A., et al. (2018). *Diachronic word embeddings and semantic shifts: a survey*. **COLING 2018**.

---

## 📄 License
MIT License. Created for the Semantic Change Research Community.