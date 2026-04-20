# AI-Motor-Evaluation

**A framework for evaluating AI performance on infant motor behavior coding from video.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://ranan122.github.io/AI-Motor-Evaluation/)

---

## Overview

This repository contains code and documentation for a structured evaluation framework assessing large language model performance on the analysis of infant motor behavior from video. The framework is designed to be general and extensible: while the initial benchmark domain is infant sitting behavior, the evaluation structure can be adapted to other motor behaviors and developmental outcomes.

The work has two goals. First, it produces rigorous methodology for applying AI tools to behavioral coding tasks that are currently done by human researchers. Second, it lays the groundwork for a broader research program using AI to support developmental science at scale.

---

## Evaluation Framework

The framework organizes tasks into two tiers:

### Tier 1: Coding Tasks
Tasks that replicate work traditionally done by human coders.

| Task | Description |
|------|-------------|
| Posture identification | Classify infant body posture from video frames into discrete categories (0–3) |
| Temporal estimation | Estimate the duration of each posture within a video clip |
| Sitter categorization | Classify infants as independent sitters, emergent sitters, or non-sitters based on posture distributions |

### Tier 2: Inference Tasks
Tasks that require reasoning beyond explicit coding.

| Task | Description |
|------|-------------|
| Age estimation | Estimate the infant's age in months from observable motor and physical cues |
| Sex identification | Identify the infant's sex from video when contextual cues are available |

---

## Sitter Classification Rules

Infants are classified into one of three categories based on the proportion of codable postural time (postures 0–3 only; transitional codes F and N are excluded from the denominator):

- **Independent sitter**: posture 3 accounts for ≥ 50% of codable time
- **Emergent sitter**: postures 2 + 3 combined account for ≥ 50% of codable time, but posture 3 alone accounts for < 50%
- **Non-sitter**: postures 0 + 1 combined account for ≥ 50% of codable time

Categories are mutually exclusive and applied in this priority order.

---

## Repository Structure

```
AI-Motor-Evaluation/
├── LICENSE
├── README.md
├── docs/                          # GitHub Pages landing page
│   └── index.html
├── protocol/                      # Coding manual and documentation
│   └── [paste your coding manual here]
├── scripts/                       # Python analysis scripts
│   └── [paste your scripts here]
└── CITATION.cff                   # Citation metadata
```

---

## Getting Started

### Requirements

- Python 3.8 or higher
- Dependencies listed in `scripts/requirements.txt` (if applicable)

### Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/ranan122/AI-Motor-Evaluation.git
   cd AI-Motor-Evaluation
   ```

2. Review the coding protocol in `protocol/` before running any analysis.

3. Run analysis scripts from the `scripts/` directory:
   ```bash
   python scripts/your_script_name.py
   ```

---

## Documentation

Full documentation is available at:
**[https://ranan122.github.io/AI-Motor-Evaluation/](https://ranan122.github.io/AI-Motor-Evaluation/)**

---

## Citation

If you use this framework in your research, please cite:

```
An, R., Libertus, K., Wang, Q., Guo, X., Chung, R.*. (in preparation). Evaluating AI as a Research Assistant in Developmental Science: A Tiered Framework Applied to Video-Based Infant Behavioral Coding.
https://github.com/ranan122/AI-Motor-Evaluation
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Ran An**  
GitHub: [@ranan122](https://github.com/ranan122)
