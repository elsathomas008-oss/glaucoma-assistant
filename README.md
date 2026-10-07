# 👁️ Glaucoma Second Opinion

**Explainable multimodal medical image intelligence for glaucoma screening and clinician decision support.**

![Status](https://img.shields.io/badge/status-hackathon%20prototype-orange)
![AI](https://img.shields.io/badge/AI-Gemini%20API-blue)
![Dataset](https://img.shields.io/badge/dataset-Drishti--GS-lightgrey)
![Notes](https://img.shields.io/badge/patient%20notes-synthetic-lightgrey)

> [!WARNING]
> **Research prototype only.** This project is not a medical device and has not been clinically validated. It must not be used as a substitute for professional diagnosis or treatment. All patient notes used in the current demonstration are synthetic.

---

## 1. What the Project Does

### The problem

Glaucoma can progress silently and may cause irreversible vision loss. Fundus photographs provide visual information about the optic nerve head, while clinical information such as intraocular pressure (IOP), RNFL thickness and visual-field measurements can provide additional context.

However, a useful clinical-support system should do more than return a binary prediction. It should also show **what evidence was considered and why the result was produced**.

### Our solution

**Glaucoma Second Opinion** is a multimodal AI prototype that combines:

- a colour fundus photograph,
- structured patient information,
- visual assessment of the optic nerve region,
- evidence-grounded explanations, and
- a clinician-in-the-loop review process.

The system produces a structured second opinion containing:

1. **Image-based findings**
2. **Clinical-note findings**
3. **Supporting evidence**
4. **Highlighted image region**
5. **An output confidence/risk score**
6. **A plain-language explanation**

The clinician remains responsible for the final decision.

### Example output

```text
Finding:
Consider glaucoma-related optic nerve changes.

Image evidence:
Enlarged optic cup relative to the disc and apparent inferior
rim thinning in the highlighted optic-disc region.

Clinical evidence:
IOP reported as 28 mmHg in the provided patient notes.

Output score:
0.87
```

> The example illustrates the intended output format. It is not a clinically validated prediction.

---

# 2. Core Idea

The project uses a **task-specific three-model architecture**.

Instead of sending every operation to the most capable model, different Gemini models are assigned different jobs:

```text
FAST PREPROCESSING
       ↓
Gemini 3.5 Flash-Lite
       ↓
STRUCTURED CLINICAL DATA
       +
VALIDATED FUNDUS IMAGE
       ↓
Gemini 3.8 Flash
Primary vision assessment
       ↓
IMAGE FINDINGS
       +
CLINICAL INFORMATION
       ↓
Gemini 3.5 Flash
Reasoning + explanation
       ↓
GROUNDING / VALIDATION
       ↓
FINAL CLINICIAN-FACING RESULT
```

This design is intended to balance **latency, API cost, reasoning, vision performance and explainability**.

---

# 3. Technologies, Libraries, and Models Used

## AI models

| Model | Main role |
|---|---|
| **Gemini 3.5 Flash-Lite** | Fast and low-cost preprocessing, image-quality checks, structured extraction from patient notes and first-pass screening |
| **Gemini 3.5 Flash** | Multimodal reasoning, explanation generation and evidence-grounded findings |
| **Gemini 3.8 Flash** | Primary fundus-image assessment where stronger vision performance is prioritized |

The model selection is based on assigning each task to an appropriate capability/cost tier. The cited model-selection comparison used during development showed stronger general vision-benchmark performance for Gemini 3.8 Flash than Gemini 3.5 Flash-Lite; this should **not** be interpreted as glaucoma-specific clinical validation.

## Main technologies

| Technology | Use |
|---|---|
| **Python 3.10+** | Application/runtime environment |
| **Google Gemini API** | Multimodal AI inference |
| **`.env` configuration** | API-key configuration |
| **`requirements.txt`** | Python dependency management |
| **Web UI + backend in `app/`** | User interaction and inference workflow |
| **Drishti-GS** | Fundus-image dataset used in the prototype |
| **Synthetic patient notes** | Demonstration of multimodal clinical context |

### Libraries

The exact Python libraries should be taken from the committed `requirements.txt` file so that the documentation matches the implementation exactly.

At a minimum, the repository expects the dependencies installed through:

```bash
pip install -r requirements.txt
```

> **Do not manually substitute library names in this README unless they match `requirements.txt`.** This keeps the documentation reproducible.

---

# 4. Data Pipeline

```text
                 ┌─────────────────────┐
                 │   FUNDUS IMAGE      │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  PATIENT NOTES      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   DATA VALIDATION   │
                 │                     │
                 │ • Format            │
                 │ • Image quality     │
                 │ • Note availability │
                 └──────────┬──────────┘
                            │
                            ▼
              ┌────────────────────────────┐
              │ Gemini 3.5 Flash-Lite     │
              │ Fast preprocessing        │
              └────────────┬───────────────┘
                           │
              ┌────────────┴─────────────┐
              ▼                          ▼
     Structured patient            Validated image
     information                    ↓
                                    │
                           ┌────────▼────────┐
                           │ Gemini 3.8 Flash│
                           │ Vision analysis │
                           └────────┬────────┘
                                    │
                                    ▼
                           Image-based findings
                                    │
              ┌─────────────────────┴──────────────────┐
              │                                        │
              ▼                                        ▼
      Clinical evidence                          Image evidence
              │                                        │
              └──────────────────┬─────────────────────┘
                                 ▼
                       ┌────────────────────┐
                       │ Gemini 3.5 Flash   │
                       │ Reasoning +        │
                       │ explanation        │
                       └─────────┬──────────┘
                                 ▼
                       Grounding / validation
                                 ▼
                   Findings + evidence + score
                                 ▼
                        Clinician review
```

### Grounding rule

Every reported finding should be traceable to one of two evidence sources:

- **Image evidence:** a relevant highlighted/identified region of the fundus image.
- **Clinical evidence:** a specific item from the patient notes.

Unsupported findings should be flagged rather than presented as grounded evidence.

---

# 5. Dataset

The current prototype uses **101 Drishti-GS colour fundus images**.

| Split | Images | Glaucoma | Normal |
|---|---:|---:|---:|
| Training | 50 | 32 | 18 |
| Testing | 51 | 38 | 13 |
| **Total** | **101** | **70** | **31** |

Patient notes are synthetic and were created to be consistent with the image labels for demonstration purposes.

### Synthetic note fields

The demonstration notes can contain:

- Age
- Sex
- Eye imaged
- Presenting complaint
- Systemic/family history
- Medication
- IOP
- Central corneal thickness
- Gonioscopy
- Disc description
- Estimated vertical C/D ratio
- RNFL thickness
- Visual-field MD and PSD
- Impression and plan

> [!IMPORTANT]
> Because the notes are label-consistent synthetic data, the **image + notes** experiment can be optimistic. These results must not be interpreted as real-world clinical accuracy.

---

# 6. Installation

## Requirements

Before starting, make sure you have:

- Python **3.10 or newer**
- Internet access for Gemini API requests
- A valid Gemini API key
- The repository files
- The dataset files required by the demonstration

Check Python:

```bash
python --version
```

Expected:

```text
Python 3.10+
```

---

## Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

---

## Create a virtual environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` file is the source of truth for the exact Python libraries used by the implementation.

---

# 7. Configuration

Create a file named:

```text
.env
```

in the project root.

Add:

```env
GEMINI_API_KEY=your_key_here
```

### Security

Never commit your real API key.

Add the following to `.gitignore`:

```text
.env
```

For team/hackathon sharing, use `.env.example` without exposing a real key:

```env
GEMINI_API_KEY=your_key_here
```

---

# 8. How to Run the System

Once the dependencies and API key are configured:

```bash
python app.py
```

The application should print or expose the local web address configured by the project.

Open that address in a browser.

> The exact host/port should be taken from the application's startup output rather than hard-coded in this README.

---

# 9. How to Use the System

### Step 1 — Upload the image

Upload a supported fundus image, such as:

```text
PNG
JPG / JPEG
```

### Step 2 — Provide patient information

Paste or select the available patient notes.

### Step 3 — Start analysis

Click the analysis button used by the application.

### Step 4 — Review the result

The system returns:

- the primary finding,
- highlighted image evidence,
- clinical evidence,
- explanation,
- output score,
- and grounding information.

### Step 5 — Human review

The clinician can accept, edit or disregard the AI suggestion.

---

# 10. Reproducing the Demonstrated Results

The current demonstration is designed to be reproducible using the same dataset split, synthetic notes and model configuration.

## Reproduction checklist

### 1. Use the same data

Use the included:

```text
data/images/
data/synthetic_notes/
```

with the same train/test split described above.

### 2. Use the same model roles

Keep the same routing:

```text
Gemini 3.5 Flash-Lite
→ preprocessing / structured extraction

Gemini 3.8 Flash
→ primary fundus-image assessment

Gemini 3.5 Flash
→ reasoning / explanation / grounded findings
```

### 3. Use the same environment

Install the exact committed dependencies:

```bash
pip install -r requirements.txt
```

### 4. Configure the API

Set:

```env
GEMINI_API_KEY=your_key_here
```

### 5. Run the application

```bash
python app.py
```

### 6. Run the demonstration workflow

For each test image:

1. Load the test fundus photograph.
2. Load its corresponding synthetic patient note.
3. Run the **image-only** workflow.
4. Record the prediction and score.
5. Run the **image + notes** workflow.
6. Record the prediction and score.
7. Compare the outputs.

### 7. Report the metrics

The evaluation should include at least:

- Accuracy
- Sensitivity
- Specificity
- Grounded-finding rate

Use the same test split for both image-only and multimodal experiments.

---

# 11. Evaluation

Use the following format when filling in the final measured results:

| Metric | Image Only | Image + Notes |
|---|---:|---:|
| Accuracy | `<measured value>` | `<measured value>` |
| Sensitivity | `<measured value>` | `<measured value>` |
| Specificity | `<measured value>` | `<measured value>` |
| Grounded-finding rate | `<measured value>` | `<measured value>` |

### Important interpretation

Do **not** claim that image + notes performance demonstrates clinical superiority because the current notes were synthesized to match the labels.

The correct interpretation is:

> The multimodal experiment demonstrates the feasibility of combining image and structured clinical context in the prototype. The current synthetic-note setup is not sufficient to establish clinical performance.

---

# 12. Explainability and Grounding

A central project goal is to make the result **reviewable rather than a black-box label**.

### Example

```text
IMAGE
  ↓
Optic-disc region
  ↓
Possible enlarged cup
  ↓
Supporting image region
       +
Clinical note:
IOP = 28 mmHg
  ↓
Multimodal reasoning
  ↓
Grounded explanation
```

This creates a chain:

```text
Finding → Evidence → Explanation
```

rather than:

```text
Image → "Glaucoma"
```

---

# 13. Limitations

- The dataset contains only 101 images.
- The dataset is from a single source.
- The classes are binary.
- The test split is imbalanced.
- Patient notes are synthetic and may leak label information.
- The system has not been clinically validated.
- A fundus photograph alone cannot confirm glaucoma.
- Clinical assessment may require IOP, visual fields, OCT and other examination findings.
- The system has not been evaluated across different populations, cameras and image-quality conditions.
- Model scores should not be interpreted as clinically calibrated probabilities unless calibration has been explicitly performed.
- Highlighted regions represent model focus/evidence and do not prove causality.
- The system is intended for research/hackathon demonstration and not autonomous diagnosis.

---

# 14. Reproducibility and Responsible Use

For a reproducible demonstration, keep the following fixed:

```text
Dataset split
      +
Synthetic-note files
      +
Model routing
      +
Application version
      +
requirements.txt
      +
Prompt/configuration files
```

For a future research release, also record:

- model/API version,
- inference timestamp,
- prompt/template version,
- preprocessing configuration,
- random seeds where applicable,
- evaluation script/version,
- and exact dataset manifest.

This makes the reported results easier to audit and reproduce.

---

# 15. Project Structure

```text
.
├── README.md
├── requirements.txt
├── .env.example
├── app/
│   ├── UI/backend code
│   └── model/inference logic
├── data/
│   ├── images/
│   └── synthetic_notes/
└── docs/
    └── screenshots/
```

---

# 16. Roadmap

- [x] Fundus image upload
- [x] Three-model Gemini architecture
- [x] Patient-note integration
- [x] Multimodal reasoning
- [x] Explainable findings
- [x] Evidence-grounding concept
- [x] Region/heatmap visualization
- [ ] Optic-disc and optic-cup segmentation
- [ ] Automated C/D-ratio measurement
- [ ] Calibrated uncertainty/confidence
- [ ] Independent clinical-note evaluation
- [ ] Longitudinal progression analysis
- [ ] Evaluation on additional public datasets
- [ ] Cross-camera and cross-population validation

---

# 17. Hackathon Information

| Item | Details |
|---|---|
| **Hackathon** | HackNex 26 |
| **Problem statement** | HNX26PSI05 — Multimodal Medical Image Intelligence |
| **Team** | Elixir |
| **Demo** | `<add final demo URL>` |

---

# 18. Acknowledgements

- Drishti-GS dataset authors
- Google Gemini API
- HackNex 26 organisers

---

# 19. License

Add the final project license after confirming the license requirements of the project code, dependencies and dataset.
