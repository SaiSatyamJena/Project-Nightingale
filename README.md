# Project Nightingale: A Hallucination Survey Function for Medical Report Summarization

<p align="center">
  <img src="Nightingale.png" alt="Project Nightingale Logo" width="200"/>
</p>

**License:** [MIT](LICENSE.md)

**Project Goal:** To develop and showcase a robust Hallucination Survey Function for critically evaluating the factual accuracy and consistency of LLM-generated summaries of medical reports, with a strong emphasis on identifying both intrinsic and extrinsic hallucinations.

## Table of Contents
1. [Introduction: The Criticality of Accuracy](#introduction)
    * [The Challenge of Hallucinations](#the-challenge-of-hallucinations)
    * [Project Goal: The Hallucination Survey Function](#project-goal-hsf)
    * [Our Approach: A Multi-Faceted Survey](#our-approach)
2. [Architecting the `HallucinationSurveyor`: An OOP Approach](#architecting-the-hallucinationsurveyor)
    * [Core Design Philosophy (OOP)](#core-design-philosophy-oop)
    * [The `HallucinationSurveyor` Class](#the-hallucinationsurveyor-class)
        * [Initialization (`__init__`)](#initialization-__init__)
        * [Helper Methods (Encapsulation)](#helper-methods-encapsulation)
        * [Core Checking Methods (Modular Design)](#core-checking-methods-modular-design)
        * [The `survey()` Method (Orchestration)](#the-survey-method-orchestration)
        * [Heuristic Scoring Mechanism](#heuristic-scoring-mechanism)
3. [Demonstrating Efficacy: Test Cases & Interpretation](#demonstrating-efficacy-test-cases--interpretation)
    * [Source Texts Used for Evaluation](#source-texts-used-for-evaluation)
    * [Scenario 1: Good, Factual Summary](#scenario-1-good-factual-summary)
    * [Scenario 2: Summary with Fabricated Critical Entity (Extrinsic Hallucination)](#scenario-2-summary-with-fabricated-critical-entity-extrinsic-hallucination)
    * [Scenario 3: Summary with Incorrect Numerical Value (Intrinsic Hallucination)](#scenario-3-summary-with-incorrect-numerical-value-intrinsic-hallucination)
    * [Scenario 4: Summary with Flipped Negation (Intrinsic Hallucination)](#scenario-4-summary-with-flipped-negation-intrinsic-hallucination)
    * [Scenario 5: Summary with Omission of Critical Information](#scenario-5-summary-with-omission-of-critical-information)
4. [Application: Evaluating LLM-Generated Summaries](#application-evaluating-llm-generated-summaries)
    * [Model 1: DistilBART-CNN-6-6 (Seq2Seq Summarizer)](#model-1-distilbart-cnn-6-6-seq2seq-summarizer)
    * [Model 2: GPT-Neo-125M (Decoder-Only LLM)](#model-2-gpt-neo-125m-decoder-only-llm)
5. [Discussion: Insights and Future Horizons](#discussion-insights-and-future-horizons)
    * [Strengths of Project Nightingale](#strengths-of-project-nightingale)
    * [Limitations and Future Enhancements](#limitations-and-future-enhancements)
6. [Conclusion: Towards Reliable Medical Summarization](#conclusion-towards-reliable-medical-summarization)
7. [Setup and Usage](#setup-and-usage)

---

## 1. Introduction: The Criticality of Accuracy <a name="introduction"></a>

### The Challenge of Hallucinations <a name="the-challenge-of-hallucinations"></a>
Large Language Models (LLMs), while demonstrating remarkable text generation capabilities, can "hallucinate." In the high-stakes medical domain, hallucinations in report summaries—generating factually incorrect, fabricated, or nonsensical content—can lead to severe patient safety risks, misinform clinicians, and carry significant legal and ethical implications. This project confronts this challenge by focusing on detecting two primary types of factual discrepancies crucial for medical accuracy:
* **Intrinsic Hallucinations:** The summary contradicts or is inconsistent with information explicitly stated in the source document.
* **Extrinsic Hallucinations:** The summary introduces new information ("facts" or details) not verifiable from the source document.

### Project Goal: The Hallucination Survey Function <a name="project-goal-hsf"></a>
Project Nightingale's core objective is the development and demonstration of a comprehensive **`HallucinationSurveyor` function**. This tool is specifically designed to meticulously evaluate summaries of medical reports. It aims not just to detect hallucinations but to provide a detailed *survey* of potential issues, offering deep insights into a summary's faithfulness to the source document and quantifying various aspects of potential hallucinations.

### Our Approach: A Multi-Faceted Survey <a name="our-approach"></a>
Instead of relying on a single, opaque metric, our `HallucinationSurveyor` employs a battery of distinct checks. Each check targets different potential failure modes common in medical text summarization. This multi-faceted approach, architected using Object-Oriented Programming (OOP) principles, provides a more nuanced, interpretable, and comprehensive assessment of summary quality and factual integrity.

---

## 2. Architecting the `HallucinationSurveyor`: An OOP Approach <a name="architecting-the-hallucinationsurveyor"></a>

The `HallucinationSurveyor` is implemented as a Python class, leveraging OOP principles to ensure the system is clear, modular, extensible, and maintainable. This design philosophy is central to Project Nightingale.

### Core Design Philosophy (OOP) <a name="core-design-philosophy-oop"></a>
The surveyor's "survey" methodology is built upon key OOP tenets:
* **Encapsulation:** The `HallucinationSurveyor` class bundles relevant data (e.g., predefined `critical_medical_terms`, `common_units`, `stopwords`) and the methods that operate on this data within a single, cohesive unit (the surveyor object). This protects internal state and provides a clean interface.
* **Modularity & Single Responsibility:** Each distinct type of check (e.g., for numerical consistency, entity coherence, negation flips) is implemented as a separate method within the class. This makes the system easier to understand, test, maintain, and extend with new checks in the future.
* **Abstraction:** Users interact with the surveyor through a simple `survey()` method, abstracting away the complex internal logic of individual checks.
* **Reusability:** Surveyor instances can be configured (e.g., with domain-specific critical terms) and reused across multiple summary evaluations.

This OOP structure allows for:
* **Granular Feedback:** Pinpointing specific *types* of hallucinations or inconsistencies.
* **Interpretability:** Clearly understanding *why* a summary is flagged, through the detailed output of each modular check.

### The `HallucinationSurveyor` Class <a name="the-hallucinationsurveyor-class"></a>

#### Initialization (`__init__`) <a name="initialization-__init__"></a>
The constructor `__init__(self, critical_medical_terms=None, common_units=None)` initializes a surveyor instance.
* It accepts optional lists of `critical_medical_terms` and `common_units`, allowing for customization to specific medical domains or datasets. If not provided, it defaults to a generic, pre-defined set.
* It stores these configurations as instance attributes (sets for efficient lookup), encapsulating the surveyor's operational context.
* It also incorporates a set of `stopwords` (derived from NLTK or a robust fallback) for text preprocessing.

#### Helper Methods (Encapsulation) <a name="helper-methods-encapsulation"></a>
The class employs several "private" helper methods (by convention, prefixed with an underscore) to encapsulate common, low-level tasks, promoting code reuse and reducing redundancy within the main checking methods:
* `_preprocess_text_for_word_analysis(text)`: This method handles the initial text cleaning (lowercasing, removing extra whitespace), tokenization (using NLTK's `word_tokenize` or a fallback), and stopword removal. It ensures that all text inputs to higher-level analysis methods are consistently processed. It also intelligently retains numerical tokens and critical medical terms.
* `_extract_numerical_phrases(text)`: A sophisticated method using refined regular expressions to accurately identify and extract numerical values along with their associated units (e.g., "10mg", "120/80mmhg", "20%"). It handles various number formats (integers, decimals, ranges, fractions) and normalizes symbols (e.g., 'μg' to 'mcg'). This method was iteratively developed for high precision.
* `_extract_negations_with_context(text, terms_to_check)`: This method identifies common negation markers (e.g., "no", "denies", "negative for") within sentences and checks for the presence of specified `terms_to_check` (typically critical medical terms) in the immediate contextual window following the negation. It extracts the relevant negated phrase for comparison.

#### Core Checking Methods (Modular Design) <a name="core-checking-methods-modular-design"></a>
Each primary evaluation facet is implemented as a distinct public method, demonstrating the modularity of the OOP design:
* `check_entity_coherence(source_text, summary_text)`:
    * Assesses if **critical medical terms** are accurately represented.
    * Identifies `hallucinated_critical_entities`: Critical terms present in the summary but not in the source (clear **extrinsic hallucination**).
    * Flags `omitted_critical_entities`: Critical terms present in the source but missing from the summary (critical information gap).
    * Lists `potentially_fabricated_general_terms`: New non-critical, non-numerical words introduced by the summary (potential **extrinsic hallucination** or rephrasing).
* `check_numerical_consistency(source_text, summary_text)`:
    * Leverages `_extract_numerical_phrases` to compare sets of numerical data between source and summary.
    * Identifies `numbers_in_summary_not_in_source`: Numerical facts (number-unit pairs or standalone numbers) in the summary that lack grounding in the source (can be **extrinsic**, or **intrinsic** if a source value is altered).
    * Lists `numbers_in_source_not_in_summary`: Numerical facts from the source omitted by the summary.
* `check_negation_consistency(source_text, summary_text)`:
    * Uses `_extract_negations_with_context` to compare how critical medical terms are negated in the source versus the summary.
    * Flags `potential_negation_issues` if the summary introduces a negation context for a critical term that isn't supported by, or contradicts, the source's negations (a key type of **intrinsic hallucination**).
* `check_n_gram_overlap(source_text, summary_text, n_values=[2, 3])`:
    * Calculates the overlap of N-grams (sequences of N words, typically bigrams and trigrams) between the source and summary.
    * A low overlap score can indicate significant textual divergence, potentially due to extensive rephrasing or the introduction of novel (possibly extrinsic) information.
    * Lists a sample of summary N-grams not found in the source, offering qualitative insight into new phrasings.
* `check_abstractiveness_metrics(source_text, summary_text)`:
    * Integrates standard **ROUGE scores** (ROUGE-1, ROUGE-2, ROUGE-L) to provide established metrics for content overlap, precision, and recall. These help characterize the summary's style (extractive vs. abstractive) and coverage.
    * Calculates the length ratio of the summary to the source. These metrics provide contextual understanding of the summary's relationship to the source text.

#### The `survey()` Method (Orchestration) <a name="the-survey-method-orchestration"></a>
The `survey(self, source_text, summary_text)` method is the primary public interface of the `HallucinationSurveyor` class. It encapsulates the complexity of the evaluation process:
1. It accepts the `source_text` and `summary_text` as inputs.
2. It systematically calls each of the individual checking methods (`check_entity_coherence`, `check_numerical_consistency`, etc.).
3. It aggregates the results from these methods into a comprehensive, structured dictionary (the "report").
4. It then calls `calculate_heuristic_score` to compute an overall risk score based on the findings in the report.
5. Finally, it returns the complete report, including all detailed findings, the overall score, and informative survey notes.
This orchestration simplifies the user's interaction with the surveyor, requiring only a single method call to get a full evaluation.

#### Heuristic Scoring Mechanism <a name="heuristic-scoring-mechanism"></a>
The `calculate_heuristic_score(self, report)` method provides a quantitative summary of potential issues.
* It assigns different weights to various findings (e.g., hallucinated critical entities receive a higher penalty than fabricated general terms or omitted numbers).
* It sums these weighted penalties to produce an **Overall Heuristic Risk Score**. A higher score indicates a greater number or severity of detected inconsistencies or fabrications.
* It's crucial to understand this is a *heuristic* score designed to provide a comparative signal. The detailed breakdown within the report offers more nuanced insights than the score alone.

---

## 3. Demonstrating Efficacy: Test Cases & Interpretation <a name="demonstrating-efficacy-test-cases--interpretation"></a>

The `HallucinationSurveyor` was rigorously tested against a primary source text (`source_text_1_cardiology`) using several handcrafted summary scenarios to validate each checking component.

### Source Texts Used for Evaluation <a name="source-texts-used-for-evaluation"></a>
Three distinct mock medical reports (`source_text_1_cardiology`, `source_text_2_diabetes`, `source_text_3_ortho`) were defined in the notebook (Chapter 4.1) to provide diverse contexts. The following scenarios primarily use `source_text_1_cardiology`.

### Scenario 1: Good, Factual Summary <a name="scenario-1-good-factual-summary"></a>
**Philosophy:** Establish a baseline with a summary that is factually accurate and well-grounded.
**Key Expectation:** Low number of direct hallucination flags. The Heuristic Risk Score should mainly reflect permissible omissions inherent in good summarization.
**Resulting Overall Heuristic Risk Score: 48**
**Interpretation:** The score of 48 was predominantly driven by `numbers_in_source_not_in_summary` (10 items * 4 points = 40 points), which is natural as a summary omits details. Critically, there were no `hallucinated_critical_entities` or `numbers_in_summary_not_in_source`. The `potentially_fabricated_general_terms` (e.g., "allergic", "clinic") were identified as reasonable abstractive word choices not directly present in the source text's vocabulary.

```json
{
  "entity_coherence": {
    "hallucinated_critical_entities": [],
    "omitted_critical_entities": [],
    "potentially_fabricated_general_terms": [
      "allergic",
      "clinic",
      "current",
      "discuss",
      "lipids",
      "medications",
      "recent",
      "showed"
    ]
  },
  "numerical_consistency": {
    "numbers_in_summary_not_in_source": [],
    "numbers_in_source_not_in_summary": [
      "001",
      "1",
      "10",
      "1970-03-15",
      "2",
      "2-3",
      "3",
      "4",
      "5",
      "72bpm"
    ]
  },
  "negation_consistency": {
    "potential_negation_issues": [],
    "source_negated_phrases_critical": [
      "...ar-old male with a known history of hypertension for 10 years and hy..."
    ],
    "summary_negated_phrases_critical": []
  },
  "n_gram_overlap": {
    "n_gram_analysis": {
      "2-gram": {
        "overlap_score": 0.324,
        "summary_ngrams_not_in_source": [
          "20mg and",
          "and slightly",
          "drinks occasionally",
          "home bp",
          "to penicillin"
        ]
      },
      "3-gram": {
        "overlap_score": 0.149,
        "summary_ngrams_not_in_source": [
          "allergic to penicillin",
          "around mmhg and",
          "he is a",
          "mmhg recent lipids",
          "symptoms his home"
        ]
      }
    }
  },
  "abstractiveness_metrics": {
    "rouge_scores": {
      "rouge1": {
        "precision": 0.854,
        "recall": 0.4,
        "fmeasure": 0.545
      },
      "rouge2": {
        "precision": 0.474,
        "recall": 0.221,
        "fmeasure": 0.301
      },
      "rougeL": {
        "precision": 0.656,
        "recall": 0.307,
        "fmeasure": 0.419
      }
    },
    "length_ratio_summary_vs_source": 0.413
  },
  "overall_hallucination_heuristic_score": 48,
  "survey_notes": [
    "This is a heuristic survey. Manual review is crucial.",
    "Heuristic score (higher indicates more potential issues): 48",
    "NLTK tokenizers used: True",
    "ROUGE metrics available: True"
  ]
}
```

### Scenario 2: Summary with Fabricated Critical Entity (Extrinsic Hallucination) <a name="scenario-2-summary-with-fabricated-critical-entity-extrinsic-hallucination"></a>
**Philosophy:** Introduce a critical medical condition ("diabetes") not mentioned in the source.
**Key Expectation:** The hallucinated_critical_entities check should clearly flag "diabetes," significantly increasing the Heuristic Risk Score.
**Resulting Overall Heuristic Risk Score: 85**
**Interpretation:** The surveyor precisely identified "diabetes" as a hallucinated_critical_entity (contributing 10 points directly). The score also increased due to related potentially_fabricated_general_terms (e.g., "diagnosed", "newly", "manage" - 9 points) and a higher count of numbers_in_source_not_in_summary (56 points) as the summary content shifted. A negation issue related to the fabricated context was also flagged (10 points). This demonstrates clear detection of a severe extrinsic hallucination.

```json
{
  "entity_coherence": {
    "hallucinated_critical_entities": [
      "diabetes"
    ],
    "omitted_critical_entities": [],
    "potentially_fabricated_general_terms": [
      "allergic",
      "clinic",
      "current",
      "diagnosed",
      "lipids",
      "manage",
      "medications",
      "newly",
      "recent"
    ]
  },
  "numerical_consistency": {
    "numbers_in_summary_not_in_source": [],
    "numbers_in_source_not_in_summary": [
      "001",
      "1",
      "10",
      "135/85mmhg",
      "170mg",
      "180mg",
      "1970-03-15",
      "2",
      "2-3",
      "3",
      "4",
      "45mg",
      "5",
      "72bpm"
    ]
  },
  "negation_consistency": {
    "potential_negation_issues": [
      "Summary has distinct negation context not found in source: '...emia, and newly diagnosed diabetes, is on lisinopril 2...'"
    ],
    "source_negated_phrases_critical": [
      "...ar-old male with a known history of hypertension for 10 years and hy..."
    ],
    "summary_negated_phrases_critical": [
      "...emia, and newly diagnosed diabetes, is on lisinopril 2..."
    ]
  },
  "n_gram_overlap": {
    "n_gram_analysis": {
      "2-gram": {
        "overlap_score": 0.275,
        "summary_ngrams_not_in_source": [
          "20mg and",
          "current medications",
          "diagnosed diabetes",
          "he denies",
          "to penicillin"
        ]
      },
      "3-gram": {
        "overlap_score": 0.14,
        "summary_ngrams_not_in_source": [
          "100 he is",
          "allergic to penicillin",
          "his clinic bp",
          "hypertension hyperlipidemia and",
          "manage his diabetes"
        ]
      }
    }
  },
  "abstractiveness_metrics": {
    "rouge_scores": {
      "rouge1": {
        "precision": 0.767,
        "recall": 0.224,
        "fmeasure": 0.347
      },
      "rouge2": {
        "precision": 0.373,
        "recall": 0.108,
        "fmeasure": 0.167
      },
      "rougeL": {
        "precision": 0.617,
        "recall": 0.18,
        "fmeasure": 0.279
      }
    },
    "length_ratio_summary_vs_source": 0.271
  },
  "overall_hallucination_heuristic_score": 85,
  "survey_notes": [
    "This is a heuristic survey. Manual review is crucial.",
    "Heuristic score (higher indicates more potential issues): 85",
    "NLTK tokenizers used: True",
    "ROUGE metrics available: True"
  ]
}
```

### Scenario 3: Summary with Incorrect Numerical Value (Intrinsic Hallucination) <a name="scenario-3-summary-with-incorrect-numerical-value-intrinsic-hallucination"></a>
**Philosophy:** Alter a key dosage (Lisinopril 20mg changed to 200mg).
**Key Expectation:** numerical_consistency should flag "200mg" as new to the summary and "20mg" as missing from the summary relative to the source.
**Resulting Overall Heuristic Risk Score: 75**
**Interpretation:** The report showed "200mg" in numbers_in_summary_not_in_source (contributing 8 points) and correctly identified "20mg" within numbers_in_source_not_in_summary (contributing to 60 points from total omitted numbers). The token "200mg" was also flagged under potentially_fabricated_general_terms (7 points total for general terms). This demonstrates strong detection of a critical intrinsic numerical error.

```json
{
  "entity_coherence": {
    "hallucinated_critical_entities": [],
    "omitted_critical_entities": [],
    "potentially_fabricated_general_terms": [
      "200mg",
      "allergic",
      "lipids",
      "managed",
      "medications",
      "recent",
      "showed"
    ]
  },
  "numerical_consistency": {
    "numbers_in_summary_not_in_source": [
      "200mg"
    ],
    "numbers_in_source_not_in_summary": [
      "001",
      "1",
      "10",
      "138/88mmhg",
      "170mg",
      "180mg",
      "1970-03-15",
      "2",
      "2-3",
      "20mg",
      "3",
      "4",
      "45mg",
      "5",
      "72bpm"
    ]
  },
  "negation_consistency": {
    "potential_negation_issues": [],
    "source_negated_phrases_critical": [
      "...ar-old male with a known history of hypertension for 10 years and hy..."
    ],
    "summary_negated_phrases_critical": []
  },
  "n_gram_overlap": {
    "n_gram_analysis": {
      "2-gram": {
        "overlap_score": 0.381,
        "summary_ngrams_not_in_source": [
          "has hypertension",
          "he denies",
          "home bp",
          "is around",
          "to penicillin"
        ]
      },
      "3-gram": {
        "overlap_score": 0.195,
        "summary_ngrams_not_in_source": [
          "100 he is",
          "54 has hypertension",
          "allergic to penicillin",
          "lipids showed ldl",
          "symptoms his home"
        ]
      }
    }
  },
  "abstractiveness_metrics": {
    "rouge_scores": {
      "rouge1": {
        "precision": 0.875,
        "recall": 0.205,
        "fmeasure": 0.332
      },
      "rouge2": {
        "precision": 0.426,
        "recall": 0.098,
        "fmeasure": 0.159
      },
      "rougeL": {
        "precision": 0.75,
        "recall": 0.176,
        "fmeasure": 0.285
      }
    },
    "length_ratio_summary_vs_source": 0.231
  },
  "overall_hallucination_heuristic_score": 75,
  "survey_notes": [
    "This is a heuristic survey. Manual review is crucial.",
    "Heuristic score (higher indicates more potential issues): 75",
    "NLTK tokenizers used: True",
    "ROUGE metrics available: True"
  ]
}
```

### Scenario 4: Summary with Flipped Negation (Intrinsic Hallucination) <a name="scenario-4-summary-with-flipped-negation-intrinsic-hallucination"></a>
**Philosophy:** Contradict the source regarding a Penicillin allergy (Source: Patient is allergic. Summary: Patient has no allergy). "Penicillin" was temporarily added to critical terms for this test.
**Key Expectation:** negation_consistency should flag a potential_negation_issue due to the summary's negated statement about penicillin contradicting the source's affirmation.
**Resulting Overall Heuristic Risk Score: 79**
**Interpretation:** The surveyor flagged: "Summary has distinct negation context not found in source: '...he has no allergy to penicillin....'" (10 points). The term "allergy" was also flagged as a hallucinated_critical_entity (10 points) because the summary presented it in a new, negated context not found in the source for this specific critical term. The score also reflects general term fabrications (3 points) and omitted numbers (56 points). This successfully identified a dangerous flip in meaning.

```json
{
  "entity_coherence": {
    "hallucinated_critical_entities": [
      "allergy"
    ],
    "omitted_critical_entities": [],
    "potentially_fabricated_general_terms": [
      "lipids",
      "medications",
      "recent"
    ]
  },
  "numerical_consistency": {
    "numbers_in_summary_not_in_source": [],
    "numbers_in_source_not_in_summary": [
      "001",
      "1",
      "10",
      "138/88mmhg",
      "170mg",
      "180mg",
      "1970-03-15",
      "2",
      "2-3",
      "3",
      "4",
      "45mg",
      "5",
      "72bpm"
    ]
  },
  "negation_consistency": {
    "potential_negation_issues": [
      "Summary has distinct negation context not found in source: '...he has no allergy to penicillin....'"
    ],
    "source_negated_phrases_critical": [
      "...ar-old male with a known history of hypertension for 10 years and hy..."
    ],
    "summary_negated_phrases_critical": [
      "...he has no allergy to penicillin...."
    ]
  },
  "n_gram_overlap": {
    "n_gram_analysis": {
      "2-gram": {
        "overlap_score": 0.325,
        "summary_ngrams_not_in_source": [
          "20mg and",
          "has hypertension",
          "he denies",
          "home bp",
          "to penicillin"
        ]
      },
      "3-gram": {
        "overlap_score": 0.128,
        "summary_ngrams_not_in_source": [
          "54 has hypertension",
          "around mmhg recent",
          "has no allergy",
          "mmhg recent lipids",
          "symptoms his home"
        ]
      }
    }
  },
  "abstractiveness_metrics": {
    "rouge_scores": {
      "rouge1": {
        "precision": 0.891,
        "recall": 0.2,
        "fmeasure": 0.327
      },
      "rouge2": {
        "precision": 0.378,
        "recall": 0.083,
        "fmeasure": 0.137
      },
      "rougeL": {
        "precision": 0.783,
        "recall": 0.176,
        "fmeasure": 0.287
      }
    },
    "length_ratio_summary_vs_source": 0.223
  },
  "overall_hallucination_heuristic_score": 79,
  "survey_notes": [
    "This is a heuristic survey. Manual review is crucial.",
    "Heuristic score (higher indicates more potential issues): 79",
    "NLTK tokenizers used: True",
    "ROUGE metrics available: True"
  ]
}
```

### Scenario 5: Summary with Omission of Critical Information <a name="scenario-5-summary-with-omission-of-critical-information"></a>
**Philosophy:** Omit a key medication (Atorvastatin 40mg). "Atorvastatin" was temporarily added to critical terms.
**Key Expectation:** omitted_critical_entities should list "atorvastatin," and numbers_in_source_not_in_summary should list "40mg."
**Resulting Overall Heuristic Risk Score: 73**
**Interpretation:** The surveyor successfully flagged "atorvastatin" as an omitted_critical_entity (7 points) and "40mg" within numbers_in_source_not_in_summary (contributing to 64 points from total omissions). This showcases the tool's ability to identify critical information gaps.

```json
{
  "entity_coherence": {
    "hallucinated_critical_entities": [],
    "omitted_critical_entities": [
      "atorvastatin"
    ],
    "potentially_fabricated_general_terms": [
      "allergic",
      "managed"
    ]
  },
  "numerical_consistency": {
    "numbers_in_summary_not_in_source": [],
    "numbers_in_source_not_in_summary": [
      "001",
      "1",
      "10",
      "100mg",
      "138/88mmhg",
      "170mg",
      "180mg",
      "1970-03-15",
      "2",
      "2-3",
      "3",
      "4",
      "40mg",
      "45mg",
      "5",
      "72bpm"
    ]
  },
  "negation_consistency": {
    "potential_negation_issues": [],
    "source_negated_phrases_critical": [
      "...ar-old male with a known history of hypertension for 10 years and hy...",
      "...ntly prescribed lisinopril 20mg daily and atorvastatin 40mg daily...."
    ],
    "summary_negated_phrases_critical": []
  },
  "n_gram_overlap": {
    "n_gram_analysis": {
      "2-gram": {
        "overlap_score": 0.441,
        "summary_ngrams_not_in_source": [
          "he denies",
          "home bp",
          "is around",
          "managed with",
          "to penicillin"
        ]
      },
      "3-gram": {
        "overlap_score": 0.212,
        "summary_ngrams_not_in_source": [
          "20mg daily he",
          "allergic to penicillin",
          "continue lisinopril follow",
          "is around mmhg",
          "mmhg he is"
        ]
      }
    }
  },
  "abstractiveness_metrics": {
    "rouge_scores": {
      "rouge1": {
        "precision": 0.902,
        "recall": 0.18,
        "fmeasure": 0.301
      },
      "rouge2": {
        "precision": 0.5,
        "recall": 0.098,
        "fmeasure": 0.164
      },
      "rougeL": {
        "precision": 0.805,
        "recall": 0.161,
        "fmeasure": 0.268
      }
    },
    "length_ratio_summary_vs_source": 0.186
  },
  "overall_hallucination_heuristic_score": 73,
  "survey_notes": [
    "This is a heuristic survey. Manual review is crucial.",
    "Heuristic score (higher indicates more potential issues): 73",
    "NLTK tokenizers used: True",
    "ROUGE metrics available: True"
  ]
}
```

## 4. Application: Evaluating LLM-Generated Summaries <a name="application-evaluating-llm-generated-summaries"></a>

The HallucinationSurveyor was applied to summaries of `source_text_1_cardiology` generated by two different LLMs from Hugging Face.

### Model 1: DistilBART-CNN-6-6 (Seq2Seq Summarizer) <a name="model-1-distilbart-cnn-6-6-seq2seq-summarizer"></a>
**LLM Summary Observation:** This model produced a very short, highly extractive summary, mostly copying the initial sentences of the source.
**Resulting Overall Heuristic Risk Score: 60**
**Surveyor Interpretation:**
- No Direct Hallucinations: No fabricated critical entities/numbers or conflicting negations were found.
- Poor Summarization Quality: The score of 60 was entirely due to a large number of `numbers_in_source_not_in_summary`, indicating significant omission of factual details.
- High N-gram Overlap (0.933 for bigrams) and specific ROUGE patterns (high Precision, low Recall) confirmed its extractive, non-comprehensive nature.
**Conclusion:** While "safe" from active fabrications in this instance, it was not a useful or complete medical summary.

```json
{
  "entity_coherence": {
    "hallucinated_critical_entities": [],
    "omitted_critical_entities": [],
    "potentially_fabricated_general_terms": []
  },
  "numerical_consistency": {
    "numbers_in_summary_not_in_source": [],
    "numbers_in_source_not_in_summary": [
      "001",
      "1",
      "100mg",
      "135/85mmhg",
      "138/88mmhg",
      "170mg",
      "180mg",
      "1970-03-15",
      "2",
      "2-3",
      "3",
      "4",
      "45mg",
      "6",
      "72bpm"
    ]
  },
  "negation_consistency": {
    "potential_negation_issues": [],
    "source_negated_phrases_critical": [
      "...ar-old male with a known history of hypertension for 10 years and hy..."
    ],
    "summary_negated_phrases_critical": [
      "...ar-old male with a known history of hypertension for 10 years and hy..."
    ]
  },
  "n_gram_overlap": {
    "n_gram_analysis": {
      "2-gram": {
        "overlap_score": 0.933,
        "summary_ngrams_not_in_source": [
          "54 is",
          "smith 54"
        ]
      },
      "3-gram": {
        "overlap_score": 0.897,
        "summary_ngrams_not_in_source": [
          "54 is a",
          "john smith 54",
          "smith 54 is"
        ]
      }
    }
  },
  "abstractiveness_metrics": {
    "rouge_scores": {
      "rouge1": {
        "precision": 0.971,
        "recall": 0.166,
        "fmeasure": 0.283
      },
      "rouge2": {
        "precision": 0.941,
        "recall": 0.157,
        "fmeasure": 0.269
      },
      "rougeL": {
        "precision": 0.971,
        "recall": 0.166,
        "fmeasure": 0.283
      }
    },
    "length_ratio_summary_vs_source": 0.15
  },
  "overall_hallucination_heuristic_score": 60,
  "survey_notes": [
    "This is a heuristic survey. Manual review is crucial.",
    "Heuristic score (higher indicates more potential issues): 60",
    "NLTK tokenizers used: True",
    "ROUGE metrics available: True"
  ]
}
```

### Model 2: GPT-Neo-125M (Decoder-Only LLM) <a name="model-2-gpt-neo-125m-decoder-only-llm"></a>
**LLM Summary Observation:** This model produced a longer, more abstractive summary that contained significant factual errors, fabricated drug names (e.g., "Lisinoplatin", "Lisinotecan", "Lisinopaol"), incorrect durations, and internal contradictions.
**Resulting Overall Heuristic Risk Score: 84**
**Surveyor Interpretation:**
- Fabricated General Terms Detected: The surveyor caught "lisinopaol" under potentially_fabricated_general_terms. Other fabrications (e.g., "diagnosed", "first", "high") also contributed. (Note: Novel misspellings of critical drugs like "Lisinoplatin" were not flagged as hallucinated_critical_entities due to the surveyor's reliance on an exact match against the predefined critical terms list for that specific check).
- Significant Semantic Divergence: Very low N-gram overlap (0.08 for bigrams) and low ROUGE scores (ROUGE-L F1: 0.143) strongly signaled that the summary's content and phrasing had diverged significantly from the source, aligning with its fabricated and contradictory nature.
- Numerical Inconsistencies (Limitations Highlighted): The `numbers_in_summary_not_in_source` list was empty. While the LLM summary did fabricate durations (e.g., "2 years", "3 months"), these were not caught because "years" and "months" are not in `self.common_units` for the numerical phrase extractor. This indicates an area for future enhancement in contextual number checking.
**Conclusion:** The high Heuristic Risk Score of 84, driven by fabricated terms, a high number of omitted source numbers (as the LLM rambled on new topics), and a distinct negation context, correctly signaled a highly problematic and unreliable summary.

```json
{
  "entity_coherence": {
    "hallucinated_critical_entities": [],
    "omitted_critical_entities": [],
    "potentially_fabricated_general_terms": [
      "also",
      "complaints",
      "continues",
      "day",
      "diagnosed",
      "diastolic",
      "first",
      "high",
      "increase",
      "lisinopaol"
    ]
  },
  "numerical_consistency": {
    "numbers_in_summary_not_in_source": [],
    "numbers_in_source_not_in_summary": [
      "001",
      "1",
      "10",
      "100mg",
      "135/85mmhg",
      "138/88mmhg",
      "170mg",
      "180mg",
      "1970-03-15",
      "2-3",
      "4",
      "45mg",
      "5",
      "54",
      "6",
      "72bpm"
    ]
  },
  "negation_consistency": {
    "potential_negation_issues": [
      "Summary has distinct negation context not found in source: '...nt who has been diagnosed with hypertension....'"
    ],
    "source_negated_phrases_critical": [
      "...ar-old male with a known history of hypertension for 10 years and hy..."
    ],
    "summary_negated_phrases_critical": [
      "...nt who has been diagnosed with hypertension...."
    ]
  },
  "n_gram_overlap": {
    "n_gram_analysis": {
      "2-gram": {
        "overlap_score": 0.08,
        "summary_ngrams_not_in_source": [
          "does not",
          "his chest",
          "mild but",
          "past 3",
          "triglycerides are"
        ]
      },
      "3-gram": {
        "overlap_score": 0.013,
        "summary_ngrams_not_in_source": [
          "does have some",
          "is a slight",
          "months he reports",
          "no new complaints",
          "on lisinotecan 40mg"
        ]
      }
    }
  },
  "abstractiveness_metrics": {
    "rouge_scores": {
      "rouge1": {
        "precision": 0.384,
        "recall": 0.298,
        "fmeasure": 0.335
      },
      "rouge2": {
        "precision": 0.07,
        "recall": 0.054,
        "fmeasure": 0.061
      },
      "rougeL": {
        "precision": 0.164,
        "recall": 0.127,
        "fmeasure": 0.143
      }
    },
    "length_ratio_summary_vs_source": 0.729
  },
  "overall_hallucination_heuristic_score": 84,
  "survey_notes": [
    "This is a heuristic survey. Manual review is crucial.",
    "Heuristic score (higher indicates more potential issues): 84",
    "NLTK tokenizers used: True",
    "ROUGE metrics available: True"
  ]
}
```

## 5. Discussion: Insights and Future Horizons <a name="discussion-insights-and-future-horizons"></a>

### Strengths of Project Nightingale <a name="strengths-of-project-nightingale"></a>
The HallucinationSurveyor developed in this project demonstrates several key strengths:
- Multi-Faceted & OOP-Driven Evaluation: Its Object-Oriented design promotes modularity and allows for a suite of checks (entity coherence, numerical consistency, negation analysis, N-gram overlap, abstractiveness metrics) providing a granular, interpretable understanding of summary quality.
- Detection of Critical Intrinsic & Extrinsic Hallucinations: As validated in Chapter 4, the surveyor effectively identifies critical medical errors such as fabricated entities (e.g., "diabetes"), incorrect numerical values (e.g., wrong dosage), flipped negations (e.g., allergy status), and significant omissions of critical information.
- Interpretability of Results: The detailed reports pinpoint where and what kind of potential issues exist, making feedback actionable for human reviewers and model developers.
- Quantitative Heuristics & Contextual Metrics: The Overall Heuristic Risk Score, combined with N-gram and ROUGE metrics, offers both a comparative measure and insights into model behavior (e.g., extractiveness vs. abstractive fabrication).
- Practical Implementation: Built with standard Python libraries, making it relatively lightweight and deployable.

### Limitations and Future Enhancements <a name="limitations-and-future-enhancements"></a>
While effective, the current lexical and rule-based OOP approach has limitations, paving the way for future enhancements:
- Novel Entity Fabrications: The reliance on predefined critical terms means novel fabrications (e.g., misspelled or entirely invented drug names like "Lisinoplatin") might only be caught as "general terms."
  - **Future Work:** Integrate fuzzy matching against medical ontologies (e.g., RxNorm, SNOMED CT) or advanced medical NER models.
- Contextual Understanding of Numbers: The current system primarily checks for presence/absence of number-unit strings. Deeper contextual analysis (e.g., "2 years" duration vs. "2mg" dosage) is an area for growth.
  - **Future Work:** Develop methods to link numerical values to their medical concepts and check for plausible ranges.
- Semantic Contradictions & Complex Reasoning: Deeper semantic contradictions or errors requiring multi-step reasoning are challenging for the current rule-based negation check.
  - **Future Work:** Explore Natural Language Inference (NLI) models or Question Answering (QA)-based validation.
- Heuristic Score Sophistication: The score weightings could be further refined.
  - **Future Work:** Explore ML models trained on annotated summaries for a more nuanced risk score.

## 6. Conclusion: Towards Reliable Medical Summarization <a name="conclusion-towards-reliable-medical-summarization"></a>
Project Nightingale's HallucinationSurveyor function establishes a vital and robust framework for scrutinizing medical report summaries. Its OOP architecture underpins a series of targeted, interpretable checks that effectively identify critical errors including intrinsic and extrinsic hallucinations. As validated through controlled scenarios and LLM evaluations, the surveyor provides multi-faceted feedback essential for guiding the development and safe deployment of LLMs in the sensitive medical domain. While the current implementation provides a strong foundation, the outlined future enhancements point towards an even more powerful tool for ensuring that AI-driven summarization in healthcare is not only efficient but, most importantly, reliable and safe for patient care. The ability to systematically "survey" for hallucinations, as demonstrated by this project, is fundamental to achieving this goal.

## 7. Setup and Usage <a name="setup-and-usage"></a>
### Prerequisites:
- Python 3.7+
- Jupyter Notebook or a compatible environment (e.g., Google Colab).

### Installation:
Clone this repository:
```bash
git clone https://github.com/SaiSatyamJena/Project-Nightingale.git
cd Project-Nightingale
```
It's highly recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install the required packages using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Running the Notebook:
The core logic and demonstrations are contained within the Jupyter Notebook: `cleaned_notebook.ipynb` (or your final notebook name).
- Open this notebook in Jupyter Lab/Notebook or Google Colab and run the cells sequentially to observe the setup, class definition, test case evaluations, and LLM summary analyses.

### Key Components in the Notebook:
- **Chapter 2: Setup:** Handles all necessary imports (NLTK, scikit-learn, ROUGE, Transformers, PyTorch) and defines basic helper functions.
- **Chapter 3: Architecting the HallucinationSurveyor:** Details the OOP design and implementation of the `HallucinationSurveyor` class and its methods.
- **Chapter 4: Test Cases:** Provides controlled scenarios to validate each aspect of the surveyor, with detailed output interpretations.
- **Chapter 5: LLM Application:** Demonstrates the surveyor's use on summaries generated by actual LLMs (DistilBART and GPT-Neo).
- **Chapters 6 & 7: Discussion & Conclusion:** Summarize findings, strengths, limitations, and future directions.

The notebook includes detailed markdown explanations throughout, guiding the user through the project's development and evaluation.

### License
This project is licensed under the MIT License - see the `LICENSE.md` file for details (you will need to create this file with the MIT license text).
