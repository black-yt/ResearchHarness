# SGI-IdeaGeneration

This directory contains the ResearchHarness benchmark role overlay for
[SGI-IdeaGeneration](https://huggingface.co/datasets/InternScience/SGI-IdeaGeneration).

SGI-IdeaGeneration is a scientific proposal-generation benchmark. It is not a
generic summarization task, so the generic QA wrappers are not recommended.

## Recommended Server Command

```bash
python3 run_server.py \
  --api-runs-dir ./api_runs \
  --host 127.0.0.1 \
  --port 8686 \
  --role-prompt-file ./benchmarks/SGI-IdeaGeneration/role_prompt.md \
  --no-input-wrapper \
  --no-output-wrapper
```

## OpenAI Test Example

The example below embeds the first real `SGI-IdeaGeneration` test item directly
and appends the same proposal JSON example shape used by the official SGI-Bench
evaluation script. It does not require the `datasets` package.

````python
import json
from pathlib import Path
from openai import OpenAI


QUESTION = r'''You are a top-tier researcher in your field. Based on the following context, please generate a novel and detailed research proposal.

##Context:

###1. Related Work:
- Senior et al. (2020): Introduced deep learning for predicting inter-residue distances, improving template-free protein structure prediction but still reliant on multiple post-processing stages and lacking atomic-level accuracy for novel folds.
- Yang et al. (2020): Employed deep neural networks to predict inter-residue orientations, integrating orientation constraints but with limited end-to-end learning and lower performance on long or complex proteins.
- AlQuraishi (2019): Proposed an end-to-end differentiable structure prediction model, directly outputting 3D coordinates; however, it exhibited lower accuracy than multi-stage pipelines and struggled without homologous templates.
- Marks et al. (2011); Jones et al. (2012): Used coevolutionary analysis of MSAs to infer residue contacts, achieving improvements in contact prediction but failing to achieve accurate atomic models, especially for proteins lacking deep MSAs or templates.

###2. Challenge:
The core challenges in the field of protein structure prediction primarily revolve around achieving high accuracy and scalability in the absence of homologous templates or deep multiple sequence alignments (MSAs). First, existing models often separate the prediction of inter-residue contacts or distances from the actual structure generation, leading to an inefficient workflow that hinders end-to-end learning and integration of physical and evolutionary constraints. This separation complicates the direct optimization of 3D coordinates and results in suboptimal performance, particularly for long or complex proteins. Second, many contemporary approaches rely on hand-crafted features and multi-stage heuristics, which not only limits their scalability but also their adaptability to diverse protein architectures. Third, the reliance on comprehensive MSAs for accurate contact prediction poses a significant challenge, particularly for proteins with sparse or under-sampled sequences, where coevolutionary signals are weak or nonexistent. Lastly, the difficulty in accurately modeling multi-chain complexes exacerbates the challenges faced in predicting conformations that depend on intricate inter-chain interactions. Addressing these technical obstacles will be crucial for advancing the field towards achieving experimental-level accuracy in structure prediction.

###3. Limitations of Existing Approaches:
Contemporary approaches fall short of experimental accuracy, particularly on targets lacking homologous templates or deep MSAs. Existing neural architectures often separate contact/distance prediction from structure generation, use hand-crafted features, or rely on multi-stage heuristics, resulting in limited scalability and suboptimal integration of physical and evolutionary constraints. Poor performance persists in under-sampled sequence regions and multi-chain complexes.

###4. Motivation for New Research:
Structural biology is constrained by the slow pace and resource demands of experimental structure determination, leaving the vast majority of protein sequences without 3D structural annotation. Accurate, scalable, and generalizable computational prediction of protein structures—especially without close templates—would transform bioinformatics, molecular biology, and drug discovery by bridging the sequence-structure knowledge gap.

###5. Task Objective:
To develop a computational method that predicts the three-dimensional atomic structure of proteins from their amino acid sequence with accuracy comparable to experimental techniques, even in the absence of close structural homologues or deep sequence alignments.

###6. Existing Solutions:
- Physics-based simulation: Uses molecular dynamics or statistical approximations to model protein folding but is computationally intractable for large proteins and sensitive to approximations in physical modeling.
- Bioinformatics/homology modeling: Predicts structures via alignment to known protein templates and infers constraints from evolutionary sequence analysis; limited by template availability and reduced accuracy for novel or divergent proteins.
- Deep learning with intermediate prediction: Predicts inter-residue distances/orientations from MSAs using CNNs or attention networks, then reconstructs structures through downstream heuristics; accuracy suffers in end-to-end integration and novel folds.

##Your Task:

Based on the context above, please generate the following sections for a new research proposal. Be specific, clear, and innovative. Please limit the generated idea to 500 characters.

###1. Idea:
(Based on the above information, please propose an innovative and feasible idea. Include the required professional methods, reasoning, and logical development. Please limit your proposal to 300-500 characters.)

###2. ImplementationSteps:
(Provide a complete and specific list of implementation steps for your idea, and number each step.)

###3. ImplementationOrder:
(Provide a formatted execution route and a checklist of implementation steps. For example, "1-2", "2-3", "3-4", "4-5".)

###4. Dataset:
(Describe the dataset required for the evaluation. If it needs to be created, explain how.)

###5. EvaluationMetrics:
(Define specific, measurable metrics to evaluate the success of the project. Explain why these metrics are relevant.)

###6. ExpectedOutcome:
(Describe the anticipated results and their potential impact on the field. Compare them to the existing solutions.)'''

EXAMPLE = {
    "Idea": "We propose an adaptive optimization framework based on a dynamic feature interaction network. This framework captures feature correlations through a hierarchical attention mechanism and combines it with a data distribution-aware dynamic weight adjustment strategy to improve the model's adaptability to heterogeneous data while ensuring computational efficiency.",
    "ImplementationSteps": {
        "1": "Data preprocessing: missing value filling, outlier handling, feature normalization and type conversion, and building a basic feature set",
        "2": "Feature engineering: generating statistically derived features, time series features, and cross-features, and building a feature candidate pool",
        "3": "Model architecture design: building a basic network module, integrating a hierarchical attention mechanism with a dynamic interaction layer",
        "4": "Dynamic weight mechanism implementation: designing a data distribution-aware weight adjustment function and embedding it into the network's intermediate layers",
        "5": "Model training and tuning: adopting a phased training strategy, using grid search and early stopping to optimize hyperparameters",
        "6": "Performance Verification: Conduct comparative experiments on multiple datasets to analyze model performance differences in different scenarios.",
    },
    "ImplementationOrder": ["1-2", "2-3", "3-4", "4-5", "1-5", "5-6"],
    "Dataset": "Contains three types of public datasets and one actual business data: 1) Public structured dataset (approximately 500,000 samples, 30+ features); 2) Text-numeric mixed dataset (approximately 200,000 samples, including text embedding features); 3) Time series sparse dataset (approximately 100,000 samples, spanning 1 year); 4) Real transaction data from an e-commerce platform (approximately 1 million samples, including user behavior and product attribute features)",
    "EvaluationMetrics": {
        "Prediction Accuracy": "AUC and F1-score are used for classification tasks; MAE and RMSE are used for regression tasks to evaluate the basic predictive ability of the model.",
        "Robustness": "Performance decay rate is calculated through data perturbation testing (adding noise and simulating feature loss) to measure model stability.",
        "Efficiency": "Record model training time, inference latency, and memory usage to evaluate computing resource consumption.",
        "Interpretability": "Use SHAP values and feature importance ranking to quantify the feature contribution to model decisions.",
        "Generalization": "Performance retention across datasets to evaluate the model's adaptability to unseen data.",
    },
    "ExpectedOutcome": "The proposed framework outperforms existing mainstream methods in comprehensive performance (accuracy, robustness, and efficiency) across multiple datasets, particularly in scenarios with uneven data distribution and cross-scenario migration. It also enhances model interpretability through a dynamic feature interaction mechanism, providing effective support for practical business decision-making.",
}

prompt = QUESTION + f"""

### Example:
```json
{json.dumps(EXAMPLE, indent=4)}
```"""

workspace = Path("./workspace/sgi_ideageneration_example").resolve()
workspace.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key="unused", base_url="http://127.0.0.1:8686/v1")

response = client.chat.completions.create(
    model="RH",
    messages=[{"role": "user", "content": prompt}],
    extra_body={"workspace-root": str(workspace)},
)

print(response.choices[0].message.content)
````

## Rationale

- The benchmark parses a structured proposal with fixed fields.
- The scoring emphasizes effectiveness, novelty, detailedness, and feasibility.
- Input rewriting can weaken domain context or constraints.
- Output rewriting can break JSON or convert a proposal into a generic summary.
- The benchmark-specific role prompt should guide the agent to inspect recent
  literature, identify limitations, and propose a concrete new method directly.

Use `RH` or `RH--<llm-model-name>` in the OpenAI SDK request as usual.
