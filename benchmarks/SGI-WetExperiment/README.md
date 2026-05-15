# SGI-WetExperiment

This directory contains the ResearchHarness benchmark role overlay for
[SGI-WetExperiment](https://huggingface.co/datasets/InternScience/SGI-WetExperiment).

SGI-WetExperiment is a strict experimental-process formatting benchmark. It is
not a generic QA/VQA task, so the generic QA wrappers are not recommended.

## Recommended Server Command

```bash
python3 run_server.py \
  --api-runs-dir ./api_runs \
  --host 127.0.0.1 \
  --port 8686 \
  --role-prompt-file ./benchmarks/SGI-WetExperiment/role_prompt.md \
  --no-input-wrapper \
  --no-output-wrapper
```

## OpenAI Test Example

The example below embeds the first real `SGI-WetExperiment` test item directly
and appends the same action-sequence output requirement used by the official
SGI-Bench evaluation script. It does not require the `datasets` package.

```python
from pathlib import Path
from openai import OpenAI


QUESTION = r'''Please design an experimental process based on the research direction (enclosed within <research direction> and </research direction> tags) provided by the user. Please organize the actions in the Action Pool (enclosed within <action> and </action> tags) into an experimental process.

<research direction>
Metastatic urothelial bladder cancer (UBC) has historically lacked effective treatments beyond chemotherapy, which often yields limited benefit and substantial toxicity, especially in older patients with comorbidities. UBC is characterized by a high mutational burden, potentially increasing tumor antigenicity and immune recognition. However, tumors evade immune destruction partly through expression of programmed death-ligand 1 (PD-L1) in the tumor microenvironment, which inhibits T cell activity by engaging PD-1 receptors.

Targeting this immune checkpoint, a human engineered monoclonal antibody against PD-L1 has been developed to block its interaction with PD-1 and B7.1, thereby restoring anti-tumor immunity. This antibody is designed to avoid depleting PD-L1-expressing activated T cells by modifying its Fc domain to eliminate antibody-dependent cellular cytotoxicity. Clinical evaluation in metastatic UBC patients demonstrated notable anti-tumor activity, with rapid and durable responses observed, particularly in tumors exhibiting PD-L1 expression on tumor-infiltrating immune cells.

In a phase I adaptive trial, patients were initially selected based on PD-L1 positivity in immune cells but later included regardless of PD-L1 status. Approximately 27% of screened tumors showed PD-L1 positivity in immune infiltrates. Among treated patients, objective response rates (ORR) were significantly higher in those with PD-L1-positive immune cells (43%) compared to PD-L1-negative/low tumors (11%). Responses included complete remissions and were ongoing at data cutoff. The association between response and PD-L1 expression was significant for immune cells but not for tumor cells, highlighting the importance of the immune microenvironment.

The safety profile was favorable, with most adverse events being low grade and manageable; importantly, no renal toxicity was observed, a critical consideration given the frequent renal impairment in this population. Common side effects included fatigue and decreased appetite, likely related to immune activation. Immune-related adverse events were minimal.

Pharmacodynamic analyses revealed transient increases in immunostimulatory cytokines such as interleukin-18 and interferon-gamma, along with proliferation of activated CD8+ T cells, consistent with immune checkpoint blockade activity. These systemic immune changes were observed in all patients but did not directly correlate with clinical response.

This therapeutic approach addresses an unmet need for effective and tolerable treatments in metastatic UBC, especially for patients ineligible for or refractory to chemotherapy. The correlation of clinical benefit with PD-L1 expression on tumor-infiltrating immune cells suggests a potential biomarker for patient selection. The adaptive trial design facilitated rapid assessment of efficacy and biomarker relevance, supporting further clinical development. Overall, PD-L1 blockade represents a promising immunotherapeutic strategy in UBC, leveraging tumor immunogenicity and modulating immune suppression within the tumor microenvironment.
</research direction>

<action>
Action Pool:

<Screen patients for PD-L1 expression>(patient_tissue, antibody)
    Args:
        patient_tissue: Archived paraffin-embedded tissue sample
        antibody: Anti-human PD-L1 monoclonal antibody
    Returns:
        PD-L1 IHC score (0, 1, 2, or 3)

<Process tissue samples>(tissue_sample, processing_method)
    Args:
        tissue_sample: Formalin-fixed paraffin-embedded tumor tissue
        processing_method: Standard IHC staining protocol
    Returns:
        Processed tissue ready for scoring

<Score PD-L1 expression>(stained_tissue, cell_type)
    Args:
        stained_tissue: IHC-stained tissue sample
        cell_type: Tumor cells or tumor-infiltrating immune cells
    Returns:
        IHC score based on percentage of positive cells

<Administer MPDL3280A treatment>(patient, dose, schedule)
    Args:
        patient: Eligible UBC patient
        dose: Dosage in mg/kg
        schedule: Treatment schedule (e.g., q3w)
    Returns:
        Treatment administration record

<Monitor patient safety>(patient, timepoint)
    Args:
        patient: Treated patient
        timepoint: Assessment time point
    Returns:
        Adverse event data and safety profile

<Perform radiological assessment>(patient, imaging_method, timepoint)
    Args:
        patient: Treated patient
        imaging_method: CT or other imaging modality
        timepoint: Week of assessment
    Returns:
        Tumor response data per RECIST v1.1

<Collect blood samples>(patient, collection_time, tube_type)
    Args:
        patient: Study participant
        collection_time: Pre-dose or post-dose timepoint
        tube_type: Collection tube specification
    Returns:
        Blood sample for analysis

<Analyze cytokine levels>(plasma_sample, cytokine_panel)
    Args:
        plasma_sample: Patient plasma sample
        cytokine_panel: Target cytokines (IL-18, IFN-γ)
    Returns:
        Cytokine concentration data

<Perform flow cytometry>(blood_sample, markers)
    Args:
        blood_sample: Whole blood sample
        markers: CD3, CD8, HLA-DR, Ki-67
    Returns:
        Cell population percentages

<Evaluate objective response>(patient_data, criteria)
    Args:
        patient_data: Complete patient assessment data
        criteria: RECIST v1.1 or irRC
    Returns:
        Response classification (CR, PR, SD, PD)

<Calculate response rates>(cohort_data, IHC_status)
    Args:
        cohort_data: All patient response data
        IHC_status: PD-L1 IHC grouping (0/1 or 2/3)
    Returns:
        Objective response rate with confidence interval

<Assess duration of response>(responder_data, followup_period)
    Args:
        responder_data: Data from responding patients
        followup_period: Time from response to progression
    Returns:
        Duration of response metrics

<Analyze safety data>(adverse_events, grade_criteria)
    Args:
        adverse_events: All reported adverse events
        grade_criteria: CTCAE version 4.0
    Returns:
        Graded safety profile summary

<Perform statistical analysis>(dataset, statistical_method)
    Args:
        dataset: Complete study data
        statistical_method: Specified statistical approach
    Returns:
        Statistical results and p-values


</action>'''

OUTPUT_REQUIREMENTS = """
The final answer should be enclosed by <answer> and </answer>.

Example:
<answer>
dataset = <Load dataset>(
    source="imagenet"
)

model_init = <Initialize model>(
    model_type="CNN"
)

model_trained = <Train model>(
    model=model_init,
    data=dataset
)

metrics = <Calculate metrics>(
    model=model_trained,
    data=dataset
)
</answer>
"""

prompt = QUESTION + OUTPUT_REQUIREMENTS

workspace = Path("./workspace/sgi_wetexperiment_example").resolve()
workspace.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key="unused", base_url="http://127.0.0.1:8686/v1")

response = client.chat.completions.create(
    model="RH",
    messages=[{"role": "user", "content": prompt}],
    extra_body={"workspace-root": str(workspace)},
)

print(response.choices[0].message.content)
```

## Rationale

- The scorer expects a strict action-call text format.
- Output rewriting can collapse required multi-line calls into one-line calls.
- Input rewriting can lose action-pool entries or move them away from the model
  context.
- The benchmark-specific role prompt should guide the agent to draft, validate,
  and return the final action sequence directly.

## Local Format Validator

The role prompt asks the agent to validate against the same broad shape used by
the benchmark parser:

```python
import re


def parse_experiment_steps(text):
    step_pattern = r'(\w+)\s*=\s*<([^>]+)>\(\s*([\s\S]*?)(?=\n\s*\)\s*$)'
    param_pattern = r'^\s*(\w+)\s*=\s*(.*?)\s*(?:,)?\s*$'
    steps = []

    for match in re.finditer(step_pattern, text, re.MULTILINE):
        output_var = match.group(1).strip()
        action_name = match.group(2).strip()
        params = match.group(3).strip()

        param_dict = {}
        param_lines = [line.strip() for line in params.split('\n') if line.strip() and line.strip() != ')']
        for line in param_lines:
            param_match = re.match(param_pattern, line)
            if param_match:
                key = param_match.group(1)
                value = param_match.group(2).strip()
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                param_dict[key] = value

        steps.append({"action": action_name, "input": param_dict, "output": output_var})

    return steps
```

This validator is documentation for the expected output shape. The actual
benchmark runner remains responsible for scoring.
