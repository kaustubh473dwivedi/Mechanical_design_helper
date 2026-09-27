# Generative AI for Automated Mechanical Design

### B.Tech Project — Department of Mechanical and Industrial Engineering, IIT Roorkee
**Academic Year:** 2025–2026

> A domain-specific generative AI framework that translates natural-language engineering requirements into structured mechanical design parameters and CAD-ready 3D outputs.

---

## Overview

Traditional mechanical design workflows require engineers to manually interpret engineering requirements, select appropriate design parameters, perform calculations, and construct CAD models.

This project investigates whether a large language model can learn engineering design relationships and convert high-level natural-language requirements into structured mechanical design parameters suitable for downstream CAD generation.

The system fine-tunes **LLaMA-3-8B** on a domain-specific mechanical engineering dataset using **QLoRA**, and implements an end-to-end pipeline:


Engineering Requirement
          ↓
   Fine-tuned LLaMA-3
          ↓
    Structured JSON
          ↓
   Parameter Validation
          ↓
     CAD Generation
          ↓
       STL Model
          ↓
   3D Visualization

The work focuses on mechanical components and mechanisms including:

Spur gears
Bolted joints
Nuts and bolts
Piston-cylinder systems
Four-bar mechanisms
Objectives

The primary objectives of the project were:

Develop a domain-specific dataset connecting engineering requirements with mechanical design parameters.
Fine-tune LLaMA-3-8B for mechanical design reasoning using parameter-efficient fine-tuning.
Convert natural-language engineering requirements into structured JSON representations.
Incorporate engineering constraints such as load, torque, speed, safety factor, material and geometric limits.
Provide a rule-based fallback mechanism for malformed or incomplete model outputs.
Connect the generated parameters to downstream CAD/mesh generation.
Evaluate the generated designs through representative engineering design cases.
System Architecture

The overall system follows a natural-language → structured parameters → CAD workflow.

                    ┌──────────────────────────┐
                    │ Natural Language Prompt │
                    │ Engineering Requirement │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     Fine-tuned LLaMA-3   │
                    │          8B Model        │
                    │        QLoRA / PEFT      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     Structured JSON      │
                    │     Design Parameters   │
                    └────────────┬─────────────┘
                                 │
                         JSON Validation
                                 │
                    ┌────────────┴─────────────┐
                    │                          │
                    ▼                          ▼
             Valid JSON                Regex Fallback
                    │                          │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    CAD / Mesh Generation │
                    │        Meshy.ai          │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       STL Output         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     3D Visualization     │
                    │         Plotly           │
                    └──────────────────────────┘
Dataset

The dataset consists of engineering requirement–design parameter pairs covering several mechanical systems.

Mechanical Systems
System	Example Design Parameters
Spur Gear	Module, number of teeth, gear ratio, face width
Bolted Joint	Diameter, pitch, thread dimensions, grade
Nut & Bolt	Size, pitch, grade, dimensions
Piston-Cylinder	Bore, stroke, rod length, speed
Four-Bar Mechanism	Link lengths, configuration, motion type

Each example represents a mapping between engineering requirements and corresponding design parameters.

Input Variables

The generated engineering cases include parameters such as:

Power
Torque
Rotational speed
Tensile load
Safety factor
Material constraints
Geometric constraints
Stress limits
Motion constraints
Desired operating life

The dataset was generated using engineering relationships, formulas and design constraints to create diverse design cases.

Engineering Constraints

The project incorporates basic mechanical design relationships into the generation and validation process.

Examples include:

Gear Design

Gear-related cases consider:

Power
Rotational speed
Number of teeth
Module
Gear ratio
Bending stress
Contact stress
Geometric constraints
Bolted Joint Design

Bolted-joint cases consider:

Tensile load
Safety factor
Bolt grade
Tensile stress area
Thread dimensions
Standard bolt sizing
Four-Bar Mechanisms

Mechanism examples consider:

Fixed-link length
Crank length
Rocker length
Input motion
Output motion
Grashof condition
Model and Fine-Tuning
Base Model

LLaMA-3-8B

The model was fine-tuned specifically for mechanical engineering design parameter generation.

Parameter-Efficient Fine-Tuning

The project uses:

QLoRA
LoRA
PEFT
4-bit quantization

QLoRA was used to make fine-tuning feasible on limited GPU resources.

The training setup used a single T4 GPU with approximately 14.5 GB available memory.

Training Configuration
Parameter	Configuration
Base Model	LLaMA-3-8B
Fine-tuning	QLoRA
Quantization	4-bit
Optimizer	AdamW 8-bit
Learning Rate	2 × 10⁻⁴
Scheduler	Cosine Decay
Batch Size	10
Training Steps	300
Loss	Cross Entropy
Hardware	Google Colab T4 GPU
Training Results

The training loss decreased substantially during the initial training phase.

The model was selected around the region where the loss had decreased significantly before the later rise in the training curve.

The curve shows the training loss decreasing from approximately 2.0 to around 0.3 during the main learning phase, followed by an increase toward approximately 0.6 near the end of the run.

This behavior was considered when selecting the model checkpoint for downstream modelling experiments.

Experimental Results

The following examples are taken from the project evaluation and demonstrate the complete pipeline from engineering prompt → generated parameters → 3D output.

1. Bolt Selection
Input
User request:

Select a bolt (nominal_diameter_mm and grade) that can sustain
tensile load = 2709 N with safety factor >= 2.500.

Return JSON 'bolt_selection'.
Generated Output
{
  "diameter_mm": 10,
  "length_mm": 50,
  "pitch_mm": 1.5,
  "head_height_mm": 6.4,
  "head_diameter_mm": 17,
  "thread": {
    "major_diameter": 10,
    "pitch_diameter": 9.02575,
    "minor_diameter": 8.15965,
    "thread_height": 1.299
  }
}
Generated 3D Result

The model generated a structured set of bolt and thread parameters which were subsequently used for downstream 3D generation.

2. Gear Performance
Input
Required power transfer: rotational power ≈ 64.3 W.

Given gear geometry:
number_of_teeth_pinion = 30
number_of_teeth_gear = 30
input speed rpm = 1213

Compute gear performance:
efficiency,
input_power_kw,
recommended_face_width_mm,
and lubrication.

Return a JSON object named 'gear_performance'.
Generated Output
{
  "gear_type": "worm",
  "module_mm": 0.8,
  "number_of_teeth_pinion": 30,
  "number_of_teeth_gear": 30,
  "input_torque_Nm": 3334.6,
  "input_speed_rpm": 1213,
  "input_power_kw": 8.1152,
  "helix_angle_deg": null,
  "recommended_face_width_mm": 7.98,
  "module_candidates": [0.8],
  "lubrication_recommendation": "ISO VG 460 (heavy)"
}
Generated 3D Result

This example demonstrates the model's ability to generate structured gear-design parameters from a natural-language engineering requirement.

3. Heavy Hex Nut
Input
Design a heavy hex nut for M20 bolt used in flange coupling
under high vibration.
Generated Output
{
  "size": "M10",
  "length_mm": 12.6,
  "pitch_mm": 1.5,
  "grade": "10.9"
}
Generated 3D Result

Observed Limitation

The requested component was an M20 nut, while the generated result specifies M10.

This is an important observed failure case rather than a result that should be hidden.

It demonstrates that although the model can produce structurally valid JSON and CAD parameters, semantic constraint adherence is not guaranteed.

This motivates further work on:

Constraint-aware decoding
Engineering-rule validation
Retrieval-augmented generation
Post-generation parameter verification
FEA-based validation
4. Four-Bar Mechanism
Input
Design a four-bar mechanism to achieve crank-rocker motion
with a fixed link of length 120 mm.

The crank should rotate fully, while the output link oscillates.

Provide link lengths and configuration details as a JSON object.
Generated Output
{
  "mechanism_type": "crank-rocker",
  "fixed_link_mm": 120,
  "input_crank_link_length": 110,
  "output_rocker_link_length": 90,
  "input_motion": "full_rotation",
  "output_motion": "oscillatory",
  "configuration": "Grashof_condition_satisfied"
}
Generated Result

The model correctly structured the requested mechanism type, fixed link and motion requirements in the demonstrated case.

5. Piston-Cylinder Design
Input
Required power transfer: indicated power ≈ 44.6 W

Peak pressure = 17.67 bar
Swept volume = 1.848932 cc
Speed = 819 rpm

Connecting rod length = 590.055 mm
Rod ratio = 3.609
Stroke = 163.482 mm

Determine the missing value (bore_mm).

Return JSON object piston_design.
Generated Output
{
  "piston_design": {
    "bore_mm": 120.0,
    "mechanism_type": "piston_cylinder"
  }
}
Generated Result

This example demonstrates the use of the model for parameter completion from a partially specified mechanical system.

Representative Engineering Case Studies

In addition to the broader demonstrations above, the project report evaluates representative engineering design cases.

Spur Gear

For a representative case:

Power = 20 kW
Speed = 1500 rpm
Safety Factor = 1.4

The generated design included:

Module = 4
Pinion teeth = 22
Gear teeth = 66

The resulting geometry was reported as feasible under the specified outer-diameter constraint.

Bolted Joint

For a representative bolted-joint case:

Tensile Load = 120 kN
Bolt Grade = 12.9

The generated design selected approximately:

Bolt Size = M20
Pitch = 2.5 mm
Stress Area ≈ 245 mm²

These values were consistent with the intended standard design selection in the project evaluation.

Robustness: Rule-Based Fallback

A practical issue with generative models is that model responses may not always follow the required JSON schema.

To improve robustness, the inference pipeline includes a regex-based fallback parser.

                 Model Output
                      │
                      ▼
                JSON Parsing
                      │
              ┌───────┴────────┐
              │                │
            Valid            Invalid
              │                │
              ▼                ▼
        Use JSON Output    Regex Fallback
                               │
                               ▼
                        Recover Parameters
                               │
                               ▼
                         CAD Pipeline

This allows selected parameters to be recovered even when the generated response does not strictly follow the expected JSON format.

End-to-End CAD Generation

The final system connects the generated parameters to downstream 3D generation.

Natural Language
      ↓
Fine-tuned LLaMA-3
      ↓
Structured JSON
      ↓
Parameter Validation
      ↓
CAD Parameters
      ↓
Mesh/CAD Generation
      ↓
STL
      ↓
3D Visualization

The demonstrated workflow produces downloadable STL-ready outputs without requiring manual CAD modelling for the demonstrated cases.

Key Contributions

The project demonstrates the following:

Domain-specific fine-tuning of LLaMA-3-8B for mechanical design.
QLoRA-based training on a single T4 GPU.
Natural-language engineering requirement → structured design parameter generation.
JSON-based intermediate representation for downstream CAD generation.
Rule-based fallback parsing for malformed model responses.
Integration with 3D/CAD generation.
Demonstrations across gears, bolts, nuts, piston-cylinder systems and four-bar mechanisms.
Explicit evaluation of both successful generations and failure cases.
Limitations

The current system should be considered a research prototype, rather than a replacement for engineering verification.

Important limitations include:

Generated parameters are not universally guaranteed to satisfy engineering constraints.
The demonstrated M20 nut case produced an M10 output.
The training dataset is substantially smaller and more controlled than real industrial CAD datasets.
Generated designs require further engineering verification before practical deployment.
The current workflow does not perform full finite-element analysis.
CAD generation relies on a downstream mesh/CAD generation component rather than directly generating native parametric CAD models.
The model may produce syntactically valid JSON even when one or more engineering constraints are violated.

These limitations motivate the next stage of the research.

Future Work

Several directions can extend this work toward a more rigorous engineering design system.

1. Industrial CAD Datasets

Train and evaluate using larger real-world engineering datasets such as:

GrabCAD
McMaster-Carr
Standard mechanical component databases
Industrial CAD repositories
2. Physics-Aware Generation

Integrate engineering equations directly into the generation and validation loop.

LLM Generation
      ↓
Engineering Constraint Checker
      ↓
Physics / Stress Validation
      ↓
Accept / Regenerate
3. Finite Element Analysis

Automatically perform FEA after parameter generation to verify:

Stress
Deformation
Factor of safety
Contact behaviour
Fatigue-related constraints
4. Native Parametric CAD

Future systems could directly generate:

STEP
IGES
SolidWorks-compatible models
Parametric CAD features

instead of relying primarily on mesh generation.

5. Constraint-Aware Decoding

Engineering constraints could be incorporated during generation rather than checked only after generation.

6. CAD API Integration

A future implementation could connect the model directly with CAD software APIs to create editable parametric models automatically.

Technology Stack
Machine Learning
LLaMA-3-8B
Hugging Face Transformers
PEFT
LoRA
QLoRA
4-bit Quantization
Engineering
Mechanical Design
Machine Design
Gear Design
Bolted Joint Design
Mechanism Design
Engineering Constraints
CAD / Visualization
Meshy.ai
STL
Plotly 3D
Development
Python
Google Colab
PyTorch
JSON
Regex-based parsing
Repository

GitHub Repository

Mechanical Design Helper

Dataset

The project dataset is available here:

Download Dataset

Project Team

Department of Mechanical and Industrial Engineering
Indian Institute of Technology Roorkee

Name	Roll Number	Contribution
Aman Kumar	22117019	Fine-tuning & 3D CAD
Kaustubh Dwivedi	22117066	Dataset curation & efficiency improvement
Samay Jain	22117124	Data preprocessing
Akshay Kumar	22117015	Dataset curation
Priyanshu	22117110	Data preprocessing

Supervisor: Prof. Anuj Bisht

Summary

This project explores the use of generative AI for translating natural-language mechanical engineering requirements into structured design parameters and CAD-ready outputs.

A domain-specific LLaMA-3-8B model was fine-tuned using QLoRA, enabling training on a single T4 GPU. The resulting pipeline combines language-model generation, structured JSON representation, engineering constraints, fallback parsing and downstream 3D generation.

The experimental results demonstrate the feasibility of using an LLM as an interface between natural-language engineering requirements and computational mechanical design workflows, while also exposing important limitations in constraint adherence and engineering verification.

The next step toward a production-grade system is to combine generative modelling with formal engineering constraints, physics-based validation, FEA and native parametric CAD generation.


### One important thing

When you paste this into GitHub, **do not paste the images themselves into the README editor**. Just upload the six `.png` files into:


Mechanical_design_helper/
└── assets/
    ├── bolt_result.png
    ├── gear_result.png
    ├── nut_result.png
    ├── fourbar_result.png
    ├── piston_result.png
    └── training_loss_curve.png

Then the image syntax will work automatically:

![Training loss curve](assets/training_loss_curve.png)

The same applies to all five experimental-result images.

Download Dataset
Google Drive
you are confusing me more than anthing else 
i want a single markdown code forthe readme.md file 
i will just copy and paste in one go
# Generative AI for Automated Mechanical Design

### B.Tech Project — Department of Mechanical and Industrial Engineering, IIT Roorkee
**Academic Year:** 2025–2026

> A domain-specific generative AI framework that translates natural-language engineering requirements into structured mechanical design parameters and CAD-ready 3D outputs.

---

## Overview

Traditional mechanical design workflows require engineers to manually interpret engineering requirements, select appropriate design parameters, perform calculations, and construct CAD models.

This project investigates whether a large language model can learn engineering design relationships and convert high-level natural-language requirements into structured mechanical design parameters suitable for downstream CAD generation.

The system fine-tunes **LLaMA-3-8B** on a domain-specific mechanical engineering dataset using **QLoRA**, and implements an end-to-end pipeline:


Engineering Requirement
          ↓
   Fine-tuned LLaMA-3
          ↓
    Structured JSON
          ↓
   Parameter Validation
          ↓
     CAD Generation
          ↓
       STL Model
          ↓
   3D Visualization

The work focuses on mechanical components and mechanisms including:

Spur gears
Bolted joints
Nuts and bolts
Piston-cylinder systems
Four-bar mechanisms
Objectives

The primary objectives of the project were:

Develop a domain-specific dataset connecting engineering requirements with mechanical design parameters.
Fine-tune LLaMA-3-8B for mechanical design reasoning using parameter-efficient fine-tuning.
Convert natural-language engineering requirements into structured JSON representations.
Incorporate engineering constraints such as load, torque, speed, safety factor, material and geometric limits.
Implement a rule-based fallback mechanism for malformed or incomplete model outputs.
Connect generated parameters to downstream CAD/mesh generation.
Evaluate generated designs through representative engineering design cases.
System Architecture
                    ┌──────────────────────────┐
                    │ Natural Language Prompt │
                    │ Engineering Requirement │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     Fine-tuned LLaMA-3   │
                    │          8B Model        │
                    │        QLoRA / PEFT      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     Structured JSON      │
                    │     Design Parameters   │
                    └────────────┬─────────────┘
                                 │
                         JSON Validation
                                 │
                    ┌────────────┴─────────────┐
                    │                          │
                    ▼                          ▼
             Valid JSON                Regex Fallback
                    │                          │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    CAD / Mesh Generation │
                    │        Meshy.ai          │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       STL Output         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     3D Visualization     │
                    │         Plotly           │
                    └──────────────────────────┘
Dataset

The dataset consists of engineering requirement–design parameter pairs covering several mechanical systems.

System	Example Design Parameters
Spur Gear	Module, number of teeth, gear ratio, face width
Bolted Joint	Diameter, pitch, thread dimensions, grade
Nut & Bolt	Size, pitch, grade, dimensions
Piston-Cylinder	Bore, stroke, rod length, speed
Four-Bar Mechanism	Link lengths, configuration, motion type

Each sample represents a mapping between engineering requirements and corresponding design parameters.

Input Variables

The generated engineering cases include:

Power
Torque
Rotational speed
Tensile load
Safety factor
Material constraints
Geometric constraints
Stress limits
Motion constraints
Desired operating life

The dataset was generated using engineering relationships, formulas and design constraints to create diverse design cases.

Engineering Constraints

The project incorporates basic mechanical design relationships into the generation and validation process.

Gear Design

Gear-related cases consider:

Power
Rotational speed
Number of teeth
Module
Gear ratio
Bending stress
Contact stress
Geometric constraints
Bolted Joint Design

Bolted-joint cases consider:

Tensile load
Safety factor
Bolt grade
Tensile stress area
Thread dimensions
Standard bolt sizing
Four-Bar Mechanisms

Mechanism examples consider:

Fixed-link length
Crank length
Rocker length
Input motion
Output motion
Grashof condition
Model and Fine-Tuning
Base Model

LLaMA-3-8B

The model was fine-tuned specifically for mechanical engineering design parameter generation.

Parameter-Efficient Fine-Tuning

The project uses:

QLoRA
LoRA
PEFT
4-bit quantization

QLoRA was used to make fine-tuning feasible on limited GPU resources.

The training setup used a single T4 GPU with approximately 14.5 GB available memory.

Training Configuration
Parameter	Configuration
Base Model	LLaMA-3-8B
Fine-tuning	QLoRA
Quantization	4-bit
Optimizer	AdamW 8-bit
Learning Rate	2 × 10⁻⁴
Scheduler	Cosine Decay
Batch Size	10
Training Steps	300
Loss	Cross Entropy
Hardware	Google Colab T4 GPU
Training Results

The training loss decreased substantially during the initial training phase.

The curve shows the training loss decreasing from approximately 2.0 to around 0.3 during the main learning phase, followed by an increase toward approximately 0.6 near the end of the run.

This behavior was considered when selecting the model checkpoint for downstream modelling experiments.

Experimental Results

The following examples are taken from the project evaluation and demonstrate the complete pipeline from engineering prompt → generated parameters → 3D output.

1. Bolt Selection
Input
User request:

Select a bolt (nominal_diameter_mm and grade) that can sustain
tensile load = 2709 N with safety factor >= 2.500.

Return JSON 'bolt_selection'.
Generated Output
{
  "diameter_mm": 10,
  "length_mm": 50,
  "pitch_mm": 1.5,
  "head_height_mm": 6.4,
  "head_diameter_mm": 17,
  "thread": {
    "major_diameter": 10,
    "pitch_diameter": 9.02575,
    "minor_diameter": 8.15965,
    "thread_height": 1.299
  }
}
Generated 3D Result

The model generated a structured set of bolt and thread parameters which were subsequently used for downstream 3D generation.

2. Gear Performance
Input
Required power transfer: rotational power ≈ 64.3 W.

Given gear geometry:
number_of_teeth_pinion = 30
number_of_teeth_gear = 30
input speed rpm = 1213

Compute gear performance:
efficiency,
input_power_kw,
recommended_face_width_mm,
and lubrication.

Return a JSON object named 'gear_performance'.
Generated Output
{
  "gear_type": "worm",
  "module_mm": 0.8,
  "number_of_teeth_pinion": 30,
  "number_of_teeth_gear": 30,
  "input_torque_Nm": 3334.6,
  "input_speed_rpm": 1213,
  "input_power_kw": 8.1152,
  "helix_angle_deg": null,
  "recommended_face_width_mm": 7.98,
  "module_candidates": [0.8],
  "lubrication_recommendation": "ISO VG 460 (heavy)"
}
Generated 3D Result

This example demonstrates the model's ability to generate structured gear-design parameters from a natural-language engineering requirement.

3. Heavy Hex Nut
Input
Design a heavy hex nut for M20 bolt used in flange coupling
under high vibration.
Generated Output
{
  "size": "M10",
  "length_mm": 12.6,
  "pitch_mm": 1.5,
  "grade": "10.9"
}
Generated 3D Result

Observed Limitation

The requested component was an M20 nut, while the generated result specifies M10.

This is an important observed failure case rather than a result that should be hidden.

It demonstrates that although the model can produce structurally valid JSON and CAD parameters, semantic constraint adherence is not guaranteed.

This motivates further work on:

Constraint-aware decoding
Engineering-rule validation
Retrieval-augmented generation
Post-generation parameter verification
FEA-based validation
4. Four-Bar Mechanism
Input
Design a four-bar mechanism to achieve crank-rocker motion
with a fixed link of length 120 mm.

The crank should rotate fully, while the output link oscillates.

Provide link lengths and configuration details as a JSON object.
Generated Output
{
  "mechanism_type": "crank-rocker",
  "fixed_link_mm": 120,
  "input_crank_link_length": 110,
  "output_rocker_link_length": 90,
  "input_motion": "full_rotation",
  "output_motion": "oscillatory",
  "configuration": "Grashof_condition_satisfied"
}
Generated Result

The model correctly structured the requested mechanism type, fixed link and motion requirements in the demonstrated case.

5. Piston-Cylinder Design
Input
Required power transfer: indicated power ≈ 44.6 W

Peak pressure = 17.67 bar
Swept volume = 1.848932 cc
Speed = 819 rpm

Connecting rod length = 590.055 mm
Rod ratio = 3.609
Stroke = 163.482 mm

Determine the missing value (bore_mm).

Return JSON object piston_design.
Generated Output
{
  "piston_design": {
    "bore_mm": 120.0,
    "mechanism_type": "piston_cylinder"
  }
}
Generated Result

This example demonstrates the use of the model for parameter completion from a partially specified mechanical system.

Representative Engineering Case Studies

In addition to the broader demonstrations above, the project report evaluates representative engineering design cases.

Spur Gear

For a representative case:

Power = 20 kW
Speed = 1500 rpm
Safety Factor = 1.4

The generated design included:

Module = 4
Pinion teeth = 22
Gear teeth = 66

The resulting geometry was reported as feasible under the specified outer-diameter constraint.

Bolted Joint

For a representative bolted-joint case:

Tensile Load = 120 kN
Bolt Grade = 12.9

The generated design selected approximately:

Bolt Size = M20
Pitch = 2.5 mm
Stress Area ≈ 245 mm²

These values were consistent with the intended standard design selection in the project evaluation.

Robustness: Rule-Based Fallback

A practical issue with generative models is that model responses may not always follow the required JSON schema.

To improve robustness, the inference pipeline includes a regex-based fallback parser.

                 Model Output
                      │
                      ▼
                JSON Parsing
                      │
              ┌───────┴────────┐
              │                │
            Valid            Invalid
              │                │
              ▼                ▼
        Use JSON Output    Regex Fallback
                               │
                               ▼
                        Recover Parameters
                               │
                               ▼
                         CAD Pipeline

This allows selected parameters to be recovered even when the generated response does not strictly follow the expected JSON format.

End-to-End CAD Generation

The final system connects the generated parameters to downstream 3D generation.

Natural Language
      ↓
Fine-tuned LLaMA-3
      ↓
Structured JSON
      ↓
Parameter Validation
      ↓
CAD Parameters
      ↓
Mesh/CAD Generation
      ↓
STL
      ↓
3D Visualization

The demonstrated workflow produces downloadable STL-ready outputs without requiring manual CAD modelling for the demonstrated cases.

Key Contributions

The project demonstrates:

Domain-specific fine-tuning of LLaMA-3-8B for mechanical design.
QLoRA-based training on a single T4 GPU.
Natural-language engineering requirement → structured design parameter generation.
JSON-based intermediate representation for downstream CAD generation.
Rule-based fallback parsing for malformed model responses.
Integration with 3D/CAD generation.
Demonstrations across gears, bolts, nuts, piston-cylinder systems and four-bar mechanisms.
Explicit evaluation of both successful generations and failure cases.
Limitations

The current system should be considered a research prototype, rather than a replacement for engineering verification.

Important limitations include:

Generated parameters are not universally guaranteed to satisfy engineering constraints.
The demonstrated M20 nut case produced an M10 output.
The training dataset is substantially smaller and more controlled than real industrial CAD datasets.
Generated designs require further engineering verification before practical deployment.
The current workflow does not perform full finite-element analysis.
CAD generation relies on a downstream mesh/CAD generation component rather than directly generating native parametric CAD models.
The model may produce syntactically valid JSON even when one or more engineering constraints are violated.

These limitations motivate the next stage of the research.

Future Work
1. Industrial CAD Datasets

Train and evaluate using larger real-world engineering datasets such as:

GrabCAD
McMaster-Carr
Standard mechanical component databases
Industrial CAD repositories
2. Physics-Aware Generation

Integrate engineering equations directly into the generation and validation loop.

LLM Generation
      ↓
Engineering Constraint Checker
      ↓
Physics / Stress Validation
      ↓
Accept / Regenerate
3. Finite Element Analysis

Automatically perform FEA after parameter generation to verify:

Stress
Deformation
Factor of safety
Contact behaviour
Fatigue-related constraints
4. Native Parametric CAD

Future systems could directly generate:

STEP
IGES
SolidWorks-compatible models
Parametric CAD features

instead of relying primarily on mesh generation.

5. Constraint-Aware Decoding

Engineering constraints could be incorporated during generation rather than checked only after generation.

6. CAD API Integration

A future implementation could connect the model directly with CAD software APIs to create editable parametric models automatically.

Technology Stack
Machine Learning
LLaMA-3-8B
Hugging Face Transformers
PEFT
LoRA
QLoRA
4-bit Quantization
Engineering
Mechanical Design
Machine Design
Gear Design
Bolted Joint Design
Mechanism Design
Engineering Constraints
CAD / Visualization
Meshy.ai
STL
Plotly 3D
Development
Python
Google Colab
PyTorch
JSON
Regex-based parsing
Repository

Mechanical Design Helper — GitHub

Dataset

Project Dataset — Google Drive

Project Team

Department of Mechanical and Industrial Engineering
Indian Institute of Technology Roorkee

Name	Roll Number	Contribution
Aman Kumar	22117019	Fine-tuning & 3D CAD
Kaustubh Dwivedi	22117066	Dataset curation & efficiency improvement
Samay Jain	22117124	Data preprocessing
Akshay Kumar	22117015	Dataset curation
Priyanshu	22117110	Data preprocessing

Supervisor: Prof. Anuj Bisht

Summary

This project explores the use of generative AI for translating natural-language mechanical engineering requirements into structured design parameters and CAD-ready outputs.

A domain-specific LLaMA-3-8B model was fine-tuned using QLoRA, enabling training on a single T4 GPU. The resulting pipeline combines generative modelling, structured JSON representation, engineering constraints, fallback parsing and downstream 3D generation.

The experimental results demonstrate the feasibility of using an LLM as an interface between natural-language engineering requirements and computational mechanical design workflows, while also exposing important limitations in constraint adherence and engineering verification.

The next step toward a more rigorous engineering design system is to combine generative modelling with formal engineering constraints, physics-based validation, FEA and native parametric CAD generation.
