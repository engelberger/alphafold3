# Visual Introduction to AlphaFold 3 Binder Design Protocols
This document provides a visual overview of the two main binder design protocols implemented in this codebase: `binder_gradient` and `binder_boltz`. These diagrams illustrate the high-level flow, key steps, and relevant code components for each approach.
## Overall Workflow
Both binder design protocols share initial setup steps before diverging into their respective optimization loops.
```mermaid
graph TD
    A[Start: run_alphafold.py] --> B{Parse Arguments}
    B -- binder_gradient / binder_boltz --> C[Load folding_input.Input]
    C --> D[Run Initial Featurisation]
    D --> E[Setup Binder Features]
    E --> F{Select Protocol}
    F -- protocol=binder_gradient --> G[Run Gradient Design Loop]
    F -- protocol=binder_boltz --> H[Run Boltz Design Loop]
    G --> I[Get Final Sequence]
    H --> I
    I --> J[Run Final Full Prediction]
    J --> K[Save Results]
    
    subgraph Common Setup
        B
        C
        D
        E
    end
    
    subgraph Design Loops
        direction LR
        G
        H
    end
    
    subgraph Final Steps
        direction LR
        I
        J
        K
    end
    
    style G fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#ccf,stroke:#333,stroke-width:2px
```
**Key Files Involved:**
*   `run_alphafold.py`: Main script, argument parsing, orchestrates the overall process, calls featurisation and design.
*   `src/alphafold3/common/folding_input.py`: Defines the `Input` object structure.
*   `src/alphafold3/data/featurisation.py`: Handles the conversion of `Input` to `feature_dict`.
*   `src/alphafold3/design/binder_utils.py`: Contains helper functions like `setup_binder_features` and `update_features_from_logits`.
*   `src/alphafold3/design/binder_design.py`: Implements the core `BinderDesigner` class and the `_design_binder_gradient` / `_design_binder_boltz` methods.
*   `src/alphafold3/design/binder_loss.py`: Defines the loss calculation logic for both protocols.
*   `src/alphafold3/model/model.py`: Defines the core `Model` architecture.
## Protocol 1: `binder_gradient` (ColabDesign-like)
This approach uses the full AlphaFold 3 model output for loss calculation and backpropagates gradients through the entire network, including the structure module.
```mermaid
graph TD
    subgraph GradientDesignLoop
        StartLoop[Start Gradient Loop] --> UpdateFeats
        UpdateFeats[Update Features] --> RunModel
        RunModel[Run Full AF3 Model] --> CalcLoss
        CalcLoss[Calculate Gradient Loss]
        CalcLoss -- Uses pLDDT, PAE, Contacts --> Grads
        Grads[Compute Gradients] --> UpdateLogits
        UpdateLogits[Update Sequence Logits] --> LoopCheck
        LoopCheck{More Steps?} -- Yes --> UpdateFeats
        LoopCheck -- No --> EndLoop[End Gradient Loop]
        
        Note1[Based on final predicted structure metrics]
    end
    
    style RunModel fill:#f9f,stroke:#333,stroke-width:2px
    style CalcLoss fill:#f9f,stroke:#333,stroke-width:2px
    style Grads fill:#f9f,stroke:#333,stroke-width:2px
    style Note1 fill:#eee,stroke:#999,stroke-dasharray:5 5
```
**Key Characteristics:**
*   **Model Use:** Leverages the complete AlphaFold 3 prediction result.
*   **Loss:** Typically based on pLDDT of the binder, PAE at the interface, and predicted contacts.
*   **Gradients:** Flow back through the entire model (Pairformer, Structure Module, Confidence Head).
*   **Cost:** More computationally expensive per step due to full backpropagation.
## Protocol 2: `binder_boltz` (BoltzDesign1-like)
This approach uses a modified forward pass, stopping gradients before the structure module, and optimizes based on intermediate outputs (Distogram, Confidence). It employs a multi-stage optimization schedule.
```mermaid
graph TD
    subgraph BoltzDesignLoop
        StartLoop[Start Boltz Loop] --> StageLoop
        StageLoop{Start Stage} --> StepLoop
        StepLoop{Start Step} --> ApplyStageLogic
        ApplyStageLogic[Apply Stage Logic] --> UpdateFeats
        UpdateFeats[Update Features] --> RunModel
        RunModel[Run Modified Forward Pass] --> CalcLoss
        CalcLoss[Calculate Boltz Loss]
        CalcLoss -- Uses Distogram, Confidence --> Grads
        Grads[Compute Gradients] --> UpdateLogits
        UpdateLogits[Update Sequence Logits] --> StepCheck
        StepCheck{More Steps?} -- Yes --> StepLoop
        StepCheck -- No --> StageCheck
        StageCheck{More Stages?} -- Yes --> StageLoop
        StageCheck -- No --> EndLoop[End Boltz Loop]
        
        Note1[Optimization based on intermediate outputs]
        Note2[Multi-stage temperature annealing]
        Note3[Computationally cheaper gradient calculation]
    end
    
    subgraph ModifiedForwardPass
        M1[Pairformer] --> M2[Structure Module]
        M2 --> M3[Stop Gradient on Atom Coords]
        M1 --> M4[Confidence Head]
        M3 --> M4
        M1 --> M5[Distogram Head]
        M4 --> M6[Partial Output]
        M5 --> M6
    end
    
    RunModel --> ModifiedForwardPass
    
    style RunModel fill:#ccf,stroke:#333,stroke-width:2px
    style M3 fill:#fcc,stroke:#c00,stroke-width:2px
    style CalcLoss fill:#ccf,stroke:#333,stroke-width:2px
    style Grads fill:#ccf,stroke:#333,stroke-width:2px
    style Note1 fill:#eee,stroke:#999,stroke-dasharray:5 5
    style Note2 fill:#eee,stroke:#999,stroke-dasharray:5 5
    style Note3 fill:#eee,stroke:#999,stroke-dasharray:5 5
```
**Key Characteristics:**
*   **Model Use:** Special `boltz_design` mode stops gradients at the structure module.
*   **Loss:** Primarily based on Distogram contacts and Confidence Head outputs (pLDDT, PAE).
*   **Gradients:** Flow back only through Pairformer and Confidence Head, *not* the Structure Module.
*   **Optimization:** Uses a 4-stage process with varying temperatures and sequence representations.
*   **Cost:** Less computationally expensive per step due to partial backpropagation.
## File/Function Call Hierarchy (Simplified)
```mermaid
graph LR
    A[run_alphafold.py] --> B[featurisation.py]
    A --> C[BinderDesigner]
    B --> D[feature_dict]
    C --> E[ModelRunner]
    A --> F[folding_input]
    F --> C
    D --> C
    E --> C
    
    subgraph DesignProcess
        C -- design_binder --> G[Setup Features]
        G --> H{Select Protocol}
        H -- binder_gradient --> I[_design_binder_gradient]
        H -- binder_boltz --> J[_design_binder_boltz]
    end
    
    I --> K[Update Features]
    I --> L[Run Full Model]
    I --> M[Calc Grad Loss]
    K --> I
    L --> I
    M --> I
    
    J --> N[Update Features]
    J --> O[Run Modified Model]
    J --> P[Calc Boltz Loss]
    N --> J
    O --> J
    P --> J
    
    L --> Q[model.py::Model]
    O --> Q
    M --> R[Loss Components]
    P --> R
    
    style C fill:#ddd,stroke:#333,stroke-width:2px
    style L fill:#f9f,stroke:#333,stroke-width:1px
    style O fill:#ccf,stroke:#333,stroke-width:1px
    style M fill:#f9f,stroke:#333,stroke-width:1px
    style P fill:#ccf,stroke:#333,stroke-width:1px
```
This hierarchy shows how `run_alphafold.py` orchestrates the process, utilizing `featurisation` and the `BinderDesigner`. The designer then calls helper utilities (`binder_utils.py`) and specific loss functions (`binder_loss.py`), interacting with the core `ModelRunner` (which wraps the `Model` from `model.py`) in different ways depending on the selected protocol.