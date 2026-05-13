# dhi-mleo

**dhi-mleo** is a Python library designed to integrate in-situ measurements and Earth Observation data with machine learning workflows. It offers:

* Spatial prediction and inference with Xarray.
* Robust spatial and temporal cross-validation strategies.
* Efficient handling of time series data.
* A flexible workflow compatible with any machine learning model or task.

## Installation 

Quick install from GitHub:

```bash
pip install "dhi-mleo @ git+https://github.com/ScaleAGData/dhi-mleo.git"
# or
uv pip install "dhi-mleo @ git+https://github.com/ScaleAGData/dhi-mleo.git"
```

**Optional Extras**
You can also install feature-specific extras:

* `ml_exta` – extra machine learning libraries
* `viz` – extra visualization libraries
* `geo` – geospatial libraries
* `dev` – developer tools (tests, linters, docs)
* `notebooks` – Jupyter support

---

For sxample
```bash
pip install "dhi-mleo[ml_extra,viz,geo,notebooks] @ git+https://github.com/ScaleAGData/dhi-mleo.git"
# or
uv pip install "dhi-mleo[ml_extra,viz,geo,notebooks] @ git+https://github.com/ScaleAGData/dhi-mleo.git"
```

## Workflow


```mermaid
flowchart TD
    A["In-situ <br/>Sensors Data"] --> B["Preprocess & Aggregate<br/>Features"]
    C["EO <br/>Meterological data"] --> B
    B --> D["Feature Engineering<br/>Temporal & Spectral Indices"]
    D --> E{{"Choose Approach"}}
    
    E -->|Traditional ML| F["S2, LAI, EA, VIs, etc..."]
    E -->|Foundation Model| G["presto_eokit.features_extractor<br/>Extract Pixel-level Embeddings"]
    
    F --> H["Split Train/Test Data"]
    G --> H
    
    H --> I["Train ML Model<br/>on Training Data"]
    I --> J["Evaluate Metrics<br/>Pearson r, R², RMSE, MAE"]
    J --> K{"Performance OK?"}
    K -->|Yes| L{{"Inference Mode"}}
    K -->|No| M["Adjust Features/<br/>Model Params"]
    M --> H
    
    L -->|Traditional ML| N["Spatial Prediction<br/>xr_predict"]
    L -->|Foundation Model| O["presto_eokit.generate_embeddings<br/>Spatial-level Embeddings"]
    
    O --> N
    N --> P["Generate Outputs"]
    P --> Q["Visualizations<br/>& Statistics"]
```