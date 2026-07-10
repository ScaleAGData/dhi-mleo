# dhi-mleo

**dhi-mleo** is a Python library designed to integrate in-situ measurements and Earth Observation data with machine learning workflows. It offers:

* Spatial prediction and inference with Xarray.
* Robust spatial and temporal cross-validation strategies.
* Efficient handling of time series data.
* A flexible workflow compatible with any machine learning model or task.

## Motivation
In-situ and satellite data often complement each other: the former provides highly accurate measurements but only at local scale and with limited temporal extent, while the latter provides spatial and temporal estimates over much larger scales but potentially with lower accuracy. By fusing the two sets of data with the help of advanced machine-learning methods, including foundation models, those synergies can be exploited to obtain robust and accurate local and regional maps. In the ScaleAgData context, the workflows presented in this repository focus on fusion of in-situ and satellite-based soil moisture estimates to be used by Water, Yield and other Research and Innovation Labs. However, those methods should be equally applicable to other biophysical parameters measurable both on the ground and from satellite sensors.   

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

## Examples

### Presto Foundation Model Embeddings
![Presto RGB Embeddings](./imgs/rgb_embeddings.png)

### Spatial Prediction Comparison
*Comparison of soil moisture predictions: Foundation Model approach (Presto 4-embeddings + Minumal number of Expert features) Vs Traditional ML approach (Sentinel-2 + Expert features).*
![Prediction Comparison](./imgs/preds_comparison.png)
