# Dependency Parsing-Based Entity and Relation Extraction (DP-ERE)

## Overview

DP-ERE introduces a novel syntactic filtering approach for entity relation extraction that addresses the precision degradation issues commonly found in long, entity-dense scientific sentences. The core innovation lies in using dependency parse trees to distinguish between syntactically plausible and implausible entity pairs during preprocessing, significantly reducing false positives and computational overhead.

## Key Innovation

The method is built on a true semantic relation between two entities is more likely to occur when the entities are close in the syntactic structure of a sentence, as measured by the shortest path in a dependency parse tree, regardless of their linear distance.*

![DP-ERE Architecture](https://raw.githubusercontent.com/basiralab/DP-ERE/main/dp_ere_main_figure.png)

## Core Features

### Syntactic Distance-Based Filtering
- **Dependency Parse Integration**: Leverages grammatical relationships between words rather than linear token distance
- **Entity Head Token Mapping**: Maps multi-word entities to syntactic root tokens for accurate distance computation
- **Threshold-Based Pruning**: Filters entity pairs using shortest dependency path length criterion
- **Strategic Inference-Only Application**: Applied exclusively during inference to maintain training robustness

### Performance Benefits
- **Precision Enhancement**: Dramatically reduces false positives in relation classification
- **Computational Efficiency**: Quadratic reduction in candidate pairs for relation classifier evaluation
- **Modular Design**: Seamlessly integrates into existing ERE pipeline architectures
- **Domain Agnostic**: Works with any dependency parser and relation classifier combination

## Methodology

### 1. Entity Representation in Syntactic Space
For a sentence `s = {w₁, w₂, ..., wₙ}` with dependency tree `Tₛ = (Vₛ, Aₛ)`:
- Each entity `eᵢ` spanning tokens `[sᵢ, tᵢ]` is mapped to its head token `hᵢ`
- Head token represents the syntactic root of the entity span

### 2. Syntactic Distance Computation
Distance between entities `eᵢ` and `eⱼ` is defined as:
```
d(hᵢ, hⱼ) = shortest_path_length(hᵢ, hⱼ)
```
Computed efficiently using Least Common Ancestor (LCA) algorithm.

### 3. Candidate Filtering Function
```
f(eᵢ, eⱼ) = {1  if d(hᵢ, hⱼ) ≤ δ
            {0  otherwise
```
Where `δ` is the distance threshold hyperparameter.

## Installation

```bash
git clone https://github.com/basiralab/DP-ERE
cd DP-ERE
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Dependencies
- Python ≥ 3.8
- PyTorch ≥ 1.9.0
- SpaCy ≥ 3.4.0 with English model
- Transformers ≥ 4.20.0
- NetworkX for graph operations
- NumPy, Pandas for data processing

## File Structure

Being Updated

### Command Line Usage

```bash
# Train with DP-ERE filtering
python dp_ere_PL_Marker.py \
    --data_dir ./data/SciERC/ \
    --train_file train.json \
    --dev_file dev.json \
    --test_file test.json \
    --distance_threshold 4 \
    --learning_rate 2e-5 \
    --epochs 10 \
    --batch_size 16
```

### Key Parameters
- `--distance_threshold`: Maximum dependency path length (default: 4)
- `--apply_filter_train`: Whether to apply filtering during training (default: False)
- `--apply_filter_test`: Whether to apply filtering during inference (default: True)
- `--dependency_parser`: SpaCy model for parsing (default: "en_core_web_sm")

## Experimental Validation

### Datasets
Validated on multiple benchmark datasets:
- **SciERC**: Scientific entity and relation corpus
- **SciER**: Extended scientific entity recognition dataset  
- **ACE05**: Automatic Content Extraction dataset

### Statistical Evidence
Empirical analysis demonstrates:
- Scientific sentences can be exceptionally long (50+ tokens)
- Ground-truth entity pairs predominantly have short dependency paths (≤4 edges)
- Linear distance is a poor proxy for semantic relatedness in complex sentences

### Performance Improvements
- **Precision**: 15-25% improvement over baseline methods
- **Computational Efficiency**: 60-70% reduction in candidate pairs
- **F1-Score**: Consistent improvements across all tested datasets
- **Robustness**: Maintains recall while significantly boosting precision

## Integration with Existing Methods

DP-ERE is designed as a modular preprocessing component that can enhance any existing ERE pipeline:

```python
# Integration example
class YourREModel:
    def __init__(self):
        self.dp_filter = DPEREFilter(distance_threshold=4)
        self.relation_classifier = YourClassifier()
    
    def predict(self, sentence, entities):
        # Apply DP-ERE filtering
        filtered_pairs = self.dp_filter.filter_entity_pairs(sentence, entities)
        
        # Run relation classification on filtered pairs
        relations = self.relation_classifier.predict(filtered_pairs)
        return relations
```

## Limitations and Future Work

While DP-ERE demonstrates significant improvements in precision and efficiency, the method represents a syntactic enhancement rather than a semantic breakthrough. Key limitations include:

- **Local Context Dependency**: Operates within document-level context without global knowledge understanding
- **Syntactic Focus**: Cannot reason about deeper methodological or conceptual dependencies
- **Threshold Sensitivity**: Performance depends on optimal threshold selection for specific domains

These limitations motivate future research directions toward more sophisticated approaches that incorporate global knowledge structures and semantic reasoning capabilities.

## Citation

If you use DP-ERE in your research, please cite:

```bibtex
@inproceedings{
    joshi2025dependency,
    title={Dependency Parsing-Based Syntactic Enhancement of Relation Extraction in Scientific Texts},
    author={Devvrat Joshi and Islem Rekik},
    booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
    year={2025},
    url={https://openreview.net/forum?id=SjCfVHO2pS}
}
```