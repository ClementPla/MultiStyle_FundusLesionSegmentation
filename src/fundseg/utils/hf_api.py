from huggingface_hub import EvalResult, ModelCard, ModelCardData

from fundseg.data import ALL_CLASSES, ALL_DATASETS


def modelcard_metadata_from_run(run, model_name) -> ModelCardData:
    """Structure the metric output."""
    m_name = "AUC Precision Recall"
    all_metrics = [
        f"{m_name} - {dataset} {class_name}" for dataset in ALL_DATASETS for class_name in ALL_CLASSES
    ]
    scores = run.history(keys=all_metrics).to_dict()
    eval_results = []
    for dataset in ALL_DATASETS:
        for lesion in ALL_CLASSES:

            metric = f"{m_name} - {dataset} {lesion}"
            score = scores[metric]
            eval_results.append(EvalResult(metric_name=f'{metric} - {lesion}', 
                                           dataset_name=dataset, 
                                           metric_type='roc_auc',
                                           dataset_type=dataset,
                                           task_type='image-segmentation',
                                           metric_value=score[0]))
        
    metadata = ModelCardData(language="en", 
                         license="mit", 
                         library="torchSeg",
                         model_name=model_name,
                         datasets=run.tags,
                         eval_results=eval_results)

    return metadata

def get_modelcard(run, arch) -> ModelCard:
    """Get the modelCard content."""
    
    metadata = modelcard_metadata_from_run(run, arch)

    default_modelcard = f"""
---
{ metadata.to_yaml() }
---
# Lesions Segmentation in Fundus

## Introduction 
We focus on the semantic segmentations of:

1. Cotton Wool Spot
2. Exudates
3. Hemmorrhages
4. Microaneurysms

For an easier use of the models, we refer to cleaned-up version of the code provided in the [fundus lesions toolkit](https://github.com/ClementPla/fundus-lesions-toolkit/tree/main/).

## Architecture

The model uses {arch} as architecture. The implementation is taken from [torchSeg](https://github.com/isaaccorley/torchseg)

## Training datasets

The model was trained on the following datasets:
{', '.join(run.tags)}

"""

    card = ModelCard(default_modelcard)

    return card
