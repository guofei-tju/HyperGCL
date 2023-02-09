# CLOSEgaps

## Abstract
Incomplete knowledge of metabolic processes hampers the accuracy of GEnome-scale Metabolic models (GEMs), hindering advancements in systems biology and metabolic engineering. To address this challenge, we present NICEgame, a generalized pipeline for automated metabolic network reconstruction, and CLOSEgaps, a key component of NICEgame, leverages a machine learning approach that considers the hypergraph topology of metabolic networks and hypothetical reactions to predict missing reactions in GEMs. Extensive results show that CLOSEgaps accurately reconstructs metabolic networks, filling over $96\%$ of artificially introduced gaps, and enhances the predictability of fermentation products in $24$ wild-type GEMs. The notable improvement in the production of four crucial metabolites (Lactate, Ethanol, Propionate, and Succinate) in two organisms (Bifidobacterium longum subsp. infantis ATCC 15697 and Faecalibacterium prausnitzii A2-165) highlights the benefits of using CLOSEgaps in metabolic network reconstruction to optimize the fermentation pathway. As a broadly applicable solution for any GEM or reaction, CLOSEgaps promises to enhance biotechnological and biomedical applications by improving GEM-based predictions and automating the NICEgame workflow.

![image](./img/model.png)

## Dependencies
The package depends on the Python==3.7.13:
```
cobra==0.22.1
joblib==1.2.0
numpy==1.21.5
optlang==1.5.2
pandas==1.3.5
torch==1.12.1
torch_geometric==2.1.0
torch_scatter==2.0.9
torch_sparse==0.6.15 
tqdm==4.62.1
scikit-learn==1.0.2
rdkit==2022.03.5
```

## Datasets
We utilized CLOSEgaps to predict missing reactions in both metabolic networks and chemical reaction datasets. The detail of all datasets is shown as below:
| oprule Dataset | Species                                    | Metabolites (vertices) | Reactions (hyperlinks) |
|----------------|--------------------------------------------|------------------------|------------------------|
| Yeast8.5       | Saccharomyces cerevisiae (Jul. 2021)       | 1136                   | 2514                   |
| iMM904         | Saccharomyces cerevisiae S288C (Oct. 2019) | 533                    | 1026                   |
| iAF1260b       | Escherichia coli str.K-12 substr.MG1655    | 765                    | 1612                   |
| iJO1366        | Escherichia coli str.K-12 substr.MG1655    | 812                    | 1713                   |
| iAF692         | Methanosarcina barkeri str.Fusaro          | 422                    | 562                    |
| USPTO\_3k      | Chemical reaction                          | 6706                   | 3000                   |
| USPTO\_8k      | Chemical reaction                          | 15405                  | 8000                   |

The datasets are stored in ```./data``` and each contains reactions and metabolites' SMILES. 
For example, 
* The folder ```./data/yeast```  contains yaset dataset. 
* The file ```./data/yeast/yeast_rxn_name_list.txt``` contains the reactions.
* The file ```./data/yeast/yeast_meta_count.csv``` contains each metabolic's name, SMILES, and atom number.
