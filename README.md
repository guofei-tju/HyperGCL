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

## Running the Experiment
To run our model in yeast dataset, based on the default conditions, which set the ratio of positive and negative reactions as 1:1, imbalanced atom number, and the ratio of replaced atoms for negative reaction as 0.5:
```bash
$ python main.py
```
If you want to run our model based on different creating negative samples strategies, run the following script:
```bash
$ python main.py --train yeast --output ./output/ --create_negative True --balanced True --atom_ratio 0.5 --negative_ratio 2
```

<kbd>train</kbd> specifies the training dataset (For example, ```yeast```, ```uspto_3k```,  ```iMM904```, and so on).

<kbd>output</kbd> specifies the path to store the model.

<kbd>create_negative</kbd> specifies whether to create negative samples based on different conditions. If <kbd>False</kbd>, the model will run on the default train, valid, and test data, and when <kbd>True</kbd>, you need to set other parameters to create negative samples. 

<kbd>balanced</kbd> specifies whether to replace metabolic based on balanced atom number.

<kbd>atom_ratio</kbd> specifies the ratio of replaced atoms for negative reaction.

<kbd>negative_ratio</kbd> specifies the ratio of negative reaction samples.

Use the command <code> python main.py -h </code>to check the meaning of other parameters.

# NICEgame
All input files should be stored in the data directory. This directory contains three sub-folders:

## data/gems
This folder contains the GEMs that will be tested. Each GEM is saved as an XML file.

## data/pools
This folder contains the reaction pool, named `universe.xml`. The pool is also a GEM with the `.xml` extension. If you wish to use your own pool, make sure to rename it to universe.xml and update the `EX_SUFFIX` and `NAMESPACE` parameters in the `input_parameters.txt` file to reflect the suffix of exchange reactions and the namespace of the biochemical reaction database being used.

## data/fermentation
This folder contains two files that are used for GEM simulations. Our algorithm performs simulations on each pair of input and gap-filled GEMs, and if a positive phenotype is observed in the gap-filled GEM but not in the input GEM, the algorithm identifies and outputs the minimum number of reactions required to produce the phenotypic change. The file `substrate_exchange_reactions.csv` contains a list of fermentation compounds that will be searched for missing phenotypes in the input GEMs. This file requires at least two columns, including a column named `compound` that specifies the conventional compound names (e.g., sucrose) and a column named by the `NAMESPACE` specified in the input_parameters.txt file, which specifies the compound IDs (e.g., sucr) in the GEMs. If you wish to use your own list of fermentation compounds, make sure to rename the file to `substrate_exchange_reactions.csv`. Additionally, the file media.csv specifies the culture medium used to simulate the GEMs and requires at least two columns, including a column named by the `NAMESPACE` specified in the `input_parameters.txt` file, which specifies the compound IDs in the GEMs, and another column named flux that specifies the maximum uptake flux for each culture medium component.
  
## Simulation Parameters

1.Score the candidate reactions in the pool for their likelihood of being missing in the input GEMs (function predict() in main.py).

2. Among the top candidate reactions with the highest likelihood, find out the minimum set that leads to new metabolic secretions that are potentially missing in the input GEMs (function validate() in `main.py`). The second program is time-consuming if the number of top candidates added to the input GEMs for simulations is too large (this parameter is controlled by `NUM_GAPFILLED_RXNS_TO_ADD` in the input_parameters.txt). If you only want the scores and rankings of candidate reactions, comment out validate() in main.py.

All simulation parameters are defined in the `input_parameters.txt`:

`CULTURE_MEDIUM (mandatory)`: filepath of culture medium. For the moment, use
