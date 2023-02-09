# CLOSEgaps

## Abstract
Incomplete knowledge of metabolic processes hampers the accuracy of GEnome-scale Metabolic models (GEMs), hindering advancements in systems biology and metabolic engineering. To address this challenge, we present NICEgame, a generalized pipeline for automated metabolic network reconstruction, and CLOSEgaps, a key component of NICEgame, leverages a machine learning approach that considers the hypergraph topology of metabolic networks and hypothetical reactions to predict missing reactions in GEMs. Extensive results show that CLOSEgaps accurately reconstructs metabolic networks, filling over $96\%$ of artificially introduced gaps, and enhances the predictability of fermentation products in $24$ wild-type GEMs. The notable improvement in the production of four crucial metabolites (Lactate, Ethanol, Propionate, and Succinate) in two organisms (Bifidobacterium longum subsp. infantis ATCC 15697 and Faecalibacterium prausnitzii A2-165) highlights the benefits of using CLOSEgaps in metabolic network reconstruction to optimize the fermentation pathway. As a broadly applicable solution for any GEM or reaction, CLOSEgaps promises to enhance biotechnological and biomedical applications by improving GEM-based predictions and automating the NICEgame workflow.

![image](./img/model.png)

## Dependencies
The package depends on the Python scientific stack:
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
