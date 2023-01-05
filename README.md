# predicting non-trivial equivalent mutants of the MART
 
## Data
Thanks to Titcheu Chekam et al., the original dataset is available at   https://mutationtesting.uni.lu/farm/.

In this repository, I provided two preprocessed versions of the dataset. *data/Codeflaws_features_Eq_101.parquet* is the preprocessed dataset on which feature selection steps of the proposed method were applied.

*data/Codeflaws_features_Eq_4.parquet* is the transformed version of the *Codeflaws_features_Eq_101.parquet*, in which the feature embedding step of the proposed method was applied.

## Code
In the *src/utils.py*, I provided the code required for transforming the *data/Codeflaws_features_Eq_101.parquet* to *data/Codeflaws_features_Eq_4.parquet*. Also, the feature embedding dictionaries were written in the code, which might be useful in some cases. Given an instance, one could retrieve the original values of the features before embedding and doing further analysis, such as extracting simple interpretable classification rules, visualizing the data, etc.

In the src/train_and_eval.py, I provided the code required to replicate the study's main findings. In particular, the code is responsible for training and validating the different classifiers on the reduced dataset (i.e., *data/Codeflaws_features_Eq_4.parquet*). However, with minimum changes, such as the file's path and the model's parameters, it also can be used for training on other versions of the dataset.
