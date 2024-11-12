# NLP_project
Project for the Natural Language Processing (NLP) course at Politecnico di Milano, A.Y 2023-2024.

The project involves the creations of models able to answer medical questions. The [dataset](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) used is composed of medical flashcards
and can be found on HuggingFace. 

### Analysis of the Dataset
In this section, we will visualize various aspects of the dataset:
  - **Word Frequency**: Count the occurrences of each word in both the questions and the answers

  - **Length Analysis**: Analyze the length of questions and answers, providing average, maximum, and minimum lengths

  - **Length Distribution**: Visualize the distribution of lengths for questions and answers, focusing on different sets of the most frequent lengths.
  Additionally, we visualize the joint plot of these distributions to better understand their relationships

  - **Joint Plot Analysis**: Visualize a joint plot of these distributions to better understand the relationship between question and answer lengths

  - **Frequent Words Analysis**: Visualize the frequency of the most common words, both with and without stopwords, and represent this data using a word cloud.


### From Texts to Vectors
In this section, we will transform our texts into vectors using two distinct approachess:

  - **Bag-of-Words (BoW) Representation**: This method involves creating a sparse vector where the length corresponds to the number of unique words in our vocabulary. Each entry in this vector represents the frequency of the corresponding word appearing in the text

  - **TF-IDF Representation**: This is an enhanced version of the BoW model, which assigns greater importance to words that are more relevant within a specific document; notice that, although the dimensions of these vectors remain the same as in the BoW approach, the values within them vary to reflect the weighted importance of each word. We generate this representation by utilizing the formulas shown below. 

$$
TF(t, d) = \frac{\text{number of times } t \text{ appears in document } d}{\text{number of terms in document } d} \quad \hspace{2cm} IDF(t, D) = \log \left( \frac{\text{number of documents }}{\text{number of documents that contain term } t} \right)
$$

$$
\text{TF-IDF}(t, d, D) = TF(t, d) \times IDF(t, D)
$$


Once the documents are vectorized, we develop a retrieval approach based on cosine similarity in order to compare BoW and TF-IDF representations: it will compare the similarity between a user-provided query and each question in our dataset.
Below, we discuss the considerations and effectiveness of using this approach.


### Text Clustering
In this section, we will perform text clustering using two main techniques: K-Means and DBSCAN. For the vectorization of texts, we will employ the TF-IDF representation and the [Universal Sentence Encoder](https://arxiv.org/pdf/1803.11175). TF-IDF is a technique that captures only syntactic information, while USE, a model developed by Google, produces embeddings for entire sentences and captures their semantics. Consequently, comparing these two methods is inherently "unfair," as USE provides a more reliable way to classify the answers into various sub-themes based on their meaning!

### Retrieval System
In this section, we experiment with different techniques that can be used to retrieve the answers given a certain query of the user. First we index the questions using PyTerrier, this permit us to retrieve the answer to the question that is most similar to the user query.

### Fine-tuning of the Retrieval Model
In this section, we fine-tune the previous retrieval model with the aim of improving the previous exact match metric. Indeed, the performances of the pretrained model were so much better than much simpler models like TF-IDF or BM25. While the pretrained model is trained on a general task, with fine tuning we aim at making it learn a better representation of the input text sequences for this task. We have a set of pairs of questions and answers, which form a set of positive pairs for the Sentence Transformer. Since we only have positive pairs, we train the model using the `MultipleNegativesRankingLoss` loss function

### Generative models
#### Minerva
In this section we fine-tune the first italian LLM, called Minerva, developed by by the Sapienza NLP Team in collaboration with Future Artificial Intelligence Research (FAIR) and CINECA; specifically, we used the version with 350 million parameters due to computational limits, though versions with 1 billion and 3 billion parameters also exist.

Although the technical report for this model is not yet available, we know from the corresponding [Hugging Face Repository](https://huggingface.co/sapienzanlp/Minerva-3B-base-v1.0) that the version used consists of 16 layers with 16 attention heads, and it can process a context with a maximum length of 16,384 tokens.  
The fine-tuned version of this model and the corresponding documentation is available [here](https://huggingface.co/FabioS08/MedicalFlashcardsMinerva).

#### LLAMA 3
[Llama3](https://llama.meta.com/llama3/) is a LLM created by Meta in 2024. Thanks to the [Unsloth framework](https://github.com/unslothai/unsloth), which allow very efficient training using less memory, we were able to finetune the smallest version of the model containing 8 billions parameters. All the arguments for the training were already defined from unsloth in order to get the best performance.

### Comparison of the models
