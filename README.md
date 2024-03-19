# Extract-Keyword-Txt

| Algorithm                                          | Pros                                                                                                                                                                           | Cons                                                                                                                                                                                            |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TF-IDF (Term Frequency-Inverse Document Frequency) | - Statistically sound and identifies document-specific keywords. - Well-established and implemented in popular libraries.                                                      | - Doesn't inherently capture multi-word keywords. - May not be ideal for short text files.                                                                                                      |
| RAKE (Rapid Automatic Keyword Extraction)          | - Designed for keyword extraction from single documents. - Scores phrases, favoring multi-word concepts. - Implemented in libraries like Gensim.                               | - May assign high scores to long, less relevant phrases. - Relies on stop word removal, potentially removing valuable terms.                                                                    |
| TextRank                                           | - Graph-based approach that models word relationships. - Prioritizes keywords based on their connections within the text. - Implemented in libraries like NetworkX.            | - Primarily focuses on single words (additional steps needed for phrases). - Computational cost can increase with larger documents.                                                             |
| SpaCy                                              | - Powerful NLP library with pre-trained models for keyword extraction. - Identifies named entities, noun phrases, and context-based keywords. - Offers domain-specific models. | - More complex setup and configuration. - Larger library size and may require additional processing power.                                                                                      |
| NLTK (Natural Language Toolkit)                    | - Highly customizable for tailoring keyword extraction. - Provides functionalities for tokenization, stop word removal, part-of-speech tagging, and frequency analysis.        | - Requires more programming effort to build a pipeline. - May require domain-specific knowledge for stop word removal and part-of-speech tagging.                                               |
| YAKE                                               | - Language-independent. - Handles multi-word phrases and acronyms well. - Adaptable to domain-specific texts.                                                                  | - Requires careful selection of parameters. - May produce redundant keywords in some cases. - Less established compared to other algorithms.                                                    |
| BERT Fine-Tuning                                   | - State-of-the-art performance. - Captures complex relationships in the text. - Adaptable to various text domains and languages.                                               | - Computationally intensive, especially for large documents. - Requires substantial computational resources and time for fine-tuning. - Dependency on pre-trained models and data availability. |