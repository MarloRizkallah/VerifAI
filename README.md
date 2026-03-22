Final_cleaned_nli_dataset --> Supports + Refutes + NEI pairs after filteration (used the Optimized Data v2, then did the filtration).
V2_dataset(Optimized) --> Supports + Refutes + NEI pairs (NEI constructed using 33% random retrieval + 67% using lexical retrieval bm25 & semantic retrieval SBERT where 33% with mid similarity and 34% with higher similarity))
V1_dataset(unoptimized) --> Supports + Refutes + NEI pairs (NEI constructed using 50% random retrieval + 50% lexical retrieval using bm25).

Supports_final --> used optimized supports + filtration (subset from the final_cleaned_nli_dataset).
Refutes_final  --> used optimized refutes  + filtration (subset from the final_cleaned_nli_dataset).
NEI_final      --> used optimized NEI      + filtration (subset from the final_cleaned_nli_dataset).
