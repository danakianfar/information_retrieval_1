# Information Retrieval

[![License](http://img.shields.io/:license-mit-blue.svg)](LICENSE)

## Description

Code for the [Information Retrieval](http://coursecatalogue.uva.nl/xmlpages/page/2016-2017-en/search-course/course/25718) course of the MSc in Artificial Intelligence at the University of Amsterdam.

### Lab 1 - Evaluation Measures, Interleaving and Click Models

<p align="justify">
One of the key questions in IR is whether offline evaluation and online evaluation outcomes agree with each other. We study the degree of agreement between offline evaluation measures and interleaving outcomes, by the means of simulations using several click models.
</p>

- [Assignment and Solutions](Lab1/11391014-11390689-hw1.ipynb)

### Lab 2 - A Study of Lexical and Semantic Language Models Applied to Information Retrieval

<p align="justify">
In a typical IR task we are interested in finding a (usually ranked) list of results that satisfy the information need of a user expressed by means of a query. Many difficulties arise as the satisfaction and information need of the user cannot be directly observed, and thus the relevance of a particular document given the query is unobserved. Moreover, the query is merely a linguistic representation of the actual information need of the user and the gap between them can not be measured either. We explore three families of models that are used to measure and rank the relevance of a set of documents given a query: lexical models, semantic models, and machine learning-based re-ranking algorithms that build on top of the former models.
</p>

- [Assignment and Solutions](Lab2/11391014-11390689-hw2-report.pdf)

### Lab 3 - Learning to Rank
<p align="justify">
We implement and compare the performance of several approaches to the Learning to Rank (LTR) problem. We received an implementation of a pointwise method based on squared-loss minimization and implemented RankNet (pairwise) and LambdaNet (listwise) algorithms.
</p>

- [Assignment and Solutions](Lab3/11391014-11390689-hw2-report.pdf)


## Testing
Refer to each lab and run the iPython notebook as follows.
```bash
jupyter notebook $notebook_name$.ipynb
```
## Dependencies

- Matplotlib
- NumPy
- pandas
- SciPy
- Pyndri
- Gensim
- Lasagne
- Theano

## Contributors

- [Dana Kianfar](https://github.com/danakianfar)
- [Jose Gallego](https://github.com/jgalle29)

## Copyright

Copyright © 2017 Dana Kianfar and Jose Gallego.

<p align="justify">
This project is distributed under the <a href="LICENSE">MIT license</a>. This was developed as part of the Information Retrieval course taught by Evangelos Kanoulas at the University of Amsterdam. Please review the <a href="http://student.uva.nl/en/content/az/plagiarism-and-fraud/plagiarism-and-fraud.html">UvA regulations governing Fraud and Plagiarism</a> in case you are a student at the UvA.
</p>
