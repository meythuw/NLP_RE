# NLP-based Information Extraction System

This repository presents a **team-based academic project** focusing on **Information Extraction (IE)** using Natural Language Processing techniques.  
The project aims to design and implement an **end-to-end NLP pipeline** capable of extracting structured information—specifically named entities and semantic relations—from unstructured textual data.

---

## Project Overview

Information Extraction is a fundamental task in Natural Language Processing that transforms raw text into structured knowledge.  
In this project, we developed a multi-stage IE system that integrates **data collection, manual annotation, Named Entity Recognition (NER), Relation Extraction (RE), and model evaluation**.

The system is designed to analyze real-world textual data and compare the effectiveness of multiple machine learning approaches for relation extraction.

---

## Project Objectives

The main objectives of this project are:

- To construct a **domain-specific annotated dataset** through web crawling and manual labeling
- To develop a **Named Entity Recognition (NER)** component for identifying entities from unstructured text
- To formulate the **Relation Extraction (RE)** task as a supervised classification problem
- To design and compare multiple machine learning models for relation extraction
- To analyze the impact of **feature engineering and vectorization methods** on model performance
- To integrate all components into a unified information extraction workflow

---

## System Architecture and Workflow

The overall workflow of the system consists of the following stages:

### 1. Data Collection and Annotation
- Textual data were collected from online sources using web crawling techniques
- The collected data were manually annotated to label:
  - Named entities
  - Semantic relations between entity pairs
- The annotated dataset serves as ground truth for both NER and RE tasks

### 2. Named Entity Recognition (NER)
- NER models were trained to automatically identify relevant entities from raw text
- The output entities were used to generate candidate entity pairs for relation extraction
- This step helps reduce noise and constrains the RE task to meaningful entity pairs

### 3. Relation Extraction (RE)
Relation Extraction was modeled as a **supervised multi-class classification problem**, where the system predicts the semantic relation between pairs of recognized entities.

Several machine learning models were implemented and compared, including:
- Support Vector Machine (SVM)
- Random Forest
- Logistic Regression
- Multi-layer Perceptron (MLP)

Each model was trained using feature representations derived from vectorized contextual information surrounding entity pairs.

### 4. Feature Engineering and Vectorization
- Textual and contextual features were transformed into numerical representations suitable for machine learning models
- Different vectorization strategies were explored to capture relational and contextual patterns
- Feature engineering played a critical role in improving relation classification performance

### 5. Evaluation and Analysis
- Model performance was evaluated using standard classification metrics
- A comparative analysis was conducted to assess:
  - Effectiveness of different machine learning models
  - Sensitivity of models to feature representations
  - Generalization performance on unseen data

---

## Team Information

**Team size:** 4 members  

| Member Name | Responsibilities |
|------------|------------------|
| Nguyễn Phan Nhật Lan | Data crawling & annotation, NER model training, RE (SVM), system development |
| Hồ Ngọc Như Quỳnh | Data crawling & annotation, input dataset construction, RE (Random Forest) |
| **Nguyễn Thị Minh Thư** | Data crawling & annotation, feature vectorization, RE (Logistic Regression), report writing |
| Lê Thủy Tiên | Data crawling, evaluation methodology, RE (MLP) |

---

## My Contribution (Nguyễn Thị Minh Thư)

My primary contributions to this project include:

- Participating in **data crawling and manual annotation**, ensuring data quality and labeling consistency
- Designing and implementing **feature engineering and vectorization methods** for relation extraction
- Training and evaluating **Logistic Regression models** for the RE task
- Contributing to **technical report writing**, particularly in methodology description and experimental analysis

All development was carried out collaboratively through shared notebooks and continuous team discussions.

---

## Tools and Technologies

- Programming Language: Python  
- Development Environment: Jupyter Notebook  
- NLP Tasks: Named Entity Recognition, Relation Extraction  
- Machine Learning: Scikit-learn  
- Data Preparation: Manual annotation, feature engineering  

---

## Academic Context

- Course: Natural Language Processing / Information Extraction  
- Institution: University of Economics Ho Chi Minh City (UEH)  
- Project Type: Team-based academic course project

---

## Notes

This repository reflects **collaborative academic work**, where all team members jointly contributed to the development of the system.  
The implementation highlights the components and responsibilities I directly contributed to within the team project.
