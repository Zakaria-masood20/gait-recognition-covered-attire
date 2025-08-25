# Gait Recognition for Covered Body Attire  

**Final Year Project (University of Sargodha, Session 2020–2024)**  
By Muhammad Zakaria Masood & Zill E Haseeb  

## Project Overview  
Traditional facial recognition systems fail when individuals wear attire that covers their faces.  
This project explores **gait recognition** as an alternative biometric, using **machine learning models** including Decision Tree, Random Forest, SVM, KNN, and LSTM networks trained on custom-built datasets.  

Our aim: Build a **robust recognition system** that identifies individuals based on walking patterns, even when their body or face is covered.  

## Key Contributions  
- Created a **multi-angle gait dataset** (0°, 45° views)  
- Extracted **body landmarks using Mediapipe** for feature generation  
- Implemented & compared multiple ML classifiers (Decision Tree, Random Forest, SVM, KNN)  
- Developed LSTM networks for temporal gait pattern analysis  
- Achieved **98% accuracy with Random Forest**  
- Provided comprehensive evaluation with precision, recall, F1-score, confusion matrices  

## Repository Structure  
```
├── documents/          # Project documentation
│   ├── CP1/           # Capstone Project 1 materials
│   │   ├── UML/       # UML diagrams (Activity, State, Use Case, etc.)
│   │   └── ...        # Proposals, specifications, reports
│   └── CP2/           # Capstone Project 2 materials
│       └── ...        # Research papers, presentations
│
├── code/              # Source code
│   ├── python/        # Python scripts
│   │   ├── extract_frames.py           # Video frame extraction
│   │   ├── frames_processing_00.py     # Frame processing pipeline
│   │   ├── sorting_csv.py              # Data organization
│   │   ├── train_test_data_spliting.py # Dataset splitting
│   │   ├── decision_tree-w-plot.py     # Decision Tree implementation
│   │   ├── random_forest-w-plot.py     # Random Forest implementation
│   │   └── svm-w-plot.py               # SVM implementation
│   └── notebooks/     # Jupyter notebooks
│       ├── Capston_Gait.ipynb
│       ├── LSTM_00.ipynb               # LSTM for 0° angle
│       ├── LSTM_45.ipynb               # LSTM for 45° angle
│       ├── Frequency_Analysis.ipynb
│       └── Raw_Raw.ipynb
│
├── datasets/          # Data files
│   ├── videos/        # Original video recordings
│   ├── frames/        # Extracted frames
│   ├── processed/     # Processed data files
│   └── train-test/    # Training and testing splits
│
├── models/            # Trained ML models
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   ├── knn_model.pkl
│   ├── lstm_00.h5                      # LSTM model for 0° angle
│   └── lstm_45.h5                      # LSTM model for 45° angle
│
└── results/           # Experimental results and visualizations
```

## Tech Stack  
- **Languages:** Python 3.x  
- **ML Libraries:** scikit-learn, TensorFlow/Keras  
- **Computer Vision:** OpenCV, Mediapipe  
- **Data Processing:** NumPy, Pandas  
- **Visualization:** Matplotlib  
- **Development:** Google Colab, Jupyter Notebooks  

## Results  
| Model           | Accuracy | Precision | Recall | F1 Score |
|-----------------|----------|-----------|--------|----------|
| Decision Tree   | 86%      | 87%       | 86%    | 86%      |
| Random Forest   | 98%      | 98%       | 98%    | 98%      |
| SVM            | 94%      | 94%       | 94%    | 94%      |
| KNN            | 92%      | 92%       | 92%    | 92%      |

> Random Forest proved to be significantly more robust and reliable for gait recognition tasks.  

## Documentation Available  
- Functional Specification Document  
- Software Requirements Specifications (AIAS)  
- UML Diagrams (Activity, State, Deployment, ER, Package, Sequence, Use Case, Class)  
- Research papers and presentations  
- Participant Information and Consent Forms  

## Future Work  
- Extend deep learning architectures (CNN-LSTM hybrid models)  
- Optimize for real-time recognition  
- Explore clothing-independent models for higher generalization  
- Implement multi-view fusion techniques  
- Deploy as a real-world security system  

## Authors  
- Muhammad Zakaria Masood  
- Zill E Haseeb  

## Institution  
University of Sargodha  
Department of Computer Science  
Session 2020–2024  

## License  
This project is part of academic research. Please contact the authors for usage permissions.

---