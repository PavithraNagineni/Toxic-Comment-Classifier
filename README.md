# 🔥 Toxic Comment Classifier (TextCNN + FastAPI)

This project implements a production-ready **toxic comment classification system** using a custom **TextCNN model in PyTorch**, a complete preprocessing pipeline, and a **FastAPI** microservice for real-time inference.

It is designed with an industry-style structure, making it ideal for AIML/ML Engineer portfolios.

---

## 🚀 Features
- Custom **TextCNN** deep learning architecture  
- Full **NLP preprocessing pipeline** (tokenization, vocab creation, padding)  
- Real-time inference with **FastAPI**  
- Dockerized deployment  
- Clean modular folder structure (training + inference separated)  
- Evaluation with accuracy, precision, recall, F1  

---

## 🏗 System Architecture

              RAW TEXT
                  |
                  v
    +---------------------------------+
    |  Preprocessing Pipeline         |
    | Tokenization, Vocab, Padding    |
    +---------------------------------+
                  |
                  v
    +---------------------------------+
    |     TextCNN Classification      |
    |  (Embedding → Conv → Pool → FC) |
    +---------------------------------+
                  |
                  v
    +---------------------------------+
    |      Saved Model (.pt file)     |
    +---------------------------------+
                  |
                  v
    +---------------------------------+
    |   FastAPI Inference Service     |
    |   /predict → returns toxicity   |
    +---------------------------------+
                  |
                  v
              DOCKER DEPLOY


## 📁 Folder Structure
toxic-comment-classifier/

│── data/

│ └── toxic_comments.csv

│── models/

│ └── textcnn.pt

│── src/

│ ├── config.py

│ ├── dataset.py

│ ├── model.py

│ ├── train.py

│ └── evaluate.py

│── service/

│ ├── app.py

│ └── schemas.py

│── requirements.txt

│── Dockerfile

│── README.md


<img width="1342" height="575" alt="image" src="https://github.com/user-attachments/assets/528bf30f-fa79-4508-9c03-2df4e9f8da6f" />


<img width="884" height="646" alt="image" src="https://github.com/user-attachments/assets/b4e33d12-9f28-41fe-8cd9-fee3779ac452" />


<img width="1205" height="665" alt="image" src="https://github.com/user-attachments/assets/9effba8d-5482-40cb-9870-fa1073e69dc8" />
