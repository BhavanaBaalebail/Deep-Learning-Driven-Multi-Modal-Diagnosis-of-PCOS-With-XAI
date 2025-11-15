#  PCOS Multimodal Diagnosis System using Deep Learning & Explainable AI

This project presents a multimodal PCOS (Polycystic Ovary Syndrome) screening system that uses:

* EfficientNet-B4 for ultrasound image classification
* MLP model for clinical data prediction
* Explainable AI (XAI) using Occlusion, LIME, and SHAP
* Streamlit Web App for easy user interaction
* Combined multimodal decision system where users can upload either images or clinical values

This project helps in early and accurate screening of PCOS using AI-driven clinical support tools.


## Features
 1. Ultrasound Image Classification (EfficientNet-B4)

* Achieved 95% accuracy
* Includes threshold tuning (0.706 optimal threshold)
* Supports classification between PCOS and Non-PCOS
* Integrated Occlusion Sensitivity Maps for explainability

2. Clinical Data Model (MLP)

* Uses patient parameters (BMI, Follicle Count, Cycle Length, Hormone levels, etc.)
* Achieved high accuracy & stable performance
* XAI using:

  * LIME for feature-wise explanations
  * SHAP for global + local interpretability

3. Multimodal Integration

* Combines predictions from:
  * EfficientNet Image Model
  * MLP Clinical Model
* User can input:
  * An ultrasound image, or
  * A set of clinical values
* Automatically selects the appropriate model

4.Streamlit Application

Intuitive UI that supports:

* Image upload & preprocessing
* Clinical input form
* Model predictions
* XAI visualization (Occlusion, LIME, SHAP)

5.End-to-End Deployment Ready

* Models saved as `.h5` files
* Streamlit app integrates both
* Simple to deploy on Render, HuggingFace, or Streamlit Cloud

Project Structure

 PCOS-Diagnosis-System
│
├── pcos_efficientnet.ipynb      # Training EfficientNet-B4 model
├── pcos_MLP.ipynb               # Training MLP clinical model
├── pcos_streamlitapp.ipynb      # Streamlit integration notebook
│
├── efficientnetb4_pcos_best.h5  # Trained image model weights
├── pcos_mlp_model.h5            # Trained clinical model weights
│
├── README.md                    # Project documentation

## Models Used
1. EfficientNet-B4 Image Model

* Pretrained on ImageNet
* Fine-tuned on PCOS ultrasound dataset
* Data Augmentation applied
* Best performing architecture among tried models (ResNet, MobileNet, etc.)

2. MLP for Clinical Parameters

* Multi-layer perceptron
* Normalized clinical input data
* Used:

  * Dense layers
  * ReLU activations
  * Dropout for regularization


##  Explainable AI (XAI)

| Model        | XAI Method                | Purpose                                               |
| ------------ | ------------------------- | ----------------------------------------------------- |
| EfficientNet | Occlusion Sensitivity     | Visualizes important ultrasound regions               |
| MLP          | LIME                      | Shows dominant clinical factors per person            |
| MLP          | SHAP                      | Global and local interpretability of clinical dataset |


##  Streamlit App Workflow

#Home Page

Choose between:
* Upload Ultrasound Image
* Enter Clinical Info

#Image Pipeline
1. Preprocess image
2. EfficientNet inference
3. Threshold-based classification
4. Display Occlusion heatmap

#Clinical Pipeline
1. Fill clinical form
2. MLP prediction
3. LIME & SHAP explanation plots
Here is **only the part you need to add** to your README.
Nothing is bold, and you can paste it directly into your file.


#Streamlit App Setup (Model Upload Instructions)
To run the Streamlit app, you must manually download the trained model files because they are not included directly in the repository.
1. Open the notebooks:
   * pcos_efficientnet.ipynb
   * pcos_MLP.ipynb
2. From the final cells in each notebook, download the following files:
   * efficientnetb4_pcos_best.h5
   * pcos_mlp_model.h5
3. Place both files in the same folder where your Streamlit app (app.py) is located.
Your folder should look like this:
streamlit_app/
│── app.py
│── efficientnetb4_pcos_best.h5
│── pcos_mlp_model.h5

4. Run the Streamlit app using the command:
streamlit run app.py
The app will load both models and allow predictions for image and clinical data.


# requirements.txt
tensorflow==2.12.0
keras==2.12.0
numpy
pandas
matplotlib
opencv-python
scikit-learn
streamlit
pillow
lime
shap
seaborn
protobuf==3.20.*

# Results Summary

| Model           | Accuracy      | Additional Notes             |
| --------------- | ------------- | ---------------------------- |
| EfficientNet-B4 | **95%**       | Best-performing architecture |
| MLP             | High accuracy | Stable and interpretable     |


#Future Improvements
* Add multimodal fusion model (late/early fusion)
* Expand dataset
* Deploy on cloud platforms
* Integrate mobile-friendly UI

#Author
Bhavana Baalebail, H J Vrishank, Aditya Bharath Raja Rao, Ayush J Shetty
AI/ML | Medical Imaging | Biosignal Analytics | Explainable AI

