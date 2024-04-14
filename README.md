# Dementia Diagnosis with MRI - University Project

## Introduction

This repository is part of a university project aimed at developing machine learning models to classify the severity of dementia using MRI scans. The project involves models named CNN, CRNN, and ResNet, each designed to process and analyze medical imaging data to assist in diagnostic procedures.

## Contents

- [CNN](#cnn)
- [CRNN](#crnn)
- [ResNet](#resnet)
- [Model Performance](#model-performance)
- [Methodology](#methodology)
- [Installation and Usage](#installation-and-usage)
- [Model Checkpoints](#model-checkpoints)

## CNN

The CNN directory contains a Convolutional Neural Network model tailored for image classification tasks. This model underwent adversarial training to enhance its robustness against adversarial attacks, a key consideration in medical diagnostics.

- Notebook: [CNN.ipynb](./CNN/CNN.ipynb)
- Log: [Training Log](./CNN/log/CNN-train-log.json)
- Adv Log: [Training Log](./CNN/log/CNN-adv-train-log.json)
- Model Checkpoints: [ModelCheckpoints](./CNN/ModelCheckPoints)

## CRNN

The CRNN model is a hybrid architecture that leverages the spatial feature extraction of Convolutional Neural Networks (CNNs) and the temporal sequence modelling of Recurrent Neural Networks (RNNs). This model is particularly adept at handling the MRI scans provided in this dataset, which consist of 61 cross-sectional images per patient. By learning the sequential features within these images, the CRNN model can make more accurate predictions, aiding in the detection and analysis of different stages of Alzheimer's disease.

MRI slices were taken along the z-axis, yielding 256 images, from which slices 100 to 160 were selected to represent the patient's condition effectively. The model's classification aligns with the provided metadata and Clinical Dementia Rating (CDR) values, categorizing patients into four classes: demented, very mild demented, mild demented, and non-demented.

- Notebook: [CRNN.ipynb](./CRNN/CRNN.ipynb)
- Log: [Training Log](./CRNN/log/CRNN-train-log.json)
- Model Checkpoints: [ModelCheckpoints](./CRNN/ModelCheckpoints)

## ResNet

A modified version of the well-known ResNet architecture, customized for this project's requirements, is present in the ResNet directory. The adjustments were made to adapt the architecture for the specificity of MRI-based dementia diagnosis.

- Notebook: [Modified Resnet.ipynb](./ResNet/modified%20resnet.ipynb)
- Log: [Training Log](./ResNet/log)
- Model Checkpoints: [ModelCheckpoints](./ResNet/ModelCheckPoints)

## Model Performance

| Model  | Training Samples | Parameters | Test Accuracy |
| ------ | ---------------- | ---------- | ------------- |
| ResNet | 86,437           | 23,509,956 | 99.94%        |
| CNN    | 86,437           | 14,316,356 | 99.86%        |
| CRNN   | 992              | 257,628    | 95.327%       |

For detailed performance metrics and visualizations, refer to the respective logs in each model's directory.

## Methodology

To evaluate the separability of the classification task for MRI-based dementia severity classification, we initially performed transfer learning with a state-of-the-art ResNet model. The high accuracy achieved by ResNet serves as a benchmark against which we compare our custom-developed models. The ultimate goal is to devise a model architecture that requires substantially fewer parameters while maintaining a comparable level of accuracy.

- **Transfer Learning (ResNet as a Benchmark)**: Utilized the pre-trained ResNet model to establish a performance baseline for the classification task. This step was crucial to understand the separability of the dataset and the potential of deep learning models in this domain.

- **Adversarial Training (CNN only)**: We incorporated adversarial training to enhance the robustness of the CNN model, thereby improving its performance against adversarial attacks, which is essential for maintaining the integrity of medical diagnoses.

- **Data Normalization**: All models underwent data normalization to ensure consistency in data scaling, which is vital for the effective training of neural networks.

- **Data Augmentation (Training samples only)**: Data augmentation techniques were applied exclusively to the training samples. This strategy was adopted to prevent overfitting and to bolster the model's ability to generalize to new, unseen data.

## Model Definition and Usage

Please refer to the individual notebooks for Model Definition and their respective Model Checkpoints folder. The notbook also contains functions to retrain and evaluate each model.

## Model Checkpoints

You can find the saved model states at various training epochs within the `ModelCheckpoints` folder of each model's directory. These checkpoints allow for the continuation of training and model evaluation without retraining from scratch.

## Conclusions

### Accuracy Implications

Accuracy is critical in medical applications, particularly for diagnostics like MRI-based dementia severity classification, due to the potential consequences of misdiagnosis. Misdiagnoses can result in incorrect treatments, delayed care, and substantial emotional distress for patients and their families.

- **ResNet**: With a test accuracy of 99.94%, the ResNet model provides exceptional performance and reliability for clinical use due to its deep architecture and residual learning capabilities.
- **CNN**: The CNN model achieves a test accuracy of 99.86%, slightly below that of ResNet, but well above the threshold required for medical diagnostics, making it a strong candidate for clinical deployment.
- **CRNN**: With a test accuracy of 95.327%, the CRNN model meets the minimum acceptability threshold for medical use. However, this necessitates further validation and potential enhancements before deployment in a clinical setting.

### Parameter Efficiency and Deployment Considerations

The number of parameters in a model has implications for computational resource requirements and inference time. While inference time is not critical in clinical settings, medical diagnostic tools with limited hardware capabilities must run models efficiently.

- **ResNet**: Containing over 23 million parameters, the high computational resource demand of the ResNet model may not be suitable for deployment on all medical devices, especially those with hardware constraints.
- **CNN**: The CNN's 14 million parameters offer a middle ground, providing a model that is complex enough for high accuracy but not as resource-intensive as ResNet. It could be effectively integrated into medical devices with moderate computational capacities.
- **CRNN**: The CRNN's relatively small size of approximately 257,628 parameters makes it an attractive model for hardware-limited environments. Nevertheless, its accuracy trade-off requires careful consideration to ensure it is used appropriately within the medical field.

### Final Thoughts

This project's models, ResNet, CNN, and CRNN, showcase different trade-offs between accuracy and computational complexity. The right choice for deployment would depend on the specific requirements of the medical diagnosis scenario, the availability of computational resources, and the need for portability in medical devices.

## Future Considerations

### Generic Considerations for All Models:

- Continuous validation against new and emerging data (considering changes in technology used and possibly expanding the models to incorporate inputs from other modalities) to ensure the models' reliability and accuracy are maintained.

### Specific Considerations for the CRNN Model:

- The CRNN model's relatively lower accuracy could be improved by addressing the small number of training samples through input sequence length reduction.
- Testing with shorter continuous subsets of images, such as sequences of 5 or 10, could allow for a more extensive training dataset and potentially improve model performance.
- Optimal sequence length determination will require rigorous hyperparameter tuning during the training phase.
- The goal is to find an effective balance between capturing essential temporal features for accurate diagnosis and providing a sufficiently large and varied training dataset.
