# Rice Variety Classification Using Deep Learning

A transfer learning project that classifies five varieties of rice grain images using MobileNetV2 and InceptionV3. Built as part of the CN7023 Artificial Intelligence and Machine Vision coursework at the University of East London.

---

## What This Project Does

The notebook trains two convolutional neural networks on a balanced subset of the Rice Image Dataset. Both models use ImageNet pre-trained weights and a custom classification head, without fine-tuning the base layers. The goal is to correctly identify which of five rice varieties appears in a given image.

The five classes are Arborio, Basmati, Ipsala, Jasmine, and Karacadag. InceptionV3 reached 98.20% test accuracy and MobileNetV2 reached 95.03% on the 600-images-per-class subset used here.

---

## Dataset

**Rice Image Dataset** by Murat Koklu et al. (2021)

- Full dataset: 75,000 images across five classes, 15,000 per class
- This project: 600 images per class sampled randomly, giving 3,000 images total
- Image size: 224 x 224 pixels, RGB
- Source: https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset
- License: CC0 Public Domain

To set up the dataset on Google Colab, upload your `kaggle.json` credentials and run the download cell in the notebook. The sampling script creates a folder called `Rice_Balanced_600` with 600 images per class copied randomly from the full dataset.

---

## Project Structure

```
rice-image-detection/
|
|-- Rice_image_detection.ipynb     Main notebook with all code
|-- README.md
```

The notebook is self-contained. Everything runs sequentially from top to bottom in a Colab or Jupyter environment.

---

## How to Run

**On Google Colab (recommended)**

1. Open `Rice_image_detection.ipynb` in Colab
2. Upload your `kaggle.json` file when the second cell prompts you
3. Run all cells in order
4. The dataset downloads, samples to 3,000 images, and trains both models

**Locally**

Install the required packages:

```
pip install tensorflow scikit-learn matplotlib seaborn pandas numpy
```

You will need to adjust the dataset path from `/content/Rice_Balanced_600` to your local path in the data loading cells.

---

## Model Architectures

Both models follow the same transfer learning pattern: freeze the pre-trained base, attach a custom head, and train only the head.

**MobileNetV2**

```
MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = False

Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])
```

**InceptionV3**

```
InceptionV3(input_shape=(224,224,3), include_top=False, weights='imagenet')
base_model.trainable = False

Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])
```

Both compiled with `optimizer='adam'`, `loss='categorical_crossentropy'`, trained for 10 epochs with batch size 32.

---

## Data Augmentation

Augmentation is applied only to the training generator, not validation or test.

```python
ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
```

The validation generator uses only `rescale=1./255` and `validation_split=0.2`. The test generator uses only `rescale=1./255` with `shuffle=False`.

---

## Results

These numbers come directly from the notebook output cell after calling `evaluate_metrics()`:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| MobileNetV2 | 95.03% | 95.53% | 95.03% | 95.03% |
| InceptionV3 | 98.20% | 98.23% | 98.20% | 98.20% |

InceptionV3 outperforms MobileNetV2 by roughly 3.2 percentage points. The difference is likely due to InceptionV3 having a more complex architecture with inception modules that capture multi-scale features, and its larger Dense head (256 units vs 128) giving it more capacity to distinguish between the five classes.

For comparison, Koklu et al. (2021) report up to 100% accuracy using the full 75,000-image dataset with a different CNN architecture. The gap between that and these results is expected given the much smaller training set used here.

---

## Evaluation

The notebook produces the following evaluation outputs for each model:

- Training and validation accuracy curves over 10 epochs
- Training and validation loss curves over 10 epochs
- Confusion matrix using seaborn heatmap
- Per-class ROC curves with AUC scores using sklearn's `roc_curve` and `auc`
- Per-class precision, recall, and F1-score bar charts using `classification_report`
- Side-by-side model comparison table printed as a DataFrame

---

## Limitations

The experiment uses 600 images per class instead of the full 15,000. This means the models have seen significantly less variation than they would in a production setting. Base layers are fully frozen, so the pre-trained features are not adapted to grain images at all. Adding fine-tuning on the top inception or depthwise layers would almost certainly push accuracy higher.

The test set in the notebook is created from the same directory as training using `ImageDataGenerator`, so there is no completely isolated held-out test set. A cleaner evaluation would separate a fixed test split before any data loading.

---

## Possible Next Steps

- Train on the full 75,000-image dataset
- Unfreeze the top layers of InceptionV3 and fine-tune with a low learning rate
- Add EarlyStopping and ReduceLROnPlateau callbacks
- Build a Flask or Streamlit inference app around the saved model
- Use Grad-CAM to visualise which parts of each grain image activate the network

---

## References

Koklu, M., Cinar, I. and Taspinar, Y.S. (2021) Classification of rice varieties with deep learning methods. Computers and Electronics in Agriculture, 187, p.106285.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.C. (2018) MobileNetV2: Inverted residuals and linear bottlenecks. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp.4510-4520.

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. and Wojna, Z. (2016) Rethinking the inception architecture for computer vision. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp.2818-2826.

---

## Author

Vinitha Abhinashree M
