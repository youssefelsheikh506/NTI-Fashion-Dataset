# 🧵 Fashion-MNIST Neural Network with Keras

This project demonstrates how to build, train, and evaluate a simple **feedforward neural network** on the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) using **TensorFlow/Keras**.

---

## 📌 What I Did
1. **Loaded the Fashion-MNIST dataset**
   - Dataset consists of **28×28 grayscale images** of clothing items (10 categories).
   - Split into **60,000 training** and **10,000 testing** samples.

2. **Preprocessed the data**
   - Normalized pixel values from `[0, 255]` to `[0, 1]` for faster convergence.

3. **Built the Neural Network**
   - Input layer: Flattened 28×28 images into a 784-dimensional vector.
   - Hidden layers:
     - Dense(128, ReLU)
     - Dense(64, ReLU)
     - Dense(32, ReLU)
   - Output layer:
     - Dense(10, Softmax) → for multi-class classification.

4. **Compiled the model**
   - Optimizer: `Adam`
   - Loss function: `Sparse Categorical Crossentropy` (since labels are integers 0–9)
   - Metric: `Accuracy`

5. **Trained the model**
   - 25 epochs
   - Batch size: 64
   - Included validation on the test set during training.

6. **Evaluated the model**
   - Reported final **test accuracy**.
   - Achieved accuracy around ~0.87–0.90 depending on training run.

---

## 🛠️ Code Structure
- Load and preprocess dataset  
- Define the Sequential model  
- Compile, train, and evaluate the model  
- Print accuracy results  

---

## 🚀 Results
- The model successfully learned to classify Fashion-MNIST images.
- Test accuracy reached **~88–90%**, showing strong performance for a simple dense network.

---

## 📚 Next Steps
- Add **Dropout layers** to reduce overfitting.
- Try **Convolutional Neural Networks (CNNs)** for better performance on image data.
- Experiment with different optimizers and learning rates.

---

## 🔧 Requirements
- Python 3.x  
- TensorFlow / Keras  
- NumPy  
- Matplotlib (optional, for visualization)  

Install dependencies:
```bash
pip install tensorflow numpy matplotlib
