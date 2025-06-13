import os
import pandas as pd
import numpy as np
import cv2
from sklearn . model_selection import train_test_split
from sklearn . preprocessing import LabelEncoder , OneHotEncoder
from sklearn . metrics import confusion_matrix ,classification_report
import matplotlib . pyplot as plt
import seaborn as sns
def relu(x):
    """
    ReLU activation : max(0, x)
    """
    assert isinstance(x, np.ndarray), "Input to ReLU must be a numpy array"
    result = np.maximum(0, x)  
    assert np.all(result >= 0), "ReLU output must be non-negative"
    return result

def relu_derivative(x):
    """
    Derivative of ReLU : 1 if x > 0, else 0
    """
    assert isinstance(x, np.ndarray), "Input to ReLU derivative must be a numpy array"
    result = (x > 0).astype(float)  
    assert np.all((result == 0) | (result == 1)), "ReLU derivative must be 0 or 1"
    return result
def softmax(x):
    """
    Softmax activation : exp(x) / sum(exp(x))
    """
    assert isinstance(x, np.ndarray), "Input to softmax must be a numpy array"
    # Soustraire le max par ligne pour stabilité numérique
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  
    result = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    assert np.all((result >= 0) & (result <= 1)), "Softmax output must be in [0, 1]"
    assert np.allclose(np.sum(result, axis=1), 1), "Softmax output must sum to 1 per sample"
    return result
# Classe MultiClassNeuralNetwork
class MultiClassNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Initialize the neural network with given layer sizes and learning rate.

        layer_sizes : List of integers [input_size, hidden1_size, ..., output_size]
        """
        assert isinstance(layer_sizes, list) and len(layer_sizes) >= 2, "layer_sizes must be a list with at least 2 elements"
        assert all(isinstance(size, int) and size > 0 for size in layer_sizes), "All layer sizes must be positive integers"
        assert isinstance(learning_rate, (int, float)) and learning_rate > 0, "Learning rate must be a positive number"

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        # Initialisation des poids et biais
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            b = np.zeros((1, layer_sizes[i + 1]))
            assert w.shape == (layer_sizes[i], layer_sizes[i + 1]), f"Weight matrix {i + 1} has incorrect shape"
            assert b.shape == (1, layer_sizes[i + 1]), f"Bias vector {i + 1} has incorrect shape"
            self.weights.append(w)
            self.biases.append(b)
def forward(self, X):
    """
    Forward propagation : Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}, A^{[l]} = g(Z^{[l]})
    """
    assert isinstance(X, np.ndarray), "Input X must be a numpy array"
    assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"

    self.activations = [X]
    self.z_values = []

    # Propagation pour les couches cachées avec ReLU
    for i in range(len(self.weights) - 1)
        assert z.shape == (X.shape[0], self.layer_sizes[i + 1]), f"Z^{i + 1} has incorrect shape"
        self.z_values.append(z)
        self.activations.append(relu(z))

    # Couche de sortie avec softmax
    z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]  
    assert z.shape == (X.shape[0], self.layer_sizes[-1]), "Output Z has incorrect shape"
    self.z_values.append(z)
    output = softmax(z)  
    assert output.shape == (X.shape[0], self.layer_sizes[-1]), "Output A has incorrect shape"
    self.activations.append(output)

    return self.activations[-1]
def compute_loss(self, y_true, y_pred):
    """
    Categorical Cross-Entropy : J = -1/m * sum(y_true * log(y_pred))
    """
    assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), "Inputs to loss must be numpy arrays"
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))  # moyenne sur les exemples, somme sur classes
    assert not np.isnan(loss), "Loss computation resulted in NaN"
    return loss

def compute_accuracy(self, y_true, y_pred):
    """
    Compute accuracy : proportion of correct predictions
    """
    assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), "Inputs to accuracy must be numpy arrays"
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

    predictions = np.argmax(y_pred, axis=1)      # prédictions classes
    true_labels = np.argmax(y_true, axis=1)      # vraies classes
    accuracy = np.mean(predictions == true_labels)  # proportion de bonnes prédictions
    assert 0 <= accuracy <= 1, "Accuracy must be between 0 and 1"
    return accuracy
def backward(self, X, y, outputs):
    """
    Backpropagation : compute dW^{[l]}, db^{[l]} for each layer
    """
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray) and isinstance(outputs, np.ndarray), "Inputs to backward must be numpy arrays"
    assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
    assert y.shape == outputs.shape, "y and outputs must have the same shape"

    m = X.shape[0]
    self.d_weights = [None] * len(self.weights)
    self.d_biases = [None] * len(self.biases)

    dZ = outputs - y  # Gradient for softmax + cross-entropy
    assert dZ.shape == outputs.shape, "dZ for output layer has incorrect shape"

    self.d_weights[-1] = (self.activations[-2].T @ dZ) / m
    self.d_biases[-1] = np.sum(dZ, axis=0, keepdims=True) / m

    for i in range(len(self.weights) - 2, -1, -1):
        dA = dZ @ self.weights[i + 1].T
        dZ = dA * relu_derivative(self.z_values[i])  # element-wise multiply by ReLU derivative
        assert dZ.shape == (X.shape[0], self.layer_sizes[i + 1]), f"dZ^{[i + 1]} has incorrect shape"

        self.d_weights[i] = (self.activations[i].T @ dZ) / m
        self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m

    # Regularisation L2 (lambda = 0.01 par exemple)
    lambda_reg = 0.01
    for i in range(len(self.weights)):
        self.d_weights[i] += (lambda_reg * self.weights[i]) / m

    for i in range(len(self.weights)):
        self.weights[i] -= self.learning_rate * self.d_weights[i]
        self.biases[i] -= self.learning_rate * self.d_biases[i]
def train(self, X, y, X_val, y_val, epochs, batch_size):
    """
    Train the neural network using mini-batch SGD, with validation
    """
    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "X and y must be numpy arrays"
    assert isinstance(X_val, np.ndarray) and isinstance(y_val, np.ndarray), "X_val and y_val must be numpy arrays"
    assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
    assert y.shape[1] == self.layer_sizes[-1], f"Output dimension ({y.shape[1]}) must match output layer size ({self.layer_sizes[-1]})"
    assert X_val.shape[1] == self.layer_sizes[0], f"Validation input dimension ({X_val.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
    assert y_val.shape[1] == self.layer_sizes[-1], f"Validation output dimension ({y_val.shape[1]}) must match output layer size ({self.layer_sizes[-1]})"
    assert isinstance(epochs, int) and epochs > 0, "Epochs must be a positive integer"
    assert isinstance(batch_size, int) and batch_size > 0, "Batch size must be a positive integer"

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_loss = 0
        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            outputs = self.forward(X_batch)
            epoch_loss += self.compute_loss(y_batch, outputs)
            self.backward(X_batch, y_batch, outputs)

        train_loss = epoch_loss / (X.shape[0] // batch_size)
        train_pred = self.forward(X)
        train_accuracy = self.compute_accuracy(y, train_pred)
        val_pred = self.forward(X_val)
        val_loss = self.compute_loss(y_val, val_pred)
        val_accuracy = self.compute_accuracy(y_val, val_pred)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies
def predict(self, X):
    """
    Predict class labels
    """
    assert isinstance(X, np.ndarray), "Input X must be a numpy array"
    assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"

    outputs = self.forward(X)
    predictions = np.argmax(outputs, axis=1)
    assert predictions.shape == (X.shape[0],), "Predictions have incorrect shape"
    return predictions
# Définir le chemin vers le dossier décompressé
data_dir = os.path.join(os.getcwd(), 'amhcd-data-64/tifinagh-images/')
print("Chemin des données :", data_dir)

# Vérification du répertoire courant
current_working_directory = os.getcwd()
print("Répertoire courant :", current_working_directory)

# Chargement du fichier CSV contenant les étiquettes (si disponible)
try:
    labels_df = pd.read_csv('/Users/Apple/Downloads/amhcd-data-64/labels-map.csv', header=None, names=['image_path', 'label'])
    assert 'image_path' in labels_df.columns and 'label' in labels_df.columns, \
        "Le fichier CSV doit contenir les colonnes 'image_path' et 'label'."
except FileNotFoundError:
    print("labels-map.csv introuvable. Construction manuelle du DataFrame...")

    # Construction manuelle du DataFrame à partir de l’arborescence des dossiers
    image_paths = []
    labels = []

    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, img_name))
                labels.append(label_dir)

    labels_df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels
    })
# Vérifier le DataFrame
assert not labels_df.empty, "Aucune donnée chargée. Vérifiez les fichiers du dataset."
print(f"{len(labels_df)} échantillons chargés avec {labels_df['label'].nunique()} classes uniques.")

# Encoder les étiquettes
label_encoder = LabelEncoder()
labels_df['label_encoded'] = label_encoder.fit_transform(labels_df['label'])
num_classes = len(label_encoder.classes_)

# Fonction pour charger et prétraiter une image
def load_and_preprocess_image(image_path, target_size=(32, 32)):
    """
    Charger et prétraiter une image : niveaux de gris, redimensionnement, normalisation.
    """
    assert os.path.exists(image_path), f"Image introuvable : {image_path}"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Échec du chargement de l'image : {image_path}"
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0  
    return img.flatten()  

# Charger et prétraiter toutes les images
data_dir = '/Users/Apple/Downloads/amhcd-data-64/tifinagh-images'
labels_df['image_path'] = labels_df['image_path'].str.replace(r'^\.\/images-data-64\/tifinagh-images\/', '', regex=True)
X = np.array([load_and_preprocess_image(os.path.join(data_dir, path)) for path in labels_df['image_path']])
y = labels_df['label_encoded'].values

# Vérifier les dimensions
assert X.shape[0] == y.shape[0], "Nombre d'images ≠ nombre de labels"
assert X.shape[1] == 32 * 32, f"Taille d'image aplatit attendue : {32*32}, obtenue : {X.shape[1]}"

# Diviser en ensembles entraînement, validation et test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

# S'assurer que ce sont bien des tableaux NumPy
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

# Vérification de la répartition
assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == X.shape[0], "La somme des tailles train/val/test ne correspond pas au total."

print(f"Train : {X_train.shape[0]} échantillons, Validation : {X_val.shape[0]}, Test : {X_test.shape[0]}")
# Encoder les étiquettes en one-hot pour la classification multiclasse
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_train_one_hot = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
y_val_one_hot = one_hot_encoder.transform(y_val.reshape(-1, 1))
y_test_one_hot = one_hot_encoder.transform(y_test.reshape(-1, 1))

# Vérifier que les tableaux one-hot sont bien des NumPy arrays
assert isinstance(y_train_one_hot, np.ndarray), "y_train_one_hot doit être un tableau numpy"
assert isinstance(y_val_one_hot, np.ndarray), "y_val_one_hot doit être un tableau numpy"
assert isinstance(y_test_one_hot, np.ndarray), "y_test_one_hot doit être un tableau numpy"

# Créer et entraîner le modèle
layer_sizes = [X_train.shape[1], 64, 32, num_classes]  
nn = MultiClassNeuralNetwork(layer_sizes, learning_rate=0.01)

train_losses, val_losses, train_accuracies, val_accuracies = nn.train(
    X_train, y_train_one_hot,
    X_val, y_val_one_hot,
    epochs=100, batch_size=32
)

# Prédictions et évaluation
y_pred = nn.predict(X_test)

print("\nRapport de classification (jeu de test) :")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matrice de confusion (jeu de test)")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.savefig("confusion_matrix.png")
plt.close()

# Courbes de perte et de précision
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Courbe de perte
ax1.plot(train_losses, label='Perte entraînement')
ax1.plot(val_losses, label='Perte validation')
ax1.set_title('Courbe de perte')
ax1.set_xlabel('Époque')
ax1.set_ylabel('Perte')
ax1.legend()

# Courbe de précision
ax2.plot(train_accuracies, label='Précision entraînement')
ax2.plot(val_accuracies, label='Précision validation')
ax2.set_title('Courbe de précision')
ax2.set_xlabel('Époque')
ax2.set_ylabel('Précision')
ax2.legend()

plt.tight_layout()
fig.savefig("loss_accuracy_plot.png")
plt.close()

