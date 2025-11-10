# Hyperparameter Tuning for Machine Learning Models
This project aims to explore and analyze metaheuristic search-based algorithms for different kinds of machine learning algorithms, in 2 different datasets. 

## Proposal
This is our [idea](./Project%20Proposal/Project%20Proposal%20-%20Fernando%20and%20Kelvin.pdf).

## Datasets
* [MNIST](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)
    * Handwritten Digit Recognition
    * Using scikit-learn's fetch_openml
    * 28x28 Grayscale Images
    * 10 Classes of digits (0-9)
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
    * Object Recognition
    * 32x32 Colored Images
    * In total 60000 images: 50000 training images and 10000 test images
    * 10 Balanced Classes

## Models in Consideration
* **Tree-based Model**: Decision Tree
* **Linear/Polynomial-based**: Linear Regression (optional)
* **Permutation-based** (especially, neural networks): Multi-Layer Perceptron (MLP)
* **Kernel-based**: K-Nearest-Neighbor (KNN)

## Metaheuristic Guided Search

### Baseline
* [**Randomized Search**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
* [**Grid Search**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

### Other Algorithms
* **Evolutionary Algorithm**: Memetic Algorithm (a specialized Genetic Algorithm)
* **Simulated Annealing**
* **Swarm Optimization**: Particle Swarm Optimization

## Prerequisites
- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Installation with uv (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Hyperparameter-Tuning-Search-Project
   ```

2. **Install dependencies** (this automatically creates a virtual environment)
   ```bash
   uv sync
   ```

3. **Run a quick demo**
   ```bash
   uv run python main.py
   ```

4. **Start Jupyter notebook**
   ```bash
   uv run jupyter notebook
   ```

### Alternative Installation with pip

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -e .
   ```

## Execution Guide
### Data Preparation
1. Run `data_download.py` to firstly download the datasets needed. 
   * Note: Data are stored into the `.cache\` folder which is gitignored.
   * Note: Should you rerun the script again, and the folder already exists with contents, please run the script with argument `--force` to enable a smooth overwriting behavior. 
2. Run `data_process.py` to process the images in the datasets. 
3. Run `data_explorer.py` to view details of processed images from different API endpoints. 
   * Note: You may need to use a client such as Postman to launch those API requests. 
   * Note: Refer to [`openapi.yaml`](openapi.yaml) for more detailed descriptions of those endpoints. 