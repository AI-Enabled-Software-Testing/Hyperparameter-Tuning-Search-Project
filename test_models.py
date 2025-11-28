"""Simple test script to verify model signatures work correctly."""

import numpy as np
from models.base import get_model_by_name


def create_dummy_data(n_samples=100, n_features=784, n_classes=10):
    """Create dummy data for testing."""
    # Create flattened image-like data (for DT and KNN)
    X_flat = np.random.rand(n_samples, n_features).astype(np.float32)
    # Ensure all classes are represented
    y_flat = np.tile(np.arange(n_classes), n_samples // n_classes + 1)[:n_samples]
    np.random.shuffle(y_flat)
    
    # Create image-like data (for CNN) - list of 2D arrays
    X_images = [np.random.rand(32, 32).astype(np.float32) for _ in range(n_samples)]
    y_images = np.tile(np.arange(n_classes), n_samples // n_classes + 1)[:n_samples]
    np.random.shuffle(y_images)
    
    return X_flat, y_flat, X_images, y_images


def test_decision_tree():
    """Test Decision Tree model."""
    print("Testing Decision Tree...")
    model = get_model_by_name("dt")
    
    X_train, y_train, _, _ = create_dummy_data(50)
    X_test, y_test, _, _ = create_dummy_data(20)
    
    # Test create_model
    model.create_model(max_depth=5, min_samples_split=2)
    
    # Test train
    model.train(X_train, y_train)
    
    # Test evaluate
    metrics = model.evaluate(X_test, y_test)
    print(f"  ✓ Decision Tree metrics: {list(metrics.keys())}")
    assert "accuracy" in metrics
    return True


def test_knn():
    """Test KNN model."""
    print("Testing KNN...")
    model = get_model_by_name("knn")
    
    X_train, y_train, _, _ = create_dummy_data(50)
    X_test, y_test, _, _ = create_dummy_data(20)
    
    # Test create_model
    model.create_model(n_neighbors=3)
    
    # Test train
    model.train(X_train, y_train)
    
    # Test evaluate
    metrics = model.evaluate(X_test, y_test)
    print(f"  ✓ KNN metrics: {list(metrics.keys())}")
    assert "accuracy" in metrics
    assert "roc_auc" in metrics  # Should have ROC AUC (standardized)
    return True


def test_cnn():
    """Test CNN model."""
    print("Testing CNN...")
    model = get_model_by_name("cnn")
    
    _, _, X_train, y_train = create_dummy_data(50)
    _, _, X_val, y_val = create_dummy_data(20)
    _, _, X_test, y_test = create_dummy_data(20)
    
    # Test create_model
    model.create_model(kernel_size=3, stride=1, learning_rate=1e-3, batch_size=16)
    
    # Test train with TrainingConfig
    from models.cnn import TrainingConfig
    config = TrainingConfig(epochs=1, patience=999, batch_size=16)  # Just 1 epoch for quick test
    model.train(X_train, y_train, X_val, y_val, config=config, verbose=False)
    
    # Test evaluate
    metrics = model.evaluate(X_test, y_test)
    print(f"  ✓ CNN metrics: {list(metrics.keys())}")
    assert "accuracy" in metrics
    assert "loss" in metrics
    return True


def main():
    """Run all tests."""
    print("Running model signature tests...\n")
    
    try:
        test_decision_tree()
        test_knn()
        test_cnn()
        print("\n✅ All tests passed!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

