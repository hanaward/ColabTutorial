# 1. Why Colab for Machine Learning? <a name="why-colab"></a>
Benefits:

- Free GPU/TPU (Tesla T4, P100, V100, TPU v2/v3)

- Pre-installed ML libraries (TensorFlow, PyTorch, scikit-learn)

- No setup headaches - Everything works out-of-the-box

- Collaborative - Share notebooks like Google Docs

-  Cloud storage - Integrates with Google Drive

Limitations:

- 12-hour session limit (disconnects after inactivity)

- Limited RAM/disk space (free tier)

- No persistent storage (unless using Drive)

# 2.  Setup & Configuration <a name="setup"></a>

  Runtime Setup:

  #1. Go to: Runtime → Change runtime type
  <img width="2879" height="1638" alt="Screenshot 2025-12-15 185728" src="https://github.com/user-attachments/assets/3229961c-79d1-40ca-bdc2-41e478ab87e7" />
  
  #2. Select: GPU or TPU accelerator
  #3. Click Save
  <img width="2879" height="1513" alt="Screenshot 2025-12-15 185753" src="https://github.com/user-attachments/assets/a740415b-f927-4d3f-ab15-1323aef8fe41" />

  You can verify the selected runtime with the following code or the bottom left corner of the page.      
        #Verify GPU
        
        !nvidia-smi
        
        #Check TensorFlow GPU access
        
        import tensorflow as tf
        print("TensorFlow version:", tf.__version__)
        print("GPU Available:", tf.config.list_physical_devices('GPU'))

        
        # Install ML packages
        !pip install -q tensorflow==2.12.0
        !pip install -q torch torchvision torchaudio
        !pip install -q scikit-learn pandas numpy matplotlib seaborn
        !pip install -q xgboost lightgbm catboost
        !pip install -q transformers datasets  # Hugging Face
        
        # Verify installations
        import sklearn
        print("scikit-learn:", sklearn.__version__)


# 3. Data Handling in Colab <a name="data"></a>
  
  Option 1: Upload Local Files
  
      from google.colab import files
      
      # Upload single file
      uploaded = files.upload()
      
      # Upload multiple
      for fn in uploaded.keys():
          print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')
<img width="2877" height="1622" alt="Screenshot 2025-12-15 185912" src="https://github.com/user-attachments/assets/d034c068-3db3-4501-804d-128217bec7fe" />

  
  Option 2: Mount Google Drive (Recommended)
  
      from google.colab import drive
      
      # Mount Drive (will prompt for authorization)
      drive.mount('/content/drive')
      
      # Access files
      import pandas as pd
      df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data.csv')

<img width="2873" height="1515" alt="Screenshot 2025-12-15 185833" src="https://github.com/user-attachments/assets/d42d8a46-74f3-47ac-be17-dc0f51f75256" />
<img width="2857" height="1517" alt="Screenshot 2025-12-15 185948" src="https://github.com/user-attachments/assets/c89547e1-08f4-4aa3-adc3-c138fae7df1e" />
<img width="2879" height="1525" alt="Screenshot 2025-12-15 185849" src="https://github.com/user-attachments/assets/9a517fe1-a1f8-4cd8-bff9-fc6c5ced242c" />

  
  Option 3: Download from URLs
  
      # Download dataset
      !wget https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
      
      # Load into pandas
      import pandas as pd
      df = pd.read_csv('iris.data', header=None)
  
  Option 4: Use Built-in Datasets
  
      # TensorFlow/Keras datasets
      from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist
      (x_train, y_train), (x_test, y_test) = mnist.load_data()
      
      # scikit-learn datasets
      from sklearn import datasets
      iris = datasets.load_iris()
      digits = datasets.load_digits()
  
  Option 5: Kaggle Datasets
  
      # Install Kaggle API
      !pip install -q kaggle
      
      # Upload kaggle.json (get from Kaggle account)
      from google.colab import files
      files.upload()  # Upload kaggle.json
      
      # Set up Kaggle
      !mkdir ~/.kaggle
      !cp kaggle.json ~/.kaggle/
      !chmod 600 ~/.kaggle/kaggle.json
      
      # Download dataset
      !kaggle competitions download -c titanic

# 4.  Basic ML Workflow <a name="basic-ml"></a>
  Complete Example: Classification with scikit-learn

      # Import libraries
      import pandas as pd
      import numpy as np
      from sklearn.model_selection import train_test_split
      from sklearn.preprocessing import StandardScaler
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.metrics import accuracy_score, classification_report
      import matplotlib.pyplot as plt
      import seaborn as sns
      
      # Load data (using Iris dataset)
      from sklearn.datasets import load_iris
      iris = load_iris()
      X = iris.data
      y = iris.target
      
      # Split data
      X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=0.2, random_state=42
      )
      
      # Scale features
      scaler = StandardScaler()
      X_train_scaled = scaler.fit_transform(X_train)
      X_test_scaled = scaler.transform(X_test)
      
      # Train model
      model = RandomForestClassifier(n_estimators=100, random_state=42)
      model.fit(X_train_scaled, y_train)
      
      # Evaluate
      y_pred = model.predict(X_test_scaled)
      accuracy = accuracy_score(y_test, y_pred)
      print(f"Accuracy: {accuracy:.2%}")
      print(classification_report(y_test, y_pred))
      
      # Feature importance visualization
      importances = model.feature_importances_
      plt.figure(figsize=(10, 6))
      plt.bar(range(len(importances)), importances)
      plt.title('Feature Importances')
      plt.show()

  Regression Example:

      # Regression with XGBoost
      from xgboost import XGBRegressor
      from sklearn.metrics import mean_squared_error, r2_score
      
      # Assuming X, y are loaded
      model = XGBRegressor(n_estimators=100, learning_rate=0.1)
      model.fit(X_train, y_train)
      
      y_pred = model.predict(X_test)
      mse = mean_squared_error(y_test, y_pred)
      r2 = r2_score(y_test, y_pred)
      
      print(f"MSE: {mse:.4f}")
      print(f"R² Score: {r2:.4f}")


# 5. Deep Learning Setup <a name="deep-learning"></a>
 TensorFlow/Keras Example:

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Check GPU availability
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    
    # Simple CNN for MNIST
    model = keras.Sequential([
        layers.Reshape((28, 28, 1), input_shape=(28, 28)),
        layers.Conv2D(32, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Train with callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    
    history = model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'\nTest accuracy: {test_acc:.4f}')


  PyTorch Example:

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Simple Neural Network
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = x.view(-1, 784)  # Flatten
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # Create model
    model = Net().to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Convert data to PyTorch tensors
    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Training loop
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    
# 6. GPU/TPU Acceleration <a name="acceleration"></a>
  
  Using GPU:
        
        # TensorFlow automatically uses GPU if available
        # For PyTorch:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        data = data.to(device)
        
        # Monitor GPU usage
        !nvidia-smi --query-gpu=memory.used --format=csv
        
  Using TPU:

       # Initialize TPU
      import tensorflow as tf
      
      try:
          tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
          tf.config.experimental_connect_to_cluster(tpu)
          tf.tpu.experimental.initialize_tpu_system(tpu)
          strategy = tf.distribute.TPUStrategy(tpu)
          print('Running on TPU:', tpu.master())
      except ValueError:
          strategy = tf.distribute.get_strategy()
          print('Running on CPU/GPU')

# 7. Model Training & Saving <a name="training"></a>

   Save/Load Models:
      
        # Save to Drive
        from google.colab import drive
        drive.mount('/content/drive')
        
        # TensorFlow/Keras
        model.save('/content/drive/MyDrive/models/my_model.h5')
        # or
        model.save('/content/drive/MyDrive/models/my_model/')  # SavedModel format
        
        # Load
        loaded_model = tf.keras.models.load_model('/content/drive/MyDrive/models/my_model.h5')
        
        # PyTorch
        torch.save(model.state_dict(), '/content/drive/MyDrive/models/model.pth')
        
        # Load PyTorch
        model.load_state_dict(torch.load('/content/drive/MyDrive/models/model.pth'))
        
        # Use strategy scope for model creation
        with strategy.scope():
            model = create_your_model()
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

   Training Visualisation:

   
        # Plot training history
        def plot_training_history(history):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot accuracy
            ax1.plot(history.history['accuracy'], label='Train Accuracy')
            ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
            ax1.set_title('Model Accuracy')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            
            # Plot loss
            ax2.plot(history.history['loss'], label='Train Loss')
            ax2.plot(history.history['val_loss'], label='Val Loss')
            ax2.set_title('Model Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()
        
        plot_training_history(history)

# 8. Advanced Features <a name="advanced"></a>

  Hugging Face Transformers:

        # Install
        !pip install -q transformers datasets
        
        # Load pre-trained model
        from transformers import pipeline
        
        # Sentiment analysis
        classifier = pipeline('sentiment-analysis')
        result = classifier("I love using Google Colab for ML!")
        print(result)
        
        # Text generation
        generator = pipeline('text-generation', model='gpt2')
        text = generator("Machine learning is", max_length=50, num_return_sequences=1)
        print(text[0]['generated_text'])

  Hyperparameter Tuning:

       # Using scikit-learn GridSearchCV
        from sklearn.model_selection import GridSearchCV
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            n_jobs=-1,  # Use all CPU cores
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")

  AutoML with PyCaret:

         !pip install -q pycaret
          
          from pycaret.classification import *
          
          # Setup
          clf1 = setup(data=df, target='target_column', session_id=123)
          
          # Compare models
          best_model = compare_models()
          
          # Create model
          model = create_model('xgboost')
          
          # Tune model
          tuned_model = tune_model(model)

# 9. Best Practices for ML in Colab <a name="best-practices"></a>

   Session Management:

        # Save important data before disconnection
        import pickle
        
        # Save variables
        with open('/content/drive/MyDrive/checkpoint.pkl', 'wb') as f:
            pickle.dump({'model': model, 'history': history}, f)
        
        # Load later
        with open('/content/drive/MyDrive/checkpoint.pkl', 'rb') as f:
            checkpoint = pickle.load(f)
            model = checkpoint['model']

  Memory Management:


       # Clear memory
        import gc
        import torch
        
        del model  # Delete large objects
        gc.collect()  # Force garbage collection
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

  Progress Tracking:

        # Use tqdm for progress bars
        !pip install -q tqdm
        from tqdm import tqdm
        
        for epoch in tqdm(range(epochs), desc='Training'):
            # Training code here
            pass

  Connectivity Tips:

    1. Automatic reconnection script:

             // Add this in a code cell
            function ClickConnect(){
                console.log("Working");
                document.querySelector("colab-toolbar-button#connect").click()
            }
            setInterval(ClickConnect,60000)
   
    2. Download results immediately:
          
             from google.colab import files
              
              # Save and download
              model.save('final_model.h5')
              files.download('final_model.h5')
   

   
   
    
