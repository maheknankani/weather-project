from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
import random

app = Flask(__name__)
models = {}
scaler = None
results = {}
pakistan_cities = {
    'karachi': {
        'name': 'Karachi',
        'description': 'Coastal city with hot and humid climate',
        'monsoon_months': [7, 8, 9],  
        'dry_months': [11, 12, 1, 2],  
        'transition_months': [3, 4, 5, 6, 10] 
    },
    'lahore': {
        'name': 'Lahore',
        'description': 'Continental climate with hot summers and mild winters',
        'monsoon_months': [7, 8, 9], 
        'dry_months': [11, 12, 1, 2],  
        'transition_months': [3, 4, 5, 6, 10] 
    },
    'islamabad': {
        'name': 'Islamabad',
        'description': 'Sub-tropical highland climate',
        'monsoon_months': [7, 8, 9],
        'dry_months': [11, 12, 1, 2],  
        'transition_months': [3, 4, 5, 6, 10]
    },
    'peshawar': {
        'name': 'Peshawar',
        'description': 'Semi-arid climate with very hot summers',
        'monsoon_months': [7, 8], 
        'dry_months': [11, 12, 1, 2, 3], 
        'transition_months': [4, 5, 6, 9, 10] 
    },
    'quetta': {
        'name': 'Quetta',
        'description': 'Highland climate with cold winters',
        'monsoon_months': [7, 8],  
        'dry_months': [11, 12, 1, 2, 3], 
        'transition_months': [4, 5, 6, 9, 10]
    }
}

def get_seasonal_bias(city, month):
    city_info = pakistan_cities[city]
    
    if month in city_info['monsoon_months']:
        return 0.3  
    elif month in city_info['dry_months']:
        return -0.3 
    else:
        return 0.0 

def add_realistic_noise(probability, noise_level=0.1):
    noise = random.uniform(-noise_level, noise_level)
    return max(0.0, min(1.0, probability + noise))

def load_city_data(city):
    try:
        df = pd.read_csv(f'data/{city.lower()}_weather.csv')
        return df
    except Exception as e:
        print(f"Error loading data for {city}: {str(e)}")
        return None

def analyze_weather_data(df, city):
    analysis = {}
    analysis['statistics'] = df.describe()
    analysis['correlation'] = df.corr()
    analysis['rain_distribution'] = df['rain_tomorrow'].value_counts(normalize=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(analysis['correlation'], annot=True, cmap='coolwarm', center=0)
    plt.title(f'Feature Correlation Heatmap - {city.title()}')
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    analysis['correlation_heatmap'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return analysis

def evaluate_model(model, X_test, y_test, model_name):
    metrics = {}
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    y_pred_proba_with_noise = np.array([
        [add_realistic_noise(p[0]), add_realistic_noise(p[1])] 
        for p in y_pred_proba
    ])
    
    y_pred_proba_with_noise = y_pred_proba_with_noise / y_pred_proba_with_noise.sum(axis=1)[:, np.newaxis]
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['classification_report'] = classification_report(
        y_test, 
        y_pred,
        zero_division=1
    )
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['No Rain', 'Rain'],
        yticklabels=['No Rain', 'Rain']
    )
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    metrics['confusion_matrix_plot'] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return metrics
def get_similar_historical_data(model, X_scaled, X_original, k=5):
    distances, indices = model.kneighbors(X_scaled)
    similar_data = []
    
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        similar_data.append({
            'distance': distance,
            'temperature': X_original.iloc[idx]['temperature'],
            'humidity': X_original.iloc[idx]['humidity'],
            'pressure': X_original.iloc[idx]['pressure'],
            'wind_speed': X_original.iloc[idx]['wind_speed'],
            'cloud_cover': X_original.iloc[idx]['cloud_cover'],
            'rain_tomorrow': X_original.iloc[idx]['rain_tomorrow'],
            'similarity': (1 - distance) * 100 
        })
    
    return similar_data

def get_decision_tree_path(model, feature_names, X):
    tree = model.tree_
    feature = tree.feature
    threshold = tree.threshold
    node_indicator = model.decision_path(X)
    leaf_id = model.apply(X)
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    decision_path = []
    for node_id in node_index:
        if leaf_id[0] == node_id:
            continue
        if X[0, feature[node_id]] <= threshold[node_id]:
            threshold_sign = "<="
        else:
            threshold_sign = ">"
            
        decision = {
            'feature': feature_names[feature[node_id]],
            'threshold': round(threshold[node_id], 2),
            'sign': threshold_sign,
            'value': round(float(X[0, feature[node_id]]), 2)
        }
        decision_path.append(decision)
    
    return decision_path

def get_mlp_activations(model, X):
    activations = []
    current_layer = X
    for i, (weights, biases) in enumerate(zip(model.coefs_, model.intercepts_)):
        current_layer = np.dot(current_layer, weights) + biases
        if i < len(model.coefs_) - 1: 
            current_layer = np.maximum(current_layer, 0)

        avg_activation = np.mean(np.abs(current_layer))
        max_activation = np.max(np.abs(current_layer))
        
        layer_info = {
            'layer': i + 1,
            'neurons': len(biases),
            'avg_activation': round(float(avg_activation), 3),
            'max_activation': round(float(max_activation), 3),
            'active_neurons': int(np.sum(current_layer > 0))
        }
        activations.append(layer_info)
    
    return activations

def initialize_models():
    global models, scaler, results
    models = {city: {} for city in pakistan_cities.keys()}
    results = {city: {} for city in pakistan_cities.keys()}
    scalers = {city: StandardScaler() for city in pakistan_cities.keys()}
    feature_columns = ['temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover', 'month']
    np.seterr(divide='ignore', invalid='ignore')
    
    initialization_errors = {}
    
    for city in pakistan_cities.keys():
        print(f"\nProcessing {city.title()}...")
        try:
            df = load_city_data(city)
            if df is None:
                initialization_errors[city] = "Failed to load data"
                print(f"Error: Failed to load data for {city.title()}")
                continue
            
            print(f"Data loaded successfully for {city.title()}")
        
            missing_columns = [col for col in feature_columns + ['rain_tomorrow'] if col not in df.columns]
            if missing_columns:
                error_msg = f"Missing required columns: {', '.join(missing_columns)}"
                initialization_errors[city] = error_msg
                print(f"Error: {error_msg}")
                continue
            print(f"Analyzing data for {city.title()}...")
            analysis_results = analyze_weather_data(df, city)
            results[city]['analysis'] = analysis_results
            X = df[feature_columns].copy()  
            y = df['rain_tomorrow'].copy()
            if X.isnull().any().any() or y.isnull().any():
                print(f"Warning: Found missing values in {city.title()} data. Handling missing values...")
                X = X.fillna(X.mean())
                y = y.fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_scaled = scalers[city].fit_transform(X_train)
            X_test_scaled = scalers[city].transform(X_test)
            results[city]['selected_features'] = feature_columns
            
            city_models = {
                'knn': KNeighborsClassifier(n_neighbors=5),
                'decision_tree': DecisionTreeClassifier(random_state=42),
                'mlp': MLPClassifier(
                    hidden_layer_sizes=(100, 50), 
                    max_iter=1000,  
                    random_state=42,
                    early_stopping=True  
                )
            }
            for name, model in city_models.items():
                print(f"Training {name} model for {city.title()}...")
                try:
                    unique_classes, class_counts = np.unique(y_train, return_counts=True)
                    print(f"Class distribution for {city.title()}: {dict(zip(unique_classes, class_counts))}")
                    
                    if len(unique_classes) < 2:
                        print(f"Warning: {city.title()} has only one class in training data. Skipping {name} model.")
                        continue
                        
                    if min(class_counts) < 2:
                        print(f"Warning: {city.title()} has too few samples of one class. Skipping {name} model.")
                        continue
                    model.fit(X_train_scaled, y_train)
                    evaluation_metrics = evaluate_model(model, X_test_scaled, y_test, name)
                    accuracy = evaluation_metrics['accuracy'] * 100
                    
                    print(f"Successfully trained {name} model for {city.title()} with {accuracy:.1f}% accuracy")
                    
                    models[city][name] = model
                    results[city][name] = {
                        'accuracy': float(accuracy),
                        'model_info': {
                            'type': name.replace('_', ' ').title(),
                            'features': feature_columns
                        },
                        'evaluation': evaluation_metrics
                    }
                except Exception as model_error:
                    print(f"Error training {name} model for {city.title()}: {str(model_error)}")
                    print(f"Error details: {type(model_error).__name__}")
                    continue
            
            print(f"Successfully processed {city.title()}")
            
        except Exception as e:
            error_msg = f"Error processing city {city}: {str(e)}"
            print(error_msg)
            initialization_errors[city] = error_msg
            continue
    global scaler
    scaler = scalers
    
    print("\nModel Initialization Status:")
    for city in pakistan_cities.keys():
        if city in initialization_errors:
            print(f"{city.title()}: Failed - {initialization_errors[city]}")
        else:
            print(f"{city.title()}: Success")
            
    return len(initialization_errors) == 0 

@app.route('/')
def dashboard():
    selected_city = request.args.get('city', 'karachi').lower()
    
    if selected_city not in pakistan_cities:
        return render_template('dashboard.html',
                            cities=pakistan_cities,
                            selected_city=None,
                            results={},
                            charts={},
                            weather_stats={})
    csv_path = f"data/{selected_city}_weather.csv"
    try:
        df = pd.read_csv(csv_path)
        avg_temp = df['temperature'].mean()
        avg_humidity = df['humidity'].mean()
        avg_wind = df['wind_speed'].mean()
        weather_stats = {
            "temperature": round(avg_temp, 1),
            "humidity": round(avg_humidity, 1),
            "wind_speed": round(avg_wind, 1)
        }
    except:
        weather_stats = {"temperature": None, "humidity": None, "wind_speed": None}
    
    city_results = results.get(selected_city, {})
    if not city_results:
        return render_template('dashboard.html',
                            cities=pakistan_cities,
                            selected_city=selected_city,
                            results={},
                            charts={},
                            weather_stats=weather_stats)
    
    charts = {}
    model_names = ['knn', 'decision_tree', 'mlp']
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    model_accuracies = []
    model_display_names = []
    for name in model_names:
        if name in city_results and isinstance(city_results[name], dict) and 'accuracy' in city_results[name]:
            model_accuracies.append(city_results[name]['accuracy'])
            model_display_names.append(city_results[name]['model_info']['type'])
    
    if model_accuracies:
        ax1.bar(model_display_names, model_accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax1.set_title(f'Model Accuracy Comparison - {selected_city.title()}')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        img = io.BytesIO()
        fig1.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        charts['accuracy_comparison'] = base64.b64encode(img.getvalue()).decode()
        plt.close(fig1)
    if 'analysis' in city_results and 'correlation_heatmap' in city_results['analysis']:
        charts['correlation_heatmap'] = city_results['analysis']['correlation_heatmap']
    for name in model_names:
        if name in city_results and 'evaluation' in city_results[name] and 'confusion_matrix_plot' in city_results[name]['evaluation']:
            charts[f'{name}_confusion_matrix'] = city_results[name]['evaluation']['confusion_matrix_plot']
    
    return render_template('dashboard.html',
                        cities=pakistan_cities,
                        selected_city=selected_city,
                        results=city_results,
                        charts=charts,
                        weather_stats=weather_stats)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        selected_city = request.form.get('city', 'karachi').lower()
    else:
        selected_city = request.args.get('city', 'karachi').lower()
    current_month = datetime.now().month
    seasonal_bias = get_seasonal_bias(selected_city, current_month)
    if selected_city not in pakistan_cities:
        return render_template('predict.html',
                             error=f"City '{selected_city}' not found in our database.",
                             cities=pakistan_cities,
                             selected_city='karachi',
                             input_data={},
                             predictions={},
                             charts={},
                             ranges={})
    
    if selected_city not in models or not models[selected_city]:
        return render_template('predict.html',
                             error=f"Models are not initialized for {selected_city.title()}. Please try restarting the application.",
                             cities=pakistan_cities,
                             selected_city=selected_city,
                             input_data={},
                             predictions={},
                             charts={},
                             ranges={})
    df = load_city_data(selected_city)
    if df is None:
        return render_template('predict.html',
                             error=f"Weather data not available for {selected_city.title()}. Please try another city.",
                             cities=pakistan_cities,
                             selected_city=selected_city,
                             input_data={},
                             predictions={},
                             charts={},
                             ranges={})
    unique_rain_values = df['rain_tomorrow'].unique()
    is_single_class = len(unique_rain_values) == 1
    single_class_value = unique_rain_values[0] if is_single_class else None
    
   
    ranges = {
        'temperature': {'min': df['temperature'].min(), 'max': df['temperature'].max()},
        'humidity': {'min': df['humidity'].min(), 'max': df['humidity'].max()},
        'pressure': {'min': df['pressure'].min(), 'max': df['pressure'].max()},
        'wind_speed': {'min': df['wind_speed'].min(), 'max': df['wind_speed'].max()},
        'cloud_cover': {'min': 0, 'max': 100}
    }
    if request.method == 'GET':
        return render_template('predict.html',
                             cities=pakistan_cities,
                             selected_city=selected_city,
                             ranges=ranges,
                             is_single_class=is_single_class,
                             single_class_value=single_class_value,
                             input_data={},
                             predictions={},
                             charts={})
    
   
    try:
        input_data = {}
        for field in ['temperature', 'humidity', 'pressure', 'wind_speed', 'cloud_cover']:
            try:
                value = float(request.form[field])
                if value < ranges[field]['min'] or value > ranges[field]['max']:
                    raise ValueError(f"{field.replace('_', ' ').title()} must be between {ranges[field]['min']} and {ranges[field]['max']}")
                input_data[field] = value
            except (KeyError, ValueError) as e:
                return render_template('predict.html',
                                     error=str(e),
                                     cities=pakistan_cities,
                                     selected_city=selected_city,
                                     ranges=ranges,
                                     input_data={},
                                     predictions={},
                                     charts={})
        input_data['month'] = current_month
        selected_features = results[selected_city]['selected_features']
     
        input_df = pd.DataFrame([input_data])[selected_features]
        
        if selected_city not in scaler:
            raise ValueError(f"Scaler not found for {selected_city.title()}. Please try restarting the application.")
        
        input_scaled = scaler[selected_city].transform(input_df)
        
      
        predictions = {}
        similar_patterns = None
        
        for name, model in models[selected_city].items():
            if model is None:
                continue
                
            try:
                base_proba = model.predict_proba(input_scaled)[0]
                rain_prob = base_proba[1] + seasonal_bias
                rain_prob = add_realistic_noise(rain_prob)
            
                rain_prob = max(0.0, min(1.0, rain_prob))
                prediction = 'Rain' if rain_prob > 0.5 else 'No Rain'
                
                model_viz = {}
                if name == 'knn':
                    city_data = load_city_data(selected_city)
                    if city_data is not None:
                        X_full = city_data[selected_features]
                        X_full_scaled = scaler[selected_city].transform(X_full)
                        model_viz['similar_patterns'] = get_similar_historical_data(model, input_scaled, city_data)
            
                elif name == 'decision_tree':
                    model_viz['decision_path'] = get_decision_tree_path(model, selected_features, input_scaled)
                
                elif name == 'mlp':
                    model_viz['layer_activations'] = get_mlp_activations(model, input_scaled)
                
                predictions[name] = {
                    'prediction': prediction,
                    'probability': rain_prob * 100,
                    'confidence_level': 'High' if abs(rain_prob - 0.5) > 0.3 else 'Medium' if abs(rain_prob - 0.5) > 0.15 else 'Low',
                    'visualizations': model_viz
                }
                
            except Exception as model_error:
                print(f"Error with {name} model for {selected_city}: {str(model_error)}")
                continue
        
        if not predictions:
            raise ValueError(f"No models were able to make predictions for {selected_city.title()}. Please try restarting the application.")
        
        charts = {}
        
        if len(predictions) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            model_names = list(predictions.keys())
            probabilities = [pred['probability'] for pred in predictions.values()]
            
            bars = ax.bar(model_names, probabilities, color=['#4361ee', '#2ecc71', '#e74c3c'])
            ax.set_title(f'Rain Probability by Model - {selected_city.title()}')
            ax.set_xlabel('Models')
            ax.set_ylabel('Probability (%)')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            img = io.BytesIO()
            fig.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            charts['prediction_comparison'] = base64.b64encode(img.getvalue()).decode()
            plt.close(fig)
        return render_template('predict.html',
                             cities=pakistan_cities,
                             selected_city=selected_city,
                             ranges=ranges,
                             input_data=input_data,
                             predictions=predictions,
                             charts=charts,
                             is_single_class=is_single_class,
                             single_class_value=single_class_value)
                             
    except Exception as e:
        print(f"Error processing prediction for {selected_city}: {str(e)}")
        return render_template('predict.html',
                             error=f"An error occurred while processing your request: {str(e)}",
                             cities=pakistan_cities,
                             selected_city=selected_city,
                             ranges=ranges,
                             input_data={},
                             predictions={},
                             charts={})

if __name__ == '__main__':
    initialize_models()
    app.run(debug=True) 