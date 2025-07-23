import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.circuit.library import RawFeatureVector
import yfinance as yf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class QuantumRegimeDetector:
    def __init__(self, n_features=8, n_qubits=4, reps=2):
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.reps = reps
        
        # quantum circuit components
        self.feature_map = ZZFeatureMap(n_qubits, reps=1)
        self.ansatz = RealAmplitudes(n_qubits, reps=reps)
        
        # ml components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.vqc = None
        
        # data
        self.features = None
        self.labels = None
        self.regime_names = ['Bull', 'Bear', 'Sideways', 'High Vol']
        
        # results
        self.training_history = []
        self.predictions = None
        
    def load_market_data(self, symbols, start_date, end_date):
        print(f"Loading market data for {len(symbols)} assets...")
        
        data = {}
        for sym in symbols:
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(start=start_date, end=end_date)
                if len(hist) > 200:
                    data[sym] = hist['Close']
                else:
                    print(f"Insufficient data for {sym}")
            except Exception as e:
                print(f"Error loading {sym}: {e}")
        
        if not data:
            raise ValueError("No data loaded")
        
        prices = pd.DataFrame(data).dropna()
        returns = prices.pct_change().dropna()
        
        print(f"Loaded {len(returns)} days of data")
        return prices, returns
    
    def extract_features(self, prices, returns, window=20):
        # extract market regime features
        features = []
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            window_prices = prices.iloc[i-window:i]
            
            # feature 1: average return
            avg_return = window_returns.mean().mean()
            
            # feature 2: volatility
            volatility = window_returns.std().mean()
            
            # feature 3: trend strength (linear regression slope)
            trend_strength = 0
            for col in window_prices.columns:
                x = np.arange(window)
                y = window_prices[col].values
                slope = np.polyfit(x, y, 1)[0] / y[0]  # normalized slope
                trend_strength += slope
            trend_strength /= len(window_prices.columns)
            
            # feature 4: momentum (recent vs old returns)
            recent_returns = window_returns.tail(5).mean().mean()
            old_returns = window_returns.head(5).mean().mean()
            momentum = recent_returns - old_returns
            
            # feature 5: correlation breakdown
            corr_matrix = window_returns.corr()
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            
            # feature 6: drawdown
            cumulative = (1 + window_returns.mean(axis=1)).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = ((cumulative - running_max) / running_max).min()
            
            # feature 7: skewness
            skewness = window_returns.mean(axis=1).skew()
            
            # feature 8: volatility clustering (garch effect)
            vol_series = window_returns.std(axis=1)
            vol_clustering = vol_series.autocorr(lag=1)
            
            features.append([
                avg_return, volatility, trend_strength, momentum,
                avg_correlation, drawdown, skewness, vol_clustering
            ])
        
        return np.array(features)
    
    def define_regimes(self, features):
        # define market regimes based on features
        labels = []
        
        for feat in features:
            avg_ret, vol, trend, momentum, corr, dd, skew, vol_clust = feat
            
            # regime classification rules
            if vol > 0.02:  # high volatility threshold
                regime = 3  # high vol
            elif avg_ret > 0.001 and trend > 0:  # positive return and trend
                regime = 0  # bull
            elif avg_ret < -0.001 and trend < 0:  # negative return and trend
                regime = 1  # bear
            else:
                regime = 2  # sideways
            
            labels.append(regime)
        
        return np.array(labels)
    
    def prepare_data(self, prices, returns):
        # extract features and labels
        features = self.extract_features(prices, returns)
        labels = self.define_regimes(features)
        
        # scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # reduce dimensionality if needed
        if features_scaled.shape[1] > self.n_features:
            # use pca or select top features
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.n_features)
            features_scaled = pca.fit_transform(features_scaled)
        
        self.features = features_scaled
        self.labels = labels
        
        print(f"Prepared {len(features)} samples with {features_scaled.shape[1]} features")
        print(f"Regime distribution: {np.bincount(labels)}")
        
        return features_scaled, labels
    
    def create_quantum_classifier(self):
        # create variational quantum classifier
        
        # combine feature map and ansatz
        qc = QuantumCircuit(self.n_qubits)
        qc.compose(self.feature_map, inplace=True)
        qc.compose(self.ansatz, inplace=True)
        
        # measurement
        qc.add_register(ClassicalRegister(self.n_qubits))
        qc.measure_all()
        
        # create vqc
        sampler = Sampler()
        optimizer = COBYLA(maxiter=100)
        
        self.vqc = VQC(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=optimizer,
            sampler=sampler
        )
        
        return self.vqc
    
    def train(self, test_size=0.3, random_state=42):
        if self.features is None:
            raise ValueError("Need to prepare data first")
        
        # split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels
        )
        
        print("Training quantum classifier...")
        
        # create and train classifier
        if self.vqc is None:
            self.create_quantum_classifier()
        
        # train
        self.vqc.fit(X_train, y_train)
        
        # predict
        train_score = self.vqc.score(X_train, y_train)
        test_score = self.vqc.score(X_test, y_test)
        
        y_pred = self.vqc.predict(X_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        # store results
        self.predictions = {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'train_score': train_score,
            'test_score': test_score
        }
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'predictions': y_pred,
            'actual': y_test
        }
    
    def classical_benchmark(self):
        # compare with classical models
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        
        if self.features is None:
            raise ValueError("Need data")
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=0.3, random_state=42, stratify=self.labels
        )
        
        # random forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_score = rf.score(X_test, y_test)
        
        # svm
        svm = SVC(kernel='rbf', random_state=42)
        svm.fit(X_train, y_train)
        svm_score = svm.score(X_test, y_test)
        
        return {
            'random_forest': rf_score,
            'svm': svm_score,
            'quantum': self.predictions['test_score'] if self.predictions else 0
        }
    
    def predict_regime(self, current_prices, current_returns, window=20):
        # predict current market regime
        if self.vqc is None:
            raise ValueError("Need to train model first")
        
        # extract features from current data
        if len(current_returns) < window:
            raise ValueError(f"Need at least {window} days of data")
        
        features = self.extract_features(
            current_prices.tail(window+1), 
            current_returns.tail(window+1), 
            window
        )
        
        # scale features
        features_scaled = self.scaler.transform(features[-1].reshape(1, -1))
        
        # predict
        regime_pred = self.vqc.predict(features_scaled)[0]
        regime_proba = self.get_prediction_probabilities(features_scaled)
        
        return {
            'regime': self.regime_names[regime_pred],
            'regime_id': regime_pred,
            'probabilities': dict(zip(self.regime_names, regime_proba))
        }
    
    def get_prediction_probabilities(self, features):
        # get quantum probabilities for each class
        # simplified implementation
        try:
            # create circuit with features
            qc = self.feature_map.assign_parameters(features.flatten())
            qc.compose(self.ansatz.assign_parameters(self.vqc._fit_result.x), inplace=True)
            
            # measure
            qc.add_register(ClassicalRegister(self.n_qubits))
            qc.measure_all()
            
            # simulate
            simulator = AerSimulator()
            job = simulator.run(qc, shots=1024)
            counts = job.result().get_counts()
            
            # convert to probabilities
            total_shots = sum(counts.values())
            probs = [0.0] * 4  # 4 regimes
            
            for bitstring, count in counts.items():
                # map bitstring to regime (simplified)
                regime_id = int(bitstring[:2], 2) % 4  # use first 2 bits
                probs[regime_id] += count / total_shots
            
            return probs
            
        except:
            # fallback to uniform distribution
            return [0.25] * 4
    
    def plot_results(self):
        if self.predictions is None:
            print("Need to train model first")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # confusion matrix
        cm = confusion_matrix(self.predictions['y_test'], self.predictions['y_pred'])
        im = axes[0, 0].imshow(cm, interpolation='nearest', cmap='Blues')
        axes[0, 0].set_title('Quantum Classifier Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0, 0].text(j, i, str(cm[i, j]), ha='center', va='center')
        
        axes[0, 0].set_xticks(range(len(self.regime_names)))
        axes[0, 0].set_xticklabels(self.regime_names, rotation=45)
        axes[0, 0].set_yticks(range(len(self.regime_names)))
        axes[0, 0].set_yticklabels(self.regime_names)
        
        # model comparison
        benchmarks = self.classical_benchmark()
        models = list(benchmarks.keys())
        scores = list(benchmarks.values())
        
        axes[0, 1].bar(models, scores)
        axes[0, 1].set_title('Model Comparison')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # feature importance (approximation)
        if hasattr(self, 'feature_importance'):
            axes[1, 0].bar(range(len(self.feature_importance)), self.feature_importance)
            axes[1, 0].set_title('Feature Importance (Quantum)')
            axes[1, 0].set_xlabel('Feature')
            axes[1, 0].set_ylabel('Importance')
            axes[1, 0].grid(True, alpha=0.3)
        
        # regime distribution over time
        if len(self.labels) > 0:
            regime_counts = np.bincount(self.labels)
            axes[1, 1].pie(regime_counts, labels=self.regime_names, autopct='%1.1f%%')
            axes[1, 1].set_title('Historical Regime Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_quantum_advantage(self):
        print("\nQuantum Advantage Analysis:")
        print("=" * 30)
        
        if self.predictions is None:
            print("Need to train model first")
            return
        
        # circuit properties
        print(f"Quantum circuit depth: {self.feature_map.depth() + self.ansatz.depth()}")
        print(f"Number of qubits: {self.n_qubits}")
        print(f"Number of parameters: {self.ansatz.num_parameters}")
        
        # performance comparison
        benchmarks = self.classical_benchmark()
        quantum_acc = benchmarks['quantum']
        best_classical = max(benchmarks['random_forest'], benchmarks['svm'])
        
        print(f"\nQuantum accuracy: {quantum_acc:.3f}")
        print(f"Best classical accuracy: {best_classical:.3f}")
        print(f"Quantum advantage: {quantum_acc - best_classical:.3f}")
        
        # expressibility analysis
        feature_space_size = 2 ** self.n_qubits
        parameter_space = self.ansatz.num_parameters
        
        print(f"\nFeature space dimension: {feature_space_size}")
        print(f"Parameter space dimension: {parameter_space}")
        print(f"Expressibility ratio: {parameter_space / feature_space_size:.3f}")
        
        return {
            'quantum_accuracy': quantum_acc,
            'classical_accuracy': best_classical,
            'advantage': quantum_acc - best_classical,
            'circuit_depth': self.feature_map.depth() + self.ansatz.depth(),
            'n_parameters': self.ansatz.num_parameters
        }
    
    def regime_transition_analysis(self, prices, returns):
        # analyze regime transitions
        if self.vqc is None:
            raise ValueError("Need trained model")
        
        # predict regimes for entire dataset
        features = self.extract_features(prices, returns)
        features_scaled = self.scaler.transform(features)
        
        regime_predictions = self.vqc.predict(features_scaled)
        
        # transition matrix
        transitions = np.zeros((4, 4))
        for i in range(len(regime_predictions) - 1):
            current = regime_predictions[i]
            next_regime = regime_predictions[i + 1]
            transitions[current, next_regime] += 1
        
        # normalize
        row_sums = transitions.sum(axis=1)
        transition_probs = transitions / row_sums[:, np.newaxis]
        
        # regime persistence
        persistence = np.diag(transition_probs)
        
        print("\nRegime Transition Analysis:")
        print("=" * 30)
        print("Transition Probability Matrix:")
        print(pd.DataFrame(transition_probs, 
                          index=self.regime_names, 
                          columns=self.regime_names).round(3))
        
        print(f"\nRegime Persistence:")
        for i, regime in enumerate(self.regime_names):
            print(f"{regime}: {persistence[i]:.3f}")
        
        return {
            'transition_matrix': transition_probs,
            'persistence': persistence,
            'regime_sequence': regime_predictions
        }

def main():
    print("Quantum Market Regime Detection")
    print("===============================\n")
    
    # market data
    symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'VIX']  # diversified etfs
    start_date = "2018-01-01"
    end_date = "2024-01-01"
    
    # initialize detector
    detector = QuantumRegimeDetector(n_features=8, n_qubits=4, reps=2)
    
    # load and prepare data
    prices, returns = detector.load_market_data(symbols, start_date, end_date)
    features, labels = detector.prepare_data(prices, returns)
    
    # train quantum classifier
    print("\n1. Training Quantum Classifier:")
    results = detector.train()
    
    print(f"Training accuracy: {results['train_accuracy']:.3f}")
    print(f"Test accuracy: {results['test_accuracy']:.3f}")
    
    # benchmark against classical
    print("\n2. Comparing with Classical Models:")
    benchmarks = detector.classical_benchmark()
    for model, score in benchmarks.items():
        print(f"{model}: {score:.3f}")
    
    # current regime prediction
    print("\n3. Current Market Regime:")
    try:
        current_regime = detector.predict_regime(prices, returns)
        print(f"Predicted regime: {current_regime['regime']}")
        print("Probabilities:")
        for regime, prob in current_regime['probabilities'].items():
            print(f"  {regime}: {prob:.3f}")
    except Exception as e:
        print(f"Error predicting current regime: {e}")
    
    # quantum advantage analysis
    print("\n4. Quantum Advantage Analysis:")
    advantage = detector.analyze_quantum_advantage()
    
    # regime transition analysis
    print("\n5. Regime Transition Analysis:")
    transitions = detector.regime_transition_analysis(prices, returns)
    
    # plot results
    print("\n6. Plotting results...")
    detector.plot_results()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
