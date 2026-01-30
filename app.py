import os
import io
import base64
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import warnings

# XAI & ML Libraries
import shap
import lime
import lime.lime_tabular
import dice_ml
from tensorflow.keras.models import load_model

# Prevent GUI errors
matplotlib.use('Agg')
# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# ===================== CONFIG =====================
DATA_FILE = 'X_test.csv'

MODEL_FILES = {
    'RandomForest': 'rf_model_best.pkl',
    'SVM': 'svm_model.pkl',
    'MLP_1Layer': 'mlp_1layer.pkl',
    'MLP_2Layer': 'mlp_2layer.pkl'
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models_final')

# ===================== GLOBALS =====================
X_data = None  # ONLY FEATURES
X_background = None  # Background for SHAP/LIME
models = {}
feature_names = []
class_names = ['No', 'Yes']


# ===================== LOAD =====================
def load_resources():
    global X_data, X_background, models, feature_names

    # ---- Load Data ----
    data_path = os.path.join(DATA_DIR, DATA_FILE)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'y' in df.columns:
        df = df.drop(columns=['y'])

    X_data = df.copy()
    feature_names = X_data.columns.tolist()

    # Background: Use median or random sample
    # Using kmeans for background summary is often faster for KernelExplainer
    X_background = X_data.sample(min(50, len(X_data)), random_state=42)

    print(f"✅ Data Loaded. Shape: {X_data.shape}")

    # ---- Load Models ----
    for name, file in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, file)
        if not os.path.exists(path):
            print(f"⚠️ Model not found: {file}")
            continue

        try:
            if file.endswith('.h5'):
                models[name] = load_model(path)
            else:
                with open(path, 'rb') as f:
                    models[name] = pickle.load(f)
            print(f"✅ Loaded Model: {name}")
        except Exception as e:
            print(f"❌ Failed loading {name}: {e}")
            models[name] = None


load_resources()


# ===================== HELPERS =====================
def plot_to_base64(fig):
    fig.patch.set_facecolor('#1e1e2f')
    ax = fig.gca()
    ax.set_facecolor('#1e1e2f')
    ax.tick_params(colors='white', which='both')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    # Handle SHAP text colors (default is often black)
    for text in ax.texts:
        text.set_color('white')

    ax.title.set_color('#bb86fc')

    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    plt.close(fig)  # Important to close properly
    return base64.b64encode(img.getvalue()).decode()


def is_keras_model(model):
    return hasattr(model, 'layers') or str(type(model)).find('keras') != -1


# Wrapper to ensure inputs have feature names (fixes sklearn warnings)
def predict_proba_wrapper(model, model_name):
    if is_keras_model(model):
        def fn(x):
            # x comes from LIME as numpy array
            if x.ndim == 2 and 'CNN' in model_name:
                x = x.reshape((x.shape[0], x.shape[1], 1))
            preds = model.predict(x, verbose=0)
            if preds.shape[1] == 1:
                return np.hstack([1 - preds, preds])
            return preds

        return fn
    else:
        def fn(x):
            # Convert numpy array back to DataFrame with feature names
            df_temp = pd.DataFrame(x, columns=feature_names)
            return model.predict_proba(df_temp)

        return fn


# ===================== ROUTES =====================
@app.route('/')
def index():
    sample_df = X_data.sample(min(15, len(X_data)))
    records = []
    for idx, row in sample_df.iterrows():
        r = row.to_dict()
        r['id'] = idx
        records.append(r)

    model_list = [k for k, v in models.items() if v is not None]
    return render_template('index.html', samples=records, columns=feature_names, model_list=model_list)


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        sample_id = int(request.form['sample_id'])
        model_name = request.form['model_name']
        model = models.get(model_name)

        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # DataFrame for SHAP/DiCE (preserves feature names)
        instance_df = X_data.loc[[sample_id]]
        # Numpy for Keras/LIME
        instance_np = instance_df.values

        results = {}

        # ---------- Prediction ----------
        try:
            if is_keras_model(model):
                inp = instance_np
                if 'CNN' in model_name:
                    inp = inp.reshape((1, inp.shape[1], 1))
                prob = model.predict(inp, verbose=0)[0]
                if len(prob) == 1:
                    prob = [1 - prob[0], prob[0]]
            else:
                prob = model.predict_proba(instance_df)[0]

            pred_class = int(np.argmax(prob))
            results['pred_label'] = class_names[pred_class]
            results['pred_confidence'] = float(prob[pred_class])
            # We don't have true label in this file, assume unknown or ignore
            results['true_label'] = "?"

        except Exception as e:
            return jsonify({'error': f'Prediction Failed: {e}'}), 500

        # ---------- SHAP ----------
        plt.clf()
        try:
            # We focus on explaining the positive class (Index 1: 'Yes')
            CLASS_INDEX = 1

            if model_name == "RandomForest":
                # TreeExplainer is fast
                explainer = shap.TreeExplainer(model)
                # shap_values can be a list [array_class0, array_class1]
                shap_values = explainer.shap_values(instance_df)

                # Logic to extract exactly one array of shape (n_features,)
                if isinstance(shap_values, list):
                    # Multi-class output (e.g. list of 2 arrays)
                    # We take the array corresponding to CLASS_INDEX, then the first sample [0]
                    shap_val = shap_values[CLASS_INDEX][0]
                    base_val = explainer.expected_value[CLASS_INDEX]
                else:
                    # Binary classification where output is just one array (rare in sklearn RF but possible)
                    # Or regression
                    if len(shap_values.shape) == 3:  # (samples, features, classes)
                        shap_val = shap_values[0, :, CLASS_INDEX]
                    elif len(shap_values.shape) == 2:
                        shap_val = shap_values[0]
                    base_val = explainer.expected_value
                    if isinstance(base_val, (list, np.ndarray)):
                        base_val = base_val[CLASS_INDEX]

            else:
                # KernelExplainer for SVM / MLP (Slower)
                # Using a smaller background sample for speed
                background_summary = shap.sample(X_background, 20)

                def predict_fn_shap(X):
                    # X comes as dataframe or numpy
                    if isinstance(X, np.ndarray):
                        X = pd.DataFrame(X, columns=feature_names)
                    # Return probability of class 1
                    return model.predict_proba(X)[:, 1]

                explainer = shap.KernelExplainer(predict_fn_shap, background_summary)
                shap_values = explainer.shap_values(instance_df, nsamples=50)  # Low nsamples for speed

                # KernelExplainer usually returns a list for classifiers
                if isinstance(shap_values, list):
                    shap_val = shap_values[
                        0]  # KernelExplainer with predict_proba[:,1] returns list of length 1 (weird, but happens) or just array
                    if len(np.array(shap_val).shape) > 1 and np.array(shap_val).shape[0] == 1:
                        shap_val = shap_val[0]
                else:
                    shap_val = shap_values[0]

                base_val = explainer.expected_value
                if isinstance(base_val, list): base_val = base_val[0]

            # Construct Explanation Object Manually to ensure correct shape for Waterfall
            # We need shap_val to be (n_features,)
            shap_exp = shap.Explanation(
                values=shap_val,
                base_values=base_val,
                data=instance_df.iloc[0].values,
                feature_names=feature_names
            )

            # Generate Plot
            fig = plt.figure()
            shap.plots.waterfall(shap_exp, show=False, max_display=10)
            results["shap_plot"] = plot_to_base64(plt.gcf())

        except Exception as e:
            print(f"❌ SHAP ERROR: {e}")
            import traceback
            traceback.print_exc()
            results["shap_plot"] = None

        # ---------- LIME ----------
        try:
            plt.clf()
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_background.values,
                feature_names=feature_names,
                class_names=class_names,
                mode='classification',
                verbose=False
            )

            pred_fn = predict_proba_wrapper(model, model_name)

            # Explain the instance
            exp = explainer.explain_instance(
                instance_np[0],
                pred_fn,
                num_features=10,
                labels=[1]  # Explain Class 1 (Yes)
            )

            results['lime_plot'] = plot_to_base64(exp.as_pyplot_figure(label=1))

        except Exception as e:
            print(f"❌ LIME ERROR: {e}")
            results['lime_plot'] = None

        # ---------- DiCE (Counterfactuals) ----------
        try:
            # DiCE needs a dataframe with target column for initialization (even if dummy)
            d_data = X_background.copy()
            d_data['target'] = 0  # Dummy target

            d = dice_ml.Data(
                dataframe=d_data,
                continuous_features=feature_names,
                outcome_name='target'
            )

            backend = 'TF2' if is_keras_model(model) else 'sklearn'
            m = dice_ml.Model(model=model, backend=backend)

            # Method: random is faster than genetic
            exp_dice = dice_ml.Dice(d, m, method='random')

            # Generate CFs
            cfs = exp_dice.generate_counterfactuals(
                instance_df,
                total_CFs=2,
                desired_class="opposite"
            )

            cf_df = cfs.cf_examples_list[0].final_cfs_df

            # Remove dummy target if present
            if 'target' in cf_df.columns:
                cf_df = cf_df.drop(columns=['target'])

            # Highlight changes? (Hard to do in simple HTML, just returning table)
            results['dice_html'] = cf_df.to_html(
                classes='table table-dark table-sm table-striped table-hover',
                index=False,
                border=0
            )

        except Exception as e:
            print(f"❌ DiCE ERROR: {e}")
            results['dice_html'] = f"<div class='alert alert-warning'>Counterfactuals generation failed: {str(e)}</div>"

        return jsonify(results)

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)