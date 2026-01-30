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
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
import urllib.request

# ===================== SETUP & LOGGING =====================
load_dotenv()

# Setup Logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = RotatingFileHandler('project_log.log', maxBytes=1024 * 1024, backupCount=5, encoding='utf-8')
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO)
logger = logging.getLogger('AHCI_Logger')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# AUTO-DETECT PROXY
if not os.getenv('HTTP_PROXY'):
    system_proxies = urllib.request.getproxies()
    if 'http' in system_proxies:
        os.environ['http_proxy'] = system_proxies['http']
        os.environ['https_proxy'] = system_proxies['http']
        logger.info(f"🔓 Auto-detected VPN Proxy: {system_proxies['http']}")
    elif 'https' in system_proxies:
        os.environ['http_proxy'] = system_proxies['https']
        os.environ['https_proxy'] = system_proxies['https']
        logger.info(f"🔓 Auto-detected VPN Proxy: {system_proxies['https']}")
else:
    os.environ['http_proxy'] = os.getenv('HTTP_PROXY')
    os.environ['https_proxy'] = os.getenv('HTTPS_PROXY')
    logger.info(f"🔒 Manual Proxy set from .env")

# --- GEMINI SETUP ---
import google.generativeai as genai

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_AVAILABLE = False

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
        GEMINI_AVAILABLE = True
        logger.info("✅ Gemini AI Connected Successfully.")
    except Exception as e:
        logger.error(f"❌ Gemini Connection Failed: {e}")
else:
    logger.warning("⚠️ Gemini API Key is missing.")

# XAI Libraries
import shap
import lime
import lime.lime_tabular
import dice_ml
from tensorflow.keras.models import load_model

matplotlib.use('Agg')
warnings.filterwarnings("ignore")

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

X_data = None
X_background = None
models = {}
feature_names = []
# کلاس‌ها بر اساس دیتاست بانکی:
class_names = ['No (Reject Term Deposit)', 'Yes (Subscribe Term Deposit)']


# ===================== LOAD =====================
def load_resources():
    global X_data, X_background, models, feature_names

    logger.info("📥 Loading resources...")
    data_path = os.path.join(DATA_DIR, DATA_FILE)
    if not os.path.exists(data_path):
        logger.critical("Data file missing!")
        raise FileNotFoundError("Data missing")

    df = pd.read_csv(data_path)
    if 'Unnamed: 0' in df.columns: df = df.drop(columns=['Unnamed: 0'])
    if 'y' in df.columns: df = df.drop(columns=['y'])

    X_data = df.copy()
    feature_names = X_data.columns.tolist()
    X_background = X_data.sample(min(50, len(X_data)), random_state=42)

    for name, file in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, file)
        try:
            if file.endswith('.h5'):
                models[name] = load_model(path)
            else:
                with open(path, 'rb') as f:
                    models[name] = pickle.load(f)
            logger.info(f"✅ Model Loaded: {name}")
        except Exception as e:
            logger.error(f"❌ Failed to load {name}: {e}")
            models[name] = None


load_resources()


# ===================== HELPERS =====================
def plot_to_base64(fig):
    BG_COLOR = '#2b2d42'
    TEXT_COLOR = 'white'
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.gca()
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for text in ax.texts: text.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color('#8d99ae')
    for spine in ax.spines.values(): spine.set_edgecolor('#8d99ae')

    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()


def is_keras_model(model):
    return hasattr(model, 'layers') or str(type(model)).find('keras') != -1


def predict_proba_wrapper(model, model_name):
    if is_keras_model(model):
        def fn(x):
            if x.ndim == 2 and 'CNN' in model_name: x = x.reshape((x.shape[0], x.shape[1], 1))
            preds = model.predict(x, verbose=0)
            if preds.shape[1] == 1: return np.hstack([1 - preds, preds])
            return preds

        return fn
    else:
        def fn(x):
            df_temp = pd.DataFrame(x, columns=feature_names)
            return model.predict_proba(df_temp)

        return fn


# ===================== ROUTES =====================
@app.route('/')
def index():
    sample_df = X_data.sample(min(15, len(X_data)))
    records = []
    for idx, row in sample_df.iterrows():
        r = row.to_dict();
        r['id'] = idx;
        records.append(r)
    return render_template('index2.html', samples=records, model_list=[k for k, v in models.items() if v])


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        sample_id = int(request.form['sample_id'])
        model_name = request.form['model_name']
        logger.info(f"🔍 Analyzing Sample ID: {sample_id} with Model: {model_name}")

        model = models.get(model_name)
        if not model: return jsonify({'error': 'Model not loaded'}), 500

        instance_df = X_data.loc[[sample_id]]
        instance_np = instance_df.values
        results = {}

        # --- 1. PREDICTION ---
        if is_keras_model(model):
            inp = instance_np
            prob = model.predict(inp, verbose=0)[0]
            if len(prob) == 1: prob = [1 - prob[0], prob[0]]
        else:
            prob = model.predict_proba(instance_df)[0]

        pred_idx = int(np.argmax(prob))
        results['pred_label'] = class_names[pred_idx]
        results['pred_confidence'] = float(prob[pred_idx])
        results['is_positive'] = bool(pred_idx == 1)

        shap_text = "N/A"
        lime_text = "N/A"
        dice_text = "No counterfactuals found"

        # --- 2. SHAP ---
        plt.clf()
        try:
            CLASS_INDEX = 1
            if model_name == "RandomForest":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(instance_df)
                if isinstance(shap_values, list):
                    shap_val = shap_values[CLASS_INDEX]
                else:
                    shap_val = shap_values

                if hasattr(shap_val, 'shape') and len(shap_val.shape) == 2: shap_val = shap_val[0]
                base_val = explainer.expected_value
                if isinstance(base_val, list): base_val = base_val[CLASS_INDEX]
            else:
                back_summary = shap.sample(X_background, 20)

                def p_fn(X):
                    if isinstance(X, np.ndarray): X = pd.DataFrame(X, columns=feature_names)
                    return model.predict_proba(X)[:, 1]

                explainer = shap.KernelExplainer(p_fn, back_summary)
                shap_values = explainer.shap_values(instance_df, nsamples=50)
                shap_val = shap_values[0] if isinstance(shap_values, list) else shap_values
                if hasattr(shap_val, 'shape') and len(shap_val.shape) == 2: shap_val = shap_val[0]
                base_val = explainer.expected_value

            shap_exp = shap.Explanation(values=shap_val, base_values=base_val, data=instance_df.iloc[0].values,
                                        feature_names=feature_names)
            shap.plots.waterfall(shap_exp, show=False, max_display=10)
            results["shap_plot"] = plot_to_base64(plt.gcf())

            # Text for AI
            feature_imp = pd.DataFrame({'feature': feature_names, 'shap': shap_val})
            feature_imp['abs_shap'] = feature_imp['shap'].abs()
            top_features = feature_imp.sort_values(by='abs_shap', ascending=False).head(5)
            shap_text = ", ".join([f"{r['feature']} ({r['shap']:.2f})" for _, r in top_features.iterrows()])

        except Exception as e:
            logger.error(f"SHAP Failed: {e}")
            results["shap_plot"] = None

        # --- 3. LIME ---
        try:
            plt.clf()
            explainer_lime = lime.lime_tabular.LimeTabularExplainer(
                X_background.values, feature_names=feature_names, class_names=class_names, mode='classification',
                verbose=False
            )
            pred_fn = predict_proba_wrapper(model, model_name)
            exp = explainer_lime.explain_instance(instance_np[0], pred_fn, num_features=5, labels=[1])
            results['lime_plot'] = plot_to_base64(exp.as_pyplot_figure(label=1))
            lime_text = ", ".join([f"{k}: {v:.2f}" for k, v in exp.as_list(label=1)])
        except Exception as e:
            logger.error(f"LIME Failed: {e}")
            results['lime_plot'] = None

        # --- 4. DiCE ---
        try:
            d_data = X_background.copy();
            d_data['target'] = 0
            d = dice_ml.Data(dataframe=d_data, continuous_features=feature_names, outcome_name='target')
            backend = 'TF2' if is_keras_model(model) else 'sklearn'
            m = dice_ml.Model(model=model, backend=backend)
            exp_dice = dice_ml.Dice(d, m, method='random')
            cfs = exp_dice.generate_counterfactuals(instance_df, total_CFs=1, desired_class="opposite")
            cf_df = cfs.cf_examples_list[0].final_cfs_df.drop(columns=['target'], errors='ignore')
            results['dice_html'] = cf_df.to_html(classes='table table-bordered table-hover text-white', index=False,
                                                 border=0)

            orig = instance_df.iloc[0];
            new_cf = cf_df.iloc[0];
            changes = []
            for col in feature_names:
                if abs(orig[col] - new_cf[col]) > 0.01: changes.append(f"{col}: {orig[col]:.2f} -> {new_cf[col]:.2f}")
            if changes: dice_text = " | ".join(changes)
        except Exception as e:
            logger.error(f"DiCE Failed: {e}")
            results['dice_html'] = "CF generation failed."

        # --- 5. GEMINI (ENHANCED BANKING PROMPT) ---
        if GEMINI_AVAILABLE:
            try:
                logger.info("🤖 Querying Gemini API...")

                # تعریف وضعیت مشتری برای AI
                status_text = "✅ مشتری تمایل به افتتاح سپرده دارد (Subscribed)." if results[
                    'is_positive'] else "❌ مشتری تمایل به افتتاح سپرده ندارد (Rejected)."

                prompt = f"""
                شما تحلیلگر ارشد داده در یک بانک پرتغالی هستید.
                هدف: پیش‌بینی اینکه آیا مشتری "سپرده بلندمدت" (Term Deposit) افتتاح می‌کند یا خیر (متغیر y).

                داده‌ها بر اساس مقاله Moro et al., 2014 هستند. مفاهیم کلیدی:
                - duration: طول مدت تماس (ثانیه). مهم‌ترین عامل. هرچه بیشتر باشد، احتمال موفقیت بیشتر است.
                - euribor3m: نرخ بهره بین بانکی اروپا (نرخ بالاتر ممکن است تمایل به سپرده‌گذاری را تغییر دهد).
                - poutcome: نتیجه کمپین قبلی (success یعنی قبلاً خرید کرده، failure یعنی نه).
                - nr.employed: شاخص تعداد کارمندان (نشان‌دهنده وضعیت اقتصادی).

                وضعیت فعلی مشتری (پیش‌بینی مدل):
                {status_text}
                (میزان اطمینان مدل: {results['pred_confidence']:.2f})

                داده‌های تفسیرپذیری مدل (XAI):
                1. SHAP (مهم‌ترین عوامل تاثیرگذار): {shap_text}
                2. LIME (تاثیرات محلی): {lime_text}
                3. DiCE (تغییرات پیشنهادی برای تغییر نظر مشتری): {dice_text}

                وظیفه:
                یک گزارش مدیریتی به زبان **فارسی** بنویسید.
                اگر پیش‌بینی "منفی" است، صادقانه بگویید مشتری تمایلی ندارد و دلایل آن را بررسی کنید. الکی امیدواری ندهید مگر اینکه DiCE راهکار داده باشد.

                ساختار خروجی (با تگ HTML):
                1. <h3>🔍 تحلیل وضعیت (Why?):</h3> توضیح دهید چرا مدل این تصمیم را گرفت؟ به متغیرهایی مثل duration و euribor3m اشاره کنید.
                2. <h3>💡 پیشنهاد به بازاریاب (Action Plan):</h3> اگر مشتری رد کرده، چه کنیم؟ (مثلاً تماس طولانی‌تر؟ تغییر زمان تماس؟). از خروجی DiCE استفاده کنید.
                3. <h3>📋 جمع‌بندی:</h3> یک جمله نهایی برای مدیر شعبه.
                """

                response = gemini_model.generate_content(prompt)
                results['gemini_response'] = response.text
                logger.info("✅ Gemini Response Received.")
            except Exception as e:
                err_msg = str(e)
                logger.error(f"❌ Gemini API Error: {err_msg}")
                results['gemini_response'] = f"<p style='color:#ff7675'>خطا در دریافت تحلیل هوشمند: {err_msg}</p>"
        else:
            results['gemini_response'] = "<p style='color:orange'>Gemini API Config Missing.</p>"

        return jsonify(results)

    except Exception as e:
        logger.critical(f"🔥 Critical Server Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("🚀 Starting Flask Server...")
    app.run(debug=True, port=5000)