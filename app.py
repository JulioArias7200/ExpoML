from flask import Flask, render_template, jsonify, request, url_for
from ml.pipeline import train_classification, train_regression, dataset_summary, load_data

app = Flask(__name__)

REG_TARGETS = ["Cholesterol Level", "BMI", "Triglyceride Level", "Blood Pressure"]

def _parse_hparams():
    """Lee y acota n_estimators / learning_rate desde query o form.
    Antes learning_rate no tenía límites: un valor negativo, cero o enorme
    podía romper el entrenamiento (o tardar mucho) sin dar un error claro."""
    n_est = int(request.values.get('n_estimators', 100))
    lr = float(request.values.get('learning_rate', 0.1))
    n_est = max(10, min(n_est, 300))
    lr = max(0.01, min(lr, 1.0))
    return n_est, lr

@app.route('/')
def index():
    # redirige a boosting como overview
    return render_template('boosting.html')

@app.route('/boosting')
def boosting():
    return render_template('boosting.html')

@app.route('/bagging')
def bagging():
    return render_template('bagging.html')

@app.route('/comparative')
def comparative():
    return render_template('comparative.html')

# --------- API ---------
@app.route('/api/summary')
def api_summary():
    try:
        s = dataset_summary()
        # derivar métricas simples
        s["plots_available"] = []
        return jsonify(s)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/train/classification', methods=['GET','POST'])
def api_train_classification():
    try:
        n_est, lr = _parse_hparams()
        result = train_classification(n_estimators=n_est, learning_rate=lr)
        # convertir paths a url_for static
        plots_url = {k: url_for('static', filename=v) for k,v in result['plots'].items() if v}
        return jsonify({"metrics": result['metrics'], "plots": plots_url, "params": {"n_estimators": n_est, "learning_rate": lr}})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/train/regression', methods=['GET','POST'])
def api_train_regression():
    try:
        target = request.values.get('target', 'Cholesterol Level')
        if target not in REG_TARGETS:
            target = "Cholesterol Level"
        n_est, lr = _parse_hparams()
        result = train_regression(target_col=target, n_estimators=n_est, learning_rate=lr)
        plots_url = {k: url_for('static', filename=v) for k,v in result['plots'].items() if v}
        return jsonify({"metrics": result['metrics'], "plots": plots_url, "target": target, "params": {"n_estimators": n_est, "learning_rate": lr}})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/train/compare', methods=['GET','POST'])
def api_train_compare():
    # ejecuta ambos y retorna comparativa
    try:
        n_est, lr = _parse_hparams()
        target = request.values.get('target', 'Cholesterol Level')
        if target not in REG_TARGETS:
            target = "Cholesterol Level"
        cls = train_classification(n_estimators=n_est, learning_rate=lr)
        reg = train_regression(target_col=target, n_estimators=n_est, learning_rate=lr)
        return jsonify({
            "classification": {"metrics": cls["metrics"], "plots": {k: url_for('static', filename=v) for k,v in cls["plots"].items() if v}},
            "regression": {"metrics": reg["metrics"], "plots": {k: url_for('static', filename=v) for k,v in reg["plots"].items() if v}},
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/demo')
def demo():
    # página demo que consume las APIs via JS
    return render_template('demo.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
