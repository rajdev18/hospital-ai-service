
from flask import Flask, request, jsonify
from symptom_checker import check_symptoms
from discharge_summary import generate_discharge_summary
from patient_qa import answer_patient_question

app = Flask(__name__)

# ── Health Check ─────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "service": "Hospital AI Service",
        "features": [
            "symptom-checker",
            "discharge-summary",
            "patient-qa"
        ]
    })

# ── Symptom Checker ───────────────────────────────
@app.route("/ai/symptom-checker", methods=["POST"])
def symptom_checker():
    data = request.get_json()

    if not data or "symptoms" not in data:
        return jsonify({"error": "Please provide symptoms"}), 400

    symptoms = data["symptoms"]
    result = check_symptoms(symptoms)

    return jsonify({
        "symptoms": symptoms,
        "recommendation": result
    })

# ── Discharge Summary ─────────────────────────────
@app.route("/ai/discharge-summary", methods=["POST"])
def discharge_summary():
    data = request.get_json()

    required = ["patient_name", "diagnosis", "prescription",
                "symptoms", "notes", "doctor_name"]

    for field in required:
        if field not in data:
            return jsonify({"error": f"{field} is required"}), 400

    summary = generate_discharge_summary(
        data["patient_name"],
        data["diagnosis"],
        data["prescription"],
        data["symptoms"],
        data["notes"],
        data["doctor_name"]
    )

    return jsonify({
        "patient_name": data["patient_name"],
        "discharge_summary": summary
    })

# ── Patient Q&A ───────────────────────────────────
@app.route("/ai/patient-qa", methods=["POST"])
def patient_qa():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Please provide a question"}), 400

    question = data["question"]
    answer = answer_patient_question(question)

    return jsonify({
        "question": question,
        "answer": answer
    })

# ── Run ───────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)