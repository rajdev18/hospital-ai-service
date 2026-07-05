
from groq import Groq

client = Groq(api_key="GROQ_API_KEY")

def generate_discharge_summary(patient_name, diagnosis, prescription, 
                                symptoms, notes, doctor_name):
    prompt = f"""You are a medical professional writing a formal hospital discharge summary.
Generate a professional, detailed discharge summary based on the information provided.

Patient Information:
- Patient Name: {patient_name}
- Attending Doctor: {doctor_name}
- Diagnosis: {diagnosis}
- Symptoms: {symptoms}
- Prescription: {prescription}
- Doctor Notes: {notes}

Generate a formal discharge summary that includes:
1. Patient Details
2. Chief Complaint
3. Diagnosis
4. Treatment Given
5. Medications Prescribed
6. Discharge Instructions
7. Follow-up Recommendations
8. Emergency Contact Instructions

Write in a professional medical format."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"