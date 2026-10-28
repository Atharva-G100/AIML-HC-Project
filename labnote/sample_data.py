"""Synthetic report for trying the interface without sending data to an API."""

from copy import deepcopy


def load_sample_analysis():
    """Return fresh demo values, never a result computed from an uploaded file."""
    data = {
        "patient_info": {
            "name": "Demo Patient",
            "age": "45 years",
            "gender": "Male",
            "report_date": "2026-10-01",
            "lab_number": "DEMO-001",
        },
        "test_categories": [
            {"category": "Complete Blood Count (CBC)", "tests": [
                {"test_name": "Hemoglobin", "value": "14.2", "unit": "g/dL", "reference_range": "13.0–17.0", "status": "normal"},
                {"test_name": "White Blood Cell Count", "value": "7.2", "unit": "10^3/µL", "reference_range": "4.0–11.0", "status": "normal"},
                {"test_name": "Platelet Count", "value": "250", "unit": "10^3/µL", "reference_range": "150–450", "status": "normal"},
            ]},
            {"category": "Blood Glucose", "tests": [
                {"test_name": "Fasting Glucose", "value": "110", "unit": "mg/dL", "reference_range": "70–100", "status": "high"},
                {"test_name": "HbA1c", "value": "5.4", "unit": "%", "reference_range": "4.0–5.6", "status": "normal"},
            ]},
            {"category": "Lipid Profile", "tests": [
                {"test_name": "Total Cholesterol", "value": "215", "unit": "mg/dL", "reference_range": "<200", "status": "high"},
                {"test_name": "HDL Cholesterol", "value": "35", "unit": "mg/dL", "reference_range": ">40", "status": "low"},
                {"test_name": "Triglycerides", "value": "120", "unit": "mg/dL", "reference_range": "<150", "status": "normal"},
            ]},
            {"category": "Kidney Function", "tests": [
                {"test_name": "Creatinine", "value": "1.0", "unit": "mg/dL", "reference_range": "0.7–1.3", "status": "normal"},
            ]},
        ],
        "abnormal_findings": [
            "Sample fasting glucose is above the sample report reference range.",
            "Sample total cholesterol is above the sample report reference range.",
            "Sample HDL cholesterol is below the sample report reference range.",
        ],
        "critical_values": [],
    }
    diagnosis = {
        "risk_assessment": {
            "overall_risk": "moderate", "cardiovascular_risk": "moderate",
            "diabetes_risk": "low", "risk_factors": ["Illustrative sample risk labels; no AI assessment was performed."],
        },
        "potential_conditions": [],
        "recommendations": [
            {"category": "medical", "recommendation": "Discuss actual lab results with a qualified healthcare professional.", "priority": "high", "rationale": "These sample values demonstrate the report layout only."},
            {"category": "follow-up", "recommendation": "Compare actual results with the reference ranges printed by your laboratory.", "priority": "medium", "rationale": "Reference ranges vary between laboratories and methods."},
        ],
        "follow_up_tests": [], "red_flags": [],
        "positive_findings": ["The sample report includes normal results to demonstrate table styling."],
        "summary": "SYNTHETIC DEMONSTRATION. These values and risk labels are sample output, not an analysis of an uploaded report or a medical assessment.",
    }
    return deepcopy(data), deepcopy(diagnosis)
