import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re
from pathlib import Path
import io

# Read project configuration explicitly; never import an inherited API key.
from dotenv import dotenv_values

# Third-party imports
from google import genai
from PIL import Image
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PDF processing imports
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReportInputError(ValueError):
    """An expected input failure with a message safe to display."""


class ReportProcessingError(RuntimeError):
    """An expected processing failure with a message safe to display."""


class MedicalReportScanner:
    
    
    def __init__(self, api_key: str = '', model: Optional[str] = None):
        self.api_key = api_key.strip()
        self.model = model or os.getenv('GEMINI_MODEL', 'gemini-3.8-flash')
        self.client = (
            genai.Client(api_key=self.api_key, http_options={'timeout': 60000})
            if self.api_key else None
        )
        self.analysis_history = []

    def _generate_content(self, contents) -> str:
        """Make one API request, reporting failures without exposing credentials."""
        if self.client is None:
            raise ReportInputError('Add a Google AI API key in Configuration or set GEMINI_API_KEY in .env to analyze your own report.')
        try:
            response = self.client.models.generate_content(model=self.model, contents=contents)
        except Exception as exc:
            code = str(getattr(exc, 'code', ''))
            if code in {'400', '401', '403'}:
                message = 'Google AI rejected the request. Check that your API key is valid and has access to the configured model.'
            elif code == '404':
                message = 'The configured Google AI model is unavailable. Update GEMINI_MODEL in .env to a model available to your account.'
            elif code == '429':
                message = 'Google AI quota or rate limit reached. Check your account quota and billing, then try again later.'
            elif code in {'500', '502', '503', '504'}:
                message = 'Google AI is temporarily unavailable. Try again later.'
            else:
                message = 'Could not contact Google AI. Check your connection, API key, and model configuration, then try again.'
            raise ReportProcessingError(message) from None
        if not isinstance(response.text, str) or not response.text.strip():
            raise ReportProcessingError('Google AI returned no readable content. Try a clearer report or check the model configuration.')
        return response.text.strip()
        

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract a text PDF, using a second reader if necessary."""
        if not PDF_AVAILABLE:
            raise ReportProcessingError('PDF support is unavailable. Install the packages in requirements.txt, then restart the app.')
        pdf_file.seek(0)
        raw = pdf_file.read()
        if not raw:
            raise ReportInputError('The uploaded PDF is empty. Choose a readable medical report.')
        extracted = []
        for reader in ('pdfplumber', 'PyPDF2'):
            try:
                if reader == 'pdfplumber':
                    with pdfplumber.open(io.BytesIO(raw)) as pdf:
                        pages = [page.extract_text() or '' for page in pdf.pages]
                else:
                    pdf = PyPDF2.PdfReader(io.BytesIO(raw))
                    pages = [page.extract_text() or '' for page in pdf.pages]
                text = '\n'.join(pages).strip()
                if len(text) >= 50:
                    return text
                if text:
                    extracted.append(text)
            except Exception:
                logger.warning('PDF text extraction failed with %s', reader)
        if extracted:
            return max(extracted, key=len)
        raise ReportInputError('No readable text could be extracted from this PDF. Use an unlocked text PDF, or upload a scanned page as an image.')

    def extract_text_from_image(self, image_source) -> str:
        """Read an image in memory without saving a patient report to disk."""
        try:
            with Image.open(image_source) as image:
                image.load()
                report_image = image.convert('RGB')
        except Exception:
            raise ReportInputError('The uploaded image could not be opened. Choose a valid PNG, JPG, TIFF, or BMP image.') from None
        prompt = """
        Extract all text from this medical report image.
        Preserve test names, values, units, reference intervals, patient information, and notes.
        Transcribe only visible information; do not invent missing values.
        """
        return self._generate_content([prompt, report_image])

    def process_uploaded_file(self, uploaded_file) -> str:
        """Extract text from supported uploads without temporary files."""
        extension = Path(uploaded_file.name).suffix.lower()
        if extension == '.pdf':
            return self.extract_text_from_pdf(uploaded_file)
        if extension in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}:
            uploaded_file.seek(0)
            return self.extract_text_from_image(io.BytesIO(uploaded_file.read()))
        raise ReportInputError('Unsupported file format. Upload a PDF, PNG, JPG, TIFF, or BMP report.')

    def clean_json_response(self, response_text: str) -> str:
        """Clean and extract JSON from AI response"""
        # Remove markdown code blocks
        if '```json' in response_text:
            start = response_text.find('```json') + 7
            end = response_text.rfind('```')
            if end > start:
                response_text = response_text[start:end]
        elif '```' in response_text:
            start = response_text.find('```') + 3
            end = response_text.rfind('```')
            if end > start:
                response_text = response_text[start:end]
        
        # Remove any leading/trailing whitespace
        response_text = response_text.strip()
        
        # Try to find JSON-like content if not already clean
        if not response_text.startswith('{'):
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group()
        
        return response_text
    
    def parse_medical_data(self, text: str) -> Dict:
        """Parse extracted text to identify medical parameters"""
        prompt = f"""
        Analyze this medical report text and extract structured information. This appears to be a comprehensive lab report.
        
        Text to analyze:
        {text}
        
        Please provide a JSON response with exactly this structure:
        {{
            "patient_info": {{
                "name": "patient name if available, otherwise 'Not specified'",
                "age": "age if available, otherwise 'Not specified'",
                "gender": "gender if available, otherwise 'Not specified'",
                "report_date": "date if available, otherwise 'Not specified'",
                "lab_number": "lab number if available, otherwise 'Not specified'"
            }},
            "test_categories": [
                {{
                    "category": "test category name (e.g., 'Complete Blood Count', 'Liver Function', 'Lipid Profile')",
                    "tests": [
                        {{
                            "test_name": "name of test",
                            "value": "measured value",
                            "unit": "unit of measurement",
                            "reference_range": "normal range",
                            "status": "normal/abnormal/high/low/borderline"
                        }}
                    ]
                }}
            ],
            "abnormal_findings": [
                "list of abnormal test results with brief description"
            ],
            "critical_values": [
                "list of any critical or extremely abnormal values"
            ]
        }}
        
        Important guidelines:
        - Look for common lab categories like CBC, Liver Panel, Kidney Panel, Lipid Screen, Thyroid Profile, HbA1c, Vitamins
        - Pay attention to numerical values and their units
        - Compare values to reference ranges when provided
        - If no test results are found, include an empty array for test_categories
        - Always include all required fields even if empty or "Not specified"
        - Only return valid JSON, no additional text or explanations
        """
        
        for attempt in range(3):
            response_text = self._generate_content(prompt)
            try:
                parsed_data = json.loads(self.clean_json_response(response_text))
            except json.JSONDecodeError:
                logger.warning('Report parsing returned invalid JSON, attempt %s', attempt + 1)
                continue
            if self.validate_parsed_data(parsed_data):
                return self.enhance_test_status(parsed_data)
            logger.warning('Report parsing returned an invalid structure, attempt %s', attempt + 1)
        raise ReportProcessingError('Google AI could not return valid report data after three attempts. Try a clearer report or a different configured model.')


    def enhance_test_status(self, parsed_data: Dict) -> Dict:
        """Normalize report statuses without replacing laboratory reference ranges."""
        for category in parsed_data.get('test_categories', []):
            for test in category.get('tests', []):
                test['status'] = test.get('status', 'unknown').strip().lower()
        parsed_data.setdefault('critical_values', [])
        return parsed_data

    @staticmethod
    def _string_list(value) -> bool:
        return isinstance(value, list) and all(isinstance(item, str) for item in value)

    def validate_parsed_data(self, data: Dict) -> bool:
        """Reject malformed nested fields before tables and reports use them."""
        if not isinstance(data, dict):
            return False
        if not isinstance(data.get('patient_info'), dict):
            return False
        if any(not isinstance(value, (str, int, float, type(None))) for value in data['patient_info'].values()):
            return False
        if not isinstance(data.get('test_categories'), list):
            return False
        if not self._string_list(data.get('abnormal_findings')) or not self._string_list(data.get('critical_values', [])):
            return False
        statuses = {'normal', 'abnormal', 'high', 'low', 'borderline', 'unknown'}
        for category in data['test_categories']:
            if not isinstance(category, dict) or not isinstance(category.get('category'), str):
                return False
            if not isinstance(category.get('tests'), list):
                return False
            for test in category['tests']:
                if not isinstance(test, dict):
                    return False
                if not all(isinstance(test.get(field), str) for field in ('test_name', 'unit', 'reference_range', 'status')):
                    return False
                if not isinstance(test.get('value'), (str, int, float)) or isinstance(test.get('value'), bool):
                    return False
                if test['status'].strip().lower() not in statuses:
                    return False
        return True

    def create_fallback_structure(self, original_text: str) -> Dict:
        """Create a basic fallback structure when parsing fails"""
        logger.info("Creating fallback structure for medical data")
        
        # Try to extract some basic information using simple regex
        lines = original_text.split('\n')
        potential_tests = []
        patient_name = "Not specified"
        patient_age = "Not specified"
        patient_gender = "Not specified"
        
        # Try to extract patient info
        for line in lines[:20]:  # Check first 20 lines
            if 'Mr.' in line or 'Mrs.' in line or 'Ms.' in line:
                name_match = re.search(r'(Mr\.|Mrs\.|Ms\.)\s+([A-Za-z\s]+)', line)
                if name_match:
                    patient_name = name_match.group(0).strip()
            
            if 'Years' in line or 'years' in line:
                age_match = re.search(r'(\d+)\s+[Yy]ears', line)
                if age_match:
                    patient_age = age_match.group(1) + " years"
            
            if 'Male' in line or 'Female' in line:
                gender_match = re.search(r'(Male|Female)', line)
                if gender_match:
                    patient_gender = gender_match.group(1)
        
        # Extract test results
        for line in lines:
            # Look for patterns like "TestName: value unit (range)"
            test_pattern = r'([A-Za-z\s]+):\s*([0-9.]+)\s*([A-Za-z/%]*)\s*\(?([0-9.-]+\s*[-–]\s*[0-9.-]+)?\)?'
            match = re.search(test_pattern, line)
            if match:
                test_name, value, unit, ref_range = match.groups()
                potential_tests.append({
                    "test_name": test_name.strip(),
                    "value": value,
                    "unit": unit or "",
                    "reference_range": ref_range or "Not specified",
                    "status": "unknown"
                })
        
        return {
            "patient_info": {
                "name": patient_name,
                "age": patient_age, 
                "gender": patient_gender,
                "report_date": "Not specified",
                "lab_number": "Not specified"
            },
            "test_categories": [{
                "category": "General Tests",
                "tests": potential_tests[:15]  # Limit to first 15 found tests
            }] if potential_tests else [],
            "abnormal_findings": ["Unable to automatically detect abnormal findings - manual review recommended"],
            "critical_values": []
        }
    
    def analyze_diagnosis(self, parsed_data: Dict) -> Dict:
        """Generate AI-powered diagnosis insights"""
        prompt = f"""
        As a medical AI assistant, analyze these comprehensive lab results and provide detailed insights:
        
        {json.dumps(parsed_data, indent=2)}
        
        Provide analysis in the following JSON format:
        {{
            "risk_assessment": {{
                "overall_risk": "low/moderate/high",
                "cardiovascular_risk": "low/moderate/high",
                "diabetes_risk": "low/moderate/high",
                "risk_factors": ["list identified risk factors"]
            }},
            "potential_conditions": [
                {{
                    "condition": "condition name",
                    "probability": "low/moderate/high",
                    "supporting_evidence": ["specific test results that support this"],
                    "description": "brief clinical explanation"
                }}
            ],
            "recommendations": [
                {{
                    "category": "lifestyle/dietary/medical/follow-up",
                    "recommendation": "specific actionable recommendation",
                    "priority": "low/medium/high",
                    "rationale": "why this recommendation is important"
                }}
            ],
            "follow_up_tests": [
                "suggested additional tests with rationale"
            ],
            "red_flags": [
                "critical findings requiring immediate medical attention"
            ],
            "positive_findings": [
                "normal or good results worth highlighting"
            ],
            "summary": "A comprehensive overall assessment summary including key findings and next steps"
        }}
        
        Consider:
        - HbA1c levels for diabetes assessment
        - Lipid profile for cardiovascular risk
        - Liver and kidney function markers
        - Vitamin levels and deficiencies
        - Blood count abnormalities
        - Thyroid function
        
        Important: Only return valid JSON, include disclaimer about professional medical consultation
        """
        
        for attempt in range(2):
            response_text = self._generate_content(prompt)
            try:
                diagnosis = json.loads(self.clean_json_response(response_text))
            except json.JSONDecodeError:
                logger.warning('AI insights returned invalid JSON, attempt %s', attempt + 1)
                continue
            if self.validate_diagnosis_data(diagnosis):
                diagnosis.setdefault('positive_findings', [])
                return diagnosis
            logger.warning('AI insights returned an invalid structure, attempt %s', attempt + 1)
        raise ReportProcessingError('Google AI could not return valid insights after two attempts. Your extracted test results can still be reviewed.')


    def validate_diagnosis_data(self, data: Dict) -> bool:
        """Validate the nested insights used by the UI and exported report."""
        if not isinstance(data, dict) or not isinstance(data.get('risk_assessment'), dict):
            return False
        risk = data['risk_assessment']
        levels = {'low', 'moderate', 'high', 'unknown'}
        if any(not isinstance(risk.get(field), str) or risk[field] not in levels
               for field in ('overall_risk', 'cardiovascular_risk', 'diabetes_risk')):
            return False
        if not self._string_list(risk.get('risk_factors', [])):
            return False
        if not isinstance(data.get('summary'), str):
            return False
        for field in ('follow_up_tests', 'red_flags', 'positive_findings'):
            if not self._string_list(data.get(field, [])):
                return False
        if not isinstance(data.get('potential_conditions'), list) or not isinstance(data.get('recommendations'), list):
            return False
        for condition in data['potential_conditions']:
            if not isinstance(condition, dict):
                return False
            if not all(isinstance(condition.get(field), str) for field in ('condition', 'description')):
                return False
            if not isinstance(condition.get('probability'), str) or condition['probability'] not in levels:
                return False
            if not self._string_list(condition.get('supporting_evidence', [])):
                return False
        for recommendation in data['recommendations']:
            if not isinstance(recommendation, dict):
                return False
            if not all(isinstance(recommendation.get(field), str) for field in ('category', 'recommendation')):
                return False
            if not isinstance(recommendation.get('rationale', ''), str):
                return False
            if not isinstance(recommendation.get('priority'), str) or recommendation['priority'] not in {'low', 'medium', 'high'}:
                return False
        return True

    def create_fallback_diagnosis(self) -> Dict:
        """Create fallback diagnosis when AI analysis fails"""
        return {
            "risk_assessment": {
                "overall_risk": "moderate",
                "cardiovascular_risk": "moderate",
                "diabetes_risk": "moderate",
                "risk_factors": ["Unable to complete automated risk assessment"]
            },
            "potential_conditions": [],
            "recommendations": [{
                "category": "medical",
                "recommendation": "Consult with healthcare provider for proper interpretation",
                "priority": "high",
                "rationale": "Professional medical interpretation required"
            }],
            "follow_up_tests": [],
            "red_flags": [],
            "positive_findings": [],
            "summary": "Automated analysis could not be completed. Please consult with a qualified healthcare professional for proper interpretation of these comprehensive lab results."
        }
    
    def generate_comprehensive_report(self, parsed_data: Dict, diagnosis: Dict) -> str:
        """Generate comprehensive medical report with enhanced formatting"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# 🏥 COMPREHENSIVE MEDICAL REPORT ANALYSIS
**Generated on:** {timestamp}  
**Analysis System:** AI Medical Report Scanner v2.0  
**Report Type:** Comprehensive Lab Panel Analysis

---

## ⚠️ MEDICAL DISCLAIMER
This analysis is generated by AI for educational and informational purposes only. 
**ALWAYS consult with qualified healthcare professionals for medical decisions and treatment.**

---

## 👤 PATIENT INFORMATION
"""
        
        if 'patient_info' in parsed_data:
            patient = parsed_data['patient_info']
            for key, value in patient.items():
                if value and value != "Not specified":
                    icon = {"name": "👤", "age": "📅", "gender": "⚧", "report_date": "📋", "lab_number": "🆔"}.get(key, "📌")
                    report += f"{icon} **{key.replace('_', ' ').title()}:** {value}\n"
        
        # Risk Assessment Summary
        if diagnosis and 'risk_assessment' in diagnosis:
            risk = diagnosis['risk_assessment']
            report += f"\n## 🎯 RISK ASSESSMENT SUMMARY\n"
            
            overall_risk = risk.get('overall_risk', 'moderate')
            risk_colors = {"low": "🟢", "moderate": "🟡", "high": "🔴"}
            report += f"**Overall Health Risk:** {risk_colors.get(overall_risk, '🟡')} **{overall_risk.upper()}**\n\n"
            
            # Specific risk categories
            if 'cardiovascular_risk' in risk:
                cv_risk = risk['cardiovascular_risk']
                report += f"**Cardiovascular Risk:** {risk_colors.get(cv_risk, '🟡')} {cv_risk.title()}\n"
            
            if 'diabetes_risk' in risk:
                dm_risk = risk['diabetes_risk']
                report += f"**Diabetes Risk:** {risk_colors.get(dm_risk, '🟡')} {dm_risk.title()}\n"
        
        # Test Results by Category
        report += "\n## 📊 DETAILED TEST RESULTS\n"
        
        if 'test_categories' in parsed_data and parsed_data['test_categories']:
            for category in parsed_data['test_categories']:
                if category.get('tests'):
                    report += f"\n### 🔬 {category.get('category', 'Unknown Category')}\n"
                    
                    # Create a table-like format
                    report += "| Test | Value | Reference Range | Status |\n"
                    report += "|------|-------|-----------------|--------|\n"
                    
                    for test in category['tests']:
                        test_name = test.get('test_name', 'Unknown Test')
                        value = f"{test.get('value', 'N/A')} {test.get('unit', '')}"
                        ref_range = test.get('reference_range', 'N/A')
                        status = test.get('status', 'unknown')
                        
                        status_icons = {
                            'normal': '🟢 Normal',
                            'abnormal': '🔴 Abnormal', 
                            'high': '🔺 High',
                            'low': '🔻 Low',
                            'borderline': '🟡 Borderline',
                            'unknown': '⚪ Unknown'
                        }
                        status_display = status_icons.get(status, '⚪ Unknown')
                        
                        report += f"| {test_name} | {value} | {ref_range} | {status_display} |\n"
                    
                    report += "\n"
        else:
            report += "No structured test results could be extracted from the report.\n"
        
        # Critical and Abnormal Findings
        if diagnosis and diagnosis.get('red_flags'):
            report += "\n## 🚨 CRITICAL FINDINGS - IMMEDIATE ATTENTION REQUIRED\n"
            for flag in diagnosis['red_flags']:
                report += f"- ⚠️ **{flag}**\n"
            report += "\n**🚑 ACTION REQUIRED:** Contact your healthcare provider immediately.\n"
        
        if 'abnormal_findings' in parsed_data and parsed_data['abnormal_findings']:
            report += "\n## ⚠️ ABNORMAL FINDINGS\n"
            for finding in parsed_data['abnormal_findings']:
                report += f"- 🔴 {finding}\n"
        
        # Positive Findings
        if diagnosis and diagnosis.get('positive_findings'):
            report += "\n## ✅ POSITIVE FINDINGS\n"
            for finding in diagnosis['positive_findings']:
                report += f"- 🟢 {finding}\n"
        
        # AI Analysis and Insights
        if diagnosis:
            if diagnosis.get('potential_conditions'):
                report += "\n## 🔍 POTENTIAL CONDITIONS TO DISCUSS WITH YOUR DOCTOR\n"
                for condition in diagnosis['potential_conditions']:
                    prob_icons = {"low": "🟢 Low", "moderate": "🟡 Moderate", "high": "🔴 High"}
                    prob_display = prob_icons.get(condition.get('probability', 'moderate'), '🟡 Moderate')
                    
                    report += f"\n**{condition.get('condition', 'Unknown')}** - Probability: {prob_display}\n"
                    report += f"- **Description:** {condition.get('description', 'No description available')}\n"
                    if condition.get('supporting_evidence'):
                        report += f"- **Supporting Evidence:** {', '.join(condition['supporting_evidence'])}\n"
            
            # Comprehensive Recommendations
            if diagnosis.get('recommendations'):
                report += "\n## 💡 PERSONALIZED RECOMMENDATIONS\n"
                categories = {}
                for rec in diagnosis['recommendations']:
                    cat = rec.get('category', 'general').title()
                    if cat not in categories:
                        categories[cat] = []
                    
                    priority_icons = {"low": "🔵", "medium": "🟡", "high": "🔴"}
                    priority = rec.get('priority', 'medium')
                    priority_display = priority_icons.get(priority, '🟡')
                    
                    rec_text = f"{priority_display} **{rec.get('recommendation', '')}**"
                    if rec.get('rationale'):
                        rec_text += f"\n  - *Rationale:* {rec['rationale']}"
                    
                    categories[cat].append(rec_text)
                
                category_icons = {
                    'Lifestyle': '🏃‍♂️',
                    'Dietary': '🥗',
                    'Medical': '⚕️',
                    'Follow-Up': '📅',
                    'General': '📌'
                }
                
                for cat, recs in categories.items():
                    icon = category_icons.get(cat, '📌')
                    report += f"\n### {icon} {cat} Recommendations\n"
                    for rec in recs:
                        report += f"- {rec}\n"
            
            # Follow-up Tests
            if diagnosis.get('follow_up_tests'):
                report += "\n## 🧪 SUGGESTED FOLLOW-UP TESTS\n"
                for test in diagnosis['follow_up_tests']:
                    report += f"- 📋 {test}\n"
            
            # Executive Summary
            if diagnosis.get('summary'):
                report += f"\n## 📋 EXECUTIVE SUMMARY\n"
                report += f"{diagnosis['summary']}\n"
        
        # Footer
        report += "\n---\n"
        report += "## 📞 NEXT STEPS\n"
        report += "1. **Schedule an appointment** with your healthcare provider to discuss these results\n"
        report += "2. **Bring this report** to your medical consultation\n"
        report += "3. **Ask questions** about any findings you don't understand\n"
        report += "4. **Follow recommended** lifestyle modifications and follow-up tests\n\n"
        
        report += "**🔒 Privacy Note:** Keep this report confidential and share only with authorized healthcare providers.\n"
        report += "**⚕️ Remember:** This AI analysis supports but does not replace professional medical advice.\n"
        
        return report
    
    def create_test_trend_chart(self, analyses: List[Dict]) -> Optional[go.Figure]:
        """Plot repeated numeric tests only when their names and units match."""
        if len(analyses) < 2:
            return None
        series = {}
        for analysis in analyses:
            timestamp = analysis.get('timestamp')
            if not timestamp:
                continue
            # One value per test per analysis, even if the report repeats a table.
            observations = {}
            for category in analysis.get('parsed_data', {}).get('test_categories', []):
                for test in category.get('tests', []):
                    name = str(test.get('test_name', '')).strip()
                    unit = str(test.get('unit', '')).strip()
                    value = str(test.get('value', '')).strip()
                    if not name or not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)', value):
                        continue
                    key = (re.sub(r'\s+', ' ', name).casefold(), unit.casefold())
                    observations[key] = (name, unit, float(value))
            for key, (name, unit, value) in observations.items():
                record = series.setdefault(key, {'name': name, 'unit': unit, 'points': []})
                record['points'].append((timestamp, value))
        repeated = [record for record in series.values() if len(record['points']) >= 2]
        if not repeated:
            return None
        columns = min(2, len(repeated))
        rows = (len(repeated) + columns - 1) // columns
        fig = make_subplots(rows=rows, cols=columns, subplot_titles=[record['name'] for record in repeated])
        for index, record in enumerate(repeated):
            row, column = divmod(index, columns)
            points = sorted(record['points'], key=lambda point: str(point[0]))
            fig.add_trace(go.Scatter(
                x=[point[0] for point in points], y=[point[1] for point in points],
                mode='lines+markers', name=record['name'],
            ), row=row + 1, col=column + 1)
            fig.update_yaxes(title_text=record['unit'], row=row + 1, col=column + 1)
        fig.update_layout(title_text='Health Markers Trend Analysis', showlegend=False, height=max(350, rows * 300))
        return fig

