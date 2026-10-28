"""Regression checks for uploads, API failures, and structured report output."""
import copy
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labnote.scanner import MedicalReportScanner, ReportInputError, ReportProcessingError
from PIL import Image


def report_data():
    return {
        'patient_info': {'name': 'Example Patient', 'age': '30', 'gender': 'Not specified'},
        'test_categories': [{
            'category': 'Example tests',
            'tests': [{
                'test_name': 'Fasting glucose', 'value': '60', 'unit': 'mg/dL',
                'reference_range': '70-100', 'status': 'low',
            }],
        }],
        'abnormal_findings': ['Example result below the interval printed on the report.'],
        'critical_values': [],
    }


def diagnosis_data():
    return {
        'risk_assessment': {
            'overall_risk': 'unknown', 'cardiovascular_risk': 'unknown',
            'diabetes_risk': 'unknown', 'risk_factors': [],
        },
        'potential_conditions': [], 'recommendations': [],
        'follow_up_tests': [], 'red_flags': [], 'positive_findings': [],
        'summary': 'Example output for software testing.',
    }


class ScannerRegressionTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {'GEMINI_MODEL': 'test-model'})
        self.environment.start()
        self.client_patch = patch('labnote.scanner.genai.Client')
        self.client_constructor = self.client_patch.start()
        self.scanner = MedicalReportScanner('test-api-key')
        self.generate = self.scanner.client.models.generate_content

    def tearDown(self):
        self.client_patch.stop()
        self.environment.stop()

    def test_client_uses_configured_model_and_timeout(self):
        self.client_constructor.assert_called_once_with(
            api_key='test-api-key', http_options={'timeout': 60000},
        )
        self.generate.return_value = SimpleNamespace(text='visible text')
        self.assertEqual(self.scanner._generate_content('input'), 'visible text')
        self.generate.assert_called_once_with(model='test-model', contents='input')

    def test_no_key_allows_preview_but_rejects_api_work(self):
        scanner = MedicalReportScanner('')
        with self.assertRaisesRegex(ReportInputError, 'GEMINI_API_KEY') as raised:
            scanner.parse_medical_data('Example text')
        self.assertIsInstance(raised.exception, ValueError)
        self.generate.assert_not_called()

    def test_api_errors_are_not_retried_or_reported_as_success(self):
        failure = RuntimeError('private request details test-api-key')
        failure.code = 429
        self.generate.side_effect = failure
        with self.assertRaisesRegex(ReportProcessingError, 'quota') as raised:
            self.scanner.parse_medical_data('Example text')
        self.assertIsInstance(raised.exception, RuntimeError)
        self.assertNotIn('test-api-key', str(raised.exception))
        self.assertEqual(self.generate.call_count, 1)

    def test_malformed_json_is_retried_and_report_status_is_preserved(self):
        self.generate.side_effect = [
            SimpleNamespace(text='invalid JSON'),
            SimpleNamespace(text='```json\n' + json.dumps(report_data()) + '\n```'),
        ]
        parsed = self.scanner.parse_medical_data('Example text')
        self.assertEqual(self.generate.call_count, 2)
        self.assertEqual(parsed['test_categories'][0]['tests'][0]['status'], 'low')
        self.assertNotIn('clinical_significance', parsed['test_categories'][0]['tests'][0])

    def test_nested_report_errors_end_in_an_error_after_bounded_attempts(self):
        malformed = report_data()
        malformed['test_categories'][0]['tests'] = [None]
        self.generate.return_value = SimpleNamespace(text=json.dumps(malformed))
        with self.assertRaisesRegex(RuntimeError, 'three attempts'):
            self.scanner.parse_medical_data('Example text')
        self.assertEqual(self.generate.call_count, 3)

    def test_nested_diagnosis_fields_are_validated_without_type_errors(self):
        self.assertTrue(self.scanner.validate_diagnosis_data(diagnosis_data()))
        malformed = diagnosis_data()
        malformed['risk_assessment']['overall_risk'] = {'bad': 'value'}
        self.assertFalse(self.scanner.validate_diagnosis_data(malformed))
        malformed = diagnosis_data()
        malformed['potential_conditions'] = [None]
        self.assertFalse(self.scanner.validate_diagnosis_data(malformed))

    def test_invalid_diagnosis_is_not_replaced_with_invented_risk(self):
        self.generate.return_value = SimpleNamespace(text='{}')
        with self.assertRaisesRegex(RuntimeError, 'two attempts'):
            self.scanner.analyze_diagnosis(report_data())
        self.assertEqual(self.generate.call_count, 2)

    def test_invalid_pdf_raises_instead_of_returning_error_text(self):
        with self.assertRaisesRegex(ReportInputError, 'No readable text'):
            self.scanner.extract_text_from_pdf(io.BytesIO(b'not a PDF'))
        self.generate.assert_not_called()

    def test_image_upload_stays_in_memory(self):
        upload = io.BytesIO()
        Image.new('RGB', (12, 12), color='white').save(upload, format='PNG')
        upload.name = '../../example.png'
        self.generate.return_value = SimpleNamespace(text='Example transcription')
        with patch('builtins.open', side_effect=AssertionError('Unexpected disk write')):
            self.assertEqual(self.scanner.process_uploaded_file(upload), 'Example transcription')
        contents = self.generate.call_args.kwargs['contents']
        self.assertIsInstance(contents[1], Image.Image)

    def test_trends_require_repeated_numeric_values_with_matching_units(self):
        first = {'timestamp': '2026-01-01T10:00:00', 'parsed_data': report_data()}
        second = {'timestamp': '2026-02-01T10:00:00', 'parsed_data': copy.deepcopy(report_data())}
        second['parsed_data']['test_categories'][0]['tests'][0]['value'] = '80'
        self.assertIsNone(self.scanner.create_test_trend_chart([first]))
        chart = self.scanner.create_test_trend_chart([first, second])
        self.assertEqual(len(chart.data), 1)
        self.assertEqual(list(chart.data[0].y), [60.0, 80.0])
        self.assertEqual(chart.data[0].name, 'Fasting glucose')
        second['parsed_data']['test_categories'][0]['tests'][0]['unit'] = 'mmol/L'
        self.assertIsNone(self.scanner.create_test_trend_chart([first, second]))

    def test_report_export_works_without_detailed_ai_insights(self):
        report = self.scanner.generate_comprehensive_report(report_data(), None)
        self.assertIn('Example Patient', report)
        self.assertIn('Fasting glucose', report)
        self.assertNotIn('Overall Health Risk', report)


if __name__ == '__main__':
    unittest.main()
