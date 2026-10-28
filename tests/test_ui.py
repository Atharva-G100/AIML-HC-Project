"""Exercise the report workflow without loading credentials or calling an API."""

import copy
import html
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'
sys.path.insert(0, str(APP_PATH.parent))
from labnote.scanner import ReportInputError, ReportProcessingError
from labnote.ui import _read_report


class ReportInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            'GEMINI_API_KEY': '', 'GOOGLE_API_KEY': '',
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        # Stub the explicit project configuration before the first script run.
        # Clearing process variables alone does not isolate the user's .env.
        self.config_patch = patch('labnote.ui.dotenv_values', return_value={
            'GEMINI_API_KEY': '', 'GEMINI_MODEL': 'test-model',
        })
        self.config = self.config_patch.start()
        self.addCleanup(self.config_patch.stop)
        self.client_patch = patch('google.genai.Client')
        self.client = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
        self.assert_no_exceptions()

    def assert_no_exceptions(self):
        self.assertEqual(len(self.app.exception), 0,
                         [exception.message for exception in self.app.exception])

    def button(self, key):
        return next(button for button in self.app.button if button.key == key)

    def checkbox(self, key):
        return next(box for box in self.app.checkbox if box.key == key)

    def load_sample(self):
        self.button('load_sample').click().run()
        self.assert_no_exceptions()

    def assert_download_controls(self):
        labels = {element.proto.label for element in self.app.get('download_button')}
        self.assertEqual(labels, {
            'Download Report (Markdown)', 'Download Data (JSON)',
        })

    def test_initial_workspace_renders_without_credentials(self):
        rendered = '\n'.join(item.value for item in self.app.markdown)
        self.assertIn('Labnote', rendered)
        self.assertIn('Make sense of', rendered)
        self.assertIn('the small print.', rendered)
        self.assertEqual(len(self.app.get('file_uploader')), 1)
        self.assertEqual(self.app.session_state['analysis_history'], [])
        self.assertEqual(len(self.app.get('download_button')), 0)
        self.assertIsNotNone(self.button('load_sample'))
        self.client.assert_not_called()

    def test_inherited_key_is_ignored_when_project_key_is_blank(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'inherited-key-must-not-be-used'}):
            self.app.run()
            self.assert_no_exceptions()
            self.assertTrue(any(field.label == 'Gemini API key'
                                for field in self.app.text_input))
            self.client.assert_not_called()

    def test_sample_populates_patient_tables_stats_and_exports(self):
        self.load_sample()

        rendered = '\n'.join(item.value for item in self.app.markdown)
        self.assertIn('Demo Patient', rendered)
        self.assertIn('Sample report', rendered)
        self.assertIn('illustrative data', rendered)
        self.assertTrue(self.app.session_state['is_demo'])
        stats = {metric.label: metric.value.strip() for metric in self.app.metric}
        self.assertEqual(stats['Total Tests'], '9')
        self.assertEqual(stats['Normal Results'], '6')
        self.assertEqual(stats['Abnormal Findings'], '3')
        self.assertEqual(stats['Test Categories'], '4')
        self.assertEqual(len(self.app.dataframe), 4)
        self.assertEqual([tab.label for tab in self.app.tabs],
                         ['At a glance', 'Test results', 'Notes & next steps', 'Full report'])
        self.assert_download_controls()
        self.client.assert_not_called()

    def test_report_without_explanations_still_exports(self):
        self.checkbox('detailed_analysis').uncheck().run()
        self.load_sample()

        self.assertIsNone(self.app.session_state['diagnosis'])
        self.assertTrue(any('Turn on Include explanations' in message.value
                            for message in self.app.info))
        self.assertTrue(any(heading.value == 'Your complete report'
                            for heading in self.app.subheader))
        self.assertTrue(any('Demo Patient' in item.value for item in self.app.markdown))
        self.assert_download_controls()

    def test_history_survives_rerun_and_can_be_viewed(self):
        self.load_sample()
        history = copy.deepcopy(self.app.session_state['analysis_history'])
        self.assertEqual(len(history), 1)

        self.checkbox('include_trends').uncheck().run()
        self.assert_no_exceptions()
        self.assertEqual(self.app.session_state['analysis_history'], history)

        self.button('view_history').click().run()
        self.assert_no_exceptions()
        self.assertEqual(self.app.session_state['active_view'], 'history')
        self.assertTrue(any(heading.value == 'Recent reports'
                            for heading in self.app.subheader))
        self.assertEqual(len(self.app.json), 1)
        self.assertEqual(self.app.session_state['analysis_history'], history)

    def test_disabling_downloads_preserves_results(self):
        self.load_sample()
        data = copy.deepcopy(self.app.session_state['parsed_data'])

        self.checkbox('export_options').uncheck().run()
        self.assert_no_exceptions()
        self.assertEqual(len(self.app.get('download_button')), 0)
        self.assertEqual(self.app.session_state['parsed_data'], data)
        self.assertEqual(len(self.app.dataframe), 4)

    def test_new_report_clears_current_results_but_keeps_history(self):
        self.load_sample()
        history = copy.deepcopy(self.app.session_state['analysis_history'])

        self.button('new_report').click().run()
        self.assert_no_exceptions()
        self.assertEqual(self.app.session_state['active_view'], 'workspace')
        self.assertNotIn('parsed_data', self.app.session_state)
        self.assertNotIn('diagnosis', self.app.session_state)
        self.assertEqual(self.app.session_state['analysis_history'], history)
        self.assertEqual(len(self.app.dataframe), 0)
        self.assertEqual(len(self.app.get('download_button')), 0)

    def test_patient_text_is_escaped_in_custom_html(self):
        self.load_sample()
        payload = '<img src=x onerror="alert(1)">'
        data = copy.deepcopy(self.app.session_state['parsed_data'])
        data['patient_info']['name'] = payload
        self.app.session_state['parsed_data'] = data

        self.app.run()
        self.assert_no_exceptions()
        html_blocks = [item.value for item in self.app.markdown
                       if getattr(item.proto, 'allow_html', False)]
        self.assertTrue(any(html.escape(payload) in block for block in html_blocks))
        self.assertFalse(any(payload in block for block in html_blocks))


class ReportErrorPrivacyTests(unittest.TestCase):
    def failure_result(self, stage, exception_type, message):
        scanner = Mock()
        scanner.process_uploaded_file.return_value = 'Example readable report text. ' * 4
        scanner.parse_medical_data.return_value = {'patient_info': {}, 'test_categories': []}
        getattr(scanner, stage).side_effect = exception_type(message)
        with patch('labnote.ui.st') as interface, patch('labnote.ui._remember'):
            interface.session_state = {}
            succeeded = _read_report(scanner, SimpleNamespace(name='example.pdf'), True)
        return interface, succeeded

    def test_unexpected_errors_do_not_expose_internal_paths_or_tokens(self):
        private_details = 'internal/session/credentials.env token=fixture-secret'
        for stage in ('process_uploaded_file', 'analyze_diagnosis'):
            for exception_type in (Exception, ValueError, RuntimeError):
                with self.subTest(stage=stage, exception=exception_type.__name__):
                    interface, succeeded = self.failure_result(stage, exception_type, private_details)
                    displayed = '\n'.join(str(call.args[0]) for kind in ('error', 'warning', 'write')
                                          for call in getattr(interface, kind).call_args_list)
                    self.assertNotIn('credentials.env', displayed)
                    self.assertNotIn('fixture-secret', displayed)
                    self.assertEqual(succeeded, stage == 'analyze_diagnosis')
                    if stage == 'analyze_diagnosis':
                        self.assertNotIn('fixture-secret', interface.session_state['reading_warning'])
                        self.assertIn('parsed_data', interface.session_state)

    def test_public_file_and_api_errors_keep_their_guidance(self):
        public_errors = (
            ('process_uploaded_file', ReportInputError,
             'The uploaded PDF is empty. Choose a readable medical report.'),
            ('process_uploaded_file', ReportProcessingError,
             'Google AI quota or rate limit reached. Check your account quota and billing, then try again later.'),
            ('analyze_diagnosis', ReportProcessingError,
             'The configured Google AI model is unavailable. Update GEMINI_MODEL in .env to a model available to your account.'),
        )
        for stage, exception_type, message in public_errors:
            with self.subTest(stage=stage, exception=exception_type.__name__):
                interface, succeeded = self.failure_result(stage, exception_type, message)
                displayed = '\n'.join(str(call.args[0]) for kind in ('error', 'warning')
                                      for call in getattr(interface, kind).call_args_list)
                self.assertIn(message, displayed)
                self.assertEqual(succeeded, stage == 'analyze_diagnosis')


if __name__ == '__main__':
    unittest.main()
