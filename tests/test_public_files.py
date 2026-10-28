"""Prove publication checks reject private content without printing it."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_public_files import check_repository, scan_content


class PublicFileTests(unittest.TestCase):
    def key(self):
        return ('AI' + 'za' + 'x' * 35).encode()

    def test_configuration_template_is_allowed(self):
        self.assertEqual(scan_content('.env.example', b'GEMINI_API_KEY=your_google_ai_studio_api_key\n'), [])

    def test_credentials_are_categorized_without_returning_the_value(self):
        result = scan_content('app.py', self.key())
        self.assertEqual(result, ['possible credential'])
        self.assertNotIn(self.key().decode(), str(result))

    def test_private_filenames_are_rejected(self):
        for name in ['.env', '.env.production', '.venv/settings.txt', '.streamlit/secrets.toml', '~$report.docx']:
            with self.subTest(name=name):
                self.assertEqual(scan_content(name, b''), ['private file'])

    def test_personal_home_paths_are_rejected(self):
        for value in ['C:' + '/Users/' + 'example/project', '/' + 'home/' + 'example/project']:
            with self.subTest(path=value):
                self.assertEqual(scan_content('settings.txt', value.encode()), ['personal filesystem path'])

    def test_internal_notes_are_rejected_only_in_markdown(self):
        note = ('## Files ' + 'kept local\n').encode()
        self.assertEqual(scan_content('README.md', note), ['internal housekeeping text'])
        self.assertEqual(scan_content('source.py', note), [])
        self.assertEqual(scan_content('README.md', b'Use .venv to run the app.'), [])

    def test_documents_are_rejected_before_contents_are_read(self):
        for name in ['report.docx', 'report.PDF']:
            with self.subTest(name=name):
                self.assertEqual(scan_content(name, None), ['private file'])

    def test_other_common_credential_formats_are_rejected(self):
        for value in ['A' + 'Q.' + 'x' * 40, 'gh' + 'p_' + 'x' * 40,
                      '-----BEGIN ' + 'PRIVATE KEY-----']:
            with self.subTest(prefix=value[:3]):
                self.assertEqual(scan_content('settings.txt', value.encode()), ['possible credential'])

    def test_staged_check_uses_index_instead_of_working_copy(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            root = Path(directory)
            self.assertEqual(root.resolve().parent, Path(__file__).resolve().parents[1])
            subprocess.run(['git', 'init', '-q', directory], check=True)
            path = root / 'app.py'
            path.write_bytes(self.key())
            subprocess.run(['git', 'add', 'app.py'], cwd=root, check=True, capture_output=True)
            path.write_text('safe working copy\n', encoding='utf-8')
            self.assertEqual(check_repository(root, staged=True)[1], [('app.py', 'possible credential')])
            self.assertEqual(check_repository(root, staged=False)[1], [])


if __name__ == '__main__':
    unittest.main()
