"""Check tracked files for common credentials and private workstation details."""

import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


SECRET_PATTERNS = (
    rb"\bAIza[0-9A-Za-z_-]{30,}\b",
    rb"\bAQ\.[0-9A-Za-z_-]{30,}\b",
    rb"\bgh[pousr]_[0-9A-Za-z]{30,}\b",
    rb"\bgithub_pat_[0-9A-Za-z_]{40,}\b",
    rb"\bsk-[0-9A-Za-z_-]{24,}\b",
    rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    rb"https?://[^\s/:@]+:[^\s/@]+@",
)
HOME_PATH = re.compile(
    rb"(?:\b[a-z]:[/\\](?:users|documents and settings)[/\\]|/(?:Users|home)/)[^\s/\\\"'<>]+",
    re.IGNORECASE,
)
INTERNAL_MARKDOWN = re.compile(
    rb"(?:^#{1,6}\s+Files kept local\b|\.(?:local|codex-reference|verification)/)",
    re.IGNORECASE | re.MULTILINE,
)
PRIVATE_DIRECTORIES = {'.venv', 'venv', '.local', '.codex-reference', '.verification', '__pycache__'}


def private_filename(path):
    parts = PurePosixPath(path.replace('\\', '/')).parts
    name = parts[-1].lower()
    return (
        any(part.lower() in PRIVATE_DIRECTORIES for part in parts)
        or ((name == '.env' or name.startswith('.env.')) and name != '.env.example')
        or name == 'secrets.toml'
        or name.startswith('~$')
        or PurePosixPath(name).suffix in {'.docx', '.pdf'}
    )


def scan_content(path, payload):
    """Return categories only; matched values must never reach logs."""
    if private_filename(path):
        return ['private file']
    findings = []
    suffix = PurePosixPath(path).suffix.lower()
    if any(re.search(pattern, payload) for pattern in SECRET_PATTERNS):
        findings.append('possible credential')
    if HOME_PATH.search(payload):
        findings.append('personal filesystem path')
    if suffix in {'.md', '.markdown'} and INTERNAL_MARKDOWN.search(payload):
        findings.append('internal housekeeping text')
    return findings


def check_repository(root, staged=False):
    """Check the index for commits, or tracked working files for manual review."""
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.PIPE)

    paths = sorted(set(path.decode('utf-8') for path in git('ls-files', '-z').split(b'\0') if path))
    findings = []
    for path in paths:
        if private_filename(path):
            findings.append((path, 'private file'))
            continue
        try:
            payload = git('show', ':' + path) if staged else (root / path).read_bytes()
        except (OSError, subprocess.CalledProcessError):
            findings.append((path, 'file could not be checked'))
            continue
        findings.extend((path, category) for category in scan_content(path, payload))
    return paths, findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true', help='check exactly the Git index before committing')
    options = parser.parse_args()
    root = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    paths, findings = check_repository(root, staged=options.staged)
    if findings:
        for path, category in findings:
            print(f'{path}: {category}')
        print('Public-file check failed. Remove private content before committing.')
        return 1
    print(f'Public-file check passed ({len(paths)} tracked files).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
