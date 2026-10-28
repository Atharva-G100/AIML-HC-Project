"""Labnote's presentation layer. Extraction and analysis live in scanner.py."""
from copy import deepcopy
from datetime import datetime
from html import escape
import json
import re

from dotenv import dotenv_values
import pandas as pd
import streamlit as st

from .config import ASSETS_DIR, ENV_FILE
from .sample_data import load_sample_analysis
from .scanner import ReportInputError, ReportProcessingError

ABNORMAL = {'abnormal', 'high', 'low', 'borderline'}


def _text(value):
    return escape(str(value if value not in (None, '', 'Not specified') else 'Not provided'))


def _remember(source):
    st.session_state.analysis_history.append({
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'source': source,
        'parsed_data': deepcopy(st.session_state.parsed_data),
        'diagnosis': deepcopy(st.session_state.get('diagnosis')),
        'is_demo': st.session_state.get('is_demo', False),
    })
    st.session_state.analysis_history = st.session_state.analysis_history[-20:]


def _new_report():
    for name in ('parsed_data', 'diagnosis', 'source_name', 'is_demo', 'extracted_text', 'reading_warning'):
        st.session_state.pop(name, None)
    st.session_state.upload_revision += 1
    st.session_state.active_view = 'workspace'


def _history_view():
    st.session_state.active_view = 'history'


def _sidebar():
    with st.sidebar:
        st.markdown('''<div class="wordmark"><span class="wordmark-icon" aria-hidden="true">l.</span>
            <span>Labnote<small>MEDICAL REPORT READER</small></span></div>
            <div class="sidebar-section-label">YOUR WORKSPACE</div>''', unsafe_allow_html=True)
        st.button('New report', key='new_report', on_click=_new_report,
                  icon=':material/add:', width='stretch', type='primary')
        st.button('Recent reports', key='view_history', on_click=_history_view,
                  icon=':material/history:', width='stretch')
        st.caption(f"{len(st.session_state.analysis_history)} reports in this session")

        with st.expander('Reading preferences'):
            trends = st.checkbox('Show trends', value=True, key='include_trends')
            detailed = st.checkbox('Include explanations', value=True, key='detailed_analysis')
            downloads = st.checkbox('Allow downloads', value=True, key='export_options')

        config = dotenv_values(ENV_FILE)
        key = config.get('GEMINI_API_KEY') or ''
        with st.expander('Connection', expanded=not bool(key)):
            if not key:
                key = st.text_input('Gemini API key', type='password', key='entered_api_key',
                                    help='Your own Google AI key. It stays in this browser session.')
            st.caption('Report content is sent to Google AI when you choose Read my report.')
        state = 'configured' if key else 'missing'
        label = 'Key configured' if key else 'Sample mode available'
        st.markdown(f'<div class="connection-state {state}"><span></span>{label}</div>', unsafe_allow_html=True)
        with st.expander('How it works'):
            st.write('1. Add a PDF or a clear photograph of your lab report.\n'
                     '2. Choose Read my report.\n'
                     '3. Explore your results and save the notes you need.')
            st.caption('Scanned PDFs without selectable text should be uploaded as images. '
                       'Recent reports are kept only in this browser session.')
        st.markdown('''<div class="sidebar-note"><small>An educational project.<br>
            Always discuss medical decisions with a qualified professional.</small></div>''',
                    unsafe_allow_html=True)
    return config, key, trends, detailed, downloads


def _hero():
    st.markdown('''<div class="workspace-masthead"><span>REPORT WORKSPACE</span><span>PDF & IMAGE REPORTS</span></div>
    <section class="intro">
      <div class="intro-copy"><div class="eyebrow">MEDICAL REPORT READER</div>
        <h1>Make sense of<br><em>the small print.</em></h1>
        <p>Lab reports can be a lot to take in. Bring the numbers together,
        understand the printed ranges, and find the questions worth asking.</p>
      </div>
      <div class="report-sketch" aria-hidden="true">
        <svg viewBox="0 0 300 245" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M57 34L222 24L236 218L71 230Z" fill="#e9e1d5"/>
          <path d="M72 18H218L243 43V211H72V18Z" fill="#fffcf5" stroke="#39332d" stroke-width="1.4"/>
          <path d="M218 18V43H243" stroke="#39332d" stroke-width="1.4"/>
          <path d="M94 52H145M94 66H193" stroke="#7b3b47" stroke-width="3"/>
          <path d="M94 94H213M94 119H213M94 144H213M94 169H164" stroke="#d8d1c5" stroke-width="1.2"/>
          <path d="M94 105H142M94 130H153M94 155H130" stroke="#5c574e" stroke-width="2"/>
          <circle cx="211" cy="160" r="36" fill="#f5eee6" stroke="#7b3b47" stroke-width="1.5"/>
          <path d="M236 187L266 218" stroke="#7b3b47" stroke-width="5" stroke-linecap="round"/>
          <path d="M191 154H230M191 166H218" stroke="#7b3b47" stroke-width="2"/>
          <path d="M36 92C22 103 20 119 28 130M28 130L18 125M28 130L30 118" stroke="#9a7d66" stroke-width="1.5" stroke-linecap="round"/>
          <path d="M95 190C119 187 142 191 157 188" stroke="#7b3b47" stroke-width="1.7" stroke-linecap="round"/>
        </svg>
      </div>
    </section>''', unsafe_allow_html=True)


def _read_report(scanner, uploaded, detailed):
    try:
        with st.status('Reading your report…', expanded=True) as progress:
            st.write('Picking up the text and numbers.')
            text = scanner.process_uploaded_file(uploaded)
            if not text or len(text.strip()) < 50:
                raise ReportInputError('There is not enough readable text. Try a clearer image or a text-based PDF.')
            st.write('Putting the test results in order.')
            data = scanner.parse_medical_data(text)
            diagnosis, notes_error = None, None
            if detailed:
                st.write('Preparing the explanatory notes.')
                try:
                    diagnosis = scanner.analyze_diagnosis(data)
                except (ReportInputError, ReportProcessingError) as exc:
                    notes_error = str(exc)
                except Exception:
                    notes_error = 'Check your connection and try reading the report again.'
            st.session_state.update(parsed_data=data, diagnosis=diagnosis, extracted_text=text,
                                    is_demo=False, source_name=uploaded.name, reading_warning=notes_error)
            _remember(uploaded.name)
            progress.update(label='Your report is ready', state='complete', expanded=False)
        if notes_error:
            st.warning(f'Your test results are ready, but the explanatory notes could not be completed: {notes_error}')
        return True
    except (ReportInputError, ReportProcessingError) as exc:
        st.error(f'This report could not be read. {exc}')
        return False
    except Exception:
        st.error('This report could not be read. Check your report file and connection, then try again.')
        return False


def _upload(scanner, key, detailed):
    with st.container(key='workbench'):
        left, right = st.columns([1.5, 1], gap='large')
        with left:
            with st.container(key='upload-panel', border=True):
                st.markdown('<div class="eyebrow">01 / ADD YOUR REPORT</div>', unsafe_allow_html=True)
                st.subheader('Upload a lab report')
                st.caption('A lab report PDF or a clear photograph.')
                upload = st.file_uploader('Choose a file', type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp'],
                                          key=f'report_upload_{st.session_state.upload_revision}',
                                          label_visibility='collapsed')
                if upload:
                    st.text(f'{upload.name} · {upload.size / 1024:.1f} KB')
                read = st.button('Read my report', key='analyze_report', type='primary',
                                 width='stretch', disabled=not bool(key and upload),
                                 icon=':material/arrow_forward:')
                st.caption('Your report is sent to Google AI only when you choose to read it.')
                if read:
                    if _read_report(scanner, upload, detailed):
                        st.rerun()
        with right:
            st.markdown('''<div class="before-upload"><div class="eyebrow">A GOOD PLACE TO START</div>
              <h3>Before you upload</h3>
              <ul><li>Include the full page and its reference ranges.</li>
              <li>Make sure the text and numbers are easy to read.</li>
              <li>Keep the original for your next appointment.</li></ul>
              <div class="sample-intro">Just looking around?</div>
              <p>See how a report comes together with a short, illustrative example.</p></div>''', unsafe_allow_html=True)
            if st.button('Try a sample report', key='load_sample', icon=':material/open_in_new:'):
                data, diagnosis = load_sample_analysis()
                st.session_state.update(parsed_data=data, diagnosis=diagnosis if detailed else None,
                                        is_demo=True, source_name='Synthetic sample report', reading_warning=None)
                _remember('Synthetic sample report')
                st.rerun()


def _patient(data):
    fields = [('Name', 'name'), ('Age', 'age'), ('Gender', 'gender'), ('Report date', 'report_date'), ('Lab reference', 'lab_number')]
    patient = data.get('patient_info', {})
    items = ''.join(f'<div><dt>{label}</dt><dd>{_text(patient.get(name))}</dd></div>' for label, name in fields)
    st.markdown(f'<dl class="patient-details">{items}</dl>', unsafe_allow_html=True)


def _findings(findings, title='Worth a closer look'):
    if findings:
        st.subheader(title)
        items = ''.join(f'<li>{_text(item)}</li>' for item in findings)
        st.markdown(f'<ul class="finding-list">{items}</ul>', unsafe_allow_html=True)


def _tables(data):
    st.subheader('The numbers, in context')
    st.caption('Reference ranges are taken from the report. They may differ between laboratories.')
    categories = data.get('test_categories', [])
    if not categories:
        st.info('No structured test results were found in this report.')
    for category in categories:
        with st.expander(category.get('category', 'Tests'), expanded=True):
            tests = category.get('tests', [])
            if not tests:
                st.caption('No results were found in this section.')
                continue
            frame = pd.DataFrame([{
                'Test Name': test.get('test_name', 'Not provided'),
                'Value': f"{test.get('value', '—')} {test.get('unit', '')}".strip(),
                'Reference Range': test.get('reference_range', 'Not provided'),
                'Status': test.get('status', 'unknown').title(),
            } for test in tests])

            def status_style(value):
                status = value.lower()
                if status == 'normal':
                    return 'background-color: #e5eee4; color: #37533c'
                if status in {'high', 'abnormal'}:
                    return 'background-color: #f1dfe0; color: #7b3b47'
                if status in {'low', 'borderline'}:
                    return 'background-color: #f3e9cc; color: #735b28'
                return 'color: #716b61'

            st.dataframe(frame.style.map(status_style, subset=['Status']), width='stretch', hide_index=True)
    _findings(data.get('abnormal_findings', []))


def _notes(diagnosis):
    if not diagnosis:
        st.info('Turn on Include explanations in Reading preferences, then read your report again to see these notes.')
        return
    st.subheader('Notes for your next conversation')
    st.caption('Generated with Google AI. These notes can help you prepare questions; they do not replace professional medical advice.')
    if diagnosis.get('summary'):
        st.write(diagnosis['summary'])
    risk = diagnosis.get('risk_assessment', {})
    items = ''.join(f'<div><span>{label}</span><strong>{_text(risk.get(name, "unknown").title())}</strong></div>'
                    for label, name in [('Overall assessment', 'overall_risk'), ('Cardiovascular', 'cardiovascular_risk'), ('Diabetes', 'diabetes_risk')])
    st.markdown(f'<div class="risk-strip">{items}</div>', unsafe_allow_html=True)
    _findings(risk.get('risk_factors', []), 'Factors noted in the report')
    for condition in diagnosis.get('potential_conditions', []):
        with st.expander(f"{condition.get('condition', 'Condition')} · {condition.get('probability', 'unknown').title()} likelihood"):
            st.write(condition.get('description', ''))
            for evidence in condition.get('supporting_evidence', []):
                st.write(f'• {evidence}')
    groups = {}
    for item in diagnosis.get('recommendations', []):
        groups.setdefault(item.get('category', 'general').title(), []).append(item)
    for category, recommendations in groups.items():
        st.subheader(f'{category} notes')
        for item in recommendations:
            st.markdown(f'<div class="recommendation"><p>{_text(item.get("recommendation"))}</p>'
                        f'<small>{_text(item.get("rationale", ""))}</small></div>', unsafe_allow_html=True)
    _findings(diagnosis.get('follow_up_tests', []), 'Follow-up tests to discuss')
    if diagnosis.get('red_flags'):
        st.error('The generated analysis flags findings that may need immediate medical attention.')
        for item in diagnosis['red_flags']:
            st.error(item)


def _full_report(scanner, data, diagnosis, downloads):
    st.subheader('Your complete report')
    report = scanner.generate_comprehensive_report(data, diagnosis)
    if st.session_state.get('is_demo'):
        report = '# SYNTHETIC DEMONSTRATION\n\n' + report
    st.caption('Keep a copy to review alongside the original lab report.')
    if downloads:
        left, right = st.columns(2)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with left:
            st.download_button('Download Report (Markdown)', data=report,
                               file_name=f'medical_report_{timestamp}.md', mime='text/markdown',
                               icon=':material/download:', width='stretch')
        with right:
            payload = {'parsed_data': data, 'diagnosis': diagnosis,
                       'is_demo': st.session_state.get('is_demo', False), 'timestamp': datetime.now().isoformat()}
            st.download_button('Download Data (JSON)', data=json.dumps(payload, indent=2),
                               file_name=f'medical_data_{timestamp}.json', mime='application/json',
                               icon=':material/download:', width='stretch')
    with st.expander('Read the full text', expanded=False):
        # The saved export retains its original format; the reader uses plain headings.
        plain = re.sub(r'[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D\u2139]+ ?', '', report)
        st.markdown(plain)


def _results(scanner, trends, downloads):
    data = st.session_state.parsed_data
    diagnosis = st.session_state.get('diagnosis')
    if st.session_state.get('is_demo'):
        st.markdown('<div class="sample-notice">Sample report <span>illustrative data, for a look around. No file was analyzed.</span></div>', unsafe_allow_html=True)
    source = _text(st.session_state.get('source_name', 'Medical report'))
    st.markdown(f'<div class="report-heading"><div class="eyebrow">02 / YOUR READING</div>'
                f'<h2>Your report, in context.</h2><p>{source}</p></div>', unsafe_allow_html=True)
    tests = [test for category in data.get('test_categories', []) for test in category.get('tests', [])]
    with st.container(key='metrics-strip'):
        columns = st.columns(4)
        values = [('Total Tests', len(tests)), ('Normal Results', sum(test.get('status', '').lower() == 'normal' for test in tests)),
                  ('Abnormal Findings', sum(test.get('status', '').lower() in ABNORMAL for test in tests)),
                  ('Test Categories', len(data.get('test_categories', [])))]
        for column, (label, value) in zip(columns, values):
            with column:
                st.metric(label, value)
    overview, table, notes, full = st.tabs(['At a glance', 'Test results', 'Notes & next steps', 'Full report'])
    with overview:
        st.subheader('The person behind the numbers')
        _patient(data)
        _findings(data.get('abnormal_findings', []))
        if not data.get('abnormal_findings'):
            st.caption('No abnormal findings were listed in the extracted report. Review every result in context with your clinician.')
    with table:
        _tables(data)
    with notes:
        _notes(diagnosis)
    with full:
        _full_report(scanner, data, diagnosis, downloads)
    if trends:
        with st.expander('Over time', expanded=False):
            patient = data.get('patient_info', {}).get('name')
            history = [entry for entry in st.session_state.analysis_history if not entry.get('is_demo')
                       and patient not in (None, '', 'Not specified')
                       and entry['parsed_data'].get('patient_info', {}).get('name') == patient]
            chart = scanner.create_test_trend_chart(history)
            if chart is None:
                st.caption('Read at least two reports for the same named patient, with matching test names and units, to compare results over time.')
            else:
                chart.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    font={'color': '#292622'}, colorway=['#7b3b47', '#61705a'])
                st.plotly_chart(chart, width='stretch')
                st.caption('Ordered by analysis time. Confirm patient identity before comparing reports.')


def _history():
    st.markdown('<div class="eyebrow">YOUR SESSION NOTEBOOK</div>', unsafe_allow_html=True)
    st.subheader('Recent reports')
    st.caption('Your five most recent readings. Up to 20 are kept in this browser session.')
    if not st.session_state.analysis_history:
        st.info('No reports yet. Start with a new report or try the sample.')
    for index, entry in enumerate(reversed(st.session_state.analysis_history[-5:])):
        with st.expander(f"{entry.get('source', 'Report')} · {entry.get('timestamp', '').replace('T', ' ')}"):
            if st.button('Open this report', key=f'open_history_{index}'):
                st.session_state.update(parsed_data=deepcopy(entry['parsed_data']),
                                        diagnosis=deepcopy(entry.get('diagnosis')), is_demo=entry.get('is_demo', False),
                                        source_name=entry.get('source', 'Medical report'), active_view='workspace', reading_warning=None)
                st.rerun()
            st.json(entry)


def render_app(scanner_type):
    st.set_page_config(page_title='Labnote · Medical report reader', page_icon=':material/description:',
                       layout='wide', initial_sidebar_state='auto')
    st.markdown(f'<style>{(ASSETS_DIR / "style.css").read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)
    st.session_state.setdefault('analysis_history', [])
    st.session_state.setdefault('active_view', 'workspace')
    st.session_state.setdefault('upload_revision', 0)
    config, key, trends, detailed, downloads = _sidebar()
    model = config.get('GEMINI_MODEL') or 'gemini-3.5-flash-lite'
    try:
        scanner = scanner_type(key, model=model)
    except Exception:
        st.error('The connection could not be prepared. Check your API configuration; the sample is still available.')
        scanner = scanner_type('', model=model)
        key = ''
    if st.session_state.active_view == 'history':
        _history()
    else:
        if 'parsed_data' not in st.session_state:
            _hero()
        else:
            st.markdown('<div class="workspace-masthead"><span>LABNOTE / REPORT WORKSPACE</span><span>RESULTS & NOTES</span></div>', unsafe_allow_html=True)
        if 'parsed_data' in st.session_state:
            if st.session_state.get('reading_warning'):
                st.warning(f'Your results are ready, but the explanatory notes could not be completed: {st.session_state.reading_warning}')
            _results(scanner, trends, downloads)
            with st.expander('Read another report', expanded=False):
                _upload(scanner, key, detailed)
        else:
            _upload(scanner, key, detailed)
            st.markdown('''<div class="reading-path"><div><span>01</span><strong>Bring the report.</strong><p>Use a PDF or a clear photo.</p></div>
                <div><span>02</span><strong>Find the context.</strong><p>Results and their reference ranges.</p></div>
                <div><span>03</span><strong>Keep the questions.</strong><p>Take useful notes to your doctor.</p></div></div>''', unsafe_allow_html=True)
    st.markdown('<footer class="page-footer"><span>Labnote · Medical report reader</span><span>Educational use. Professional care comes first.</span></footer>', unsafe_allow_html=True)
