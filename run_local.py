import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import ingestion as ing

ing.DATA_DIR = str(BASE_DIR / 'course_materials')
ing.DB_DIR = str(BASE_DIR / 'chroma_db')
ing.APP_DIR = str(BASE_DIR / 'gradio_app')

import agentic_rag as arag

def bootstrap_vector_store():
    col = ing.get_collection()
    if col.count() > 0:
        print(f'ChromaDB already populated with {col.count()} chunks.')
        return
    data_path = Path(ing.DATA_DIR)
    if not data_path.exists():
        print('No course_materials directory found.')
        return
    records = []
    for p in sorted(data_path.rglob('*.*')):
        if p.suffix.lower() in ing.LOADERS:
            course_name = p.parent.name.replace('_', ' ').title()
            for rec in ing.extract_records(str(p)):
                records.append({
                    **rec,
                    'course': course_name,
                    'file': p.name,
                    'type': p.suffix[1:].upper(),
                    'path': str(p),
                })
    chunks = ing.records_to_chunks(records)
    stored = ing.store_chunks(chunks, reset=True)
    print(f'Indexed {stored} chunks into ChromaDB.')

if __name__ == '__main__':
    bootstrap_vector_store()
    app_file = Path(ing.APP_DIR) / 'app.py'
    if app_file.exists():
        import runpy
        runpy.run_path(str(app_file), run_name='__main__')
    else:
        sample_q = 'What does the learning rate control in gradient descent?'
        out = arag.run_agentic_rag(sample_q, 'flan-t5-base')
        print('Question:', sample_q)
        print('Answer  :', out['answer'])
        print('Sources :', out['sources'])
