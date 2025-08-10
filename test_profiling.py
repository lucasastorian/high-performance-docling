#!/usr/bin/env python3
"""Test with profiling enabled to see where time is spent"""

import os
import requests
from io import BytesIO

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions
from gpu_processor import GPUProcessor
from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler

# Disable MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

def main():
    # Download a small test PDF with tables
    pdf_url = "https://arxiv.org/pdf/2408.09869.pdf"  # DocLing paper - 9 pages
    print(f"📥 Downloading test PDF...")
    response = requests.get(pdf_url)
    pdf_stream = BytesIO(response.content)
    
    # Load document
    backend = DoclingParseDocumentBackend(stream=pdf_stream, document_hash="test")
    input_doc = backend.document_cls(backend)
    print(f"✓ Loaded {input_doc.page_count} pages")
    
    # Test with just first 3 pages to see profiling quickly
    backend.pages = backend.pages[:3]
    input_doc = backend.document_cls(backend)
    
    # Configure pipeline
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        images_scale=1.0,
        accelerator_options=AcceleratorOptions(device="mps")
    )
    
    print("\n🔍 Running with profiling enabled...")
    print("-" * 40)
    
    # Create GPU processor
    gpu_processor = GPUProcessor(pipeline_options)
    
    # Preprocess pages (simplified)
    preprocessed_pages = []
    for page_idx, page in enumerate(input_doc.pages):
        page_backend = input_doc.backend.load_page(page_idx)
        preprocessed_pages.append({
            'page_no': page_idx,
            'page': page,
            'page_backend': page_backend
        })
    
    # Process pages
    results = gpu_processor.process_batch(preprocessed_pages)
    
    # Print profiling results
    print("\n📊 PROFILING RESULTS")
    print("=" * 60)
    
    # Get profiling data from AggProfiler
    profiler = AggProfiler()
    if hasattr(profiler, 'get_summary'):
        summary = profiler.get_summary()
        for key, times in summary.items():
            if times:
                avg_time = sum(times) / len(times)
                total_time = sum(times)
                print(f"{key:30s}: avg={avg_time*1000:.2f}ms, total={total_time:.2f}s, count={len(times)}")
    else:
        print("No profiling data available")

if __name__ == "__main__":
    main()