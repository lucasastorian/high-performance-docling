import requests
from io import BytesIO
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import InputDocument
from docling.datamodel.settings import DocumentLimits
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorOptions

from document_assembler import DocumentAssembler
from lambda_preprocessor import distribute_preprocessing
from gpu_processor import GPUProcessor

if __name__ == '__main__':
    pdf_url = "https://d18rn0p25nwr6d.cloudfront.net/CIK-0000320193/a411a029-368f-4479-b416-25c404acca3d.pdf"  # DocLing paper

    print("=" * 60)
    print("🚀 DISTRIBUTED DOCLING PIPELINE")
    print("=" * 60)

    # Download PDF
    print(f"\n📥 Downloading PDF...")
    response = requests.get(pdf_url)
    pdf_bytes = BytesIO(response.content)
    pdf_bytes.seek(0)

    # Create InputDocument
    input_doc = InputDocument(
        path_or_stream=pdf_bytes,
        format=InputFormat.PDF,
        filename="document.pdf",
        limits=DocumentLimits(),
        backend=DoclingParseV4DocumentBackend
    )
    print(f"✓ Loaded {input_doc.page_count} pages")

    # Configure pipeline with CPU accelerator
    pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        do_table_structure=True,
        images_scale=1.0,
        accelerator_options=AcceleratorOptions(device="mps")
    )

    # ========================================
    # PHASE 1: Distributed Lambda Preprocessing
    # ========================================
    print(f"\n📦 PHASE 1: Lambda Preprocessing (distributed)")
    print("-" * 40)
    preprocessed_pages = distribute_preprocessing(input_doc, batch_size=10)
    print(f"✓ Preprocessed {len(preprocessed_pages)} pages across Lambda functions")

    # ========================================
    # PHASE 2: GPU Processing (RunPod)
    # ========================================
    print(f"\n⚡ PHASE 2: GPU Processing (RunPod)")
    print("-" * 40)
    gpu_processor = GPUProcessor(pipeline_options)
    processed_pages = gpu_processor.process_all_pages(input_doc, preprocessed_pages)

    # ========================================
    # PHASE 3: Document Assembly
    # ========================================
    print(f"\n📄 PHASE 3: Document Assembly")
    print("-" * 40)
    assembler = DocumentAssembler(pipeline_options)
    conv_res = assembler.assemble_document(input_doc, processed_pages)

    # ========================================
    # RESULTS
    # ========================================
    print("\n" + "=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print(f"✓ Document: {conv_res.document.name}")
    print(f"✓ Pages: {len(conv_res.pages)}")
    print(f"✓ Elements: {len(conv_res.assembled.elements)}")

    # Export
    markdown = conv_res.document.export_to_markdown()
    print(f"✓ Markdown length: {len(markdown)} chars")

    print(f"{markdown}")
