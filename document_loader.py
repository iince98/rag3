# data/document_loader.py

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader

def remove_empty_lines(text):
    return "\n".join([line for line in text.splitlines() if line.strip() != ""])

def load_documents(data_path):
    all_documents = []
    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # if file.endswith(".pdf"):
                #     loader = PyPDFLoader(file_path)
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                    for doc in documents:
                        doc.page_content = remove_empty_lines(doc.page_content)

                elif file.endswith(".txt"):
                    loader = TextLoader(file_path)
                elif file.endswith(".docx"):
                    loader = UnstructuredWordDocumentLoader(file_path)

                elif file.endswith(".pptx"):
                    loader = UnstructuredPowerPointLoader(file_path)

                else:
                    continue
                all_documents.extend(loader.load())
                print(f"Loaded: {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return all_documents


# import os
# from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader
# from pptx import Presentation
# from langchain_community.vectorstores.utils import filter_complex_metadata

# def extract_pptx_metadata(file_path):
#     """Extract metadata from pptx file (e.g. number of slides, slide titles)."""
#     try:
#         prs = Presentation(file_path)
#         slide_titles = []
#         for slide in prs.slides:
#             # Try to get title shape text
#             title_shapes = [shape for shape in slide.shapes if shape.has_text_frame and shape == slide.shapes.title]
#             title_text = title_shapes[0].text if title_shapes else "No Title"
#             slide_titles.append(title_text)
#         metadata = {
#             "num_slides": len(prs.slides),
#             "slide_titles": slide_titles,
#         }
#         return metadata
#     except Exception as e:
#         print(f"Error extracting pptx metadata: {e}")
#         return {}

# def load_documents(data_path):
#     all_documents = []
#     for root, _, files in os.walk(data_path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             try:
#                 if file.endswith(".pdf"):
#                     loader = PyPDFLoader(file_path)

#                 elif file.endswith(".txt"):
#                     loader = TextLoader(file_path)
#                 elif file.endswith(".docx"):
#                     loader = UnstructuredWordDocumentLoader(file_path)

#                 elif file.endswith(".pptx"):
#                     # Extract metadata separately
#                     metadata = filter_complex_metadata(extract_pptx_metadata(file_path))

#                     print(f"Metadata for {file}: {metadata}")

#                     loader = UnstructuredPowerPointLoader(file_path)

#                 else:
#                     continue

#                 loaded_docs = loader.load()
#                 # Optionally, you can attach metadata to each document:
#                 for doc in loaded_docs:
#                     if file.endswith(".pptx"):
#                         doc.metadata = {**getattr(doc, "metadata", {}), **metadata}

#                 all_documents.extend(loaded_docs)
#                 print(f"Loaded: {file}")

#             except Exception as e:
#                 print(f"Error loading {file}: {e}")

#     return all_documents



