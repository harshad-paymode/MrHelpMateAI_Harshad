from chunking import crop_pdf_footer,chunk_pdf_with_metadata
from build_index import index_and_store
from core.config import PATHS
import json
from build_index import index_and_store

raw_pdf_path = PATHS.POLICY_PDF_PATH
cleaned_pdf_path = PATHS.POLICY_PDF_PATH_OUTPUT
chunk_output_path = PATHS.OUTPUT_CHUNK_PATH

def main():
    writer = crop_pdf_footer(raw_pdf_path)
    # Write the output PDF
    with open(cleaned_pdf_path, "wb") as output_file:
        writer.write(output_file)
    print(f"Cropped PDF saved")
    chunks = chunk_pdf_with_metadata(cleaned_pdf_path,chunk_output_path)
     # Open a file in write mode and dump the data
    with open(chunk_output_path, 'w') as json_file:
        json.dump(chunks, json_file, indent = 4)

    #Create docs to store as chunk
    with open(chunk_output_path,'r') as chunks:
        data = json.load(chunks)
    #Now build chroma store
    index_and_store(data)

if __name__ == '__main__':
    main()
