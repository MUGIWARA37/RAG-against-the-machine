from pathlib import Path                                                                                                             
from tqdm import tqdm                                                                                                                
from src.indexing.chunking import chunk_markdown, chunk_python                                                                       
                                                                                                                                     
def gather_corpus(directory_path: str):                                                                                              
    """                                                                                                                              
    Crawls a directory, reads .md and .py files, and chunks them.                                                                    
    """                                                                                                                              
    master_chunks = []                                                                                                               
                                                                                                                                                                                                                                               
    base_path = Path(directory_path)                                                                                                                                                     
    all_files = [p for p in base_path.rglob("*") if p.is_file()]                                                                     
                                                                                                                                     
                                                                               
    for file_path in tqdm(all_files, desc="Chunking files"):                                                                         
                                                                                                                                     
        ext = file_path.suffix                                                                            
                                                                                                                                     
        if ext in [".md", ".py"]:                                                                                                    
            try:                                                                                                                     
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                                                                            
                if ext == ".md":
                    chunks = chunk_markdown(text, str(file_path))   
                else:
                    chunks = chunk_python(text, str(file_path))                                                                                                              
                master_chunks.extend(chunks)
            except Exception as e:                                                                                                   
                print(f"Skipping {file_path} due to error: {e}")                                                                     
                                                                                                                                     
    return master_chunks