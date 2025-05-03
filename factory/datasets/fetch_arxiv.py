def tokenize_and_chunk_text(text, tokenizer, chunk_size=4096, overlap_size=0):
    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Calculate the step size (chunk_size - overlap_size)
    step_size = chunk_size - overlap_size
    
    # Split tokens into overlapping chunks
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), step_size)]
    
    return chunks

def decode_chunks(chunks, tokenizer):
    return [tokenizer.decode(chunk, clean_up_tokenization_spaces=True) for chunk in chunks]

def split_content(content, tokenizer):
    tokenized_chunks = tokenize_and_chunk_text(content, tokenizer)
    text_chunks = decode_chunks(tokenized_chunks, tokenizer)
    return text_chunks

