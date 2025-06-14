def load_prompt(prompt_name):
    """Load a prompt from a file."""
    try:
        with open(f"prompts/{prompt_name}.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None

def generate_method_prompt(prompt_name, user_input):
    """Generate a prompt for a specific method."""
    prompt_text = load_prompt(prompt_name)
    main_prompt = load_prompt("main_prompt1")
    if prompt_text and main_prompt:
        return f"""{main_prompt + prompt_text}
Must Provide Concise Answer:"""
    return None

def rag_prompt(context, question):
    """Create a RAG prompt."""
    main_prompt = load_prompt("main_prompt1")
    if main_prompt and context and question:
        return f"""{main_prompt}
Context: {context}
Question: {question}
Concise and accurate answer:"""
    return None

def sales_pitch_prompt():
     return load_prompt("sales_pitch1")
