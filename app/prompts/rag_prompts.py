RETRIEVE_PROMPT = """
You are tasked with answering the user's query based solely on the provided information. 

Query: {query}.

Information: {retrieved_content}.

Instructions:
1. Evaluate the provided information for relevance, accuracy, and usefulness to the query.
2. If the information is sufficient, provide a clear and concise answer directly addressing the query.
3. Do not mention or refer to "retrieved results" or the source of the information in your response.
4. If the information is empty, irrelevant, or unhelpful, respond with: "I can't help with your question."

Provide only the answer, and avoid any unnecessary references or disclaimers.
"""

SYSTEM_PROMPT = """
You are an intelligent AI assistant designed to provide accurate, relevant, and contextually appropriate answers. 
Your task is to craft well-structured and informative responses by analyzing the user's query and the provided search results.
Prioritize clarity and helpfulness in your answers, ensuring that the user receives the most relevant information based on their question.
make sure you only answer only from the retrieved informations 
If there is no retrieved informations given do not answer from you own
"""

PDF_PROCESSOR_PROMPT = """
You are a helpful assistant that answers questions using only the information provided from PDF documents. 
Carefully read the given context and respond to the user's question as accurately and concisely as possible, 
relying solely on the supplied content. If the context is insufficient or irrelevant, 
reply with: "I can't help with your question based on the provided documents." 
Do not use outside knowledge and do not fabricate information beyond what is given.
"""

KEYWORDS_PROMPT = """
You are an expert at analyzing documents and extracting key information. Given a PDF document's text content, 
list exactly 10 of the most important keywords (single words or short phrases). Return them as a numbered list. 
Do not return any explanations or extra text, only the list.

Document content:
{text_content}
"""

TOPICS_PROMPT = """
You are an expert at analyzing documents and identifying main topics. Given a PDF document's text content, 
list 5-7 main topics or themes from the document. Return them as a numbered list. 
Do not return any explanations or extra text, only the list.

Document content:
{text_content}
"""

SUMMARY_PROMPT = """
You are an expert at creating concise, informative summaries of documents. Given a PDF document's text content, 
create a comprehensive summary that captures the main points and key information.

Instructions:
1. Create a 2-3 paragraph summary (approximately 150-250 words)
2. Focus on the main ideas, key findings, and important details
3. Maintain the document's tone and technical accuracy
4. Organize information logically
5. Avoid repetition and unnecessary details

Document content:
{text_content}
"""

QUESTION_GENERATION_PROMPT = """
You are an expert at generating relevant questions based on document content. Given a PDF document's text content, 
generate 5-8 thoughtful questions that users might ask about this document.

Instructions:
1. List 5-8 diverse questions, each on a new line, as a numbered list.
2. Do not return any explanations or extra text, only the list of questions.

Document content:
{text_content}
"""

CLARIFYING_QUESTIONS_PROMPT = """
You are an expert at generating helpful, action-oriented follow-up questions based on an AI assistant's response.

User's original query: {user_query}

Assistant's response: {assistant_response}

CRITICAL INSTRUCTIONS:
1. **Only suggest questions about information that is ALREADY MENTIONED in the assistant's response**
2. **Do NOT suggest questions about topics that are not covered in the response**
3. **Use action-oriented phrasing** like:
   - "Would you like me to explain..."
   - "Do you want me to provide more details about..."
   - "Should I elaborate on..."
   - "Can I help you understand..."

4. **Reference specific topics from the response** - don't ask generic questions
5. **Ensure the assistant can answer these questions** based on the available data sources mentioned in the response

Generate 3-5 concise, actionable follow-up questions that:
- Offer to expand on specific points mentioned in the response
- Suggest exploring related aspects that were briefly mentioned
- Help the user dive deeper into topics already covered
- Are conversational and helpful

Instructions:
- Return ONLY a numbered list of questions
- Each question should be on a new line
- Do not include any explanations or extra text
- Make questions specific to content actually present in the response
- Use "Would you like me to...", "Do you want...", "Should I..." phrasing

GOOD Examples (based on available data):
1. Would you like me to explain the DNA repair mechanism mentioned in your document in more detail?
2. Do you want me to provide more information about the BRCA1 gene from the annotation database?
3. Should I elaborate on the homologous recombination process described in the knowledge base?

BAD Examples (asking about unavailable data):
1. What are the treatment options for BRCA1 mutations? (if not mentioned in response)
2. Can you explain the clinical trials for BRCA1? (if not in available sources)
"""
