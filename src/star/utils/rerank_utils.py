# remembr/utils/rerank_utils.py

def should_rerank(query: str) -> bool:
    """Decide whether to rerank based on query content."""
    # If the question is open-ended, long, or has multiple conditions, rerank
    keywords = ['where', 'when', 'what', 'which', 'how', 'find', 'describe', 'show', 'seen']
    query_lower = query.lower()
    if any(word in query_lower for word in keywords):
        return True
    return False
