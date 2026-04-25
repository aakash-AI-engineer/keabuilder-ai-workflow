from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextMatcher:
    def __init__(self, database_inputs):
        self.database = database_inputs
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.database)

    def find_best_match(self, query):
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_idx = similarities.argmax()
        return {
            "query": query,
            "best_match": self.database[best_idx],
            "confidence_score": round(similarities[best_idx], 4)
        }

if __name__ == "__main__":
    saved_prompts = [
        "Generate a welcome email sequence for new leads",
        "Create a fitness landing page funnel",
        "Build a lead capture chatbot flow",
        "Write a sales script for a marketing agency"
    ]
    
    matcher = TextMatcher(saved_prompts)
    user_query = "I need a funnel for capturing leads"
    result = matcher.find_best_match(user_query)
    
    print(f"Query: {result['query']}")
    print(f"Top Match: {result['best_match']}")
    print(f"Score: {result['confidence_score']}")