import os
from typing import Dict, Any
from dotenv import load_dotenv
load_dotenv()

class ContextGrader:
    """Evaluates whether the retrieved context is sufficient to answer the query."""
    
    def __init__(self):
        self.mock_mode = os.getenv("MOCK_MODE", "False").lower() == "true"
        self.gemini_key = os.getenv("GEMINI_API_KEY", "")
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
        
    def grade(self, query: str, context: str) -> Dict[str, Any]:
        """Grades the sufficiency of the context.
        Returns a dict: {'sufficient': bool, 'score': float, 'reason': str}
        """
        if not context.strip():
            return {
                "sufficient": False,
                "score": 0.0,
                "reason": "Context is completely empty."
            }
            
        if not self.mock_mode and self.gemini_key:
            return self._grade_via_llm(query, context)
        else:
            return self._grade_via_heuristics(query, context)
            
    def _grade_via_heuristics(self, query: str, context: str) -> Dict[str, Any]:
        """Simple rule-based grader for mock/offline usage."""
        query_words = set(query.lower().split())
        context_lower = context.lower()
        
        # Check how many query words are in the context
        matches = sum(1 for word in query_words if word in context_lower)
        score = matches / max(len(query_words), 1)
        
        # Boost score slightly if we have clear source indicators
        if "source" in context_lower or "pdf" in context_lower or "plan" in context_lower:
            score = min(score + 0.2, 1.0)
            
        sufficient = score >= self.confidence_threshold
        
        reason = (
            f"Heuristic score of {score:.2f} meets or exceeds threshold {self.confidence_threshold}."
            if sufficient
            else f"Heuristic score of {score:.2f} is below threshold {self.confidence_threshold}."
        )
        
        return {
            "sufficient": sufficient,
            "score": score,
            "reason": reason
        }

    def _grade_via_llm(self, query: str, context: str) -> Dict[str, Any]:
        """Asks Google Gemini to grade the context for question-answering sufficiency."""
        try:
            from google import genai
            from google.genai import types
            import json
            client = genai.Client(api_key=self.gemini_key)
            
            prompt = """You are a quality control grader for a Retrieval-Augmented Generation pipeline.
            Evaluate whether the provided context contains sufficient information to directly answer the user's question.
            
            Output your assessment ONLY as a raw JSON object with the following fields:
            {
                "score": 0.85, // float between 0.0 and 1.0
                "sufficient": true, // boolean, true if score >= 0.7, false otherwise
                "reason": "Explain briefly why the context is or is not sufficient to answer the question"
            }
            Do not include markdown code block backticks or explanation outside the JSON.

            User Question: {query}
            
            Retrieved Context:
            {context}
            
            JSON Evaluation:"""
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt.format(query=query, context=context),
                config=types.GenerateContentConfig(
                    max_output_tokens=250,
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            
            text_resp = response.text.strip()
            # Clean up JSON formatting if model included backticks
            if text_resp.startswith("```json"):
                text_resp = text_resp.replace("```json", "").replace("```", "").strip()
            elif text_resp.startswith("```"):
                text_resp = text_resp.replace("```", "").strip()
                
            result = json.loads(text_resp)
            return {
                "sufficient": bool(result.get("sufficient", False)),
                "score": float(result.get("score", 0.0)),
                "reason": str(result.get("reason", "Graded by Gemini LLM"))
            }
        except Exception as e:
            print(f"Error in LLM context grading: {e}. Falling back to heuristics.")
            return self._grade_via_heuristics(query, context)
