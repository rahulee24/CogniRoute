import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

class SessionMemory:
    """Manages chat history per session using Redis or a local dictionary fallback."""
    
    _local_store: Dict[str, List[Dict[str, str]]] = {}
    
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "")
        self.memory_window = int(os.getenv("MEMORY_WINDOW", "5"))
        self.redis_client = None
        
        if self.redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
                # Quick health check
                self.redis_client.ping()
                print(f"Connected to Redis at: {self.redis_url}")
            except Exception as e:
                print(f"Redis not available ({e}). Using in-memory fallback store.")
                self.redis_client = None
                
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieves history list for a session, limited to the memory window."""
        if self.redis_client:
            try:
                data = self.redis_client.get(f"session:{session_id}")
                if data:
                    history = json.loads(data)
                    return history[-self.memory_window * 2:] # Retrieve last N rounds (2 messages per round)
            except Exception as e:
                print(f"Redis read error: {e}")
                
        # Fallback to local store
        history = self._local_store.get(session_id, [])
        return history[-self.memory_window * 2:]
        
    def add_message(self, session_id: str, role: str, content: str):
        """Adds a message to the history session."""
        message = {"role": role, "content": content}
        
        if self.redis_client:
            try:
                history = self.get_history(session_id)
                history.append(message)
                # Keep last 50 messages to avoid unbound database growth
                history = history[-50:]
                self.redis_client.set(f"session:{session_id}", json.dumps(history), ex=86400) # 24h expiry
                return
            except Exception as e:
                print(f"Redis write error: {e}")
                
        # Fallback to local store
        if session_id not in self._local_store:
            self._local_store[session_id] = []
        self._local_store[session_id].append(message)
        self._local_store[session_id] = self._local_store[session_id][-50:]
        
    def clear_history(self, session_id: str):
        """Clears history session."""
        if self.redis_client:
            try:
                self.redis_client.delete(f"session:{session_id}")
                return
            except Exception as e:
                print(f"Redis delete error: {e}")
                
        if session_id in self._local_store:
            del self._local_store[session_id]
